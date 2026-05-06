#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from simdistserve.benchmarks.search_hetero import (
    _config_to_dict,
    _config_profile_status,
    _load_handoff,
    _load_pool,
    _model_object,
    _normalize_model,
)
from simdistserve.hetero.count_optimizer import SearchMode
from simdistserve.hetero.profile import SimulationGoodputProfiler
from simdistserve.hetero.search import search_hetero_configs


SEARCH_MODES: tuple[SearchMode, ...] = ("milp", "no_cross", "cuda_prefill_ascend_decode")


def run_suite(config_path: Path, output_root: Path, mu_cache_path: Path | None = None) -> dict:
    payload = json.loads(config_path.read_text())
    model_label = str(payload["model"])
    model = _normalize_model(model_label)
    model_obj = _model_object(model)
    search = payload.get("search", {})
    slo = payload.get("slo", {})

    profiler = SimulationGoodputProfiler(
        model=model,
        workload=str(payload["workload"]),
        prefill_target_ms=float(slo.get("prefill_target_ms", payload.get("prefill_slo_ms", 1000.0))),
        decode_target_ms=float(slo.get("decode_target_ms", payload.get("decode_slo_ms", 1000.0))),
        prefill_attainment=int(slo.get("prefill_attainment", payload.get("prefill_attainment", 90))),
        decode_attainment=int(slo.get("decode_attainment", payload.get("decode_attainment", 90))),
        max_rate=float(search.get("single_instance_max_rate", search.get("max_rate", 8.0))),
        epsilon=float(search.get("single_instance_epsilon", search.get("rate_epsilon", 0.25))),
        profile_num_requests=int(search.get("profile_num_requests", 120)),
        arrival=str(search.get("arrival", "poisson")),
        cv=float(search.get("cv", 1.0)),
        seed=int(search.get("seed", 0)),
        cache_path=str(mu_cache_path or search.get("mu_cache_path", "")) or None,
        auto_expand_max_rate=bool(search.get("auto_expand_max_rate", True)),
        profile_max_rate_cap=float(search.get("profile_max_rate_cap", 8192.0)),
        profile_min_profile_duration_s=float(search.get("profile_min_profile_duration_s", 1.0)),
        profile_timeout_s=(
            None if "profile_timeout_s" not in search else float(search["profile_timeout_s"])
        ),
    )

    cuda_pool = _load_pool(payload["pools"], "cuda")
    ascend_pool = _load_pool(payload["pools"], "ascend")
    handoff = _load_handoff(payload)
    top_k = int(search.get("top_k", 10))
    capped_mu_policy = str(search.get("capped_mu_policy", "keep"))
    if capped_mu_policy not in {"keep", "exclude"}:
        raise ValueError(f"Unknown capped_mu_policy: {capped_mu_policy}")

    summary = {"model": model_label, "config": str(config_path), "modes": {}}
    for mode in SEARCH_MODES:
        started_at = time.perf_counter()
        result = search_hetero_configs(
            model_type=model_obj,
            cuda_pool=cuda_pool,
            ascend_pool=ascend_pool,
            handoff=handoff,
            profiler=profiler,
            top_k=top_k,
            search_mode=mode,
            capped_mu_policy=capped_mu_policy,  # type: ignore[arg-type]
        )
        rendered = {
            "search_mode": result.search_mode,
            "capped_mu_policy": capped_mu_policy,
            "best_config": _config_to_dict(result.best_config) if result.best_config is not None else None,
            "best_config_profile_status": _config_profile_status(result.best_config, profiler),
            "top_configs": [_config_to_dict(config) for config in result.configs],
            "top_config_profile_status": [
                _config_profile_status(config, profiler) for config in result.configs
            ],
            "num_shape_tuples": result.num_shape_tuples,
            "num_allocation_problems": result.num_allocation_problems,
            "num_count_tuples": result.num_count_tuples,
            "profiled_shapes": len(profiler.cache),
            "profile_summary": profiler.summary(),
            "timing": result.timing,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
        mode_dir = output_root / model_label / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        output_path = mode_dir / "search_4nodes_high_affinity.json"
        output_path.write_text(json.dumps(rendered, indent=2) + "\n")
        summary["modes"][mode] = {
            "output": str(output_path),
            "estimated_goodput": (
                None
                if rendered["best_config"] is None
                else rendered["best_config"]["estimated_goodput"]
            ),
            "elapsed_seconds": rendered["elapsed_seconds"],
            "profile_seconds": rendered["timing"]["profile_seconds"],
            "allocation_milp_seconds": rendered["timing"]["allocation_milp_seconds"],
            "num_capped_shapes": rendered["profile_summary"]["num_capped_shapes"],
            "num_timed_out_shapes": rendered["profile_summary"]["num_timed_out_shapes"],
            "best_uses_capped_profile": rendered["best_config_profile_status"]["uses_capped_profile"],
            "best_uses_timed_out_profile": rendered["best_config_profile_status"]["uses_timed_out_profile"],
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all heterogeneous search modes for one config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/hetero/results/search"),
    )
    parser.add_argument(
        "--distserve-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_5p4d.json"),
    )
    parser.add_argument(
        "--vllm-ascend-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_filtered.json"),
    )
    parser.add_argument(
        "--mu-cache-path",
        type=Path,
        default=None,
        help="Persistent single-instance goodput cache JSON. Defaults to simdistserve/hetero/results/cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["SIMDISTSERVE_DISTSERVE_PROFILE"] = str(args.distserve_profile)
    os.environ["SIMDISTSERVE_VLLM_ASCEND_PROFILE"] = str(args.vllm_ascend_profile)
    summary = run_suite(args.config, args.output_root, args.mu_cache_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

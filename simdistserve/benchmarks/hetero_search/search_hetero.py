#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from simdistserve.constants import ModelTypes
from simdistserve.hetero.count_optimizer import SearchMode
from simdistserve.hetero.profile import SimulationGoodputProfiler
from simdistserve.hetero.search import search_hetero_configs
from simdistserve.hetero.types import DevicePool, HandoffGoodput, HeteroConfig, RoleConfig, RoleShape


MODEL_ALIASES = {
    "llama_1B": ModelTypes.LLAMA_3_2_1B_LOCAL_PATH,
    "llama_3B": ModelTypes.LLAMA_3_2_3B_LOCAL_PATH,
    "llama_7B": ModelTypes.LLAMA_2_7B_LOCAL_PATH,
    "llama_8B": ModelTypes.LLAMA_3_1_8B_LOCAL_PATH,
}


def _normalize_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _model_object(model: str):
    return ModelTypes.model_str_to_object(_normalize_model(model))


def _load_pool(payload: dict[str, Any], kind: str) -> DevicePool:
    raw = payload[kind]
    return DevicePool(
        kind=kind,
        num_nodes=int(raw["num_nodes"]),
        devices_per_node=int(raw["devices_per_node"]),
        high_affinity=bool(raw.get("high_affinity", False)),
    )


def _load_handoff(payload: dict[str, Any]) -> HandoffGoodput:
    raw = payload.get("handoff", {})
    ca = raw.get("cuda_to_ascend", raw.get("ca", {}))
    ac = raw.get("ascend_to_cuda", raw.get("ac", {}))
    return HandoffGoodput(
        cuda_to_ascend=float(ca.get("handoff_goodput_upper_bound", ca.get("goodput", 0.0))),
        ascend_to_cuda=float(ac.get("handoff_goodput_upper_bound", ac.get("goodput", 0.0))),
    )


def _shape_to_dict(shape: RoleShape | None) -> dict[str, Any] | None:
    return None if shape is None else asdict(shape)


def _role_config_to_dict(config: RoleConfig) -> dict[str, Any]:
    return {
        "shape": _shape_to_dict(config.shape),
        "num_instances": config.num_instances,
    }


def _config_to_dict(config: HeteroConfig) -> dict[str, Any]:
    return {
        "cuda_prefill": _role_config_to_dict(config.cuda_prefill),
        "cuda_decode": _role_config_to_dict(config.cuda_decode),
        "ascend_prefill": _role_config_to_dict(config.ascend_prefill),
        "ascend_decode": _role_config_to_dict(config.ascend_decode),
        "flows": asdict(config.flows),
        "estimated_goodput": config.estimated_goodput,
    }


def _config_profile_status(config: HeteroConfig | None, profiler: SimulationGoodputProfiler) -> dict[str, Any]:
    if config is None:
        return {
            "uses_capped_profile": False,
            "uses_timed_out_profile": False,
            "capped_roles": [],
            "timed_out_roles": [],
        }
    roles = {
        "cuda_prefill": config.cuda_prefill.shape,
        "cuda_decode": config.cuda_decode.shape,
        "ascend_prefill": config.ascend_prefill.shape,
        "ascend_decode": config.ascend_decode.shape,
    }
    capped_roles = [
        {
            "role": name,
            "shape": _shape_to_dict(shape),
            "goodput": None if profiler.result_for(shape) is None else profiler.result_for(shape).goodput,
            "effective_cap": None
            if profiler.result_for(shape) is None
            else profiler.result_for(shape).effective_cap,
        }
        for name, shape in roles.items()
        if shape is not None and profiler.is_capped(shape)
    ]
    timed_out_roles = [
        {
            "role": name,
            "shape": _shape_to_dict(shape),
            "goodput": None if profiler.result_for(shape) is None else profiler.result_for(shape).goodput,
            "effective_cap": None
            if profiler.result_for(shape) is None
            else profiler.result_for(shape).effective_cap,
        }
        for name, shape in roles.items()
        if shape is not None and profiler.is_timed_out(shape)
    ]
    return {
        "uses_capped_profile": bool(capped_roles),
        "uses_timed_out_profile": bool(timed_out_roles),
        "capped_roles": capped_roles,
        "timed_out_roles": timed_out_roles,
    }


def run_search(config_path: Path, mu_cache_path: Path | None = None) -> dict[str, Any]:
    started_at = time.perf_counter()
    payload = json.loads(config_path.read_text())
    model = _normalize_model(str(payload["model"]))
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
    search_mode_raw = str(search.get("mode", "milp"))
    if search_mode_raw not in {"milp", "no_cross", "cuda_prefill_ascend_decode"}:
        raise ValueError(f"Unknown search mode: {search_mode_raw}")
    search_mode: SearchMode = search_mode_raw  # type: ignore[assignment]
    capped_mu_policy = str(search.get("capped_mu_policy", "keep"))
    if capped_mu_policy not in {"keep", "exclude"}:
        raise ValueError(f"Unknown capped_mu_policy: {capped_mu_policy}")

    result = search_hetero_configs(
        model_type=model_obj,
        cuda_pool=_load_pool(payload["pools"], "cuda"),
        ascend_pool=_load_pool(payload["pools"], "ascend"),
        handoff=_load_handoff(payload),
        profiler=profiler,
        top_k=int(search.get("top_k", 10)),
        search_mode=search_mode,
        capped_mu_policy=capped_mu_policy,  # type: ignore[arg-type]
    )

    return {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search heterogeneous DistServe configs.")
    parser.add_argument("--config", type=Path, required=True, help="Search config JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--search-mode",
        choices=["milp", "no_cross", "cuda_prefill_ascend_decode"],
        default=None,
        help="Override search mode in config.",
    )
    parser.add_argument(
        "--distserve-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_5p4d.json"),
        help="DistServe CUDA profile JSON used by simulate_dist.",
    )
    parser.add_argument(
        "--vllm-ascend-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_filtered.json"),
        help="vLLM Ascend profile JSON used by simulate_dist.",
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
    if args.search_mode is not None:
        payload = json.loads(args.config.read_text())
        payload.setdefault("search", {})["mode"] = args.search_mode
        temp_config = args.config.with_suffix(f".{args.search_mode}.tmp.json")
        temp_config.write_text(json.dumps(payload, indent=2) + "\n")
        try:
            result = run_search(temp_config, args.mu_cache_path)
        finally:
            temp_config.unlink(missing_ok=True)
    else:
        result = run_search(args.config, args.mu_cache_path)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()

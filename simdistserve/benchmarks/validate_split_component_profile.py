#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simdistserve.benchmarks.component_fit_utils import (
    DEFAULT_MODELS,
    DEFAULT_RATES,
    build_calibrated_latency_rows,
    default_validation_root,
    extract_request_components,
    flatten_case_for_csv,
    load_actual_requests,
    load_model_profile,
    resolve_actual_exp_path,
    run_simulation,
    summarize_actual_requests,
    summarize_error_metrics,
    summarize_latency_rows,
    workload_path_for_split,
    write_latency_csv,
)


WORKLOAD_ROOT_DEFAULT = Path("/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0")


def default_real_valid_root(backend: str) -> Path:
    if backend == "distserve":
        return Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result_split/val")
    if backend == "vllm_ascend":
        return Path("/users/rh/ascend_data/ascend_vllm_split/val")
    raise ValueError(f"Unsupported backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate split-aware component profiles on val.jsonl.")
    parser.add_argument("--backend", choices=["distserve", "vllm_ascend"], required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workload-root", type=Path, default=WORKLOAD_ROOT_DEFAULT)
    parser.add_argument("--real-valid-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--python-bin", type=str, default="python3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=120)
    parser.add_argument("--arrival", type=str, default="poisson")
    parser.add_argument("--cv", type=float, default=1.0)
    parser.add_argument("--prefill-slo", type=float, default=1.0)
    parser.add_argument("--decode-slo", type=float, default=1.0)
    parser.add_argument("--total-slo", type=float, default=1.0)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--rates", nargs="+", default=DEFAULT_RATES)
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = default_validation_root(args.backend)
    if args.real_valid_root is None:
        args.real_valid_root = default_real_valid_root(args.backend)
    return args


def raw_latency_rows_from_components(components):
    rows = []
    for component in components:
        first_token_latency = (
            component.prefill_queue_ms
            + component.prefill_compute_ms
            + component.handoff_queue_ms
            + component.handoff_service_ms
            + component.decode_queue_first_ms
            + component.first_decode_compute_ms
        )
        total_latency = max(component.total_latency_ms_raw, first_token_latency)
        rows.append(
            {
                "req_id": component.req_id,
                "first_token_latency": first_token_latency,
                "decoding_latency": max(total_latency - first_token_latency, 0.0),
                "total_latency": total_latency,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cases = []

    for model_alias in args.models:
        workload_path = workload_path_for_split(args.workload_root, model_alias, args.split)
        if not workload_path.exists():
            raise FileNotFoundError(f"Missing workload: {workload_path}")

        model_profile = load_model_profile(args.profile, model_alias)
        handoff_delay_ms = float(model_profile.get("handoff_delay_ms", 0.0))
        handoff_delay_per_token_ms = float(model_profile.get("handoff_delay_per_token_ms", 0.0))

        for rate in args.rates:
            exp_path = resolve_actual_exp_path(args.backend, args.real_valid_root, model_alias, args.num_prompts, str(rate))
            if not exp_path.exists():
                raise FileNotFoundError(f"Missing real validation trace: {exp_path}")

            output_dir = args.output_root / model_alias / f"rate_{rate}"
            print(f"VALIDATE backend={args.backend} model={model_alias} rate={rate}", flush=True)
            run_simulation(
                backend=args.backend,
                python_bin=args.python_bin,
                profile=args.profile,
                model_alias=model_alias,
                rate=str(rate),
                output_dir=output_dir,
                workload_path=workload_path,
                num_prompts=args.num_prompts,
                seed=args.seed,
                arrival=args.arrival,
                cv=args.cv,
                handoff_delay_ms=handoff_delay_ms,
                handoff_delay_per_token_ms=handoff_delay_per_token_ms,
            )

            components = extract_request_components(output_dir / "request_event.csv", backend=args.backend)
            raw_rows = raw_latency_rows_from_components(components)
            calibrated_rows = build_calibrated_latency_rows(model_profile, components, rate=rate)
            write_latency_csv(output_dir / "request_latency.csv", calibrated_rows)

            actual_requests = load_actual_requests(exp_path)
            if len(actual_requests) != len(calibrated_rows):
                raise ValueError(
                    f"{exp_path}: length mismatch actual={len(actual_requests)} predicted={len(calibrated_rows)}"
                )

            actual = summarize_actual_requests(actual_requests, args.prefill_slo, args.decode_slo, args.total_slo)
            raw_sim = summarize_latency_rows(raw_rows, args.prefill_slo, args.decode_slo, args.total_slo)
            new_sim = summarize_latency_rows(calibrated_rows, args.prefill_slo, args.decode_slo, args.total_slo)
            errors = {
                "new_slo_abs_diff_prefill_pct": abs(actual["slo"]["prefill"] - new_sim["slo"]["prefill"]),
                "raw_slo_abs_diff_prefill_pct": abs(actual["slo"]["prefill"] - raw_sim["slo"]["prefill"]),
                "new_slo_abs_diff_decode_pct": abs(actual["slo"]["decode"] - new_sim["slo"]["decode"]),
                "raw_slo_abs_diff_decode_pct": abs(actual["slo"]["decode"] - raw_sim["slo"]["decode"]),
                "new_slo_abs_diff_total_pct": abs(actual["slo"]["total"] - new_sim["slo"]["total"]),
                "raw_slo_abs_diff_total_pct": abs(actual["slo"]["total"] - raw_sim["slo"]["total"]),
                "new_ftl_mean_ms_abs_ms": abs(actual["ftl"]["mean_ms"] - new_sim["ftl"]["mean_ms"]),
                "raw_ftl_mean_ms_abs_ms": abs(actual["ftl"]["mean_ms"] - raw_sim["ftl"]["mean_ms"]),
                "new_total_p95_ms_abs_ms": abs(actual["total"]["p95_ms"] - new_sim["total"]["p95_ms"]),
                "raw_total_p95_ms_abs_ms": abs(actual["total"]["p95_ms"] - raw_sim["total"]["p95_ms"]),
            }
            cases.append(
                {
                    "model": model_alias,
                    "rate": float(rate),
                    "exp_path": str(exp_path),
                    "actual": actual,
                    "raw_sim": raw_sim,
                    "new_sim": new_sim,
                    "errors": errors,
                }
            )

    summary = {
        "backend": args.backend,
        "profile": str(args.profile),
        "output_root": str(args.output_root),
        "split": args.split,
        "cases": cases,
        "aggregate": summarize_error_metrics(cases),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_root / "summary.json"
    summary_csv = args.output_root / "summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    flat_rows = [flatten_case_for_csv(case) for case in cases]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    for key, stats in summary["aggregate"].items():
        print(f"{key} {stats}")
    print(f"WROTE {summary_json}")
    print(f"WROTE {summary_csv}")


if __name__ == "__main__":
    main()

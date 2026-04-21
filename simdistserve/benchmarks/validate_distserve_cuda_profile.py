#!/usr/bin/env python3
"""
Validate a fitted DistServe CUDA profile against real benchmark traces.

This mirrors `validate_vllm_ascend_profile.py`:
1. Replays the benchmark workload through `simulate_dist.py`.
2. Compares simulated per-request latencies to the real DistServe `.exp` traces.
   For total latency, prefer the visible completion boundary
   `last_token_timestamp - start_time` instead of the client-side
   `end_time - start_time`, which can include large post-last-token
   response-path stalls in some DistServe runs.
3. Writes `summary.json` and `summary.csv` under the output root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from simdistserve.benchmarks.component_fit_utils import (
    build_calibrated_latency_rows,
    extract_request_components,
    load_model_profile,
    summarize_latency_rows,
)


REPO_ROOT = Path("/users/rh/DistServe")
SIMULATE_DIST = REPO_ROOT / "simdistserve" / "benchmarks" / "simulate_dist.py"

MODEL_CONFIGS = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}

DEFAULT_MODELS = list(MODEL_CONFIGS.keys())
DEFAULT_RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a fitted DistServe CUDA profile from scratch.")
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Fitted DistServe CUDA profile JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where replay outputs and summary files will be written.",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        default=REPO_ROOT / "evaluation" / "2-benchmark-serving" / "data" / "sampled_sharegpt_pure.jsonl",
        help="Workload file used by the real benchmark traces.",
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=REPO_ROOT / "evaluation" / "2-benchmark-serving" / "result",
        help="Root directory containing real DistServe `.exp` traces.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run simulate_dist.py.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--arrival", type=str, default="poisson")
    parser.add_argument("--cv", type=float, default=1.0)
    parser.add_argument("--prefill-slo", type=float, default=1.0, help="Seconds.")
    parser.add_argument("--decode-slo", type=float, default=1.0, help="Seconds.")
    parser.add_argument("--total-slo", type=float, default=1.0, help="Seconds.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        default=DEFAULT_RATES,
    )
    return parser.parse_args()


def percentile_ms(values_ms: list[float], pct: float) -> float:
    if not values_ms:
        return 0.0
    return float(np.percentile(np.array(values_ms, dtype=float), pct))


def summarize_series_ms(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms, dtype=float)
    return {
        "mean_ms": float(arr.mean()) if len(arr) else 0.0,
        "p50_ms": percentile_ms(values_ms, 50),
        "p95_ms": percentile_ms(values_ms, 95),
        "p99_ms": percentile_ms(values_ms, 99),
    }


def compute_actual_case_stats(exp_path: Path, prefill_slo_s: float, decode_slo_s: float, total_slo_s: float) -> dict[str, Any]:
    data = json.loads(exp_path.read_text())

    ftl_ms: list[float] = []
    decode_ms: list[float] = []
    total_ms: list[float] = []
    prefill_ok = decode_ok = total_ok = both_ok = 0

    for request in data:
        ftl_s = float(request.get("ftl", 0.0))
        start_time = request.get("start_time")
        token_timestamps = request.get("token_timestamps") or []
        if start_time is not None and token_timestamps:
            total_s = max(float(token_timestamps[-1]) - float(start_time), 0.0)
        else:
            total_s = float(request.get("latency", 0.0))
        lifecycle_events = request.get("lifecycle_events") or []
        decode_s = None

        if lifecycle_events:
            lifecycle = {event["event_type"]: event["timestamp"] for event in lifecycle_events}
            dec_begin = lifecycle.get("decoding_begin")
            dec_end = lifecycle.get("decoding_end")
            if dec_begin is not None and dec_end is not None:
                decode_s = float(dec_end) - float(dec_begin)

        if decode_s is None:
            decode_s = max(total_s - ftl_s, 0.0)

        prefill_ok_i = ftl_s <= prefill_slo_s
        decode_ok_i = decode_s <= decode_slo_s
        total_ok_i = total_s <= total_slo_s
        both_ok_i = prefill_ok_i and decode_ok_i

        prefill_ok += int(prefill_ok_i)
        decode_ok += int(decode_ok_i)
        total_ok += int(total_ok_i)
        both_ok += int(both_ok_i)

        ftl_ms.append(ftl_s * 1000.0)
        decode_ms.append(decode_s * 1000.0)
        total_ms.append(total_s * 1000.0)

    total = len(data)
    return {
        "slo": {
            "prefill": 100.0 * prefill_ok / total,
            "decode": 100.0 * decode_ok / total,
            "total": 100.0 * total_ok / total,
            "both": 100.0 * both_ok / total,
        },
        "ftl": summarize_series_ms(ftl_ms),
        "decode": summarize_series_ms(decode_ms),
        "total": summarize_series_ms(total_ms),
    }


def compute_sim_case_stats(csv_path: Path, prefill_slo_s: float, decode_slo_s: float, total_slo_s: float) -> dict[str, Any]:
    rows = list(csv.DictReader(csv_path.open()))

    ftl_ms: list[float] = []
    decode_ms: list[float] = []
    total_ms: list[float] = []
    prefill_ok = decode_ok = total_ok = both_ok = 0

    for row in rows:
        ftl_ms_i = float(row["first_token_latency"])
        decode_ms_i = float(row["decoding_latency"])
        total_ms_i = float(row["total_latency"])

        ftl_s = ftl_ms_i / 1000.0
        decode_s = decode_ms_i / 1000.0
        total_s = total_ms_i / 1000.0

        prefill_ok_i = ftl_s <= prefill_slo_s
        decode_ok_i = decode_s <= decode_slo_s
        total_ok_i = total_s <= total_slo_s
        both_ok_i = prefill_ok_i and decode_ok_i

        prefill_ok += int(prefill_ok_i)
        decode_ok += int(decode_ok_i)
        total_ok += int(total_ok_i)
        both_ok += int(both_ok_i)

        ftl_ms.append(ftl_ms_i)
        decode_ms.append(decode_ms_i)
        total_ms.append(total_ms_i)

    total = len(rows)
    return {
        "slo": {
            "prefill": 100.0 * prefill_ok / total,
            "decode": 100.0 * decode_ok / total,
            "total": 100.0 * total_ok / total,
            "both": 100.0 * both_ok / total,
        },
        "ftl": summarize_series_ms(ftl_ms),
        "decode": summarize_series_ms(decode_ms),
        "total": summarize_series_ms(total_ms),
    }


def compute_calibrated_sim_case_stats(
    profile_path: Path,
    model_alias: str,
    rate: float | str,
    request_event_csv: Path,
    prefill_slo_s: float,
    decode_slo_s: float,
    total_slo_s: float,
) -> dict[str, Any]:
    model_profile = load_model_profile(profile_path, model_alias)
    has_global_queue_model = bool(model_profile.get("queue_model_ftl")) and bool(model_profile.get("queue_model_total"))
    has_piecewise_queue_model = bool(model_profile.get("queue_model_piecewise"))
    if not has_global_queue_model and not has_piecewise_queue_model:
        request_latency_csv = request_event_csv.parent / "request_latency.csv"
        return compute_sim_case_stats(request_latency_csv, prefill_slo_s, decode_slo_s, total_slo_s)
    components = extract_request_components(request_event_csv, backend="distserve")
    rows = build_calibrated_latency_rows(model_profile, components, rate=rate)
    return summarize_latency_rows(rows, prefill_slo_s, decode_slo_s, total_slo_s)


def run_simulation(
    python_bin: str,
    profile: Path,
    model_alias: str,
    rate: str,
    output_dir: Path,
    workload_path: Path,
    model_path: str,
    num_prompts: int,
    seed: int,
    arrival: str,
    cv: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SIMDISTSERVE_DISTSERVE_PROFILE"] = str(profile)

    cmd = [
        python_bin,
        str(SIMULATE_DIST),
        "--backend", "distserve",
        "--model", model_path,
        "--seed", str(seed),
        "--rate", str(rate),
        "--N", str(num_prompts),
        "--arrival", arrival,
        "--workload", str(workload_path),
        "--name", f"{model_alias}/rate_{rate}",
        "--output", str(output_dir / "sharegpt.json.sim.csv"),
        "--output-request-info", str(output_dir / "request_info.csv"),
        "--output-request-event", str(output_dir / "request_event.csv"),
        "--output-request-latency", str(output_dir / "request_latency.csv"),
        "--slo-scales", "[1.0]",
        "--cv", str(cv),
    ]
    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def flatten_case_for_csv(case: dict[str, Any]) -> dict[str, Any]:
    errors = case["errors"]
    return {
        "model": case["model"],
        "rate": case["rate"],
        "actual_prefill_slo_pct": case["actual"]["slo"]["prefill"],
        "new_prefill_slo_pct": case["new_sim"]["slo"]["prefill"],
        "new_prefill_slo_abs_diff_pct": errors["new_slo_abs_diff_prefill_pct"],
        "prev_prefill_slo_abs_diff_pct": errors["prev_slo_abs_diff_prefill_pct"],
        "actual_decode_slo_pct": case["actual"]["slo"]["decode"],
        "new_decode_slo_pct": case["new_sim"]["slo"]["decode"],
        "new_decode_slo_abs_diff_pct": errors["new_slo_abs_diff_decode_pct"],
        "prev_decode_slo_abs_diff_pct": errors["prev_slo_abs_diff_decode_pct"],
        "actual_total_slo_pct": case["actual"]["slo"]["total"],
        "new_total_slo_pct": case["new_sim"]["slo"]["total"],
        "new_total_slo_abs_diff_pct": errors["new_slo_abs_diff_total_pct"],
        "prev_total_slo_abs_diff_pct": errors["prev_slo_abs_diff_total_pct"],
        "actual_both_slo_pct": case["actual"]["slo"]["both"],
        "new_both_slo_pct": case["new_sim"]["slo"]["both"],
        "new_both_slo_abs_diff_pct": errors["new_slo_abs_diff_both_pct"],
        "prev_both_slo_abs_diff_pct": errors["prev_slo_abs_diff_both_pct"],
        "actual_ftl_mean_ms": case["actual"]["ftl"]["mean_ms"],
        "new_ftl_mean_ms": case["new_sim"]["ftl"]["mean_ms"],
        "new_ftl_mean_ms_abs_ms": errors["new_ftl_mean_ms_abs_ms"],
        "prev_ftl_mean_ms_abs_ms": errors["prev_ftl_mean_ms_abs_ms"],
        "actual_ftl_p95_ms": case["actual"]["ftl"]["p95_ms"],
        "new_ftl_p95_ms": case["new_sim"]["ftl"]["p95_ms"],
        "new_ftl_p95_ms_abs_ms": errors["new_ftl_p95_ms_abs_ms"],
        "prev_ftl_p95_ms_abs_ms": errors["prev_ftl_p95_ms_abs_ms"],
        "actual_ftl_p99_ms": case["actual"]["ftl"]["p99_ms"],
        "new_ftl_p99_ms": case["new_sim"]["ftl"]["p99_ms"],
        "new_ftl_p99_ms_abs_ms": errors["new_ftl_p99_ms_abs_ms"],
        "prev_ftl_p99_ms_abs_ms": errors["prev_ftl_p99_ms_abs_ms"],
        "actual_decode_mean_ms": case["actual"]["decode"]["mean_ms"],
        "new_decode_mean_ms": case["new_sim"]["decode"]["mean_ms"],
        "new_decode_mean_ms_abs_ms": errors["new_decode_mean_ms_abs_ms"],
        "prev_decode_mean_ms_abs_ms": errors["prev_decode_mean_ms_abs_ms"],
        "actual_decode_p95_ms": case["actual"]["decode"]["p95_ms"],
        "new_decode_p95_ms": case["new_sim"]["decode"]["p95_ms"],
        "new_decode_p95_ms_abs_ms": errors["new_decode_p95_ms_abs_ms"],
        "prev_decode_p95_ms_abs_ms": errors["prev_decode_p95_ms_abs_ms"],
        "actual_decode_p99_ms": case["actual"]["decode"]["p99_ms"],
        "new_decode_p99_ms": case["new_sim"]["decode"]["p99_ms"],
        "new_decode_p99_ms_abs_ms": errors["new_decode_p99_ms_abs_ms"],
        "prev_decode_p99_ms_abs_ms": errors["prev_decode_p99_ms_abs_ms"],
        "actual_total_mean_ms": case["actual"]["total"]["mean_ms"],
        "new_total_mean_ms": case["new_sim"]["total"]["mean_ms"],
        "new_total_mean_ms_abs_ms": errors["new_total_mean_ms_abs_ms"],
        "prev_total_mean_ms_abs_ms": errors["prev_total_mean_ms_abs_ms"],
        "actual_total_p95_ms": case["actual"]["total"]["p95_ms"],
        "new_total_p95_ms": case["new_sim"]["total"]["p95_ms"],
        "new_total_p95_ms_abs_ms": errors["new_total_p95_ms_abs_ms"],
        "prev_total_p95_ms_abs_ms": errors["prev_total_p95_ms_abs_ms"],
        "actual_total_p99_ms": case["actual"]["total"]["p99_ms"],
        "new_total_p99_ms": case["new_sim"]["total"]["p99_ms"],
        "new_total_p99_ms_abs_ms": errors["new_total_p99_ms_abs_ms"],
        "prev_total_p99_ms_abs_ms": errors["prev_total_p99_ms_abs_ms"],
    }


def summarize_error_metrics(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_key: dict[str, list[float]] = {}
    for case in cases:
        for key, value in case["errors"].items():
            if value is None:
                continue
            by_key.setdefault(key, []).append(float(value))

    aggregate: dict[str, dict[str, float]] = {}
    for key, values in by_key.items():
        arr = np.array(values, dtype=float)
        aggregate[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }
    return aggregate


def main() -> None:
    args = parse_args()

    cases: list[dict[str, Any]] = []

    for model_alias in args.models:
        model_path = MODEL_CONFIGS[model_alias]

        for rate in args.rates:
            output_dir = args.output_root / model_alias / f"rate_{rate}"
            exp_path = args.bench_root / model_alias / f"distserve-{args.num_prompts}-{rate}.exp"
            if not exp_path.exists():
                raise FileNotFoundError(f"Real benchmark file not found: {exp_path}")

            print(f"RUN {model_alias} rate={rate}", flush=True)
            run_simulation(
                python_bin=args.python_bin,
                profile=args.profile,
                model_alias=model_alias,
                rate=str(rate),
                output_dir=output_dir,
                workload_path=args.workload,
                model_path=model_path,
                num_prompts=args.num_prompts,
                seed=args.seed,
                arrival=args.arrival,
                cv=args.cv,
            )

            request_event_csv = output_dir / "request_event.csv"
            actual = compute_actual_case_stats(exp_path, args.prefill_slo, args.decode_slo, args.total_slo)
            new_sim = compute_calibrated_sim_case_stats(
                args.profile,
                model_alias,
                rate,
                request_event_csv,
                args.prefill_slo,
                args.decode_slo,
                args.total_slo,
            )

            errors = {
                "new_slo_abs_diff_prefill_pct": abs(actual["slo"]["prefill"] - new_sim["slo"]["prefill"]),
                "prev_slo_abs_diff_prefill_pct": None,
                "new_slo_abs_diff_decode_pct": abs(actual["slo"]["decode"] - new_sim["slo"]["decode"]),
                "prev_slo_abs_diff_decode_pct": None,
                "new_slo_abs_diff_total_pct": abs(actual["slo"]["total"] - new_sim["slo"]["total"]),
                "prev_slo_abs_diff_total_pct": None,
                "new_slo_abs_diff_both_pct": abs(actual["slo"]["both"] - new_sim["slo"]["both"]),
                "prev_slo_abs_diff_both_pct": None,
                "new_ftl_mean_ms_abs_ms": abs(actual["ftl"]["mean_ms"] - new_sim["ftl"]["mean_ms"]),
                "prev_ftl_mean_ms_abs_ms": None,
                "new_ftl_p50_ms_abs_ms": abs(actual["ftl"]["p50_ms"] - new_sim["ftl"]["p50_ms"]),
                "prev_ftl_p50_ms_abs_ms": None,
                "new_ftl_p95_ms_abs_ms": abs(actual["ftl"]["p95_ms"] - new_sim["ftl"]["p95_ms"]),
                "prev_ftl_p95_ms_abs_ms": None,
                "new_ftl_p99_ms_abs_ms": abs(actual["ftl"]["p99_ms"] - new_sim["ftl"]["p99_ms"]),
                "prev_ftl_p99_ms_abs_ms": None,
                "new_decode_mean_ms_abs_ms": abs(actual["decode"]["mean_ms"] - new_sim["decode"]["mean_ms"]),
                "prev_decode_mean_ms_abs_ms": None,
                "new_decode_p50_ms_abs_ms": abs(actual["decode"]["p50_ms"] - new_sim["decode"]["p50_ms"]),
                "prev_decode_p50_ms_abs_ms": None,
                "new_decode_p95_ms_abs_ms": abs(actual["decode"]["p95_ms"] - new_sim["decode"]["p95_ms"]),
                "prev_decode_p95_ms_abs_ms": None,
                "new_decode_p99_ms_abs_ms": abs(actual["decode"]["p99_ms"] - new_sim["decode"]["p99_ms"]),
                "prev_decode_p99_ms_abs_ms": None,
                "new_total_mean_ms_abs_ms": abs(actual["total"]["mean_ms"] - new_sim["total"]["mean_ms"]),
                "prev_total_mean_ms_abs_ms": None,
                "new_total_p50_ms_abs_ms": abs(actual["total"]["p50_ms"] - new_sim["total"]["p50_ms"]),
                "prev_total_p50_ms_abs_ms": None,
                "new_total_p95_ms_abs_ms": abs(actual["total"]["p95_ms"] - new_sim["total"]["p95_ms"]),
                "prev_total_p95_ms_abs_ms": None,
                "new_total_p99_ms_abs_ms": abs(actual["total"]["p99_ms"] - new_sim["total"]["p99_ms"]),
                "prev_total_p99_ms_abs_ms": None,
            }

            cases.append({
                "model": model_alias,
                "rate": float(rate),
                "exp_path": str(exp_path),
                "actual": actual,
                "new_sim": new_sim,
                "errors": errors,
            })

    summary = {
        "profile": str(args.profile),
        "output_root": str(args.output_root),
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

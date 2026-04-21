#!/usr/bin/env python3
"""
Validate a fitted vLLM-Ascend profile against holdout traces.

This script:
1. Replays the holdout workloads through `simulate_dist.py`.
2. Compares simulated per-request latencies to the real Ascend `.exp` traces.
3. Writes `summary.json` and `summary.csv` under the output root.

The CSV is intentionally compatible with `plot_validation_summary.py`.
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


REPO_ROOT = Path("/users/rh/DistServe")
SIMULATE_DIST = REPO_ROOT / "simdistserve" / "benchmarks" / "simulate_dist.py"

MODEL_CONFIGS = {
    "llama_1B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
        "workload_dir": "llama-3.2-1B",
        "real_dir": "llama1B",
    },
    "llama_3B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
        "workload_dir": "llama-3.2-3B",
        "real_dir": "llama3B",
    },
    "llama_7B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
        "workload_dir": "llama-2-7b",
        "real_dir": "llama7B",
    },
    "llama_8B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
        "workload_dir": "llama-3.1-8B",
        "real_dir": "llama8B",
    },
}

DEFAULT_MODELS = list(MODEL_CONFIGS.keys())
DEFAULT_RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a fitted vLLM-Ascend profile from scratch.")
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Fitted Ascend profile JSON.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where replay outputs and summary files will be written.",
    )
    parser.add_argument(
        "--workload-root",
        type=Path,
        default=REPO_ROOT / "simdistserve" / "dataset" / "splits" / "sharegpt_four_models_common_ascend1900_seed0",
        help="Root directory containing per-model holdout workloads.",
    )
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("/users/rh/ascend_data/ascend_vllm_holdout"),
        help="Root directory containing real Ascend holdout `.exp` traces.",
    )
    parser.add_argument(
        "--prev-summary",
        type=Path,
        default=Path("/users/rh/ascend_data/validation/refit_profile/summary.json"),
        help="Optional previous summary to compare against.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run simulate_dist.py.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=120)
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


def maybe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_actual_case_stats(exp_path: Path, prefill_slo_s: float, decode_slo_s: float, total_slo_s: float) -> dict[str, Any]:
    data = json.loads(exp_path.read_text())

    ftl_ms: list[float] = []
    decode_ms: list[float] = []
    total_ms: list[float] = []
    prefill_ok = decode_ok = total_ok = both_ok = 0

    for request in data:
        ftl_s = float(request.get("ftl", 0.0))
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


def load_prev_case_errors(prev_summary_path: Path | None) -> dict[tuple[str, str], dict[str, float | None]]:
    if prev_summary_path is None or not prev_summary_path.exists():
        return {}

    data = json.loads(prev_summary_path.read_text())
    result: dict[tuple[str, str], dict[str, float | None]] = {}
    for case in data.get("cases", []):
        key = (case["model"], str(case["rate"]))
        result[key] = {
            key_: maybe_float(value)
            for key_, value in case.get("errors", {}).items()
        }
    return result


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
    env["SIMDISTSERVE_VLLM_ASCEND_PROFILE"] = str(profile)

    cmd = [
        python_bin,
        str(SIMULATE_DIST),
        "--backend", "vllm_ascend",
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


def holdout_rate_label(rate: str) -> str:
    return "1.0" if str(rate) == "1" else str(rate)


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

    prev_case_errors = load_prev_case_errors(args.prev_summary)
    cases: list[dict[str, Any]] = []

    for model_alias in args.models:
        cfg = MODEL_CONFIGS[model_alias]
        workload_path = args.workload_root / cfg["workload_dir"] / "val.jsonl"
        if not workload_path.exists():
            raise FileNotFoundError(f"Workload file not found: {workload_path}")

        for rate in args.rates:
            output_dir = args.output_root / model_alias / f"rate_{rate}"
            exp_rate = holdout_rate_label(str(rate))
            exp_path = args.bench_root / cfg["real_dir"] / f"ascend-vllm-120-{exp_rate}.exp"
            if not exp_path.exists():
                raise FileNotFoundError(f"Real benchmark file not found: {exp_path}")

            print(f"RUN {model_alias} rate={rate}", flush=True)
            run_simulation(
                python_bin=args.python_bin,
                profile=args.profile,
                model_alias=model_alias,
                rate=str(rate),
                output_dir=output_dir,
                workload_path=workload_path,
                model_path=cfg["model_path"],
                num_prompts=args.num_prompts,
                seed=args.seed,
                arrival=args.arrival,
                cv=args.cv,
            )

            sim_csv = output_dir / "request_latency.csv"
            actual = compute_actual_case_stats(exp_path, args.prefill_slo, args.decode_slo, args.total_slo)
            new_sim = compute_sim_case_stats(sim_csv, args.prefill_slo, args.decode_slo, args.total_slo)

            prev_errors = prev_case_errors.get((model_alias, str(float(rate))))
            if prev_errors is None:
                prev_errors = prev_case_errors.get((model_alias, str(rate)), {})

            errors = {
                "new_slo_abs_diff_prefill_pct": abs(actual["slo"]["prefill"] - new_sim["slo"]["prefill"]),
                "prev_slo_abs_diff_prefill_pct": prev_errors.get("new_slo_abs_diff_prefill_pct"),
                "new_slo_abs_diff_decode_pct": abs(actual["slo"]["decode"] - new_sim["slo"]["decode"]),
                "prev_slo_abs_diff_decode_pct": prev_errors.get("new_slo_abs_diff_decode_pct"),
                "new_slo_abs_diff_total_pct": abs(actual["slo"]["total"] - new_sim["slo"]["total"]),
                "prev_slo_abs_diff_total_pct": prev_errors.get("new_slo_abs_diff_total_pct"),
                "new_slo_abs_diff_both_pct": abs(actual["slo"]["both"] - new_sim["slo"]["both"]),
                "prev_slo_abs_diff_both_pct": prev_errors.get("new_slo_abs_diff_both_pct"),
                "new_ftl_mean_ms_abs_ms": abs(actual["ftl"]["mean_ms"] - new_sim["ftl"]["mean_ms"]),
                "prev_ftl_mean_ms_abs_ms": prev_errors.get("new_ftl_mean_ms_abs_ms"),
                "new_ftl_p50_ms_abs_ms": abs(actual["ftl"]["p50_ms"] - new_sim["ftl"]["p50_ms"]),
                "prev_ftl_p50_ms_abs_ms": prev_errors.get("new_ftl_p50_ms_abs_ms"),
                "new_ftl_p95_ms_abs_ms": abs(actual["ftl"]["p95_ms"] - new_sim["ftl"]["p95_ms"]),
                "prev_ftl_p95_ms_abs_ms": prev_errors.get("new_ftl_p95_ms_abs_ms"),
                "new_ftl_p99_ms_abs_ms": abs(actual["ftl"]["p99_ms"] - new_sim["ftl"]["p99_ms"]),
                "prev_ftl_p99_ms_abs_ms": prev_errors.get("new_ftl_p99_ms_abs_ms"),
                "new_decode_mean_ms_abs_ms": abs(actual["decode"]["mean_ms"] - new_sim["decode"]["mean_ms"]),
                "prev_decode_mean_ms_abs_ms": prev_errors.get("new_decode_mean_ms_abs_ms"),
                "new_decode_p50_ms_abs_ms": abs(actual["decode"]["p50_ms"] - new_sim["decode"]["p50_ms"]),
                "prev_decode_p50_ms_abs_ms": prev_errors.get("new_decode_p50_ms_abs_ms"),
                "new_decode_p95_ms_abs_ms": abs(actual["decode"]["p95_ms"] - new_sim["decode"]["p95_ms"]),
                "prev_decode_p95_ms_abs_ms": prev_errors.get("new_decode_p95_ms_abs_ms"),
                "new_decode_p99_ms_abs_ms": abs(actual["decode"]["p99_ms"] - new_sim["decode"]["p99_ms"]),
                "prev_decode_p99_ms_abs_ms": prev_errors.get("new_decode_p99_ms_abs_ms"),
                "new_total_mean_ms_abs_ms": abs(actual["total"]["mean_ms"] - new_sim["total"]["mean_ms"]),
                "prev_total_mean_ms_abs_ms": prev_errors.get("new_total_mean_ms_abs_ms"),
                "new_total_p50_ms_abs_ms": abs(actual["total"]["p50_ms"] - new_sim["total"]["p50_ms"]),
                "prev_total_p50_ms_abs_ms": prev_errors.get("new_total_p50_ms_abs_ms"),
                "new_total_p95_ms_abs_ms": abs(actual["total"]["p95_ms"] - new_sim["total"]["p95_ms"]),
                "prev_total_p95_ms_abs_ms": prev_errors.get("new_total_p95_ms_abs_ms"),
                "new_total_p99_ms_abs_ms": abs(actual["total"]["p99_ms"] - new_sim["total"]["p99_ms"]),
                "prev_total_p99_ms_abs_ms": prev_errors.get("new_total_p99_ms_abs_ms"),
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

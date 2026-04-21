#!/usr/bin/env python3
"""Compare real benchmark .exp results against simulator replay CSV outputs.

This script is intended for the split-trace workflow:
1. real benchmark runs on an ordered split dataset
2. `build_replay_trace_from_exp.py` converts the real run into a replay trace
3. `simulate_dist.py --arrival custom` runs the simulator on that replay trace

The simulator request order may differ from the original split order because the
replay trace is sorted by actual issue time. We therefore align by `source_index`
from `request_info.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_mape(pairs: list[tuple[float, float]]) -> float | None:
    non_zero = [abs(pred - actual) / abs(actual) for actual, pred in pairs if actual != 0]
    if not non_zero:
        return None
    return 100.0 * statistics.mean(non_zero)


def summarize_metric(pairs: list[tuple[float, float]]) -> dict[str, float | None]:
    abs_errors = [abs(pred - actual) for actual, pred in pairs]
    signed_errors = [pred - actual for actual, pred in pairs]
    return {
        "samples": len(pairs),
        "mae_ms": statistics.mean(abs_errors) if abs_errors else None,
        "rmse_ms": math.sqrt(statistics.mean(err * err for err in signed_errors)) if signed_errors else None,
        "mean_signed_error_ms": statistics.mean(signed_errors) if signed_errors else None,
        "max_abs_error_ms": max(abs_errors) if abs_errors else None,
        "mape_pct": safe_mape(pairs),
    }


def summarize_attainment(
    rows: list[dict[str, Any]],
    ttft_key: str,
    tpot_key: str,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, float | int]:
    ttft_ok = [float(row[ttft_key]) <= ttft_slo_ms for row in rows]
    tpot_ok = [float(row[tpot_key]) <= tpot_slo_ms for row in rows]
    both_ok = [a and b for a, b in zip(ttft_ok, tpot_ok)]
    total = len(rows)
    return {
        "samples": total,
        "ttft_attainment_percent": 100.0 * sum(ttft_ok) / total if total else 0.0,
        "tpot_attainment_percent": 100.0 * sum(tpot_ok) / total if total else 0.0,
        "combined_attainment_percent": 100.0 * sum(both_ok) / total if total else 0.0,
    }


def build_real_rows(exp_path: Path) -> list[dict[str, Any]]:
    payload = load_json(exp_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected {exp_path} to contain a list of request results.")

    rows = []
    for source_index, item in enumerate(payload):
        start_time = float(item["start_time"])
        end_time = float(item["end_time"])
        token_timestamps = [float(ts) for ts in item.get("token_timestamps", [])]
        output_len = int(item["output_len"])
        rows.append(
            {
                "source_index": source_index,
                "prompt_len": int(item["prompt_len"]),
                "output_len": output_len,
                "real_ttft_ms": (
                    (token_timestamps[0] - start_time) * 1000.0 if token_timestamps else 0.0
                ),
                "real_tpot_ms": (
                    0.0
                    if output_len <= 1 or len(token_timestamps) < 2
                    else (token_timestamps[-1] - token_timestamps[0]) * 1000.0 / (output_len - 1)
                ),
                "real_total_latency_ms": (end_time - start_time) * 1000.0,
            }
        )
    return rows


def build_sim_rows(request_info_path: Path, request_latency_path: Path) -> list[dict[str, Any]]:
    info_rows = load_csv_rows(request_info_path)
    latency_rows = load_csv_rows(request_latency_path)
    if len(info_rows) != len(latency_rows):
        raise ValueError(
            f"CSV length mismatch: {request_info_path} has {len(info_rows)} rows but "
            f"{request_latency_path} has {len(latency_rows)} rows."
        )

    info_by_req_id = {int(row["req_id"]): row for row in info_rows}
    sim_rows = []
    for row_index, latency_row in enumerate(latency_rows):
        if "req_id" in latency_row and latency_row["req_id"] != "":
            req_id = int(latency_row["req_id"])
        else:
            req_id = int(info_rows[row_index]["req_id"])
        info_row = info_by_req_id[req_id]
        sim_rows.append(
            {
                "req_id": req_id,
                "source_index": int(info_row["source_index"]) if "source_index" in info_row else req_id,
                "prompt_len": int(info_row["prefill_lens"]),
                "output_len": int(info_row["output_lens"]),
                "sim_ttft_ms": float(latency_row["first_token_latency"]),
                "sim_tpot_ms": float(latency_row["tpot"]),
                "sim_total_latency_ms": float(latency_row["total_latency"]),
            }
        )
    return sim_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", type=Path, required=True, help="Path to the real benchmark .exp file.")
    parser.add_argument(
        "--request-info",
        type=Path,
        required=True,
        help="Path to simulator request_info.csv.",
    )
    parser.add_argument(
        "--request-latency",
        type=Path,
        required=True,
        help="Path to simulator request_latency.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional merged per-request comparison CSV.",
    )
    parser.add_argument(
        "--ftl-slo-ms",
        type=float,
        default=None,
        help="Optional TTFT/FTL SLO threshold in milliseconds.",
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=None,
        help="Optional TPOT SLO threshold in milliseconds.",
    )
    args = parser.parse_args()

    real_rows = build_real_rows(args.exp)
    sim_rows = build_sim_rows(args.request_info, args.request_latency)

    real_by_source_index = {row["source_index"]: row for row in real_rows}
    sim_by_source_index = {row["source_index"]: row for row in sim_rows}
    if set(real_by_source_index) != set(sim_by_source_index):
        raise ValueError(
            "Source-index mismatch between real benchmark and simulator outputs: "
            f"real has {len(real_by_source_index)} rows, simulator has {len(sim_by_source_index)} rows."
        )

    merged_rows = []
    for source_index in sorted(real_by_source_index):
        real_row = real_by_source_index[source_index]
        sim_row = sim_by_source_index[source_index]
        if (
            real_row["prompt_len"] != sim_row["prompt_len"]
            or real_row["output_len"] != sim_row["output_len"]
        ):
            raise ValueError(
                f"Prompt/output mismatch at source_index={source_index}: "
                f"real ({real_row['prompt_len']}, {real_row['output_len']}) vs "
                f"sim ({sim_row['prompt_len']}, {sim_row['output_len']})."
            )

        merged_rows.append(
            {
                "source_index": source_index,
                "sim_req_id": sim_row["req_id"],
                "prompt_len": real_row["prompt_len"],
                "output_len": real_row["output_len"],
                "real_ttft_ms": real_row["real_ttft_ms"],
                "sim_ttft_ms": sim_row["sim_ttft_ms"],
                "ttft_error_ms": sim_row["sim_ttft_ms"] - real_row["real_ttft_ms"],
                "real_tpot_ms": real_row["real_tpot_ms"],
                "sim_tpot_ms": sim_row["sim_tpot_ms"],
                "tpot_error_ms": sim_row["sim_tpot_ms"] - real_row["real_tpot_ms"],
                "real_total_latency_ms": real_row["real_total_latency_ms"],
                "sim_total_latency_ms": sim_row["sim_total_latency_ms"],
                "total_latency_error_ms": sim_row["sim_total_latency_ms"] - real_row["real_total_latency_ms"],
            }
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(merged_rows[0].keys()) if merged_rows else [])
            if merged_rows:
                writer.writeheader()
                writer.writerows(merged_rows)

    ttft_pairs = [(row["real_ttft_ms"], row["sim_ttft_ms"]) for row in merged_rows]
    tpot_pairs = [(row["real_tpot_ms"], row["sim_tpot_ms"]) for row in merged_rows]
    total_pairs = [(row["real_total_latency_ms"], row["sim_total_latency_ms"]) for row in merged_rows]
    summary = {
        "exp": str(args.exp),
        "request_info": str(args.request_info),
        "request_latency": str(args.request_latency),
        "samples": len(merged_rows),
        "ttft": summarize_metric(ttft_pairs),
        "tpot": summarize_metric(tpot_pairs),
        "total_latency": summarize_metric(total_pairs),
    }
    if (args.ftl_slo_ms is None) != (args.tpot_slo_ms is None):
        raise ValueError("Both --ftl-slo-ms and --tpot-slo-ms must be provided together.")
    if args.ftl_slo_ms is not None and args.tpot_slo_ms is not None:
        real_attainment = summarize_attainment(
            merged_rows,
            ttft_key="real_ttft_ms",
            tpot_key="real_tpot_ms",
            ttft_slo_ms=args.ftl_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        sim_attainment = summarize_attainment(
            merged_rows,
            ttft_key="sim_ttft_ms",
            tpot_key="sim_tpot_ms",
            ttft_slo_ms=args.ftl_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        real_combined = float(real_attainment["combined_attainment_percent"])
        sim_combined = float(sim_attainment["combined_attainment_percent"])
        summary["slo_attainment"] = {
            "ttft_slo_ms": args.ftl_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "real": real_attainment,
            "sim": sim_attainment,
            "combined_error_pct_points": sim_combined - real_combined,
            "combined_abs_error_pct_points": abs(sim_combined - real_combined),
            "combined_relative_error_percent": (
                abs(sim_combined - real_combined) / real_combined * 100.0
                if real_combined != 0.0
                else None
            ),
        }
    if args.output_csv is not None:
        summary["output_csv"] = str(args.output_csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

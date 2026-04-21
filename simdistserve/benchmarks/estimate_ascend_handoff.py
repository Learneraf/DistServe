#!/usr/bin/env python3
"""
Estimate the Ascend prefill-to-proxy TTFT residual from matched grid runs.

This compares compute-grid and proxy-grid `case_summaries.json` files on the
shared key:

    (model_id, batch_size, input_len_actual, output_len_target)

and reports TTFT deltas in milliseconds. A practical same-node fixed handoff
estimate is the batch-size-1 median delta.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list payload in {path}")
    return payload


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "mean_ms": None,
            "median_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Ascend TTFT handoff residual from matched grid summaries.")
    parser.add_argument(
        "--compute",
        type=Path,
        default=Path("/users/rh/ascend_data/ascend_compute_grid/case_summaries.json"),
        help="Compute-grid case_summaries.json path.",
    )
    parser.add_argument(
        "--proxy",
        type=Path,
        default=Path("/users/rh/ascend_data/ascend_proxy_grid/case_summaries.json"),
        help="Proxy-grid case_summaries.json path.",
    )
    parser.add_argument(
        "--ttft-stat",
        type=str,
        default="mean_ttft",
        choices=["mean_ttft", "p50_ttft", "p95_ttft"],
        help="Which TTFT statistic to compare.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    compute_rows = load_rows(args.compute)
    proxy_rows = load_rows(args.proxy)

    compute_by_key = {}
    for row in compute_rows:
        if int(row.get("failed_requests", 0)) > 0 or row.get(args.ttft_stat) is None:
            continue
        key = (
            row.get("model_id"),
            int(row["batch_size"]),
            int(row["input_len_actual"]),
            int(row["output_len_target"]),
        )
        compute_by_key[key] = row

    matched_deltas: list[dict] = []
    by_batch_size: dict[int, list[float]] = defaultdict(list)
    by_model: dict[str, list[float]] = defaultdict(list)

    for row in proxy_rows:
        if int(row.get("failed_requests", 0)) > 0 or row.get(args.ttft_stat) is None:
            continue
        key = (
            row.get("model_id"),
            int(row["batch_size"]),
            int(row["input_len_actual"]),
            int(row["output_len_target"]),
        )
        compute_row = compute_by_key.get(key)
        if compute_row is None:
            continue

        delta_ms = 1000.0 * float(row[args.ttft_stat] - compute_row[args.ttft_stat])
        matched_deltas.append(
            {
                "model_id": row.get("model_id"),
                "batch_size": int(row["batch_size"]),
                "input_len_actual": int(row["input_len_actual"]),
                "output_len_target": int(row["output_len_target"]),
                "delta_ttft_ms": delta_ms,
            }
        )
        by_batch_size[int(row["batch_size"])].append(delta_ms)
        by_model[str(row.get("model_id"))].append(delta_ms)

    overall_values = [row["delta_ttft_ms"] for row in matched_deltas]
    batch1_values = by_batch_size.get(1, [])

    result = {
        "compute_path": str(args.compute),
        "proxy_path": str(args.proxy),
        "ttft_stat": args.ttft_stat,
        "matched_cases": len(matched_deltas),
        "overall": summarize(overall_values),
        "by_batch_size": {
            str(batch_size): summarize(values)
            for batch_size, values in sorted(by_batch_size.items())
        },
        "by_model": {
            model_id: summarize(values)
            for model_id, values in sorted(by_model.items())
        },
        "recommended_fixed_handoff_ms": statistics.median(batch1_values) if batch1_values else None,
        "matched_deltas": matched_deltas,
    }

    print(json.dumps(result, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

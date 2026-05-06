#!/usr/bin/env python3
"""Fit DistServe component profiles from two-phase explicit batch logs.

Input files are produced by:

    evaluation/0-test-single-forward-performance/run_dataset_arrival_profile.py

Unlike the serving-result fitter, this script does not infer batches from
request timestamps. It consumes explicit prefill batch and decode round records.


python \
  /users/rh/DistServe/simdistserve/estimators/fit_params/fit_explicit_batch.py \
  --input-root /users/rh/DistServe/evaluation/0-test-single-forward-performance/result/llama8B \
  --model-key /users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2 \
  --tp-world-size 2 \
  --output /users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_tp2_llama8B.json

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PrefillSample:
    batch_size: int
    sum_prompt_len: int
    max_prompt_len: int
    sum_prompt_len_sq: int
    duration_ms: float


@dataclass
class DecodeSample:
    batch_size: int
    sum_context_len: int
    max_context_len: int
    sum_remaining_output_tokens: int
    duration_ms: float


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def discover_two_phase_files(input_root: Path) -> tuple[list[Path], list[Path]]:
    prefill_names = {"tp2_prefill_trace.jsonl", "prefill_trace.jsonl"}
    decode_names = {"tp2_decode_rounds.jsonl", "decode_rounds.jsonl"}
    prefill_traces = sorted(
        path for path in input_root.rglob("*.jsonl")
        if path.name in prefill_names
    )
    decode_rounds = sorted(
        path for path in input_root.rglob("*.jsonl")
        if path.name in decode_names
    )
    return prefill_traces, decode_rounds


def load_prefill_samples(path: Path) -> list[PrefillSample]:
    rows = read_jsonl(path)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["batch_id"]), []).append(row)

    samples = []
    for batch_id, batch_rows in sorted(grouped.items()):
        prompt_lens = _prefill_prompt_lens(batch_rows)
        durations = [float(row["prefill_latency_ms"]) for row in batch_rows]
        samples.append(
            PrefillSample(
                batch_size=len(prompt_lens),
                sum_prompt_len=sum(prompt_lens),
                max_prompt_len=max(prompt_lens),
                sum_prompt_len_sq=sum(length * length for length in prompt_lens),
                duration_ms=statistics.median(durations),
            )
        )
    return samples


def load_prefill_samples_many(paths: list[Path]) -> list[PrefillSample]:
    samples: list[PrefillSample] = []
    for path in paths:
        path_samples = load_prefill_samples(path)
        print(f"Loaded {len(path_samples)} prefill sample(s) from {path}")
        samples.extend(path_samples)
    return samples


def _prefill_prompt_lens(batch_rows: list[dict[str, Any]]) -> list[int]:
    first = batch_rows[0]
    batch_prompt_lens = first.get("batch_prompt_lens")
    if isinstance(batch_prompt_lens, list) and batch_prompt_lens:
        return [int(value) for value in batch_prompt_lens]
    return [int(row["prompt_len"]) for row in batch_rows]


def load_decode_samples(path: Path) -> list[DecodeSample]:
    samples = []
    for row in read_jsonl(path):
        current_context_lens = [int(value) for value in row["current_context_lens"]]
        remaining_output_tokens = [int(value) for value in row["remaining_output_tokens"]]
        samples.append(
            DecodeSample(
                batch_size=int(row["batch_size"]),
                sum_context_len=sum(current_context_lens),
                max_context_len=max(current_context_lens),
                sum_remaining_output_tokens=sum(remaining_output_tokens),
                duration_ms=float(row["forward_time_ms"]),
            )
        )
    return samples


def load_decode_samples_many(paths: list[Path]) -> list[DecodeSample]:
    samples: list[DecodeSample] = []
    for path in paths:
        path_samples = load_decode_samples(path)
        print(f"Loaded {len(path_samples)} decode sample(s) from {path}")
        samples.extend(path_samples)
    return samples


def fit_prefill(samples: list[PrefillSample]) -> tuple[list[float], dict[str, float]]:
    """Fit: ms = a + b*bs + c*sum_len + d*max_len + e*sum_len_sq."""
    if not samples:
        raise ValueError("No prefill samples.")
    rows = [
        [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
        ]
        for sample in samples
    ]
    durations = [max(float(sample.duration_ms), 1e-6) for sample in samples]
    return fit_least_squares(rows, durations)


def fit_decode(samples: list[DecodeSample], include_remaining: bool) -> tuple[list[float], dict[str, float]]:
    """Fit decode round time.

    Default 4-param form:
        ms = a + b*bs + c*sum_context_len + d*max_context_len

    Optional 5-param form:
        ms = a + b*bs + c*sum_context_len + d*max_context_len
             + e*sum_remaining_output_tokens
    """
    if not samples:
        raise ValueError("No decode samples.")
    rows = []
    for sample in samples:
        row = [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.max_context_len),
        ]
        if include_remaining:
            row.append(float(sample.sum_remaining_output_tokens))
        rows.append(row)
    durations = [max(float(sample.duration_ms), 1e-6) for sample in samples]
    return fit_least_squares(rows, durations)


def fit_least_squares(rows: list[list[float]], durations: list[float]) -> tuple[list[float], dict[str, float]]:
    matrix = np.array(rows, dtype=float)
    target = np.array(durations, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    predictions = matrix @ coeffs
    rel_errors = (predictions - target) / target
    abs_rel_errors = np.abs(rel_errors)
    metrics = {
        "count": float(len(target)),
        "mean_abs_rel_error_pct": float(100.0 * np.mean(abs_rel_errors)),
        "rmse_rel_error_pct": float(100.0 * math.sqrt(float(np.mean(rel_errors * rel_errors)))),
        "max_abs_rel_error_pct": float(100.0 * np.max(abs_rel_errors)),
    }
    return [float(value) for value in coeffs], metrics


def write_prefill_samples_csv(path: Path, samples: list[PrefillSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size",
            "sum_prompt_len",
            "max_prompt_len",
            "sum_prompt_len_sq",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.batch_size,
                sample.sum_prompt_len,
                sample.max_prompt_len,
                sample.sum_prompt_len_sq,
                sample.duration_ms,
            ])


def write_decode_samples_csv(path: Path, samples: list[DecodeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "batch_size",
            "sum_context_len",
            "max_context_len",
            "sum_remaining_output_tokens",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.batch_size,
                sample.sum_context_len,
                sample.max_context_len,
                sample.sum_remaining_output_tokens,
                sample.duration_ms,
            ])


def update_profile(
    existing_profile: Path | None,
    model_key: str,
    tp_world_size: int,
    prefill_coeffs: list[float],
    decode_coeffs: list[float],
    decode_large_small_bs_threshold: int,
) -> dict[str, Any]:
    if existing_profile is not None and existing_profile.exists():
        profile = json.loads(existing_profile.read_text())
    else:
        profile = {}

    model_profile = profile.setdefault(model_key, {})
    tp_profile = model_profile.setdefault(str(tp_world_size), {})
    tp_profile["prefill"] = prefill_coeffs
    tp_profile["decoding_smallbs"] = decode_coeffs
    tp_profile["decoding_largebs"] = decode_coeffs
    tp_profile["decoding_large_small_bs_threshold"] = decode_large_small_bs_threshold
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit profile from explicit two-phase batch logs.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing per-rate two-phase outputs. Recursively "
            "loads tp2_prefill_trace.jsonl and tp2_decode_rounds.jsonl."
        ),
    )
    parser.add_argument(
        "--prefill-trace",
        type=Path,
        action="append",
        default=[],
        help="Explicit prefill trace JSONL. Can be passed multiple times.",
    )
    parser.add_argument(
        "--decode-rounds",
        type=Path,
        action="append",
        default=[],
        help="Explicit decode rounds JSONL. Can be passed multiple times.",
    )
    parser.add_argument("--model-key", required=True, help="Model key used in the output profile JSON.")
    parser.add_argument("--tp-world-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--existing-profile",
        type=Path,
        default=None,
        help="Optional profile JSON to update instead of creating a new one.",
    )
    parser.add_argument(
        "--decode-include-remaining",
        action="store_true",
        help="Fit a 5-parameter decode model including sum_remaining_output_tokens.",
    )
    parser.add_argument(
        "--decode-large-small-bs-threshold",
        type=int,
        default=95,
        help="Compatibility threshold retained in the output JSON.",
    )
    parser.add_argument("--prefill-samples-csv", type=Path, default=None)
    parser.add_argument("--decode-samples-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefill_trace_paths = list(args.prefill_trace)
    decode_round_paths = list(args.decode_rounds)
    if args.input_root is not None:
        discovered_prefill, discovered_decode = discover_two_phase_files(args.input_root)
        prefill_trace_paths.extend(discovered_prefill)
        decode_round_paths.extend(discovered_decode)

    # Deduplicate while preserving order.
    prefill_trace_paths = list(dict.fromkeys(prefill_trace_paths))
    decode_round_paths = list(dict.fromkeys(decode_round_paths))
    if not prefill_trace_paths:
        raise ValueError("No prefill trace files provided. Use --prefill-trace or --input-root.")
    if not decode_round_paths:
        raise ValueError("No decode round files provided. Use --decode-rounds or --input-root.")

    print(f"Prefill trace files: {len(prefill_trace_paths)}")
    print(f"Decode round files: {len(decode_round_paths)}")
    prefill_samples = load_prefill_samples_many(prefill_trace_paths)
    decode_samples = load_decode_samples_many(decode_round_paths)

    if args.prefill_samples_csv is not None:
        write_prefill_samples_csv(args.prefill_samples_csv, prefill_samples)
    if args.decode_samples_csv is not None:
        write_decode_samples_csv(args.decode_samples_csv, decode_samples)

    prefill_coeffs, prefill_metrics = fit_prefill(prefill_samples)
    decode_coeffs, decode_metrics = fit_decode(
        decode_samples,
        include_remaining=args.decode_include_remaining,
    )

    profile = update_profile(
        existing_profile=args.existing_profile,
        model_key=args.model_key,
        tp_world_size=args.tp_world_size,
        prefill_coeffs=prefill_coeffs,
        decode_coeffs=decode_coeffs,
        decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=4) + "\n")

    print(f"Prefill samples: {len(prefill_samples)}")
    print(f"Prefill coeffs: {prefill_coeffs}")
    print(f"Prefill fit: {prefill_metrics}")
    print(f"Decode samples: {len(decode_samples)}")
    print(f"Decode coeffs: {decode_coeffs}")
    print(f"Decode fit: {decode_metrics}")
    print(f"Wrote profile: {args.output}")


if __name__ == "__main__":
    main()

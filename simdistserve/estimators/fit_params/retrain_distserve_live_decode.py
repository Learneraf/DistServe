#!/usr/bin/env python3
"""
Retrain DistServe decode parameters from real benchmark .exp files.

This script reconstructs live decode rounds by clustering near-simultaneous token
timestamps from the real DistServe benchmark results. It then fits a decode-round
latency model using batch-state features:

    round_ms = A
             + B * batch_size
             + C * sum_current_context_len
             + D * max_current_context_len

The output JSON preserves the existing prefill fit and any untouched model/TP
entries, while replacing TP=1 decode coefficients for the models that have
benchmark results available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_ALIAS_TO_KEY = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "huggyllama/llama-7b",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}


@dataclass
class TokenEvent:
    timestamp: float
    interval_ms: float
    prompt_len: int
    output_len: int
    token_idx: int
    current_context_len: int
    remaining_output_tokens: int


@dataclass
class RoundSample:
    model_key: str
    source_file: str
    rate: float
    batch_size: int
    sum_context_len: int
    max_context_len: int
    sum_remaining_output_tokens: int
    duration_ms: float


def infer_model_key(exp_path: Path) -> str | None:
    parent_name = exp_path.parent.name
    return MODEL_ALIAS_TO_KEY.get(parent_name)


def infer_rate(exp_path: Path) -> float:
    match = re.search(r"distserve-\d+-([\d.]+)\.exp$", exp_path.name)
    if not match:
        raise ValueError(f"Cannot infer request rate from {exp_path}")
    return float(match.group(1))


def extract_decode_events(exp_path: Path) -> tuple[str | None, list[TokenEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_key = infer_model_key(exp_path)
    rate = infer_rate(exp_path)
    events: list[TokenEvent] = []

    for request in requests:
        prompt_len = int(request["prompt_len"])
        output_len = int(request["output_len"])
        token_timestamps = request.get("token_timestamps", [])
        if len(token_timestamps) != output_len or output_len <= 1:
            continue

        # Skip the first token because the client-side timestamps do not share the
        # same clock as server-side lifecycle events, so we cannot isolate the
        # first decode round from prefill/migration robustly.
        for token_idx in range(1, output_len):
            interval_ms = (token_timestamps[token_idx] - token_timestamps[token_idx - 1]) * 1000.0
            current_context_len = prompt_len + token_idx
            remaining_output_tokens = output_len - token_idx
            events.append(
                TokenEvent(
                    timestamp=token_timestamps[token_idx],
                    interval_ms=interval_ms,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    token_idx=token_idx,
                    current_context_len=current_context_len,
                    remaining_output_tokens=remaining_output_tokens,
                )
            )

    return model_key, events, rate


def cluster_events_into_rounds(
    model_key: str,
    exp_path: Path,
    rate: float,
    events: list[TokenEvent],
    cluster_gap_ms: float,
) -> list[RoundSample]:
    if not events:
        return []

    events = sorted(events, key=lambda event: event.timestamp)
    cluster_gap_s = cluster_gap_ms / 1000.0
    clusters: list[list[TokenEvent]] = []
    current_cluster: list[TokenEvent] = [events[0]]

    for event in events[1:]:
        if (event.timestamp - current_cluster[-1].timestamp) <= cluster_gap_s:
            current_cluster.append(event)
        else:
            clusters.append(current_cluster)
            current_cluster = [event]
    clusters.append(current_cluster)

    round_samples: list[RoundSample] = []
    for cluster in clusters:
        duration_ms = statistics.median(event.interval_ms for event in cluster)
        round_samples.append(
            RoundSample(
                model_key=model_key,
                source_file=str(exp_path),
                rate=rate,
                batch_size=len(cluster),
                sum_context_len=sum(event.current_context_len for event in cluster),
                max_context_len=max(event.current_context_len for event in cluster),
                sum_remaining_output_tokens=sum(event.remaining_output_tokens for event in cluster),
                duration_ms=duration_ms,
            )
        )

    return round_samples


def fit_relative_error_linear_model(samples: list[RoundSample]) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No samples to fit.")

    design_matrix = []
    rhs = []
    raw_rows = []
    durations = []

    for sample in samples:
        row = [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.max_context_len),
        ]
        duration = max(sample.duration_ms, 1e-6)
        design_matrix.append([value / duration for value in row])
        rhs.append(1.0)
        raw_rows.append(row)
        durations.append(duration)

    coeffs, _, _, _ = np.linalg.lstsq(np.array(design_matrix), np.array(rhs), rcond=None)
    predictions = [float(np.dot(coeffs, row)) for row in raw_rows]
    rel_errors = [
        (prediction - duration) / duration
        for prediction, duration in zip(predictions, durations)
    ]
    abs_rel_errors = [abs(error) for error in rel_errors]

    metrics = {
        "count": float(len(samples)),
        "mean_abs_rel_error_pct": 100.0 * statistics.mean(abs_rel_errors),
        "rmse_rel_error_pct": 100.0 * math.sqrt(statistics.mean(error * error for error in rel_errors)),
        "max_abs_rel_error_pct": 100.0 * max(abs_rel_errors),
    }
    return coeffs.tolist(), metrics


def load_base_profile(path: Path) -> dict:
    if not path.exists():
        fallback_path = path.parent / "fit_params.json"
        if fallback_path.exists():
            path = fallback_path
    with open(path) as f:
        return json.load(f)


def discover_exp_files(results_root: Path) -> list[Path]:
    return sorted(path for path in results_root.rglob("*.exp") if path.is_file())


def write_rounds_csv(path: Path, samples: list[RoundSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "source_file",
            "rate",
            "batch_size",
            "sum_context_len",
            "max_context_len",
            "sum_remaining_output_tokens",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.model_key,
                sample.source_file,
                sample.rate,
                sample.batch_size,
                sample.sum_context_len,
                sample.max_context_len,
                sample.sum_remaining_output_tokens,
                sample.duration_ms,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain DistServe live decode parameters from benchmark .exp files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result"),
        help="Root directory containing DistServe benchmark .exp files.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live.json"),
        help="Existing DistServe fit JSON used as the merge base.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_decode.json"),
        help="Output path for the merged profile JSON.",
    )
    parser.add_argument(
        "--cluster-gap-ms",
        type=float,
        default=1.0,
        help="Maximum timestamp gap, in milliseconds, to group token emissions into the same decode round.",
    )
    parser.add_argument(
        "--decode-large-small-bs-threshold",
        type=int,
        default=95,
        help="Batch-size threshold used to split small/large decode fits.",
    )
    parser.add_argument(
        "--rounds-csv",
        type=Path,
        default=None,
        help="Optional CSV output for reconstructed round samples.",
    )
    args = parser.parse_args()

    base_profile = load_base_profile(args.base_profile)
    round_samples_by_model: dict[str, list[RoundSample]] = {}

    exp_files = discover_exp_files(args.results_root)
    if not exp_files:
        raise FileNotFoundError(f"No .exp files found under {args.results_root}")

    ignored_files = []
    for exp_path in exp_files:
        model_key, events, rate = extract_decode_events(exp_path)
        if model_key is None:
            ignored_files.append(str(exp_path))
            continue
        round_samples = cluster_events_into_rounds(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=args.cluster_gap_ms,
        )
        round_samples_by_model.setdefault(model_key, []).extend(round_samples)

    all_round_samples = [sample for samples in round_samples_by_model.values() for sample in samples]
    if args.rounds_csv is not None:
        write_rounds_csv(args.rounds_csv, all_round_samples)

    print(f"Discovered {len(exp_files)} exp files")
    if ignored_files:
        print(f"Ignored {len(ignored_files)} files with unknown model aliases")
        for path in ignored_files:
            print(f"  - {path}")

    threshold = args.decode_large_small_bs_threshold
    for model_key, samples in sorted(round_samples_by_model.items()):
        if model_key not in base_profile:
            print(f"Skipping {model_key}: not present in base profile")
            continue
        if "1" not in base_profile[model_key]:
            print(f"Skipping {model_key}: TP=1 missing in base profile")
            continue

        small_bs_samples = [sample for sample in samples if sample.batch_size < threshold]
        large_bs_samples = [sample for sample in samples if sample.batch_size >= threshold]

        print(f"\nModel: {model_key}")
        print(f"  total reconstructed rounds: {len(samples)}")
        print(f"  small-batch rounds: {len(small_bs_samples)}")
        print(f"  large-batch rounds: {len(large_bs_samples)}")

        tp_profile = base_profile[model_key]["1"]

        if small_bs_samples:
            small_coeffs, small_metrics = fit_relative_error_linear_model(small_bs_samples)
            tp_profile["decoding_smallbs"] = small_coeffs
            print(f"  small-batch coeffs: {small_coeffs}")
            print(f"  small-batch fit: {small_metrics}")
        else:
            print("  small-batch coeffs unchanged: no samples")

        if large_bs_samples:
            large_coeffs, large_metrics = fit_relative_error_linear_model(large_bs_samples)
            tp_profile["decoding_largebs"] = large_coeffs
            print(f"  large-batch coeffs: {large_coeffs}")
            print(f"  large-batch fit: {large_metrics}")
        else:
            if not tp_profile.get("decoding_largebs"):
                tp_profile["decoding_largebs"] = tp_profile.get("decoding_smallbs", [])
            print("  large-batch coeffs fallback to small-batch fit: no samples")

        tp_profile["decoding_large_small_bs_threshold"] = threshold

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(base_profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote merged live-decode profile to {args.output}")


if __name__ == "__main__":
    main()

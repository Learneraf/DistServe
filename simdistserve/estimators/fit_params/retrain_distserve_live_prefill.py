#!/usr/bin/env python3
"""
Retrain DistServe prefill parameters from real benchmark .exp files.

This script reconstructs live prefill batches from request lifecycle events and fits
the prefill batch runtime as:

    round_ms = A
             + B * batch_size
             + C * sum_prompt_len
             + D * max_prompt_len
             + E * sum_prompt_len_sq

The output JSON preserves any existing decode coefficients and untouched model/TP
entries while replacing TP=1 prefill coefficients for the models that have
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
class PrefillEvent:
    timestamp: float
    end_timestamp: float
    prompt_len: int


@dataclass
class PrefillBatchSample:
    model_key: str
    source_file: str
    rate: float
    batch_size: int
    sum_prompt_len: int
    max_prompt_len: int
    sum_prompt_len_sq: int
    duration_ms: float


def infer_model_key(exp_path: Path) -> str | None:
    return MODEL_ALIAS_TO_KEY.get(exp_path.parent.name)


def infer_rate(exp_path: Path) -> float:
    match = re.search(r"distserve-\d+-([\d.]+)\.exp$", exp_path.name)
    if not match:
        raise ValueError(f"Cannot infer request rate from {exp_path}")
    return float(match.group(1))


def extract_prefill_events(exp_path: Path) -> tuple[str | None, list[PrefillEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_key = infer_model_key(exp_path)
    rate = infer_rate(exp_path)
    events: list[PrefillEvent] = []

    for request in requests:
        lifecycle = {event["event_type"]: event["timestamp"] for event in request.get("lifecycle_events", [])}
        context_begin = lifecycle.get("context_begin")
        context_end = lifecycle.get("context_end")
        if context_begin is None or context_end is None:
            continue
        events.append(
            PrefillEvent(
                timestamp=float(context_begin),
                end_timestamp=float(context_end),
                prompt_len=int(request["prompt_len"]),
            )
        )

    return model_key, events, rate


def cluster_events_into_batches(
    model_key: str,
    exp_path: Path,
    rate: float,
    events: list[PrefillEvent],
    cluster_gap_ms: float,
) -> list[PrefillBatchSample]:
    if not events:
        return []

    events = sorted(events, key=lambda event: event.timestamp)
    cluster_gap_s = cluster_gap_ms / 1000.0
    clusters: list[list[PrefillEvent]] = []
    current_cluster: list[PrefillEvent] = [events[0]]

    for event in events[1:]:
        if (event.timestamp - current_cluster[-1].timestamp) <= cluster_gap_s:
            current_cluster.append(event)
        else:
            clusters.append(current_cluster)
            current_cluster = [event]
    clusters.append(current_cluster)

    batch_samples: list[PrefillBatchSample] = []
    for cluster in clusters:
        prompt_lens = [event.prompt_len for event in cluster]
        batch_samples.append(
            PrefillBatchSample(
                model_key=model_key,
                source_file=str(exp_path),
                rate=rate,
                batch_size=len(cluster),
                sum_prompt_len=sum(prompt_lens),
                max_prompt_len=max(prompt_lens),
                sum_prompt_len_sq=sum(prompt_len * prompt_len for prompt_len in prompt_lens),
                duration_ms=1000.0 * statistics.median(
                    event.end_timestamp - event.timestamp for event in cluster
                ),
            )
        )

    return batch_samples


def fit_relative_error_linear_model(
    samples: list[PrefillBatchSample],
) -> tuple[list[float], dict[str, float]]:
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
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
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


def write_batches_csv(path: Path, samples: list[PrefillBatchSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "source_file",
            "rate",
            "batch_size",
            "sum_prompt_len",
            "max_prompt_len",
            "sum_prompt_len_sq",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.model_key,
                sample.source_file,
                sample.rate,
                sample.batch_size,
                sample.sum_prompt_len,
                sample.max_prompt_len,
                sample.sum_prompt_len_sq,
                sample.duration_ms,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain DistServe live prefill parameters from benchmark .exp files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result"),
        help="Root directory containing DistServe benchmark .exp files.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_decode.json"),
        help="Existing DistServe fit JSON used as the merge base.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live.json"),
        help="Output path for the merged profile JSON.",
    )
    parser.add_argument(
        "--cluster-gap-ms",
        type=float,
        default=1.0,
        help="Maximum timestamp gap, in milliseconds, to group requests into the same prefill batch.",
    )
    parser.add_argument(
        "--batches-csv",
        type=Path,
        default=None,
        help="Optional CSV output for reconstructed prefill batches.",
    )
    args = parser.parse_args()

    base_profile = load_base_profile(args.base_profile)
    batch_samples_by_model: dict[str, list[PrefillBatchSample]] = {}

    exp_files = discover_exp_files(args.results_root)
    if not exp_files:
        raise FileNotFoundError(f"No .exp files found under {args.results_root}")

    ignored_files = []
    for exp_path in exp_files:
        model_key, events, rate = extract_prefill_events(exp_path)
        if model_key is None:
            ignored_files.append(str(exp_path))
            continue
        batch_samples = cluster_events_into_batches(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=args.cluster_gap_ms,
        )
        batch_samples_by_model.setdefault(model_key, []).extend(batch_samples)

    all_batch_samples = [sample for samples in batch_samples_by_model.values() for sample in samples]
    if args.batches_csv is not None:
        write_batches_csv(args.batches_csv, all_batch_samples)

    print(f"Discovered {len(exp_files)} exp files")
    if ignored_files:
        print(f"Ignored {len(ignored_files)} files with unknown model aliases")
        for path in ignored_files:
            print(f"  - {path}")

    for model_key, samples in sorted(batch_samples_by_model.items()):
        if model_key not in base_profile:
            print(f"Skipping {model_key}: not present in base profile")
            continue
        if "1" not in base_profile[model_key]:
            print(f"Skipping {model_key}: TP=1 missing in base profile")
            continue

        coeffs, metrics = fit_relative_error_linear_model(samples)
        base_profile[model_key]["1"]["prefill"] = coeffs

        print(f"\nModel: {model_key}")
        print(f"  total reconstructed prefill batches: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(base_profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote merged live profile to {args.output}")


if __name__ == "__main__":
    main()

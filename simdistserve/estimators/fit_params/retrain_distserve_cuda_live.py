#!/usr/bin/env python3
"""
Retrain the full DistServe CUDA live profile from real benchmark `.exp` files.

This merged entry point mirrors `retrain_vllm_ascend_live.py`:
1. Fit prefill in the 3-parameter form:
       prefill_ms = A + B * sum_input_tokens + C * sum(input_tokens^2)
2. Fit decode in the 3-parameter form:
       decode_ms = A + B * total_tokens + C * batch_size

Only the final merged JSON profile is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import retrain_distserve_live_decode as decode_fit
import retrain_distserve_live_prefill as prefill_fit


def collect_prefill_samples(results_root: Path, cluster_gap_ms: float) -> tuple[dict, list[str], int]:
    batch_samples_by_model: dict[str, list[prefill_fit.PrefillBatchSample]] = {}
    exp_files = prefill_fit.discover_exp_files(results_root)
    ignored_files: list[str] = []

    for exp_path in exp_files:
        model_key, events, rate = prefill_fit.extract_prefill_events(exp_path)
        if model_key is None:
            ignored_files.append(str(exp_path))
            continue
        batch_samples = prefill_fit.cluster_events_into_batches(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=cluster_gap_ms,
        )
        batch_samples_by_model.setdefault(model_key, []).extend(batch_samples)

    return batch_samples_by_model, ignored_files, len(exp_files)


def collect_decode_samples(results_root: Path, cluster_gap_ms: float) -> tuple[dict, list[str], int]:
    round_samples_by_model: dict[str, list[decode_fit.RoundSample]] = {}
    exp_files = decode_fit.discover_exp_files(results_root)
    ignored_files: list[str] = []

    for exp_path in exp_files:
        model_key, events, rate = decode_fit.extract_decode_events(exp_path)
        if model_key is None:
            ignored_files.append(str(exp_path))
            continue
        round_samples = decode_fit.cluster_events_into_rounds(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=cluster_gap_ms,
        )
        round_samples_by_model.setdefault(model_key, []).extend(round_samples)

    return round_samples_by_model, ignored_files, len(exp_files)


def write_prefill_samples_csv(path: Path, samples_by_model: dict) -> None:
    samples = [sample for model_samples in samples_by_model.values() for sample in model_samples]
    prefill_fit.write_batches_csv(path, samples)


def write_decode_samples_csv(path: Path, samples_by_model: dict) -> None:
    samples = [sample for model_samples in samples_by_model.values() for sample in model_samples]
    decode_fit.write_rounds_csv(path, samples)


def create_tp1_entry(profile: dict, model_key: str, decode_large_small_bs_threshold: int) -> dict:
    model_profile = profile.setdefault(model_key, {})
    tp_profile = model_profile.setdefault("1", {})
    tp_profile.setdefault("decoding_large_small_bs_threshold", decode_large_small_bs_threshold)
    return tp_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the full DistServe CUDA live profile.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result"),
        help="Root directory containing DistServe benchmark `.exp` files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_cuda_data.json"),
        help="Output path for the final merged live profile JSON.",
    )
    parser.add_argument(
        "--cluster-gap-ms",
        type=float,
        default=1.0,
        help="Maximum timestamp gap, in milliseconds, for reconstructing live batches/rounds.",
    )
    parser.add_argument(
        "--decode-large-small-bs-threshold",
        type=int,
        default=95,
        help="Threshold retained for JSON compatibility. Large-batch coefficients fall back to the same fit.",
    )
    parser.add_argument(
        "--prefill-samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for reconstructed prefill samples.",
    )
    parser.add_argument(
        "--decode-samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for reconstructed decode samples.",
    )
    args = parser.parse_args()

    profile: dict[str, dict[str, dict[str, list[float] | int]]] = {}

    prefill_samples_by_model, prefill_ignored_files, prefill_file_count = collect_prefill_samples(
        args.results_root,
        args.cluster_gap_ms,
    )
    print(f"Discovered {prefill_file_count} prefill exp file(s)")
    if prefill_ignored_files:
        print(f"Ignored {len(prefill_ignored_files)} prefill files with unknown model aliases")

    if args.prefill_samples_csv is not None:
        write_prefill_samples_csv(args.prefill_samples_csv, prefill_samples_by_model)

    for model_key, samples in sorted(prefill_samples_by_model.items()):
        coeffs, metrics = prefill_fit.fit_relative_error_quadratic_model(samples)
        tp_profile = create_tp1_entry(
            profile,
            model_key,
            decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
        )
        tp_profile["prefill"] = coeffs

        print(f"\nPrefill model: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    decode_samples_by_model, decode_ignored_files, decode_file_count = collect_decode_samples(
        args.results_root,
        args.cluster_gap_ms,
    )
    print(f"\nDiscovered {decode_file_count} decode exp file(s)")
    if decode_ignored_files:
        print(f"Ignored {len(decode_ignored_files)} decode files with unknown model aliases")

    if args.decode_samples_csv is not None:
        write_decode_samples_csv(args.decode_samples_csv, decode_samples_by_model)

    for model_key, samples in sorted(decode_samples_by_model.items()):
        coeffs, metrics = decode_fit.fit_relative_error_three_param_model(samples)
        tp_profile = create_tp1_entry(
            profile,
            model_key,
            decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
        )
        tp_profile["decoding_smallbs"] = coeffs
        tp_profile["decoding_largebs"] = coeffs
        tp_profile["decoding_large_small_bs_threshold"] = args.decode_large_small_bs_threshold

        print(f"\nDecode model: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote scratch-built live profile to {args.output}")


if __name__ == "__main__":
    main()

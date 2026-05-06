#!/usr/bin/env python3
"""
Filtered variant of `retrain_distserve_cuda_live_5p4d.py`.

This keeps the same 5-parameter prefill fit and 4-parameter decode fit, but
adds Ascend-specific decode sample filtering:

1. Optionally drop the last emitted decode interval.
2. Optionally ignore implausibly small decode intervals.

The original script is left untouched so both variants can be compared side by
side.

python3 /users/rh/DistServe/simdistserve/estimators/fit_params/fit_infer_batch.py \
  --backend vllm_ascend \
  --results-root /users/rh/ascend_data/ascend_vllm_holdout_fit_4ascend \
  --output /users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_tp_2.json \
  --tp-world-size 2

python3 /users/rh/DistServe/simdistserve/estimators/fit_params/fit_infer_batch.py \
  --backend distserve_cuda \
  --results-root /users/rh/DistServe/evaluation/2-benchmark-serving/result/fit \
  --output /users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_5p4d.json


"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import utils as base_fit


def create_tp_entry(
    profile: dict,
    model_key: str,
    tp_world_size: int,
    decode_large_small_bs_threshold: int,
) -> dict:
    model_profile = profile.setdefault(model_key, {})
    tp_profile = model_profile.setdefault(str(tp_world_size), {})
    tp_profile.setdefault("decoding_large_small_bs_threshold", decode_large_small_bs_threshold)
    return tp_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a live profile with 5-param prefill and 4-param decode, "
            "with optional Ascend decode interval filtering."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="distserve_cuda",
        choices=["distserve_cuda", "vllm_ascend"],
        help="Which request-trace backend to fit.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root directory containing benchmark `.exp` files. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the fitted profile JSON. Defaults depend on --backend.",
    )
    parser.add_argument(
        "--cluster-gap-ms",
        type=float,
        default=1.0,
        help="Maximum timestamp gap, in milliseconds, for reconstructing live batches and rounds.",
    )
    parser.add_argument(
        "--decode-large-small-bs-threshold",
        type=int,
        default=95,
        help="Compatibility threshold retained in the output JSON.",
    )
    parser.add_argument(
        "--tp-world-size",
        type=int,
        default=1,
        help="Tensor parallel size key to write in the output profile JSON.",
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
    parser.add_argument(
        "--exclude-last-interval",
        dest="exclude_last_interval",
        action="store_true",
        default=True,
        help="For vllm_ascend decode fitting, drop the last emitted interval from each request.",
    )
    parser.add_argument(
        "--include-last-interval",
        dest="exclude_last_interval",
        action="store_false",
        help="For vllm_ascend decode fitting, keep the last emitted interval.",
    )
    parser.add_argument(
        "--min-interval-ms",
        type=float,
        default=5.0,
        help="For vllm_ascend decode fitting, ignore intervals smaller than this threshold.",
    )
    args = parser.parse_args()

    results_root = args.results_root or base_fit.default_results_root(args.backend)
    output_path = args.output or base_fit.default_output_path(args.backend)
    profile: dict[str, dict[str, dict[str, list[float] | int]]] = {}

    prefill_samples_by_model, prefill_ignored_files, prefill_file_count = base_fit.collect_prefill_samples(
        results_root,
        args.backend,
        args.cluster_gap_ms,
    )
    print(f"Discovered {prefill_file_count} prefill exp file(s)")
    if prefill_ignored_files:
        print(f"Ignored {len(prefill_ignored_files)} prefill files with unknown model aliases")

    if args.prefill_samples_csv is not None:
        prefill_samples = [
            sample
            for model_samples in prefill_samples_by_model.values()
            for sample in model_samples
        ]
        base_fit.write_prefill_samples_csv(args.prefill_samples_csv, prefill_samples)

    for model_key, samples in sorted(prefill_samples_by_model.items()):
        coeffs, metrics = base_fit.fit_prefill_five_param_model(samples)
        tp_profile = create_tp_entry(
            profile,
            model_key,
            args.tp_world_size,
            decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
        )
        tp_profile["prefill"] = coeffs

        print(f"\nPrefill model: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    decode_samples_by_model, decode_ignored_files, decode_file_count = base_fit.collect_decode_samples(
        results_root=results_root,
        backend=args.backend,
        cluster_gap_ms=args.cluster_gap_ms,
        exclude_last_interval=args.exclude_last_interval,
        min_interval_ms=args.min_interval_ms,
    )
    print(f"\nDiscovered {decode_file_count} decode exp file(s)")
    if decode_ignored_files:
        print(f"Ignored {len(decode_ignored_files)} decode files with unknown model aliases")

    if args.decode_samples_csv is not None:
        decode_samples = [
            sample
            for model_samples in decode_samples_by_model.values()
            for sample in model_samples
        ]
        base_fit.write_decode_samples_csv(args.decode_samples_csv, decode_samples)

    for model_key, samples in sorted(decode_samples_by_model.items()):
        coeffs, metrics = base_fit.fit_decode_four_param_model(samples)
        tp_profile = create_tp_entry(
            profile,
            model_key,
            args.tp_world_size,
            decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
        )
        tp_profile["decoding_smallbs"] = coeffs
        tp_profile["decoding_largebs"] = coeffs
        tp_profile["decoding_large_small_bs_threshold"] = args.decode_large_small_bs_threshold

        print(f"\nDecode model: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote filtered live profile to {output_path}")


if __name__ == "__main__":
    main()

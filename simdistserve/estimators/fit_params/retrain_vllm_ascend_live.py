#!/usr/bin/env python3
"""
Retrain the full vLLM Ascend live profile from benchmark JSONs.

This merged entry point runs:
1. Prefill fit on compute-grid or legacy prefill summaries.
2. Decode fit on proxy-grid or legacy decode summaries.

Only the final merged JSON profile is written.

/users/rh/miniconda3/envs/distserve/bin/python \
/users/rh/DistServe/simdistserve/estimators/fit_params/retrain_vllm_ascend_live.py \
  --prefill-results-root /users/rh/ascend_data/ascend_compute_grid \
  --decode-results-root /users/rh/ascend_data/ascend_proxy_grid \
  --output /users/rh/ascend_data/fitted_profiles/fit_params_live_ascend_data.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import retrain_vllm_ascend_decode as decode_fit
import retrain_vllm_ascend_prefill as prefill_fit


def collect_prefill_samples(
    results_root: Path,
    ttft_stat: str,
) -> tuple[str, dict, set[str], int]:
    grid_files = prefill_fit.discover_grid_case_summary_files(results_root)
    legacy_files = [] if grid_files else prefill_fit.discover_legacy_summary_files(results_root)

    if grid_files:
        mode = "grid"
        samples_by_model = {}
        ignored_models: set[str] = set()
        for summary_path in grid_files:
            file_samples_by_model, file_ignored_models = prefill_fit.extract_grid_samples(
                summary_path,
                ttft_stat=ttft_stat,
            )
            ignored_models.update(file_ignored_models)
            for model_key, model_samples in file_samples_by_model.items():
                samples_by_model.setdefault(model_key, []).extend(model_samples)
        return mode, samples_by_model, ignored_models, len(grid_files)

    if legacy_files:
        mode = "legacy"
        samples_by_model = {}
        ignored_models = set()
        for summary_path in legacy_files:
            model_key, model_samples = prefill_fit.extract_legacy_samples(summary_path)
            if model_key is None:
                ignored_models.add(str(summary_path))
                continue
            samples_by_model.setdefault(model_key, []).extend(model_samples)
        return mode, samples_by_model, ignored_models, len(legacy_files)

    raise FileNotFoundError(
        f"No supported prefill benchmark JSONs found under {results_root}. "
        "Expected either legacy `rate_*.json` files or grid `case_summaries.json`."
    )


def collect_decode_samples(
    results_root: Path,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[str, dict, set[str], int]:
    grid_files = decode_fit.discover_grid_request_metrics_files(results_root)
    legacy_files = [] if grid_files else decode_fit.discover_legacy_summary_files(results_root)

    if grid_files:
        mode = "grid"
        samples_by_model = {}
        ignored_models: set[str] = set()
        for summary_path in grid_files:
            file_samples_by_model, file_ignored_models = decode_fit.extract_grid_samples(
                summary_path,
                exclude_last_interval=exclude_last_interval,
                min_interval_ms=min_interval_ms,
            )
            ignored_models.update(file_ignored_models)
            for model_key, model_samples in file_samples_by_model.items():
                samples_by_model.setdefault(model_key, []).extend(model_samples)
        return mode, samples_by_model, ignored_models, len(grid_files)

    if legacy_files:
        mode = "legacy"
        samples_by_model = {}
        ignored_models = set()
        for summary_path in legacy_files:
            model_key, model_samples = decode_fit.extract_legacy_samples(summary_path)
            if model_key is None:
                ignored_models.add(str(summary_path))
                continue
            samples_by_model.setdefault(model_key, []).extend(model_samples)
        return mode, samples_by_model, ignored_models, len(legacy_files)

    raise FileNotFoundError(
        f"No supported decode benchmark JSONs found under {results_root}. "
        "Expected either legacy `rate_*.json` files or grid `request_metrics.json`."
    )


def create_tp1_entry(profile: dict, model_key: str, decode_large_small_bs_threshold: int) -> dict:
    model_profile = profile.setdefault(model_key, {})
    tp_profile = model_profile.setdefault("1", {})
    tp_profile.setdefault("decoding_large_small_bs_threshold", decode_large_small_bs_threshold)
    return tp_profile


def apply_prefill_fit(profile: dict, mode: str, samples_by_model: dict, decode_large_small_bs_threshold: int) -> None:
    for model_key, samples in sorted(samples_by_model.items()):
        if mode == "grid":
            coeffs, metrics = prefill_fit.fit_relative_error_live_batch(samples)
        else:
            coeffs, metrics = prefill_fit.fit_relative_error_quadratic(samples)

        tp_profile = create_tp1_entry(
            profile,
            model_key,
            decode_large_small_bs_threshold=decode_large_small_bs_threshold,
        )
        tp_profile["prefill"] = coeffs
        print(f"\nPrefill model: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")


def apply_decode_fit(
    profile: dict,
    mode: str,
    samples_by_model: dict,
    decode_large_small_bs_threshold: int,
) -> None:
    for model_key, samples in sorted(samples_by_model.items()):
        if mode == "grid":
            coeffs, metrics = decode_fit.fit_grid_interval_model(samples)
        else:
            coeffs, metrics = decode_fit.fit_legacy_interval_model(samples)
            calibrated_coeffs = decode_fit.CALIBRATED_DECODE_COEFFS.get(model_key)
            if calibrated_coeffs is not None:
                coeffs = calibrated_coeffs

        tp_profile = create_tp1_entry(
            profile,
            model_key,
            decode_large_small_bs_threshold=decode_large_small_bs_threshold,
        )
        tp_profile["decoding_smallbs"] = coeffs
        tp_profile["decoding_largebs"] = coeffs
        tp_profile["decoding_large_small_bs_threshold"] = decode_large_small_bs_threshold

        print(f"\nDecode model: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        if mode == "legacy" and model_key in decode_fit.CALIBRATED_DECODE_COEFFS:
            print("  note: applied validated legacy calibration")
        print(f"  fit: {metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the full vLLM Ascend live profile.")
    parser.add_argument(
        "--prefill-results-root",
        type=Path,
        default=Path("/users/rh/ascend_data/ascend_compute_grid"),
        help="Compute-grid directory/file or legacy prefill benchmark root.",
    )
    parser.add_argument(
        "--decode-results-root",
        type=Path,
        default=Path("/users/rh/ascend_data/ascend_proxy_grid"),
        help="Proxy-grid directory/file or legacy decode benchmark root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live.json"),
        help="Output path for the final merged live profile JSON.",
    )
    parser.add_argument(
        "--grid-ttft-stat",
        type=str,
        default="p50_ttft",
        choices=["mean_ttft", "p50_ttft", "p95_ttft"],
        help="Which TTFT statistic to use from grid `case_summaries.json`.",
    )
    parser.add_argument(
        "--decode-large-small-bs-threshold",
        type=int,
        default=95,
        help="Threshold retained for JSON compatibility. Large-batch coefficients fall back to the same fit.",
    )
    parser.add_argument(
        "--exclude-last-interval",
        dest="exclude_last_interval",
        action="store_true",
        default=True,
        help="Drop the final emitted interval from each request when fitting grid decode data.",
    )
    parser.add_argument(
        "--include-last-interval",
        dest="exclude_last_interval",
        action="store_false",
        help="Keep the final emitted interval from each request when fitting grid decode data.",
    )
    parser.add_argument(
        "--min-interval-ms",
        type=float,
        default=5.0,
        help="Ignore grid decode intervals below this threshold.",
    )
    parser.add_argument(
        "--prefill-samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for extracted prefill samples.",
    )
    parser.add_argument(
        "--decode-samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for extracted decode samples.",
    )
    args = parser.parse_args()

    profile: dict[str, dict[str, dict[str, list[float] | int]]] = {}

    prefill_mode, prefill_samples_by_model, prefill_ignored_models, prefill_file_count = collect_prefill_samples(
        args.prefill_results_root,
        args.grid_ttft_stat,
    )
    print(f"Discovered {prefill_file_count} prefill {'grid' if prefill_mode == 'grid' else 'legacy'} file(s)")
    if prefill_ignored_models:
        print(f"Ignored {len(prefill_ignored_models)} unknown prefill model entries")

    if args.prefill_samples_csv is not None:
        prefill_fit.write_samples_csv(
            args.prefill_samples_csv,
            prefill_mode,
            [sample for samples in prefill_samples_by_model.values() for sample in samples],
        )

    apply_prefill_fit(
        profile,
        prefill_mode,
        prefill_samples_by_model,
        decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
    )

    decode_mode, decode_samples_by_model, decode_ignored_models, decode_file_count = collect_decode_samples(
        args.decode_results_root,
        exclude_last_interval=args.exclude_last_interval,
        min_interval_ms=args.min_interval_ms,
    )
    print(f"\nDiscovered {decode_file_count} decode {'grid' if decode_mode == 'grid' else 'legacy'} file(s)")
    if decode_ignored_models:
        print(f"Ignored {len(decode_ignored_models)} unknown decode model entries")

    if args.decode_samples_csv is not None:
        decode_fit.write_samples_csv(
            args.decode_samples_csv,
            decode_mode,
            [sample for samples in decode_samples_by_model.values() for sample in samples],
        )

    apply_decode_fit(
        profile,
        decode_mode,
        decode_samples_by_model,
        decode_large_small_bs_threshold=args.decode_large_small_bs_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote scratch-built live profile to {args.output}")


if __name__ == "__main__":
    main()

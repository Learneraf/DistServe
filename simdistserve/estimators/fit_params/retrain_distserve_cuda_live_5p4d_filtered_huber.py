#!/usr/bin/env python3
"""
Huber-IRLS variant of `retrain_distserve_cuda_live_5p4d_filtered.py`.

This keeps the same:
1. sample reconstruction logic,
2. Ascend-specific decode filtering,
3. output JSON schema,

but replaces the ordinary least squares fit with Huber IRLS.

python3 /users/rh/DistServe/simdistserve/estimators/fit_params/retrain_distserve_cuda_live_5p4d_filtered_huber.py \
  --backend vllm_ascend \
  --results-root /users/rh/ascend_data/ascend_vllm_holdout_fit \
  --output /users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/ablation/fit_params_live_5p4d_huber_no_maxlen.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import retrain_distserve_cuda_live_5p4d as base_fit


def default_output_path(backend: str) -> Path:
    if backend == "distserve_cuda":
        return Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/"
            "distserve-cuda/fit_params_live_5p4d_filtered_huber.json"
        )
    if backend == "vllm_ascend":
        return Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/"
            "vllm-ascend/fit_params_live_5p4d_filtered_huber.json"
        )
    raise ValueError(f"Unsupported backend: {backend}")


def extract_decode_events(
    exp_path: Path,
    backend: str,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[str | None, list[base_fit.TokenEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_key = base_fit.infer_model_key(exp_path)
    rate = base_fit.infer_rate(exp_path)
    events: list[base_fit.TokenEvent] = []

    for request in requests:
        prompt_len = int(request["prompt_len"])
        output_len = int(request["output_len"])
        token_timestamps = request.get("token_timestamps", [])
        if len(token_timestamps) != output_len or output_len <= 1:
            continue

        stop = output_len
        if backend == "vllm_ascend" and exclude_last_interval:
            stop = output_len - 1

        for token_idx in range(1, stop):
            interval_ms = (token_timestamps[token_idx] - token_timestamps[token_idx - 1]) * 1000.0
            if backend == "vllm_ascend" and interval_ms < min_interval_ms:
                continue

            current_context_len = prompt_len + token_idx
            remaining_output_tokens = output_len - token_idx
            events.append(
                base_fit.TokenEvent(
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


def collect_decode_samples(
    results_root: Path,
    backend: str,
    cluster_gap_ms: float,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[dict[str, list[base_fit.RoundSample]], list[str], int]:
    samples_by_model: dict[str, list[base_fit.RoundSample]] = {}
    ignored_files: list[str] = []
    exp_files = base_fit.discover_exp_files(results_root, backend)

    for exp_path in exp_files:
        model_key, events, rate = extract_decode_events(
            exp_path=exp_path,
            backend=backend,
            exclude_last_interval=exclude_last_interval,
            min_interval_ms=min_interval_ms,
        )
        if model_key is None:
            ignored_files.append(str(exp_path))
            continue
        samples = base_fit.cluster_events_into_rounds(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=cluster_gap_ms,
        )
        samples_by_model.setdefault(model_key, []).extend(samples)

    return samples_by_model, ignored_files, len(exp_files)


def _compute_fit_metrics(
    coeffs: np.ndarray,
    raw_rows: list[list[float]],
    durations: list[float],
) -> dict[str, float]:
    predictions = [float(np.dot(coeffs, row)) for row in raw_rows]
    rel_errors = [
        (prediction - duration) / duration
        for prediction, duration in zip(predictions, durations)
    ]
    abs_rel_errors = [abs(error) for error in rel_errors]
    return {
        "count": float(len(durations)),
        "mean_abs_rel_error_pct": 100.0 * statistics.mean(abs_rel_errors),
        "rmse_rel_error_pct": 100.0 * math.sqrt(statistics.mean(error * error for error in rel_errors)),
        "max_abs_rel_error_pct": 100.0 * max(abs_rel_errors),
    }


def _solve_huber_irls(
    design_matrix: np.ndarray,
    rhs: np.ndarray,
    huber_delta: float,
    max_iters: int,
    tol: float,
) -> tuple[np.ndarray, int, float]:
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, rhs, rcond=None)
    eps = 1e-12

    for iteration in range(1, max_iters + 1):
        residuals = design_matrix @ coeffs - rhs
        centered = residuals - np.median(residuals)
        sigma = float(np.median(np.abs(centered)) / 0.6744897501960817)
        sigma = max(sigma, eps)

        scaled = residuals / sigma
        abs_scaled = np.abs(scaled)
        weights = np.ones_like(abs_scaled)
        mask = abs_scaled > huber_delta
        weights[mask] = huber_delta / abs_scaled[mask]
        sqrt_weights = np.sqrt(np.maximum(weights, eps))

        weighted_design = design_matrix * sqrt_weights[:, None]
        weighted_rhs = rhs * sqrt_weights
        new_coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)

        denom = max(np.linalg.norm(coeffs), eps)
        step = float(np.linalg.norm(new_coeffs - coeffs) / denom)
        coeffs = new_coeffs
        if step <= tol:
            return coeffs, iteration, sigma

    residuals = design_matrix @ coeffs - rhs
    centered = residuals - np.median(residuals)
    sigma = float(np.median(np.abs(centered)) / 0.6744897501960817)
    return coeffs, max_iters, max(sigma, eps)


def fit_prefill_five_param_model_huber(
    samples: list[base_fit.PrefillBatchSample],
    huber_delta: float,
    max_iters: int,
    tol: float,
) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No prefill samples to fit.")

    design_matrix = []
    rhs = []
    raw_rows = []
    durations = []

    for sample in samples:
        duration = max(sample.duration_ms, 1e-6)
        row = [
            1.0,
            # float(sample.batch_size),
            float(sample.sum_prompt_len),
            # float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
        ]
        design_matrix.append([value / duration for value in row])
        rhs.append(1.0)
        raw_rows.append(row)
        durations.append(duration)

    coeffs, iterations, sigma = _solve_huber_irls(
        design_matrix=np.array(design_matrix, dtype=float),
        rhs=np.array(rhs, dtype=float),
        huber_delta=huber_delta,
        max_iters=max_iters,
        tol=tol,
    )
    metrics = _compute_fit_metrics(coeffs, raw_rows, durations)
    metrics["irls_iterations"] = float(iterations)
    metrics["irls_residual_scale"] = sigma
    metrics["huber_delta"] = huber_delta
    return coeffs.tolist(), metrics


def fit_decode_four_param_model_huber(
    samples: list[base_fit.RoundSample],
    huber_delta: float,
    max_iters: int,
    tol: float,
) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No decode samples to fit.")

    design_matrix = []
    rhs = []
    raw_rows = []
    durations = []

    for sample in samples:
        duration = max(sample.duration_ms, 1e-6)
        row = [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            # float(sample.max_context_len),
        ]
        design_matrix.append([value / duration for value in row])
        rhs.append(1.0)
        raw_rows.append(row)
        durations.append(duration)

    coeffs, iterations, sigma = _solve_huber_irls(
        design_matrix=np.array(design_matrix, dtype=float),
        rhs=np.array(rhs, dtype=float),
        huber_delta=huber_delta,
        max_iters=max_iters,
        tol=tol,
    )
    metrics = _compute_fit_metrics(coeffs, raw_rows, durations)
    metrics["irls_iterations"] = float(iterations)
    metrics["irls_residual_scale"] = sigma
    metrics["huber_delta"] = huber_delta
    return coeffs.tolist(), metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a live profile with 5-param prefill and 4-param decode, "
            "with Ascend decode filtering and Huber IRLS fitting."
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
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.345,
        help="Huber cutoff on standardized residuals used by IRLS.",
    )
    parser.add_argument(
        "--irls-max-iters",
        type=int,
        default=50,
        help="Maximum number of IRLS iterations.",
    )
    parser.add_argument(
        "--irls-tol",
        type=float,
        default=1e-8,
        help="Relative coefficient-change tolerance for IRLS convergence.",
    )
    args = parser.parse_args()

    results_root = args.results_root or base_fit.default_results_root(args.backend)
    output_path = args.output or default_output_path(args.backend)
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
        coeffs, metrics = fit_prefill_five_param_model_huber(
            samples=samples,
            huber_delta=args.huber_delta,
            max_iters=args.irls_max_iters,
            tol=args.irls_tol,
        )
        tp_profile = base_fit.create_tp1_entry(
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
        coeffs, metrics = fit_decode_four_param_model_huber(
            samples=samples,
            huber_delta=args.huber_delta,
            max_iters=args.irls_max_iters,
            tol=args.irls_tol,
        )
        tp_profile = base_fit.create_tp1_entry(
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote filtered Huber live profile to {output_path}")


if __name__ == "__main__":
    main()

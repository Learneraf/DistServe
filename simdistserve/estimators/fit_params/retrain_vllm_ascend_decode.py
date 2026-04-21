#!/usr/bin/env python3
"""
Retrain vLLM Ascend decode parameters from benchmark JSONs.

Supported input formats:
1. Legacy `rate_*.json` summaries:
       itl_ms = A + C * current_context_len + E * remaining_output_tokens
       stored as [A, 0.0, C, 0.0, E]
2. New grid `request_metrics.json` outputs from `run_grid_benchmark.py`:
       round_ms = A
                + B * total_tokens
                + C * batch_size

   where `total_tokens = sum(current_context_len + 1)` for the active decode round,
   which matches the 3-coefficient runtime path used by `time_estimator.py`.

The grid path drops the last emitted interval by default and filters
pathologically tiny intervals, which are transport flush artifacts rather than
real steady-state decode rounds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_NAME_TO_KEY = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
    "llama1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
    "huggyllama/llama-7b": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "anonymous4chan/llama-2-7b": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "LLM-Research/llama-2-7b": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "LLM-Research/Llama-3.2-1B":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "LLM-Research/Llama-3.2-3B":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "LLM-Research/Meta-Llama-3.1-8B":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
    "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2":
        "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}

FALLBACK_MODEL_KEYS = {
    "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2": (
        "anonymous4chan/llama-2-7b",
        "huggyllama/llama-7b",
    ),
    "anonymous4chan/llama-2-7b": ("huggyllama/llama-7b",),
}

# Retained for legacy rate-summary workflows only.
CALIBRATED_DECODE_COEFFS = {
    "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2":
        [41.0, 0.12, -0.0006, 0.0, 5.1567739390013045e-05],
}


@dataclass
class LegacyDecodeSample:
    model_key: str
    source_file: str
    request_rate: float
    prompt_len: int
    output_len: int
    token_idx: int
    current_context_len: int
    remaining_output_tokens: int
    interval_ms: float


@dataclass
class GridDecodeSample:
    model_key: str
    source_file: str
    batch_size: int
    input_len: int
    output_len: int
    token_idx: int
    sum_context_len: int
    max_context_len: int
    sum_remaining_output_tokens: int
    interval_ms: float


def normalize_model_key(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return MODEL_NAME_TO_KEY.get(model_name)


def discover_legacy_summary_files(results_root: Path) -> list[Path]:
    if results_root.is_file():
        return [results_root] if results_root.name.startswith("rate_") else []
    return sorted(path for path in results_root.rglob("rate_*.json") if path.is_file())


def discover_grid_request_metrics_files(results_root: Path) -> list[Path]:
    if results_root.is_file():
        return [results_root] if results_root.name == "request_metrics.json" else []
    direct = results_root / "request_metrics.json"
    if direct.is_file():
        return [direct]

    matches = sorted(path for path in results_root.rglob("request_metrics.json") if path.is_file())
    if not matches:
        return []

    preferred = [
        path
        for path in matches
        if any(token in str(path).lower() for token in ("proxy", "decode"))
    ]
    if preferred:
        return preferred
    if len(matches) == 1:
        return matches
    raise ValueError(
        f"Found multiple request_metrics.json files under {results_root}. "
        "Pass the proxy-grid directory or file explicitly."
    )


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def extract_legacy_samples(summary_path: Path) -> tuple[str | None, list[LegacyDecodeSample]]:
    payload = load_json(summary_path)
    model_key = normalize_model_key(summary_path.parent.name)
    if model_key is None:
        return None, []

    prompt_lens = payload.get("input_lens", [])
    output_lens = payload.get("output_lens", [])
    itls = payload.get("itls", [])
    request_rate = float(payload.get("request_rate", 0.0))

    if not (len(prompt_lens) == len(output_lens) == len(itls)):
        raise ValueError(
            f"Mismatched sample lengths in {summary_path}: "
            f"{len(prompt_lens)} prompt lengths, {len(output_lens)} output lengths, {len(itls)} itl lists"
        )

    samples: list[LegacyDecodeSample] = []
    for prompt_len, output_len, request_itls in zip(prompt_lens, output_lens, itls):
        prompt_len = int(prompt_len)
        output_len = int(output_len)
        expected_itl_count = max(output_len - 1, 0)
        if len(request_itls) != expected_itl_count:
            continue
        for token_idx, itl_s in enumerate(request_itls, start=1):
            current_context_len = prompt_len + token_idx
            remaining_output_tokens = output_len - token_idx
            samples.append(
                LegacyDecodeSample(
                    model_key=model_key,
                    source_file=str(summary_path),
                    request_rate=request_rate,
                    prompt_len=prompt_len,
                    output_len=output_len,
                    token_idx=token_idx,
                    current_context_len=current_context_len,
                    remaining_output_tokens=remaining_output_tokens,
                    interval_ms=1000.0 * float(itl_s),
                )
            )
    return model_key, samples


def extract_grid_samples(
    summary_path: Path,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[dict[str, list[GridDecodeSample]], list[str]]:
    payload = load_json(summary_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list payload in {summary_path}")

    samples_by_model: dict[str, list[GridDecodeSample]] = {}
    ignored_models: list[str] = []

    for row in payload:
        if row.get("error") is not None:
            continue

        token_timestamps = row.get("token_timestamps", [])
        output_len = int(row.get("generated_tokens", 0))
        if len(token_timestamps) != output_len or output_len <= 1:
            continue

        model_key = normalize_model_key(row.get("model_id")) or normalize_model_key(row.get("model_alias"))
        if model_key is None:
            ignored_models.append(str(row.get("model_id") or row.get("model_alias")))
            continue

        input_len = int(row["input_len_actual"])
        batch_size = int(row["batch_size"])
        stop = output_len - 1 if exclude_last_interval else output_len

        for token_idx in range(1, stop):
            interval_ms = 1000.0 * float(token_timestamps[token_idx] - token_timestamps[token_idx - 1])
            if interval_ms < min_interval_ms:
                continue
            current_context_len = input_len + token_idx
            remaining_output_tokens = output_len - token_idx
            samples_by_model.setdefault(model_key, []).append(
                GridDecodeSample(
                    model_key=model_key,
                    source_file=str(summary_path),
                    batch_size=batch_size,
                    input_len=input_len,
                    output_len=output_len,
                    token_idx=token_idx,
                    sum_context_len=batch_size * current_context_len,
                    max_context_len=current_context_len,
                    sum_remaining_output_tokens=batch_size * remaining_output_tokens,
                    interval_ms=interval_ms,
                )
            )

    return samples_by_model, ignored_models


def fit_legacy_interval_model(
    samples: list[LegacyDecodeSample],
) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No legacy decode samples to fit.")

    design_matrix = []
    durations = []

    for sample in samples:
        row = [
            1.0,
            float(sample.current_context_len),
            float(sample.remaining_output_tokens),
        ]
        design_matrix.append(row)
        durations.append(max(sample.interval_ms, 1e-6))

    coeffs, _, _, _ = np.linalg.lstsq(np.array(design_matrix), np.array(durations), rcond=None)
    predictions = [float(np.dot(coeffs, row)) for row in design_matrix]
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

    a, c, e = coeffs.tolist()
    return [a, 0.0, c, 0.0, e], metrics


def fit_grid_interval_model(
    samples: list[GridDecodeSample],
) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No grid decode samples to fit.")

    design_matrix = []
    rhs = []
    raw_rows = []
    durations = []

    for sample in samples:
        duration = max(sample.interval_ms, 1e-6)
        total_tokens = sample.sum_context_len + sample.batch_size
        row = [
            1.0,
            float(total_tokens),
            float(sample.batch_size),
        ]
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
    with open(path) as f:
        return json.load(f)


def ensure_model_profile(base_profile: dict, model_key: str) -> None:
    if model_key in base_profile:
        return
    fallback_model_keys = FALLBACK_MODEL_KEYS.get(model_key, ())
    if isinstance(fallback_model_keys, str):
        fallback_model_keys = (fallback_model_keys,)
    for fallback_model_key in fallback_model_keys:
        if fallback_model_key in base_profile:
            base_profile[model_key] = copy.deepcopy(base_profile[fallback_model_key])
            return
    if not fallback_model_keys:
        raise KeyError(f"Missing base profile entry for {model_key}")
    raise KeyError(
        f"Missing base profile entry for {model_key}; tried fallbacks {list(fallback_model_keys)}"
    )


def write_samples_csv(path: Path, mode: str, samples: list[LegacyDecodeSample] | list[GridDecodeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if mode == "legacy":
            writer.writerow([
                "model_key",
                "source_file",
                "request_rate",
                "prompt_len",
                "output_len",
                "token_idx",
                "current_context_len",
                "remaining_output_tokens",
                "interval_ms",
            ])
            for sample in samples:
                writer.writerow([
                    sample.model_key,
                    sample.source_file,
                    sample.request_rate,
                    sample.prompt_len,
                    sample.output_len,
                    sample.token_idx,
                    sample.current_context_len,
                    sample.remaining_output_tokens,
                    sample.interval_ms,
                ])
        else:
            writer.writerow([
                "model_key",
                "source_file",
                "batch_size",
                "input_len",
                "output_len",
                "token_idx",
                "sum_context_len",
                "max_context_len",
                "sum_remaining_output_tokens",
                "interval_ms",
            ])
            for sample in samples:
                writer.writerow([
                    sample.model_key,
                    sample.source_file,
                    sample.batch_size,
                    sample.input_len,
                    sample.output_len,
                    sample.token_idx,
                    sample.sum_context_len,
                    sample.max_context_len,
                    sample.sum_remaining_output_tokens,
                    sample.interval_ms,
                ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain vLLM Ascend decode-round runtime profile.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/benchmarks/results/latency/vllm_ascend/raw_data"),
        help="Legacy raw-data root or a grid benchmark proxy directory/file.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_prefill.json"),
        help="Existing Ascend profile JSON used as the merge base.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live.json"),
        help="Output path for the merged live profile JSON.",
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
        "--samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for extracted decode samples.",
    )
    args = parser.parse_args()

    base_profile = load_base_profile(args.base_profile)

    grid_files = discover_grid_request_metrics_files(args.results_root)
    legacy_files = [] if grid_files else discover_legacy_summary_files(args.results_root)

    if grid_files:
        mode = "grid"
        samples_by_model: dict[str, list[GridDecodeSample]] = {}
        ignored_models: set[str] = set()
        for summary_path in grid_files:
            file_samples_by_model, file_ignored_models = extract_grid_samples(
                summary_path,
                exclude_last_interval=args.exclude_last_interval,
                min_interval_ms=args.min_interval_ms,
            )
            ignored_models.update(file_ignored_models)
            for model_key, model_samples in file_samples_by_model.items():
                samples_by_model.setdefault(model_key, []).extend(model_samples)
        print(f"Discovered {len(grid_files)} grid request-metrics file(s)")
    elif legacy_files:
        mode = "legacy"
        samples_by_model = {}
        ignored_models = set()
        for summary_path in legacy_files:
            model_key, model_samples = extract_legacy_samples(summary_path)
            if model_key is None:
                ignored_models.add(str(summary_path))
                continue
            samples_by_model.setdefault(model_key, []).extend(model_samples)
        print(f"Discovered {len(legacy_files)} legacy summary file(s)")
    else:
        raise FileNotFoundError(
            f"No supported benchmark JSONs found under {args.results_root}. "
            "Expected either legacy `rate_*.json` files or grid `request_metrics.json`."
        )

    if ignored_models:
        print(f"Ignored {len(ignored_models)} unknown model entries")

    all_samples = [sample for samples in samples_by_model.values() for sample in samples]
    if args.samples_csv is not None:
        write_samples_csv(args.samples_csv, mode, all_samples)

    for model_key, samples in sorted(samples_by_model.items()):
        ensure_model_profile(base_profile, model_key)
        if "1" not in base_profile[model_key]:
            print(f"Skipping {model_key}: TP=1 missing in base profile")
            continue

        if mode == "grid":
            coeffs, metrics = fit_grid_interval_model(samples)
        else:
            coeffs, metrics = fit_legacy_interval_model(samples)
            calibrated_coeffs = CALIBRATED_DECODE_COEFFS.get(model_key)
            if calibrated_coeffs is not None:
                coeffs = calibrated_coeffs

        tp_profile = base_profile[model_key]["1"]
        tp_profile["decoding_smallbs"] = coeffs
        tp_profile["decoding_largebs"] = coeffs
        tp_profile["decoding_large_small_bs_threshold"] = args.decode_large_small_bs_threshold

        print(f"\nModel: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        if mode == "legacy" and model_key in CALIBRATED_DECODE_COEFFS:
            print("  note: applied validated legacy calibration")
        print(f"  fit: {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(base_profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote merged live profile to {args.output}")


if __name__ == "__main__":
    main()

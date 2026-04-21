#!/usr/bin/env python3
"""
Retrain vLLM Ascend prefill parameters from benchmark JSONs.

Supported input formats:
1. Legacy `rate_*.json` summaries:
       ttft_ms = A + B * prompt_len + C * prompt_len^2
2. New grid `case_summaries.json` outputs from `run_grid_benchmark.py`:
       round_ms = A
                + B * sum_prompt_len
                + C * sum_prompt_len_sq

The script preserves untouched model/TP entries and updates TP=1 for any model
found in the supplied benchmark data.
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


@dataclass
class LegacyPrefillSample:
    model_key: str
    source_file: str
    request_rate: float
    prompt_len: int
    ttft_ms: float


@dataclass
class GridPrefillSample:
    model_key: str
    source_file: str
    batch_size: int
    sum_prompt_len: int
    max_prompt_len: int
    sum_prompt_len_sq: int
    ttft_ms: float
    output_len_target: int


def normalize_model_key(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    return MODEL_NAME_TO_KEY.get(model_name)


def discover_legacy_summary_files(results_root: Path) -> list[Path]:
    if results_root.is_file():
        return [results_root] if results_root.name.startswith("rate_") else []
    return sorted(path for path in results_root.rglob("rate_*.json") if path.is_file())


def discover_grid_case_summary_files(results_root: Path) -> list[Path]:
    if results_root.is_file():
        return [results_root] if results_root.name == "case_summaries.json" else []
    direct = results_root / "case_summaries.json"
    if direct.is_file():
        return [direct]

    matches = sorted(path for path in results_root.rglob("case_summaries.json") if path.is_file())
    if not matches:
        return []

    preferred = [path for path in matches if "compute" in str(path).lower()]
    if preferred:
        return preferred
    if len(matches) == 1:
        return matches
    raise ValueError(
        f"Found multiple case_summaries.json files under {results_root}. "
        "Pass the compute-grid directory or file explicitly."
    )


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def extract_legacy_samples(summary_path: Path) -> tuple[str | None, list[LegacyPrefillSample]]:
    model_key = normalize_model_key(summary_path.parent.name)
    if model_key is None:
        return None, []

    payload = load_json(summary_path)
    prompt_lens = payload.get("input_lens", [])
    ttfts = payload.get("ttfts", [])
    request_rate = float(payload.get("request_rate", 0.0))

    if len(prompt_lens) != len(ttfts):
        raise ValueError(
            f"Mismatched sample lengths in {summary_path}: "
            f"{len(prompt_lens)} prompt lengths vs {len(ttfts)} ttfts"
        )

    samples = [
        LegacyPrefillSample(
            model_key=model_key,
            source_file=str(summary_path),
            request_rate=request_rate,
            prompt_len=int(prompt_len),
            ttft_ms=1000.0 * float(ttft),
        )
        for prompt_len, ttft in zip(prompt_lens, ttfts)
    ]
    return model_key, samples


def extract_grid_samples(
    summary_path: Path,
    ttft_stat: str,
) -> tuple[dict[str, list[GridPrefillSample]], list[str]]:
    payload = load_json(summary_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list payload in {summary_path}")

    samples_by_model: dict[str, list[GridPrefillSample]] = {}
    ignored_models: list[str] = []

    for row in payload:
        if int(row.get("failed_requests", 0)) > 0:
            continue

        ttft_value = row.get(ttft_stat)
        if ttft_value is None:
            continue

        model_key = normalize_model_key(row.get("model_id")) or normalize_model_key(row.get("model_alias"))
        if model_key is None:
            ignored_models.append(str(row.get("model_id") or row.get("model_alias")))
            continue

        batch_size = int(row["batch_size"])
        prompt_len = int(row["input_len_actual"])
        samples_by_model.setdefault(model_key, []).append(
            GridPrefillSample(
                model_key=model_key,
                source_file=str(summary_path),
                batch_size=batch_size,
                sum_prompt_len=batch_size * prompt_len,
                max_prompt_len=prompt_len,
                sum_prompt_len_sq=batch_size * prompt_len * prompt_len,
                ttft_ms=1000.0 * float(ttft_value),
                output_len_target=int(row["output_len_target"]),
            )
        )

    return samples_by_model, ignored_models


def fit_relative_error_quadratic(
    samples: list[LegacyPrefillSample],
) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No legacy prefill samples to fit.")

    design_matrix = []
    rhs = []
    raw_rows = []
    durations = []

    for sample in samples:
        duration = max(sample.ttft_ms, 1e-6)
        row = [
            1.0,
            float(sample.prompt_len),
            float(sample.prompt_len * sample.prompt_len),
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


def fit_relative_error_live_batch(
    samples: list[GridPrefillSample],
) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No grid prefill samples to fit.")

    design_matrix = []
    rhs = []
    raw_rows = []
    durations = []

    for sample in samples:
        duration = max(sample.ttft_ms, 1e-6)
        row = [
            1.0,
            float(sample.sum_prompt_len),
            float(sample.sum_prompt_len_sq),
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


def write_samples_csv(path: Path, mode: str, samples: list[LegacyPrefillSample] | list[GridPrefillSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if mode == "legacy":
            writer.writerow(["model_key", "source_file", "request_rate", "prompt_len", "ttft_ms"])
            for sample in samples:
                writer.writerow([
                    sample.model_key,
                    sample.source_file,
                    sample.request_rate,
                    sample.prompt_len,
                    sample.ttft_ms,
                ])
        else:
            writer.writerow([
                "model_key",
                "source_file",
                "batch_size",
                "sum_prompt_len",
                "max_prompt_len",
                "sum_prompt_len_sq",
                "ttft_ms",
                "output_len_target",
            ])
            for sample in samples:
                writer.writerow([
                    sample.model_key,
                    sample.source_file,
                    sample.batch_size,
                    sample.sum_prompt_len,
                    sample.max_prompt_len,
                    sample.sum_prompt_len_sq,
                    sample.ttft_ms,
                    sample.output_len_target,
                ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain vLLM Ascend prefill runtime profile.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/benchmarks/results/latency/vllm_ascend/raw_data"),
        help="Legacy raw-data root or a grid benchmark compute directory/file.",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_all.json"),
        help="Existing Ascend profile JSON used as the merge base.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_prefill.json"),
        help="Output path for the merged prefill profile JSON.",
    )
    parser.add_argument(
        "--grid-ttft-stat",
        type=str,
        default="p50_ttft",
        choices=["mean_ttft", "p50_ttft", "p95_ttft"],
        help="Which TTFT statistic to use from grid `case_summaries.json`.",
    )
    parser.add_argument(
        "--samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for extracted prefill samples.",
    )
    args = parser.parse_args()

    base_profile = load_base_profile(args.base_profile)

    grid_files = discover_grid_case_summary_files(args.results_root)
    legacy_files = [] if grid_files else discover_legacy_summary_files(args.results_root)

    if grid_files:
        mode = "grid"
        samples_by_model: dict[str, list[GridPrefillSample]] = {}
        ignored_models: set[str] = set()
        for summary_path in grid_files:
            file_samples_by_model, file_ignored_models = extract_grid_samples(
                summary_path,
                ttft_stat=args.grid_ttft_stat,
            )
            ignored_models.update(file_ignored_models)
            for model_key, model_samples in file_samples_by_model.items():
                samples_by_model.setdefault(model_key, []).extend(model_samples)
        print(f"Discovered {len(grid_files)} grid case summary file(s)")
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
            "Expected either legacy `rate_*.json` files or grid `case_summaries.json`."
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
            coeffs, metrics = fit_relative_error_live_batch(samples)
        else:
            coeffs, metrics = fit_relative_error_quadratic(samples)

        base_profile[model_key]["1"]["prefill"] = coeffs
        print(f"\nModel: {model_key}")
        print(f"  samples: {len(samples)}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(base_profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote merged prefill profile to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Retrain vLLM Ascend first-token runtime parameters from benchmark summary JSONs.

The Ascend disaggregated benchmark exposes per-request TTFT, which already
captures the path to the first token in the proxy-based deployment. We fit:

    ttft_ms = A + B * prompt_len + C * prompt_len^2

and write the coefficients into the backend-specific profile JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
class PrefillSample:
    model_key: str
    source_file: str
    request_rate: float
    prompt_len: int
    ttft_ms: float


def discover_summary_files(results_root: Path) -> list[Path]:
    return sorted(path for path in results_root.rglob("rate_*.json") if path.is_file())


def infer_model_key(summary_path: Path) -> str | None:
    return MODEL_ALIAS_TO_KEY.get(summary_path.parent.name)


def load_summary(summary_path: Path) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def extract_samples(summary_path: Path) -> tuple[str | None, list[PrefillSample]]:
    model_key = infer_model_key(summary_path)
    if model_key is None:
        return None, []

    payload = load_summary(summary_path)
    prompt_lens = payload.get("input_lens", [])
    ttfts = payload.get("ttfts", [])
    request_rate = float(payload.get("request_rate", 0.0))

    if len(prompt_lens) != len(ttfts):
        raise ValueError(
            f"Mismatched sample lengths in {summary_path}: "
            f"{len(prompt_lens)} prompt lengths vs {len(ttfts)} ttfts"
        )

    samples = [
        PrefillSample(
            model_key=model_key,
            source_file=str(summary_path),
            request_rate=request_rate,
            prompt_len=int(prompt_len),
            ttft_ms=1000.0 * float(ttft),
        )
        for prompt_len, ttft in zip(prompt_lens, ttfts)
    ]
    return model_key, samples


def fit_relative_error_quadratic(samples: list[PrefillSample]) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No prefill samples to fit.")

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


def load_base_profile(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_samples_csv(path: Path, samples: list[PrefillSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_key", "source_file", "request_rate", "prompt_len", "ttft_ms"])
        for sample in samples:
            writer.writerow([
                sample.model_key,
                sample.source_file,
                sample.request_rate,
                sample.prompt_len,
                sample.ttft_ms,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain vLLM Ascend first-token runtime profile.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/benchmarks/results/latency/vllm_ascend/raw_data"),
        help="Root directory containing vLLM Ascend benchmark summary JSONs.",
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
        "--samples-csv",
        type=Path,
        default=None,
        help="Optional CSV output for extracted prefill samples.",
    )
    args = parser.parse_args()

    base_profile = load_base_profile(args.base_profile)
    samples_by_model: dict[str, list[PrefillSample]] = {}

    summary_files = discover_summary_files(args.results_root)
    if not summary_files:
        raise FileNotFoundError(f"No summary JSON files found under {args.results_root}")

    ignored_files = []
    for summary_path in summary_files:
        model_key, samples = extract_samples(summary_path)
        if model_key is None:
            ignored_files.append(str(summary_path))
            continue
        samples_by_model.setdefault(model_key, []).extend(samples)

    all_samples = [sample for samples in samples_by_model.values() for sample in samples]
    if args.samples_csv is not None:
        write_samples_csv(args.samples_csv, all_samples)

    print(f"Discovered {len(summary_files)} benchmark summary files")
    if ignored_files:
        print(f"Ignored {len(ignored_files)} files with unknown model aliases")

    for model_key, samples in sorted(samples_by_model.items()):
        if model_key not in base_profile or "1" not in base_profile[model_key]:
            print(f"Skipping {model_key}: missing base profile entry")
            continue
        coeffs, metrics = fit_relative_error_quadratic(samples)
        base_profile[model_key]["1"]["prefill"] = coeffs
        print(f"\nModel: {model_key}")
        print(f"  coeffs: {coeffs}")
        print(f"  fit: {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(base_profile, f, indent=4)
        f.write("\n")

    print(f"\nWrote merged prefill profile to {args.output}")


if __name__ == "__main__":
    main()

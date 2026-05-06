#!/usr/bin/env python3
"""Evaluate fit_infer_batch.py style profiles on reconstructed live batches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import utils as base_fit


def summarize_errors(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "mean_abs_rel_error_pct": math.nan,
            "rmse_rel_error_pct": math.nan,
            "median_abs_rel_error_pct": math.nan,
            "p90_abs_rel_error_pct": math.nan,
            "p95_abs_rel_error_pct": math.nan,
            "max_abs_rel_error_pct": math.nan,
            "mean_signed_rel_error_pct": math.nan,
        }
    rel = np.array([row["rel_error"] for row in rows], dtype=float)
    abs_rel = np.abs(rel)
    return {
        "count": int(len(rows)),
        "mean_abs_rel_error_pct": float(100.0 * np.mean(abs_rel)),
        "rmse_rel_error_pct": float(100.0 * math.sqrt(np.mean(rel * rel))),
        "median_abs_rel_error_pct": float(100.0 * np.percentile(abs_rel, 50)),
        "p90_abs_rel_error_pct": float(100.0 * np.percentile(abs_rel, 90)),
        "p95_abs_rel_error_pct": float(100.0 * np.percentile(abs_rel, 95)),
        "max_abs_rel_error_pct": float(100.0 * np.max(abs_rel)),
        "mean_signed_rel_error_pct": float(100.0 * np.mean(rel)),
    }


def model_name_from_key(model_key: str) -> str:
    for alias, key in base_fit.MODEL_ALIAS_TO_KEY.items():
        if key == model_key and alias.startswith("llama_"):
            return alias
    return model_key


def predict_prefill(sample: base_fit.PrefillBatchSample, coeffs: list[float]) -> float:
    row = np.array(
        [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        dtype=float,
    )
    return float(np.dot(np.array(coeffs, dtype=float), row))


def predict_decode(sample: base_fit.RoundSample, coeffs: list[float]) -> float:
    row = np.array(
        [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.max_context_len),
        ],
        dtype=float,
    )
    return float(np.dot(np.array(coeffs, dtype=float), row))


def evaluate_prefill(samples_by_model: dict[str, list[base_fit.PrefillBatchSample]], profile: dict) -> tuple[dict, list[dict]]:
    summary = {}
    rows = []
    for model_key, samples in sorted(samples_by_model.items()):
        tp_profile = profile.get(model_key, {}).get("1")
        if not tp_profile or "prefill" not in tp_profile:
            continue
        coeffs = tp_profile["prefill"]
        model_rows = []
        for sample in samples:
            actual = float(sample.duration_ms)
            pred = predict_prefill(sample, coeffs)
            row = {
                "phase": "prefill",
                "model": model_name_from_key(model_key),
                "source_file": sample.source_file,
                "rate": sample.rate,
                "batch_size": sample.batch_size,
                "actual_ms": actual,
                "predicted_ms": pred,
                "rel_error": (pred - actual) / max(actual, 1e-6),
            }
            rows.append(row)
            model_rows.append(row)
        summary[model_name_from_key(model_key)] = summarize_errors(model_rows)
    summary["overall"] = summarize_errors(rows)
    return summary, rows


def evaluate_decode(samples_by_model: dict[str, list[base_fit.RoundSample]], profile: dict) -> tuple[dict, list[dict]]:
    summary = {}
    rows = []
    for model_key, samples in sorted(samples_by_model.items()):
        tp_profile = profile.get(model_key, {}).get("1")
        if not tp_profile or "decoding_smallbs" not in tp_profile:
            continue
        coeffs = tp_profile["decoding_smallbs"]
        model_rows = []
        for sample in samples:
            actual = float(sample.duration_ms)
            pred = predict_decode(sample, coeffs)
            row = {
                "phase": "decode",
                "model": model_name_from_key(model_key),
                "source_file": sample.source_file,
                "rate": sample.rate,
                "batch_size": sample.batch_size,
                "actual_ms": actual,
                "predicted_ms": pred,
                "rel_error": (pred - actual) / max(actual, 1e-6),
            }
            rows.append(row)
            model_rows.append(row)
        summary[model_name_from_key(model_key)] = summarize_errors(model_rows)
    summary["overall"] = summarize_errors(rows)
    return summary, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "model",
        "source_file",
        "rate",
        "batch_size",
        "actual_ms",
        "predicted_ms",
        "rel_error_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "phase": row["phase"],
                    "model": row["model"],
                    "source_file": row["source_file"],
                    "rate": f"{row['rate']:.6g}",
                    "batch_size": row["batch_size"],
                    "actual_ms": f"{row['actual_ms']:.6f}",
                    "predicted_ms": f"{row['predicted_ms']:.6f}",
                    "rel_error_pct": f"{100.0 * row['rel_error']:.6f}",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["distserve_cuda", "vllm_ascend"], default="vllm_ascend")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-samples", type=Path, default=None)
    parser.add_argument("--cluster-gap-ms", type=float, default=1.0)
    parser.add_argument("--exclude-last-interval", dest="exclude_last_interval", action="store_true", default=True)
    parser.add_argument("--include-last-interval", dest="exclude_last_interval", action="store_false")
    parser.add_argument("--min-interval-ms", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.profile.open(encoding="utf-8") as f:
        profile = json.load(f)

    prefill_samples_by_model, prefill_ignored, prefill_file_count = base_fit.collect_prefill_samples(
        args.results_root,
        args.backend,
        args.cluster_gap_ms,
    )
    decode_samples_by_model, decode_ignored, decode_file_count = base_fit.collect_decode_samples(
        results_root=args.results_root,
        backend=args.backend,
        cluster_gap_ms=args.cluster_gap_ms,
        exclude_last_interval=args.exclude_last_interval,
        min_interval_ms=args.min_interval_ms,
    )

    prefill_summary, prefill_rows = evaluate_prefill(prefill_samples_by_model, profile)
    decode_summary, decode_rows = evaluate_decode(decode_samples_by_model, profile)
    summary = {
        "results_root": str(args.results_root),
        "profile": str(args.profile),
        "prefill_file_count": prefill_file_count,
        "decode_file_count": decode_file_count,
        "prefill_ignored_files": prefill_ignored,
        "decode_ignored_files": decode_ignored,
        "prefill": prefill_summary,
        "decode": decode_summary,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_samples is not None:
        write_rows(args.output_samples, prefill_rows + decode_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

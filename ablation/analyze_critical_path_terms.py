#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from ablate_live_length_terms import (
    OUTPUT_ROOT,
    build_decode_samples,
    build_prefill_samples,
    discover_exp_files,
    fit_relative_error_model,
    metric_dict,
)


CRITICAL_OUTPUT_ROOT = OUTPUT_ROOT / "critical_path_analysis"


def prefill_variant_rows(sample) -> dict[str, list[float]]:
    return {
        "pressure_only": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "pressure_plus_max": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "pressure_plus_mid": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.mid_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "pressure_plus_min": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.min_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
    }


def decode_variant_rows(sample) -> dict[str, list[float]]:
    return {
        "pressure_only": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
        ],
        "critical_only": [
            1.0,
            float(sample.batch_size),
            float(sample.max_context_len),
        ],
        "pressure_plus_max": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.max_context_len),
        ],
        "pressure_plus_mid": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.mid_context_len),
        ],
        "pressure_plus_min": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.min_context_len),
        ],
    }


def grouped_cv_predictions(samples: list, variant: str, feature_fn) -> tuple[list[float], list[float]]:
    by_file = defaultdict(list)
    for sample in samples:
        by_file[sample.source_file].append(sample)

    predictions: list[float] = []
    durations: list[float] = []

    for held_out_file, held_out_samples in by_file.items():
        train_samples = [
            sample
            for source_file, file_samples in by_file.items()
            if source_file != held_out_file
            for sample in file_samples
        ]
        if not train_samples:
            continue
        train_rows = [feature_fn(sample)[variant] for sample in train_samples]
        if len(train_rows) < len(train_rows[0]):
            continue
        coeffs = fit_relative_error_model(train_rows, [sample.duration_ms for sample in train_samples])
        for sample in held_out_samples:
            predictions.append(float(np.dot(coeffs, feature_fn(sample)[variant])))
            durations.append(sample.duration_ms)
    return predictions, durations


def evaluate_subsets(samples_by_model: dict[str, list], subsets: dict[str, callable], variants: list[str], feature_fn):
    results = {}
    for subset_name, subset_fn in subsets.items():
        subset_results = {
            "counts": {},
            "variants": {},
        }
        for variant in variants:
            per_model = {}
            overall_predictions: list[float] = []
            overall_durations: list[float] = []
            for model_name, model_samples in sorted(samples_by_model.items()):
                selected = subset_fn(model_samples)
                subset_results["counts"][model_name] = len(selected)
                if not selected:
                    continue
                predictions, durations = grouped_cv_predictions(selected, variant, feature_fn)
                metrics = metric_dict(predictions, durations)
                per_model[model_name] = metrics
                overall_predictions.extend(predictions)
                overall_durations.extend(durations)
            subset_results["variants"][variant] = {
                "overall_grouped_cv_metrics": metric_dict(overall_predictions, overall_durations),
                "per_model": per_model,
            }
        results[subset_name] = subset_results
    return results


def build_prefill_subsets(samples_by_model: dict[str, list]) -> dict[str, callable]:
    positive_spreads = [
        sample.max_prompt_len - sample.min_prompt_len
        for samples in samples_by_model.values()
        for sample in samples
        if sample.batch_size >= 2 and sample.max_prompt_len > sample.min_prompt_len
    ]
    high_spread_threshold = statistics.median(positive_spreads) if positive_spreads else 0
    return {
        "all": lambda samples: list(samples),
        "multi_request": lambda samples: [sample for sample in samples if sample.batch_size >= 2],
        "heterogeneous_multi": lambda samples: [
            sample for sample in samples
            if sample.batch_size >= 2 and sample.max_prompt_len > sample.min_prompt_len
        ],
        "high_spread_multi": lambda samples: [
            sample for sample in samples
            if sample.batch_size >= 2 and (sample.max_prompt_len - sample.min_prompt_len) >= high_spread_threshold and sample.max_prompt_len > sample.min_prompt_len
        ],
    }


def build_decode_subsets(samples_by_model: dict[str, list]) -> dict[str, callable]:
    positive_spreads = [
        sample.max_context_len - sample.min_context_len
        for samples in samples_by_model.values()
        for sample in samples
        if sample.batch_size >= 2 and sample.max_context_len > sample.min_context_len
    ]
    high_spread_threshold = statistics.median(positive_spreads) if positive_spreads else 0
    return {
        "all_small": lambda samples: [sample for sample in samples if sample.split == "small"],
        "multi_request_small": lambda samples: [
            sample for sample in samples
            if sample.split == "small" and sample.batch_size >= 2
        ],
        "heterogeneous_small": lambda samples: [
            sample for sample in samples
            if sample.split == "small" and sample.batch_size >= 2 and sample.max_context_len > sample.min_context_len
        ],
        "high_spread_small": lambda samples: [
            sample for sample in samples
            if sample.split == "small"
            and sample.batch_size >= 2
            and (sample.max_context_len - sample.min_context_len) >= high_spread_threshold
            and sample.max_context_len > sample.min_context_len
        ],
    }


def stage_table(stage_name: str, results: dict, variants: list[str]) -> str:
    lines = [f"## {stage_name}", ""]
    for subset_name, subset_result in results.items():
        total_count = sum(subset_result["counts"].values())
        lines.append(f"### {subset_name}")
        lines.append("")
        lines.append(f"- sample count: {total_count}")
        lines.append("")
        lines.append("| Variant | CV mean abs % | CV RMSE % | CV max abs % |")
        lines.append("| --- | ---: | ---: | ---: |")
        for variant in variants:
            metrics = subset_result["variants"][variant]["overall_grouped_cv_metrics"]
            lines.append(
                f"| {variant} | {metrics['mean_abs_rel_error_pct']:.2f} | "
                f"{metrics['rmse_rel_error_pct']:.2f} | {metrics['max_abs_rel_error_pct']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    CRITICAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exp_files = discover_exp_files(Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result"))

    prefill_samples_by_model = defaultdict(list)
    decode_samples_by_model = defaultdict(list)
    for exp_file in exp_files:
        for sample in build_prefill_samples(exp_file):
            prefill_samples_by_model[sample.model_name].append(sample)
        for sample in build_decode_samples(exp_file):
            decode_samples_by_model[sample.model_name].append(sample)

    prefill_variants = ["pressure_only", "pressure_plus_max", "pressure_plus_mid", "pressure_plus_min"]
    decode_variants = ["pressure_only", "critical_only", "pressure_plus_max", "pressure_plus_mid", "pressure_plus_min"]

    prefill_results = evaluate_subsets(
        prefill_samples_by_model,
        build_prefill_subsets(prefill_samples_by_model),
        prefill_variants,
        prefill_variant_rows,
    )
    decode_results = evaluate_subsets(
        decode_samples_by_model,
        build_decode_subsets(decode_samples_by_model),
        decode_variants,
        decode_variant_rows,
    )

    summary = {
        "prefill": prefill_results,
        "decode": decode_results,
    }
    with open(CRITICAL_OUTPUT_ROOT / "critical_path_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    report = "\n".join(
        [
            "# Critical Path Term Analysis",
            "",
            stage_table("Prefill", prefill_results, prefill_variants),
            stage_table("Decode", decode_results, decode_variants),
        ]
    )
    with open(CRITICAL_OUTPUT_ROOT / "critical_path_report.md", "w") as f:
        f.write(report)
        f.write("\n")

    print(f"Wrote {CRITICAL_OUTPUT_ROOT / 'critical_path_summary.json'}")
    print(f"Wrote {CRITICAL_OUTPUT_ROOT / 'critical_path_report.md'}")


if __name__ == "__main__":
    main()

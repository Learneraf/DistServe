#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESULTS_ROOT = Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result")
OUTPUT_ROOT = Path("/users/rh/tmp/distserve_ablation/results")
DECODE_THRESHOLD = 95

MODEL_ALIAS_TO_NAME = {
    "llama_1B": "llama_1B",
    "llama_3B": "llama_3B",
    "llama_7B": "llama_7B",
    "llama_8B": "llama_8B",
}


@dataclass
class PrefillEvent:
    timestamp: float
    end_timestamp: float
    prompt_len: int


@dataclass
class PrefillBatchSample:
    model_name: str
    source_file: str
    rate: float
    batch_size: int
    sum_prompt_len: int
    max_prompt_len: int
    min_prompt_len: int
    mid_prompt_len: float
    sum_prompt_len_sq: int
    duration_ms: float


@dataclass
class TokenEvent:
    timestamp: float
    interval_ms: float
    current_context_len: int


@dataclass
class DecodeRoundSample:
    model_name: str
    source_file: str
    rate: float
    batch_size: int
    sum_context_len: int
    max_context_len: int
    min_context_len: int
    mid_context_len: float
    duration_ms: float
    split: str


def infer_model_name(exp_path: Path) -> str | None:
    return MODEL_ALIAS_TO_NAME.get(exp_path.parent.name)


def infer_rate(exp_path: Path) -> float:
    match = re.search(r"distserve-\d+-([\d.]+)\.exp$", exp_path.name)
    if not match:
        raise ValueError(f"Cannot infer rate from {exp_path}")
    return float(match.group(1))


def discover_exp_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.exp") if path.parent.name in MODEL_ALIAS_TO_NAME)


def extract_prefill_events(exp_path: Path) -> tuple[str | None, list[PrefillEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_name = infer_model_name(exp_path)
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
    return model_name, events, rate


def cluster_prefill_events(events: list[PrefillEvent], cluster_gap_ms: float) -> list[list[PrefillEvent]]:
    if not events:
        return []
    cluster_gap_s = cluster_gap_ms / 1000.0
    events = sorted(events, key=lambda event: event.timestamp)
    clusters: list[list[PrefillEvent]] = [[events[0]]]
    for event in events[1:]:
        if (event.timestamp - clusters[-1][-1].timestamp) <= cluster_gap_s:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def build_prefill_samples(exp_path: Path, cluster_gap_ms: float = 1.0) -> list[PrefillBatchSample]:
    model_name, events, rate = extract_prefill_events(exp_path)
    if model_name is None:
        return []
    samples: list[PrefillBatchSample] = []
    for cluster in cluster_prefill_events(events, cluster_gap_ms):
        prompt_lens = sorted(event.prompt_len for event in cluster)
        samples.append(
            PrefillBatchSample(
                model_name=model_name,
                source_file=str(exp_path),
                rate=rate,
                batch_size=len(cluster),
                sum_prompt_len=sum(prompt_lens),
                max_prompt_len=max(prompt_lens),
                min_prompt_len=min(prompt_lens),
                mid_prompt_len=float(statistics.median(prompt_lens)),
                sum_prompt_len_sq=sum(prompt_len * prompt_len for prompt_len in prompt_lens),
                duration_ms=1000.0 * statistics.median(event.end_timestamp - event.timestamp for event in cluster),
            )
        )
    return samples


def extract_decode_events(exp_path: Path) -> tuple[str | None, list[TokenEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_name = infer_model_name(exp_path)
    rate = infer_rate(exp_path)
    events: list[TokenEvent] = []
    for request in requests:
        prompt_len = int(request["prompt_len"])
        output_len = int(request["output_len"])
        token_timestamps = request.get("token_timestamps", [])
        if len(token_timestamps) != output_len or output_len <= 1:
            continue
        for token_idx in range(1, output_len):
            events.append(
                TokenEvent(
                    timestamp=float(token_timestamps[token_idx]),
                    interval_ms=(token_timestamps[token_idx] - token_timestamps[token_idx - 1]) * 1000.0,
                    current_context_len=prompt_len + token_idx,
                )
            )
    return model_name, events, rate


def cluster_decode_events(events: list[TokenEvent], cluster_gap_ms: float) -> list[list[TokenEvent]]:
    if not events:
        return []
    cluster_gap_s = cluster_gap_ms / 1000.0
    events = sorted(events, key=lambda event: event.timestamp)
    clusters: list[list[TokenEvent]] = [[events[0]]]
    for event in events[1:]:
        if (event.timestamp - clusters[-1][-1].timestamp) <= cluster_gap_s:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def build_decode_samples(exp_path: Path, cluster_gap_ms: float = 1.0) -> list[DecodeRoundSample]:
    model_name, events, rate = extract_decode_events(exp_path)
    if model_name is None:
        return []
    samples: list[DecodeRoundSample] = []
    for cluster in cluster_decode_events(events, cluster_gap_ms):
        context_lens = sorted(event.current_context_len for event in cluster)
        batch_size = len(cluster)
        samples.append(
            DecodeRoundSample(
                model_name=model_name,
                source_file=str(exp_path),
                rate=rate,
                batch_size=batch_size,
                sum_context_len=sum(context_lens),
                max_context_len=max(context_lens),
                min_context_len=min(context_lens),
                mid_context_len=float(statistics.median(context_lens)),
                duration_ms=float(statistics.median(event.interval_ms for event in cluster)),
                split="small" if batch_size < DECODE_THRESHOLD else "large",
            )
        )
    return samples


def prefill_feature_map(sample: PrefillBatchSample) -> dict[str, list[float]]:
    return {
        "baseline": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "no_bs": [
            1.0,
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "min_len": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.min_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "mid_len": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.mid_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
        "no_max_len": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.sum_prompt_len_sq),
        ],
    }


def decode_feature_map(sample: DecodeRoundSample) -> dict[str, list[float]]:
    return {
        "baseline": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.max_context_len),
        ],
        "min_len": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.min_context_len),
        ],
        "mid_len": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
            float(sample.mid_context_len),
        ],
        "no_max_len": [
            1.0,
            float(sample.batch_size),
            float(sample.sum_context_len),
        ],
    }


def fit_relative_error_model(rows: list[list[float]], durations: list[float]) -> np.ndarray:
    design_matrix = np.array(
        [[value / max(duration, 1e-6) for value in row] for row, duration in zip(rows, durations)],
        dtype=float,
    )
    rhs = np.ones(len(rows), dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(design_matrix, rhs, rcond=None)
    return coeffs


def metric_dict(predictions: list[float], durations: list[float]) -> dict[str, float]:
    rel_errors = [
        (prediction - duration) / max(duration, 1e-6)
        for prediction, duration in zip(predictions, durations)
    ]
    abs_rel_errors = [abs(value) for value in rel_errors]
    if not rel_errors:
        return {
            "count": 0.0,
            "mean_abs_rel_error_pct": 0.0,
            "rmse_rel_error_pct": 0.0,
            "max_abs_rel_error_pct": 0.0,
        }
    return {
        "count": float(len(rel_errors)),
        "mean_abs_rel_error_pct": 100.0 * statistics.mean(abs_rel_errors),
        "rmse_rel_error_pct": 100.0 * math.sqrt(statistics.mean(value * value for value in rel_errors)),
        "max_abs_rel_error_pct": 100.0 * max(abs_rel_errors),
    }


def evaluate_variant(samples: list, variant: str, feature_getter) -> dict[str, object]:
    grouped: dict[str, list] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_file].append(sample)

    cv_predictions: list[float] = []
    cv_durations: list[float] = []
    folds_used = 0

    for held_out_file, held_out_samples in grouped.items():
        train_samples = [
            sample
            for source_file, file_samples in grouped.items()
            if source_file != held_out_file
            for sample in file_samples
        ]
        if not train_samples:
            continue
        train_rows = [feature_getter(sample)[variant] for sample in train_samples]
        if len(train_rows) < len(train_rows[0]):
            continue
        train_durations = [sample.duration_ms for sample in train_samples]
        coeffs = fit_relative_error_model(train_rows, train_durations)
        test_rows = [feature_getter(sample)[variant] for sample in held_out_samples]
        test_durations = [sample.duration_ms for sample in held_out_samples]
        cv_predictions.extend(float(np.dot(coeffs, row)) for row in test_rows)
        cv_durations.extend(test_durations)
        folds_used += 1

    all_rows = [feature_getter(sample)[variant] for sample in samples]
    all_durations = [sample.duration_ms for sample in samples]
    fit_all_coeffs = fit_relative_error_model(all_rows, all_durations).tolist()
    fit_all_predictions = [float(np.dot(fit_all_coeffs, row)) for row in all_rows]

    return {
        "fit_all_coeffs": fit_all_coeffs,
        "fit_all_metrics": metric_dict(fit_all_predictions, all_durations),
        "grouped_cv_metrics": metric_dict(cv_predictions, cv_durations),
        "folds_used": folds_used,
    }


def evaluate_stage_by_model(samples_by_model: dict[str, list], variants: list[str], feature_getter, split_key: str | None = None) -> dict:
    results = {}
    for variant in variants:
        variant_result = {"per_model": {}, "overall_grouped_cv_metrics": {}, "overall_fit_all_metrics": {}}
        overall_cv_predictions: list[float] = []
        overall_cv_durations: list[float] = []
        overall_fit_predictions: list[float] = []
        overall_fit_durations: list[float] = []

        for model_name, model_samples in sorted(samples_by_model.items()):
            if split_key is not None:
                model_samples = [sample for sample in model_samples if sample.split == split_key]
            if not model_samples:
                continue
            model_result = evaluate_variant(model_samples, variant, feature_getter)
            variant_result["per_model"][model_name] = model_result

            grouped = defaultdict(list)
            for sample in model_samples:
                grouped[sample.source_file].append(sample)
            for held_out_file, held_out_samples in grouped.items():
                train_samples = [
                    sample
                    for source_file, file_samples in grouped.items()
                    if source_file != held_out_file
                    for sample in file_samples
                ]
                if not train_samples:
                    continue
                train_rows = [feature_getter(sample)[variant] for sample in train_samples]
                if len(train_rows) < len(train_rows[0]):
                    continue
                coeffs = fit_relative_error_model(train_rows, [sample.duration_ms for sample in train_samples])
                for sample in held_out_samples:
                    overall_cv_predictions.append(float(np.dot(coeffs, feature_getter(sample)[variant])))
                    overall_cv_durations.append(sample.duration_ms)

            coeffs = np.array(model_result["fit_all_coeffs"], dtype=float)
            for sample in model_samples:
                overall_fit_predictions.append(float(np.dot(coeffs, feature_getter(sample)[variant])))
                overall_fit_durations.append(sample.duration_ms)

        variant_result["overall_grouped_cv_metrics"] = metric_dict(overall_cv_predictions, overall_cv_durations)
        variant_result["overall_fit_all_metrics"] = metric_dict(overall_fit_predictions, overall_fit_durations)
        results[variant] = variant_result
    return results


def markdown_table(stage_results: dict, variants: list[str], label: str) -> str:
    lines = [
        f"### {label}",
        "",
        "| Variant | CV mean abs % | CV RMSE % | CV max abs % | Fit-all mean abs % | Count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in variants:
        cv = stage_results[variant]["overall_grouped_cv_metrics"]
        fit_all = stage_results[variant]["overall_fit_all_metrics"]
        lines.append(
            f"| {variant} | {cv['mean_abs_rel_error_pct']:.2f} | {cv['rmse_rel_error_pct']:.2f} | "
            f"{cv['max_abs_rel_error_pct']:.2f} | {fit_all['mean_abs_rel_error_pct']:.2f} | {int(cv['count'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def per_model_table(stage_results: dict, variant: str, models: list[str], label: str) -> str:
    lines = [
        f"#### {label}: {variant}",
        "",
        "| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        result = stage_results[variant]["per_model"].get(model)
        if result is None:
            lines.append(f"| {model} | - | - | - | 0 |")
            continue
        cv = result["grouped_cv_metrics"]
        lines.append(
            f"| {model} | {cv['mean_abs_rel_error_pct']:.2f} | {cv['rmse_rel_error_pct']:.2f} | "
            f"{cv['max_abs_rel_error_pct']:.2f} | {int(cv['count'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exp_files = discover_exp_files(RESULTS_ROOT)

    prefill_samples_by_model: dict[str, list[PrefillBatchSample]] = defaultdict(list)
    decode_samples_by_model: dict[str, list[DecodeRoundSample]] = defaultdict(list)

    for exp_file in exp_files:
        for sample in build_prefill_samples(exp_file):
            prefill_samples_by_model[sample.model_name].append(sample)
        for sample in build_decode_samples(exp_file):
            decode_samples_by_model[sample.model_name].append(sample)

    prefill_variants = ["baseline", "no_bs", "min_len", "mid_len", "no_max_len"]
    decode_variants = ["baseline", "min_len", "mid_len", "no_max_len"]

    prefill_results = evaluate_stage_by_model(prefill_samples_by_model, prefill_variants, prefill_feature_map)
    decode_small_results = evaluate_stage_by_model(
        decode_samples_by_model,
        decode_variants,
        decode_feature_map,
        split_key="small",
    )
    decode_large_results = evaluate_stage_by_model(
        decode_samples_by_model,
        decode_variants,
        decode_feature_map,
        split_key="large",
    )

    summary = {
        "results_root": str(RESULTS_ROOT),
        "decode_threshold": DECODE_THRESHOLD,
        "num_exp_files": len(exp_files),
        "prefill_results": prefill_results,
        "decode_small_results": decode_small_results,
        "decode_large_results": decode_large_results,
    }
    with open(OUTPUT_ROOT / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    models = sorted(prefill_samples_by_model.keys())
    report_parts = [
        "# DistServe Live Fit Ablation",
        "",
        f"- Benchmark files: {len(exp_files)}",
        f"- Decode split threshold: {DECODE_THRESHOLD}",
        "",
        markdown_table(prefill_results, prefill_variants, "Prefill"),
        markdown_table(decode_small_results, decode_variants, "Decode (small batch)"),
        markdown_table(decode_large_results, decode_variants, "Decode (large batch)"),
        per_model_table(prefill_results, "baseline", models, "Prefill per-model"),
        per_model_table(prefill_results, "no_bs", models, "Prefill per-model"),
        per_model_table(prefill_results, "min_len", models, "Prefill per-model"),
        per_model_table(prefill_results, "mid_len", models, "Prefill per-model"),
        per_model_table(prefill_results, "no_max_len", models, "Prefill per-model"),
        per_model_table(decode_small_results, "baseline", models, "Decode small per-model"),
        per_model_table(decode_small_results, "min_len", models, "Decode small per-model"),
        per_model_table(decode_small_results, "mid_len", models, "Decode small per-model"),
        per_model_table(decode_small_results, "no_max_len", models, "Decode small per-model"),
    ]
    with open(OUTPUT_ROOT / "ablation_report.md", "w") as f:
        f.write("\n".join(report_parts))
        f.write("\n")

    print(f"Wrote {OUTPUT_ROOT / 'ablation_summary.json'}")
    print(f"Wrote {OUTPUT_ROOT / 'ablation_report.md'}")


if __name__ == "__main__":
    main()

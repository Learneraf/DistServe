from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_ALIAS_TO_KEY = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
    "llama1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}

DISTSERVE_RATE_PATTERN = re.compile(r"distserve-\d+-([\d.]+)\.exp$")
VLLM_ASCEND_RATE_PATTERN = re.compile(r"ascend-vllm-\d+-([\d.]+)\.exp$")


@dataclass
class PrefillEvent:
    timestamp: float
    end_timestamp: float
    prompt_len: int


@dataclass
class PrefillBatchSample:
    model_key: str
    source_file: str
    rate: float
    batch_size: int
    sum_prompt_len: int
    max_prompt_len: int
    sum_prompt_len_sq: int
    duration_ms: float


@dataclass
class TokenEvent:
    timestamp: float
    interval_ms: float
    prompt_len: int
    output_len: int
    token_idx: int
    current_context_len: int
    remaining_output_tokens: int


@dataclass
class RoundSample:
    model_key: str
    source_file: str
    rate: float
    batch_size: int
    sum_context_len: int
    max_context_len: int
    sum_remaining_output_tokens: int
    duration_ms: float


def default_results_root(backend: str) -> Path:
    if backend == "distserve_cuda":
        return Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result/fit")
    if backend == "vllm_ascend":
        return Path("/users/rh/ascend_data/ascend_vllm_holdout_fit")
    raise ValueError(f"Unsupported backend: {backend}")


def default_output_path(backend: str) -> Path:
    if backend == "distserve_cuda":
        return Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/"
            "distserve-cuda/fit_params_live_5p4d_filtered.json"
        )
    if backend == "vllm_ascend":
        return Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/"
            "vllm-ascend/fit_params_live_5p4d_filtered.json"
        )
    raise ValueError(f"Unsupported backend: {backend}")

def infer_backend_from_exp_path(exp_path: Path) -> str | None:
    if DISTSERVE_RATE_PATTERN.search(exp_path.name):
        return "distserve_cuda"
    if VLLM_ASCEND_RATE_PATTERN.search(exp_path.name):
        return "vllm_ascend"
    return None


def infer_model_key(exp_path: Path) -> str | None:
    return MODEL_ALIAS_TO_KEY.get(exp_path.parent.name)


def infer_rate(exp_path: Path) -> float:
    match = DISTSERVE_RATE_PATTERN.search(exp_path.name)
    if match:
        return float(match.group(1))
    match = VLLM_ASCEND_RATE_PATTERN.search(exp_path.name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Cannot infer request rate from {exp_path}")


def discover_exp_files(results_root: Path, backend: str) -> list[Path]:
    matches = sorted(path for path in results_root.rglob("*.exp") if path.is_file())
    if backend == "distserve_cuda":
        return [path for path in matches if infer_backend_from_exp_path(path) == "distserve_cuda"]
    if backend == "vllm_ascend":
        return [path for path in matches if infer_backend_from_exp_path(path) == "vllm_ascend"]
    raise ValueError(f"Unsupported backend: {backend}")


def extract_prefill_events(exp_path: Path) -> tuple[str | None, list[PrefillEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_key = infer_model_key(exp_path)
    rate = infer_rate(exp_path)
    events: list[PrefillEvent] = []

    for request in requests:
        lifecycle_events = request.get("lifecycle_events") or []
        lifecycle = {
            event["event_type"]: event["timestamp"]
            for event in lifecycle_events
            if isinstance(event, dict) and event.get("event_type") is not None
        }
        context_begin = lifecycle.get("context_begin")
        context_end = lifecycle.get("context_end")

        if context_begin is None or context_end is None:
            start_time = request.get("start_time")
            ftl = request.get("ftl")
            if start_time is None or ftl is None:
                continue
            context_begin = float(start_time)
            context_end = float(start_time) + float(ftl)

        events.append(
            PrefillEvent(
                timestamp=float(context_begin),
                end_timestamp=float(context_end),
                prompt_len=int(request["prompt_len"]),
            )
        )

    return model_key, events, rate


def cluster_events_into_batches(
    model_key: str,
    exp_path: Path,
    rate: float,
    events: list[PrefillEvent],
    cluster_gap_ms: float,
) -> list[PrefillBatchSample]:
    if not events:
        return []

    events = sorted(events, key=lambda event: event.timestamp)
    cluster_gap_s = cluster_gap_ms / 1000.0
    clusters: list[list[PrefillEvent]] = []
    current_cluster: list[PrefillEvent] = [events[0]]

    for event in events[1:]:
        if (event.timestamp - current_cluster[-1].timestamp) <= cluster_gap_s:
            current_cluster.append(event)
        else:
            clusters.append(current_cluster)
            current_cluster = [event]
    clusters.append(current_cluster)

    samples: list[PrefillBatchSample] = []
    for cluster in clusters:
        prompt_lens = [event.prompt_len for event in cluster]
        samples.append(
            PrefillBatchSample(
                model_key=model_key,
                source_file=str(exp_path),
                rate=rate,
                batch_size=len(cluster),
                sum_prompt_len=sum(prompt_lens),
                max_prompt_len=max(prompt_lens),
                sum_prompt_len_sq=sum(prompt_len * prompt_len for prompt_len in prompt_lens),
                duration_ms=1000.0 * statistics.median(
                    event.end_timestamp - event.timestamp for event in cluster
                ),
            )
        )

    return samples


def extract_decode_events(
    exp_path: Path,
    backend: str,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[str | None, list[TokenEvent], float]:
    with open(exp_path) as f:
        requests = json.load(f)

    model_key = infer_model_key(exp_path)
    rate = infer_rate(exp_path)
    events: list[TokenEvent] = []

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
                TokenEvent(
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


def cluster_events_into_rounds(
    model_key: str,
    exp_path: Path,
    rate: float,
    events: list[TokenEvent],
    cluster_gap_ms: float,
) -> list[RoundSample]:
    if not events:
        return []

    events = sorted(events, key=lambda event: event.timestamp)
    cluster_gap_s = cluster_gap_ms / 1000.0
    clusters: list[list[TokenEvent]] = []
    current_cluster: list[TokenEvent] = [events[0]]

    for event in events[1:]:
        if (event.timestamp - current_cluster[-1].timestamp) <= cluster_gap_s:
            current_cluster.append(event)
        else:
            clusters.append(current_cluster)
            current_cluster = [event]
    clusters.append(current_cluster)

    samples: list[RoundSample] = []
    for cluster in clusters:
        duration_ms = statistics.median(event.interval_ms for event in cluster)
        samples.append(
            RoundSample(
                model_key=model_key,
                source_file=str(exp_path),
                rate=rate,
                batch_size=len(cluster),
                sum_context_len=sum(event.current_context_len for event in cluster),
                max_context_len=max(event.current_context_len for event in cluster),
                sum_remaining_output_tokens=sum(event.remaining_output_tokens for event in cluster),
                duration_ms=duration_ms,
            )
        )

    return samples


def fit_prefill_five_param_model(
    samples: list[PrefillBatchSample],
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
            float(sample.batch_size),
            float(sample.sum_prompt_len),
            float(sample.max_prompt_len),
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


def fit_decode_four_param_model(
    samples: list[RoundSample],
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
            float(sample.max_context_len),
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


def write_prefill_samples_csv(path: Path, samples: list[PrefillBatchSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "source_file",
            "rate",
            "batch_size",
            "sum_prompt_len",
            "max_prompt_len",
            "sum_prompt_len_sq",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.model_key,
                sample.source_file,
                sample.rate,
                sample.batch_size,
                sample.sum_prompt_len,
                sample.max_prompt_len,
                sample.sum_prompt_len_sq,
                sample.duration_ms,
            ])


def write_decode_samples_csv(path: Path, samples: list[RoundSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "source_file",
            "rate",
            "batch_size",
            "sum_context_len",
            "max_context_len",
            "sum_remaining_output_tokens",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.model_key,
                sample.source_file,
                sample.rate,
                sample.batch_size,
                sample.sum_context_len,
                sample.max_context_len,
                sample.sum_remaining_output_tokens,
                sample.duration_ms,
            ])


def collect_prefill_samples(
    results_root: Path,
    backend: str,
    cluster_gap_ms: float,
) -> tuple[dict[str, list[PrefillBatchSample]], list[str], int]:
    samples_by_model: dict[str, list[PrefillBatchSample]] = {}
    ignored_files: list[str] = []
    exp_files = discover_exp_files(results_root, backend)

    for exp_path in exp_files:
        model_key, events, rate = extract_prefill_events(exp_path)
        if model_key is None:
            ignored_files.append(str(exp_path))
            continue
        samples = cluster_events_into_batches(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=cluster_gap_ms,
        )
        samples_by_model.setdefault(model_key, []).extend(samples)

    return samples_by_model, ignored_files, len(exp_files)


def collect_decode_samples(
    results_root: Path,
    backend: str,
    cluster_gap_ms: float,
    exclude_last_interval: bool,
    min_interval_ms: float,
) -> tuple[dict[str, list[RoundSample]], list[str], int]:
    samples_by_model: dict[str, list[RoundSample]] = {}
    ignored_files: list[str] = []
    exp_files = discover_exp_files(results_root, backend)

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
        samples = cluster_events_into_rounds(
            model_key=model_key,
            exp_path=exp_path,
            rate=rate,
            events=events,
            cluster_gap_ms=cluster_gap_ms,
        )
        samples_by_model.setdefault(model_key, []).extend(samples)

    return samples_by_model, ignored_files, len(exp_files)


def create_tp1_entry(profile: dict, model_key: str, decode_large_small_bs_threshold: int) -> dict:
    model_profile = profile.setdefault(model_key, {})
    tp_profile = model_profile.setdefault("1", {})
    tp_profile.setdefault("decoding_large_small_bs_threshold", decode_large_small_bs_threshold)
    return tp_profile
from __future__ import annotations

import ast
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path("/users/rh/DistServe")
SIMULATE_DIST = REPO_ROOT / "simdistserve" / "benchmarks" / "simulate_dist.py"


MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "llama_1B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
        "workload_dir": "llama-3.2-1B",
        "cuda_real_dir": "llama_1B",
        "ascend_real_dir": "llama1B",
    },
    "llama_3B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
        "workload_dir": "llama-3.2-3B",
        "cuda_real_dir": "llama_3B",
        "ascend_real_dir": "llama3B",
    },
    "llama_7B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
        "workload_dir": "llama-2-7b",
        "cuda_real_dir": "llama_7B",
        "ascend_real_dir": "llama7B",
    },
    "llama_8B": {
        "model_path": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
        "workload_dir": "llama-3.1-8B",
        "cuda_real_dir": "llama_8B",
        "ascend_real_dir": "llama8B",
    },
}

DEFAULT_MODELS = list(MODEL_CONFIGS.keys())
DEFAULT_RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]


@dataclass
class ComputeSample:
    duration_ms: float
    x1: float
    x2: float


@dataclass
class RequestComponentSample:
    req_id: int
    prefill_queue_ms: float
    prefill_compute_ms: float
    handoff_queue_ms: float
    handoff_service_ms: float
    decode_queue_first_ms: float
    decode_queue_total_ms: float
    first_decode_compute_ms: float
    decode_compute_total_ms: float
    total_latency_ms_raw: float


def default_seed_profile(backend: str) -> Path:
    if backend == "distserve":
        return REPO_ROOT / "simdistserve" / "estimators" / "profiled_data" / "distserve-cuda" / "fit_params_live_cuda_data.json"
    if backend == "vllm_ascend":
        preferred = Path("/users/rh/ascend_data/fitted_profiles/fit_params_live_ascend_data.json")
        if preferred.exists():
            return preferred
        return REPO_ROOT / "simdistserve" / "estimators" / "profiled_data" / "vllm-ascend" / "fit_params_live.json"
    raise ValueError(f"Unsupported backend: {backend}")


def default_output_profile(backend: str) -> Path:
    if backend == "distserve":
        return REPO_ROOT / "simdistserve" / "estimators" / "profiled_data" / "distserve-cuda" / "fit_params_split_component.json"
    if backend == "vllm_ascend":
        return Path("/users/rh/ascend_data/fitted_profiles/fit_params_split_component.json")
    raise ValueError(f"Unsupported backend: {backend}")


def default_validation_root(backend: str) -> Path:
    if backend == "distserve":
        return REPO_ROOT / "simdistserve" / "benchmarks" / "results" / "validation" / "distserve_cuda_split_component"
    if backend == "vllm_ascend":
        return Path("/users/rh/ascend_data/validation/split_component_profile")
    raise ValueError(f"Unsupported backend: {backend}")


def workload_path_for_split(workload_root: Path, model_alias: str, split: str) -> Path:
    return workload_root / MODEL_CONFIGS[model_alias]["workload_dir"] / f"{split}.jsonl"


def normalize_rate_label_for_backend(backend: str, rate: str) -> str:
    if backend == "vllm_ascend" and str(rate) == "1":
        return "1.0"
    return str(rate)


def resolve_actual_exp_path(
    backend: str,
    bench_root: Path,
    model_alias: str,
    num_prompts: int,
    rate: str,
) -> Path:
    if backend == "distserve":
        model_dir = MODEL_CONFIGS[model_alias]["cuda_real_dir"]
        return bench_root / model_dir / f"distserve-{num_prompts}-{rate}.exp"

    if backend == "vllm_ascend":
        model_dir = MODEL_CONFIGS[model_alias]["ascend_real_dir"]
        rate_label = normalize_rate_label_for_backend(backend, rate)
        return bench_root / model_dir / f"ascend-vllm-{num_prompts}-{rate_label}.exp"

    raise ValueError(f"Unsupported backend: {backend}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_actual_requests(exp_path: Path) -> list[dict[str, Any]]:
    with open(exp_path) as f:
        return json.load(f)


def actual_total_latency_ms(request: dict[str, Any], prefer_visible_completion: bool = True) -> float:
    if prefer_visible_completion:
        start_time = request.get("start_time")
        token_timestamps = request.get("token_timestamps") or []
        if start_time is not None and token_timestamps:
            return max(float(token_timestamps[-1]) - float(start_time), 0.0) * 1000.0
    return float(request.get("latency", 0.0)) * 1000.0


def actual_ftl_ms(request: dict[str, Any]) -> float:
    return float(request.get("ftl", 0.0)) * 1000.0


def extract_actual_migration_ms(request: dict[str, Any]) -> float | None:
    lifecycle_events = request.get("lifecycle_events") or []
    if not lifecycle_events:
        return None

    lifecycle = {event.get("event_type"): event.get("timestamp") for event in lifecycle_events}
    begin = lifecycle.get("migration_begin")
    end = lifecycle.get("migration_end")
    if begin is None or end is None:
        return None
    return max(float(end) - float(begin), 0.0) * 1000.0


def profile_env_var(backend: str) -> str:
    if backend == "distserve":
        return "SIMDISTSERVE_DISTSERVE_PROFILE"
    if backend == "vllm_ascend":
        return "SIMDISTSERVE_VLLM_ASCEND_PROFILE"
    raise ValueError(f"Unsupported backend: {backend}")


def prefill_first_token_visible_immediately_for_backend(backend: str) -> bool:
    return backend != "vllm_ascend"


def load_profile(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def model_path_for_alias(model_alias: str) -> str:
    return MODEL_CONFIGS[model_alias]["model_path"]


def model_key_for_alias(model_alias: str) -> str:
    return MODEL_CONFIGS[model_alias]["model_path"]


def load_model_profile(profile_path: Path, model_alias: str) -> dict[str, Any]:
    profile = load_profile(profile_path)
    return profile[model_key_for_alias(model_alias)]["1"]


def run_simulation(
    backend: str,
    python_bin: str,
    profile: Path,
    model_alias: str,
    rate: str,
    output_dir: Path,
    workload_path: Path,
    num_prompts: int,
    seed: int,
    arrival: str,
    cv: float,
    handoff_delay_ms: float | None = None,
    handoff_delay_per_token_ms: float | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env[profile_env_var(backend)] = str(profile)

    cmd = [
        python_bin,
        str(SIMULATE_DIST),
        "--backend", backend,
        "--model", model_path_for_alias(model_alias),
        "--seed", str(seed),
        "--rate", str(rate),
        "--N", str(num_prompts),
        "--arrival", arrival,
        "--workload", str(workload_path),
        "--name", f"{model_alias}/rate_{rate}",
        "--output", str(output_dir / "sharegpt.json.sim.csv"),
        "--output-request-info", str(output_dir / "request_info.csv"),
        "--output-request-event", str(output_dir / "request_event.csv"),
        "--output-request-latency", str(output_dir / "request_latency_raw.csv"),
        "--output-worker", str(output_dir / "worker_event.csv"),
        "--slo-scales", "[1.0]",
        "--cv", str(cv),
    ]
    if handoff_delay_ms is not None:
        cmd.extend(["--handoff-delay-ms", str(handoff_delay_ms)])
    if handoff_delay_per_token_ms is not None:
        cmd.extend(["--handoff-delay-per-token-ms", str(handoff_delay_per_token_ms)])

    subprocess.run(cmd, check=True, env=env, cwd=REPO_ROOT)


def _parse_list_field(raw_value: str) -> list[float]:
    if raw_value in ("", None):
        return []
    value = ast.literal_eval(raw_value)
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def collect_prefill_compute_samples(worker_csv: Path) -> list[ComputeSample]:
    rows = list(csv.DictReader(worker_csv.open()))
    samples: list[ComputeSample] = []
    for row in rows:
        if row["event_type"] != "do_prefill":
            continue
        prefill_lens = _parse_list_field(row["prefill_batch"])
        if not prefill_lens:
            continue
        duration_ms = float(row["duration"])
        sum_prompt_len = float(sum(prefill_lens))
        sum_prompt_len_sq = float(sum(length * length for length in prefill_lens))
        samples.append(
            ComputeSample(
                duration_ms=duration_ms,
                x1=sum_prompt_len,
                x2=sum_prompt_len_sq,
            )
        )
    return samples


def collect_decode_compute_samples(worker_csv: Path) -> list[ComputeSample]:
    rows = list(csv.DictReader(worker_csv.open()))
    samples: list[ComputeSample] = []
    for row in rows:
        if row["event_type"] != "do_decode":
            continue
        context_lens = _parse_list_field(row["decode_batch"])
        if not context_lens:
            continue
        batch_size = float(row["decode_bs"])
        duration_ms = float(row["duration"])
        total_tokens = float(sum(context_lens) + batch_size)
        samples.append(
            ComputeSample(
                duration_ms=duration_ms,
                x1=total_tokens,
                x2=batch_size,
            )
        )
    return samples


def _fit_relative_error_three_param(samples: list[ComputeSample]) -> tuple[list[float], dict[str, float]]:
    if not samples:
        raise ValueError("No samples to fit.")

    design = []
    rhs = []
    raw_rows = []
    durations = []
    for sample in samples:
        duration = max(sample.duration_ms, 1e-6)
        row = [1.0, sample.x1, sample.x2]
        design.append([value / duration for value in row])
        rhs.append(1.0)
        raw_rows.append(row)
        durations.append(duration)

    coeffs, _, _, _ = np.linalg.lstsq(np.array(design, dtype=float), np.array(rhs, dtype=float), rcond=None)
    predictions = np.array([float(np.dot(coeffs, row)) for row in raw_rows], dtype=float)
    durations_arr = np.array(durations, dtype=float)
    rel_errors = (predictions - durations_arr) / durations_arr
    abs_rel_errors = np.abs(rel_errors)
    metrics = {
        "count": float(len(samples)),
        "mean_abs_rel_error_pct": float(100.0 * abs_rel_errors.mean()),
        "rmse_rel_error_pct": float(100.0 * math.sqrt(np.mean(rel_errors * rel_errors))),
        "max_abs_rel_error_pct": float(100.0 * abs_rel_errors.max()),
    }
    return coeffs.tolist(), metrics


def fit_prefill_compute(samples: list[ComputeSample]) -> tuple[list[float], dict[str, float]]:
    return _fit_relative_error_three_param(samples)


def fit_decode_compute(samples: list[ComputeSample]) -> tuple[list[float], dict[str, float]]:
    return _fit_relative_error_three_param(samples)


def extract_request_components(
    request_event_csv: Path,
    *,
    backend: str,
) -> list[RequestComponentSample]:
    rows = list(csv.DictReader(request_event_csv.open()))
    by_req_id: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        req_id = int(row["req_id"])
        row_copy = dict(row)
        row_copy["start_time"] = float(row["start_time"])
        row_copy["end_time"] = float(row["end_time"])
        row_copy["duration"] = float(row["duration"])
        by_req_id.setdefault(req_id, []).append(row_copy)

    samples: list[RequestComponentSample] = []
    for req_id in sorted(by_req_id):
        events = sorted(by_req_id[req_id], key=lambda item: (item["start_time"], item["end_time"]))
        by_type: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            by_type.setdefault(event["event_type"], []).append(event)

        init_rows = by_type.get("init")
        exit_rows = by_type.get("exit_system")
        if not init_rows or not exit_rows:
            raise ValueError(f"Request {req_id} is missing init/exit_system events in {request_event_csv}")

        first_visible_rows = by_type.get("first_token_visible") or []
        do_prefill_rows = by_type.get("do_prefill") or []
        do_decode_rows = by_type.get("do_decode") or []
        wait_decode_rows = by_type.get("wait_decode") or []

        init_start = init_rows[0]["start_time"]
        first_do_prefill = do_prefill_rows[0] if do_prefill_rows else None
        first_visible_time = first_visible_rows[0]["start_time"] if first_visible_rows else None
        first_do_decode = do_decode_rows[0] if do_decode_rows else None

        if first_do_decode is None:
            decode_queue_first_ms = 0.0
            first_decode_compute_ms = 0.0
        elif first_visible_time is not None and first_visible_time <= first_do_decode["start_time"]:
            decode_queue_first_ms = 0.0
            first_decode_compute_ms = 0.0
        else:
            decode_queue_first_ms = wait_decode_rows[0]["duration"] if wait_decode_rows else 0.0
            first_decode_compute_ms = first_do_decode["duration"]

        if backend == "distserve" and first_do_prefill is not None:
            prefill_queue_ms = max(first_do_prefill["start_time"] - init_start, 0.0)
        else:
            prefill_queue_ms = sum(event["duration"] for event in by_type.get("wait_prefill", []))

        total_latency_ms_raw = max(exit_rows[-1]["end_time"] - init_rows[0]["start_time"], 0.0)

        samples.append(
            RequestComponentSample(
                req_id=req_id,
                prefill_queue_ms=prefill_queue_ms,
                prefill_compute_ms=sum(event["duration"] for event in by_type.get("do_prefill", [])),
                handoff_queue_ms=sum(event["duration"] for event in by_type.get("wait_handoff", [])),
                handoff_service_ms=sum(event["duration"] for event in by_type.get("do_handoff", [])),
                decode_queue_first_ms=decode_queue_first_ms,
                decode_queue_total_ms=sum(event["duration"] for event in wait_decode_rows),
                first_decode_compute_ms=first_decode_compute_ms,
                decode_compute_total_ms=sum(event["duration"] for event in do_decode_rows),
                total_latency_ms_raw=total_latency_ms_raw,
            )
        )
    return samples


def handoff_tokens_for_request(backend: str, workload_item: dict[str, Any]) -> float:
    prompt_len = float(workload_item["prompt_len"])
    if backend == "vllm_ascend" and float(workload_item.get("output_len", workload_item.get("output_tokens", 0))) > 0:
        return prompt_len + 1.0
    return prompt_len


def fit_linear_model_ms(
    rows: list[list[float]],
    targets_ms: list[float],
) -> tuple[list[float], dict[str, float]]:
    if not rows:
        raise ValueError("No rows to fit.")

    design = np.array(rows, dtype=float)
    targets = np.array(targets_ms, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    predictions = design @ coeffs
    abs_errors = np.abs(predictions - targets)
    metrics = {
        "count": float(len(rows)),
        "mae_ms": float(abs_errors.mean()),
        "rmse_ms": float(math.sqrt(np.mean((predictions - targets) ** 2))),
        "p95_abs_error_ms": float(np.percentile(abs_errors, 95)),
        "max_abs_error_ms": float(abs_errors.max()),
    }
    return coeffs.tolist(), metrics


def fit_huber_linear_model_ms(
    rows: list[list[float]],
    targets_ms: list[float],
    *,
    delta: float = 1.5,
    max_iter: int = 25,
    tol: float = 1e-9,
) -> tuple[list[float], dict[str, float]]:
    if not rows:
        raise ValueError("No rows to fit.")

    design = np.array(rows, dtype=float)
    targets = np.array(targets_ms, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)

    for _ in range(max_iter):
        residuals = targets - (design @ coeffs)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        scale = 1.4826 * mad
        if not math.isfinite(scale) or scale < tol:
            break

        threshold = delta * scale
        weights = np.ones_like(residuals)
        large_mask = np.abs(residuals) > threshold
        weights[large_mask] = threshold / np.abs(residuals[large_mask])

        sqrt_weights = np.sqrt(weights)
        weighted_design = design * sqrt_weights[:, None]
        weighted_targets = targets * sqrt_weights
        next_coeffs, _, _, _ = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
        if np.allclose(next_coeffs, coeffs, atol=tol, rtol=0.0):
            coeffs = next_coeffs
            break
        coeffs = next_coeffs

    predictions = design @ coeffs
    abs_errors = np.abs(predictions - targets)
    metrics = {
        "count": float(len(rows)),
        "mae_ms": float(abs_errors.mean()),
        "rmse_ms": float(math.sqrt(np.mean((predictions - targets) ** 2))),
        "p95_abs_error_ms": float(np.percentile(abs_errors, 95)),
        "max_abs_error_ms": float(abs_errors.max()),
    }
    return coeffs.tolist(), metrics


def build_compute_only_profile(
    backend: str,
    prefill_coeffs_by_model: dict[str, list[float]],
    decode_coeffs_by_model: dict[str, list[float]],
) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    prefill_first_token_visible_immediately = prefill_first_token_visible_immediately_for_backend(backend)
    for model_alias in sorted(prefill_coeffs_by_model):
        model_key = model_key_for_alias(model_alias)
        profile[model_key] = {
            "1": {
                "decoding_large_small_bs_threshold": 95,
                "prefill": prefill_coeffs_by_model[model_alias],
                "decoding_smallbs": decode_coeffs_by_model[model_alias],
                "decoding_largebs": decode_coeffs_by_model[model_alias],
                "handoff_delay_ms": 0.0,
                "handoff_delay_per_token_ms": 0.0,
                "prefill_first_token_visible_immediately": prefill_first_token_visible_immediately,
            }
        }
    return profile


def add_handoff_and_queue_models(
    profile: dict[str, Any],
    handoff_coeffs_by_model: dict[str, list[float]],
    queue_ftl_coeffs_by_model: dict[str, list[float]],
    queue_total_coeffs_by_model: dict[str, list[float]],
    queue_piecewise_by_model: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    for model_alias in sorted(handoff_coeffs_by_model):
        tp_profile = profile[model_key_for_alias(model_alias)]["1"]
        handoff_coeffs = handoff_coeffs_by_model[model_alias]
        tp_profile["handoff_delay_ms"] = float(handoff_coeffs[0])
        tp_profile["handoff_delay_per_token_ms"] = float(handoff_coeffs[1])
        tp_profile["queue_model_ftl"] = queue_ftl_coeffs_by_model[model_alias]
        tp_profile["queue_model_total"] = queue_total_coeffs_by_model[model_alias]
        if queue_piecewise_by_model and model_alias in queue_piecewise_by_model:
            tp_profile["queue_model_piecewise"] = queue_piecewise_by_model[model_alias]
    return profile


def write_profile(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(profile, f, indent=4)
        f.write("\n")


def _select_queue_models_for_rate(
    model_profile: dict[str, Any],
    rate: float | str | None,
) -> tuple[Any, Any]:
    piecewise = model_profile.get("queue_model_piecewise")
    if piecewise and rate is not None:
        threshold = float(piecewise["rate_threshold"])
        bucket = "low" if float(rate) <= threshold else "high"
        bucket_profile = piecewise[bucket]
        return bucket_profile.get("ftl"), bucket_profile.get("total")
    return model_profile.get("queue_model_ftl"), model_profile.get("queue_model_total")


def build_calibrated_latency_rows(
    model_profile: dict[str, Any],
    components: list[RequestComponentSample],
    rate: float | str | None = None,
) -> list[dict[str, float]]:
    queue_ftl, queue_total = _select_queue_models_for_rate(model_profile, rate)
    prefill_first_token_visible_immediately = bool(
        model_profile.get("prefill_first_token_visible_immediately", True)
    )

    rows: list[dict[str, float]] = []
    for component in components:
        if queue_ftl and queue_total:
            if prefill_first_token_visible_immediately and len(queue_ftl) == 2:
                ftl_queue_pred = float(
                    np.dot(
                        np.array(queue_ftl, dtype=float),
                        np.array(
                            [
                                1.0,
                                component.prefill_queue_ms,
                            ],
                            dtype=float,
                        ),
                    )
                )
            else:
                ftl_queue_pred = float(
                    np.dot(
                        np.array(queue_ftl, dtype=float),
                        np.array(
                            [
                                1.0,
                                component.prefill_queue_ms,
                                component.handoff_queue_ms,
                                component.decode_queue_first_ms,
                            ],
                            dtype=float,
                        ),
                    )
                )
            total_queue_pred = float(
                np.dot(
                    np.array(queue_total, dtype=float),
                    np.array(
                        [
                            1.0,
                            component.prefill_queue_ms,
                            component.handoff_queue_ms,
                            component.decode_queue_total_ms,
                        ],
                        dtype=float,
                    ),
                )
            )
            ftl_queue_pred = max(ftl_queue_pred, 0.0)
            total_queue_pred = max(total_queue_pred, 0.0)
            if prefill_first_token_visible_immediately and len(queue_ftl) == 2:
                first_token_latency = component.prefill_compute_ms + ftl_queue_pred
            else:
                first_token_latency = (
                    component.prefill_compute_ms
                    + component.handoff_service_ms
                    + component.first_decode_compute_ms
                    + ftl_queue_pred
                )
            total_latency = (
                component.prefill_compute_ms
                + component.handoff_service_ms
                + component.decode_compute_total_ms
                + total_queue_pred
            )
        else:
            first_token_latency = (
                component.prefill_queue_ms
                + component.prefill_compute_ms
                + component.handoff_queue_ms
                + component.handoff_service_ms
                + component.decode_queue_first_ms
                + component.first_decode_compute_ms
            )
            total_latency = component.total_latency_ms_raw

        total_latency = max(total_latency, first_token_latency)
        decoding_latency = max(total_latency - first_token_latency, 0.0)
        rows.append(
            {
                "req_id": component.req_id,
                "first_token_latency": first_token_latency,
                "decoding_latency": decoding_latency,
                "total_latency": total_latency,
            }
        )
    return rows


def summarize_latency_rows(rows: list[dict[str, float]], prefill_slo_s: float, decode_slo_s: float, total_slo_s: float) -> dict[str, Any]:
    ftl_ms = [float(row["first_token_latency"]) for row in rows]
    decode_ms = [float(row["decoding_latency"]) for row in rows]
    total_ms = [float(row["total_latency"]) for row in rows]

    prefill_ok = decode_ok = total_ok = both_ok = 0
    for ftl_ms_i, decode_ms_i, total_ms_i in zip(ftl_ms, decode_ms, total_ms):
        prefill_ok_i = ftl_ms_i / 1000.0 <= prefill_slo_s
        decode_ok_i = decode_ms_i / 1000.0 <= decode_slo_s
        total_ok_i = total_ms_i / 1000.0 <= total_slo_s
        both_ok_i = prefill_ok_i and decode_ok_i
        prefill_ok += int(prefill_ok_i)
        decode_ok += int(decode_ok_i)
        total_ok += int(total_ok_i)
        both_ok += int(both_ok_i)

    total = len(rows)
    return {
        "slo": {
            "prefill": 100.0 * prefill_ok / total,
            "decode": 100.0 * decode_ok / total,
            "total": 100.0 * total_ok / total,
            "both": 100.0 * both_ok / total,
        },
        "ftl": summarize_series_ms(ftl_ms),
        "decode": summarize_series_ms(decode_ms),
        "total": summarize_series_ms(total_ms),
    }


def summarize_actual_requests(
    requests: list[dict[str, Any]],
    prefill_slo_s: float,
    decode_slo_s: float,
    total_slo_s: float,
) -> dict[str, Any]:
    rows = []
    for req_id, request in enumerate(requests):
        ftl_ms = actual_ftl_ms(request)
        total_ms = actual_total_latency_ms(request)
        rows.append(
            {
                "req_id": req_id,
                "first_token_latency": ftl_ms,
                "decoding_latency": max(total_ms - ftl_ms, 0.0),
                "total_latency": total_ms,
            }
        )
    return summarize_latency_rows(rows, prefill_slo_s, decode_slo_s, total_slo_s)


def summarize_series_ms(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms, dtype=float)
    if len(arr) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def flatten_case_for_csv(case: dict[str, Any]) -> dict[str, Any]:
    errors = case["errors"]
    return {
        "model": case["model"],
        "rate": case["rate"],
        "actual_prefill_slo_pct": case["actual"]["slo"]["prefill"],
        "new_prefill_slo_pct": case["new_sim"]["slo"]["prefill"],
        "raw_prefill_slo_pct": case["raw_sim"]["slo"]["prefill"],
        "new_prefill_slo_abs_diff_pct": errors["new_slo_abs_diff_prefill_pct"],
        "raw_prefill_slo_abs_diff_pct": errors["raw_slo_abs_diff_prefill_pct"],
        "actual_decode_slo_pct": case["actual"]["slo"]["decode"],
        "new_decode_slo_pct": case["new_sim"]["slo"]["decode"],
        "raw_decode_slo_pct": case["raw_sim"]["slo"]["decode"],
        "new_decode_slo_abs_diff_pct": errors["new_slo_abs_diff_decode_pct"],
        "raw_decode_slo_abs_diff_pct": errors["raw_slo_abs_diff_decode_pct"],
        "actual_total_slo_pct": case["actual"]["slo"]["total"],
        "new_total_slo_pct": case["new_sim"]["slo"]["total"],
        "raw_total_slo_pct": case["raw_sim"]["slo"]["total"],
        "new_total_slo_abs_diff_pct": errors["new_slo_abs_diff_total_pct"],
        "raw_total_slo_abs_diff_pct": errors["raw_slo_abs_diff_total_pct"],
        "actual_ftl_mean_ms": case["actual"]["ftl"]["mean_ms"],
        "new_ftl_mean_ms": case["new_sim"]["ftl"]["mean_ms"],
        "raw_ftl_mean_ms": case["raw_sim"]["ftl"]["mean_ms"],
        "new_ftl_mean_ms_abs_ms": errors["new_ftl_mean_ms_abs_ms"],
        "raw_ftl_mean_ms_abs_ms": errors["raw_ftl_mean_ms_abs_ms"],
        "actual_total_p95_ms": case["actual"]["total"]["p95_ms"],
        "new_total_p95_ms": case["new_sim"]["total"]["p95_ms"],
        "raw_total_p95_ms": case["raw_sim"]["total"]["p95_ms"],
        "new_total_p95_ms_abs_ms": errors["new_total_p95_ms_abs_ms"],
        "raw_total_p95_ms_abs_ms": errors["raw_total_p95_ms_abs_ms"],
    }


def summarize_error_metrics(cases: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_key: dict[str, list[float]] = {}
    for case in cases:
        for key, value in case["errors"].items():
            if value is None:
                continue
            by_key.setdefault(key, []).append(float(value))

    aggregate: dict[str, dict[str, float]] = {}
    for key, values in by_key.items():
        arr = np.array(values, dtype=float)
        aggregate[key] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }
    return aggregate


def write_latency_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["req_id", "first_token_latency", "decoding_latency", "total_latency"])
        writer.writeheader()
        writer.writerows(rows)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def make_python_bin_default() -> str:
    return sys.executable

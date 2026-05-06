#!/usr/bin/env python3
"""Fit vLLM P/D profiles from explicit scheduler batch traces.

Inputs are the traces produced under /users/rh/cdua_data/<model>/:
prefill_batch_trace.jsonl.gz, decode_batch_trace.jsonl.gz, and vllm-pd-*.exp.

Default model:
    prefill_ms = a + b * sum_prompt_len + c * sum_prompt_len_sq
    decode_ms  = a + b * sum_next_context_len + c * batch_size
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from utils import MODEL_ALIAS_TO_KEY
except ImportError:
    MODEL_ALIAS_TO_KEY = {
        "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
        "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
        "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
        "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
    }


@dataclass
class RequestInfo:
    output_len: int
    context_end: float | None


@dataclass
class PrefillSample:
    model_key: str
    source_file: str
    batch_id: str
    batch_size: int
    sum_prompt_len: int
    max_prompt_len: int
    sum_prompt_len_sq: int
    duration_ms: float


@dataclass
class DecodeSample:
    model_key: str
    source_file: str
    batch_id: str
    batch_size: int
    sum_next_context_len: int
    sum_context_len: int
    max_context_len: int
    sum_remaining_output_tokens: int
    duration_ms: float


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def read_exp_requests(exp_path: Path) -> dict[str, RequestInfo]:
    requests = json.loads(exp_path.read_text())
    request_infos: dict[str, RequestInfo] = {}
    for request in requests:
        lifecycle = {
            event.get("event_type"): event.get("timestamp")
            for event in request.get("lifecycle_events") or []
            if isinstance(event, dict)
        }
        context_end = lifecycle.get("context_end")
        info = RequestInfo(
            output_len=int(request["output_len"]),
            context_end=float(context_end) if context_end is not None else None,
        )
        for key in ("vllm_internal_request_id", "client_request_id"):
            request_id = request.get(key)
            if request_id:
                request_infos[str(request_id)] = info
    return request_infos


def discover_model_dirs(results_root: Path) -> list[Path]:
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and (
            (path / "prefill_batch_trace.jsonl.gz").exists()
            or (path / "prefill_batch_trace.jsonl").exists()
        )
        and (
            (path / "decode_batch_trace.jsonl.gz").exists()
            or (path / "decode_batch_trace.jsonl").exists()
        )
    )


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def infer_model_key(model_dir: Path, explicit_model_key: str | None) -> str:
    if explicit_model_key is not None:
        return explicit_model_key
    return MODEL_ALIAS_TO_KEY.get(model_dir.name, model_dir.name)


def infer_exp_path(model_dir: Path) -> Path:
    matches = sorted(model_dir.glob("vllm-pd-*.exp"))
    if not matches:
        raise ValueError(f"No vllm-pd-*.exp found in {model_dir}")
    if len(matches) > 1:
        print(f"Using first exp file under {model_dir}: {matches[0]}")
    return matches[0]


def read_model_exp_requests(model_dir: Path) -> dict[str, RequestInfo]:
    matches = sorted(model_dir.glob("vllm-pd-*.exp"))
    if not matches:
        raise ValueError(f"No vllm-pd-*.exp found in {model_dir}")
    merged: dict[str, RequestInfo] = {}
    for exp_path in matches:
        merged.update(read_exp_requests(exp_path))
    return merged


def infer_event_trace_path(
    model_dir: Path,
    side: str,
    trace_root: Path | None,
) -> Path | None:
    candidates = [
        model_dir / f"{side}_event_trace.jsonl.gz",
        model_dir / f"{side}_event_trace.jsonl",
    ]
    if trace_root is not None:
        candidates.extend(
            [
                trace_root / model_dir.name / f"{side}_event_trace.jsonl.gz",
                trace_root / model_dir.name / f"{side}_event_trace.jsonl",
            ]
        )
    return first_existing(candidates)


def load_event_durations(
    event_trace_path: Path | None,
    duration_source: str,
) -> dict[str, float]:
    if event_trace_path is None or duration_source == "scheduler":
        return {}
    event_type = f"{duration_source}_end"
    durations: dict[str, float] = {}
    for row in read_jsonl(event_trace_path):
        if row.get("event_type") != event_type:
            continue
        batch_id = row.get("batch_id")
        duration_ms = row.get("duration_ms")
        if batch_id is None or duration_ms is None:
            continue
        durations[str(batch_id)] = float(duration_ms)
    return durations


def load_prefill_samples(
    trace_path: Path,
    exp_requests: dict[str, RequestInfo],
    model_key: str,
    min_duration_ms: float,
    max_duration_ms: float | None,
    event_durations: dict[str, float],
) -> list[PrefillSample]:
    samples: list[PrefillSample] = []
    for row in read_jsonl(trace_path):
        records = row.get("requests") or []
        if not records:
            continue
        batch_id = str(row.get("batch_id", ""))
        if batch_id in event_durations:
            duration_ms = event_durations[batch_id]
        else:
            context_ends = [
                exp_requests[req_id].context_end
                for req_id in row.get("request_ids") or []
                if req_id in exp_requests
                and exp_requests[req_id].context_end is not None
            ]
            if not context_ends:
                continue
            duration_ms = 1000.0 * (max(context_ends) - float(row["monotonic_time"]))
        if duration_ms < min_duration_ms:
            continue
        if max_duration_ms is not None and duration_ms > max_duration_ms:
            continue

        prompt_lens = [
            int(record.get("num_scheduled_tokens") or record.get("num_prompt_tokens") or 0)
            for record in records
        ]
        prompt_lens = [value for value in prompt_lens if value > 0]
        if not prompt_lens:
            continue
        samples.append(
            PrefillSample(
                model_key=model_key,
                source_file=str(trace_path),
                batch_id=batch_id,
                batch_size=len(prompt_lens),
                sum_prompt_len=sum(prompt_lens),
                max_prompt_len=max(prompt_lens),
                sum_prompt_len_sq=sum(value * value for value in prompt_lens),
                duration_ms=duration_ms,
            )
        )
    return samples


def load_decode_samples(
    trace_path: Path,
    exp_requests: dict[str, RequestInfo],
    model_key: str,
    min_duration_ms: float,
    max_duration_ms: float | None,
    include_prefill_phase: bool,
    event_durations: dict[str, float],
) -> list[DecodeSample]:
    rows = read_jsonl(trace_path)
    samples: list[DecodeSample] = []
    for row, next_row in zip(rows, rows[1:]):
        phase = str(row.get("phase", ""))
        if phase != "decode" and not include_prefill_phase:
            continue
        batch_id = str(row.get("batch_id", ""))
        if batch_id in event_durations:
            duration_ms = event_durations[batch_id]
        else:
            duration_ms = 1000.0 * (
                float(next_row["monotonic_time"]) - float(row["monotonic_time"])
            )
        if duration_ms < min_duration_ms:
            continue
        if max_duration_ms is not None and duration_ms > max_duration_ms:
            continue

        records = row.get("requests") or []
        if not records:
            continue
        context_lens: list[int] = []
        next_context_lens: list[int] = []
        remaining_output_tokens: list[int] = []
        for record in records:
            computed_before = int(record.get("num_computed_tokens_before") or 0)
            scheduled_tokens = int(record.get("num_scheduled_tokens") or 1)
            request_id = str(record.get("request_id", ""))
            output_len = exp_requests.get(request_id, RequestInfo(0, None)).output_len
            output_tokens = int(record.get("num_output_tokens") or 0)
            context_lens.append(computed_before)
            next_context_lens.append(computed_before + scheduled_tokens)
            remaining_output_tokens.append(max(output_len - output_tokens, 0))

        samples.append(
            DecodeSample(
                model_key=model_key,
                source_file=str(trace_path),
                batch_id=batch_id,
                batch_size=len(context_lens),
                sum_next_context_len=sum(next_context_lens),
                sum_context_len=sum(context_lens),
                max_context_len=max(context_lens),
                sum_remaining_output_tokens=sum(remaining_output_tokens),
                duration_ms=duration_ms,
            )
        )
    return samples


def fit_prefill(samples: list[PrefillSample], model_name: str):
    if not samples:
        raise ValueError("No prefill samples.")
    rows = []
    for sample in samples:
        if model_name == "3p":
            rows.append([1.0, float(sample.sum_prompt_len), float(sample.sum_prompt_len_sq)])
        elif model_name == "5p":
            rows.append([
                1.0,
                float(sample.batch_size),
                float(sample.sum_prompt_len),
                float(sample.max_prompt_len),
                float(sample.sum_prompt_len_sq),
            ])
        else:
            raise ValueError(f"Unsupported prefill model: {model_name}")
    return fit_least_squares(rows, [max(sample.duration_ms, 1e-6) for sample in samples])


def fit_decode(samples: list[DecodeSample], model_name: str):
    if not samples:
        raise ValueError("No decode samples.")
    rows = []
    for sample in samples:
        if model_name == "3d":
            rows.append([1.0, float(sample.sum_next_context_len), float(sample.batch_size)])
        elif model_name == "4d":
            rows.append([
                1.0,
                float(sample.batch_size),
                float(sample.sum_context_len),
                float(sample.max_context_len),
            ])
        elif model_name == "5d":
            rows.append([
                1.0,
                float(sample.batch_size),
                float(sample.sum_context_len),
                float(sample.max_context_len),
                float(sample.sum_remaining_output_tokens),
            ])
        else:
            raise ValueError(f"Unsupported decode model: {model_name}")
    return fit_least_squares(rows, [max(sample.duration_ms, 1e-6) for sample in samples])


def fit_least_squares(rows: list[list[float]], durations: list[float]):
    matrix = np.array(rows, dtype=float)
    target = np.array(durations, dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    predictions = matrix @ coeffs
    errors = predictions - target
    rel_errors = errors / target
    abs_rel_errors = np.abs(rel_errors)
    metrics = {
        "count": float(len(target)),
        "mean_abs_error_ms": float(np.mean(np.abs(errors))),
        "rmse_ms": float(math.sqrt(float(np.mean(errors * errors)))),
        "mean_abs_rel_error_pct": float(100.0 * np.mean(abs_rel_errors)),
        "rmse_rel_error_pct": float(
            100.0 * math.sqrt(float(np.mean(rel_errors * rel_errors)))
        ),
        "p50_abs_rel_error_pct": float(100.0 * np.percentile(abs_rel_errors, 50)),
        "p90_abs_rel_error_pct": float(100.0 * np.percentile(abs_rel_errors, 90)),
        "p99_abs_rel_error_pct": float(100.0 * np.percentile(abs_rel_errors, 99)),
        "max_abs_rel_error_pct": float(100.0 * np.max(abs_rel_errors)),
    }
    return [float(value) for value in coeffs], metrics


def write_prefill_samples_csv(path: Path, samples: list[PrefillSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "source_file",
            "batch_id",
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
                sample.batch_id,
                sample.batch_size,
                sample.sum_prompt_len,
                sample.max_prompt_len,
                sample.sum_prompt_len_sq,
                sample.duration_ms,
            ])


def write_decode_samples_csv(path: Path, samples: list[DecodeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_key",
            "source_file",
            "batch_id",
            "batch_size",
            "sum_next_context_len",
            "sum_context_len",
            "max_context_len",
            "sum_remaining_output_tokens",
            "duration_ms",
        ])
        for sample in samples:
            writer.writerow([
                sample.model_key,
                sample.source_file,
                sample.batch_id,
                sample.batch_size,
                sample.sum_next_context_len,
                sample.sum_context_len,
                sample.max_context_len,
                sample.sum_remaining_output_tokens,
                sample.duration_ms,
            ])


def update_profile(
    existing_profile: Path | None,
    prefill_fits: dict[str, tuple[list[float], dict[str, float]]],
    decode_fits: dict[str, tuple[list[float], dict[str, float]]],
    tp_world_size: int,
    decode_large_small_bs_threshold: int,
) -> dict[str, Any]:
    profile = json.loads(existing_profile.read_text()) if existing_profile and existing_profile.exists() else {}
    for model_key in sorted(set(prefill_fits) | set(decode_fits)):
        tp_profile = profile.setdefault(model_key, {}).setdefault(str(tp_world_size), {})
        if model_key in prefill_fits:
            tp_profile["prefill"] = prefill_fits[model_key][0]
        if model_key in decode_fits:
            coeffs = decode_fits[model_key][0]
            tp_profile["decoding_smallbs"] = coeffs
            tp_profile["decoding_largebs"] = coeffs
            tp_profile["decoding_large_small_bs_threshold"] = decode_large_small_bs_threshold
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit profiles from vLLM P/D scheduler traces.")
    parser.add_argument("--results-root", type=Path, default=Path("/users/rh/cdua_data"))
    parser.add_argument("--model-dir", type=Path, action="append", default=[])
    parser.add_argument("--model-key", default=None)
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=None,
        help="Root containing live *_event_trace.jsonl files, if not copied to results-root.",
    )
    parser.add_argument("--tp-world-size", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/"
            "distserve-cuda/fit_params_explicit_batch_vllm_3p3d.json"
        ),
    )
    parser.add_argument("--existing-profile", type=Path, default=None)
    parser.add_argument("--prefill-model", choices=["3p", "5p"], default="3p")
    parser.add_argument("--decode-model", choices=["3d", "4d", "5d"], default="3d")
    parser.add_argument(
        "--duration-source",
        choices=["scheduler", "forward", "model_forward"],
        default="model_forward",
        help=(
            "scheduler uses old inferred timings; forward uses full model-runner "
            "context; model_forward uses only the _model_forward call."
        ),
    )
    parser.add_argument("--prefill-min-duration-ms", type=float, default=0.01)
    parser.add_argument("--prefill-max-duration-ms", type=float, default=None)
    parser.add_argument("--decode-min-duration-ms", type=float, default=0.01)
    parser.add_argument("--decode-max-duration-ms", type=float, default=50.0)
    parser.add_argument("--decode-include-prefill-phase", action="store_true")
    parser.add_argument("--decode-large-small-bs-threshold", type=int, default=95)
    parser.add_argument("--prefill-samples-csv", type=Path, default=None)
    parser.add_argument("--decode-samples-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dirs = list(args.model_dir) if args.model_dir else discover_model_dirs(args.results_root)
    if not model_dirs:
        raise ValueError("No model directories found.")
    if args.model_key is not None and len(model_dirs) != 1:
        raise ValueError("--model-key can only be used with exactly one --model-dir.")
    decode_max_duration_ms = None if args.decode_max_duration_ms < 0 else args.decode_max_duration_ms

    prefill_samples: list[PrefillSample] = []
    decode_samples: list[DecodeSample] = []
    for model_dir in model_dirs:
        model_key = infer_model_key(model_dir, args.model_key)
        exp_requests = read_model_exp_requests(model_dir)
        prefill_trace = first_existing([
            model_dir / "prefill_batch_trace.jsonl.gz",
            model_dir / "prefill_batch_trace.jsonl",
        ])
        decode_trace = first_existing([
            model_dir / "decode_batch_trace.jsonl.gz",
            model_dir / "decode_batch_trace.jsonl",
        ])
        if prefill_trace is None or decode_trace is None:
            raise ValueError(f"Missing batch trace under {model_dir}")
        prefill_event_trace = infer_event_trace_path(
            model_dir, "prefill", args.trace_root
        )
        decode_event_trace = infer_event_trace_path(model_dir, "decode", args.trace_root)
        prefill_event_durations = load_event_durations(
            prefill_event_trace, args.duration_source
        )
        decode_event_durations = load_event_durations(
            decode_event_trace, args.duration_source
        )
        print(
            f"{model_dir.name}: duration_source={args.duration_source}, "
            f"prefill_event_trace={prefill_event_trace}, "
            f"decode_event_trace={decode_event_trace}"
        )
        model_prefill_samples = load_prefill_samples(
            prefill_trace,
            exp_requests,
            model_key,
            args.prefill_min_duration_ms,
            args.prefill_max_duration_ms,
            prefill_event_durations,
        )
        model_decode_samples = load_decode_samples(
            decode_trace,
            exp_requests,
            model_key,
            args.decode_min_duration_ms,
            decode_max_duration_ms,
            args.decode_include_prefill_phase,
            decode_event_durations,
        )
        print(
            f"{model_dir.name}: prefill_samples={len(model_prefill_samples)}, "
            f"decode_samples={len(model_decode_samples)}"
        )
        prefill_samples.extend(model_prefill_samples)
        decode_samples.extend(model_decode_samples)

    if args.prefill_samples_csv is not None:
        write_prefill_samples_csv(args.prefill_samples_csv, prefill_samples)
    if args.decode_samples_csv is not None:
        write_decode_samples_csv(args.decode_samples_csv, decode_samples)

    prefill_by_model: dict[str, list[PrefillSample]] = {}
    decode_by_model: dict[str, list[DecodeSample]] = {}
    for sample in prefill_samples:
        prefill_by_model.setdefault(sample.model_key, []).append(sample)
    for sample in decode_samples:
        decode_by_model.setdefault(sample.model_key, []).append(sample)

    prefill_fits = {
        model_key: fit_prefill(samples, args.prefill_model)
        for model_key, samples in sorted(prefill_by_model.items())
    }
    decode_fits = {
        model_key: fit_decode(samples, args.decode_model)
        for model_key, samples in sorted(decode_by_model.items())
    }

    profile = update_profile(
        args.existing_profile,
        prefill_fits,
        decode_fits,
        args.tp_world_size,
        args.decode_large_small_bs_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=4) + "\n")

    print(f"\nPrefill model: {args.prefill_model}")
    for model_key, (coeffs, metrics) in prefill_fits.items():
        print(f"  {model_key}")
        print(f"    coeffs={coeffs}")
        print(f"    metrics={metrics}")

    print(f"\nDecode model: {args.decode_model}")
    for model_key, (coeffs, metrics) in decode_fits.items():
        print(f"  {model_key}")
        print(f"    coeffs={coeffs}")
        print(f"    metrics={metrics}")

    print(f"\nWrote profile: {args.output}")


if __name__ == "__main__":
    main()

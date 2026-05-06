#!/usr/bin/env python3
"""Fit a vLLM-shaped execution profile from CUDA P/D traces.

The model is fitted on the fit split and evaluated on the val split. It writes
a simulator profile that uses explicit feature names instead of relying on the
legacy positional 3p/4d/5d formulas.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ["llama_1B", "llama_3B", "llama_8B"]
MODEL_ALIAS_TO_KEY = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}

PREFILL_FEATURES = [
    "constant",
    "batch_size",
    "sum_scheduled_tokens",
    "attention_work",
    "final_kv_blocks",
]
DECODE_FEATURES = [
    "constant",
    "batch_size",
    "sum_context_len",
    "max_context_len",
]


@dataclass(frozen=True)
class Sample:
    model: str
    batch_id: str
    phase: str
    features: dict[str, float]
    duration_ms: float


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open_text(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(paths[0])


def load_event_durations(path: Path, target: str) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    durations: dict[str, float] = {}
    begins: dict[str, dict[str, Any]] = {}
    begin_type = f"{target}_begin"
    end_type = f"{target}_end"
    for row in read_jsonl(path):
        event_type = row.get("event_type")
        batch_id = row.get("batch_id")
        if batch_id is None:
            continue
        if event_type == begin_type:
            begins[str(batch_id)] = row
        elif event_type == end_type and row.get("duration_ms") is not None:
            durations[str(batch_id)] = float(row["duration_ms"])
    return durations, begins


def batch_features(row: dict[str, Any], begin_row: dict[str, Any] | None, block_size: int) -> dict[str, float]:
    records = row.get("requests") or []
    scheduled = [int(record.get("num_scheduled_tokens") or 0) for record in records]
    computed_before = [
        int(record.get("num_computed_tokens_before") or 0) for record in records
    ]
    prompt_lens = [int(record.get("num_prompt_tokens") or 0) for record in records]

    sum_scheduled = sum(scheduled)
    max_scheduled = max(scheduled) if scheduled else 0
    context_lens = [c + q for c, q in zip(computed_before, scheduled)]
    sum_context = sum(context_lens)
    max_context = max(context_lens) if context_lens else 0
    attention_work = sum(
        q * c + q * (q + 1) / 2
        for q, c in zip(scheduled, computed_before)
    )
    final_kv_blocks = sum(
        math.ceil(prompt_len / block_size)
        for prompt_len, context_len in zip(prompt_lens, context_lens)
        if prompt_len > 0 and context_len >= prompt_len
    )
    padded_tokens = (
        int(begin_row["num_tokens_padded"])
        if begin_row and begin_row.get("num_tokens_padded") is not None
        else sum_scheduled
    )
    return {
        "constant": 1.0,
        "batch_size": float(len(records)),
        "sum_scheduled_tokens": float(sum_scheduled),
        "num_tokens_padded": float(padded_tokens),
        "max_scheduled_tokens": float(max_scheduled),
        "sum_scheduled_tokens_sq": float(sum(q * q for q in scheduled)),
        "attention_work": float(attention_work),
        "final_kv_blocks": float(final_kv_blocks),
        "final_prefill_req_count": float(
            sum(
                1
                for prompt_len, context_len in zip(prompt_lens, context_lens)
                if prompt_len > 0 and context_len >= prompt_len
            )
        ),
        "sum_context_len": float(sum_context),
        "max_context_len": float(max_context),
        "sum_next_context_len": float(sum_context),
    }


def load_samples(
    root: Path,
    split: str,
    model: str,
    side: str,
    target: str,
    block_size: int,
) -> list[Sample]:
    model_dir = root / split / model
    batch_path = first_existing(
        [
            model_dir / f"{side}_batch_trace.jsonl.gz",
            model_dir / f"{side}_batch_trace.jsonl",
        ]
    )
    event_path = first_existing(
        [
            model_dir / f"{side}_event_trace.jsonl.gz",
            model_dir / f"{side}_event_trace.jsonl",
        ]
    )
    durations, begins = load_event_durations(event_path, target)
    samples = []
    for row in read_jsonl(batch_path):
        batch_id = str(row.get("batch_id", ""))
        if batch_id not in durations:
            continue
        samples.append(
            Sample(
                model=model,
                batch_id=batch_id,
                phase=str(row.get("phase", "")),
                features=batch_features(row, begins.get(batch_id), block_size),
                duration_ms=durations[batch_id],
            )
        )
    return samples


def fit_lstsq(samples: list[Sample], features: list[str]) -> list[float]:
    matrix = np.array(
        [[sample.features[feature] for feature in features] for sample in samples],
        dtype=float,
    )
    target = np.array([sample.duration_ms for sample in samples], dtype=float)
    coeffs, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    return [float(value) for value in coeffs]


def evaluate(samples: list[Sample], features: list[str], coeffs: list[float]) -> dict[str, float]:
    if not samples:
        return {"count": 0}
    matrix = np.array(
        [[sample.features[feature] for feature in features] for sample in samples],
        dtype=float,
    )
    target = np.array([sample.duration_ms for sample in samples], dtype=float)
    pred = matrix @ np.array(coeffs, dtype=float)
    errors = pred - target
    abs_errors = np.abs(errors)
    rel = abs_errors / np.maximum(target, 1e-6)
    return {
        "count": int(len(samples)),
        "mean_abs_error_ms": float(np.mean(abs_errors)),
        "p50_abs_error_ms": float(np.percentile(abs_errors, 50)),
        "p90_abs_error_ms": float(np.percentile(abs_errors, 90)),
        "p99_abs_error_ms": float(np.percentile(abs_errors, 99)),
        "max_abs_error_ms": float(np.max(abs_errors)),
        "mean_abs_rel_error_pct": float(100.0 * np.mean(rel)),
        "p90_abs_rel_error_pct": float(100.0 * np.percentile(rel, 90)),
    }


def legacy_prefill_features() -> list[str]:
    return ["constant", "sum_scheduled_tokens", "sum_scheduled_tokens_sq"]


def legacy_decode_features() -> list[str]:
    return ["constant", "sum_next_context_len", "batch_size"]


def model_profile(prefill_coeffs: list[float], decode_coeffs: list[float], block_size: int) -> dict[str, Any]:
    decode_model = {
        "features": DECODE_FEATURES,
        "coeffs": decode_coeffs,
    }
    return {
        "1": {
            "block_size": block_size,
            "prefill": {
                "features": PREFILL_FEATURES,
                "coeffs": prefill_coeffs,
            },
            "decoding_smallbs": decode_model,
            "decoding_largebs": decode_model,
            "decoding_large_small_bs_threshold": 95,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/users/rh/cuda_data"))
    parser.add_argument(
        "--output-profile",
        type=Path,
        default=Path(
            "/users/rh/DistServe/simdistserve/estimators/profiled_data/"
            "vllm-ascend/fit_params_cuda_data_split_execution.json"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "/users/rh/cuda_data/sim/3p3d_fit_model_forward/"
            "split_execution_profile_eval/summary.json"
        ),
    )
    parser.add_argument("--prefill-target", choices=["forward", "model_forward"], default="forward")
    parser.add_argument("--decode-target", choices=["forward", "model_forward"], default="forward")
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "train_split": "fit",
        "test_split": "val",
        "prefill_target": args.prefill_target,
        "decode_target": args.decode_target,
        "block_size": args.block_size,
        "models": {},
    }

    for model in MODELS:
        fit_prefill = load_samples(
            args.data_root, "fit", model, "prefill", args.prefill_target, args.block_size
        )
        val_prefill = load_samples(
            args.data_root, "val", model, "prefill", args.prefill_target, args.block_size
        )
        fit_decode = [
            sample
            for sample in load_samples(
                args.data_root, "fit", model, "decode", args.decode_target, args.block_size
            )
            if sample.phase == "decode"
        ]
        val_decode = [
            sample
            for sample in load_samples(
                args.data_root, "val", model, "decode", args.decode_target, args.block_size
            )
            if sample.phase == "decode"
        ]

        prefill_coeffs = fit_lstsq(fit_prefill, PREFILL_FEATURES)
        decode_coeffs = fit_lstsq(fit_decode, DECODE_FEATURES)
        legacy_prefill_coeffs = fit_lstsq(fit_prefill, legacy_prefill_features())
        legacy_decode_coeffs = fit_lstsq(fit_decode, legacy_decode_features())

        model_key = MODEL_ALIAS_TO_KEY[model]
        profile[model_key] = model_profile(prefill_coeffs, decode_coeffs, args.block_size)
        summary["models"][model] = {
            "new": {
                "prefill_features": PREFILL_FEATURES,
                "prefill_coeffs": prefill_coeffs,
                "decode_features": DECODE_FEATURES,
                "decode_coeffs": decode_coeffs,
                "fit_prefill": evaluate(fit_prefill, PREFILL_FEATURES, prefill_coeffs),
                "val_prefill": evaluate(val_prefill, PREFILL_FEATURES, prefill_coeffs),
                "fit_decode": evaluate(fit_decode, DECODE_FEATURES, decode_coeffs),
                "val_decode": evaluate(val_decode, DECODE_FEATURES, decode_coeffs),
            },
            "legacy_shape": {
                "prefill_features": legacy_prefill_features(),
                "prefill_coeffs": legacy_prefill_coeffs,
                "decode_features": legacy_decode_features(),
                "decode_coeffs": legacy_decode_coeffs,
                "fit_prefill": evaluate(
                    fit_prefill, legacy_prefill_features(), legacy_prefill_coeffs
                ),
                "val_prefill": evaluate(
                    val_prefill, legacy_prefill_features(), legacy_prefill_coeffs
                ),
                "fit_decode": evaluate(
                    fit_decode, legacy_decode_features(), legacy_decode_coeffs
                ),
                "val_decode": evaluate(
                    val_decode, legacy_decode_features(), legacy_decode_coeffs
                ),
            },
        }

    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(json.dumps(profile, indent=4) + "\n", encoding="utf-8")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote profile: {args.output_profile}")
    print(f"Wrote summary: {args.summary_output}")


if __name__ == "__main__":
    main()

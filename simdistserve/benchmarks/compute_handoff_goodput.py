#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from simdistserve.benchmarks.search_hetero import MODEL_ALIASES, _normalize_model


DTYPE_BYTES = {
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float32": 4,
    "fp32": 4,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_model_path(model: str) -> Path:
    normalized = _normalize_model(model)
    if normalized in MODEL_ALIASES:
        normalized = MODEL_ALIASES[normalized]
    return Path(normalized)


def _load_model_kv_config(model: str, dtype_override: str | None = None) -> dict[str, Any]:
    model_path = _resolve_model_path(model)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    config = _load_json(config_path)
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_layers = int(config["num_hidden_layers"])
    num_kv_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = int(config.get("head_dim") or (hidden_size // num_attention_heads))
    dtype = dtype_override or str(config.get("torch_dtype", "float16"))
    dtype_bytes = DTYPE_BYTES.get(dtype.lower())
    if dtype_bytes is None:
        raise ValueError(f"Unsupported dtype {dtype!r}; pass --dtype explicitly.")

    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "num_hidden_layers": num_layers,
        "head_dim": head_dim,
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "kv_bytes_per_token": num_layers * num_kv_heads * head_dim * 2 * dtype_bytes,
    }


def _load_prompt_lengths(workload: Path, field: str) -> list[int]:
    lengths: list[int] = []
    with workload.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if field not in item:
                raise KeyError(f"{workload}:{line_no} missing field {field!r}")
            lengths.append(int(item[field]))
    if not lengths:
        raise ValueError(f"No requests found in workload: {workload}")
    return lengths


def _percentile(values: list[int], pct: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _length_stat(lengths: list[int], aggregation: str) -> float:
    if aggregation == "mean":
        return statistics.fmean(lengths)
    if aggregation == "median":
        return statistics.median(lengths)
    if aggregation.startswith("p"):
        return _percentile(lengths, float(aggregation[1:]))
    if aggregation == "max":
        return float(max(lengths))
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def _load_bandwidth(
    summary_csv: Path,
    direction_flag: str,
    parallel_streams: int,
    bandwidth_column: str,
) -> dict[str, Any]:
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [
        row for row in rows
        if row["direction_flag"] == direction_flag and int(row["parallel_streams"]) == parallel_streams
    ]
    if not matches:
        raise ValueError(
            f"No iperf row for direction={direction_flag}, parallel_streams={parallel_streams}"
        )
    row = matches[0]
    return {
        "direction_flag": direction_flag,
        "transfer_direction": row.get("transfer_direction"),
        "parallel_streams": parallel_streams,
        "bandwidth_column": bandwidth_column,
        "bits_per_second": float(row[bandwidth_column]),
        "mbits_per_second": float(row[bandwidth_column]) / 1e6,
        "source_row": row,
    }


def compute_handoff_goodput(
    model: str,
    workload: Path,
    network_summary_csv: Path,
    prompt_len_field: str,
    length_aggregation: str,
    dtype: str | None,
    parallel_streams: int,
    bandwidth_column: str,
    cuda_to_ascend_direction: str,
    ascend_to_cuda_direction: str,
) -> dict[str, Any]:
    model_config = _load_model_kv_config(model, dtype)
    prompt_lens = _load_prompt_lengths(workload, prompt_len_field)
    effective_prompt_len = _length_stat(prompt_lens, length_aggregation)
    avg_payload_bytes = effective_prompt_len * model_config["kv_bytes_per_token"]
    avg_payload_bits = avg_payload_bytes * 8

    cuda_to_ascend_bw = _load_bandwidth(
        network_summary_csv, cuda_to_ascend_direction, parallel_streams, bandwidth_column
    )
    ascend_to_cuda_bw = _load_bandwidth(
        network_summary_csv, ascend_to_cuda_direction, parallel_streams, bandwidth_column
    )

    cuda_to_ascend_goodput = cuda_to_ascend_bw["bits_per_second"] / avg_payload_bits
    ascend_to_cuda_goodput = ascend_to_cuda_bw["bits_per_second"] / avg_payload_bits
    kv_bits_per_token = model_config["kv_bytes_per_token"] * 8
    cuda_to_ascend_delay_per_token_ms = (
        kv_bits_per_token / cuda_to_ascend_bw["bits_per_second"] * 1000.0
    )
    ascend_to_cuda_delay_per_token_ms = (
        kv_bits_per_token / ascend_to_cuda_bw["bits_per_second"] * 1000.0
    )

    return {
        "model": model,
        "workload": str(workload),
        "network_summary_csv": str(network_summary_csv),
        "prompt_len_field": prompt_len_field,
        "length_aggregation": length_aggregation,
        "num_requests": len(prompt_lens),
        "prompt_len_stats": {
            "mean": statistics.fmean(prompt_lens),
            "median": statistics.median(prompt_lens),
            "p90": _percentile(prompt_lens, 90),
            "p95": _percentile(prompt_lens, 95),
            "max": max(prompt_lens),
            "effective": effective_prompt_len,
        },
        "model_config": model_config,
        "payload": {
            "kv_bytes_per_token": model_config["kv_bytes_per_token"],
            "effective_payload_bytes_per_request": avg_payload_bytes,
            "effective_payload_mib_per_request": avg_payload_bytes / (1024 * 1024),
            "effective_payload_bits_per_request": avg_payload_bits,
        },
        "bandwidth": {
            "cuda_to_ascend": cuda_to_ascend_bw,
            "ascend_to_cuda": ascend_to_cuda_bw,
        },
        "handoff": {
            "cuda_to_ascend": {
                "handoff_goodput_upper_bound": cuda_to_ascend_goodput,
                "fixed_delay_ms": 0.0,
                "delay_per_token_ms": cuda_to_ascend_delay_per_token_ms,
                "effective_payload_delay_ms": (
                    cuda_to_ascend_delay_per_token_ms * effective_prompt_len
                ),
            },
            "ascend_to_cuda": {
                "handoff_goodput_upper_bound": ascend_to_cuda_goodput,
                "fixed_delay_ms": 0.0,
                "delay_per_token_ms": ascend_to_cuda_delay_per_token_ms,
                "effective_payload_delay_ms": (
                    ascend_to_cuda_delay_per_token_ms * effective_prompt_len
                ),
            },
        },
    }


def _update_search_config(config_path: Path, handoff: dict[str, Any]) -> None:
    payload = _load_json(config_path)
    payload["handoff"] = {
        "cuda_to_ascend": {
            "handoff_goodput_upper_bound": handoff["cuda_to_ascend"]["handoff_goodput_upper_bound"],
            "fixed_delay_ms": handoff["cuda_to_ascend"]["fixed_delay_ms"],
            "delay_per_token_ms": handoff["cuda_to_ascend"]["delay_per_token_ms"],
        },
        "ascend_to_cuda": {
            "handoff_goodput_upper_bound": handoff["ascend_to_cuda"]["handoff_goodput_upper_bound"],
            "fixed_delay_ms": handoff["ascend_to_cuda"]["fixed_delay_ms"],
            "delay_per_token_ms": handoff["ascend_to_cuda"]["delay_per_token_ms"],
        },
    }
    config_path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute shape-independent handoff goodput from workload and model config.")
    parser.add_argument("--model", required=True, help="Model alias or local model path.")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--network-summary-csv",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/benchmarks/results/network/iperf3/20260423T090835Z/summary.csv"),
    )
    parser.add_argument("--prompt-len-field", default="prompt_len")
    parser.add_argument(
        "--length-aggregation",
        choices=["mean", "median", "p90", "p95", "max"],
        default="mean",
        help="Request length statistic used to convert bandwidth to req/s.",
    )
    parser.add_argument("--dtype", default=None, help="Override KV dtype. Defaults to model config torch_dtype.")
    parser.add_argument("--parallel-streams", type=int, default=8)
    parser.add_argument("--bandwidth-column", default="mean_receiver_bits_per_second")
    parser.add_argument(
        "--cuda-to-ascend-direction",
        choices=["forward", "reverse"],
        default="reverse",
        help="iperf direction for CUDA -> Ascend. Defaults to reverse for the recorded test topology.",
    )
    parser.add_argument(
        "--ascend-to-cuda-direction",
        choices=["forward", "reverse"],
        default="forward",
        help="iperf direction for Ascend -> CUDA. Defaults to forward for the recorded test topology.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--update-config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_handoff_goodput(
        model=args.model,
        workload=args.workload,
        network_summary_csv=args.network_summary_csv,
        prompt_len_field=args.prompt_len_field,
        length_aggregation=args.length_aggregation,
        dtype=args.dtype,
        parallel_streams=args.parallel_streams,
        bandwidth_column=args.bandwidth_column,
        cuda_to_ascend_direction=args.cuda_to_ascend_direction,
        ascend_to_cuda_direction=args.ascend_to_cuda_direction,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.update_config is not None:
        _update_search_config(args.update_config, result["handoff"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

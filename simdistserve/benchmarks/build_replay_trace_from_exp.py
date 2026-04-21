#!/usr/bin/env python3
"""Build a replayable simulator workload from an ordered dataset split and a real benchmark .exp.

The input dataset split must preserve request order. This script aligns each
request with the corresponding benchmark result by index, validates prompt/output
lengths, then emits a JSONL file compatible with `simulate_dist.py --arrival custom`.

Each output record contains:
  - prompt / prompt_len / output_len
  - start_time relative to the first issued request
  - optional benchmark metadata for later analysis
"""

from __future__ import annotations

import argparse
import json
import marshal
from pathlib import Path
from typing import Any


def load_dataset_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(
                    {
                        "prompt": payload["prompt"],
                        "prompt_len": int(payload["prompt_len"]),
                        "output_len": int(payload["output_len"]),
                    }
                )
        return records

    if path.suffix == ".ds":
        with path.open("rb") as f:
            payload = marshal.load(f)
        reqs = payload["reqs"]
        return [
            {
                "prompt": prompt,
                "prompt_len": int(prompt_len),
                "output_len": int(output_len),
            }
            for prompt, prompt_len, output_len in reqs
        ]

    raise ValueError(f"Unsupported dataset format: {path}. Expected .ds or .jsonl")


def load_exp_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected {path} to contain a list of benchmark request results.")
    return payload


def build_records_for_one_exp(
    dataset_records: list[dict[str, Any]],
    exp_path: Path,
    omit_benchmark_metadata: bool,
) -> list[dict[str, Any]]:
    exp_results = load_exp_results(exp_path)
    if len(dataset_records) != len(exp_results):
        raise ValueError(
            f"Dataset length mismatch: {len(dataset_records)} requests in dataset but "
            f"{len(exp_results)} benchmark results in {exp_path}."
        )

    enriched_records = []
    for index, (dataset_record, exp_result) in enumerate(zip(dataset_records, exp_results)):
        prompt_len = int(exp_result["prompt_len"])
        output_len = int(exp_result["output_len"])
        if prompt_len != dataset_record["prompt_len"] or output_len != dataset_record["output_len"]:
            raise ValueError(
                "Request alignment mismatch at index "
                f"{index} for {exp_path}: dataset has "
                f"(prompt_len={dataset_record['prompt_len']}, output_len={dataset_record['output_len']}) "
                f"but exp has (prompt_len={prompt_len}, output_len={output_len})."
            )

        token_timestamps = [float(ts) for ts in exp_result.get("token_timestamps", [])]
        start_time = float(exp_result["start_time"])
        end_time = float(exp_result["end_time"])
        record = {
            "source_index": index,
            "prompt": dataset_record["prompt"],
            "prompt_len": prompt_len,
            "output_len": output_len,
            "start_time": start_time,
        }
        if not omit_benchmark_metadata:
            record.update(
                {
                    "benchmark_end_time": end_time,
                    "benchmark_latency_sec": end_time - start_time,
                    "benchmark_ttft_sec": (
                        token_timestamps[0] - start_time if token_timestamps else None
                    ),
                    "benchmark_tpot_sec": (
                        0.0
                        if output_len <= 1 or len(token_timestamps) < 2
                        else (token_timestamps[-1] - token_timestamps[0]) / (output_len - 1)
                    ),
                    "benchmark_token_timestamps_rel": [
                        ts - start_time for ts in token_timestamps
                    ],
                }
            )
        enriched_records.append(record)

    enriched_records.sort(key=lambda item: (float(item["start_time"]), int(item["source_index"])))
    first_start_time = float(enriched_records[0]["start_time"]) if enriched_records else 0.0
    for replay_index, record in enumerate(enriched_records):
        record["req_id"] = replay_index
        record["start_time"] = float(record["start_time"]) - first_start_time
    return enriched_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the ordered dataset split (.ds or .jsonl).",
    )
    parser.add_argument(
        "--exp",
        type=Path,
        required=True,
        help="Path to the real benchmark .exp file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path for simulator replay.",
    )
    parser.add_argument(
        "--omit-benchmark-metadata",
        action="store_true",
        help="Omit benchmark latency/TTFT/TPOT/timestamp metadata from each output record.",
    )
    args = parser.parse_args()

    dataset_records = load_dataset_records(args.dataset)
    if args.exp.is_dir():
        exp_files = sorted(path for path in args.exp.rglob("*.exp") if path.is_file())
        if not exp_files:
            raise FileNotFoundError(f"No .exp files found under {args.exp}")
        args.output.mkdir(parents=True, exist_ok=True)
        outputs = []
        for exp_path in exp_files:
            records = build_records_for_one_exp(
                dataset_records=dataset_records,
                exp_path=exp_path,
                omit_benchmark_metadata=args.omit_benchmark_metadata,
            )
            output_path = args.output / f"{exp_path.stem}.jsonl"
            write_jsonl(output_path, records)
            outputs.append(
                {
                    "exp": str(exp_path),
                    "output": str(output_path),
                    "num_requests": len(records),
                }
            )
        metadata = {
            "dataset": str(args.dataset),
            "exp_root": str(args.exp),
            "output_dir": str(args.output),
            "num_traces": len(outputs),
            "time_origin": "first_request_start_time",
            "sorted_by": ["start_time", "source_index"],
            "included_benchmark_metadata": not bool(args.omit_benchmark_metadata),
            "traces": outputs,
        }
        print(json.dumps(metadata, indent=2))
        return

    records = build_records_for_one_exp(
        dataset_records=dataset_records,
        exp_path=args.exp,
        omit_benchmark_metadata=args.omit_benchmark_metadata,
    )
    write_jsonl(args.output, records)
    metadata = {
        "dataset": str(args.dataset),
        "exp": str(args.exp),
        "output": str(args.output),
        "num_requests": len(records),
        "time_origin": "first_request_start_time",
        "sorted_by": ["start_time", "source_index"],
        "included_benchmark_metadata": not bool(args.omit_benchmark_metadata),
    }
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

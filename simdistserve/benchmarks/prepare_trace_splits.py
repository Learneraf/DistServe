#!/usr/bin/env python3
"""Prepare canonical fit/val/test request splits from a preprocessed dataset.

The script consumes the existing marshal-backed `.ds` dataset format and emits:

- ordered `.ds` files for the real benchmark path
- ordered `.jsonl` files for the simulator custom workload path
- a metadata JSON file describing the split composition

The split is approximately stratified by prompt/output length buckets.
"""

from __future__ import annotations

import argparse
import json
import marshal
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_split_sizes(raw: str) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid split spec {part!r}. Expected comma-separated name:size entries."
            )
        name, size = part.split(":", 1)
        items.append((name.strip(), int(size.strip())))
    if not items:
        raise ValueError("At least one split must be provided.")
    return items


def load_dataset(path: Path) -> tuple[str, list[tuple[str, int, int]]]:
    with path.open("rb") as f:
        payload = marshal.load(f)
    return payload["dataset_name"], list(payload["reqs"])


def dump_dataset(path: Path, dataset_name: str, reqs: list[tuple[str, int, int]]) -> None:
    with path.open("wb") as f:
        marshal.dump({"dataset_name": dataset_name, "reqs": reqs}, f)


def prompt_bucket(prompt_len: int) -> str:
    if prompt_len <= 256:
        return "short"
    if prompt_len <= 1024:
        return "medium"
    return "long"


def output_bucket(output_len: int) -> str:
    if output_len <= 64:
        return "short"
    if output_len <= 256:
        return "medium"
    return "long"


def bucket_key(req: tuple[str, int, int]) -> tuple[str, str]:
    _, prompt_len, output_len = req
    return prompt_bucket(prompt_len), output_bucket(output_len)


def allocate_counts(total: int, weights: list[int]) -> list[int]:
    weight_sum = sum(weights)
    if total < 0 or weight_sum <= 0:
        raise ValueError("Invalid allocation request.")
    exact = [total * weight / weight_sum for weight in weights]
    base = [int(x) for x in exact]
    remainder = total - sum(base)
    order = sorted(
        range(len(weights)),
        key=lambda i: (exact[i] - base[i], weights[i]),
        reverse=True,
    )
    for i in order[:remainder]:
        base[i] += 1
    return base


def summarize_buckets(reqs: Iterable[tuple[str, int, int]]) -> dict[str, int]:
    counter = Counter(f"{p}/{o}" for p, o in (bucket_key(req) for req in reqs))
    return dict(sorted(counter.items()))


def load_tokenizer(tokenizer_name: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=trust_remote_code,
    )


def normalize_and_filter_reqs(
    reqs: list[tuple[str, int, int]],
    tokenizer_name: str | None,
    trust_remote_code: bool,
    max_input_len: int | None,
    max_total_len: int | None,
) -> tuple[list[tuple[str, int, int]], dict[str, object]]:
    tokenizer = None
    if tokenizer_name is not None:
        tokenizer = load_tokenizer(tokenizer_name, trust_remote_code)

    filtered_reqs: list[tuple[str, int, int]] = []
    dropped_counts = {
        "input_too_long": 0,
        "total_too_long": 0,
    }
    prompt_len_deltas: list[int] = []

    for prompt, stored_prompt_len, output_len in reqs:
        prompt_len = stored_prompt_len
        if tokenizer is not None:
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompt_len_deltas.append(prompt_len - stored_prompt_len)

        if max_input_len is not None and prompt_len > max_input_len:
            dropped_counts["input_too_long"] += 1
            continue

        if max_total_len is not None and prompt_len + output_len > max_total_len:
            dropped_counts["total_too_long"] += 1
            continue

        filtered_reqs.append((prompt, prompt_len, output_len))

    prompt_len_stats = None
    if prompt_len_deltas:
        prompt_len_stats = {
            "mean_delta": sum(prompt_len_deltas) / len(prompt_len_deltas),
            "min_delta": min(prompt_len_deltas),
            "max_delta": max(prompt_len_deltas),
        }

    metadata = {
        "tokenizer": tokenizer_name,
        "retokenized_prompt_lengths": tokenizer is not None,
        "max_input_len": max_input_len,
        "max_total_len": max_total_len,
        "dropped_counts": dropped_counts,
        "kept_requests": len(filtered_reqs),
        "original_requests": len(reqs),
        "prompt_len_delta_stats": prompt_len_stats,
    }
    return filtered_reqs, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to preprocessed .ds dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write fit/val/test split artifacts into",
    )
    parser.add_argument(
        "--split-sizes",
        type=str,
        default="fit:100,val:100,test:100",
        help="Comma-separated split sizes, e.g. fit:100,val:100,test:100",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Optional tokenizer/model name used to recompute prompt lengths before splitting.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading --tokenizer.",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=None,
        help="Optional maximum prompt token length after retokenization/filtering.",
    )
    parser.add_argument(
        "--max-total-len",
        type=int,
        default=None,
        help="Optional maximum prompt_len + output_len after retokenization/filtering.",
    )
    args = parser.parse_args()

    split_specs = parse_split_sizes(args.split_sizes)
    split_names = [name for name, _ in split_specs]
    split_sizes = [size for _, size in split_specs]
    total_requested = sum(split_sizes)

    dataset_name, original_reqs = load_dataset(args.dataset)
    reqs, filter_metadata = normalize_and_filter_reqs(
        reqs=original_reqs,
        tokenizer_name=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        max_input_len=args.max_input_len,
        max_total_len=args.max_total_len,
    )
    if total_requested > len(reqs):
        raise ValueError(
            f"Requested {total_requested} requests across splits, but dataset only has {len(reqs)} entries."
        )

    rng = random.Random(args.seed)

    bucket_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, req in enumerate(reqs):
        bucket_to_indices[bucket_key(req)].append(idx)
    for indices in bucket_to_indices.values():
        rng.shuffle(indices)

    bucket_keys = sorted(bucket_to_indices)
    bucket_sizes = [len(bucket_to_indices[key]) for key in bucket_keys]
    selected_counts = allocate_counts(total_requested, bucket_sizes)

    selected_by_bucket: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
    for key, count in zip(bucket_keys, selected_counts):
        indices = bucket_to_indices[key][:count]
        selected_by_bucket[key] = [reqs[idx] for idx in indices]

    split_to_reqs: dict[str, list[tuple[str, int, int]]] = {name: [] for name in split_names}
    split_weights = split_sizes
    for key in bucket_keys:
        bucket_reqs = selected_by_bucket[key]
        bucket_counts = allocate_counts(len(bucket_reqs), split_weights)
        offset = 0
        for split_name, count in zip(split_names, bucket_counts):
            split_to_reqs[split_name].extend(bucket_reqs[offset:offset + count])
            offset += count

    actual_sizes = {name: len(split_to_reqs[name]) for name in split_names}
    deficits = {name: target - actual_sizes[name] for name, target in split_specs}
    while True:
        need = next((name for name in split_names if deficits[name] > 0), None)
        give = next((name for name in split_names if deficits[name] < 0), None)
        if need is None or give is None:
            break
        split_to_reqs[need].append(split_to_reqs[give].pop())
        deficits[need] -= 1
        deficits[give] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_dataset": str(args.dataset),
        "dataset_name": dataset_name,
        "seed": args.seed,
        "requested_split_sizes": dict(split_specs),
        "filtering": filter_metadata,
        "source_bucket_counts": summarize_buckets(original_reqs),
        "filtered_bucket_counts": summarize_buckets(reqs),
        "selected_bucket_counts": summarize_buckets(
            req for split_reqs in selected_by_bucket.values() for req in split_reqs
        ),
        "splits": {},
    }

    for split_index, split_name in enumerate(split_names):
        split_rng = random.Random(args.seed + split_index + 1)
        split_rng.shuffle(split_to_reqs[split_name])
        split_dataset_name = f"{dataset_name}-{split_name}"
        ds_path = args.output_dir / f"{split_name}.ds"
        jsonl_path = args.output_dir / f"{split_name}.jsonl"
        dump_dataset(ds_path, split_dataset_name, split_to_reqs[split_name])
        with jsonl_path.open("w", encoding="utf-8") as f:
            for req_id, (prompt, prompt_len, output_len) in enumerate(split_to_reqs[split_name]):
                record = {
                    "req_id": req_id,
                    "prompt": prompt,
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        metadata["splits"][split_name] = {
            "size": len(split_to_reqs[split_name]),
            "dataset_name": split_dataset_name,
            "ds_path": str(ds_path),
            "jsonl_path": str(jsonl_path),
            "bucket_counts": summarize_buckets(split_to_reqs[split_name]),
        }

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

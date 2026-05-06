#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from simdistserve.benchmarks.compute_handoff_goodput import compute_handoff_goodput
from simdistserve.benchmarks.search_hetero import MODEL_ALIASES, _normalize_model


MODEL_DATASETS = {
    "llama_1B": "llama-3.2-1B",
    "llama_3B": "llama-3.2-3B",
    "llama_7B": "llama-2-7b",
    "llama_8B": "llama-3.1-8B",
}


def _load_raw_sharegpt(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected raw ShareGPT JSON array: {path}")
    return payload


def _first_user_assistant_pair(item: dict[str, Any]) -> tuple[str, str] | None:
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None
    for idx, turn in enumerate(conversations[:-1]):
        next_turn = conversations[idx + 1]
        if (
            isinstance(turn, dict)
            and isinstance(next_turn, dict)
            and turn.get("from") == "human"
            and next_turn.get("from") == "gpt"
            and isinstance(turn.get("value"), str)
            and isinstance(next_turn.get("value"), str)
        ):
            prompt = turn["value"].strip()
            output = next_turn["value"].strip()
            if prompt and output:
                return prompt, output
    return None


def _build_workload(
    raw_items: list[dict[str, Any]],
    model: str,
    num_requests: int,
    seed: int,
    min_prompt_len: int,
    min_output_len: int,
    max_prompt_len: int,
    max_output_len: int,
    output_path: Path,
) -> dict[str, Any]:
    model_path = Path(_normalize_model(model))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rng = random.Random(seed)
    indices = list(range(len(raw_items)))
    rng.shuffle(indices)

    rows: list[dict[str, Any]] = []
    for source_index in indices:
        pair = _first_user_assistant_pair(raw_items[source_index])
        if pair is None:
            continue
        prompt, output = pair
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        output_len = len(tokenizer.encode(output, add_special_tokens=False))
        if prompt_len <= 0 or output_len <= 0:
            continue
        if prompt_len < min_prompt_len or output_len < min_output_len:
            continue
        if prompt_len > max_prompt_len or output_len > max_output_len:
            continue
        rows.append(
            {
                "source_index": source_index,
                "prompt_len": prompt_len,
                "output_len": output_len,
            }
        )
        if len(rows) >= num_requests:
            break

    if len(rows) < num_requests:
        raise RuntimeError(
            f"Only found {len(rows)} valid requests for {model}; requested {num_requests}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_lens = [row["prompt_len"] for row in rows]
    output_lens = [row["output_len"] for row in rows]
    return {
        "model": model,
        "model_path": str(model_path),
        "output": str(output_path),
        "num_requests": len(rows),
        "filters": {
            "min_prompt_len": min_prompt_len,
            "min_output_len": min_output_len,
            "max_prompt_len": max_prompt_len,
            "max_output_len": max_output_len,
        },
        "prompt_len": {
            "min": min(prompt_lens),
            "mean": sum(prompt_lens) / len(prompt_lens),
            "max": max(prompt_lens),
        },
        "output_len": {
            "min": min(output_lens),
            "mean": sum(output_lens) / len(output_lens),
            "max": max(output_lens),
        },
    }


def _update_config(config_path: Path, workload_path: Path, handoff: dict[str, Any], num_requests: int) -> None:
    payload = json.loads(config_path.read_text())
    payload["workload"] = str(workload_path)
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
    search = payload.setdefault("search", {})
    search.setdefault("profile_num_requests", min(num_requests, 32))
    search.setdefault("profile_min_profile_duration_s", 1.0)
    search.setdefault("profile_max_rate_cap", 256.0)
    search.setdefault("capped_mu_policy", "keep")
    config_path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hetero profiling workloads and update configs.")
    parser.add_argument(
        "--raw-sharegpt",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/dataset/raw/ShareGPT_V3_unfiltered_cleaned_split.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/hetero/examples/workloads"),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/hetero/examples/configs"),
    )
    parser.add_argument(
        "--handoff-output-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/hetero/results/handoff/profile_workloads"),
    )
    parser.add_argument(
        "--network-summary-csv",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/benchmarks/results/network/iperf3/20260423T090835Z/summary.csv"),
    )
    parser.add_argument("--num-requests", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-prompt-len", type=int, default=1)
    parser.add_argument("--min-output-len", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=2048)
    parser.add_argument("--parallel-streams", type=int, default=8)
    parser.add_argument("--length-aggregation", default="mean", choices=["mean", "median", "p90", "p95", "max"])
    parser.add_argument("--models", nargs="+", default=["llama_1B", "llama_3B", "llama_7B", "llama_8B"])
    parser.add_argument(
        "--update-example-config",
        action="store_true",
        help="Also update example_search_config.json. By default only the measured-bandwidth search config is updated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_items = _load_raw_sharegpt(args.raw_sharegpt)
    summary: dict[str, Any] = {"raw_sharegpt": str(args.raw_sharegpt), "models": {}}

    for model in args.models:
        dataset_name = MODEL_DATASETS[model]
        filter_tag = (
            f"minp{args.min_prompt_len}_mino{args.min_output_len}"
            f"_maxp{args.max_prompt_len}_maxo{args.max_output_len}"
        )
        workload_path = (
            args.output_root
            / dataset_name
            / f"profile_{args.num_requests}_seed{args.seed}_{filter_tag}.jsonl"
        )
        workload_summary = _build_workload(
            raw_items=raw_items,
            model=model,
            num_requests=args.num_requests,
            seed=args.seed,
            min_prompt_len=args.min_prompt_len,
            min_output_len=args.min_output_len,
            max_prompt_len=args.max_prompt_len,
            max_output_len=args.max_output_len,
            output_path=workload_path,
        )
        handoff_result = compute_handoff_goodput(
            model=model,
            workload=workload_path,
            network_summary_csv=args.network_summary_csv,
            prompt_len_field="prompt_len",
            length_aggregation=args.length_aggregation,
            dtype=None,
            parallel_streams=args.parallel_streams,
            bandwidth_column="mean_receiver_bits_per_second",
            cuda_to_ascend_direction="reverse",
            ascend_to_cuda_direction="forward",
        )
        handoff_path = args.handoff_output_root / model / "handoff_goodput.json"
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.write_text(json.dumps(handoff_result, indent=2) + "\n")

        config_path = args.config_root / model / "search_4nodes_high_affinity.json"
        _update_config(config_path, workload_path, handoff_result["handoff"], args.num_requests)

        example_config_path = args.config_root / model / "example_search_config.json"
        if args.update_example_config and example_config_path.exists():
            _update_config(example_config_path, workload_path, handoff_result["handoff"], args.num_requests)

        summary["models"][model] = {
            "workload": workload_summary,
            "handoff": handoff_result["handoff"],
            "handoff_output": str(handoff_path),
            "updated_config": str(config_path),
        }

    summary_path = (
        args.output_root
        / (
            f"profile_{args.num_requests}_seed{args.seed}"
            f"_minp{args.min_prompt_len}_mino{args.min_output_len}"
            f"_maxp{args.max_prompt_len}_maxo{args.max_output_len}_summary.json"
        )
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), "models": summary["models"]}, indent=2))


if __name__ == "__main__":
    main()

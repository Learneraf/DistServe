#!/usr/bin/env python3
"""Two-phase TP profiler driven by a dataset arrival trace.

This script is for the "two GPUs, TP=2 component profile" case:

1. Use all visible GPUs for a prefill TP group and replay request arrivals from
   a dataset. The script records each request's prefill completion time.
2. Release the prefill workers, create a decode TP group on the same GPUs, and
   replay decode arrivals at the recorded prefill completion times.

The measurement is intentionally component-level, not end-to-end DistServe
serving. It avoids needing four GPUs for simultaneous prefill_tp=2 and
decode_tp=2 instances.

python \
  /users/rh/DistServe/evaluation/0-test-single-forward-performance/run_dataset_arrival_profile.py \
  --phase both \
  --model /users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2 \
  --dataset /users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.2-3B/fit.ds \
  --tp-world-size 2 \
  --request-rates 1,1.5,2,2.5,3,3.5,4 \
  --num-prompts 120 \
  --output-root /users/rh/DistServe/evaluation/0-test-single-forward-performance/result/llama3B

"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import marshal
import random
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import ray
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from distserve.config import CacheConfig, ModelConfig, ParallelConfig
from sut.abstract_sut import get_input_ids
from sut.sut_distserve import BLOCK_SIZE, Worker


PREFILL_TRACE_NAME = "tp2_prefill_trace.jsonl"
DECODE_ROUNDS_NAME = "tp2_decode_rounds.jsonl"
REQUEST_RESULTS_NAME = "tp2_two_phase.exp"
SUMMARY_NAME = "tp2_two_phase_summary.json"


@dataclasses.dataclass(frozen=True)
class DatasetRequest:
    req_id: int
    prompt: str
    prompt_len: int
    output_len: int
    arrival_time: float


@dataclasses.dataclass
class PrefillTraceRecord:
    req_id: int
    prompt_len: int
    output_len: int
    arrival_time: float
    prefill_start_time: float
    prefill_end_time: float
    prefill_latency_ms: float
    batch_id: int
    batch_size: int
    batch_prompt_lens: list[int]


@dataclasses.dataclass
class DecodeRequestState:
    req_id: int
    prompt_len: int
    output_len: int
    arrival_time: float
    prefill_start_time: float
    prefill_end_time: float
    generated_tokens: int
    last_token_id: int
    token_timestamps: list[float]

    @property
    def remaining_tokens(self) -> int:
        return max(self.output_len - self.generated_tokens, 0)

    @property
    def current_context_len(self) -> int:
        return self.prompt_len + self.generated_tokens


@dataclasses.dataclass
class DecodeRoundRecord:
    round_id: int
    decode_start_time: float
    decode_end_time: float
    forward_time_ms: float
    batch_size: int
    req_ids: list[int]
    input_lens: list[int]
    output_lens: list[int]
    current_context_lens: list[int]
    remaining_output_tokens: list[int]


@dataclasses.dataclass
class RequestResultRecord:
    prompt_len: int
    output_len: int
    start_time: float
    end_time: float
    token_timestamps: list[float]
    lifecycle_events: list[dict[str, Any]]
    latency: float
    ftl: float
    tpot: float


def load_dataset(path: Path, request_rate: float, process_name: str, cv: float, seed: int) -> list[DatasetRequest]:
    raw_requests = _load_raw_requests(path)
    intervals = make_interarrival_intervals(len(raw_requests), request_rate, process_name, cv, seed)
    arrival_time = 0.0
    requests: list[DatasetRequest] = []
    for req_id, (prompt, prompt_len, output_len) in enumerate(raw_requests):
        requests.append(
            DatasetRequest(
                req_id=req_id,
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                arrival_time=arrival_time,
            )
        )
        arrival_time += float(intervals[req_id])
    return requests


def _load_raw_requests(path: Path) -> list[tuple[str, int, int]]:
    try:
        with path.open("rb") as f:
            payload = marshal.load(f)
        return [
            (str(prompt), int(prompt_len), int(output_len))
            for prompt, prompt_len, output_len in payload["reqs"]
        ]
    except (EOFError, KeyError, TypeError, ValueError):
        pass

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Dataset file is empty: {path}")
    if raw_text[0] == "[":
        items = json.loads(raw_text)
    else:
        items = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    result: list[tuple[str, int, int]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{idx} expected object, got {type(item).__name__}")
        prompt = item.get("prompt")
        prompt_len = item.get("prompt_len")
        output_len = item.get("output_len", item.get("output_tokens"))
        if prompt is None or prompt_len is None or output_len is None:
            raise ValueError(f"{path}:{idx} requires prompt, prompt_len, and output_len/output_tokens")
        result.append((str(prompt), int(prompt_len), int(output_len)))
    return result


def make_interarrival_intervals(
    num_requests: int,
    request_rate: float,
    process_name: str,
    cv: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if request_rate in (0.0, float("inf")):
        return np.zeros(num_requests)
    if process_name == "uniform":
        return np.full(num_requests, 1.0 / request_rate)
    if process_name in ("poisson", "possion"):
        return rng.gamma(shape=1.0, scale=1.0 / request_rate, size=num_requests)
    if process_name == "gamma":
        shape = 1.0 / (cv * cv)
        scale = cv * cv / request_rate
        return rng.gamma(shape=shape, scale=scale, size=num_requests)
    raise ValueError(f"Unsupported process_name: {process_name}")


def maybe_sample_requests(requests: list[DatasetRequest], num_prompts: int, seed: int) -> list[DatasetRequest]:
    if num_prompts <= 0 or num_prompts == len(requests):
        return list(requests)
    if num_prompts > len(requests):
        raise ValueError(f"num_prompts={num_prompts} exceeds dataset size {len(requests)}")
    # Preserve canonical trace order after sampling.
    selected_ids = {req.req_id for req in random.Random(seed).sample(requests, num_prompts)}
    return [req for req in requests if req.req_id in selected_ids]


class TPForwardProfiler:
    def __init__(
        self,
        model_dir: str,
        tp_world_size: int,
        max_num_blocks: int,
        gpu_memory_utilization: float,
        use_dummy_weights: bool,
    ) -> None:
        self.model_dir = model_dir
        self.tp_world_size = tp_world_size
        self.model_config = ModelConfig(
            model=model_dir,
            tokenizer=model_dir,
            dtype="fp16",
            use_dummy_weights=use_dummy_weights,
        )
        self.cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            max_num_blocks_per_req=2048 // BLOCK_SIZE + 2,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.workers = []

        pp_nccl_comm_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        tp_nccl_comm_id = copy.deepcopy(torch.ops.nccl_ops.generate_nccl_id())
        for tp_rank in range(tp_world_size):
            parallel_config = ParallelConfig(
                tensor_parallel_rank=tp_rank,
                tensor_parallel_size=tp_world_size,
                pipeline_parallel_rank=0,
                pipeline_parallel_size=1,
            )
            self.workers.append(
                Worker.remote(
                    worker_id=tp_rank,
                    model_config=self.model_config,
                    cache_config=self.cache_config,
                    parallel_config=parallel_config,
                    tensor_parallel_id=tp_nccl_comm_id,
                    pipeline_parallel_id=pp_nccl_comm_id,
                )
            )

        print(f"Loading model with tp_world_size={tp_world_size} ...")
        ray.get([worker.init_model.remote() for worker in self.workers])
        print(f"Initializing KV cache with num_gpu_blocks={max_num_blocks} ...")
        ray.get([worker.init_kvcache.remote(max_num_blocks) for worker in self.workers])

    def close(self) -> None:
        for worker in self.workers:
            ray.kill(worker, no_restart=True)
        self.workers = []

    def warmup(self) -> None:
        prompt = get_input_ids(self.model_dir, 16).view(1, 16).tolist()
        block_table = [list(range(blocks_for_request(32)))]
        ray.get([
            worker.step.remote(prompt, [0], block_table)
            for worker in self.workers
        ])
        ray.get([
            worker.step.remote([[1]], [16], block_table)
            for worker in self.workers
        ])

    def prefill_batch(self, requests: list[DatasetRequest]) -> tuple[list[int], float]:
        prompt_token_ids = build_prompt_token_ids(self.model_dir, [req.prompt_len for req in requests])
        block_table = build_block_table([req.prompt_len + max(req.output_len, 1) for req in requests])
        started_at = time.perf_counter()
        outputs = ray.get([
            worker.step.remote(
                prompt_token_ids,
                [0 for _ in requests],
                block_table,
            )
            for worker in self.workers
        ])[0]
        ended_at = time.perf_counter()
        return [int(token) for token in outputs], ended_at - started_at

    def decode_batch(self, states: list[DecodeRequestState]) -> tuple[list[int], float]:
        block_table = build_block_table([
            state.prompt_len + max(state.output_len, 1)
            for state in states
        ])
        started_at = time.perf_counter()
        outputs = ray.get([
            worker.step.remote(
                [[state.last_token_id] for state in states],
                [state.current_context_len for state in states],
                block_table,
            )
            for worker in self.workers
        ])[0]
        ended_at = time.perf_counter()
        return [int(token) for token in outputs], ended_at - started_at


def blocks_for_request(total_len: int) -> int:
    return max(1, (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE)


def build_block_table(total_lens: list[int]) -> list[list[int]]:
    block_table = []
    next_block = 0
    for total_len in total_lens:
        num_blocks = blocks_for_request(total_len)
        block_table.append(list(range(next_block, next_block + num_blocks)))
        next_block += num_blocks
    return block_table


def build_prompt_token_ids(model_dir: str, prompt_lens: list[int]) -> list[list[int]]:
    total_tokens = sum(prompt_lens)
    flat_ids = get_input_ids(model_dir, total_tokens).tolist()
    result = []
    offset = 0
    for prompt_len in prompt_lens:
        result.append(flat_ids[offset:offset + prompt_len])
        offset += prompt_len
    return result


def select_prefill_batch(
    pending: list[DatasetRequest],
    max_batch_size: int,
    max_tokens_per_batch: int,
) -> list[DatasetRequest]:
    batch = []
    tokens = 0
    while pending and len(batch) < max_batch_size:
        candidate = pending[0]
        if batch and tokens + candidate.prompt_len > max_tokens_per_batch:
            break
        batch.append(pending.pop(0))
        tokens += candidate.prompt_len
    return batch


def select_decode_batch(
    active: list[DecodeRequestState],
    max_batch_size: int,
    max_tokens_per_batch: int,
) -> list[DecodeRequestState]:
    batch = []
    tokens = 0
    for state in active:
        if state.remaining_tokens <= 0:
            continue
        round_tokens = state.current_context_len + 1
        if batch and tokens + round_tokens > max_tokens_per_batch:
            break
        batch.append(state)
        tokens += round_tokens
        if len(batch) >= max_batch_size:
            break
    return batch


def sleep_until_relative(start_perf: float, target_relative_time: float) -> None:
    delay = target_relative_time - (time.perf_counter() - start_perf)
    if delay > 0:
        time.sleep(delay)


def run_prefill_phase(args: argparse.Namespace, requests: list[DatasetRequest]) -> list[PrefillTraceRecord]:
    max_blocks = args.max_prefill_batch_size * max(
        blocks_for_request(req.prompt_len + max(req.output_len, 1))
        for req in requests
    )
    profiler = TPForwardProfiler(
        model_dir=args.model,
        tp_world_size=args.tp_world_size,
        max_num_blocks=max_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_dummy_weights=args.use_dummy_weights,
    )
    if args.warmup:
        profiler.warmup()

    pending: list[DatasetRequest] = []
    next_arrival_idx = 0
    records: list[PrefillTraceRecord] = []
    batch_id = 0
    phase_start = time.perf_counter()

    try:
        while next_arrival_idx < len(requests) or pending:
            now = time.perf_counter() - phase_start
            while next_arrival_idx < len(requests) and requests[next_arrival_idx].arrival_time <= now:
                pending.append(requests[next_arrival_idx])
                next_arrival_idx += 1

            if not pending:
                sleep_until_relative(phase_start, requests[next_arrival_idx].arrival_time)
                continue

            batch = select_prefill_batch(
                pending,
                max_batch_size=args.max_prefill_batch_size,
                max_tokens_per_batch=args.max_prefill_tokens_per_batch,
            )
            relative_start = time.perf_counter() - phase_start
            _, elapsed = profiler.prefill_batch(batch)
            relative_end = time.perf_counter() - phase_start
            for req in batch:
                records.append(
                    PrefillTraceRecord(
                        req_id=req.req_id,
                        prompt_len=req.prompt_len,
                        output_len=req.output_len,
                        arrival_time=req.arrival_time,
                        prefill_start_time=relative_start,
                        prefill_end_time=relative_end,
                        prefill_latency_ms=elapsed * 1000.0,
                        batch_id=batch_id,
                        batch_size=len(batch),
                        batch_prompt_lens=[item.prompt_len for item in batch],
                    )
                )
            batch_id += 1
            if batch_id == 1 or batch_id % 10 == 0:
                print(f"prefill batches={batch_id}, finished_requests={len(records)}/{len(requests)}")
    finally:
        profiler.close()

    records.sort(key=lambda record: record.req_id)
    return records


def run_decode_phase(
    args: argparse.Namespace,
    prefill_records: list[PrefillTraceRecord],
) -> tuple[list[DecodeRoundRecord], list[RequestResultRecord]]:
    max_blocks = args.max_decode_batch_size * max(
        blocks_for_request(record.prompt_len + max(record.output_len, 1))
        for record in prefill_records
    )
    profiler = TPForwardProfiler(
        model_dir=args.model,
        tp_world_size=args.tp_world_size,
        max_num_blocks=max_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_dummy_weights=args.use_dummy_weights,
    )
    if args.warmup:
        profiler.warmup()

    records_by_arrival = sorted(prefill_records, key=lambda record: (record.prefill_end_time, record.req_id))
    states_by_id: dict[int, DecodeRequestState] = {}
    active: list[DecodeRequestState] = []
    finished: list[DecodeRequestState] = []
    round_records: list[DecodeRoundRecord] = []
    next_arrival_idx = 0
    round_id = 0
    phase_start = time.perf_counter()

    try:
        while next_arrival_idx < len(records_by_arrival) or active:
            now = time.perf_counter() - phase_start
            while (
                next_arrival_idx < len(records_by_arrival)
                and records_by_arrival[next_arrival_idx].prefill_end_time <= now
            ):
                record = records_by_arrival[next_arrival_idx]
                state = DecodeRequestState(
                    req_id=record.req_id,
                    prompt_len=record.prompt_len,
                    output_len=record.output_len,
                    arrival_time=record.arrival_time,
                    prefill_start_time=record.prefill_start_time,
                    prefill_end_time=record.prefill_end_time,
                    generated_tokens=1 if record.output_len > 0 else 0,
                    last_token_id=1 + (record.req_id % 997),
                    token_timestamps=([record.prefill_end_time] if record.output_len > 0 else []),
                )
                states_by_id[state.req_id] = state
                if state.remaining_tokens > 0:
                    active.append(state)
                else:
                    finished.append(state)
                next_arrival_idx += 1

            if not active:
                if next_arrival_idx < len(records_by_arrival):
                    sleep_until_relative(phase_start, records_by_arrival[next_arrival_idx].prefill_end_time)
                continue

            batch = select_decode_batch(
                active,
                max_batch_size=args.max_decode_batch_size,
                max_tokens_per_batch=args.max_decode_tokens_per_batch,
            )
            if not batch:
                # Ensure forward progress for a single very long context.
                batch = [active[0]]

            relative_start = time.perf_counter() - phase_start
            output_ids, elapsed = profiler.decode_batch(batch)
            relative_end = time.perf_counter() - phase_start

            round_records.append(
                DecodeRoundRecord(
                    round_id=round_id,
                    decode_start_time=relative_start,
                    decode_end_time=relative_end,
                    forward_time_ms=elapsed * 1000.0,
                    batch_size=len(batch),
                    req_ids=[state.req_id for state in batch],
                    input_lens=[state.prompt_len for state in batch],
                    output_lens=[state.output_len for state in batch],
                    current_context_lens=[state.current_context_len for state in batch],
                    remaining_output_tokens=[state.remaining_tokens for state in batch],
                )
            )

            for state, token_id in zip(batch, output_ids):
                state.last_token_id = token_id
                state.generated_tokens += 1
                state.token_timestamps.append(relative_end)

            still_active = []
            for state in active:
                if state.remaining_tokens > 0:
                    still_active.append(state)
                else:
                    finished.append(state)
            active = still_active
            round_id += 1
            if round_id == 1 or round_id % 50 == 0:
                print(
                    f"decode rounds={round_id}, finished_requests={len(finished)}/{len(prefill_records)}, "
                    f"active={len(active)}"
                )
    finally:
        profiler.close()

    for state in states_by_id.values():
        if state not in finished and state.remaining_tokens <= 0:
            finished.append(state)
    request_results = build_request_results(states_by_id, prefill_records)
    return round_records, request_results


def build_request_results(
    states_by_id: dict[int, DecodeRequestState],
    prefill_records: list[PrefillTraceRecord],
) -> list[RequestResultRecord]:
    results = []
    for record in sorted(prefill_records, key=lambda item: item.req_id):
        state = states_by_id[record.req_id]
        end_time = state.token_timestamps[-1] if state.token_timestamps else record.prefill_end_time
        first_token_time = state.token_timestamps[0] if state.token_timestamps else record.prefill_end_time
        tpot = 0.0
        if record.output_len > 1 and len(state.token_timestamps) > 1:
            tpot = (state.token_timestamps[-1] - state.token_timestamps[0]) / (record.output_len - 1)
        lifecycle_events = [
            {"timestamp": record.arrival_time, "event_type": "issued"},
            {"timestamp": record.prefill_start_time, "event_type": "context_begin"},
            {"timestamp": record.prefill_end_time, "event_type": "context_end"},
            {"timestamp": record.prefill_end_time, "event_type": "migration_begin"},
            {"timestamp": record.prefill_end_time, "event_type": "migration_end"},
            {"timestamp": record.prefill_end_time, "event_type": "decoding_begin"},
            {"timestamp": end_time, "event_type": "decoding_end"},
        ]
        results.append(
            RequestResultRecord(
                prompt_len=record.prompt_len,
                output_len=record.output_len,
                start_time=record.arrival_time,
                end_time=end_time,
                token_timestamps=list(state.token_timestamps),
                lifecycle_events=lifecycle_events,
                latency=end_time - record.arrival_time,
                ftl=first_token_time - record.arrival_time,
                tpot=tpot,
            )
        )
    return results


def write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def read_prefill_trace(path: Path) -> list[PrefillTraceRecord]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(PrefillTraceRecord(**json.loads(line)))
    return records


def write_exp(path: Path, records: list[RequestResultRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dataclasses.asdict(record) for record in records], indent=2) + "\n")


def write_summary(path: Path, prefill_records: list[PrefillTraceRecord], round_records: list[DecodeRoundRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefill_latencies = [record.prefill_latency_ms for record in prefill_records]
    decode_forward_times = [record.forward_time_ms for record in round_records]
    summary = {
        "num_requests": len(prefill_records),
        "num_prefill_batches": len({record.batch_id for record in prefill_records}),
        "num_decode_rounds": len(round_records),
        "prefill_latency_ms": summarize(prefill_latencies),
        "decode_forward_time_ms": summarize(decode_forward_times),
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def parse_float_csv(value: str | None) -> list[float]:
    if value is None:
        return []
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def safe_rate_name(rate: float) -> str:
    if rate == float("inf"):
        return "rate_inf"
    text = f"{rate:g}".replace(".", "p").replace("-", "m")
    return f"rate_{text}"


def resolve_output_paths(args: argparse.Namespace, request_rate: float, multi_rate: bool) -> dict[str, Path | None]:
    if args.output_root is not None:
        rate_dir = args.output_root / safe_rate_name(request_rate)
        return {
            "prefill_trace": rate_dir / PREFILL_TRACE_NAME,
            "decode_rounds": rate_dir / DECODE_ROUNDS_NAME,
            "request_output": rate_dir / REQUEST_RESULTS_NAME,
            "summary": rate_dir / SUMMARY_NAME,
        }

    if multi_rate:
        raise ValueError("--output-root is required when using multiple request rates.")
    if args.prefill_trace_output is None:
        raise ValueError("--prefill-trace-output is required when --output-root is not set.")
    return {
        "prefill_trace": args.prefill_trace_output,
        "decode_rounds": args.decode_round_output,
        "request_output": args.request_output,
        "summary": args.summary_output,
    }


def run_for_rate(args: argparse.Namespace, request_rate: float, output_paths: dict[str, Path | None]) -> None:
    prefill_trace_output = output_paths["prefill_trace"]
    assert prefill_trace_output is not None

    prefill_records: list[PrefillTraceRecord]
    if args.phase in ("both", "prefill"):
        if args.dataset is None:
            raise ValueError("--dataset is required for phase=prefill/both")
        requests = load_dataset(
            args.dataset,
            request_rate=request_rate,
            process_name=args.process_name,
            cv=args.request_cv,
            seed=args.seed,
        )
        requests = maybe_sample_requests(requests, args.num_prompts, args.seed)
        print("=" * 80)
        print(f"dataset={args.dataset}")
        print(f"requests={len(requests)}")
        print(f"request_rate={request_rate}")
        print(f"tp_world_size={args.tp_world_size}")
        prefill_records = run_prefill_phase(args, requests)
        write_jsonl(prefill_trace_output, prefill_records)
        print(f"Wrote prefill trace: {prefill_trace_output}")
    else:
        prefill_records = read_prefill_trace(prefill_trace_output)

    if args.phase in ("both", "decode"):
        round_records, request_results = run_decode_phase(args, prefill_records)
        decode_round_output = output_paths["decode_rounds"]
        request_output = output_paths["request_output"]
        summary_output = output_paths["summary"]
        if decode_round_output is not None:
            write_jsonl(decode_round_output, round_records)
            print(f"Wrote decode rounds: {decode_round_output}")
        if request_output is not None:
            write_exp(request_output, request_results)
            print(f"Wrote request results: {request_output}")
        if summary_output is not None:
            write_summary(summary_output, prefill_records, round_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase TP profiler driven by dataset arrivals.")
    parser.add_argument("--phase", choices=["both", "prefill", "decode"], default="both")
    parser.add_argument("--model", required=True, help="Local model path.")
    parser.add_argument("--dataset", type=Path, default=None, help="Required for phase=prefill/both.")
    parser.add_argument("--num-prompts", type=int, default=0, help="0 means use all requests.")
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument(
        "--request-rates",
        type=str,
        default=None,
        help="Comma-separated request rates. If set, each rate is measured separately under --output-root.",
    )
    parser.add_argument("--request-cv", type=float, default=1.0)
    parser.add_argument("--process-name", choices=["poisson", "possion", "gamma", "uniform"], default="poisson")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tp-world-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--use-dummy-weights", action="store_true", default=True)
    parser.add_argument("--real-weights", dest="use_dummy_weights", action="store_false")
    parser.add_argument("--ray-temp-dir", type=Path, default=Path("/users/rh/tmp/ray_two_phase_profile"))
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--max-prefill-batch-size", type=int, default=8)
    parser.add_argument("--max-prefill-tokens-per-batch", type=int, default=8192)
    parser.add_argument("--max-decode-batch-size", type=int, default=8)
    parser.add_argument("--max-decode-tokens-per-batch", type=int, default=16384)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for multi-rate outputs. Files are written under "
            "<output-root>/rate_<rate>/."
        ),
    )
    parser.add_argument("--prefill-trace-output", type=Path, default=None)
    parser.add_argument("--decode-round-output", type=Path, default=None)
    parser.add_argument("--request-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.multiprocessing.set_start_method("spawn", force=True)
    args.ray_temp_dir.mkdir(parents=True, exist_ok=True)
    ray.init(address="local", ignore_reinit_error=True, _temp_dir=str(args.ray_temp_dir))

    request_rates = parse_float_csv(args.request_rates) or [args.request_rate]
    multi_rate = len(request_rates) > 1
    for request_rate in request_rates:
        output_paths = resolve_output_paths(args, request_rate, multi_rate)
        run_for_rate(args, request_rate, output_paths)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

#!/usr/bin/env python3
"""Run a vLLM P/D benchmark with explicit scheduler batch traces.

This client sends each request to a prefill vLLM instance first, then sends the
same request id plus the first generated token to a decode vLLM instance. The
P2pNcclConnector uses the request id to locate peer KV endpoints.
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class DatasetRequest:
    prompt: str
    prompt_len: int
    output_len: int


@dataclass
class RequestResult:
    prompt_len: int
    output_len: int
    start_time: float
    end_time: float
    token_timestamps: list[float]
    lifecycle_events: list[dict[str, Any]]
    client_request_id: str
    vllm_internal_request_id: str
    prefill_response_id: str | None
    decode_response_id: str | None
    prefill_token_ids: list[int]
    decode_token_ids: list[int]
    latency: float
    ftl: float
    tpot: float


def load_dataset(path: Path, num_prompts: int) -> list[DatasetRequest]:
    requests: list[DatasetRequest] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            requests.append(
                DatasetRequest(
                    prompt=str(item["prompt"]),
                    prompt_len=int(item["prompt_len"]),
                    output_len=int(item.get("output_len", item.get("output_tokens"))),
                )
            )
    if num_prompts > len(requests):
        raise ValueError(
            f"num_prompts={num_prompts} exceeds dataset size {len(requests)}"
        )
    return requests[:num_prompts]


def make_intervals(count: int, request_rate: float, process_name: str, cv: float):
    if request_rate in (float("inf"), 0.0):
        return [0.0 for _ in range(count)]
    if process_name == "uniform":
        return [1.0 / request_rate for _ in range(count)]
    if process_name in {"poisson", "possion"}:
        cv = 1.0
    if process_name in {"gamma", "poisson", "possion"}:
        shape = 1.0 / (cv * cv)
        scale = cv * cv / request_rate
        return np.random.gamma(shape, scale, size=count).tolist()
    raise ValueError(f"Unsupported process_name={process_name}")


def parse_openai_sse_line(raw_line: bytes) -> tuple[bool, str | None, list[int], str | None]:
    line = raw_line.decode("utf-8", errors="ignore").strip()
    if not line or not line.startswith("data:"):
        return False, None, [], None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return True, None, [], None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False, None, [], None
    if "error" in data:
        raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
    response_id = data.get("id")
    choices = data.get("choices") or []
    if not choices:
        return False, None, [], response_id
    choice = choices[0]
    return False, choice.get("text"), choice.get("token_ids") or [], response_id


async def post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    async with session.post(url, json=payload, headers=headers) as response:
        text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"POST {url} failed: status={response.status}, body={text}")
        data = json.loads(text)
        if "error" in data:
            raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
        return data


async def send_pd_request(
    request: DatasetRequest,
    request_index: int,
    token_ids: list[int],
    args: argparse.Namespace,
    pbar: tqdm,
) -> RequestResult:
    client_request_id = (
        f"{args.request_id_prefix}-{request_index}"
        f"___prefill_addr_{args.kv_host}:{args.prefill_kv_port}___"
        f"___decode_addr_{args.kv_host}:{args.decode_kv_port}"
    )
    internal_request_id = f"cmpl-{client_request_id}-0"
    headers = {
        "User-Agent": "PD Benchmark Client",
        "X-Request-Id": client_request_id,
    }
    prefill_url = f"http://{args.host}:{args.prefill_port}/v1/completions"
    decode_url = f"http://{args.host}:{args.decode_port}/v1/completions"

    sampling_common = {
        "model": args.model_alias,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "return_token_ids": True,
        "add_special_tokens": False,
    }

    request_start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout_s)) as session:
        async def run_decode_stream():
            decode_token_timestamps: list[float] = []
            decode_token_ids: list[int] = []
            decode_response_id = None
            decode_begin = time.perf_counter()
            decode_payload = {
                **sampling_common,
                "prompt": token_ids,
                "max_tokens": request.output_len,
                "min_tokens": request.output_len,
                "stream": True,
                "request_id": client_request_id,
            }
            async with session.post(
                decode_url,
                json=decode_payload,
                headers={**headers, "Accept": "text/event-stream"},
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"Decode failed: status={response.status}, body={body}"
                    )
                async for raw_line in response.content:
                    is_done, _, delta_token_ids, response_id = parse_openai_sse_line(
                        raw_line
                    )
                    if response_id is not None and decode_response_id is None:
                        decode_response_id = response_id
                    if is_done:
                        break
                    if not delta_token_ids:
                        continue
                    now = time.perf_counter()
                    for token_id in delta_token_ids:
                        decode_token_ids.append(int(token_id))
                        decode_token_timestamps.append(now)
            decode_end = time.perf_counter()
            return (
                decode_begin,
                decode_end,
                decode_response_id,
                decode_token_ids,
                decode_token_timestamps,
            )

        decode_task = asyncio.create_task(run_decode_stream())
        if args.decode_lead_s > 0:
            await asyncio.sleep(args.decode_lead_s)

        prefill_payload = {
            **sampling_common,
            "prompt": token_ids,
            "max_tokens": 1,
            "min_tokens": 1,
            "stream": False,
            "request_id": client_request_id,
        }
        try:
            prefill_response = await post_json(
                session, prefill_url, prefill_payload, headers
            )
        except Exception:
            decode_task.cancel()
            raise
        prefill_end = time.perf_counter()

        prefill_choice = prefill_response["choices"][0]
        first_token_ids = prefill_choice.get("token_ids") or []
        if not first_token_ids:
            raise RuntimeError(f"Prefill returned no token_ids for request {request_index}")

        (
            decode_begin,
            decode_end,
            decode_response_id,
            decode_token_ids,
            decode_token_timestamps,
        ) = await decode_task

    token_timestamps = decode_token_timestamps
    request_end = decode_end
    latency = request_end - request_start
    ftl = (
        token_timestamps[0] - request_start
        if token_timestamps
        else request_end - request_start
    )
    tpot = (
        0.0
        if len(token_timestamps) <= 1
        else (token_timestamps[-1] - token_timestamps[0]) / (len(token_timestamps) - 1)
    )
    lifecycle_events = [
        {"timestamp": request_start, "event_type": "issued"},
        {"timestamp": request_start, "event_type": "context_begin"},
        {"timestamp": decode_begin, "event_type": "decode_request_begin"},
        {"timestamp": prefill_end, "event_type": "context_end"},
        {"timestamp": prefill_end, "event_type": "migration_begin"},
        {"timestamp": decode_begin, "event_type": "migration_end"},
        {"timestamp": decode_begin, "event_type": "decoding_begin"},
        {"timestamp": decode_end, "event_type": "decoding_end"},
    ]
    pbar.update(1)
    return RequestResult(
        prompt_len=request.prompt_len,
        output_len=request.output_len,
        start_time=request_start,
        end_time=request_end,
        token_timestamps=token_timestamps,
        lifecycle_events=lifecycle_events,
        client_request_id=client_request_id,
        vllm_internal_request_id=internal_request_id,
        prefill_response_id=prefill_response.get("id"),
        decode_response_id=decode_response_id,
        prefill_token_ids=[int(x) for x in first_token_ids],
        decode_token_ids=decode_token_ids,
        latency=latency,
        ftl=ftl,
        tpot=tpot,
    )


async def run_once(args: argparse.Namespace, requests: list[DatasetRequest]):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenized = [
        tokenizer.encode(req.prompt, add_special_tokens=True)
        for req in requests
    ]
    intervals = make_intervals(
        len(requests), args.request_rate, args.process_name, args.request_cv
    )
    print(f"First 10 intervals: {intervals[:10]}")

    tasks = []
    pbar = tqdm(total=len(requests))
    for idx, (req, token_ids) in enumerate(zip(requests, tokenized)):
        tasks.append(
            asyncio.create_task(send_pd_request(req, idx, token_ids, args, pbar))
        )
        if intervals[idx] > 0:
            await asyncio.sleep(intervals[idx])
    results = await asyncio.gather(*tasks)
    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-port", type=int, default=18000)
    parser.add_argument("--decode-port", type=int, default=18001)
    parser.add_argument("--kv-host", default="127.0.0.1")
    parser.add_argument("--prefill-kv-port", type=int, default=19000)
    parser.add_argument("--decode-kv-port", type=int, default=19001)
    parser.add_argument("--num-prompts", type=int, default=120)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--request-cv", type=float, default=1.0)
    parser.add_argument("--request-id-prefix", default="bench")
    parser.add_argument(
        "--process-name", choices=["poisson", "possion", "gamma", "uniform"], default="poisson"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=10800)
    parser.add_argument("--decode-lead-s", type=float, default=0.02)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    requests = load_dataset(args.dataset, args.num_prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = asyncio.run(run_once(args, requests))
    elapsed = time.time() - start
    with args.output.open("w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f)
    print(f"Wrote {args.output}")
    print(f"Total time: {elapsed:.2f}s, throughput: {len(results) / elapsed:.3f} req/s")


if __name__ == "__main__":
    main()

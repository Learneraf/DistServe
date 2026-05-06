"""Benchmark online serving throughput.
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional
import os
import sys

import aiohttp
import numpy as np
from tqdm import tqdm

from structs import TestRequest, Dataset, RequestResult
from backends import BACKEND_TO_PORTS

pbar: Optional[tqdm] = None


def make_lifecycle_event(event_type: str, timestamp: float) -> dict[str, float | str]:
    return {
        "timestamp": float(timestamp),
        "event_type": event_type,
    }


def ensure_migration_lifecycle_events(
    lifecycle_events: Optional[list[dict]],
    start_time: float,
    token_timestamps: list[float],
    end_time: float,
) -> list[dict]:
    events = list(lifecycle_events or [])
    event_types = {event.get("event_type") for event in events if isinstance(event, dict)}
    if "migration_begin" in event_types and "migration_end" in event_types:
        return events

    first_token_time = token_timestamps[0] if token_timestamps else end_time
    if not events:
        events = [
            make_lifecycle_event("issued", start_time),
            make_lifecycle_event("context_begin", start_time),
            make_lifecycle_event("context_end", first_token_time),
            make_lifecycle_event("migration_begin", first_token_time),
            make_lifecycle_event("migration_end", first_token_time),
            make_lifecycle_event("decoding_begin", first_token_time),
            make_lifecycle_event("decoding_end", token_timestamps[-1] if token_timestamps else end_time),
        ]
        return events

    if "migration_begin" not in event_types:
        insert_ts = next(
            (
                float(event["timestamp"])
                for event in events
                if event.get("event_type") in {"context_end", "decoding_begin"}
            ),
            first_token_time,
        )
        events.append(make_lifecycle_event("migration_begin", insert_ts))
    if "migration_end" not in event_types:
        insert_ts = next(
            (
                float(event["timestamp"])
                for event in events
                if event.get("event_type") in {"decoding_begin", "context_end"}
            ),
            first_token_time,
        )
        events.append(make_lifecycle_event("migration_end", insert_ts))

    events.sort(key=lambda event: (float(event.get("timestamp", 0.0)), str(event.get("event_type", ""))))
    return events


def parse_openai_sse_line(
    raw_line: bytes,
) -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    line = raw_line.decode("utf-8", errors="ignore").strip()
    if not line or not line.startswith("data:"):
        return False, None, None, None

    payload = line[5:].strip()
    if payload == "[DONE]":
        return True, None, None, None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return False, None, None, None

    response_id = data.get("id")

    if "error" in data:
        return False, None, json.dumps(data["error"], ensure_ascii=False), response_id

    choices = data.get("choices") or []
    if not choices:
        return False, None, None, response_id

    choice = choices[0]
    delta_text = choice.get("text")
    if isinstance(delta_text, str):
        return False, delta_text, None, response_id

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return False, content, None, response_id

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return False, content, None, response_id

    return False, None, None, response_id

def sample_requests(dataset_path: str, num_prompts: int) -> List[TestRequest]:
    """
    sample_requests: Sample the given number of requests from the dataset.
    """
    dataset = Dataset.load(dataset_path)
    if num_prompts > len(dataset.reqs):
        raise ValueError(
            f"Number of prompts ({num_prompts}) is larger than the dataset size ({len(dataset.reqs)})."
        )
    # If the caller asks for the whole dataset, preserve the stored order so
    # split datasets can serve as canonical traces for fit/val/test runs.
    if num_prompts == len(dataset.reqs):
        return list(dataset.reqs)
    return random.sample(dataset.reqs, num_prompts)


async def get_request(
    input_requests: List[TestRequest],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
) -> AsyncGenerator[TestRequest, None]:
    interval_lens = len(input_requests)
    input_requests = iter(input_requests)

    if request_rate not in [float("inf"), 0.0]:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        else:
            raise ValueError(
                f"Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
            
    print(f"First {10} intervals: {intervals[:10]}")

    for idx, request in enumerate(input_requests):
        yield request
        if request_rate == float("inf") or request_rate == 0.0:
            continue

        interval = intervals[idx]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    verbose: bool,
    api_model: Optional[str] = None,
    allow_eos: bool = False,
    request_index: Optional[int] = None,
) -> RequestResult:
    global pbar
    if backend == "deepspeed":
        payload = {
            "prompt": prompt,
            "max_tokens": output_len,
            "min_new_tokens": output_len,
            "max_new_tokens": output_len,
            "stream": True,
            "max_length": int((prompt_len + output_len)*1.2+10) # *1.2 to prevent tokenization error
        }
        
        request_start_time = time.perf_counter()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3*3600)) as session:
            token_timestamps = []
            generated_text = ""
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for data in response.content.iter_any():
                            token_timestamps.append(time.perf_counter())
                            try:
                                generated_text += json.loads(data.decode("utf-8")[6:])["text"][0]
                            except:
                                generated_text += data.decode("utf-8")
                        complete_time = time.perf_counter()
                    else:
                        print(response)
                        print(response.status)
                        print(response.reason)
                        sys.exit(1)
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
                print(e)
                sys.exit(1)
        request_end_time = time.perf_counter()
        lifecycle_events = ensure_migration_lifecycle_events(
            None,
            request_start_time,
            token_timestamps,
            request_end_time,
        )
        
        if verbose:
            print(f"Prompt: {prompt}, Output: {generated_text}")
        
        pbar.update(1)
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=token_timestamps,
            lifetime_events=lifecycle_events
        )
    elif backend in ("distserve", "vllm"):
        headers = {"User-Agent": "Benchmark Client"}
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }

        # The maximum length of the input is 2048, limited by the embedding
        # table size.
        assert prompt_len+output_len < 2048
        
        request_start_time = time.perf_counter()
        request_output = None

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers, json=pload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                try:
                    output = json.loads(output)
                except:
                    print("Failed to parse the response:")
                    print(output)
                    continue
                if verbose:
                    print(f"Prompt: {prompt}\n\nOutput: {output['text']}")

                # Re-send the request if it failed.
                if "error" not in output:
                    request_output = output
                    break
                else:
                    print(f"Failed to process the request: {output['error']}")
                    print(f"Resending the request: {pload}")

        request_end_time = time.perf_counter()
        lifecycle_events = ensure_migration_lifecycle_events(
            request_output.get("lifetime_events", None),
            request_start_time,
            request_output["timestamps"],
            request_end_time,
        )
        
        pbar.update(1)        
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=request_output["timestamps"],
            lifetime_events=lifecycle_events
        )
    elif backend == "openai":
        client_request_id = (
            f"bench-{request_index}" if request_index is not None else None
        )
        headers = {
            "User-Agent": "Benchmark Client",
            "Accept": "text/event-stream",
        }
        if client_request_id is not None:
            headers["X-Request-Id"] = client_request_id
        payload = {
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "stream": True,
            "ignore_eos": not allow_eos,
        }
        if api_model:
            payload["model"] = api_model

        request_start_time = time.perf_counter()
        token_timestamps = []
        generated_text = ""
        sse_preview: list[str] = []
        sse_error: Optional[str] = None
        server_request_id: Optional[str] = None

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"Request {request_index} failed with status {response.status}: {body}"
                    )
                async for raw_line in response.content:
                    decoded_line = raw_line.decode("utf-8", errors="ignore").strip()
                    if decoded_line:
                        sse_preview.append(decoded_line)
                        if len(sse_preview) > 20:
                            sse_preview = sse_preview[-20:]

                    (
                        is_done,
                        delta_text,
                        line_error,
                        line_request_id,
                    ) = parse_openai_sse_line(raw_line)
                    if line_request_id is not None and server_request_id is None:
                        server_request_id = line_request_id
                    if is_done:
                        break
                    if line_error is not None:
                        sse_error = line_error
                        break
                    if delta_text is None:
                        continue
                    generated_text += delta_text
                    token_timestamps.append(time.perf_counter())

        request_end_time = time.perf_counter()
        lifecycle_events = ensure_migration_lifecycle_events(
            None,
            request_start_time,
            token_timestamps,
            request_end_time,
        )
        if sse_error is not None:
            raise RuntimeError(
                f"Request {request_index} received SSE error payload. "
                f"Prompt len={prompt_len}, output len={output_len}, error={sse_error}"
            )
        if not token_timestamps:
            raise RuntimeError(
                "Request produced no streamed token timestamps from the OpenAI-compatible backend. "
                f"Request={request_index}, prompt len={prompt_len}, output len={output_len}, "
                f"allow_eos={allow_eos}, last_sse_lines={sse_preview}"
            )
        if verbose:
            print(f"Prompt: {prompt}\n\nOutput: {generated_text}")

        pbar.update(1)
        result = RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=token_timestamps,
            lifetime_events=lifecycle_events
        )
        result.client_request_id = client_request_id
        result.server_request_id = server_request_id
        result.vllm_internal_request_id = (
            f"{server_request_id}-0" if server_request_id is not None else None
        )
        return result
    else:
        raise ValueError(f"Unknown backend: {backend}")


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[TestRequest],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
    verbose: bool = False,
    api_model: Optional[str] = None,
    allow_eos: bool = False,
) -> List[RequestResult]:
    tasks: List[asyncio.Task] = []
    async for request_index, request in async_enumerate(
        get_request(
            input_requests, process_name, request_rate, request_cv
        )
    ):
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                request.prompt,
                request.prompt_len,
                request.output_len,
                best_of,
                use_beam_search,
                verbose,
                api_model,
                allow_eos,
                request_index,
            )
        )
        tasks.append(task)
    request_results = await asyncio.gather(*tasks)
    return request_results


async def async_enumerate(generator: AsyncGenerator[TestRequest, None]):
    index = 0
    async for item in generator:
        yield index, item
        index += 1


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.backend == "openai":
        api_url = f"http://{args.host}:{args.port}/v1/completions"
    else:
        api_url = f"http://{args.host}:{args.port}/generate"
    input_requests = sample_requests(
        args.dataset, args.num_prompts
    )
    print("Sampling done. Start benchmarking...")

    global pbar
    pbar = tqdm(total=args.num_prompts)
    benchmark_start_time = time.time()
    request_results = asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            args.request_cv,
            args.process_name,
            args.verbose,
            args.api_model,
            args.allow_eos,
        )
    )
    benchmark_end_time = time.time()
    pbar.close()
    
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput:")
    print(f"\t{args.num_prompts / benchmark_time:.2f} requests/s")
    print(f"\t{sum([req.prompt_len + req.output_len for req in input_requests]) / benchmark_time:.2f} tokens/s")
    print(f"\t{sum([req.output_len for req in input_requests]) / benchmark_time:.2f} output tokens/s")

    with open(args.output, "w") as f:
        json.dump(request_results, f, default=vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend", type=str, default="distserve", choices=["distserve", "vllm", "deepspeed", "openai"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="Optional model field to include for OpenAI-compatible backends.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the (preprocessed) dataset."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="Allow early EOS on OpenAI-compatible backends. By default, ignore EOS to match target output lengths.",
    )
    parser.add_argument(
        "--num-prompts-req-rates", type=str, required=True,
        help="[(num_prompts, request_rate), ...] where num_prompts is the number of prompts to process and request_rate is the number of requests per second.",
    )
    parser.add_argument(
        "--request-cv",
        type=float,
        default=1.0,
        help="the coefficient of variation of the gap between" "the requests.",
    )
    parser.add_argument(
        "--process-name",
        type=str,
        default="possion",
        choices=["possion", "gamma", "uniform"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    
    parser.add_argument(
        "--exp-result-root",
        type=str,
        default=None,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: env var EXP_RESULT_ROOT)"
    )
    parser.add_argument(
        "--exp-result-dir",
        type=str,
        required=True,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: <model_name>-<dataset.name>)"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default=None,
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default: <backend>)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logs (prompts and outputs)."
    )
    
    args = parser.parse_args()
    
    if args.exp_result_root == None:
        if "EXP_RESULT_ROOT" not in os.environ:
            print(f"Error: EXP_RESULT_ROOT is not set in environment variables")
            sys.exit(1)
        args.exp_result_root = os.getenv("EXP_RESULT_ROOT")
        
    if args.exp_result_prefix == None:
        args.exp_result_prefix = args.backend
        
    if args.port == None and args.backend != "openai":
        args.port = BACKEND_TO_PORTS[args.backend]
    elif args.port is None:
        print("Error: --port is required for backend=openai")
        sys.exit(1)
        
    num_prompts_request_rates = eval(args.num_prompts_req_rates)
    for (num_prompts, request_rate) in num_prompts_request_rates:
        print("===================================================================")
        print(f"Running with num_prompts={num_prompts}, request_rate={request_rate}")
        args.num_prompts = num_prompts
        args.request_rate = request_rate
        output_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{args.exp_result_prefix}-{num_prompts}-{request_rate}.exp")
        main(args)
        time.sleep(1)
    

    '''
    usage:
    python ./2-benchmark-serving.py \
        --dataset sharegpt.json \
        --host 127.0.0.1 \
        --port 8400 \
        --num-prompts-req-rates "[(100, 1), (100, 1.5), (100, 2), (100, 2.5), (100, 3), (100, 3.5), (100, 4)]" \
        --exp-result-root "./result" \
        --exp-result-dir "llama_7B"  \
        --verbose


/users/rh/miniconda3/envs/distserve/bin/python ./2-benchmark-serving.py \
    --backend distserve \
    --dataset /users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.1-8B/val.jsonl \
    --host 127.0.0.1 \
    --port 8400 \
    --seed 0 \
    --num-prompts-req-rates "[(120, 1), (120, 1.5), (120, 2), (120, 2.5), (120, 3), (120, 3.5), (120, 4)]" \
    --exp-result-root ./result/val \
    --exp-result-dir llama_8B

    '''

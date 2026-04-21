"""
Simulate DistServe

Output a JSON (list) where each item is the lifecycle for a request.

"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Literal, Union

import numpy as np
import pandas as pd
import simpy
from transformers import AutoTokenizer

from simdistserve.base.organize_data import organize_request_df, organize_request_event_df, \
    calculate_per_request_latency, organize_worker_event_df
from simdistserve.base.scheduler import put_requests_with_interarrivals
from simdistserve.base.worker import WorkerConfig
from simdistserve.base.workload import (
    get_gamma_interarrival,
    get_fixed_interarrival,
    convert_absolutearrival_to_interarrival, convert_pd_pair_to_request, sample_requests
)
from simdistserve.base.request import Request
from simdistserve.clusters.disagg import DisaggCluster
from simdistserve.clusters.vllm import VLLMCluster
from simdistserve.constants import ModelTypes
from simdistserve.estimators.memory_estimator import get_max_num_tokens, is_model_runnable


# VLLM_ASCEND_HANDOFF_DELAY_MS = 65.40447611397082
# VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS = 0.09436758011886258
VLLM_ASCEND_HANDOFF_DELAY_MS = 0
VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS = 0


def parse_args(args_=None):
    parser = argparse.ArgumentParser(description='Simulation: vLLM, DistServe')
    parser.add_argument('--backend', type=str, default='distserve',
                        help='Backend to simulate (distserve, vllm, vllm_ascend)')
    parser.add_argument('--model', type=str, default='facebook/opt-13b',
                        help='Model type (opt_13b, opt_66b, opt_175b,'
                             'llama_7b, llama_2_7b,'
                             'or facebook/opt-13b, facebook/opt-66b, facebook/opt-175b,'
                             'huggyllama/llama-7b, anonymous4chan/llama-2-7b)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rate', type=float, default=float("inf"),
                        help='Rate of requests per second')
    parser.add_argument('--N', type=int, default=64, help='Number of requests')
    parser.add_argument(
        '--arrival', type=str, default='poisson',
        help=('Arrival distribution (gamma, poisson, fixed, custom). '
              'If custom, then require the JSON file workload to specify '
              'the "start_time" field for each incoming request.'))
    parser.add_argument(
        '--workload', type=str, default='sharegpt',
        help=(
            'Workload type, or a JSON file that contains the workload. '
            'The workload file should be a list of pairs with (prompt_len, decode_len) length. '
            '(e.g.: "sharegpt", "longbench", "humaneval", or specify your own path like "./workload/workload.json")')
    )
    parser.add_argument('--cv', type=float, default=1.0)
    parser.add_argument('--tp-prefill', type=int, default=1, help='Number of TP per prefill worker (used in DistServe)')
    parser.add_argument('--pp-prefill', type=int, default=1, help='Number of PP per prefill worker (used in DistServe)')
    parser.add_argument('--tp-decode', type=int, default=1, help='Number of TP per decode worker (used in DistServe)')
    parser.add_argument('--pp-decode', type=int, default=1, help='Number of PP per decode worker (used in DistServe)')
    parser.add_argument('--name', type=str, default=None)  # Experiment name
    parser.add_argument('--output', type=str, default=None, help='Output SLA (csv)')
    parser.add_argument('--output-request-info', type=str, default=None, help='Output request info (csv)')
    parser.add_argument('--output-request-event', type=str, default=None,
                        help='Output per-request event dataframe (csv)')
    parser.add_argument('--output-request-latency', type=str, default=None, help='Output per-request latency (csv)')
    parser.add_argument('--output-worker', type=str, default=None,
                        help='Output per-worker per-iteration time (csv)')
    parser.add_argument('--prefill-containment', type=int, default=None,
                        help='Containment target for prefill')
    parser.add_argument('--prefill-target', type=int, default=200,
                        help='Target latency for prefill')
    parser.add_argument('--decode-containment', type=int, default=None,
                        help='Containment target for decode')
    parser.add_argument('--slas', type=str, default='[85, 90, 95, 98, 99]',
                        help='Fix attainment and get the target.'),
    parser.add_argument('--slo-scales', type=str, default='[1.0, 0.4, 0.6, 0.8, 1.2]',
                        help='SLO scales in a python list.'),
    parser.add_argument('--decode-target', type=int, default=100,
                        help='Target latency for decode')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')
    parser.add_argument(
        '--handoff-delay-ms',
        type=float,
        default=None,
        help=('Fixed prefill-to-decode handoff delay in ms for disaggregated backends. '
              f'Defaults to {VLLM_ASCEND_HANDOFF_DELAY_MS:.6f} for vllm_ascend and 0 otherwise.'),
    )
    parser.add_argument(
        '--handoff-delay-per-token-ms',
        type=float,
        default=None,
        help=('Additional handoff delay per live token at transfer time. '
              f'Defaults to {VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS:.6f} for vllm_ascend and 0 otherwise.'),
    )
    parser.add_argument(
        '--handoff-capacity',
        type=int,
        default=1,
        help='Maximum number of concurrent handoffs in the simulated cluster.',
    )
    parser.add_argument(
        '--prefill-first-token-visible-immediately',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=('Whether a first token generated during prefill is immediately visible '
              'to the user before the handoff completes. Defaults to false for '
              'vllm_ascend and true otherwise.'),
    )

    args = parser.parse_args(args=args_)

    assert args.backend in ['distserve', 'vllm', 'vllm_ascend'], f'Unknown backend: {args.backend}'
    assert args.arrival in ['poisson', 'gamma', 'fixed', 'custom'], f'Unknown arrival process: {args.arrival}'
    args.slo_scales = eval(args.slo_scales)
    args.slas = eval(args.slas)
    assert isinstance(args.slo_scales, list)
    assert isinstance(args.slas, list)
    return args


def check_dataset_existence(x):
    if not Path(x).exists():
        raise FileNotFoundError(f"Dataset {x} does not exist.")
    return


def load_workload(
    workload,
    N,
    rate,
    cv,
    seed,
    process: Literal["fixed", "gamma", "custom"],
    model_path: str,
    do_sample=False,
):
    random.seed(seed)
    np.random.seed(seed)

    # 加载数据集
    import sys
    import os
    structs_path = os.path.join(os.path.dirname(__file__), '../..', 'evaluation', '2-benchmark-serving', 'structs.py')
    sys.path.append(os.path.dirname(structs_path))
    from structs import Dataset, TestRequest
    
    
    if not do_sample:
        dataset = []
        with open(workload, "r", encoding="utf-8") as f:
            for line in f:
                # 去除空行/换行符，避免报错
                line = line.strip()
                if not line:
                    continue
                # 解析每行 JSON
                json_data = json.loads(line)
                dataset.append(json_data)

        sampled_test_requests = dataset
    else:
        dataset = Dataset.load(workload)
        if N > len(dataset.reqs):
            raise ValueError(
                f"Number of prompts ({N}) is larger than the dataset size ({len(dataset.reqs)})."
            )
        sampled_test_requests = random.sample(dataset.reqs, N)

    model_name = model_path
    tokenizer = None

    requests = []
    for i, req in enumerate(sampled_test_requests):
        if isinstance(req, dict) and req.get("prompt_len") is not None:
            prefill_len = req["prompt_len"]
        elif isinstance(req, dict) and req.get("prompt") is not None:
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            token_ids = tokenizer.encode(req["prompt"])
            prefill_len = len(token_ids)

        # 获取 prompt 文本（假设 req 有 prompt 属性）
        elif isinstance(req, TestRequest) and hasattr(req, 'prompt_len') and req.prompt_len is not None:
            prefill_len = req.prompt_len
        elif isinstance(req, TestRequest) and hasattr(req, 'prompt') and req.prompt is not None:
            # 仅在数据集中缺少 prompt_len 时才重新分词，避免与真实基准的长度定义不一致。
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            token_ids = tokenizer.encode(req.prompt)
            prefill_len = len(token_ids)
        else:
            # 回退：使用原有 prompt_len 字段（兼容旧数据集）
            prefill_len = req.prompt_len
            print(f"Warning: Request {i} missing 'prompt' attribute, falling back to prompt_len={prefill_len}")

        if isinstance(req, dict):
            output_len = req.get("output_len", req.get("output_tokens"))
            if output_len is None:
                raise ValueError(
                    f"Custom workload request {i} must provide either 'output_len' or 'output_tokens'."
                )
        else:
            output_len = req.output_len

        requests.append(
            Request(
                env=None,
                req_id=i,
                source_index=(int(req["source_index"]) if isinstance(req, dict) and req.get("source_index") is not None else i),
                prefill_length=prefill_len,
                output_lens=output_len,
            )
        )

    request_count = len(sampled_test_requests)

    # 生成到达间隔（与原逻辑相同）
    if process == 'custom':
        start_times = []
        for i, req in enumerate(sampled_test_requests):
            if not isinstance(req, dict) or req.get("start_time") is None:
                raise ValueError(
                    f"Custom workload request {i} must provide 'start_time' when --arrival custom is used."
                )
            start_times.append(float(req["start_time"]))
        arrival = convert_absolutearrival_to_interarrival(start_times)
    elif process == 'fixed':
        delay = 1 / rate * 1000  # ms
        arrival = get_fixed_interarrival(request_count, delay)
    else:
        arrival = get_gamma_interarrival(request_count, rate, cv, seed=seed)

    print(f"First {10} requests: {requests[:10]}")
    print(f"First {10} arrivals: {arrival[:10]}")

    return requests, arrival


def main(args, outputs=None):
    outputs = outputs if outputs is not None else {}

    cv = args.cv
    N = args.N
    rate = args.rate
    seed = args.seed
    workload: Union[Literal["sharegpt", "longbench", "humaneval"], str] = args.workload
    model_type = ModelTypes.model_str_to_object(args.model)
    process = args.arrival

    TP_Prefill = args.tp_prefill
    PP_prefill = args.pp_prefill
    TP_Decode = args.tp_decode
    PP_decode = args.pp_decode

    #
    # Handle vllm in data processing
    #
    if not is_model_runnable(model_type, TP_Prefill, PP_prefill):
        raise ValueError(
            f"Model {model_type} is not runnable with TP={TP_Prefill}, PP={PP_prefill}. "
            f"Skipping by throwing exception..."
        )

    prefill_max_tokens = get_max_num_tokens(model_type, TP_Prefill, PP_prefill)
    if args.backend == 'vllm_ascend':
        # The Ascend path is served through a proxy that disaggregates
        # prefill and decode onto different devices.
        if not is_model_runnable(model_type, TP_Decode, PP_decode):
            raise ValueError(
                f"Model {model_type} is not runnable with decode TP={TP_Decode}, PP={PP_decode}. "
                f"Skipping by throwing exception..."
            )
        decode_max_tokens = get_max_num_tokens(model_type, TP_Decode, PP_decode)
    else:
        decode_max_tokens = get_max_num_tokens(model_type, TP_Decode, PP_decode)

    # Setting the seed to sample request / process
    requests, arrival = load_workload(workload, N, rate, cv, seed, process, args.model, False)
    N = len(requests)
    handoff_delay_ms = args.handoff_delay_ms
    if handoff_delay_ms is None:
        handoff_delay_ms = VLLM_ASCEND_HANDOFF_DELAY_MS if args.backend == 'vllm_ascend' else 0.0
    handoff_delay_per_token_ms = args.handoff_delay_per_token_ms
    if handoff_delay_per_token_ms is None:
        handoff_delay_per_token_ms = (
            VLLM_ASCEND_HANDOFF_DELAY_PER_TOKEN_MS if args.backend == 'vllm_ascend' else 0.0
        )
    prefill_first_token_visible_immediately = args.prefill_first_token_visible_immediately
    if prefill_first_token_visible_immediately is None:
        prefill_first_token_visible_immediately = args.backend != 'vllm_ascend'

    # Run simulation
    env = simpy.Environment()
    if args.backend == 'vllm':
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Prefill,
            prefill_max_batch_size=10 ** 7,  # inf
            decode_max_batch_size=10 ** 7,  # inf
            prefill_max_tokens=prefill_max_tokens,
            decode_max_tokens=prefill_max_tokens,
            enable_chunked_prefill=False,
            engine_type=args.backend,
        )

        cluster = VLLMCluster(
            env=env, PP=PP_prefill, worker_configs=worker_config,
        )
    elif args.backend == 'vllm_ascend':
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
            prefill_max_batch_size=10 ** 7,
            decode_max_batch_size=10 ** 7,  # inf
            prefill_max_tokens=prefill_max_tokens,
            decode_max_tokens=decode_max_tokens,
            enable_chunked_prefill=False,
            engine_type=args.backend,
            prefill_generates_first_token=True,
            handoff_delay_ms=handoff_delay_ms,
            handoff_delay_per_token_ms=handoff_delay_per_token_ms,
            handoff_capacity=args.handoff_capacity,
            prefill_first_token_visible_immediately=prefill_first_token_visible_immediately,
        )

        cluster = DisaggCluster(
            env=env, PP_prefill=PP_prefill, PP_decode=PP_decode,
            worker_configs=worker_config,
        )
    elif args.backend == 'distserve':
        worker_config = WorkerConfig(
            model_type=model_type,
            TP=TP_Prefill, TP_Prefill=TP_Prefill, TP_Decode=TP_Decode,
            prefill_max_batch_size=10 ** 7,  # inf
            decode_max_batch_size=10 ** 7,  # inf
            prefill_max_tokens=prefill_max_tokens,
            decode_max_tokens=decode_max_tokens,
            enable_chunked_prefill=False,
            engine_type=args.backend,
            handoff_delay_ms=handoff_delay_ms,
            handoff_delay_per_token_ms=handoff_delay_per_token_ms,
            handoff_capacity=args.handoff_capacity,
            prefill_first_token_visible_immediately=prefill_first_token_visible_immediately,
        )

        cluster = DisaggCluster(
            env=env, PP_prefill=PP_prefill, PP_decode=PP_decode,
            worker_configs=worker_config,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    cluster.run()
    put_requests_with_interarrivals(env, cluster.scheduler, arrival, requests)
    env.run()

    #
    # Collect request-level data and containment
    #
    request_df = organize_request_df(requests)
    request_event_df = organize_request_event_df(requests)
    per_request_latency_df = calculate_per_request_latency(
        request_event_df,
        request_df.output_lens,
        request_df.first_token_prefill,
    )
    outputs['request_df'] = request_df
    outputs['request_event_df'] = request_event_df
    outputs['per_request_latency_df'] = per_request_latency_df
    per_request_latency_export_df = per_request_latency_df.reset_index()
    if args.output_request_info:
        os.makedirs(os.path.dirname(args.output_request_info), exist_ok=True)
        with open(args.output_request_info, 'w') as f:
            request_df.to_csv(f, index=False)
    if args.output_request_event:
        os.makedirs(os.path.dirname(args.output_request_event), exist_ok=True)
        with open(args.output_request_event, 'w') as f:
            request_event_df.to_csv(f, index=False)
    if args.output_request_latency:
        os.makedirs(os.path.dirname(args.output_request_latency), exist_ok=True)
        with open(args.output_request_latency, 'w') as f:
            per_request_latency_export_df.to_csv(f, index=False)

    columns = [
        "backend", "model_type", "pd", "rate", "target", "attainment",
        "tp_prefill", "pp_prefill", "tp_decode", "pp_decode",
    ]
    output_results = []
    # Fix the prefill & decode target (SLO & scale),
    # then find the attainment (percentage of requests that meet the SLO)
    for scale in args.slo_scales:
        prefill_target = args.prefill_target * scale
        prefill_attainment = (per_request_latency_df['first_token_latency'] <= prefill_target).sum() / N
        prefill_attainment *= 100
        item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        decode_target = args.decode_target * scale
        decode_attainment = (per_request_latency_df['tpot'] <= decode_target).sum() / N
        decode_attainment *= 100
        item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)

        both_attainment = (
                              (per_request_latency_df['first_token_latency'] <= prefill_target) &
                              (per_request_latency_df['tpot'] <= decode_target)
                          ).sum() / N
        both_attainment *= 100
        item = [args.backend, model_type, 'both', rate, (prefill_target, decode_target), both_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        pass

    # Fix the attainment (percentage of requests that meet the SLO),
    # then find the prefill /  decode SLO target that it can meet.
    slas = args.slas
    for sla in slas:
        prefill_attainment = decode_attainment = sla
        prefill_target = per_request_latency_df['first_token_latency'].quantile(prefill_attainment / 100)
        decode_target = per_request_latency_df['tpot'].quantile(decode_attainment / 100)
        item = [args.backend, model_type, 'prefill', rate, prefill_target, prefill_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        item = [args.backend, model_type, 'decode', rate, decode_target, decode_attainment,
                TP_Prefill, PP_prefill, TP_Decode, PP_decode]
        output_results.append(item)
        pass

    df = pd.DataFrame(output_results, columns=columns)
    outputs['latency_df'] = df

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            df.to_csv(f, index=False)

    if args.verbose:
        print(df.to_markdown())

    #
    # Collect worker-level data
    #
    if args.output_worker:
        os.makedirs(os.path.dirname(args.output_worker), exist_ok=True)
        worker_df = organize_worker_event_df(cluster)
        worker_df.to_csv(args.output_worker, index=False)

        outputs['worker_df'] = worker_df

    #
    # Return if the agreement of prefill/decode is met
    #
    is_prefill_contained = None
    is_decode_contained = None

    prefill_containment = args.prefill_containment
    prefill_target = args.prefill_target
    if prefill_containment:
        # See if the P{prefill_containment} is less than prefill_target
        t = per_request_latency_df['first_token_latency'].quantile(prefill_containment / 100)
        is_prefill_contained = t < prefill_target
        pass

    decode_containment = args.decode_containment
    decode_target = args.decode_target
    if decode_containment:
        t = per_request_latency_df['tpot'].quantile(decode_containment / 100)
        is_decode_contained = t < decode_target
        pass

    return is_prefill_contained, is_decode_contained, df


run_experiment = main


def test_opt_13b_grid_search_serial():
    arg_lists = [
        [
            '--arrival', 'poisson',
            '--seed', '0',
            '--N', '100',
            '--prefill-containment', '90',  # P90
            '--prefill-target', '200',  # ms
            '--decode-containment', '90',  # P90
            '--decode-target', '100',  # ms
            '--model', 'opt_13b',
            '--workload', 'sharegpt',
        ]
    ]

    config_list = [
        [
            '--rate', f'{rate}',
            # '--output',
            # f'raw_results/request.opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv',
            # '--output-worker',
            # f'raw_results/worker.opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv',
            '--pp-prefill', f'{pp_prefill}',
            '--pp-decode', f'{pp_decode}',
            '--tp-prefill', f'{tp_prefill}',
            '--tp-decode', f'{tp_decode}',
        ]
        # for rate in range(1, 8)
        for rate in range(1, 9)
        for pp_prefill in [1, 2, 4, 8]
        for pp_decode in [1, 2, 4, 8]
        for tp_prefill in [1, 2, 4, 8]
        for tp_decode in [1, 2, 4, 8]
    ]

    best_config = None
    best_goodput = 0
    # pbar = tqdm(total=len(arg_lists) * len(config_list))

    print("tp_prefill,pp_prefill,tp_decode,pp_decode,rate,goodput")
    for machine_config in config_list:
        best_config_this_iter = None
        for task_config in arg_lists:
            # pbar.update(1)

            args = parse_args(args_=task_config + machine_config)
            key = (
                args.tp_prefill, args.pp_prefill,
                args.tp_decode, args.pp_decode,
            )
            num_gpu = args.pp_prefill * args.tp_prefill + args.pp_decode * args.tp_decode
            rate = args.rate * num_gpu
            if num_gpu > 32:
                continue
            goodput = args.rate / num_gpu
            if goodput < best_goodput:
                continue

            # print(args.rate, args.pp_prefill, args.tp_prefill, args.pp_decode, args.tp_decode)
            is_prefill_contained, is_decode_contained, containment_df = main(args)

            if not is_prefill_contained or not is_decode_contained:
                break

            if goodput > best_goodput:
                best_config = args
                best_goodput = goodput

                print(
                    f"{best_config.tp_prefill},{best_config.pp_prefill},"
                    f"{best_config.tp_decode},{best_config.pp_decode},"
                    f"{rate},{best_goodput}")

    best_config_str = (f"(Prefill TP = {best_config.tp_prefill}, Prefill PP = {best_config.pp_prefill},"
                       f" Decode TP = {best_config.tp_decode}, Decode PP = {best_config.pp_decode})")
    print(f"Best Config: {best_config_str} with goodput {best_goodput}")


def test_opt_13b_one_case(
    rate=1, tp_prefill=1, pp_prefill=1, tp_decode=1, pp_decode=1,
    backend='vllm',
):
    suffix = f"opt-13b-p{tp_prefill}{pp_prefill}{tp_decode}{pp_decode}-rate{rate}.csv"
    args = [
        '--arrival', 'poisson',
        '--seed', '0',
        '--N', '1000',
        '--backend', backend,
        '--prefill-containment', '90',  # P90
        '--prefill-target', '200',  # ms
        '--decode-containment', '90',  # P90
        '--decode-target', '100',  # ms
        '--model', 'opt_13b',
        '--workload', 'sharegpt',
        '--rate', f'{rate}',
        '--output', f'logs/request.{suffix}',
        '--output-request-event', f'logs/request-event.{suffix}',
        '--output-request-latency', f'logs/request-latency.{suffix}',
        '--output-worker', f'logs/worker.{suffix}',
        '--pp-prefill', f'{pp_prefill}',
        '--pp-decode', f'{pp_decode}',
        '--tp-prefill', f'{tp_prefill}',
        '--tp-decode', f'{tp_decode}',
        '--verbose',
    ]
    args = parse_args(args_=args)
    main(args)
    return


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    # test_opt_13b_grid_search_serial()
    pass

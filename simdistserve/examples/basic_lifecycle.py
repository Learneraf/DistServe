import argparse

from simdistserve.examples.common import (
    ExampleConfig,
    describe_config,
    format_frame,
    request_event_view,
    request_table,
    run_simulation,
    worker_event_view,
    latency_summary,
)

REQUEST_SPECS = [
    (64, 4),
    (256, 6),
    (128, 5),
]
INTERARRIVAL_MS = [0.0, 6.0, 14.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Show the event-level lifecycle of a tiny simdistserve run.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        default="distserve",
        choices=["distserve", "vllm", "vllm_ascend"],
    )
    parser.add_argument("--tp-prefill", type=int, default=1)
    parser.add_argument("--pp-prefill", type=int, default=1)
    parser.add_argument("--tp-decode", type=int, default=1)
    parser.add_argument("--pp-decode", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model or ("facebook/opt-13b" if args.backend == "vllm" else "huggyllama/llama-7b")
    config = ExampleConfig(
        backend=args.backend,
        model=model,
        tp_prefill=args.tp_prefill,
        pp_prefill=args.pp_prefill,
        tp_decode=args.tp_decode,
        pp_decode=args.pp_decode,
    )

    result = run_simulation(REQUEST_SPECS, INTERARRIVAL_MS, config)

    print("Configuration")
    print(describe_config(config))
    print()
    print("Workload")
    print(format_frame(request_table(REQUEST_SPECS, INTERARRIVAL_MS)))
    print()
    print("Latency Summary (ms)")
    print(format_frame(latency_summary(result.latency_df)))
    print()
    print("Request Timeline")
    print(format_frame(request_event_view(result.request_event_df)))
    print()
    print("Worker Timeline")
    print(format_frame(worker_event_view(result.worker_df)))


if __name__ == "__main__":
    main()

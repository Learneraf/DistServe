import argparse

from simdistserve.examples.common import (
    ExampleConfig,
    chunk_slicing_plan,
    decode_round_membership_view,
    describe_config,
    format_frame,
    handoff_event_view,
    request_event_view,
    request_table,
    run_simulation,
)


def trace_chunked_prefill():
    request_specs = [(1536, 6)]
    interarrival_ms = [0.0]
    config = ExampleConfig(
        backend="distserve",
        model="huggyllama/llama-7b",
        enable_chunked_prefill=True,
        prefill_max_tokens_cap=384,
    )
    result = run_simulation(request_specs, interarrival_ms, config)

    print("Chunked Prefill Trace")
    print(describe_config(config))
    print()
    print("Workload")
    print(format_frame(request_table(request_specs, interarrival_ms)))
    print()
    print("Pure Python Slice Plan")
    print(format_frame(chunk_slicing_plan(prefill_len=1536, chunk_cap=384)))
    print()
    print("Simulator Request Timeline")
    timeline = request_event_view(result.request_event_df)
    timeline = timeline[timeline.req_id == 0]
    print(format_frame(timeline))
    handoff_timeline = handoff_event_view(result.request_event_df)
    handoff_timeline = handoff_timeline[handoff_timeline.req_id == 0]
    if not handoff_timeline.empty:
        print()
        print("Handoff / Visibility Events")
        print(format_frame(handoff_timeline))
    print()
    print("How to read it")
    print(
        "Each `do_prefill` consumes one chunk, `wait_prefill` means `remain_prefill_lens > 0`, and only the final "
        "`finish_prefill` completes prefill. In the current disaggregated path, the request then goes through "
        "`wait_handoff` / `do_handoff` / `finish_handoff` before decode starts."
    )


def trace_continuous_batching():
    request_specs = [
        (512, 24),
        (64, 8),
        (64, 8),
        (64, 8),
    ]
    interarrival_ms = [0.0, 70.0, 70.0, 70.0]
    config = ExampleConfig(
        backend="vllm",
        model="facebook/opt-13b",
        tp_prefill=1,
        pp_prefill=1,
    )
    result = run_simulation(request_specs, interarrival_ms, config)

    print("Continuous Batching Trace")
    print(describe_config(config))
    print()
    print("Workload")
    print(format_frame(request_table(request_specs, interarrival_ms)))
    print()
    print("Decode Round Membership by Request ID")
    print(format_frame(decode_round_membership_view(result.request_event_df)))
    print()
    print("Condensed Request Timeline")
    timeline = request_event_view(result.request_event_df)
    timeline = timeline[timeline.event_type.isin(["init", "finish_prefill", "do_decode", "exit_system"])]
    print(format_frame(timeline))
    print()
    print("How to read it")
    print(
        "The `req_ids` column is the active decode batch for that round. `joiners` are requests that became decode-"
        "ready since the previous round, and `leavers` are requests that completed decoding."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trace the worker's chunked-prefill and continuous-batching mechanics."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "chunked_prefill", "continuous_batching"],
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode in {"all", "chunked_prefill"}:
        trace_chunked_prefill()
    if args.mode == "all":
        print()
        print("=" * 80)
        print()
    if args.mode in {"all", "continuous_batching"}:
        trace_continuous_batching()


if __name__ == "__main__":
    main()

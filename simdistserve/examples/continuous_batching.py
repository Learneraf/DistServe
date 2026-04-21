from simdistserve.examples.common import (
    ExampleConfig,
    decode_round_view,
    describe_config,
    format_frame,
    request_event_view,
    request_latency_view,
    request_table,
    run_simulation,
)

REQUEST_SPECS = [
    (512, 24),
    (64, 8),
    (64, 8),
    (64, 8),
]
INTERARRIVAL_MS = [0.0, 70.0, 70.0, 70.0]


def main():
    config = ExampleConfig(
        backend="vllm",
        model="facebook/opt-13b",
        tp_prefill=1,
        pp_prefill=1,
    )
    result = run_simulation(REQUEST_SPECS, INTERARRIVAL_MS, config)

    decode_df = decode_round_view(result.worker_df)
    joined_df = decode_df.assign(active_requests=decode_df["decode_batch"].apply(len))

    print("Configuration")
    print(describe_config(config))
    print()
    print("Workload")
    print(format_frame(request_table(REQUEST_SPECS, INTERARRIVAL_MS)))
    print()
    print("Decode Rounds")
    print(format_frame(joined_df))
    print()
    print("Request Latency")
    print(format_frame(request_latency_view(result.latency_df)))
    print()
    print("Request Timeline")
    timeline = request_event_view(result.request_event_df)
    print(format_frame(timeline[timeline.event_type.isin(['init', 'do_prefill', 'do_decode', 'exit_system'])]))
    print()
    print("Observation")
    print(
        "The decode batch grows from 2 to 3 to 4 active requests as later arrivals finish prefill and join ongoing "
        "decode rounds. That is the continuous-batching behavior modeled by this simulator."
    )


if __name__ == "__main__":
    main()

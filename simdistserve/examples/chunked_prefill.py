import pandas as pd

from simdistserve.examples.common import (
    ExampleConfig,
    describe_config,
    format_frame,
    prefill_round_view,
    request_event_view,
    request_table,
    run_simulation,
)

REQUEST_SPECS = [
    (1536, 6),
    (256, 3)
]
INTERARRIVAL_MS = [0.0, 10.0]


def scenario_rows(name: str, result) -> dict[str, float]:
    latency = result.latency_df.iloc[0]
    return {
        "scenario": name,
        "ttft_ms": latency["first_token_latency"],
        "tpot_ms": latency["tpot"],
        "total_latency_ms": latency["total_latency"],
    }


def main():
    non_chunked = ExampleConfig(
        backend="distserve",
        model="huggyllama/llama-7b",
    )
    chunked = ExampleConfig(
        backend="distserve",
        model="huggyllama/llama-7b",
        enable_chunked_prefill=True,
        prefill_max_tokens_cap=384,
    )

    base_result = run_simulation(REQUEST_SPECS, INTERARRIVAL_MS, non_chunked)
    chunk_result = run_simulation(REQUEST_SPECS, INTERARRIVAL_MS, chunked)

    print("Workload")
    print(format_frame(request_table(REQUEST_SPECS, INTERARRIVAL_MS)))
    print()
    print("Configurations")
    print(describe_config(non_chunked))
    print(describe_config(chunked))
    print()
    print("Latency Comparison")
    print(
        format_frame(
            pd.DataFrame(
                [
                    scenario_rows("non_chunked", base_result),
                    scenario_rows("chunked_cap_384", chunk_result),
                ]
            )
        )
    )
    print()
    print("Non-Chunked Prefill Rounds")
    print(format_frame(prefill_round_view(base_result.worker_df)))
    print()
    print("Chunked Prefill Rounds")
    print(format_frame(prefill_round_view(chunk_result.worker_df)))
    print()
    print("Non-Chunked Request Timeline")
    print(format_frame(request_event_view(base_result.request_event_df)))
    print()
    print("Chunked Request Timeline")
    print(format_frame(request_event_view(chunk_result.request_event_df)))
    print()
    print("Observation")
    print(
        "With chunked prefill enabled, the same 1536-token prompt is split into four 384-token prefill rounds, so the "
        "request alternates between `do_prefill` and `wait_prefill` before the final `finish_prefill` hands it off "
        "to decode."
    )


if __name__ == "__main__":
    main()

import pandas as pd

from simdistserve.examples.common import (
    ExampleConfig,
    describe_config,
    format_frame,
    request_latency_view,
    request_table,
    run_simulation,
    worker_event_view,
)

REQUEST_SPECS = [
    (512, 16),
    (512, 16),
    (512, 16),
    (512, 16),
]
INTERARRIVAL_MS = [0.0, 0.0, 0.0, 0.0]


def scenario_summary(name: str, result) -> dict[str, float]:
    latency = result.latency_df
    return {
        "scenario": name,
        "mean_ttft_ms": latency["first_token_latency"].mean(),
        "mean_tpot_ms": latency["tpot"].mean(),
        "mean_total_ms": latency["total_latency"].mean(),
    }


def main():
    pp1 = ExampleConfig(
        backend="vllm",
        model="facebook/opt-13b",
        tp_prefill=1,
        pp_prefill=1,
    )
    pp2 = ExampleConfig(
        backend="vllm",
        model="facebook/opt-13b",
        tp_prefill=1,
        pp_prefill=2,
    )

    pp1_result = run_simulation(REQUEST_SPECS, INTERARRIVAL_MS, pp1)
    pp2_result = run_simulation(REQUEST_SPECS, INTERARRIVAL_MS, pp2)

    print("Workload")
    print(format_frame(request_table(REQUEST_SPECS, INTERARRIVAL_MS)))
    print()
    print("Configurations")
    print(describe_config(pp1))
    print(describe_config(pp2))
    print()
    print("Latency Comparison")
    print(
        format_frame(
            pd.DataFrame(
                [
                    scenario_summary("pp=1", pp1_result),
                    scenario_summary("pp=2", pp2_result),
                ]
            )
        )
    )
    print()
    print("PP=1 Worker Timeline")
    print(format_frame(worker_event_view(pp1_result.worker_df).head(16)))
    print()
    print("PP=2 Worker Timeline")
    print(format_frame(worker_event_view(pp2_result.worker_df).head(24)))
    print()
    print("PP=1 Request Latency")
    print(format_frame(request_latency_view(pp1_result.latency_df)))
    print()
    print("PP=2 Request Latency")
    print(format_frame(request_latency_view(pp2_result.latency_df)))
    print()
    print("Observation")
    print(
        "With `pp=2`, the workload flows through two pipeline stages, so the worker timeline alternates across two "
        "worker ids instead of one."
    )


if __name__ == "__main__":
    main()

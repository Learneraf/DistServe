import argparse

import pandas as pd

from simdistserve.examples.common import (
    ExampleConfig,
    describe_config,
    fixed_interarrivals_from_rate,
    format_frame,
    request_table,
    run_simulation,
)

REQUEST_SPECS = [
    (256, 24),
    (1024, 24),
    (64, 32),
    (768, 24),
    (64, 32),
    (512, 24),
    (64, 32),
    (128, 32),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the same request list under different arrival patterns."
    )
    parser.add_argument("--backend", type=str, default="distserve", choices=["distserve", "vllm"])
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def scenario_summary(name: str, result) -> dict[str, float]:
    latency = result.latency_df
    return {
        "scenario": name,
        "mean_ttft_ms": latency["first_token_latency"].mean(),
        "p90_ttft_ms": latency["first_token_latency"].quantile(0.90),
        "mean_tpot_ms": latency["tpot"].mean(),
        "p90_tpot_ms": latency["tpot"].quantile(0.90),
        "mean_total_ms": latency["total_latency"].mean(),
        "max_total_ms": latency["total_latency"].max(),
    }


def per_request_delta(burst_result, spread_result) -> pd.DataFrame:
    burst_df = burst_result.latency_df.loc[:, ["req_id", "first_token_latency", "tpot", "total_latency"]].rename(
        columns={
            "first_token_latency": "burst_ttft_ms",
            "tpot": "burst_tpot_ms",
            "total_latency": "burst_total_ms",
        }
    )
    spread_df = spread_result.latency_df.loc[:, ["req_id", "first_token_latency", "tpot", "total_latency"]].rename(
        columns={
            "first_token_latency": "spread_ttft_ms",
            "tpot": "spread_tpot_ms",
            "total_latency": "spread_total_ms",
        }
    )
    merged = burst_df.merge(spread_df, on="req_id", how="inner")
    merged["ttft_delta_ms"] = merged["burst_ttft_ms"] - merged["spread_ttft_ms"]
    merged["total_delta_ms"] = merged["burst_total_ms"] - merged["spread_total_ms"]
    return merged


def main():
    args = parse_args()
    model = args.model or ("facebook/opt-13b" if args.backend == "vllm" else "huggyllama/llama-7b")
    if args.backend == "vllm":
        config = ExampleConfig(backend="vllm", model=model, tp_prefill=1, pp_prefill=1)
    else:
        config = ExampleConfig(
            backend="distserve",
            model=model,
            tp_prefill=1,
            pp_prefill=1,
            tp_decode=1,
            pp_decode=1,
        )

    burst_interarrival_ms = [0.0] * len(REQUEST_SPECS)
    spread_interarrival_ms = fixed_interarrivals_from_rate(len(REQUEST_SPECS), rate=6.0)

    burst_result = run_simulation(REQUEST_SPECS, burst_interarrival_ms, config)
    spread_result = run_simulation(REQUEST_SPECS, spread_interarrival_ms, config)

    print("Configuration")
    print(describe_config(config))
    print()
    print("Burst Workload")
    print(format_frame(request_table(REQUEST_SPECS, burst_interarrival_ms)))
    print()
    print("Spread Workload")
    print(format_frame(request_table(REQUEST_SPECS, spread_interarrival_ms)))
    print()
    print("Scenario Comparison (ms)")
    print(
        format_frame(
            pd.DataFrame(
                [
                    scenario_summary("burst", burst_result),
                    scenario_summary("spread", spread_result),
                ]
            )
        )
    )
    print()
    print("Per-Request Delta (positive delta means the burst case is slower)")
    print(format_frame(per_request_delta(burst_result, spread_result)))


if __name__ == "__main__":
    main()
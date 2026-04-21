import argparse

import pandas as pd

from simdistserve.examples.common import (
    ExampleConfig,
    attainment_summary,
    fixed_interarrivals_from_rate,
    format_frame,
    repeated_pattern,
    run_simulation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep request arrival rate and show how latency/SLO attainment changes."
    )
    parser.add_argument("--backend", type=str, default="distserve", choices=["distserve", "vllm"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--prefill-target", type=float, default=200.0)
    parser.add_argument("--decode-target", type=float, default=100.0)
    return parser.parse_args()


def main():
    args = parse_args()
    request_specs = repeated_pattern(
        pattern=[
            (256, 24),
            (512, 24),
            (128, 24),
            (768, 24),
        ],
        repeats=8,
    )
    rates = [10.0, 20.0, 30.0, 40.0, 60.0, 80.0]
    rows = []

    if args.model is None:
        default_model = "facebook/opt-13b" if args.backend == "vllm" else "huggyllama/llama-7b"
    else:
        default_model = args.model

    if args.backend == "distserve":
        config = ExampleConfig(
            backend="distserve",
            model=default_model,
            tp_prefill=1,
            pp_prefill=1,
            tp_decode=1,
            pp_decode=1,
        )
    elif args.backend == "vllm":
        config = ExampleConfig(
            backend="vllm",
            model=default_model,
            tp_prefill=1,
            pp_prefill=1,
        )

    for rate in rates:
        interarrival_ms = fixed_interarrivals_from_rate(len(request_specs), rate=rate)
        result = run_simulation(request_specs, interarrival_ms, config)
        attainment = attainment_summary(
            result.latency_df,
            prefill_target_ms=args.prefill_target,
            decode_target_ms=args.decode_target,
        )
        rows.append(
            {
                "rate_req_per_s": rate,
                "mean_ttft_ms": result.latency_df["first_token_latency"].mean(),
                "p90_ttft_ms": result.latency_df["first_token_latency"].quantile(0.90),
                "mean_tpot_ms": result.latency_df["tpot"].mean(),
                "p90_tpot_ms": result.latency_df["tpot"].quantile(0.90),
                "prefill_attainment_pct": attainment["prefill_attainment_pct"],
                "decode_attainment_pct": attainment["decode_attainment_pct"],
                "joint_attainment_pct": attainment["joint_attainment_pct"],
            }
        )

    print("Arrival-rate sweep")
    print(
        f"backend={config.backend}, model={config.model}, "
        f"prefill_target_ms={args.prefill_target}, decode_target_ms={args.decode_target}"
    )
    print()
    print(format_frame(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()

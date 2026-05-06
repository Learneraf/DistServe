#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODEL_PATHS = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}


def load_sim_ftl_ms(path: Path) -> list[float]:
    with path.open(newline="") as f:
        return [float(row["first_token_latency"]) for row in csv.DictReader(f)]


def load_exp_ftl_ms(path: Path) -> list[float]:
    with path.open(encoding="utf-8") as f:
        requests = json.load(f)
    return [1000.0 * float(request["ftl"]) for request in requests]


def collapse_duplicate_x(x_values: list[float], y_values: list[float]):
    collapsed: list[tuple[float, float]] = []
    for x_value, y_value in zip(x_values, y_values):
        if collapsed and abs(x_value - collapsed[-1][0]) < 1e-9:
            collapsed[-1] = (x_value, y_value)
        else:
            collapsed.append((x_value, y_value))
    return collapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a per-model FTL quantile calibration file."
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        required=True,
        help="Root containing vllm_ascend/organized_data/<model>/rate_<rate>/request_latency.csv.",
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        required=True,
        help="Root containing <model>/vllm-pd-120-<rate>.exp.",
    )
    parser.add_argument("--rate", default="1")
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS))
    parser.add_argument("--num-quantiles", type=int, default=121)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probabilities = np.linspace(0.0, 1.0, args.num_quantiles)
    calibration = {}
    for model in args.models:
        sim_latency_path = (
            args.sim_root
            / "vllm_ascend"
            / "organized_data"
            / model
            / f"rate_{args.rate}"
            / "request_latency.csv"
        )
        exp_path = args.exp_root / model / f"vllm-pd-120-{args.rate}.exp"
        raw_ftl = np.array(load_sim_ftl_ms(sim_latency_path), dtype=float)
        real_ftl = np.array(load_exp_ftl_ms(exp_path), dtype=float)
        x_values = np.quantile(raw_ftl, probabilities).tolist()
        y_values = np.quantile(real_ftl, probabilities).tolist()
        collapsed = collapse_duplicate_x(x_values, y_values)
        calibration[MODEL_PATHS[model]] = {
            "first_token_latency": {
                "source": "quantile_map_from_sim_ftl_to_real_exp_ftl",
                "x_ms": [float(x_value) for x_value, _ in collapsed],
                "y_ms": [float(y_value) for _, y_value in collapsed],
            }
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

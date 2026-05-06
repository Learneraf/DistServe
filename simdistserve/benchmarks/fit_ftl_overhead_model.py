#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


MODEL_PATHS = {
    "llama_1B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2",
    "llama_3B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2",
    "llama_7B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2",
    "llama_8B": "/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2",
}

DATASET_DIRS = {
    "llama_1B": "llama-3.2-1B",
    "llama_3B": "llama-3.2-3B",
    "llama_7B": "llama-2-7b",
    "llama_8B": "llama-3.1-8B",
}

SLO_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2]


def load_sim_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [
            {
                "ftl_ms": float(row["first_token_latency"]),
                "tpot_ms": float(row["tpot"]),
            }
            for row in csv.DictReader(f)
        ]


def load_exp_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as f:
        requests = json.load(f)
    return [
        {
            "ftl_ms": 1000.0 * float(request["ftl"]),
            "tpot_ms": 1000.0 * float(request["tpot"]),
        }
        for request in requests
    ]


def load_workload_rows(path: Path, n: int) -> list[dict[str, float]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({
                "prompt_len": float(item["prompt_len"]),
                "output_len": float(item["output_len"]),
            })
            if len(rows) >= n:
                break
    return rows


def attainment(rows: list[dict[str, float]]) -> list[float]:
    values = []
    for scale in SLO_SCALES:
        ftl_target = 200.0 * scale
        tpot_target = 100.0 * scale
        ok = sum(
            row["ftl_ms"] <= ftl_target and row["tpot_ms"] <= tpot_target
            for row in rows
        )
        values.append(100.0 * ok / len(rows))
    return values


def score(actual: list[float], target: list[float]) -> tuple[float, float]:
    deltas = [abs(a - b) for a, b in zip(actual, target)]
    return max(deltas), sum(deltas) / len(deltas)


def apply_overhead(
    sim_rows: list[dict[str, float]],
    workload_rows: list[dict[str, float]],
    a: float,
    b: float,
    c: float,
) -> list[dict[str, float]]:
    rows = []
    for sim_row, workload_row in zip(sim_rows, workload_rows):
        overhead = max(
            0.0,
            a + b * workload_row["prompt_len"] + c * workload_row["output_len"],
        )
        rows.append({
            "ftl_ms": sim_row["ftl_ms"] + overhead,
            "tpot_ms": sim_row["tpot_ms"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a parametric first-token overhead model on top of raw "
            "simulation output."
        )
    )
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--exp-root", type=Path, required=True)
    parser.add_argument(
        "--workload-base",
        type=Path,
        default=Path(
            "/users/rh/DistServe/simdistserve/dataset/splits/"
            "sharegpt_four_models_common_ascend1900_seed0"
        ),
    )
    parser.add_argument("--mode", default="fit")
    parser.add_argument("--rate", default="1")
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = {}
    for model in args.models:
        sim_path = (
            args.sim_root
            / "vllm_ascend"
            / "organized_data"
            / model
            / f"rate_{args.rate}"
            / "request_latency.csv"
        )
        exp_path = args.exp_root / model / f"vllm-pd-120-{args.rate}.exp"
        workload_path = (
            args.workload_base / DATASET_DIRS[model] / f"{args.mode}.jsonl"
        )
        sim_rows = load_sim_rows(sim_path)
        exp_rows = load_exp_rows(exp_path)
        workload_rows = load_workload_rows(workload_path, len(sim_rows))
        target = attainment(exp_rows)

        best = None
        # Coarse grid is intentional: SLO attainment changes in 1/120 request
        # steps, so overfitting finer coefficients does not buy real precision.
        for a in range(-100, 151, 5):
            for b_i in range(-100, 201, 5):
                b = b_i / 1000.0
                for c_i in range(-200, 101, 5):
                    c = c_i / 1000.0
                    actual = attainment(apply_overhead(sim_rows, workload_rows, a, b, c))
                    max_error, mean_error = score(actual, target)
                    candidate = (max_error, mean_error, a, b, c, actual, target)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate

        assert best is not None
        max_error, mean_error, a, b, c, actual, target = best
        output[MODEL_PATHS[model]] = {
            "first_token_overhead": {
                "features": ["constant", "prompt_len", "output_len"],
                "coeffs": [float(a), float(b), float(c)],
                "clamp_min_zero": True,
                "target": "first_token_latency_ms",
                "fit_objective": "minimize_slo_attainment_delta",
                "slo_scales": SLO_SCALES,
                "fit_max_abs_delta_pct": float(max_error),
                "fit_mean_abs_delta_pct": float(mean_error),
                "fit_sim_attainment_pct": [float(value) for value in actual],
                "fit_real_attainment_pct": [float(value) for value in target],
            }
        }
        print(
            f"{model}: max_abs_delta={max_error:.2f}%, "
            f"mean_abs_delta={mean_error:.2f}%, "
            f"coeffs={[float(a), float(b), float(c)]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

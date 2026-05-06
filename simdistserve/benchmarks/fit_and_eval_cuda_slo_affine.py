#!/usr/bin/env python3
"""Fit SLO-aware FTL affine corrections on fit and evaluate on val.

This is intentionally not a quantile map.  For each model/rate pair, it learns
one monotonic affine correction from the fit split:

    corrected_ftl_ms = max(0, alpha * simulated_ftl_ms + beta)

The objective is the maximum SLO-attainment delta over the configured SLO
scales on the fit split.  The learned alpha/beta are then applied unchanged to
the val split.  Decode/TPOT is left as the raw simulator output because the
model_forward decode fit is already within 1 percentage point on these runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODELS = ["llama_1B", "llama_3B", "llama_7B", "llama_8B"]
RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
SLO_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2]
REQUEST_INDEX_RE = re.compile(r"-bench-(\d+)")


@dataclass(frozen=True)
class CaseData:
    real_ftl_ms: list[float]
    real_tpot_ms: list[float]
    sim_ftl_ms: list[float]
    sim_tpot_ms: list[float]


def request_index(item: dict, fallback: int) -> int:
    for key in ("client_request_id", "vllm_internal_request_id"):
        match = REQUEST_INDEX_RE.search(str(item.get(key, "")))
        if match:
            return int(match.group(1))
    return fallback


def load_exp(path: Path) -> dict[int, tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for fallback, item in enumerate(data):
        rows[request_index(item, fallback)] = (
            1000.0 * float(item["ftl"]),
            1000.0 * float(item["tpot"]),
        )
    return rows


def load_sim(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            int(row["req_id"]): (
                float(row["first_token_latency"]),
                float(row["tpot"]),
            )
            for row in csv.DictReader(f)
        }


def load_case(exp_root: Path, sim_root: Path, split: str, model: str, rate: str) -> CaseData | None:
    exp_path = exp_root / split / model / f"vllm-pd-120-{rate}.exp"
    sim_path = (
        sim_root
        / split
        / "vllm_ascend"
        / "organized_data"
        / model
        / f"rate_{rate}"
        / "request_latency.csv"
    )
    if not exp_path.exists() or not sim_path.exists():
        return None
    exp = load_exp(exp_path)
    sim = load_sim(sim_path)
    keys = sorted(set(exp) & set(sim))
    return CaseData(
        real_ftl_ms=[exp[key][0] for key in keys],
        real_tpot_ms=[exp[key][1] for key in keys],
        sim_ftl_ms=[sim[key][0] for key in keys],
        sim_tpot_ms=[sim[key][1] for key in keys],
    )


def rate(values: list[float], target: float) -> float:
    return 100.0 * sum(value <= target for value in values) / len(values)


def both_rate(ftl_values: list[float], tpot_values: list[float], scale: float) -> float:
    ftl_target = 200.0 * scale
    tpot_target = 100.0 * scale
    return 100.0 * sum(
        ftl <= ftl_target and tpot <= tpot_target
        for ftl, tpot in zip(ftl_values, tpot_values)
    ) / len(ftl_values)


def fit_affine(case: CaseData, alphas: np.ndarray, betas: np.ndarray) -> dict:
    best = None
    real_rates = [
        rate(case.real_ftl_ms, 200.0 * scale)
        for scale in SLO_SCALES
    ]
    raw = np.array(case.sim_ftl_ms, dtype=float)
    for alpha in alphas:
        scaled = alpha * raw
        for beta in betas:
            corrected = np.maximum(0.0, scaled + beta).tolist()
            deltas = [
                abs(rate(corrected, 200.0 * scale) - real_rate)
                for scale, real_rate in zip(SLO_SCALES, real_rates)
            ]
            candidate = (max(deltas), sum(deltas) / len(deltas), float(alpha), float(beta), deltas)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    assert best is not None
    return {
        "alpha": best[2],
        "beta_ms": best[3],
        "fit_max_prefill_delta_pct": best[0],
        "fit_mean_prefill_delta_pct": best[1],
        "fit_prefill_deltas_pct": [float(value) for value in best[4]],
    }


def corrected_ftl(case: CaseData, params: dict) -> list[float]:
    alpha = float(params["alpha"])
    beta = float(params["beta_ms"])
    return [max(0.0, alpha * value + beta) for value in case.sim_ftl_ms]


def write_eval(
    path: Path,
    split: str,
    cases: dict[tuple[str, str], CaseData],
    params: dict[str, dict[str, dict]],
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "model",
        "rate",
        "slo_scale",
        "metric",
        "real_attainment_pct",
        "raw_attainment_pct",
        "corrected_attainment_pct",
        "raw_delta_pct",
        "corrected_delta_pct",
    ]
    max_raw = 0.0
    max_corrected = 0.0
    within_5 = 0
    total = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, request_rate), case in sorted(cases.items(), key=lambda item: (item[0][0], float(item[0][1]))):
            ftl = corrected_ftl(case, params[model][request_rate])
            for scale in SLO_SCALES:
                metrics = {
                    "prefill": (
                        rate(case.real_ftl_ms, 200.0 * scale),
                        rate(case.sim_ftl_ms, 200.0 * scale),
                        rate(ftl, 200.0 * scale),
                    ),
                    "decode": (
                        rate(case.real_tpot_ms, 100.0 * scale),
                        rate(case.sim_tpot_ms, 100.0 * scale),
                        rate(case.sim_tpot_ms, 100.0 * scale),
                    ),
                    "both": (
                        both_rate(case.real_ftl_ms, case.real_tpot_ms, scale),
                        both_rate(case.sim_ftl_ms, case.sim_tpot_ms, scale),
                        both_rate(ftl, case.sim_tpot_ms, scale),
                    ),
                }
                for metric, (real_value, raw_value, corrected_value) in metrics.items():
                    raw_delta = abs(raw_value - real_value)
                    corrected_delta = abs(corrected_value - real_value)
                    max_raw = max(max_raw, raw_delta)
                    max_corrected = max(max_corrected, corrected_delta)
                    total += 1
                    within_5 += corrected_delta <= 5.0
                    writer.writerow(
                        {
                            "split": split,
                            "model": model,
                            "rate": request_rate,
                            "slo_scale": scale,
                            "metric": metric,
                            "real_attainment_pct": f"{real_value:.6f}",
                            "raw_attainment_pct": f"{raw_value:.6f}",
                            "corrected_attainment_pct": f"{corrected_value:.6f}",
                            "raw_delta_pct": f"{raw_delta:.6f}",
                            "corrected_delta_pct": f"{corrected_delta:.6f}",
                        }
                    )
    return {
        "max_raw_delta_pct": max_raw,
        "max_corrected_delta_pct": max_corrected,
        "cases": total,
        "corrected_cases_within_5pct": within_5,
        "corrected_fraction_within_5pct": within_5 / total if total else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("/users/rh/cuda_data"))
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-min", type=float, default=0.4)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-step", type=float, default=0.025)
    parser.add_argument("--beta-min", type=float, default=-100.0)
    parser.add_argument("--beta-max", type=float, default=200.0)
    parser.add_argument("--beta-step", type=float, default=2.0)
    args = parser.parse_args()

    fit_cases: dict[tuple[str, str], CaseData] = {}
    val_cases: dict[tuple[str, str], CaseData] = {}
    for split, cases in (("fit", fit_cases), ("val", val_cases)):
        for model in MODELS:
            for request_rate in RATES:
                case = load_case(args.exp_root, args.sim_root, split, model, request_rate)
                if case is not None:
                    cases[(model, request_rate)] = case

    alphas = np.arange(args.alpha_min, args.alpha_max + 0.5 * args.alpha_step, args.alpha_step)
    betas = np.arange(args.beta_min, args.beta_max + 0.5 * args.beta_step, args.beta_step)
    params: dict[str, dict[str, dict]] = {model: {} for model in MODELS}
    for model in MODELS:
        for request_rate in RATES:
            case = fit_cases.get((model, request_rate))
            if case is None:
                continue
            params[model][request_rate] = fit_affine(case, alphas, betas)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "slo_affine_model.json").write_text(
        json.dumps(
            {
                "train_split": "fit",
                "test_split": "val",
                "formula": "corrected_ftl_ms = max(0, alpha * simulated_ftl_ms + beta_ms)",
                "decode_tpot": "raw simulator output",
                "slo_scales": SLO_SCALES,
                "params": params,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "fit": write_eval(args.output_dir / "fit_slo_delta.csv", "fit", fit_cases, params),
        "val": write_eval(args.output_dir / "val_slo_delta.csv", "val", val_cases, params),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

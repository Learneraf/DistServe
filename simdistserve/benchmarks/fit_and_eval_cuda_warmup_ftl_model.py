#!/usr/bin/env python3
"""Fit a run-order warmup FTL model on top of raw 3p3d output.

This keeps the raw 3p3d prefill/decode execution profile unchanged.  The FTL
mapping is

    modeled_ftl_ms =
        max(0, alpha * raw_3p3d_ftl_ms + beta_ms
               + warmup_amp_ms * exp(-global_request_index / warmup_tau))

`global_request_index` follows the benchmark run order, i.e. all requests at
rate 1, then rate 1.5, etc.  The term models connector/CUDA graph/cache warmup
observed in the real traces without using request_rate as a direct feature.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODELS = ["llama_1B", "llama_3B", "llama_8B"]
RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
SLO_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2]
REQUEST_INDEX_RE = re.compile(r"-bench-(\d+)")


@dataclass(frozen=True)
class CaseRows:
    split: str
    model: str
    rate: str
    req_ids: np.ndarray
    global_request_index: np.ndarray
    raw_ftl_ms: np.ndarray
    raw_tpot_ms: np.ndarray
    real_ftl_ms: np.ndarray
    real_tpot_ms: np.ndarray


def request_index(item: dict, fallback: int) -> int:
    for key in ("client_request_id", "vllm_internal_request_id"):
        match = REQUEST_INDEX_RE.search(str(item.get(key, "")))
        if match:
            return int(match.group(1))
    return fallback


def load_exp(path: Path) -> dict[int, tuple[float, float]]:
    rows = {}
    data = json.loads(path.read_text(encoding="utf-8"))
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


def load_case(split: str, model: str, rate: str, exp_root: Path, sim_root: Path) -> CaseRows | None:
    exp_path = exp_root / split / model / f"vllm-pd-120-{rate}.exp"
    sim_path = sim_root / split / "vllm_ascend" / "organized_data" / model / f"rate_{rate}" / "request_latency.csv"
    if not exp_path.exists() or not sim_path.exists():
        return None

    exp = load_exp(exp_path)
    sim = load_sim(sim_path)
    req_ids = sorted(set(exp) & set(sim))
    rate_offset = RATES.index(rate) * len(req_ids)
    return CaseRows(
        split=split,
        model=model,
        rate=rate,
        req_ids=np.array(req_ids, dtype=int),
        global_request_index=np.array([rate_offset + idx for idx in req_ids], dtype=float),
        raw_ftl_ms=np.array([sim[idx][0] for idx in req_ids], dtype=float),
        raw_tpot_ms=np.array([sim[idx][1] for idx in req_ids], dtype=float),
        real_ftl_ms=np.array([exp[idx][0] for idx in req_ids], dtype=float),
        real_tpot_ms=np.array([exp[idx][1] for idx in req_ids], dtype=float),
    )


def predict_case(case: CaseRows, config: dict) -> np.ndarray:
    alpha = float(config["alpha"])
    beta_ms = float(config["beta_ms"])
    warmup_amp_ms = float(config["warmup_amp_ms"])
    warmup_tau_requests = float(config["warmup_tau_requests"])
    return np.maximum(
        0.0,
        alpha * case.raw_ftl_ms
        + beta_ms
        + warmup_amp_ms * np.exp(-case.global_request_index / warmup_tau_requests),
    )


def metric_deltas_for_case(
    real_ftl: np.ndarray,
    real_tpot: np.ndarray,
    pred_ftl: np.ndarray,
    pred_tpot: np.ndarray,
) -> dict[tuple[float, str], tuple[float, float, float]]:
    output = {}
    count = len(real_ftl)
    for scale in SLO_SCALES:
        ftl_target = 200.0 * scale
        tpot_target = 100.0 * scale
        real_prefill = 100.0 * np.mean(real_ftl <= ftl_target)
        pred_prefill = 100.0 * np.mean(pred_ftl <= ftl_target)
        real_decode = 100.0 * np.mean(real_tpot <= tpot_target)
        pred_decode = 100.0 * np.mean(pred_tpot <= tpot_target)
        real_both = 100.0 * np.count_nonzero(
            (real_ftl <= ftl_target) & (real_tpot <= tpot_target)
        ) / count
        pred_both = 100.0 * np.count_nonzero(
            (pred_ftl <= ftl_target) & (pred_tpot <= tpot_target)
        ) / count
        output[(scale, "prefill")] = (real_prefill, pred_prefill, abs(real_prefill - pred_prefill))
        output[(scale, "decode")] = (real_decode, pred_decode, abs(real_decode - pred_decode))
        output[(scale, "both")] = (real_both, pred_both, abs(real_both - pred_both))
    return output


def candidate_params(model: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + MODELS.index(model))
    params = []
    for alpha in np.arange(0.5, 1.51, 0.02):
        for beta_ms in np.arange(-80.0, 101.0, 4.0):
            params.append((alpha, beta_ms, 0.0, 60.0))

    random_trials = 50000
    max_warmup = 800.0 if model == "llama_8B" else 300.0
    tau_choices = np.array([8, 12, 16, 24, 32, 45, 60, 90, 120, 180, 240], dtype=float)
    random_params = np.column_stack(
        [
            rng.uniform(0.5, 1.5, random_trials),
            rng.uniform(-120.0, 140.0, random_trials),
            rng.uniform(0.0, max_warmup, random_trials),
            rng.choice(tau_choices, random_trials),
        ]
    )
    return np.vstack([np.array(params, dtype=float), random_params])


def score_params(cases: list[CaseRows], params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    max_delta = np.zeros(len(params), dtype=float)
    sum_delta = np.zeros(len(params), dtype=float)
    for case in cases:
        for start in range(0, len(params), 4000):
            chunk = params[start : start + 4000]
            pred = np.maximum(
                0.0,
                chunk[:, 0, None] * case.raw_ftl_ms[None, :]
                + chunk[:, 1, None]
                + chunk[:, 2, None]
                * np.exp(-case.global_request_index[None, :] / chunk[:, 3, None]),
            )
            deltas = []
            n = len(case.req_ids)
            for scale in SLO_SCALES:
                ftl_target = 200.0 * scale
                tpot_target = 100.0 * scale
                real_prefill = 100.0 * np.mean(case.real_ftl_ms <= ftl_target)
                pred_prefill = 100.0 * np.mean(pred <= ftl_target, axis=1)
                deltas.append(np.abs(pred_prefill - real_prefill))

                real_both = 100.0 * np.count_nonzero(
                    (case.real_ftl_ms <= ftl_target) & (case.real_tpot_ms <= tpot_target)
                ) / n
                pred_both = 100.0 * np.count_nonzero(
                    (pred <= ftl_target) & (case.raw_tpot_ms[None, :] <= tpot_target),
                    axis=1,
                ) / n
                deltas.append(np.abs(pred_both - real_both))

            delta_matrix = np.vstack(deltas)
            max_delta[start : start + len(chunk)] = np.maximum(
                max_delta[start : start + len(chunk)], delta_matrix.max(axis=0)
            )
            sum_delta[start : start + len(chunk)] += delta_matrix.sum(axis=0)
        count += len(SLO_SCALES) * 2
    return max_delta, sum_delta / count


def fit_model(model: str, cases: list[CaseRows], seed: int) -> dict:
    params = candidate_params(model, seed)
    max_delta, mean_delta = score_params(cases, params)
    best_idx = np.lexsort((mean_delta, max_delta))[0]
    return {
        "alpha": float(params[best_idx, 0]),
        "beta_ms": float(params[best_idx, 1]),
        "warmup_amp_ms": float(params[best_idx, 2]),
        "warmup_tau_requests": float(params[best_idx, 3]),
        "fit_max_prefill_or_both_delta_pct": float(max_delta[best_idx]),
        "fit_mean_prefill_or_both_delta_pct": float(mean_delta[best_idx]),
        "formula": (
            "ftl_ms = max(0, alpha * raw_3p3d_ftl_ms + beta_ms + "
            "warmup_amp_ms * exp(-global_request_index / warmup_tau_requests))"
        ),
    }


def write_eval_csv(path: Path, split: str, cases: dict[tuple[str, str], CaseRows], configs: dict[str, dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "model",
        "rate",
        "slo_scale",
        "metric",
        "real_attainment_pct",
        "raw_attainment_pct",
        "modeled_attainment_pct",
        "raw_delta_pct",
        "modeled_delta_pct",
    ]
    max_raw = 0.0
    max_modeled = 0.0
    within_5 = 0
    total = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, rate), case in sorted(cases.items(), key=lambda item: (item[0][0], float(item[0][1]))):
            pred_ftl = predict_case(case, configs[model])
            raw = metric_deltas_for_case(case.real_ftl_ms, case.real_tpot_ms, case.raw_ftl_ms, case.raw_tpot_ms)
            modeled = metric_deltas_for_case(case.real_ftl_ms, case.real_tpot_ms, pred_ftl, case.raw_tpot_ms)
            for scale in SLO_SCALES:
                for metric in ("prefill", "decode", "both"):
                    real_value, raw_value, raw_delta = raw[(scale, metric)]
                    _, modeled_value, modeled_delta = modeled[(scale, metric)]
                    max_raw = max(max_raw, raw_delta)
                    max_modeled = max(max_modeled, modeled_delta)
                    within_5 += modeled_delta <= 5.0
                    total += 1
                    writer.writerow(
                        {
                            "split": split,
                            "model": model,
                            "rate": rate,
                            "slo_scale": scale,
                            "metric": metric,
                            "real_attainment_pct": f"{real_value:.6f}",
                            "raw_attainment_pct": f"{raw_value:.6f}",
                            "modeled_attainment_pct": f"{modeled_value:.6f}",
                            "raw_delta_pct": f"{raw_delta:.6f}",
                            "modeled_delta_pct": f"{modeled_delta:.6f}",
                        }
                    )
    return {
        "max_raw_delta_pct": float(max_raw),
        "max_modeled_delta_pct": float(max_modeled),
        "cases": int(total),
        "modeled_cases_within_5pct": int(within_5),
        "modeled_fraction_within_5pct": float(within_5 / total) if total else math.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("/users/rh/cuda_data"))
    parser.add_argument("--sim-root", type=Path, default=Path("/users/rh/cuda_data/sim/3p3d_fit_model_forward"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/users/rh/cuda_data/sim/3p3d_fit_model_forward/warmup_ftl_eval"),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_cases: dict[tuple[str, str], CaseRows] = {}
    val_cases: dict[tuple[str, str], CaseRows] = {}
    for split, target in (("fit", fit_cases), ("val", val_cases)):
        for model in MODELS:
            for rate in RATES:
                case = load_case(split, model, rate, args.exp_root, args.sim_root)
                if case is not None:
                    target[(model, rate)] = case

    configs = {}
    for model in MODELS:
        train_cases = [case for (case_model, _), case in fit_cases.items() if case_model == model]
        configs[model] = fit_model(model, train_cases, args.seed)
        print(f"{model}: {json.dumps(configs[model])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "warmup_ftl_model.json").write_text(
        json.dumps(
            {
                "train_split": "fit",
                "test_split": "val",
                "raw_simulator": str(args.sim_root),
                "rate_handling": "request_rate is not a fitted feature; run order only defines global_request_index",
                "tpot": "raw simulator output",
                "models": configs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fit_summary = write_eval_csv(args.output_dir / "fit_slo_delta.csv", "fit", fit_cases, configs)
    val_summary = write_eval_csv(args.output_dir / "val_slo_delta.csv", "val", val_cases, configs)
    summary = {"fit": fit_summary, "val": val_summary}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

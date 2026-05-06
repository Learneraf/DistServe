#!/usr/bin/env python3
"""Fit split-only latency residuals and evaluate SLO deltas.

This script deliberately keeps the train/test split explicit:

* fit split: learn per-model residual models for FTL and TPOT
* val split: apply the learned residuals without looking at val targets

The input simulation files are the request-level CSVs emitted by
simulate_dist.py.  The real files are the vLLM P/D benchmark .exp JSONs.
All latency units in this script are milliseconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


MODELS = ["llama_1B", "llama_3B", "llama_8B"]
RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
SLO_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2]

REQUEST_INDEX_RE = re.compile(r"-bench-(\d+)")


@dataclass(frozen=True)
class RequestRow:
    model: str
    rate: float
    req_index: int
    prompt_len: float
    output_len: float
    sim_ftl_ms: float
    sim_tpot_ms: float
    real_ftl_ms: float
    real_tpot_ms: float


def load_workload(path: Path, limit: int) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            rows[idx] = (float(item["prompt_len"]), float(item["output_len"]))
    return rows


def request_index_from_exp(item: dict, fallback: int) -> int:
    for key in ("client_request_id", "vllm_internal_request_id"):
        value = item.get(key)
        if not value:
            continue
        match = REQUEST_INDEX_RE.search(str(value))
        if match:
            return int(match.group(1))
    return fallback


def load_exp(path: Path) -> dict[int, tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[int, tuple[float, float]] = {}
    for fallback, item in enumerate(data):
        idx = request_index_from_exp(item, fallback)
        rows[idx] = (1000.0 * float(item["ftl"]), 1000.0 * float(item["tpot"]))
    return rows


def load_sim(path: Path) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["req_id"])
            rows[idx] = (float(row["first_token_latency"]), float(row["tpot"]))
    return rows


def load_joined_rows(
    split: str,
    model: str,
    rate: str,
    exp_root: Path,
    sim_root: Path,
    workload_base: Path,
) -> list[RequestRow]:
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
        return []
    dataset_dir = {
        "llama_1B": "llama-3.2-1B",
        "llama_3B": "llama-3.2-3B",
        "llama_8B": "llama-3.1-8B",
    }[model]
    workload = load_workload(workload_base / dataset_dir / f"{split}.jsonl", 120)
    exp = load_exp(exp_path)
    sim = load_sim(sim_path)
    joined = []
    for idx in sorted(set(exp) & set(sim) & set(workload)):
        prompt_len, output_len = workload[idx]
        sim_ftl, sim_tpot = sim[idx]
        real_ftl, real_tpot = exp[idx]
        joined.append(
            RequestRow(
                model=model,
                rate=float(rate),
                req_index=idx,
                prompt_len=prompt_len,
                output_len=output_len,
                sim_ftl_ms=sim_ftl,
                sim_tpot_ms=sim_tpot,
                real_ftl_ms=real_ftl,
                real_tpot_ms=real_tpot,
            )
        )
    return joined


def feature_vector(row: RequestRow, metric: str) -> list[float]:
    # Keep this intentionally small and interpretable.  No request id or split
    # dependent feature is used.
    if metric == "ftl":
        raw = row.sim_ftl_ms
    elif metric == "tpot":
        raw = row.sim_tpot_ms
    else:
        raise ValueError(metric)
    return [
        1.0,
        raw,
        math.log1p(row.prompt_len),
        math.log1p(row.output_len),
        row.rate,
    ]


def fit_ridge(rows: list[RequestRow], metric: str, ridge: float) -> list[float]:
    x = np.array([feature_vector(row, metric) for row in rows], dtype=float)
    if metric == "ftl":
        y = np.array([row.real_ftl_ms for row in rows], dtype=float)
    else:
        y = np.array([row.real_tpot_ms for row in rows], dtype=float)
    penalty = np.eye(x.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    coeffs = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return [float(value) for value in coeffs]


def predict(row: RequestRow, coeffs: list[float], metric: str) -> float:
    return max(0.0, float(np.dot(np.array(feature_vector(row, metric)), np.array(coeffs))))


def attainment(rows: Iterable[tuple[float, float]], scale: float) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"prefill": float("nan"), "decode": float("nan"), "both": float("nan")}
    ftl_target = 200.0 * scale
    tpot_target = 100.0 * scale
    prefill = sum(ftl <= ftl_target for ftl, _ in rows)
    decode = sum(tpot <= tpot_target for _, tpot in rows)
    both = sum(ftl <= ftl_target and tpot <= tpot_target for ftl, tpot in rows)
    total = len(rows)
    return {
        "prefill": 100.0 * prefill / total,
        "decode": 100.0 * decode / total,
        "both": 100.0 * both / total,
    }


def write_delta_rows(
    path: Path,
    split: str,
    rows_by_case: dict[tuple[str, str], list[RequestRow]],
    coeffs_by_model: dict[str, dict[str, list[float]]],
) -> dict[str, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "model",
        "rate",
        "slo_scale",
        "metric",
        "real_attainment_pct",
        "sim_raw_attainment_pct",
        "sim_calibrated_attainment_pct",
        "raw_delta_pct",
        "calibrated_delta_pct",
    ]
    max_raw = 0.0
    max_cal = 0.0
    count_cal_le_5 = 0
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, rate), rows in sorted(rows_by_case.items(), key=lambda item: (item[0][0], float(item[0][1]))):
            coeffs = coeffs_by_model[model]
            real_pairs = [(row.real_ftl_ms, row.real_tpot_ms) for row in rows]
            raw_pairs = [(row.sim_ftl_ms, row.sim_tpot_ms) for row in rows]
            cal_pairs = [
                (
                    predict(row, coeffs["ftl"], "ftl"),
                    predict(row, coeffs["tpot"], "tpot"),
                )
                for row in rows
            ]
            for scale in SLO_SCALES:
                real = attainment(real_pairs, scale)
                raw = attainment(raw_pairs, scale)
                cal = attainment(cal_pairs, scale)
                for metric in ("prefill", "decode", "both"):
                    raw_delta = abs(raw[metric] - real[metric])
                    cal_delta = abs(cal[metric] - real[metric])
                    max_raw = max(max_raw, raw_delta)
                    max_cal = max(max_cal, cal_delta)
                    count += 1
                    if cal_delta <= 5.0:
                        count_cal_le_5 += 1
                    writer.writerow(
                        {
                            "split": split,
                            "model": model,
                            "rate": rate,
                            "slo_scale": scale,
                            "metric": metric,
                            "real_attainment_pct": f"{real[metric]:.6f}",
                            "sim_raw_attainment_pct": f"{raw[metric]:.6f}",
                            "sim_calibrated_attainment_pct": f"{cal[metric]:.6f}",
                            "raw_delta_pct": f"{raw_delta:.6f}",
                            "calibrated_delta_pct": f"{cal_delta:.6f}",
                        }
                    )
    return {
        "max_raw_delta_pct": max_raw,
        "max_calibrated_delta_pct": max_cal,
        "cases": count,
        "calibrated_cases_within_5pct": count_cal_le_5,
        "calibrated_fraction_within_5pct": count_cal_le_5 / count if count else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("/users/rh/cuda_data"))
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument(
        "--workload-base",
        type=Path,
        default=Path(
            "/users/rh/DistServe/simdistserve/dataset/splits/"
            "sharegpt_four_models_common_ascend1900_seed0"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    args = parser.parse_args()

    fit_by_case: dict[tuple[str, str], list[RequestRow]] = {}
    val_by_case: dict[tuple[str, str], list[RequestRow]] = {}
    for split, target in (("fit", fit_by_case), ("val", val_by_case)):
        for model in MODELS:
            for rate in RATES:
                rows = load_joined_rows(
                    split, model, rate, args.exp_root, args.sim_root, args.workload_base
                )
                if rows:
                    target[(model, rate)] = rows

    coeffs_by_model: dict[str, dict[str, list[float]]] = {}
    for model in MODELS:
        train_rows = [
            row
            for (case_model, _), rows in fit_by_case.items()
            if case_model == model
            for row in rows
        ]
        coeffs_by_model[model] = {
            "ftl": fit_ridge(train_rows, "ftl", args.ridge),
            "tpot": fit_ridge(train_rows, "tpot", args.ridge),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "residual_model.json").write_text(
        json.dumps(
            {
                "features": ["constant", "raw_metric_ms", "log1p_prompt_len", "log1p_output_len", "request_rate"],
                "models": coeffs_by_model,
                "train_split": "fit",
                "test_split": "val",
                "slo_scales": SLO_SCALES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fit_summary = write_delta_rows(
        args.output_dir / "fit_slo_delta.csv", "fit", fit_by_case, coeffs_by_model
    )
    val_summary = write_delta_rows(
        args.output_dir / "val_slo_delta.csv", "val", val_by_case, coeffs_by_model
    )
    summary = {"fit": fit_summary, "val": val_summary}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

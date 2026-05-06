#!/usr/bin/env python3
"""Fit a request-level FTL path model on top of 3p3d simulation output.

The simulator's prefill/decode execution profile is left unchanged.  This
script only adjusts first-token latency with an additive, interpretable
overhead model:

    ftl_ms = raw_sim_ftl_ms + max(0, feature_model(request))

The model is fitted on the fit split with an SLO-attainment objective and then
evaluated unchanged on the val split.  TPOT is always the raw simulator TPOT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


MODELS = ["llama_1B", "llama_3B", "llama_8B"]
RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
SLO_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2]
REQUEST_INDEX_RE = re.compile(r"-bench-(\d+)")
DATASET_DIRS = {
    "llama_1B": "llama-3.2-1B",
    "llama_3B": "llama-3.2-3B",
    "llama_8B": "llama-3.1-8B",
}


@dataclass(frozen=True)
class CaseRows:
    split: str
    model: str
    rate: str
    req_ids: np.ndarray
    prompt_len: np.ndarray
    prompt_blocks: np.ndarray
    output_len: np.ndarray
    raw_ftl_ms: np.ndarray
    raw_tpot_ms: np.ndarray
    real_ftl_ms: np.ndarray
    real_tpot_ms: np.ndarray


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    features: tuple[str, ...]
    ranges: tuple[tuple[float, float], ...]
    random_trials: int


FEATURE_SPECS = {
    "constant": FeatureSpec(
        "constant",
        ("constant",),
        ((-20.0, 120.0),),
        2000,
    ),
    "blocks": FeatureSpec(
        "blocks",
        ("constant", "prompt_blocks"),
        ((-20.0, 140.0), (-2.0, 2.0)),
        30000,
    ),
    "blocks_rate": FeatureSpec(
        "blocks_rate",
        ("constant", "prompt_blocks", "request_rate"),
        ((-20.0, 160.0), (-2.0, 2.0), (-30.0, 30.0)),
        50000,
    ),
    "blocks_output_rate": FeatureSpec(
        "blocks_output_rate",
        ("constant", "prompt_blocks", "output_len", "request_rate"),
        ((-20.0, 160.0), (-2.0, 2.0), (-0.10, 0.10), (-30.0, 30.0)),
        70000,
    ),
}


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


def load_workload(path: Path, limit: int = 120) -> dict[int, tuple[float, float]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            rows[idx] = (float(item["prompt_len"]), float(item["output_len"]))
    return rows


def load_case(
    split: str,
    model: str,
    rate: str,
    exp_root: Path,
    sim_root: Path,
    workload_base: Path,
) -> CaseRows | None:
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
    workload_path = workload_base / DATASET_DIRS[model] / f"{split}.jsonl"
    if not exp_path.exists() or not sim_path.exists() or not workload_path.exists():
        return None

    exp = load_exp(exp_path)
    sim = load_sim(sim_path)
    workload = load_workload(workload_path)
    req_ids = sorted(set(exp) & set(sim) & set(workload))
    prompt_len = np.array([workload[idx][0] for idx in req_ids], dtype=float)
    output_len = np.array([workload[idx][1] for idx in req_ids], dtype=float)
    return CaseRows(
        split=split,
        model=model,
        rate=rate,
        req_ids=np.array(req_ids, dtype=int),
        prompt_len=prompt_len,
        prompt_blocks=np.ceil(prompt_len / 16.0),
        output_len=output_len,
        raw_ftl_ms=np.array([sim[idx][0] for idx in req_ids], dtype=float),
        raw_tpot_ms=np.array([sim[idx][1] for idx in req_ids], dtype=float),
        real_ftl_ms=np.array([exp[idx][0] for idx in req_ids], dtype=float),
        real_tpot_ms=np.array([exp[idx][1] for idx in req_ids], dtype=float),
    )


def feature_matrix(cases: list[CaseRows], features: tuple[str, ...]) -> np.ndarray:
    columns = []
    for case in cases:
        request_rate = np.full_like(case.prompt_len, float(case.rate), dtype=float)
        values = {
            "constant": np.ones_like(case.prompt_len, dtype=float),
            "prompt_blocks": case.prompt_blocks,
            "prompt_len": case.prompt_len,
            "output_len": case.output_len,
            "request_rate": request_rate,
            "raw_ftl_ms": case.raw_ftl_ms,
        }
        columns.append(np.column_stack([values[name] for name in features]))
    return np.vstack(columns)


def concatenate(cases: list[CaseRows], attr: str) -> np.ndarray:
    return np.concatenate([getattr(case, attr) for case in cases])


def candidate_params(spec: FeatureSpec, seed: int) -> list[tuple[float, ...]]:
    rng = random.Random(seed)
    params: list[tuple[float, ...]] = []
    dims = len(spec.features)
    for intercept in np.arange(-20.0, 120.1, 2.0):
        params.append((float(intercept),) + tuple(0.0 for _ in range(dims - 1)))
    for _ in range(spec.random_trials):
        params.append(tuple(rng.uniform(low, high) for low, high in spec.ranges))
    return params


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
        output[(scale, "prefill")] = (
            real_prefill,
            pred_prefill,
            abs(real_prefill - pred_prefill),
        )
        output[(scale, "decode")] = (
            real_decode,
            pred_decode,
            abs(real_decode - pred_decode),
        )
        output[(scale, "both")] = (
            real_both,
            pred_both,
            abs(real_both - pred_both),
        )
    return output


def score_params(cases: list[CaseRows], matrix: np.ndarray, params: tuple[float, ...]) -> tuple[float, float]:
    overhead = np.maximum(0.0, matrix @ np.array(params, dtype=float))
    raw_ftl = concatenate(cases, "raw_ftl_ms")
    pred_ftl = raw_ftl + overhead
    offset = 0
    deltas = []
    for case in cases:
        n = len(case.req_ids)
        case_pred_ftl = pred_ftl[offset : offset + n]
        offset += n
        for (_, metric), (_, _, delta) in metric_deltas_for_case(
            case.real_ftl_ms,
            case.real_tpot_ms,
            case_pred_ftl,
            case.raw_tpot_ms,
        ).items():
            if metric in ("prefill", "both"):
                deltas.append(delta)
    return max(deltas), sum(deltas) / len(deltas)


def fit_model(cases: list[CaseRows], spec: FeatureSpec, seed: int) -> dict:
    matrix = feature_matrix(cases, spec.features)
    best: tuple[float, float, tuple[float, ...]] | None = None
    for params in candidate_params(spec, seed):
        max_delta, mean_delta = score_params(cases, matrix, params)
        candidate = (max_delta, mean_delta, params)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return {
        "features": list(spec.features),
        "coeffs": [float(value) for value in best[2]],
        "fit_max_prefill_or_both_delta_pct": float(best[0]),
        "fit_mean_prefill_or_both_delta_pct": float(best[1]),
        "formula": "ftl_ms = raw_sim_ftl_ms + max(0, dot(features, coeffs))",
    }


def predict_case(case: CaseRows, model_config: dict) -> np.ndarray:
    matrix = feature_matrix([case], tuple(model_config["features"]))
    overhead = np.maximum(0.0, matrix @ np.array(model_config["coeffs"], dtype=float))
    return case.raw_ftl_ms + overhead


def write_eval_csv(
    path: Path,
    split: str,
    cases: dict[tuple[str, str], CaseRows],
    model_configs: dict[str, dict],
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
            pred_ftl = predict_case(case, model_configs[model])
            raw = metric_deltas_for_case(
                case.real_ftl_ms,
                case.real_tpot_ms,
                case.raw_ftl_ms,
                case.raw_tpot_ms,
            )
            modeled = metric_deltas_for_case(
                case.real_ftl_ms,
                case.real_tpot_ms,
                pred_ftl,
                case.raw_tpot_ms,
            )
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
        "modeled_fraction_within_5pct": float(within_5 / total) if total else float("nan"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", type=Path, default=Path("/users/rh/cuda_data"))
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path("/users/rh/cuda_data/sim/3p3d_fit_model_forward"),
    )
    parser.add_argument(
        "--workload-base",
        type=Path,
        default=Path(
            "/users/rh/DistServe/simdistserve/dataset/splits/"
            "sharegpt_four_models_common_ascend1900_seed0"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/users/rh/cuda_data/sim/3p3d_fit_model_forward/ftl_path_model_eval"),
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SPECS),
        default="blocks_rate",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = FEATURE_SPECS[args.feature_set]
    fit_cases: dict[tuple[str, str], CaseRows] = {}
    val_cases: dict[tuple[str, str], CaseRows] = {}
    for split, target in (("fit", fit_cases), ("val", val_cases)):
        for model in MODELS:
            for rate in RATES:
                case = load_case(
                    split,
                    model,
                    rate,
                    args.exp_root,
                    args.sim_root,
                    args.workload_base,
                )
                if case is not None:
                    target[(model, rate)] = case

    model_configs = {}
    for model in MODELS:
        train_cases = [
            case
            for (case_model, _), case in fit_cases.items()
            if case_model == model
        ]
        model_configs[model] = fit_model(train_cases, spec, args.seed)
        print(f"{model}: {json.dumps(model_configs[model])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ftl_path_model.json").write_text(
        json.dumps(
            {
                "train_split": "fit",
                "test_split": "val",
                "feature_set": args.feature_set,
                "raw_simulator": str(args.sim_root),
                "tpot": "raw simulator output",
                "models": model_configs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fit_summary = write_eval_csv(
        args.output_dir / "fit_slo_delta.csv", "fit", fit_cases, model_configs
    )
    val_summary = write_eval_csv(
        args.output_dir / "val_slo_delta.csv", "val", val_cases, model_configs
    )
    summary = {"fit": fit_summary, "val": val_summary}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

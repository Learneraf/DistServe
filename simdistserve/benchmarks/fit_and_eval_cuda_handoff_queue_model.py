#!/usr/bin/env python3
"""Fit a mechanistic P/D handoff queue on top of raw 3p3d output.

The raw 3p3d simulator output is treated as the prefill/decode execution
model.  This script only changes first-token visibility: after simulated
prefill finishes, a request must pass through a P->D handoff resource whose
service time is

    service_ms = base_ms + per_token_ms * prompt_len

with a small integer capacity.  This represents KV save/load and decode-side
first-token readiness without using request_rate as a fitted feature.  The
request rate affects the result only through the simulated finish_prefill
timeline and the handoff queue.
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
    finish_prefill_ms: np.ndarray
    prompt_len: np.ndarray
    output_len: np.ndarray
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


def load_sim_latency(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            int(row["req_id"]): (
                float(row["first_token_latency"]),
                float(row["tpot"]),
            )
            for row in csv.DictReader(f)
        }


def load_request_info(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {
            int(row["req_id"]): (float(row["prefill_lens"]), float(row["output_lens"]))
            for row in csv.DictReader(f)
        }


def load_finish_prefill(path: Path) -> dict[int, float]:
    finish = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["event_type"] == "finish_prefill":
                finish[int(row["req_id"])] = float(row["start_time"])
    return finish


def load_case(split: str, model: str, rate: str, exp_root: Path, sim_root: Path) -> CaseRows | None:
    exp_path = exp_root / split / model / f"vllm-pd-120-{rate}.exp"
    sim_dir = sim_root / split / "vllm_ascend" / "organized_data" / model / f"rate_{rate}"
    latency_path = sim_dir / "request_latency.csv"
    info_path = sim_dir / "request_info.csv"
    event_path = sim_dir / "request_event.csv"
    if not all(path.exists() for path in (exp_path, latency_path, info_path, event_path)):
        return None

    exp = load_exp(exp_path)
    sim = load_sim_latency(latency_path)
    info = load_request_info(info_path)
    finish = load_finish_prefill(event_path)
    req_ids = sorted(set(exp) & set(sim) & set(info) & set(finish))
    return CaseRows(
        split=split,
        model=model,
        rate=rate,
        req_ids=np.array(req_ids, dtype=int),
        finish_prefill_ms=np.array([finish[idx] for idx in req_ids], dtype=float),
        prompt_len=np.array([info[idx][0] for idx in req_ids], dtype=float),
        output_len=np.array([info[idx][1] for idx in req_ids], dtype=float),
        raw_ftl_ms=np.array([sim[idx][0] for idx in req_ids], dtype=float),
        raw_tpot_ms=np.array([sim[idx][1] for idx in req_ids], dtype=float),
        real_ftl_ms=np.array([exp[idx][0] for idx in req_ids], dtype=float),
        real_tpot_ms=np.array([exp[idx][1] for idx in req_ids], dtype=float),
    )


def simulate_handoff_delay(
    finish_prefill_ms: np.ndarray,
    prompt_len: np.ndarray,
    base_ms: float,
    per_token_ms: float,
    capacity: int,
) -> np.ndarray:
    order = np.argsort(finish_prefill_ms, kind="stable")
    available = np.zeros(capacity, dtype=float)
    delay = np.zeros(len(finish_prefill_ms), dtype=float)
    for idx in order:
        slot = int(np.argmin(available))
        start = max(finish_prefill_ms[idx], available[slot])
        service = max(0.0, base_ms + per_token_ms * prompt_len[idx])
        end = start + service
        available[slot] = end
        delay[idx] = end - finish_prefill_ms[idx]
    return delay


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


def predict_case(case: CaseRows, config: dict) -> np.ndarray:
    delay = simulate_handoff_delay(
        case.finish_prefill_ms,
        case.prompt_len,
        float(config["base_ms"]),
        float(config["per_token_ms"]),
        int(config["capacity"]),
    )
    return case.raw_ftl_ms + delay


def score_config(cases: list[CaseRows], config: dict) -> tuple[float, float]:
    deltas = []
    for case in cases:
        pred_ftl = predict_case(case, config)
        for (_, metric), (_, _, delta) in metric_deltas_for_case(
            case.real_ftl_ms,
            case.real_tpot_ms,
            pred_ftl,
            case.raw_tpot_ms,
        ).items():
            if metric in ("prefill", "both"):
                deltas.append(delta)
    return max(deltas), sum(deltas) / len(deltas)


def candidate_configs() -> list[dict]:
    configs = []
    # Include the no-handoff baseline.
    configs.append({"base_ms": 0.0, "per_token_ms": 0.0, "capacity": 1})
    for capacity in (1, 2, 4, 8):
        for base_ms in np.arange(0.0, 121.0, 2.0):
            configs.append({"base_ms": float(base_ms), "per_token_ms": 0.0, "capacity": capacity})
        for base_ms in np.arange(0.0, 81.0, 4.0):
            for per_token_ms in np.arange(0.0, 0.081, 0.004):
                configs.append(
                    {
                        "base_ms": float(base_ms),
                        "per_token_ms": float(per_token_ms),
                        "capacity": capacity,
                    }
                )
    return configs


def fit_model(cases: list[CaseRows]) -> dict:
    best: tuple[float, float, dict] | None = None
    for config in candidate_configs():
        max_delta, mean_delta = score_config(cases, config)
        candidate = (max_delta, mean_delta, config)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    config = dict(best[2])
    config.update(
        {
            "fit_max_prefill_or_both_delta_pct": float(best[0]),
            "fit_mean_prefill_or_both_delta_pct": float(best[1]),
            "formula": "ftl_ms = raw_3p3d_ftl_ms + queue_delay(base_ms + per_token_ms * prompt_len, capacity)",
        }
    )
    return config


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
        default=Path("/users/rh/cuda_data/sim/3p3d_fit_model_forward/handoff_queue_eval"),
    )
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
        configs[model] = fit_model(train_cases)
        print(f"{model}: {json.dumps(configs[model])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "handoff_queue_model.json").write_text(
        json.dumps(
            {
                "train_split": "fit",
                "test_split": "val",
                "raw_simulator": str(args.sim_root),
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

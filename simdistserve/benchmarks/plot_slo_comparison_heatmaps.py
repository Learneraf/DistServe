#!/usr/bin/env python3
"""Plot heatmaps from merged_analyze comparison.txt files.

The parser expects files like:

    results/slo/<backend>/compared/<model>/rate_<rate>/scale_<scale>/comparison.txt

and reads the signed "Difference (Exp - CSV)" percentage for each SLO metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("prefill", "Prefill SLO met rate", "Prefill"),
    ("decode", "Decode SLO met rate", "Decode"),
    ("total", "Total SLO met rate", "Total"),
    ("both", "Both (Prefill+Decode) SLO met rate", "Both"),
]
METRIC_BY_LABEL = {label: (key, short) for key, label, short in METRICS}
PATH_RE = re.compile(
    r"compared/(?P<model>[^/]+)/rate_(?P<rate>[^/]+)/scale_(?P<scale>[^/]+)/comparison\.txt$"
)
DIFF_RE = re.compile(r"(?P<metric>.+?)\s+\d+/\d+\s+\([^)]+\)\s+\d+/\d+\s+\([^)]+\)\s+(?P<diff>[+-]?\d+(?:\.\d+)?)%")


@dataclass(frozen=True)
class Row:
    backend: str
    model: str
    rate: float
    slo_scale: float
    metric: str
    delta_pct: float


def parse_float_tag(value: str) -> float:
    return float(value.replace("p", "."))


def rate_label(value: float) -> str:
    return f"{value:g}"


def scale_label(value: float) -> str:
    return f"{value:g}"


def parse_comparison(path: Path, backend: str, root: Path) -> list[Row]:
    rel = path.relative_to(root).as_posix()
    match = PATH_RE.search(rel)
    if not match:
        return []

    model = match.group("model")
    rate = parse_float_tag(match.group("rate"))
    scale = parse_float_tag(match.group("scale").removeprefix("scale_"))
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        diff_match = DIFF_RE.match(line)
        if not diff_match:
            continue
        metric_label = diff_match.group("metric").strip()
        metric_info = METRIC_BY_LABEL.get(metric_label)
        if metric_info is None:
            continue
        metric_key, _ = metric_info
        rows.append(
            Row(
                backend=backend,
                model=model,
                rate=rate,
                slo_scale=scale,
                metric=metric_key,
                delta_pct=float(diff_match.group("diff")),
            )
        )
    return rows


def collect_rows(slo_root: Path, backend: str) -> list[Row]:
    backend_root = slo_root / backend
    rows: list[Row] = []
    for path in sorted((backend_root / "compared").glob("*/rate_*/scale_*/comparison.txt")):
        rows.extend(parse_comparison(path, backend, backend_root))
    return rows


def write_rows_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["backend", "model", "rate", "slo_scale", "metric", "delta_pct", "abs_delta_pct"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.backend, r.model, r.rate, r.slo_scale, r.metric)):
            writer.writerow(
                {
                    "backend": row.backend,
                    "model": row.model,
                    "rate": f"{row.rate:g}",
                    "slo_scale": f"{row.slo_scale:g}",
                    "metric": row.metric,
                    "delta_pct": f"{row.delta_pct:.6f}",
                    "abs_delta_pct": f"{abs(row.delta_pct):.6f}",
                }
            )


def matrix_for_metric(rows: list[Row], model: str, metric: str) -> tuple[list[float], list[float], np.ndarray]:
    subset = [row for row in rows if row.model == model and row.metric == metric]
    rates = sorted({row.rate for row in subset})
    scales = sorted({row.slo_scale for row in subset})
    values = np.full((len(rates), len(scales)), np.nan)
    index = {(row.rate, row.slo_scale): row.delta_pct for row in subset}
    for i, rate in enumerate(rates):
        for j, scale in enumerate(scales):
            if (rate, scale) in index:
                values[i, j] = index[(rate, scale)]
    return rates, scales, values


def annotate_heatmap(ax, values: np.ndarray, fmt: str = ".1f") -> None:
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if math.isnan(value):
                continue
            color = "white" if abs(value) >= 35 else "#1f2933"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color=color)


def plot_signed_heatmap(
    rows: list[Row],
    backend: str,
    model: str,
    metric: str,
    metric_title: str,
    output_path: Path,
) -> None:
    rates, scales, values = matrix_for_metric(rows, model, metric)
    if values.size == 0:
        return

    finite = values[np.isfinite(values)]
    limit = max(5.0, float(np.max(np.abs(finite))) if finite.size else 5.0)
    fig_width = max(7.0, 0.9 * len(scales) + 2.8)
    fig_height = max(4.8, 0.55 * len(rates) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_title(f"{backend}: {model} {metric_title} SLO Delta", fontsize=13, weight="bold")
    ax.set_xlabel("SLO scale")
    ax.set_ylabel("Request rate")
    ax.set_xticks(range(len(scales)), [scale_label(scale) for scale in scales])
    ax.set_yticks(range(len(rates)), [rate_label(rate) for rate in rates])
    annotate_heatmap(ax, values)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("EXP - simdistserve (percentage points)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_heatmap(
    rows: list[Row],
    backend: str,
    metric: str,
    metric_title: str,
    output_path: Path,
    reducer: str,
) -> None:
    subset = [row for row in rows if row.metric == metric]
    models = sorted({row.model for row in subset})
    rates = sorted({row.rate for row in subset})
    values = np.full((len(models), len(rates)), np.nan)
    by_cell: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in subset:
        by_cell[(row.model, row.rate)].append(abs(row.delta_pct))
    for i, model in enumerate(models):
        for j, rate in enumerate(rates):
            cell = by_cell.get((model, rate), [])
            if cell:
                if reducer == "max":
                    values[i, j] = max(cell)
                elif reducer == "mean":
                    values[i, j] = float(np.mean(cell))
                else:
                    raise ValueError(f"Unsupported reducer: {reducer}")

    if values.size == 0:
        return
    finite = values[np.isfinite(values)]
    limit = max(5.0, float(np.max(finite)) if finite.size else 5.0)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(rates) + 2.5), 4.8), constrained_layout=True)
    image = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=limit, aspect="auto")
    reducer_title = "Max" if reducer == "max" else "Mean"
    ax.set_title(f"{backend}: {reducer_title} |{metric_title} SLO Delta| Across Scales", fontsize=13, weight="bold")
    ax.set_xlabel("Request rate")
    ax.set_ylabel("Model")
    ax.set_xticks(range(len(rates)), [rate_label(rate) for rate in rates])
    ax.set_yticks(range(len(models)), models)
    annotate_heatmap(ax, values)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(f"{reducer_title} |EXP - simdistserve| (percentage points)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize(rows: list[Row]) -> dict:
    summary: dict[str, dict] = {}
    for metric, _, _ in METRICS:
        metric_rows = [row for row in rows if row.metric == metric]
        if not metric_rows:
            continue
        worst = max(metric_rows, key=lambda row: abs(row.delta_pct))
        summary[metric] = {
            "cases": len(metric_rows),
            "max_abs_delta_pct": abs(worst.delta_pct),
            "worst": {
                "model": worst.model,
                "rate": worst.rate,
                "slo_scale": worst.slo_scale,
                "delta_pct": worst.delta_pct,
            },
        }
    summary["overall"] = {
        "cases": len(rows),
        "max_abs_delta_pct": max((abs(row.delta_pct) for row in rows), default=float("nan")),
    }
    return summary


def plot_backend(slo_root: Path, backend: str) -> dict:
    rows = collect_rows(slo_root, backend)
    if not rows:
        raise FileNotFoundError(f"No scale comparison files found for backend: {backend}")

    output_root = slo_root / backend / "plots" / "heatmaps"
    write_rows_csv(output_root / "slo_delta_rows.csv", rows)

    models = sorted({row.model for row in rows})
    for model in models:
        for metric, _, metric_title in METRICS:
            plot_signed_heatmap(
                rows,
                backend,
                model,
                metric,
                metric_title,
                output_root / "by_model" / model / f"{metric}_delta_heatmap.png",
            )

    for metric, _, metric_title in METRICS:
        plot_summary_heatmap(
            rows,
            backend,
            metric,
            metric_title,
            output_root / "summary" / f"{metric}_max_abs_delta_heatmap.png",
            reducer="max",
        )
        plot_summary_heatmap(
            rows,
            backend,
            metric,
            metric_title,
            output_root / "summary" / f"{metric}_mean_abs_delta_heatmap.png",
            reducer="mean",
        )

    backend_summary = summarize(rows)
    (output_root / "summary.json").write_text(json.dumps(backend_summary, indent=2) + "\n", encoding="utf-8")
    return {
        "backend": backend,
        "rows": len(rows),
        "models": models,
        "output_root": str(output_root),
        "summary": backend_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SLO comparison heatmaps for backend folders.")
    parser.add_argument(
        "--slo-root",
        type=Path,
        default=Path("/users/rh/DistServe/simdistserve/benchmarks/results/slo"),
    )
    parser.add_argument("--backends", nargs="+", default=["distserve_cuda", "vllm_ascend"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [plot_backend(args.slo_root, backend) for backend in args.backends]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

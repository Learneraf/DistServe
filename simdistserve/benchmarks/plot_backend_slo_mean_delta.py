#!/usr/bin/env python3
"""Plot per-backend SLO attainment mean absolute error bars for thesis figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


MODELS = ["llama_1B", "llama_3B", "llama_7B", "llama_8B"]
METRICS = ["prefill", "decode", "both"]
METRIC_LABELS = {
    "prefill": "预填充",
    "decode": "解码",
    "both": "整体",
}
METRIC_COLORS = {
    "prefill": "#4c78a8",
    "decode": "#f58518",
    "both": "#2f7d6d",
}


def load_mean_abs_delta(csv_path: Path) -> dict[str, dict[str, float]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    values: dict[str, dict[str, float]] = {}
    for model in MODELS:
        values[model] = {}
        for metric in METRICS:
            deltas = [
                float(row["abs_delta_pct"])
                for row in rows
                if row["model"] == model and row["metric"] == metric
            ]
            values[model][metric] = float(np.mean(deltas)) if deltas else 0.0
    return values


def plot_backend(
    csv_path: Path,
    backend_label: str,
    output_path: Path,
    font_path: Path | None,
) -> None:
    font = FontProperties(fname=str(font_path)) if font_path and font_path.exists() else None
    values = load_mean_abs_delta(csv_path)

    x = np.arange(len(MODELS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for i, metric in enumerate(METRICS):
        y = [values[model][metric] for model in MODELS]
        bars = ax.bar(
            x + (i - 1) * width,
            y,
            width,
            label=METRIC_LABELS[metric],
            color=METRIC_COLORS[metric],
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title(f"{backend_label} backend SLO达标率平均绝对误差", fontproperties=font, fontsize=14)
    ax.set_ylabel("平均绝对误差(%)", fontproperties=font)
    ax.set_xlabel("模型", fontproperties=font)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontproperties=font)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(prop=font, frameon=False, ncol=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_slo_root = repo_root / "simdistserve" / "benchmarks" / "results" / "slo"
    default_figures = repo_root / "docs" / "figures"
    default_font = repo_root / "docs" / "fonts" / "NotoSerifSC-Regular.ttf"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slo-root", type=Path, default=default_slo_root)
    parser.add_argument("--output-dir", type=Path, default=default_figures)
    parser.add_argument("--font-path", type=Path, default=default_font)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = [
        (
            "DistServe",
            args.slo_root / "distserve_cuda" / "plots" / "heatmaps" / "slo_delta_rows.csv",
            args.output_dir / "fig_5_1a_distserve_slo_mean_delta.png",
        ),
        (
            "vLLM",
            args.slo_root / "vllm_ascend" / "plots" / "heatmaps" / "slo_delta_rows.csv",
            args.output_dir / "fig_5_1b_vllm_slo_mean_delta.png",
        ),
    ]

    for backend_label, csv_path, output_path in backends:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing SLO delta CSV: {csv_path}")
        plot_backend(csv_path, backend_label, output_path, args.font_path)
        print(output_path)


if __name__ == "__main__":
    main()

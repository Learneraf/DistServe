#!/usr/bin/env python3
"""
Plot CUDA vLLM P/D SLO fit/eval results from fit_slo_delta.csv and val_slo_delta.csv.

The input CSVs are produced by fit_and_eval_cuda_slo_affine.py and contain one row per
split/model/rate/slo_scale/metric. This script creates:
  - per split/model/rate SLO-scale curves comparing real, raw simulator, corrected simulator
  - per split/metric corrected-delta heatmaps
  - per split raw-vs-corrected max-delta summary bars
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("/users/rh/cuda_data/sim/3p3d_fit_model_forward/slo_affine_eval")
METRICS = [
    ("prefill", "Prefill"),
    ("decode", "Decode"),
    ("both", "Both"),
]
LINE_SPECS = [
    ("real_attainment_pct", "Real", "#1f4e79", "-", "o"),
    ("raw_attainment_pct", "Raw Sim", "#6f7378", "--", "s"),
    ("corrected_attainment_pct", "Corrected Sim", "#c46a2c", "--", "^"),
]


def style_axis(ax) -> None:
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color="#666666")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")


def rate_label(rate: float) -> str:
    if abs(rate - round(rate)) < 1e-9:
        return str(int(round(rate)))
    return f"{rate:g}"


def rate_stem(rate: float) -> str:
    return rate_label(rate).replace(".", "p")


def load_results(input_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("fit", "val"):
        csv_path = input_dir / f"{split}_slo_delta.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")
        frame = pd.read_csv(csv_path)
        if "split" not in frame.columns:
            frame.insert(0, "split", split)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "rate",
        "slo_scale",
        "real_attainment_pct",
        "raw_attainment_pct",
        "corrected_attainment_pct",
        "raw_delta_pct",
        "corrected_delta_pct",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col])
    return data


def plot_case(group: pd.DataFrame, split: str, model: str, rate: float, output_path: Path) -> None:
    scales = sorted(group["slo_scale"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor="#f3f0e8", sharey=True)
    fig.suptitle(
        f"{split.upper()} SLO Scale Sweep: {model} at rate={rate_label(rate)}",
        fontsize=15,
        y=1.02,
        weight="bold",
    )
    fig.text(
        0.5,
        0.935,
        "Real benchmark vs raw 3p3d simulator vs fitted execution-time model",
        ha="center",
        va="center",
        fontsize=10,
        color="#5f5a53",
    )

    for ax, (metric_key, metric_name) in zip(axes, METRICS):
        metric_rows = (
            group[group["metric"] == metric_key]
            .sort_values("slo_scale")
            .set_index("slo_scale")
            .reindex(scales)
            .reset_index()
        )
        style_axis(ax)

        real = metric_rows["real_attainment_pct"].to_numpy(dtype=float)
        corrected = metric_rows["corrected_attainment_pct"].to_numpy(dtype=float)
        raw = metric_rows["raw_attainment_pct"].to_numpy(dtype=float)
        scale_arr = np.asarray(scales, dtype=float)
        valid = ~(np.isnan(real) | np.isnan(corrected))
        if np.any(valid):
            ax.fill_between(
                scale_arr[valid],
                real[valid],
                corrected[valid],
                color="#e8b85c",
                alpha=0.18,
                linewidth=0,
                label="Corrected Gap",
            )

        for col, label, color, linestyle, marker in LINE_SPECS:
            ax.plot(
                scales,
                metric_rows[col],
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2.2,
                markersize=5.5,
                label=label,
            )

        raw_gap = np.abs(real - raw)
        corrected_gap = np.abs(real - corrected)
        ax.text(
            0.03,
            0.96,
            f"raw max |Δ| = {np.nanmax(raw_gap):.1f}%\ncorrected max |Δ| = {np.nanmax(corrected_gap):.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#d7d7cf",
                alpha=0.92,
            ),
        )
        ax.axhline(95, color="#8c8c8c", linestyle=":", linewidth=1.0, alpha=0.55)
        ax.set_title(metric_name, fontsize=12, pad=10, weight="bold")
        ax.set_xlabel("SLO Scale", fontsize=10)
        ax.set_ylim(-2, 102)
        ax.set_xlim(min(scales) - 0.04, max(scales) + 0.04)
        ax.set_xticks(scales)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False)

    axes[0].set_ylabel("SLO Met Rate (%)", fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(data: pd.DataFrame, split: str, metric: str, output_path: Path) -> None:
    subset = data[(data["split"] == split) & (data["metric"] == metric)].copy()
    subset["case"] = subset["model"] + " r=" + subset["rate"].map(rate_label)
    subset = subset.sort_values(["model", "rate", "slo_scale"])
    pivot = subset.pivot_table(
        index="case",
        columns="slo_scale",
        values="corrected_delta_pct",
        aggfunc="max",
    )

    fig_width = max(8.5, 1.05 * len(pivot.columns) + 4.0)
    fig_height = max(7.0, 0.34 * len(pivot.index) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="#f3f0e8")
    ax.set_facecolor("#fbfbf8")
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(12.0, np.nanmax(values)))

    ax.set_title(
        f"{split.upper()} Corrected SLO Delta Heatmap: {metric}",
        fontsize=15,
        pad=14,
        weight="bold",
    )
    ax.set_xlabel("SLO Scale")
    ax.set_ylabel("Model / Request Rate")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{col:g}" for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    exceeded_cells: list[tuple[int, int]] = []
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                continue
            text_color = "white" if value >= 7.5 else "#242424"
            ax.text(col_idx, row_idx, f"{value:.1f}", ha="center", va="center", fontsize=8.5, color=text_color)
            if value > 5.0:
                exceeded_cells.append((row_idx, col_idx))

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_idx, col_idx in exceeded_cells:
        rect = plt.Rectangle(
            (col_idx - 0.5, row_idx - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#2a0f0f",
            linewidth=1.8,
        )
        ax.add_patch(rect)
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Corrected |SLO delta| (%)")
    fig.text(
        0.5,
        0.02,
        "Cells outlined in black are above the 5% target.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#5f5a53",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_delta_heatmap(data: pd.DataFrame, split: str, metric: str, output_path: Path) -> None:
    subset = data[(data["split"] == split) & (data["metric"] == metric)].copy()
    if subset.empty:
        return

    pivot = subset.pivot_table(
        index="model",
        columns="rate",
        values="corrected_delta_pct",
        aggfunc="mean",
    ).sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    values = pivot.to_numpy(dtype=float)
    vmax = max(5.0, float(np.nanmax(values)) if values.size else 5.0)
    fig_width = max(7.5, 0.95 * len(pivot.columns) + 2.8)
    fig_height = max(4.8, 0.55 * len(pivot.index) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="#f3f0e8")
    ax.set_facecolor("#fbfbf8")
    image = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)

    ax.set_title(
        f"{split.upper()} Mean Corrected |SLO Delta|: {metric}",
        fontsize=15,
        pad=14,
        weight="bold",
    )
    ax.set_xlabel("Request Rate")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([rate_label(float(col)) for col in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                continue
            text_color = "white" if value >= 0.65 * vmax else "#242424"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Mean corrected |SLO delta| (%) across SLO scales")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_delta_summary(data: pd.DataFrame, split: str, output_path: Path) -> None:
    subset = data[data["split"] == split].copy()
    grouped = (
        subset.groupby(["model", "rate", "metric"], as_index=False)
        .agg(raw_max=("raw_delta_pct", "max"), corrected_max=("corrected_delta_pct", "max"))
        .sort_values(["model", "rate", "metric"])
    )
    grouped["case"] = grouped["model"] + "\n" + grouped["metric"] + "\nr=" + grouped["rate"].map(rate_label)

    x = np.arange(len(grouped))
    fig_width = max(14.0, 0.38 * len(grouped) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2), facecolor="#f3f0e8")
    style_axis(ax)
    width = 0.38
    ax.bar(x - width / 2, grouped["raw_max"], width=width, color="#8a8f96", label="Raw max |Δ|")
    ax.bar(x + width / 2, grouped["corrected_max"], width=width, color="#c46a2c", label="Corrected max |Δ|")
    ax.axhline(5, color="#8f1d1d", linestyle="--", linewidth=1.2, alpha=0.9, label="5% target")
    ax.set_title(f"{split.upper()} Max SLO Delta by Model, Rate, Metric", fontsize=15, pad=14, weight="bold")
    ax.set_ylabel("Max |SLO delta| (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["case"], rotation=90, fontsize=7.5)
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(8.0, float(grouped[["raw_max", "corrected_max"]].to_numpy().max()) * 1.12))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_plot_index(data: pd.DataFrame, output_dir: Path) -> None:
    summary = {}
    for split, split_rows in data.groupby("split"):
        corrected = split_rows["corrected_delta_pct"].to_numpy(dtype=float)
        raw = split_rows["raw_delta_pct"].to_numpy(dtype=float)
        summary[split] = {
            "cases": int(len(split_rows)),
            "max_raw_delta_pct": float(np.nanmax(raw)),
            "max_corrected_delta_pct": float(np.nanmax(corrected)),
            "corrected_cases_within_5pct": int(np.sum(corrected <= 5.0)),
            "corrected_fraction_within_5pct": float(np.mean(corrected <= 5.0)),
        }

    index_path = output_dir / "plot_summary.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CUDA SLO affine fit/eval results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf"],
        help="Plot file formats to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "plots"
    data = load_results(input_dir)

    for split in sorted(data["split"].unique()):
        split_rows = data[data["split"] == split]
        for (model, rate), group in split_rows.groupby(["model", "rate"], sort=True):
            for fmt in args.formats:
                output_path = (
                    output_dir
                    / "slo_scale"
                    / split
                    / model
                    / f"rate_{rate_stem(float(rate))}_slo_scale.{fmt}"
                )
                plot_case(group, split, model, float(rate), output_path)

        for metric, _ in METRICS:
            for fmt in args.formats:
                output_path = output_dir / "heatmaps" / split / f"{metric}_corrected_delta_heatmap.{fmt}"
                plot_delta_heatmap(data, split, metric, output_path)
                mean_output_path = output_dir / "heatmaps" / "summary" / f"{split}_{metric}_mean_abs_delta_heatmap.{fmt}"
                plot_mean_delta_heatmap(data, split, metric, mean_output_path)

        for fmt in args.formats:
            output_path = output_dir / "summary" / f"{split}_raw_vs_corrected_delta_summary.{fmt}"
            plot_delta_summary(data, split, output_path)

    write_plot_index(data, output_dir)
    print(f"Wrote plots under: {output_dir}")


if __name__ == "__main__":
    main()

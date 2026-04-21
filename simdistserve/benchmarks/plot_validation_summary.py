#!/usr/bin/env python3
"""
Plot validation-summary CSV files as 2x2 SLO dashboards.

This is intended for summary tables like:
    /users/rh/ascend_data/validation/final_3param_profile/summary.csv

Each model gets one figure with four panels:
    Prefill / Decode / Total / Both
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("prefill", "Prefill SLO Met Rate"),
    ("decode", "Decode SLO Met Rate"),
    ("total", "Total SLO Met Rate"),
    ("both", "Both SLO Met Rate"),
]


def style_axis(ax) -> None:
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color="#666666")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")


def add_gap_annotation(ax, actual_vals: np.ndarray, sim_vals: np.ndarray) -> None:
    diffs = np.abs(actual_vals - sim_vals)
    valid_diffs = diffs[~np.isnan(diffs)]
    if len(valid_diffs) == 0:
        return
    ax.text(
        0.03,
        0.96,
        f"mean |Δ| = {float(np.mean(valid_diffs)):.1f}%\nmax |Δ| = {float(np.max(valid_diffs)):.1f}%",
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


def load_rows(summary_csv: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with summary_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["model"]].append(row)
    return grouped


def plot_model(model: str, rows: list[dict[str, str]], sim_prefix: str, output_path: Path) -> None:
    rows = sorted(rows, key=lambda row: float(row["rate"]))
    rates = np.array([float(row["rate"]) for row in rows], dtype=float)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor="#f3f0e8")
    fig.suptitle(f"SLO Satisfaction Comparison: {model}", fontsize=17, y=0.98, weight="bold")
    fig.text(
        0.5,
        0.945,
        "Actual benchmark vs simulator prediction across request rates",
        ha="center",
        va="center",
        fontsize=10,
        color="#5f5a53",
    )

    for ax, (metric_key, display_name) in zip(axes.flatten(), METRICS):
        style_axis(ax)

        actual = np.array([float(row[f"actual_{metric_key}_slo_pct"]) for row in rows], dtype=float)
        sim = np.array([float(row[f"{sim_prefix}_{metric_key}_slo_pct"]) for row in rows], dtype=float)
        valid_mask = ~(np.isnan(actual) | np.isnan(sim))

        if np.any(valid_mask):
            ax.fill_between(
                rates[valid_mask],
                actual[valid_mask],
                sim[valid_mask],
                color="#e8b85c",
                alpha=0.16,
                linewidth=0,
                label="Absolute Gap",
            )

        ax.plot(
            rates,
            actual,
            marker="o",
            linestyle="-",
            color="#1f4e79",
            linewidth=2.4,
            markersize=6,
            label="Actual",
        )
        ax.plot(
            rates,
            sim,
            marker="s",
            linestyle="--",
            color="#c46a2c",
            linewidth=2.2,
            markersize=5.5,
            label="Simulator",
        )

        ax.set_title(display_name, fontsize=12, pad=10, weight="bold")
        ax.set_xlabel("Request Rate", fontsize=10)
        ax.set_ylabel("SLO Met Rate (%)", fontsize=10)
        ax.set_ylim(-2, 102)
        ax.set_xlim(float(np.min(rates)) - 0.1, float(np.max(rates)) + 0.1)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False)
        add_gap_annotation(ax, actual, sim)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.16, hspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a validation summary CSV into one 2x2 SLO dashboard per model."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sim-prefix",
        default="new",
        help="Simulator column prefix in the CSV, for example 'new' or 'old'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped_rows = load_rows(args.summary_csv)
    if not grouped_rows:
        raise ValueError(f"No rows found in {args.summary_csv}")

    for model, rows in sorted(grouped_rows.items()):
        output_path = args.output_dir / f"{model}_slo_subplots.png"
        plot_model(model, rows, args.sim_prefix, output_path)
        print(output_path)


if __name__ == "__main__":
    main()

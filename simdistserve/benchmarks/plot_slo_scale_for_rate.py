#!/usr/bin/env python3
"""
Plot benchmark-vs-simulator SLO met rates across a sweep of SLO scales for one fixed rate.

This script reads pre-generated per-scale `comparison.txt` files under:
    results/slo/<backend>/compared/<model>/rate_<rate>/scale_<scale>/comparison.txt

Example:
    python3 /users/rh/DistServe/simdistserve/benchmarks/plot_slo_scale_for_rate.py \
        --backend vllm_ascend \
        --model llama_7B \
        --rate 4 \
        --output-dir /users/rh/DistServe/simdistserve/benchmarks/results/slo_scale_plots
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/users/rh/DistServe")
BENCH_ROOT = ROOT / "simdistserve" / "benchmarks"
RESULTS_ROOT = BENCH_ROOT / "results"

METRICS = [
    ("prefill", "Prefill SLO met rate"),
    ("decode", "Decode SLO met rate"),
    ("total", "Total SLO met rate"),
    ("both", "Both (Prefill+Decode) SLO met rate"),
]
DEFAULT_SLO_SCALES = [0.4, 0.6, 0.8, 1.0, 1.2]
BACKEND_OUTPUT_NAMES = {
    "distserve_cuda": "dist-serve",
    "vllm_ascend": "vllm-ascend",
}
THRESHOLD_RE = re.compile(
    r"Unified SLO thresholds:\s*Prefill=([0-9.]+)s,\s*Decode=([0-9.]+)s,\s*Total=([0-9.]+)s"
)


def compared_dir_for(backend: str, model: str, rate: str) -> Path:
    return RESULTS_ROOT / "slo" / backend / "compared" / model / f"rate_{rate}"


def backend_output_name(backend: str) -> str:
    try:
        return BACKEND_OUTPUT_NAMES[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend: {backend}") from exc


def output_dir_for_case(root_output_dir: Path, backend: str, model: str) -> Path:
    return root_output_dir / backend_output_name(backend) / model


def parse_scales(raw_scales: str | None) -> list[float] | None:
    if raw_scales is None:
        return None
    values = ast.literal_eval(raw_scales)
    if not isinstance(values, list):
        raise ValueError("--slo-scales must be a Python list")
    scales = sorted(float(value) for value in values)
    for scale in scales:
        if scale <= 0:
            raise ValueError("All SLO scales must be positive")
    return scales


def parse_scale_from_label(label: str) -> float:
    return float(label.replace("p", "."))


def infer_scale(
    thresholds: dict[str, float] | None,
    scale_label: str | None,
    base_prefill_slo: float,
    base_decode_slo: float,
    base_total_slo: float,
) -> float:
    if scale_label is not None:
        return parse_scale_from_label(scale_label)

    ratios: list[float] = []
    if thresholds:
        if base_prefill_slo > 0:
            ratios.append(thresholds["prefill"] / base_prefill_slo)
        if base_decode_slo > 0:
            ratios.append(thresholds["decode"] / base_decode_slo)
        if base_total_slo > 0:
            ratios.append(thresholds["total"] / base_total_slo)
    if ratios:
        return float(sum(ratios) / len(ratios))
    if scale_label is None:
        raise ValueError("Unable to infer SLO scale from comparison data.")
    return parse_scale_from_label(scale_label)


def parse_comparison_txt(filepath: Path) -> dict:
    results = {metric_label: None for _, metric_label in METRICS}
    thresholds = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if thresholds is None:
                match = THRESHOLD_RE.search(line)
                if match:
                    thresholds = {
                        "prefill": float(match.group(1)),
                        "decode": float(match.group(2)),
                        "total": float(match.group(3)),
                    }

            for _, metric_label in METRICS:
                if metric_label not in line:
                    continue
                percents = re.findall(r"(\d+\.?\d*)%", line)
                if len(percents) < 2:
                    continue
                benchmark = float(percents[0])
                simulator = float(percents[1])
                results[metric_label] = {
                    "benchmark": benchmark,
                    "simulator": simulator,
                    "gap": benchmark - simulator,
                }
                break

    missing_metrics = [label for _, label in METRICS if results[label] is None]
    if missing_metrics:
        raise ValueError(f"Missing metrics in {filepath}: {', '.join(missing_metrics)}")

    return {
        "thresholds": thresholds,
        "metrics": results,
    }


def matches_requested_scale(scale: float, requested_scales: list[float] | None) -> bool:
    if requested_scales is None:
        return True
    return any(abs(scale - requested) <= 1e-9 for requested in requested_scales)


def load_summary_from_comparisons(
    backend: str,
    model: str,
    rate: str,
    compared_root: Path,
    base_prefill_slo: float,
    base_decode_slo: float,
    base_total_slo: float,
    requested_scales: list[float] | None,
) -> dict:
    if not compared_root.exists():
        raise FileNotFoundError(f"Compared root not found: {compared_root}")

    points: list[dict] = []

    scale_dirs = sorted(
        path for path in compared_root.iterdir()
        if path.is_dir() and path.name.startswith("scale_")
    )
    if scale_dirs:
        for scale_dir in scale_dirs:
            comparison_path = scale_dir / "comparison.txt"
            if not comparison_path.exists():
                continue
            scale_label = scale_dir.name.removeprefix("scale_")
            parsed = parse_comparison_txt(comparison_path)
            scale = infer_scale(
                parsed["thresholds"],
                scale_label,
                base_prefill_slo,
                base_decode_slo,
                base_total_slo,
            )
            if not matches_requested_scale(scale, requested_scales):
                continue
            points.append(
                {
                    "scale": scale,
                    "comparison_file": str(comparison_path),
                    "thresholds": parsed["thresholds"],
                    "metrics": {
                        metric_key: parsed["metrics"][metric_label]
                        for metric_key, metric_label in METRICS
                    },
                }
            )
    else:
        comparison_path = compared_root / "comparison.txt"
        if not comparison_path.exists():
            raise FileNotFoundError(
                f"No per-scale comparison outputs found under {compared_root}"
            )
        parsed = parse_comparison_txt(comparison_path)
        scale = infer_scale(
            parsed["thresholds"],
            "1p0",
            base_prefill_slo,
            base_decode_slo,
            base_total_slo,
        )
        if matches_requested_scale(scale, requested_scales):
            points.append(
                {
                    "scale": scale,
                    "comparison_file": str(comparison_path),
                    "thresholds": parsed["thresholds"],
                    "metrics": {
                        metric_key: parsed["metrics"][metric_label]
                        for metric_key, metric_label in METRICS
                    },
                }
            )

    if not points:
        raise ValueError(f"No comparison points matched under {compared_root}")

    points.sort(key=lambda point: point["scale"])

    metric_summary = {}
    for metric_key, _ in METRICS:
        gaps = [abs(point["metrics"][metric_key]["gap"]) for point in points]
        metric_summary[metric_key] = {
            "mean_abs_gap": float(np.mean(gaps)),
            "max_abs_gap": float(np.max(gaps)),
        }

    return {
        "backend": backend,
        "model": model,
        "rate": rate,
        "compared_root": str(compared_root),
        "base_slos": {
            "prefill": base_prefill_slo,
            "decode": base_decode_slo,
            "total": base_total_slo,
        },
        "scales": [point["scale"] for point in points],
        "points": points,
        "metric_summary": metric_summary,
    }


def style_axis(ax):
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color="#666666")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")


def plot_summary(summary: dict, model: str, rate: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor="#f3f0e8")
    fig.suptitle(
        f"SLO Scale Sweep: {model} at rate={rate}",
        fontsize=17,
        y=0.98,
        weight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "Actual benchmark vs simulator prediction from saved comparison reports",
        ha="center",
        va="center",
        fontsize=10,
        color="#5f5a53",
    )

    scales = np.array(summary["scales"], dtype=float)
    axes = axes.flatten()

    for idx, (metric_key, title) in enumerate(METRICS):
        ax = axes[idx]
        style_axis(ax)
        benchmark = np.array(
            [point["metrics"][metric_key]["benchmark"] for point in summary["points"]],
            dtype=float,
        )
        simulator = np.array(
            [point["metrics"][metric_key]["simulator"] for point in summary["points"]],
            dtype=float,
        )
        gap = np.abs(benchmark - simulator)

        ax.fill_between(
            scales,
            benchmark,
            simulator,
            color="#e8b85c",
            alpha=0.16,
            linewidth=0,
            label="Absolute Gap",
        )
        ax.plot(
            scales,
            benchmark,
            marker="o",
            linestyle="-",
            color="#1f4e79",
            linewidth=2.4,
            markersize=6,
            label="Actual",
        )
        ax.plot(
            scales,
            simulator,
            marker="s",
            linestyle="--",
            color="#c46a2c",
            linewidth=2.2,
            markersize=5.5,
            label="Simulator",
        )

        ax.set_title(title, fontsize=12, pad=10, weight="bold")
        ax.set_xlabel("SLO Scale", fontsize=10)
        ax.set_ylabel("SLO Met Rate (%)", fontsize=10)
        ax.set_ylim(-2, 102)
        ax.set_xlim(scales.min() - 0.03, scales.max() + 0.03)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False)
        ax.text(
            0.03,
            0.96,
            f"mean |Δ| = {gap.mean():.1f}%\nmax |Δ| = {gap.max():.1f}%",
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

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.16, hspace=0.24)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SLO-scale curves directly from saved comparison.txt files."
    )
    parser.add_argument("--backend", choices=["distserve_cuda", "vllm_ascend"], required=True)
    parser.add_argument("--model", required=True, help="Model alias such as llama_1B, llama_3B, llama_7B, llama_8B")
    parser.add_argument("--rate", required=True, help="Specific rate to evaluate, for example 1 or 2.5")
    parser.add_argument(
        "--compared-root",
        type=Path,
        default=None,
        help="Optional override for the compared/<model>/rate_<rate> directory.",
    )
    parser.add_argument("--prefill-slo", type=float, default=1.0)
    parser.add_argument("--decode-slo", type=float, default=1.0)
    parser.add_argument("--total-slo", type=float, default=1.0)
    parser.add_argument(
        "--slo-scales",
        type=str,
        default=str(DEFAULT_SLO_SCALES),
        help="Optional list of scales to include. The script reads saved comparison files instead of recomputing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_ROOT / "results" / "slo_scale_plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_scales = parse_scales(args.slo_scales)
    compared_root = args.compared_root or compared_dir_for(args.backend, args.model, args.rate)

    summary = load_summary_from_comparisons(
        args.backend,
        args.model,
        args.rate,
        compared_root,
        args.prefill_slo,
        args.decode_slo,
        args.total_slo,
        requested_scales,
    )

    case_output_dir = output_dir_for_case(args.output_dir, args.backend, args.model)
    stem = f"rate_{str(args.rate).replace('.', 'p')}"
    json_path = case_output_dir / f"{stem}_slo_scale_summary.json"
    png_path = case_output_dir / f"{stem}_slo_scale_plot.png"
    case_output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    plot_summary(summary, args.model, args.rate, png_path)

    print(f"Wrote summary: {json_path}")
    print(f"Wrote plot: {png_path}")


if __name__ == "__main__":
    main()

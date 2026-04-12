#!/usr/bin/env python3
"""
Plot benchmark-vs-simulator SLO met rates across a sweep of SLO scales for one fixed rate.

Example:
    python3 /users/rh/DistServe/simdistserve/benchmarks/plot_slo_scale_for_rate.py \
        --backend distserve_cuda \
        --model llama_7B \
        --rate 1 \
        --output-dir /users/rh/DistServe/simdistserve/benchmarks/results/slo_scale_plots
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from merged_analyze import analyze_csv, analyze_exp, analyze_json


ROOT = Path("/users/rh/DistServe")
BENCH_ROOT = ROOT / "simdistserve" / "benchmarks"
RESULTS_ROOT = BENCH_ROOT / "results"
DISTSERVE_BENCH_ROOT = ROOT / "evaluation" / "2-benchmark-serving" / "result"
VLLM_ASCEND_BENCH_ROOT = RESULTS_ROOT / "latency" / "vllm_ascend" / "raw_data"

METRICS = [
    ("prefill", "Prefill SLO met rate", "prefill_slo_rate"),
    ("decode", "Decode SLO met rate", "decode_slo_rate"),
    ("total", "Total SLO met rate", "total_slo_rate"),
    ("both", "Both (Prefill+Decode) SLO met rate", "both_slo_rate"),
]
DEFAULT_MODELS = ["llama_1B", "llama_3B", "llama_7B", "llama_8B"]
DEFAULT_RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
DEFAULT_SLO_SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
BACKEND_OUTPUT_NAMES = {
    "distserve_cuda": "dist-serve",
    "vllm_ascend": "vllm-ascend",
}


def benchmark_file_for(backend: str, model: str, rate: str) -> Path:
    if backend == "distserve_cuda":
        return DISTSERVE_BENCH_ROOT / model / f"distserve-100-{rate}.exp"
    if backend == "vllm_ascend":
        return VLLM_ASCEND_BENCH_ROOT / model / f"rate_{rate}.json"
    raise ValueError(f"Unsupported backend: {backend}")


def simulator_file_for(backend: str, model: str, rate: str) -> Path:
    return RESULTS_ROOT / "latency" / backend / "organized_data" / model / f"rate_{rate}" / "request_latency.csv"


def backend_output_name(backend: str) -> str:
    try:
        return BACKEND_OUTPUT_NAMES[backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend: {backend}") from exc


def output_dir_for_case(root_output_dir: Path, backend: str, model: str) -> Path:
    return root_output_dir / backend_output_name(backend) / model


def parse_scales(raw_scales: str) -> list[float]:
    values = ast.literal_eval(raw_scales)
    if not isinstance(values, list):
        raise ValueError("--slo-scales must be a Python list")
    scales = sorted(float(value) for value in values)
    for scale in scales:
        if scale <= 0:
            raise ValueError("All SLO scales must be positive")
    return scales


def sweep_one_case(
    backend: str,
    model: str,
    rate: str,
    benchmark_file: Path,
    simulator_file: Path,
    base_prefill_slo: float,
    base_decode_slo: float,
    base_total_slo: float,
    scales: list[float],
) -> dict:
    summary = {
        "backend": backend,
        "model": model,
        "rate": rate,
        "benchmark_file": str(benchmark_file),
        "simulator_file": str(simulator_file),
        "base_slos": {
            "prefill": base_prefill_slo,
            "decode": base_decode_slo,
            "total": base_total_slo,
        },
        "scales": scales,
        "points": [],
    }

    for scale in scales:
        prefill_slo = base_prefill_slo * scale
        decode_slo = base_decode_slo * scale
        total_slo = base_total_slo * scale

        if backend == "vllm_ascend":
            exp_stats, _ = analyze_json(str(benchmark_file), prefill_slo, decode_slo, total_slo)
        else:
            exp_stats, _ = analyze_exp(str(benchmark_file), prefill_slo, decode_slo, total_slo)
        csv_stats, _ = analyze_csv(str(simulator_file), prefill_slo, decode_slo, total_slo)

        point = {
            "scale": scale,
            "metrics": {},
        }
        for metric_key, _, stats_key in METRICS:
            point["metrics"][metric_key] = {
                "benchmark": exp_stats[stats_key],
                "simulator": csv_stats[stats_key],
                "gap": exp_stats[stats_key] - csv_stats[stats_key],
        }
        summary["points"].append(point)

    metric_summary = {}
    for metric_key, _, _ in METRICS:
        gaps = [abs(point["metrics"][metric_key]["gap"]) for point in summary["points"]]
        metric_summary[metric_key] = {
            "mean_abs_gap": float(np.mean(gaps)),
            "max_abs_gap": float(np.max(gaps)),
        }
    summary["metric_summary"] = metric_summary

    return summary


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
        "Actual benchmark vs simulator prediction across SLO scales",
        ha="center",
        va="center",
        fontsize=10,
        color="#5f5a53",
    )

    scales = np.array(summary["scales"], dtype=float)
    axes = axes.flatten()

    for idx, (metric_key, title, _) in enumerate(METRICS):
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
        description="Sweep SLO scales for one fixed (backend, model, rate) case and plot the result."
    )
    parser.add_argument("--backend", choices=["distserve_cuda", "vllm_ascend"], required=True)
    parser.add_argument("--model", required=True, help="Model alias such as llama_1B, llama_3B, llama_7B, llama_8B")
    parser.add_argument("--rate", required=True, help="Specific rate to evaluate, for example 1 or 2.5")
    parser.add_argument("--prefill-slo", type=float, default=1.0)
    parser.add_argument("--decode-slo", type=float, default=1.0)
    parser.add_argument("--total-slo", type=float, default=1.0)
    parser.add_argument(
        "--slo-scales",
        type=str,
        default=str(DEFAULT_SLO_SCALES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCH_ROOT / "results" / "slo_scale_plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = parse_scales(args.slo_scales)
    benchmark_file = benchmark_file_for(args.backend, args.model, args.rate)
    simulator_file = simulator_file_for(args.backend, args.model, args.rate)

    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")
    if not simulator_file.exists():
        raise FileNotFoundError(f"Simulator file not found: {simulator_file}")

    summary = sweep_one_case(
        args.backend,
        args.model,
        args.rate,
        benchmark_file,
        simulator_file,
        args.prefill_slo,
        args.decode_slo,
        args.total_slo,
        scales,
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

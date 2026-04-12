#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result")
OUTPUT_ROOT = Path("/users/rh/tmp/distserve_ablation/results/completion_pathology")
MODELS = ["llama_1B", "llama_3B", "llama_7B", "llama_8B"]
RATES = ["1", "1.5", "2", "2.5", "3", "3.5", "4"]
NUM_PROMPTS = 100
SEED = 0


@dataclass
class RateMetrics:
    model: str
    rate: str
    total_p95_ms: float
    total_p99_ms: float
    total_max_ms: float
    ftl_p99_ms: float
    decode_p99_ms: float
    post_last_token_p95_ms: float
    post_last_token_p99_ms: float
    post_last_token_max_ms: float
    observed_start_gap_max_ms: float
    expected_gap_max_ms: float
    observed_start_gap_p99_ms: float
    expected_gap_p99_ms: float
    max_concurrency: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[idx])


def load_requests(model: str, rate: str) -> list[dict]:
    path = RESULTS_ROOT / model / f"distserve-100-{rate}.exp"
    with open(path) as f:
        return json.load(f)


def expected_interarrival_gaps(rate: str, n: int = NUM_PROMPTS, seed: int = SEED) -> list[float]:
    np.random.seed(seed)
    rate_value = float(rate)
    intervals = np.random.gamma(1.0, 1.0 / rate_value, size=n)
    return [float(interval * 1000.0) for interval in intervals[:-1]]


def observed_start_gaps_ms(requests: list[dict]) -> list[float]:
    starts = sorted(float(request["start_time"]) for request in requests)
    return [(right - left) * 1000.0 for left, right in zip(starts, starts[1:])]


def max_concurrency(requests: list[dict]) -> int:
    events: list[tuple[float, int]] = []
    for request in requests:
        events.append((float(request["start_time"]), 1))
        events.append((float(request["end_time"]), -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    maximum = 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    return maximum


def request_metrics(model: str, rate: str) -> RateMetrics:
    requests = load_requests(model, rate)
    total_ms = [float(request["latency"]) * 1000.0 for request in requests]
    ftl_ms = [float(request["ftl"]) * 1000.0 for request in requests]
    post_last_token_ms = [
        (float(request["end_time"]) - float(request["token_timestamps"][-1])) * 1000.0
        for request in requests
    ]
    decode_ms = []
    for request in requests:
        lifecycle = {event["event_type"]: float(event["timestamp"]) for event in request.get("lifecycle_events", [])}
        decode_ms.append((lifecycle["decoding_end"] - lifecycle["decoding_begin"]) * 1000.0)

    observed_gaps = observed_start_gaps_ms(requests)
    expected_gaps = expected_interarrival_gaps(rate, n=len(requests))

    return RateMetrics(
        model=model,
        rate=rate,
        total_p95_ms=percentile(total_ms, 0.95),
        total_p99_ms=percentile(total_ms, 0.99),
        total_max_ms=max(total_ms),
        ftl_p99_ms=percentile(ftl_ms, 0.99),
        decode_p99_ms=percentile(decode_ms, 0.99),
        post_last_token_p95_ms=percentile(post_last_token_ms, 0.95),
        post_last_token_p99_ms=percentile(post_last_token_ms, 0.99),
        post_last_token_max_ms=max(post_last_token_ms),
        observed_start_gap_max_ms=max(observed_gaps),
        expected_gap_max_ms=max(expected_gaps),
        observed_start_gap_p99_ms=percentile(observed_gaps, 0.99),
        expected_gap_p99_ms=percentile(expected_gaps, 0.99),
        max_concurrency=max_concurrency(requests),
    )


def style_axis(ax):
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color="#666666")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")


def write_summary(metrics: list[RateMetrics]) -> None:
    summary = {
        "seed": SEED,
        "num_prompts": NUM_PROMPTS,
        "models": MODELS,
        "rates": RATES,
        "metrics": [metric.__dict__ for metric in metrics],
    }
    with open(OUTPUT_ROOT / "completion_pathology_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def write_report(metrics: list[RateMetrics]) -> None:
    by_model = {model: [metric for metric in metrics if metric.model == model] for model in MODELS}
    lines = [
        "# DistServe CUDA Completion Pathology",
        "",
        "This analysis focuses on two non-compute indicators:",
        "",
        "- `post_last_token_ms = end_time - token_timestamps[-1]`",
        "- observed request-start gaps compared with the seeded Poisson schedule used by the benchmark",
        "",
        "If these values spike while server-side decode does not, the anomaly is more consistent with client/response-path delay than with model compute.",
        "",
    ]

    for model in MODELS:
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| Rate | total p99 ms | decode p99 ms | post-last-token p99 ms | post-last-token max ms | observed max start gap ms | expected max gap ms | max concurrency |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for metric in by_model[model]:
            lines.append(
                f"| {metric.rate} | {metric.total_p99_ms:.2f} | {metric.decode_p99_ms:.2f} | "
                f"{metric.post_last_token_p99_ms:.2f} | {metric.post_last_token_max_ms:.2f} | "
                f"{metric.observed_start_gap_max_ms:.2f} | {metric.expected_gap_max_ms:.2f} | {metric.max_concurrency} |"
            )
        lines.append("")

    with open(OUTPUT_ROOT / "completion_pathology_report.md", "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def plot_heatmaps(metrics: list[RateMetrics]) -> None:
    by_key = {(metric.model, metric.rate): metric for metric in metrics}
    model_labels = MODELS
    rate_labels = RATES

    post_last_token = np.array(
        [[by_key[(model, rate)].post_last_token_p99_ms for rate in rate_labels] for model in model_labels],
        dtype=float,
    )
    start_gap_ratio = np.array(
        [[by_key[(model, rate)].observed_start_gap_max_ms / max(by_key[(model, rate)].expected_gap_max_ms, 1e-6) for rate in rate_labels] for model in model_labels],
        dtype=float,
    )
    total_vs_decode_gap = np.array(
        [[by_key[(model, rate)].total_p99_ms - by_key[(model, rate)].decode_p99_ms for rate in rate_labels] for model in model_labels],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#f3f0e8")
    fig.suptitle("DistServe CUDA Completion Pathology", fontsize=17, y=0.98, weight="bold")
    fig.text(0.5, 0.94, "Looking for response-path stalls rather than pure compute slowdowns", ha="center", va="center", fontsize=10, color="#5f5a53")

    panels = [
        ("Post-last-token p99 (ms)", post_last_token, "YlOrBr"),
        ("Observed / Expected max start-gap ratio", start_gap_ratio, "YlGnBu"),
        ("Total p99 - decode p99 (ms)", total_vs_decode_gap, "YlOrRd"),
    ]

    for ax, (title, matrix, cmap) in zip(axes, panels):
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks(range(len(rate_labels)))
        ax.set_xticklabels(rate_labels)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels)
        ax.set_xlabel("Rate")
        ax.set_ylabel("Model")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8, color="#1f1f1f")
        fig.colorbar(image, ax=ax, shrink=0.84)

    plt.tight_layout()
    plt.subplots_adjust(top=0.86, wspace=0.28)
    plt.savefig(OUTPUT_ROOT / "completion_pathology_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_llama7b_detail(metrics: list[RateMetrics]) -> None:
    llama_metrics = [metric for metric in metrics if metric.model == "llama_7B"]
    rates = np.array([float(metric.rate) for metric in llama_metrics], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#f3f0e8")
    fig.suptitle("llama_7B DistServe CUDA Detail", fontsize=17, y=0.98, weight="bold")
    fig.text(0.5, 0.945, "Separating server decode behavior from response-path and client-side anomalies", ha="center", va="center", fontsize=10, color="#5f5a53")
    axes = axes.flatten()

    panels = [
        ("Total / Decode p99", [
            ("total p99", [metric.total_p99_ms for metric in llama_metrics], "#1f4e79"),
            ("decode p99", [metric.decode_p99_ms for metric in llama_metrics], "#c46a2c"),
            ("post-last-token p99", [metric.post_last_token_p99_ms for metric in llama_metrics], "#7a8f2a"),
        ], "Latency (ms)"),
        ("Start-gap Max", [
            ("observed max gap", [metric.observed_start_gap_max_ms for metric in llama_metrics], "#8c3f8c"),
            ("expected max gap", [metric.expected_gap_max_ms for metric in llama_metrics], "#4b6f44"),
        ], "Gap (ms)"),
        ("Total / Decode Difference", [
            ("total p99 - decode p99", [metric.total_p99_ms - metric.decode_p99_ms for metric in llama_metrics], "#b84a3a"),
        ], "Difference (ms)"),
        ("Concurrency", [
            ("max concurrency", [metric.max_concurrency for metric in llama_metrics], "#2f7f9f"),
        ], "Requests"),
    ]

    for ax, (title, series, ylabel) in zip(axes, panels):
        style_axis(ax)
        for label, values, color in series:
            ax.plot(rates, values, marker="o", linewidth=2.3, markersize=5.5, label=label, color=color)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Rate")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8.5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.20, hspace=0.24)
    plt.savefig(OUTPUT_ROOT / "llama_7b_completion_detail.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = [request_metrics(model, rate) for model in MODELS for rate in RATES]
    write_summary(metrics)
    write_report(metrics)
    plot_heatmaps(metrics)
    plot_llama7b_detail(metrics)
    print(f"Wrote {OUTPUT_ROOT / 'completion_pathology_summary.json'}")
    print(f"Wrote {OUTPUT_ROOT / 'completion_pathology_report.md'}")
    print(f"Wrote {OUTPUT_ROOT / 'completion_pathology_heatmaps.png'}")
    print(f"Wrote {OUTPUT_ROOT / 'llama_7b_completion_detail.png'}")


if __name__ == "__main__":
    main()

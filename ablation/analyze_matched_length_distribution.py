#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from ablate_live_length_terms import (
    build_decode_samples,
    build_prefill_samples,
    discover_exp_files,
)


RESULTS_ROOT = Path("/users/rh/DistServe/evaluation/2-benchmark-serving/result")
OUTPUT_ROOT = Path(__file__).resolve().parent / "results" / "matched_length_distribution"
DECODE_SPREAD_DIFF_CUTOFFS = [1, 32, 64, 128, 256, 512, 1024]
PREFILL_NEAR_MATCH_TOLERANCE = 32


def decode_spread(sample) -> int:
    return int(sample.max_context_len - sample.min_context_len)


def prefill_spread(sample) -> int:
    return int(sample.max_prompt_len - sample.min_prompt_len)


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def exact_group_coverage(samples: list, stage: str) -> dict[str, object]:
    grouped = defaultdict(list)
    for sample in samples:
        if sample.batch_size < 2:
            continue
        if stage == "decode" and getattr(sample, "split", None) != "small":
            continue
        total_len = sample.sum_context_len if stage == "decode" else sample.sum_prompt_len
        grouped[(sample.model_name, sample.rate, sample.batch_size, total_len)].append(sample)

    multi_groups = [group for group in grouped.values() if len(group) >= 2]
    varying_spread_groups = [
        group
        for group in multi_groups
        if len({decode_spread(s) if stage == "decode" else prefill_spread(s) for s in group}) >= 2
    ]
    return {
        "total_groups": len(grouped),
        "multi_sample_groups": len(multi_groups),
        "varying_spread_groups": len(varying_spread_groups),
        "samples_in_varying_spread_groups": sum(len(group) for group in varying_spread_groups),
    }


def build_decode_exact_pair_records(samples: list) -> list[dict[str, float | int | str]]:
    grouped = defaultdict(list)
    for sample in samples:
        if sample.batch_size < 2 or sample.split != "small":
            continue
        grouped[(sample.model_name, sample.rate, sample.batch_size, sample.sum_context_len)].append(sample)

    records: list[dict[str, float | int | str]] = []
    for (model_name, rate, batch_size, sum_context_len), group in grouped.items():
        spreads = [decode_spread(sample) for sample in group]
        if len(set(spreads)) < 2:
            continue
        min_spread = min(spreads)
        max_spread = max(spreads)
        low_samples = [sample.duration_ms for sample in group if decode_spread(sample) == min_spread]
        high_samples = [sample.duration_ms for sample in group if decode_spread(sample) == max_spread]
        low_duration_ms = mean(low_samples)
        high_duration_ms = mean(high_samples)
        delta_ms = high_duration_ms - low_duration_ms
        slowdown_pct = 100.0 * delta_ms / max(low_duration_ms, 1e-6)
        records.append(
            {
                "model_name": model_name,
                "rate": float(rate),
                "batch_size": int(batch_size),
                "sum_len": int(sum_context_len),
                "low_spread": int(min_spread),
                "high_spread": int(max_spread),
                "spread_diff": int(max_spread - min_spread),
                "low_duration_ms": float(low_duration_ms),
                "high_duration_ms": float(high_duration_ms),
                "delta_ms": float(delta_ms),
                "slowdown_pct": float(slowdown_pct),
            }
        )
    return records


def build_prefill_near_match_records(samples: list, sum_tolerance: int) -> list[dict[str, float | int | str]]:
    grouped = defaultdict(list)
    for sample in samples:
        if sample.batch_size < 2:
            continue
        grouped[(sample.model_name, sample.rate, sample.batch_size)].append(sample)

    records: list[dict[str, float | int | str]] = []
    for (model_name, rate, batch_size), group in grouped.items():
        sorted_group = sorted(group, key=lambda sample: (sample.sum_prompt_len, prefill_spread(sample)))
        used_indices: set[int] = set()
        for index, left in enumerate(sorted_group):
            if index in used_indices:
                continue
            best_match_index: int | None = None
            best_score: tuple[int, int] | None = None
            for other_index in range(index + 1, len(sorted_group)):
                if other_index in used_indices:
                    continue
                right = sorted_group[other_index]
                abs_sum_diff = abs(right.sum_prompt_len - left.sum_prompt_len)
                if abs_sum_diff > sum_tolerance:
                    break
                left_spread = prefill_spread(left)
                right_spread = prefill_spread(right)
                if left_spread == right_spread:
                    continue
                score = (abs_sum_diff, -abs(right_spread - left_spread))
                if best_score is None or score < best_score:
                    best_score = score
                    best_match_index = other_index
            if best_match_index is None:
                continue
            used_indices.add(index)
            used_indices.add(best_match_index)
            right = sorted_group[best_match_index]
            if prefill_spread(left) <= prefill_spread(right):
                low_sample, high_sample = left, right
            else:
                low_sample, high_sample = right, left
            delta_ms = high_sample.duration_ms - low_sample.duration_ms
            slowdown_pct = 100.0 * delta_ms / max(low_sample.duration_ms, 1e-6)
            records.append(
                {
                    "model_name": model_name,
                    "rate": float(rate),
                    "batch_size": int(batch_size),
                    "low_sum_len": int(low_sample.sum_prompt_len),
                    "high_sum_len": int(high_sample.sum_prompt_len),
                    "abs_sum_diff": int(abs(high_sample.sum_prompt_len - low_sample.sum_prompt_len)),
                    "low_spread": int(prefill_spread(low_sample)),
                    "high_spread": int(prefill_spread(high_sample)),
                    "spread_diff": int(prefill_spread(high_sample) - prefill_spread(low_sample)),
                    "low_duration_ms": float(low_sample.duration_ms),
                    "high_duration_ms": float(high_sample.duration_ms),
                    "delta_ms": float(delta_ms),
                    "slowdown_pct": float(slowdown_pct),
                }
            )
    return records


def summarize_pair_records(records: list[dict[str, float | int | str]]) -> dict[str, object]:
    if not records:
        return {
            "group_count": 0,
            "high_spread_slower_pct": 0.0,
            "median_delta_ms": 0.0,
            "mean_delta_ms": 0.0,
            "median_slowdown_pct": 0.0,
            "mean_slowdown_pct": 0.0,
            "per_model": {},
        }

    delta_values = [float(record["delta_ms"]) for record in records]
    slowdown_values = [float(record["slowdown_pct"]) for record in records]
    per_model = {}
    for model_name in sorted({str(record["model_name"]) for record in records}):
        model_records = [record for record in records if record["model_name"] == model_name]
        model_delta_values = [float(record["delta_ms"]) for record in model_records]
        model_slowdown_values = [float(record["slowdown_pct"]) for record in model_records]
        per_model[model_name] = {
            "group_count": len(model_records),
            "high_spread_slower_pct": 100.0 * sum(value > 0.0 for value in model_delta_values) / len(model_delta_values),
            "median_delta_ms": median(model_delta_values),
            "mean_delta_ms": mean(model_delta_values),
            "median_slowdown_pct": median(model_slowdown_values),
            "mean_slowdown_pct": mean(model_slowdown_values),
        }

    return {
        "group_count": len(records),
        "high_spread_slower_pct": 100.0 * sum(value > 0.0 for value in delta_values) / len(delta_values),
        "median_delta_ms": median(delta_values),
        "mean_delta_ms": mean(delta_values),
        "median_slowdown_pct": median(slowdown_values),
        "mean_slowdown_pct": mean(slowdown_values),
        "per_model": per_model,
    }


def summarize_decode_cutoffs(records: list[dict[str, float | int | str]]) -> dict[str, dict[str, float]]:
    summary = {}
    for cutoff in DECODE_SPREAD_DIFF_CUTOFFS:
        selected = [record for record in records if int(record["spread_diff"]) >= cutoff]
        stats = summarize_pair_records(selected)
        summary[str(cutoff)] = {
            "group_count": int(stats["group_count"]),
            "high_spread_slower_pct": float(stats["high_spread_slower_pct"]),
            "median_delta_ms": float(stats["median_delta_ms"]),
            "mean_delta_ms": float(stats["mean_delta_ms"]),
            "median_slowdown_pct": float(stats["median_slowdown_pct"]),
            "mean_slowdown_pct": float(stats["mean_slowdown_pct"]),
        }
    return summary


def top_examples(records: list[dict[str, float | int | str]], reverse: bool, limit: int = 8) -> list[dict[str, float | int | str]]:
    return sorted(records, key=lambda record: float(record["slowdown_pct"]), reverse=reverse)[:limit]


def write_report(
    output_path: Path,
    prefill_coverage: dict[str, object],
    decode_coverage: dict[str, object],
    decode_exact_summary: dict[str, object],
    decode_cutoff_summary: dict[str, dict[str, float]],
    prefill_near_summary: dict[str, object],
    decode_records: list[dict[str, float | int | str]],
    prefill_near_records: list[dict[str, float | int | str]],
) -> None:
    lines = [
        "# Matched Length Distribution Analysis",
        "",
        "This analysis asks a narrower question than the fit ablations:",
        "holding total tokens fixed inside a batch, does a more extreme within-batch length distribution run slower than a more uniform one?",
        "",
        "## Coverage",
        "",
        "| Stage | Matched key | Total groups | Multi-sample groups | Varying-spread groups | Samples in varying-spread groups |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| Prefill exact | `(model, rate, batch_size, sum_prompt_len)` | "
            f"{prefill_coverage['total_groups']} | {prefill_coverage['multi_sample_groups']} | "
            f"{prefill_coverage['varying_spread_groups']} | {prefill_coverage['samples_in_varying_spread_groups']} |"
        ),
        (
            f"| Decode exact | `(model, rate, batch_size, sum_context_len)` | "
            f"{decode_coverage['total_groups']} | {decode_coverage['multi_sample_groups']} | "
            f"{decode_coverage['varying_spread_groups']} | {decode_coverage['samples_in_varying_spread_groups']} |"
        ),
        "",
        "Exact prefill duplicates do not occur in this corpus, so the strict same-sum experiment is only available for decode.",
        "A small prefill near-match addendum is reported separately and should be treated as suggestive rather than decisive.",
        "",
        "## Decode Exact Match",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| usable matched groups | {decode_exact_summary['group_count']} |",
        f"| high-spread slower groups % | {decode_exact_summary['high_spread_slower_pct']:.2f} |",
        f"| median delta ms | {decode_exact_summary['median_delta_ms']:.3f} |",
        f"| mean delta ms | {decode_exact_summary['mean_delta_ms']:.3f} |",
        f"| median slowdown % | {decode_exact_summary['median_slowdown_pct']:.2f} |",
        f"| mean slowdown % | {decode_exact_summary['mean_slowdown_pct']:.2f} |",
        "",
        "### Decode Per Model",
        "",
        "| Model | Groups | High-spread slower % | Median delta ms | Median slowdown % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, metrics in decode_exact_summary["per_model"].items():
        lines.append(
            f"| {model_name} | {metrics['group_count']} | {metrics['high_spread_slower_pct']:.2f} | "
            f"{metrics['median_delta_ms']:.3f} | {metrics['median_slowdown_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Decode By Minimum Spread Difference",
            "",
            "| Min spread diff | Groups | High-spread slower % | Median delta ms | Median slowdown % |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for cutoff in DECODE_SPREAD_DIFF_CUTOFFS:
        metrics = decode_cutoff_summary[str(cutoff)]
        lines.append(
            f"| {cutoff} | {metrics['group_count']} | {metrics['high_spread_slower_pct']:.2f} | "
            f"{metrics['median_delta_ms']:.3f} | {metrics['median_slowdown_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Decode Strongest Positive Examples",
            "",
            "| Model | Rate | Batch | Sum len | Spread low->high | Runtime low->high ms | Slowdown % |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for record in top_examples(decode_records, reverse=True):
        lines.append(
            f"| {record['model_name']} | {record['rate']} | {record['batch_size']} | {record['sum_len']} | "
            f"{record['low_spread']} -> {record['high_spread']} | "
            f"{record['low_duration_ms']:.2f} -> {record['high_duration_ms']:.2f} | {record['slowdown_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            "### Decode Strongest Negative Examples",
            "",
            "| Model | Rate | Batch | Sum len | Spread low->high | Runtime low->high ms | Slowdown % |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for record in top_examples(decode_records, reverse=False):
        lines.append(
            f"| {record['model_name']} | {record['rate']} | {record['batch_size']} | {record['sum_len']} | "
            f"{record['low_spread']} -> {record['high_spread']} | "
            f"{record['low_duration_ms']:.2f} -> {record['high_duration_ms']:.2f} | {record['slowdown_pct']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Prefill Near-Match Addendum",
            "",
            (
                f"Near-match rule: same `(model, rate, batch_size)` and total prompt length difference <= "
                f"{PREFILL_NEAR_MATCH_TOLERANCE} tokens."
            ),
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| near-matched pairs | {prefill_near_summary['group_count']} |",
            f"| high-spread slower pairs % | {prefill_near_summary['high_spread_slower_pct']:.2f} |",
            f"| median delta ms | {prefill_near_summary['median_delta_ms']:.3f} |",
            f"| mean delta ms | {prefill_near_summary['mean_delta_ms']:.3f} |",
            f"| median slowdown % | {prefill_near_summary['median_slowdown_pct']:.2f} |",
            f"| mean slowdown % | {prefill_near_summary['mean_slowdown_pct']:.2f} |",
            "",
            "Because this section only has a handful of pairs, it is not strong enough to carry the main argument on its own.",
            "",
        ]
    )
    if prefill_near_records:
        lines.extend(
            [
                "| Model | Rate | Batch | Sum len low->high | Spread low->high | Runtime low->high ms | Slowdown % |",
                "| --- | ---: | ---: | --- | --- | --- | ---: |",
            ]
        )
        for record in prefill_near_records:
            lines.append(
                f"| {record['model_name']} | {record['rate']} | {record['batch_size']} | "
                f"{record['low_sum_len']} -> {record['high_sum_len']} | "
                f"{record['low_spread']} -> {record['high_spread']} | "
                f"{record['low_duration_ms']:.2f} -> {record['high_duration_ms']:.2f} | {record['slowdown_pct']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "For decode, the same total context tokens do not imply the same runtime.",
            "When the spread difference is large, the extreme distribution is more often slower, not just different.",
            "That is direct evidence that total-work terms alone miss a real within-batch critical-path effect.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def make_plot(
    output_path: Path,
    decode_records: list[dict[str, float | int | str]],
    decode_cutoff_summary: dict[str, dict[str, float]],
    prefill_near_records: list[dict[str, float | int | str]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    decode_low = [float(record["low_duration_ms"]) for record in decode_records]
    decode_high = [float(record["high_duration_ms"]) for record in decode_records]
    decode_spread_diff = [float(record["spread_diff"]) for record in decode_records]
    max_axis = max(decode_low + decode_high) if decode_records else 1.0

    scatter = axes[0].scatter(
        decode_low,
        decode_high,
        c=decode_spread_diff,
        cmap="viridis",
        alpha=0.45,
        s=18,
        linewidths=0,
    )
    axes[0].plot([0, max_axis], [0, max_axis], color="#444444", linestyle="--", linewidth=1.0)
    axes[0].set_title("Decode Exact Matches")
    axes[0].set_xlabel("Low-spread runtime (ms)")
    axes[0].set_ylabel("High-spread runtime (ms)")
    cbar = fig.colorbar(scatter, ax=axes[0])
    cbar.set_label("Spread difference")

    cutoff_x = DECODE_SPREAD_DIFF_CUTOFFS
    slowdown_y = [decode_cutoff_summary[str(cutoff)]["median_slowdown_pct"] for cutoff in cutoff_x]
    positive_y = [decode_cutoff_summary[str(cutoff)]["high_spread_slower_pct"] for cutoff in cutoff_x]
    axes[1].plot(cutoff_x, slowdown_y, marker="o", color="#1f77b4", label="Median slowdown %")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Minimum spread difference")
    axes[1].set_ylabel("Median slowdown %", color="#1f77b4")
    axes[1].tick_params(axis="y", labelcolor="#1f77b4")
    twin = axes[1].twinx()
    twin.plot(cutoff_x, positive_y, marker="s", color="#d62728", label="High-spread slower %")
    twin.set_ylabel("High-spread slower %", color="#d62728")
    twin.tick_params(axis="y", labelcolor="#d62728")
    axes[1].set_title("Decode Effect Strength vs Spread Gap")

    if prefill_near_records:
        prefill_low = [float(record["low_duration_ms"]) for record in prefill_near_records]
        prefill_high = [float(record["high_duration_ms"]) for record in prefill_near_records]
        prefill_spread_diff = [float(record["spread_diff"]) for record in prefill_near_records]
        prefill_axis = max(prefill_low + prefill_high)
        scatter_prefill = axes[2].scatter(
            prefill_low,
            prefill_high,
            c=prefill_spread_diff,
            cmap="magma",
            alpha=0.8,
            s=55,
            linewidths=0,
        )
        axes[2].plot([0, prefill_axis], [0, prefill_axis], color="#444444", linestyle="--", linewidth=1.0)
        axes[2].set_title("Prefill Near Matches")
        axes[2].set_xlabel("Low-spread runtime (ms)")
        axes[2].set_ylabel("High-spread runtime (ms)")
        cbar_prefill = fig.colorbar(scatter_prefill, ax=axes[2])
        cbar_prefill.set_label("Spread difference")
    else:
        axes[2].axis("off")
        axes[2].text(
            0.5,
            0.55,
            "No usable prefill near-match pairs",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[2].text(
            0.5,
            0.42,
            "Exact same-sum prefill groups are absent\nin the current benchmark corpus.",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )

    fig.suptitle("Uniform vs Extreme Batch Length Distribution", fontsize=15)
    fig.text(
        0.5,
        0.93,
        "Decode uses exact same-sum matches. Prefill is shown only as a small near-match addendum.",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    exp_files = discover_exp_files(RESULTS_ROOT)

    prefill_samples = []
    decode_samples = []
    for exp_file in exp_files:
        prefill_samples.extend(build_prefill_samples(exp_file))
        decode_samples.extend(build_decode_samples(exp_file))

    prefill_coverage = exact_group_coverage(prefill_samples, stage="prefill")
    decode_coverage = exact_group_coverage(decode_samples, stage="decode")
    decode_records = build_decode_exact_pair_records(decode_samples)
    prefill_near_records = build_prefill_near_match_records(prefill_samples, PREFILL_NEAR_MATCH_TOLERANCE)

    decode_exact_summary = summarize_pair_records(decode_records)
    decode_cutoff_summary = summarize_decode_cutoffs(decode_records)
    prefill_near_summary = summarize_pair_records(prefill_near_records)

    summary = {
        "prefill_exact_coverage": prefill_coverage,
        "decode_exact_coverage": decode_coverage,
        "decode_exact_summary": decode_exact_summary,
        "decode_cutoff_summary": decode_cutoff_summary,
        "prefill_near_match_tolerance": PREFILL_NEAR_MATCH_TOLERANCE,
        "prefill_near_match_summary": prefill_near_summary,
        "top_decode_positive_examples": top_examples(decode_records, reverse=True),
        "top_decode_negative_examples": top_examples(decode_records, reverse=False),
        "prefill_near_match_records": prefill_near_records,
    }
    (OUTPUT_ROOT / "matched_length_distribution_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        OUTPUT_ROOT / "matched_length_distribution_report.md",
        prefill_coverage=prefill_coverage,
        decode_coverage=decode_coverage,
        decode_exact_summary=decode_exact_summary,
        decode_cutoff_summary=decode_cutoff_summary,
        prefill_near_summary=prefill_near_summary,
        decode_records=decode_records,
        prefill_near_records=prefill_near_records,
    )
    make_plot(
        OUTPUT_ROOT / "matched_length_distribution_plot.png",
        decode_records=decode_records,
        decode_cutoff_summary=decode_cutoff_summary,
        prefill_near_records=prefill_near_records,
    )

    print(f"Wrote {OUTPUT_ROOT / 'matched_length_distribution_summary.json'}")
    print(f"Wrote {OUTPUT_ROOT / 'matched_length_distribution_report.md'}")
    print(f"Wrote {OUTPUT_ROOT / 'matched_length_distribution_plot.png'}")


if __name__ == "__main__":
    main()

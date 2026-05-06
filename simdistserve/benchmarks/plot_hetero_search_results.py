#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


MODE_LABELS = {
    "milp": "Hetero MILP",
    "no_cross": "No Cross",
    "cuda_prefill_ascend_decode": "CUDA-P / Ascend-D",
}
THESIS_MODE_LABELS = {
    "milp": "本文方法",
    "no_cross": "不跨设备池",
    "cuda_prefill_ascend_decode": "固定分工",
}
MODE_ORDER = ("no_cross", "cuda_prefill_ascend_decode", "milp")
THESIS_MODE_ORDER = ("milp", "no_cross", "cuda_prefill_ascend_decode")
ROLE_KEYS = ("cuda_prefill", "cuda_decode", "ascend_prefill", "ascend_decode")
ROLE_LABELS = {
    "cuda_prefill": "CUDA prefill",
    "cuda_decode": "CUDA decode",
    "ascend_prefill": "Ascend prefill",
    "ascend_decode": "Ascend decode",
}
THESIS_ROLE_LABELS = {
    "cuda_prefill": "NVIDIA预填充",
    "cuda_decode": "NVIDIA解码",
    "ascend_prefill": "Ascend预填充",
    "ascend_decode": "Ascend解码",
}
ROLE_COLORS = {
    "cuda_prefill": "#4c78a8",
    "cuda_decode": "#72b7b2",
    "ascend_prefill": "#f58518",
    "ascend_decode": "#e45756",
}
FLOW_KEYS = ("x_cc", "x_ca", "x_ac", "x_aa")
FLOW_LABELS = {
    "x_cc": "CUDA P -> CUDA D",
    "x_ca": "CUDA P -> Ascend D",
    "x_ac": "Ascend P -> CUDA D",
    "x_aa": "Ascend P -> Ascend D",
}
THESIS_FLOW_LABELS = {
    "x_cc": "NVIDIA→NVIDIA",
    "x_ca": "NVIDIA→Ascend",
    "x_ac": "Ascend→NVIDIA",
    "x_aa": "Ascend→Ascend",
}
FLOW_COLORS = {
    "x_cc": "#4c78a8",
    "x_ca": "#9ecae9",
    "x_ac": "#f58518",
    "x_aa": "#e45756",
}
THESIS_MODE_COLORS = {
    "milp": "#2f7d6d",
    "no_cross": "#7c8a99",
    "cuda_prefill_ascend_decode": "#d97745",
}


def _font(font_path: Path | None) -> FontProperties | None:
    if font_path and font_path.exists():
        return FontProperties(fname=str(font_path))
    return None


def load_rows(result_root: Path, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for model in models:
        for mode in MODE_ORDER:
            path = result_root / model / mode / "search_4nodes_high_affinity.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            best = data.get("best_config") or {}
            flows = best.get("flows") or {}
            profile_status = data.get("best_config_profile_status") or {}
            row = {
                "model": model,
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
                "estimated_goodput": float(best.get("estimated_goodput", 0.0)),
                "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
                "profile_seconds": float(data.get("timing", {}).get("profile_seconds", 0.0)),
                "allocation_milp_seconds": float(data.get("timing", {}).get("allocation_milp_seconds", 0.0)),
                "uses_capped_profile": bool(profile_status.get("uses_capped_profile", False)),
                "uses_timed_out_profile": bool(profile_status.get("uses_timed_out_profile", False)),
                "num_capped_roles": len(profile_status.get("capped_roles") or []),
                "num_timed_out_roles": len(profile_status.get("timed_out_roles") or []),
                "path": str(path),
            }
            for role in ROLE_KEYS:
                role_config = best.get(role) or {}
                row[f"instances_{role}"] = int(role_config.get("num_instances") or 0)
                shape = role_config.get("shape") or {}
                row[f"shape_{role}"] = (
                    f"tp{shape.get('tp')}_ppl{shape.get('pp_local')}_ppc{shape.get('pp_cross')}"
                    if shape
                    else ""
                )
            for flow in FLOW_KEYS:
                row[flow] = float(flows.get(flow, 0.0))
            rows.append(
                row
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_goodput(rows: list[dict], models: list[str], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    values = {
        mode: [
            next((row["estimated_goodput"] for row in rows if row["model"] == model and row["mode"] == mode), 0.0)
            for model in models
        ]
        for mode in MODE_ORDER
    }

    x = np.arange(len(models))
    width = 0.25
    colors = {
        "no_cross": "#7c8a99",
        "cuda_prefill_ascend_decode": "#d97745",
        "milp": "#2f7d6d",
    }

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, mode in enumerate(MODE_ORDER):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values[mode], width, label=MODE_LABELS[mode], color=colors[mode])
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    ax.set_title("Heterogeneous DistServe Search Goodput")
    ax.set_ylabel("Estimated goodput (req/s)")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_fig_5_2_real_bandwidth_goodput(
    rows: list[dict],
    models: list[str],
    output_png: Path,
    font_path: Path | None = None,
) -> None:
    """Plot thesis Figure 5-2 from real-bandwidth heterogeneous search results."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    font = _font(font_path)
    values = {
        mode: [
            next((row["estimated_goodput"] for row in rows if row["model"] == model and row["mode"] == mode), 0.0)
            for model in models
        ]
        for mode in THESIS_MODE_ORDER
    }

    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for i, mode in enumerate(THESIS_MODE_ORDER):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            values[mode],
            width,
            label=THESIS_MODE_LABELS[mode],
            color=THESIS_MODE_COLORS[mode],
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_title("有效吞吐对比", fontproperties=font, fontsize=14)
    ax.set_ylabel("有效吞吐(req/s)", fontproperties=font)
    ax.set_xlabel("模型", fontproperties=font)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontproperties=font)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(prop=font, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def _rows_by_model_mode(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["model"], row["mode"]): row for row in rows}


def _x_labels(models: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    keys = [(model, mode) for model in models for mode in MODE_ORDER]
    labels = [f"{model}\n{MODE_LABELS[mode]}" for model, mode in keys]
    return keys, labels


def plot_instance_allocation(rows: list[dict], models: list[str], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    by_key = _rows_by_model_mode(rows)
    keys, labels = _x_labels(models)
    x = np.arange(len(keys))
    bottoms = np.zeros(len(keys))

    fig, ax = plt.subplots(figsize=(14, 6.2))
    for role in ROLE_KEYS:
        values = np.array([
            by_key.get((model, mode), {}).get(f"instances_{role}", 0)
            for model, mode in keys
        ], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=ROLE_COLORS[role],
            label=ROLE_LABELS[role],
            width=0.72,
        )
        for i, value in enumerate(values):
            if value > 0:
                ax.text(
                    x[i],
                    bottoms[i] + value / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 2 else "#202020",
                )
        bottoms += values

    ax.set_title("Search Resource Allocation")
    ax.set_ylabel("Number of instances")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_fig_5_3_real_bandwidth_allocation(
    rows: list[dict],
    models: list[str],
    output_png: Path,
    font_path: Path | None = None,
) -> None:
    """Plot thesis Figure 5-3: resource allocation of the proposed method."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    font = _font(font_path)
    by_key = _rows_by_model_mode(rows)
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for role in ROLE_KEYS:
        values = np.array([
            by_key.get((model, "milp"), {}).get(f"instances_{role}", 0)
            for model in models
        ], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=ROLE_COLORS[role],
            label=THESIS_ROLE_LABELS[role],
            width=0.62,
        )
        for i, value in enumerate(values):
            if value > 0:
                ax.text(
                    x[i],
                    bottoms[i] + value / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if value >= 2 else "#202020",
                    fontproperties=font,
                )
        bottoms += values

    ax.set_title("本文方法的资源分配", fontproperties=font, fontsize=15, pad=10)
    ax.set_ylabel("实例数量", fontproperties=font)
    ax.set_xlabel("模型", fontproperties=font)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontproperties=font)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(prop=font, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_flow_stacked(rows: list[dict], models: list[str], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    by_key = _rows_by_model_mode(rows)
    keys, labels = _x_labels(models)
    x = np.arange(len(keys))
    bottoms = np.zeros(len(keys))

    fig, ax = plt.subplots(figsize=(14, 6.2))
    for flow in FLOW_KEYS:
        values = np.array([
            by_key.get((model, mode), {}).get(flow, 0.0)
            for model, mode in keys
        ], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=FLOW_COLORS[flow],
            label=FLOW_LABELS[flow],
            width=0.72,
        )
        for i, value in enumerate(values):
            if value >= max(1.0, 0.08 * max(1.0, values.max())):
                ax.text(
                    x[i],
                    bottoms[i] + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white",
                )
        bottoms += values

    ax.set_title("Search Request Flow Decomposition")
    ax.set_ylabel("Flow goodput (req/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_fig_5_4_real_bandwidth_flow(
    rows: list[dict],
    models: list[str],
    output_png: Path,
    font_path: Path | None = None,
) -> None:
    """Plot thesis Figure 5-4: request flow decomposition of the proposed method."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    font = _font(font_path)
    by_key = _rows_by_model_mode(rows)
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    max_value = max(
        (
            by_key.get((model, "milp"), {}).get(flow, 0.0)
            for model in models
            for flow in FLOW_KEYS
        ),
        default=1.0,
    )
    for flow in FLOW_KEYS:
        values = np.array([
            by_key.get((model, "milp"), {}).get(flow, 0.0)
            for model in models
        ], dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=FLOW_COLORS[flow],
            label=THESIS_FLOW_LABELS[flow],
            width=0.62,
        )
        for i, value in enumerate(values):
            if value >= max(1.0, 0.08 * max(1.0, max_value)):
                ax.text(
                    x[i],
                    bottoms[i] + value / 2,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontproperties=font,
                )
        bottoms += values

    ax.set_title("本文方法的请求流向", fontproperties=font, fontsize=15, pad=10)
    ax.set_ylabel("流量有效吞吐(req/s)", fontproperties=font)
    ax.set_xlabel("模型", fontproperties=font)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontproperties=font)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(prop=font, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_flow_matrix(rows: list[dict], models: list[str], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    by_key = _rows_by_model_mode(rows)
    max_flow = max((row[flow] for row in rows for flow in FLOW_KEYS), default=1.0)
    max_flow = max(max_flow, 1.0)

    fig, axes = plt.subplots(
        len(models),
        len(MODE_ORDER),
        figsize=(3.4 * len(MODE_ORDER), 2.65 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, model in enumerate(models):
        for col_idx, mode in enumerate(MODE_ORDER):
            ax = axes[row_idx][col_idx]
            row = by_key.get((model, mode))
            if row is None:
                ax.axis("off")
                continue
            matrix = np.array(
                [
                    [row["x_cc"], row["x_ca"]],
                    [row["x_ac"], row["x_aa"]],
                ],
                dtype=float,
            )
            im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=max_flow)
            for i in range(2):
                for j in range(2):
                    value = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if value > 0.55 * max_flow else "#202020",
                    )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["CUDA D", "Ascend D"], fontsize=8)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["CUDA P", "Ascend P"], fontsize=8)
            if row_idx == 0:
                ax.set_title(MODE_LABELS[mode], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(model, fontsize=10)
            ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Flow goodput (req/s)")
    fig.suptitle("Search Flow Matrix: Prefill Device -> Decode Device")
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_png.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Plot heterogeneous search goodput results.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=repo_root / "simdistserve" / "hetero" / "results" / "search",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "simdistserve" / "hetero" / "results" / "plots",
    )
    parser.add_argument(
        "--thesis-figures-dir",
        type=Path,
        default=repo_root / "docs" / "figures",
        help="Directory for thesis figures. Figure 5-2 is written here.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=repo_root / "docs" / "fonts" / "NotoSerifSC-Regular.ttf",
        help="Chinese font used by thesis figures.",
    )
    parser.add_argument("--models", nargs="+", default=["llama_1B", "llama_3B", "llama_7B", "llama_8B"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.result_root, args.models)
    if not rows:
        raise SystemExit(f"No result rows found under {args.result_root}")
    csv_path = args.output_dir / "hetero_search_goodput_summary.csv"
    png_path = args.output_dir / "hetero_search_goodput.png"
    write_csv(rows, csv_path)
    plot_goodput(rows, args.models, png_path)
    fig_5_2_png = args.thesis_figures_dir / "fig_5_2_real_bandwidth_goodput.png"
    fig_5_3_png = args.thesis_figures_dir / "fig_5_3_real_bandwidth_allocation.png"
    fig_5_4_png = args.thesis_figures_dir / "fig_5_4_real_bandwidth_flow.png"
    plot_fig_5_2_real_bandwidth_goodput(rows, args.models, fig_5_2_png, args.font_path)
    plot_fig_5_3_real_bandwidth_allocation(rows, args.models, fig_5_3_png, args.font_path)
    plot_fig_5_4_real_bandwidth_flow(rows, args.models, fig_5_4_png, args.font_path)
    allocation_png = args.output_dir / "hetero_search_instance_allocation.png"
    flow_stacked_png = args.output_dir / "hetero_search_flow_stacked.png"
    flow_matrix_png = args.output_dir / "hetero_search_flow_matrix.png"
    plot_instance_allocation(rows, args.models, allocation_png)
    plot_flow_stacked(rows, args.models, flow_stacked_png)
    plot_flow_matrix(rows, args.models, flow_matrix_png)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "goodput_png": str(png_path),
                "goodput_pdf": str(png_path.with_suffix(".pdf")),
                "fig_5_2_png": str(fig_5_2_png),
                "fig_5_2_pdf": str(fig_5_2_png.with_suffix(".pdf")),
                "fig_5_3_png": str(fig_5_3_png),
                "fig_5_3_pdf": str(fig_5_3_png.with_suffix(".pdf")),
                "fig_5_4_png": str(fig_5_4_png),
                "fig_5_4_pdf": str(fig_5_4_png.with_suffix(".pdf")),
                "allocation_png": str(allocation_png),
                "allocation_pdf": str(allocation_png.with_suffix(".pdf")),
                "flow_stacked_png": str(flow_stacked_png),
                "flow_stacked_pdf": str(flow_stacked_png.with_suffix(".pdf")),
                "flow_matrix_png": str(flow_matrix_png),
                "flow_matrix_pdf": str(flow_matrix_png.with_suffix(".pdf")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

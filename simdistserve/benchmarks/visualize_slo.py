#!/usr/bin/env python3
"""
Visualize SLO satisfaction comparisons as polished 2x2 model dashboards.

Each model gets one figure with four panels:
    Prefill / Decode / Total / Both

Each panel shows:
    - Actual benchmark line
    - Simulator prediction line
    - Shaded absolute-gap band
    - Mean/max absolute gap summary

Usage:

VLLM Side:
python ./visualize_slo.py \
    --input_dir "./results/slo/vllm_ascend/compared/" \
    --output_dir "./results/slo/vllm_ascend/plots/"

CUDA Side:
python ./visualize_slo.py \
    --input_dir "./results/slo/distserve_cuda/compared/" \
    --output_dir "./results/slo/distserve_cuda/plots/"
"""

import os
import re
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_comparison_txt(filepath):
    """
    解析 comparison.txt 文件，提取各项指标的 Exp 和 CSV 满足率（百分比）
    
    参数:
        filepath: comparison.txt 文件路径
    
    返回:
        dict: {指标名称: (exp_percent, csv_percent)}
        指标名称如 'Prefill SLO met rate', 'Decode SLO met rate' 等
    """
    target_metrics = [
        'Prefill SLO met rate',
        'Decode SLO met rate',
        'Total SLO met rate',
        'Both (Prefill+Decode) SLO met rate'
    ]
    
    results = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not any(metric in line for metric in target_metrics):
                continue
            percents = re.findall(r'(\d+\.?\d*)%', line)
            if len(percents) >= 2:
                exp_percent = float(percents[0])
                csv_percent = float(percents[1])
                for metric in target_metrics:
                    if metric in line:
                        results[metric] = (exp_percent, csv_percent)
                        break
    return results


def collect_all_data(root_dir):
    """
    遍历根目录，收集所有模型在不同 rate 下的数据
    
    返回:
        dict: {模型名称: {rate数值: {指标: (exp, csv)}}}
    """
    all_data = defaultdict(dict)
    
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")
    
    for model_name in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        for rate_dir in os.listdir(model_path):
            rate_path = os.path.join(model_path, rate_dir)
            if not os.path.isdir(rate_path):
                continue
            
            match = re.match(r'rate_([\d\.]+)', rate_dir)
            if not match:
                continue
            rate_value = float(match.group(1))
            
            txt_file = os.path.join(rate_path, 'comparison.txt')
            if not os.path.isfile(txt_file):
                print(f"警告: 未找到 {txt_file}，跳过该 rate")
                continue
            
            try:
                metrics = parse_comparison_txt(txt_file)
                if metrics:
                    all_data[model_name][rate_value] = metrics
                else:
                    print(f"警告: {txt_file} 中未解析到有效数据")
            except Exception as e:
                print(f"解析 {txt_file} 时出错: {e}")
    
    return all_data


def style_axis(ax):
    ax.set_facecolor("#fbfbf8")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, color="#666666")
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")


def add_gap_annotation(ax, exp_vals, csv_vals):
    diffs = np.abs(np.array(exp_vals) - np.array(csv_vals))
    valid_diffs = diffs[~np.isnan(diffs)]
    if len(valid_diffs) == 0:
        return
    mean_gap = float(np.mean(valid_diffs))
    max_gap = float(np.max(valid_diffs))
    ax.text(
        0.03,
        0.96,
        f"mean |Δ| = {mean_gap:.1f}%\nmax |Δ| = {max_gap:.1f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d7d7cf", alpha=0.92),
    )


def plot_model_subplots(model_name, data, output_path):
    """
    为单个模型绘制 2x2 子图对比图
    
    参数:
        model_name: 模型名称（用于标题和文件名）
        data: dict {rate: {指标: (exp, csv)}}
        output_path: 保存图片的完整路径
    """
    sorted_rates = sorted(data.keys())
    
    # 指标映射：原始名称 -> 子图显示名称
    metrics_info = {
        'Prefill SLO met rate': ('Prefill SLO Met Rate', 'Prefill'),
        'Decode SLO met rate': ('Decode SLO Met Rate', 'Decode'),
        'Total SLO met rate': ('Total SLO Met Rate', 'Total'),
        'Both (Prefill+Decode) SLO met rate': ('Both SLO Met Rate', 'Both')
    }
    
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor="#f3f0e8")
    fig.suptitle(f'SLO Satisfaction Comparison: {model_name}', fontsize=17, y=0.98, weight='bold')
    fig.text(
        0.5,
        0.945,
        'Actual benchmark vs simulator prediction across request rates',
        ha='center',
        va='center',
        fontsize=10,
        color='#5f5a53',
    )
    
    # 将 axes 展平为一维列表方便索引
    ax_list = axes.flatten()
    
    for idx, (metric_full, (display_name, short_name)) in enumerate(metrics_info.items()):
        ax = ax_list[idx]
        
        # 提取 Exp 和 CSV 数据
        exp_vals = []
        csv_vals = []
        for rate in sorted_rates:
            if metric_full in data[rate]:
                exp, csv = data[rate][metric_full]
                exp_vals.append(exp)
                csv_vals.append(csv)
            else:
                exp_vals.append(np.nan)
                csv_vals.append(np.nan)
        
        style_axis(ax)

        exp_arr = np.array(exp_vals, dtype=float)
        csv_arr = np.array(csv_vals, dtype=float)
        valid_mask = ~(np.isnan(exp_arr) | np.isnan(csv_arr))

        if np.any(valid_mask):
            ax.fill_between(
                np.array(sorted_rates)[valid_mask],
                exp_arr[valid_mask],
                csv_arr[valid_mask],
                color="#e8b85c",
                alpha=0.16,
                linewidth=0,
                label="Absolute Gap",
            )

        ax.plot(
            sorted_rates,
            exp_vals,
            marker='o',
            linestyle='-',
            color='#1f4e79',
            linewidth=2.4,
            markersize=6,
            label='Actual',
        )
        ax.plot(
            sorted_rates,
            csv_vals,
            marker='s',
            linestyle='--',
            color='#c46a2c',
            linewidth=2.2,
            markersize=5.5,
            label='Simulator',
        )
        
        # 设置子图标题和标签
        ax.set_title(display_name, fontsize=12, pad=10, weight='bold')
        ax.set_xlabel('Request Rate', fontsize=10)
        ax.set_ylabel('SLO Met Rate (%)', fontsize=10)
        ax.legend(loc='lower left', fontsize=8.5, frameon=False)
        add_gap_annotation(ax, exp_vals, csv_vals)
        
        # 设置 y 轴范围 0-100（留一点边距）
        ax.set_ylim(-2, 102)
        ax.set_xlim(min(sorted_rates) - 0.1, max(sorted_rates) + 0.1)
    
    # 如果某个子图没有数据（理论上不会），隐藏它
    for i in range(len(metrics_info), 4):
        ax_list[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.16, hspace=0.24)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 DistServe vs simdistserve SLO 满足率对比图（2x2子图）"
    )
    parser.add_argument(
        '--input_dir', type=str, default='./compared',
        help='包含模型子目录的根目录（默认为 ./compared）'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='保存图片的输出目录（将自动创建）'
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"正在扫描目录: {args.input_dir}")
    all_data = collect_all_data(args.input_dir)
    
    if not all_data:
        print("未找到任何有效数据，请检查输入目录结构。")
        return
    
    print(f"发现 {len(all_data)} 个模型: {list(all_data.keys())}")
    
    for model_name, model_data in all_data.items():
        if not model_data:
            print(f"模型 {model_name} 无有效数据，跳过")
            continue
        
        safe_name = model_name.replace('/', '_')
        output_path = os.path.join(args.output_dir, f"{safe_name}_slo_subplots.png")
        
        print(f"正在绘制 {model_name} ...")
        plot_model_subplots(model_name, model_data, output_path)
    
    print("全部完成。")


if __name__ == "__main__":
    main()

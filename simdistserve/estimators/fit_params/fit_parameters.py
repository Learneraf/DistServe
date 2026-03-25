#!/usr/bin/env python3
"""
根据测量时间反向推导出time_estimator.py中使用的a,b,c参数

According to the measured time in CSV_PATH, we can derive the a,b,c parameters used in time_estimator.py
We use the least squares method to fit the parameters.
Finally, we save the fit results to OUTPUT_JSON_PATH, and the charts to OUTPUT_PLOT_DIR.
"""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("/users/rh/DistServe/evaluation/0-test-single-forward-performance/result/read_db.csv")
OUTPUT_PLOT_DIR = Path("/users/rh/DistServe/simdistserve/estimators/fit_params/plots")
OUTPUT_JSON_PATH = Path("/users/rh/DistServe/simdistserve/estimators/profile_data/profiler-a100-80g.distserve.fitted.json")

OUTPUT_PLOT_DIR.mkdir(exist_ok=True)



def fit_prefill_params(data):
    """
    拟合预填充阶段的a,b,c参数
    公式：delay = a + b * num_total_tokens + c * sum_num_tokens_sqr
    """
    X = []
    y = []
    
    for row in data:
        batch_size = row['batch_size']
        input_len = row['input_len']
        prefill_time = row['avg_prefill_time_usage']
        
        num_total_tokens = batch_size * input_len
        sum_num_tokens_sqr = batch_size * (input_len ** 2)
        
        X.append([1, num_total_tokens, sum_num_tokens_sqr])
        y.append(prefill_time)
    
    # 使用最小二乘法拟合
    X = np.array(X)
    y = np.array(y)
    params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    a, b, c = params
    return [float(a), float(b), float(c)]


def fit_decode_params(data, threshold=95):
    """
    拟合解码阶段的a,b,c参数
    公式：delay = a + b * num_total_tokens + c * batch_size
    """
    # 分离小批量和大批量数据
    small_bs_data = [row for row in data if row['batch_size'] < threshold]
    large_bs_data = [row for row in data if row['batch_size'] >= threshold]
    
    def fit_bs_params(bs_data):
        X = []
        y = []
        
        for row in bs_data:
            batch_size = row['batch_size']
            output_len = row['output_len']
            decode_time = row['avg_decoding_time_usage']
            
            num_total_tokens = batch_size * (output_len - 1)  # 第一个token在prefill阶段生成
            
            X.append([1, num_total_tokens, batch_size])
            y.append(decode_time)
        
        if not X:
            return [0, 0, 0]  # 如果没有数据，返回默认值
        
        # 使用最小二乘法拟合
        X = np.array(X)
        y = np.array(y)
        params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        a, b, c = params
        return [float(a), float(b), float(c)]
    
    small_bs_params = fit_bs_params(small_bs_data)
    large_bs_params = fit_bs_params(large_bs_data)
    
    return small_bs_params, large_bs_params


def calculate_prefill_time(params, batch_size, input_len):
    """
    使用拟合参数计算预填充时间
    """
    a, b, c = params
    num_total_tokens = batch_size * input_len
    sum_num_tokens_sqr = batch_size * (input_len ** 2)
    return a + b * num_total_tokens + c * sum_num_tokens_sqr


def calculate_decode_time(params, batch_size, output_len):
    """
    使用拟合参数计算解码时间
    """
    a, b, c = params
    num_total_tokens = batch_size * (output_len - 1)  # 第一个token在prefill阶段生成
    return a + b * num_total_tokens + c * batch_size


def plot_prefill_fit(data, params, model, tp):
    """
    绘制预填充时间的拟合曲线
    """
    # 准备数据
    actual_times = []
    predicted_times = []
    total_tokens = []
    
    for row in data:
        batch_size = row['batch_size']
        input_len = row['input_len']
        actual_time = row['avg_prefill_time_usage']
        
        total_token = batch_size * input_len
        predicted_time = calculate_prefill_time(params, batch_size, input_len)
        
        actual_times.append(actual_time)
        predicted_times.append(predicted_time)
        total_tokens.append(total_token)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制原始数据点
    plt.scatter(total_tokens, actual_times, label='Actual Runtime', alpha=0.6)
    
    # 绘制拟合曲线
    sorted_tokens = sorted(set(total_tokens))
    sorted_predicted = [calculate_prefill_time(params, 1, token) for token in sorted_tokens]  # 假设batch_size=1
    plt.plot(sorted_tokens, sorted_predicted, 'r-', label='Fit Curve', linewidth=2)
    
    # 设置图表属性
    plt.title(f'Prefill Time Fit - {model} (TP={tp})')
    plt.xlabel('Total Tokens')
    plt.ylabel('Runtime (ms)')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    output_path = OUTPUT_PLOT_DIR / f"prefill_fit_{model.replace('/', '_')}_tp{tp}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Prefill Time Fit Chart Saved to: {output_path}")


def plot_decode_fit(data, small_bs_params, large_bs_params, model, tp, threshold=95):
    """
    绘制解码时间的拟合曲线
    """
    # 准备数据
    actual_times = []
    predicted_times = []
    batch_sizes = []
    
    for row in data:
        batch_size = row['batch_size']
        output_len = row['output_len']
        actual_time = row['avg_decoding_time_usage']
        
        if batch_size < threshold:
            predicted_time = calculate_decode_time(small_bs_params, batch_size, output_len)
        else:
            predicted_time = calculate_decode_time(large_bs_params, batch_size, output_len)
        
        actual_times.append(actual_time)
        predicted_times.append(predicted_time)
        batch_sizes.append(batch_size)
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 绘制原始数据点
    plt.scatter(batch_sizes, actual_times, label='Actual Runtime', alpha=0.6)
    
    # 绘制拟合曲线
    sorted_bs = sorted(set(batch_sizes))
    small_bs_range = [bs for bs in sorted_bs if bs < threshold]
    large_bs_range = [bs for bs in sorted_bs if bs >= threshold]
    
    if small_bs_range:
        small_bs_predicted = [calculate_decode_time(small_bs_params, bs, 16) for bs in small_bs_range]  # 假设output_len=16
        plt.plot(small_bs_range, small_bs_predicted, 'r-', label='Small Batch Fit Curve', linewidth=2)
    
    if large_bs_range:
        large_bs_predicted = [calculate_decode_time(large_bs_params, bs, 16) for bs in large_bs_range]  # 假设output_len=16
        plt.plot(large_bs_range, large_bs_predicted, 'g-', label='Large Batch Fit Curve', linewidth=2)
    
    # 设置图表属性
    plt.title(f'Decodeoding Time Fit - {model} (TP={tp})')
    plt.xlabel('Batch Size')
    plt.ylabel('Runtime (ms)')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    output_path = OUTPUT_PLOT_DIR / f"decode_fit_{model.replace('/', '_')}_tp{tp}.png"
    plt.savefig(output_path)
    plt.close()
    
    print(f"Decodeoding Time Fit Chart Saved to: {output_path}")


def main():
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Fit Parameters')
    parser.add_argument('--visualize', action='store_true', help='Fit Visualization')
    args = parser.parse_args()
    
    # 读取CSV文件
    data = []
    
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数据类型
            row['tp_world_size'] = int(row['tp_world_size'])
            row['batch_size'] = int(row['batch_size'])
            row['input_len'] = int(row['input_len'])
            row['output_len'] = int(row['output_len'])
            row['avg_prefill_time_usage'] = float(row['avg_prefill_time_usage'])
            row['avg_decoding_time_usage'] = float(row['avg_decoding_time_usage'])
            data.append(row)
    
    # 按模型和张量并行度分组
    grouped_data = {}
    for row in data:
        model = row['tag']
        tp = str(row['tp_world_size'])
        
        if model not in grouped_data:
            grouped_data[model] = {}
        if tp not in grouped_data[model]:
            grouped_data[model][tp] = []
        
        grouped_data[model][tp].append(row)
    
    # 拟合参数
    result = {}
    for model, tp_data in grouped_data.items():
        result[model] = {}
        for tp, rows in tp_data.items():
            print(f"Fit Model: {model}, TP: {tp}")
            
            # 拟合预填充参数
            prefill_params = fit_prefill_params(rows)
            print(f"  Prefill Parameters: {prefill_params}")
            
            # 拟合解码参数
            small_bs_params, large_bs_params = fit_decode_params(rows)
            print(f"  Small BS Parameters: {small_bs_params}")
            print(f"  Large BS Parameters: {large_bs_params}")
            
            # 保存结果
            result[model][tp] = {
                "decoding_large_small_bs_threshold": 95,
                "prefill": prefill_params,
                "decoding_smallbs": small_bs_params,
                "decoding_largebs": large_bs_params
            }
            
            # 生成可视化图表
            if args.visualize:
                plot_prefill_fit(rows, prefill_params, model, tp)
                plot_decode_fit(rows, small_bs_params, large_bs_params, model, tp)
    
    # 写入JSON文件
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFit Results Saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()

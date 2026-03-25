#!/usr/bin/env python3
"""
This script verifies Fit Results, and calculates Average Prefill and Decode Errors.
"""

import csv
import json
from pathlib import Path

PARAMS_PATH = Path("/users/rh/DistServe/simdistserve/estimators/profile_data/profiler-a100-80g.distserve.fitted.json")
CSV_PATH = Path("/users/rh/DistServe/evaluation/0-test-single-forward-performance/result/read_db.csv")

def load_fitted_params():
    """加载拟合的参数"""
    with open(PARAMS_PATH, 'r') as f:
        return json.load(f)


def calculate_prefill_time(params, batch_size, input_len):
    """使用拟合参数计算预填充时间"""
    a, b, c = params
    num_total_tokens = batch_size * input_len
    sum_num_tokens_sqr = batch_size * (input_len ** 2)
    return a + b * num_total_tokens + c * sum_num_tokens_sqr


def calculate_decode_time(params, batch_size, output_len):
    """使用拟合参数计算解码时间"""
    a, b, c = params
    num_total_tokens = batch_size * (output_len - 1)  # 第一个token在prefill阶段生成
    return a + b * num_total_tokens + c * batch_size


def main():
    # 加载拟合参数
    fitted_params = load_fitted_params()
    
    # 读取原始数据
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
    
    # 验证拟合结果并计算误差
    print("Verify Fit Results:")
    print("=" * 80)
    
    total_prefill_error = 0
    total_decode_error = 0
    count = 0
    
    for row in data:
        model = row['tag']
        tp = str(row['tp_world_size'])
        batch_size = row['batch_size']
        input_len = row['input_len']
        output_len = row['output_len']
        actual_prefill = row['avg_prefill_time_usage']
        actual_decode = row['avg_decoding_time_usage']
        
        # 获取拟合参数
        prefill_params = fitted_params[model][tp]['prefill']
        decode_params = fitted_params[model][tp]['decoding_smallbs']
        
        # 计算预测时间
        predicted_prefill = calculate_prefill_time(prefill_params, batch_size, input_len)
        predicted_decode = calculate_decode_time(decode_params, batch_size, output_len)
        
        # 计算误差
        prefill_error = abs(predicted_prefill - actual_prefill) / actual_prefill * 100
        decode_error = abs(predicted_decode - actual_decode) / actual_decode * 100
        
        total_prefill_error += prefill_error
        total_decode_error += decode_error
        count += 1
        
        # 每10个数据点打印一次
        if count % 10 == 0:
            print(f"Model: {model}, TP: {tp}, Batch: {batch_size}, Input: {input_len}")
            print(f"  Actual Prefill: {actual_prefill:.2f}ms, Predicted: {predicted_prefill:.2f}ms, Error: {prefill_error:.2f}%")
            print(f"  Actual Decode: {actual_decode:.2f}ms, Predicted: {predicted_decode:.2f}ms, Error: {decode_error:.2f}%")
            print("-" * 80)
    
    # 计算平均误差
    avg_prefill_error = total_prefill_error / count
    avg_decode_error = total_decode_error / count
    
    print(f"Average Prefill Error: {avg_prefill_error:.2f}%")
    print(f"Average Decode Error: {avg_decode_error:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()

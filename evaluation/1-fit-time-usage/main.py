"""
# Introduction

This program reads data (batch size, input len, output len, prefill time usage,
decoding time usage) from a sqlite database, and fits a model to predict the prefill/decoding
time usage.

# Methodology

For the prefill stage, we assume the time usage to be:

A + B*#total_tokens + C*\sum_{i=1}^{batch_size} input_len_i^2

Where #total_tokens = batch_size * input_len

While for the decoding stage, we assume the time usage to be:

A + B*input_len + C*output_len + D*input_len*output_len + E*output_len^2 + F*input_len^2

We use the least squares method ("最小二乘法" in Chinese) to fit the model, with
the goal of minimizing \sum relative_error^2.

usage:
python ./main.py \
    -i "../0-test-single-forward-performance/db-identical-req.sqlite" \
    -o "./params/fit_params.json" > "./main.log"


"""
import math
import dataclasses
from typing import Callable, Union
import json

import numpy as np
import csv
import sqlite3
import argparse

@dataclasses.dataclass
class DataPoint:
    model: str
    tp_world_size: int
    
    batch_size: int
    input_len: int
    output_len: int           # <-- 新增字段
    
    prefill_time: float
    decoding_time: float

def read_all_data_points(db_path: str) -> list[DataPoint]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # 新增 output_len 列读取
    cur.execute("SELECT tag, tp_world_size, batch_size, input_len, output_len, avg_prefill_time_usage, avg_decoding_time_usage FROM records")
    return [
        DataPoint(
            model = row[0],
            tp_world_size = row[1],
            batch_size = row[2],
            input_len = row[3],
            output_len = row[4],          # <-- 新增
            prefill_time = row[5],
            decoding_time = row[6]
        )
        for row in cur.fetchall()
    ]

def fit_linear_model(
    data_points: list[DataPoint],
    design_matrix_func: Callable[[DataPoint], list[float]],
    time_getter: Callable[[DataPoint], float],
    weight_getter: Callable[[DataPoint], float]
) -> np.ndarray:
    """
    通用最小二乘拟合。
    design_matrix_func 返回一个列表，对应每个数据点的设计矩阵行。
    time_getter 返回观测时间。
    weight_getter 返回该点的权重。
    返回拟合系数数组。
    """
    a_matrix = []
    b_vec = []
    for dp in data_points:
        row = design_matrix_func(dp)
        t = time_getter(dp)
        weight = weight_getter(dp)
        a_matrix.append([val / t * weight for val in row])
        b_vec.append(weight)
    a_matrix = np.array(a_matrix)
    b_vec = np.array(b_vec)
    coeffs, _, _, _ = np.linalg.lstsq(a_matrix, b_vec, rcond=None)
    
    # 打印诊断信息
    print(f"{'bs':>3s}  {'ilen':>6s}  {'olen':>6s}  {'actual':>9s}  {'pred':>9s}  {'rel_err':>6s}")
    rel_errs = []
    for dp in data_points:
        row = design_matrix_func(dp)
        t = time_getter(dp)
        pred_time_usage = np.dot(coeffs, row)
        cur_rel_err = (pred_time_usage - t) / t
        rel_errs.append(cur_rel_err)
        print(f"{dp.batch_size:3d}  {dp.input_len:6d}  {dp.output_len:6d}  {t:9.2f}  {pred_time_usage:9.2f}  {cur_rel_err*100:6.2f}%")
    
    rel_errs = np.array(rel_errs)
    print(f"Max relative error: {np.max(np.abs(rel_errs))*100:.2f}%")
    print(f"Avg relative error: {np.mean(np.abs(rel_errs))*100:.2f}%")
    print(f"Avg sqrt(relerr^2): {np.sqrt(np.mean(rel_errs**2))*100:.2f}%")
    
    return coeffs

def main(args: argparse.Namespace):
    print(args)
    input_path = args.input
    output_path = args.output
    
    data_points = read_all_data_points(input_path)
    
    data_points.sort(key=lambda dp: (dp.model, dp.tp_world_size, dp.batch_size, dp.input_len))
    
    models_and_tp_sizes = []
    for dp in data_points:
        if (dp.model, dp.tp_world_size) not in models_and_tp_sizes:
            models_and_tp_sizes.append((dp.model, dp.tp_world_size))
    
    DECODING_LARGE_SMALL_BS_THRESHOLD = 95   # 与需求代码保持一致
    
    result = {}
    for (model, tp_world_size) in models_and_tp_sizes:
        print(f"Fitting model {model} with tp_world_size {tp_world_size} (Prefill stage)")
        cur_data_points = [
            dp
            for dp in data_points
            if dp.model == model and dp.tp_world_size == tp_world_size
        ]

        prefill_coeffs = None
        if len(cur_data_points) != 0:
            prefill_coeffs = fit_linear_model(
                cur_data_points,
                lambda dp: [1, dp.batch_size * dp.input_len, dp.batch_size * dp.input_len**2],
                lambda dp: dp.prefill_time,
                lambda dp: 1.0
            )
            print(prefill_coeffs)
        else:
            print("No data points for prefill stage.")
        
        # Decoding small batch size (6 parameters)
        print(f"Fitting model {model} with tp_world_size {tp_world_size} (Decoding stage, small batch size)")
        cur_data_points_small = [
            dp
            for dp in data_points
            if dp.model == model and dp.tp_world_size == tp_world_size
            if dp.batch_size <= DECODING_LARGE_SMALL_BS_THRESHOLD
        ]

        decoding_smallbs_coeffs = None
        if len(cur_data_points_small) != 0:
            decoding_smallbs_coeffs = fit_linear_model(
                cur_data_points_small,
                lambda dp: [1, dp.input_len, dp.output_len, dp.input_len * dp.output_len, dp.output_len**2, dp.input_len**2],
                lambda dp: dp.decoding_time,
                lambda dp: 1.0
            )
            print(decoding_smallbs_coeffs)
        else:
            print("No data points for decoding stage, small batch size.")
        
        # Decoding large batch size (6 parameters)
        print(f"Fitting model {model} with tp_world_size {tp_world_size} (Decoding stage, large batch size)")
        cur_data_points_large = [
            dp
            for dp in data_points
            if dp.model == model and dp.tp_world_size == tp_world_size
            if dp.batch_size > DECODING_LARGE_SMALL_BS_THRESHOLD
        ]   
        
        decoding_largebs_coeffs = None
        if len(cur_data_points_large) != 0:
            decoding_largebs_coeffs = fit_linear_model(
                cur_data_points_large,
                lambda dp: [1, dp.input_len, dp.output_len, dp.input_len * dp.output_len, dp.output_len**2, dp.input_len**2],
                lambda dp: dp.decoding_time,
                lambda dp: 1.0
            )
            print(decoding_largebs_coeffs)
        else:
            print("No data points for decoding stage, large batch size.")
        
        if model not in result:
            result[model] = {}
        result[model][tp_world_size] = {
            "decoding_large_small_bs_threshold": DECODING_LARGE_SMALL_BS_THRESHOLD,
            "prefill": prefill_coeffs.tolist() if prefill_coeffs is not None else [],
            "decoding_smallbs": decoding_smallbs_coeffs.tolist() if decoding_smallbs_coeffs is not None else [],
            "decoding_largebs": decoding_largebs_coeffs.tolist() if decoding_largebs_coeffs is not None else []
        }
    
    with open(output_path, "w") as f:
        f.write(json.dumps(result, indent=4))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input sqlite database")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output json file")
    args = parser.parse_args()
    main(args)
#!/usr/bin/env python3
"""
合并 SLO 分析脚本：同时处理 DistServe 的 .exp (JSON) 文件和 simdistserve 的 .csv 文件，
生成各自的原始分析报告，并输出对比报告。

usage: For DistServe:

model=llama_7B
rate=1
python3 merged_analyze.py \
    --exp-file /users/rh/DistServe/evaluation/2-benchmark-serving/result/$model/distserve-100-$rate.exp \
    --csv-file /users/rh/DistServe/simdistserve/benchmarks/results/$model/rate_$rate/request_latency.csv \
    --prefill-slo 1.0 \
    --decode-slo 1.0 \
    --total-slo 1.0 \
    --exp-output /users/rh/Distserve_result/raw/$model/rate_$rate/actual/exp_analysis.txt \
    --csv-output /users/rh/Distserve_result/raw/$model/rate_$rate/sim/csv_analysis.txt \
    --compare-output /users/rh/Distserve_result/compared/$model/rate_$rate/comparison.txt


For VLLM-Ascend:

model=llama_1B
rate=4
python3 merged_analyze.py \
    --exp-file /users/rh/vllm_ascend_raw_data/$model/rate_$rate.json \
    --csv-file /users/rh/DistServe/simdistserve/benchmarks/vllm_ascend_results/$model/rate_$rate/request_latency.csv \
    --prefill-slo 1.0 \
    --decode-slo 1.0 \
    --total-slo 1.0 \
    --exp-output /users/rh/Distserve_result/vllm_ascend/raw/$model/rate_$rate/actual/exp_analysis.txt \
    --csv-output /users/rh/Distserve_result/vllm_ascend/raw/$model/rate_$rate/sim/csv_analysis.txt \
    --compare-output /users/rh/Distserve_result/vllm_ascend/compared/$model/rate_$rate/comparison.txt \
    --is_vllm_ascend
"""

import json
import csv
import argparse
import os

def analyze_exp(file_path, prefill_slo, decode_slo, total_slo):
    """分析 .exp 文件，返回 (stats_dict, output_string)"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_requests = len(data)
    prefill_slo_met = 0
    decode_slo_met = 0
    total_slo_met = 0
    both_slo_met = 0

    lines = []  # 收集输出行

    for i, request in enumerate(data):
        lifecycle_events = request['lifecycle_events']
        context_begin = None
        context_end = None
        decoding_begin = None
        decoding_end = None

        for event in lifecycle_events:
            if event['event_type'] == 'context_begin':
                context_begin = event['timestamp']
            elif event['event_type'] == 'context_end':
                context_end = event['timestamp']
            elif event['event_type'] == 'decoding_begin':
                decoding_begin = event['timestamp']
            elif event['event_type'] == 'decoding_end':
                decoding_end = event['timestamp']

        # Compare against the same user-visible first-token latency that the simulator reports.
        prefill_time = request.get('ftl', 0)
        decode_time = decoding_end - decoding_begin if decoding_begin and decoding_end else 0
        total_time = request['latency']

        prefill_ok = prefill_time <= prefill_slo
        decode_ok = decode_time <= decode_slo
        total_ok = total_time <= total_slo
        both_ok = prefill_ok and decode_ok

        if prefill_ok:
            prefill_slo_met += 1
        if decode_ok:
            decode_slo_met += 1
        if total_ok:
            total_slo_met += 1
        if both_ok:
            both_slo_met += 1

        lines.append(f"Request {i+1}:")
        lines.append(f"  Prompt length: {request['prompt_len']}")
        lines.append(f"  Output length: {request['output_len']}")
        lines.append(f"  First token latency: {prefill_time:.4f}s (SLO: {prefill_slo}s) {'✓' if prefill_ok else '✗'}")
        lines.append(f"  Decode time: {decode_time:.4f}s (SLO: {decode_slo}s) {'✓' if decode_ok else '✗'}")
        lines.append(f"  Total time: {total_time:.4f}s (SLO: {total_slo}s) {'✓' if total_ok else '✗'}")
        lines.append(f"  Both prefill and decode SLO met: {'✓' if both_ok else '✗'}")
        lines.append("")

    prefill_slo_rate = prefill_slo_met / total_requests * 100
    decode_slo_rate = decode_slo_met / total_requests * 100
    total_slo_rate = total_slo_met / total_requests * 100
    both_slo_rate = both_slo_met / total_requests * 100

    lines.append("=" * 60)
    lines.append("SLO Analysis Results")
    lines.append("=" * 60)
    lines.append(f"Total requests: {total_requests}")
    lines.append(f"Prefill SLO met: {prefill_slo_met}/{total_requests} ({prefill_slo_rate:.2f}%)")
    lines.append(f"Decode SLO met: {decode_slo_met}/{total_requests} ({decode_slo_rate:.2f}%)")
    lines.append(f"Total SLO met: {total_slo_met}/{total_requests} ({total_slo_rate:.2f}%)")
    lines.append(f"Both prefill and decode SLO met: {both_slo_met}/{total_requests} ({both_slo_rate:.2f}%)")
    lines.append("=" * 60)

    stats = {
        'total_requests': total_requests,
        'prefill_slo_met': prefill_slo_met,
        'decode_slo_met': decode_slo_met,
        'total_slo_met': total_slo_met,
        'both_slo_met': both_slo_met,
        'prefill_slo_rate': prefill_slo_rate,
        'decode_slo_rate': decode_slo_rate,
        'total_slo_rate': total_slo_rate,
        'both_slo_rate': both_slo_rate,
    }
    return stats, "\n".join(lines)

# This function is for vllm-ascend benchmark
def analyze_json(file_path, prefill_slo, decode_slo, total_slo):
    """分析 vLLM benchmark 输出的 JSON 文件（每个请求一次运行），返回 (stats_dict, output_string)"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 提取数组
    input_lens = data.get('input_lens', [])
    output_lens = data.get('output_lens', [])
    ttfts = data.get('ttfts', [])
    itls_list = data.get('itls', [])

    total_requests = len(input_lens)
    # 确保所有数组长度一致
    assert len(output_lens) == total_requests, "output_lens length mismatch"
    assert len(ttfts) == total_requests, "ttfts length mismatch"
    assert len(itls_list) == total_requests, "itls length mismatch"

    prefill_slo_met = 0
    decode_slo_met = 0
    total_slo_met = 0
    both_slo_met = 0

    lines = []

    for i in range(total_requests):
        prompt_len = input_lens[i]
        output_len = output_lens[i]
        prefill_time = ttfts[i]
        decode_time = sum(itls_list[i]) if itls_list[i] else 0.0
        total_time = prefill_time + decode_time

        prefill_ok = prefill_time <= prefill_slo
        decode_ok = decode_time <= decode_slo
        total_ok = total_time <= total_slo
        both_ok = prefill_ok and decode_ok

        if prefill_ok:
            prefill_slo_met += 1
        if decode_ok:
            decode_slo_met += 1
        if total_ok:
            total_slo_met += 1
        if both_ok:
            both_slo_met += 1

        lines.append(f"Request {i+1}:")
        lines.append(f"  Prompt length: {prompt_len}")
        lines.append(f"  Output length: {output_len}")
        lines.append(f"  Prefill time: {prefill_time:.4f}s (SLO: {prefill_slo}s) {'✓' if prefill_ok else '✗'}")
        lines.append(f"  Decode time: {decode_time:.4f}s (SLO: {decode_slo}s) {'✓' if decode_ok else '✗'}")
        lines.append(f"  Total time: {total_time:.4f}s (SLO: {total_slo}s) {'✓' if total_ok else '✗'}")
        lines.append(f"  Both prefill and decode SLO met: {'✓' if both_ok else '✗'}")
        lines.append("")

    prefill_slo_rate = prefill_slo_met / total_requests * 100 if total_requests else 0
    decode_slo_rate = decode_slo_met / total_requests * 100 if total_requests else 0
    total_slo_rate = total_slo_met / total_requests * 100 if total_requests else 0
    both_slo_rate = both_slo_met / total_requests * 100 if total_requests else 0

    lines.append("=" * 60)
    lines.append("SLO Analysis Results")
    lines.append("=" * 60)
    lines.append(f"Total requests: {total_requests}")
    lines.append(f"Prefill SLO met: {prefill_slo_met}/{total_requests} ({prefill_slo_rate:.2f}%)")
    lines.append(f"Decode SLO met: {decode_slo_met}/{total_requests} ({decode_slo_rate:.2f}%)")
    lines.append(f"Total SLO met: {total_slo_met}/{total_requests} ({total_slo_rate:.2f}%)")
    lines.append(f"Both prefill and decode SLO met: {both_slo_met}/{total_requests} ({both_slo_rate:.2f}%)")
    lines.append("=" * 60)

    stats = {
        'total_requests': total_requests,
        'prefill_slo_met': prefill_slo_met,
        'decode_slo_met': decode_slo_met,
        'total_slo_met': total_slo_met,
        'both_slo_met': both_slo_met,
        'prefill_slo_rate': prefill_slo_rate,
        'decode_slo_rate': decode_slo_rate,
        'total_slo_rate': total_slo_rate,
        'both_slo_rate': both_slo_rate,
    }
    return stats, "\n".join(lines)

def analyze_csv(file_path, prefill_slo, decode_slo, total_slo):
    """分析 .csv 文件，返回 (stats_dict, output_string)"""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['first_token_latency'] = float(row['first_token_latency'])
            row['decoding_latency'] = float(row['decoding_latency'])
            row['total_latency'] = float(row['total_latency'])
            data.append(row)

    total_requests = len(data)
    prefill_slo_met = 0
    decode_slo_met = 0
    total_slo_met = 0
    both_slo_met = 0

    lines = []

    for i, request in enumerate(data):
        prefill_time = request['first_token_latency'] / 1000   # 秒
        decode_time = request['decoding_latency'] / 1000
        total_time = request['total_latency'] / 1000

        prefill_ok = prefill_time <= prefill_slo
        decode_ok = decode_time <= decode_slo
        total_ok = total_time <= total_slo
        both_ok = prefill_ok and decode_ok

        if prefill_ok:
            prefill_slo_met += 1
        if decode_ok:
            decode_slo_met += 1
        if total_ok:
            total_slo_met += 1
        if both_ok:
            both_slo_met += 1

        lines.append(f"Request {i+1}:")
        lines.append(f"  First token latency: {prefill_time:.4f}s (SLO: {prefill_slo}s) {'✓' if prefill_ok else '✗'}")
        lines.append(f"  Decoding latency: {decode_time:.4f}s (SLO: {decode_slo}s) {'✓' if decode_ok else '✗'}")
        lines.append(f"  Total latency: {total_time:.4f}s (SLO: {total_slo}s) {'✓' if total_ok else '✗'}")
        lines.append(f"  Both prefill and decode SLO met: {'✓' if both_ok else '✗'}")
        lines.append("")

    prefill_slo_rate = prefill_slo_met / total_requests * 100
    decode_slo_rate = decode_slo_met / total_requests * 100
    total_slo_rate = total_slo_met / total_requests * 100
    both_slo_rate = both_slo_met / total_requests * 100

    lines.append("=" * 60)
    lines.append("SLO Analysis Results")
    lines.append("=" * 60)
    lines.append(f"Total requests: {total_requests}")
    lines.append(f"Prefill SLO met: {prefill_slo_met}/{total_requests} ({prefill_slo_rate:.2f}%)")
    lines.append(f"Decode SLO met: {decode_slo_met}/{total_requests} ({decode_slo_rate:.2f}%)")
    lines.append(f"Total SLO met: {total_slo_met}/{total_requests} ({total_slo_rate:.2f}%)")
    lines.append(f"Both prefill and decode SLO met: {both_slo_met}/{total_requests} ({both_slo_rate:.2f}%)")
    lines.append("=" * 60)

    stats = {
        'total_requests': total_requests,
        'prefill_slo_met': prefill_slo_met,
        'decode_slo_met': decode_slo_met,
        'total_slo_met': total_slo_met,
        'both_slo_met': both_slo_met,
        'prefill_slo_rate': prefill_slo_rate,
        'decode_slo_rate': decode_slo_rate,
        'total_slo_rate': total_slo_rate,
        'both_slo_rate': both_slo_rate,
    }
    return stats, "\n".join(lines)


def generate_comparison(exp_stats, csv_stats, slo_values):
    """生成对比报告字符串"""
    lines = []
    lines.append("=" * 70)
    lines.append("COMPARISON REPORT: SLO Satisfaction Rates")
    lines.append("=" * 70)
    lines.append(f"Unified SLO thresholds: Prefill={slo_values['prefill']}s, Decode={slo_values['decode']}s, Total={slo_values['total']}s")
    lines.append("")

    # 表头
    lines.append(f"{'Metric':<30} {'EXP (DistServe)':<25} {'CSV (simdistserve)':<25} {'Difference (Exp - CSV)':<20}")
    lines.append("-" * 100)

    # Prefill
    exp_prefill = f"{exp_stats['prefill_slo_met']}/{exp_stats['total_requests']} ({exp_stats['prefill_slo_rate']:.2f}%)"
    csv_prefill = f"{csv_stats['prefill_slo_met']}/{csv_stats['total_requests']} ({csv_stats['prefill_slo_rate']:.2f}%)"
    diff_prefill = f"{exp_stats['prefill_slo_rate'] - csv_stats['prefill_slo_rate']:+.2f}%"
    lines.append(f"{'Prefill SLO met rate':<30} {exp_prefill:<25} {csv_prefill:<25} {diff_prefill:<20}")

    # Decode
    exp_decode = f"{exp_stats['decode_slo_met']}/{exp_stats['total_requests']} ({exp_stats['decode_slo_rate']:.2f}%)"
    csv_decode = f"{csv_stats['decode_slo_met']}/{csv_stats['total_requests']} ({csv_stats['decode_slo_rate']:.2f}%)"
    diff_decode = f"{exp_stats['decode_slo_rate'] - csv_stats['decode_slo_rate']:+.2f}%"
    lines.append(f"{'Decode SLO met rate':<30} {exp_decode:<25} {csv_decode:<25} {diff_decode:<20}")

    # Total
    exp_total = f"{exp_stats['total_slo_met']}/{exp_stats['total_requests']} ({exp_stats['total_slo_rate']:.2f}%)"
    csv_total = f"{csv_stats['total_slo_met']}/{csv_stats['total_requests']} ({csv_stats['total_slo_rate']:.2f}%)"
    diff_total = f"{exp_stats['total_slo_rate'] - csv_stats['total_slo_rate']:+.2f}%"
    lines.append(f"{'Total SLO met rate':<30} {exp_total:<25} {csv_total:<25} {diff_total:<20}")

    # Both (prefill+decode)
    exp_both = f"{exp_stats['both_slo_met']}/{exp_stats['total_requests']} ({exp_stats['both_slo_rate']:.2f}%)"
    csv_both = f"{csv_stats['both_slo_met']}/{csv_stats['total_requests']} ({csv_stats['both_slo_rate']:.2f}%)"
    diff_both = f"{exp_stats['both_slo_rate'] - csv_stats['both_slo_rate']:+.2f}%"
    lines.append(f"{'Both (Prefill+Decode) SLO met rate':<30} {exp_both:<25} {csv_both:<25} {diff_both:<20}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare SLO compliance between DistServe .exp and simdistserve .csv results")
    parser.add_argument("--exp-file", required=True, help="Path to the .exp file (JSON)")
    parser.add_argument("--csv-file", required=True, help="Path to the .csv file")
    parser.add_argument("--prefill-slo", type=float, required=True, help="Prefill SLO in seconds (unified)")
    parser.add_argument("--decode-slo", type=float, required=True, help="Decode SLO in seconds (unified)")
    parser.add_argument("--total-slo", type=float, required=True, help="Total SLO in seconds (unified)")
    parser.add_argument("--exp-output", required=True, help="Output file for EXP raw analysis")
    parser.add_argument("--csv-output", required=True, help="Output file for CSV raw analysis")
    parser.add_argument("--compare-output", required=True, help="Output file for comparison report")
    parser.add_argument("--is_vllm_ascend", action="store_true", default=False, help="Is the model VLLM-Ascend?")



    args = parser.parse_args()

    if args.is_vllm_ascend:
        exp_stats, exp_output = analyze_json(args.exp_file, args.prefill_slo, args.decode_slo, args.total_slo)
        os.makedirs(os.path.dirname(args.exp_output), exist_ok=True)
        with open(args.exp_output, 'w') as f:
            f.write(exp_output)
    else:
        # 分析 EXP
        exp_stats, exp_output = analyze_exp(args.exp_file, args.prefill_slo, args.decode_slo, args.total_slo)
        os.makedirs(os.path.dirname(args.exp_output), exist_ok=True)
        with open(args.exp_output, 'w') as f:
            f.write(exp_output)

    # 分析 CSV
    csv_stats, csv_output = analyze_csv(args.csv_file, args.prefill_slo, args.decode_slo, args.total_slo)
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    with open(args.csv_output, 'w') as f:
        f.write(csv_output)

    # 生成对比报告
    slo_values = {'prefill': args.prefill_slo, 'decode': args.decode_slo, 'total': args.total_slo}
    compare_report = generate_comparison(exp_stats, csv_stats, slo_values)
    os.makedirs(os.path.dirname(args.compare_output), exist_ok=True)
    with open(args.compare_output, 'w') as f:
        f.write(compare_report)

    print(f"Done. Raw EXP output written to: {args.exp_output}")
    print(f"Raw CSV output written to: {args.csv_output}")
    print(f"Comparison report written to: {args.compare_output}")


if __name__ == "__main__":
    main()

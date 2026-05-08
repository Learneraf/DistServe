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
    --ttft-slo 0.2 \
    --tpot-slo 0.03 \
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
    --ttft-slo 0.2 \
    --tpot-slo 0.03 \
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


def calculate_tpot_from_request_trace(request):
    """Return request-level TPOT in seconds from benchmark trace fields."""
    if request.get('tpot') is not None:
        return float(request['tpot'])

    token_timestamps = request.get('token_timestamps') or []
    if len(token_timestamps) > 1:
        return (token_timestamps[-1] - token_timestamps[0]) / (len(token_timestamps) - 1)

    output_len = int(request.get('output_len', 0) or 0)
    if output_len <= 1:
        return 0.0

    lifecycle_events = request.get('lifecycle_events') or []
    decoding_begin = None
    decoding_end = None
    for event in lifecycle_events:
        if event['event_type'] == 'decoding_begin':
            decoding_begin = event['timestamp']
        elif event['event_type'] == 'decoding_end':
            decoding_end = event['timestamp']

    if decoding_begin is not None and decoding_end is not None:
        decode_time = decoding_end - decoding_begin
    else:
        ttft_time = float(request.get('ftl', 0))
        total_time = float(request['latency'])
        decode_time = max(total_time - ttft_time, 0)

    return decode_time / max(output_len - 1, 1)


def analyze_request_trace_payload(data, ttft_slo, tpot_slo, total_slo):
    """Analyze request-trace payloads used by DistServe CUDA and Ascend vLLM `.exp` files."""
    total_requests = len(data)
    ttft_slo_met = 0
    tpot_slo_met = 0
    total_slo_met = 0
    both_slo_met = 0

    lines = []

    for i, request in enumerate(data):
        ttft_time = float(request.get('ftl', 0))
        total_time = float(request['latency'])
        tpot_time = calculate_tpot_from_request_trace(request)

        ttft_ok = ttft_time <= ttft_slo
        tpot_ok = tpot_time <= tpot_slo
        total_ok = total_time <= total_slo
        both_ok = ttft_ok and tpot_ok

        if ttft_ok:
            ttft_slo_met += 1
        if tpot_ok:
            tpot_slo_met += 1
        if total_ok:
            total_slo_met += 1
        if both_ok:
            both_slo_met += 1

        lines.append(f"Request {i+1}:")
        lines.append(f"  Prompt length: {request['prompt_len']}")
        lines.append(f"  Output length: {request['output_len']}")
        lines.append(f"  TTFT: {ttft_time:.4f}s (SLO: {ttft_slo}s) {'✓' if ttft_ok else '✗'}")
        lines.append(f"  TPOT: {tpot_time:.4f}s (SLO: {tpot_slo}s) {'✓' if tpot_ok else '✗'}")
        lines.append(f"  Total time: {total_time:.4f}s (SLO: {total_slo}s) {'✓' if total_ok else '✗'}")
        lines.append(f"  Both TTFT and TPOT SLO met: {'✓' if both_ok else '✗'}")
        lines.append("")

    ttft_slo_rate = ttft_slo_met / total_requests * 100 if total_requests else 0
    tpot_slo_rate = tpot_slo_met / total_requests * 100 if total_requests else 0
    total_slo_rate = total_slo_met / total_requests * 100 if total_requests else 0
    both_slo_rate = both_slo_met / total_requests * 100 if total_requests else 0

    lines.append("=" * 60)
    lines.append("SLO Analysis Results")
    lines.append("=" * 60)
    lines.append(f"Total requests: {total_requests}")
    lines.append(f"TTFT SLO met: {ttft_slo_met}/{total_requests} ({ttft_slo_rate:.2f}%)")
    lines.append(f"TPOT SLO met: {tpot_slo_met}/{total_requests} ({tpot_slo_rate:.2f}%)")
    lines.append(f"Total SLO met: {total_slo_met}/{total_requests} ({total_slo_rate:.2f}%)")
    lines.append(f"Both TTFT and TPOT SLO met: {both_slo_met}/{total_requests} ({both_slo_rate:.2f}%)")
    lines.append("=" * 60)

    stats = {
        'total_requests': total_requests,
        'ttft_slo_met': ttft_slo_met,
        'tpot_slo_met': tpot_slo_met,
        'total_slo_met': total_slo_met,
        'both_slo_met': both_slo_met,
        'ttft_slo_rate': ttft_slo_rate,
        'tpot_slo_rate': tpot_slo_rate,
        'total_slo_rate': total_slo_rate,
        'both_slo_rate': both_slo_rate,
    }
    return stats, "\n".join(lines)

def analyze_exp(file_path, ttft_slo, tpot_slo, total_slo):
    """分析 .exp 文件，返回 (stats_dict, output_string)"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    return analyze_request_trace_payload(data, ttft_slo, tpot_slo, total_slo)

# This function is for vllm-ascend benchmark
def analyze_json(file_path, ttft_slo, tpot_slo, total_slo):
    """分析 vLLM benchmark 输出的 JSON 文件（每个请求一次运行），返回 (stats_dict, output_string)"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return analyze_request_trace_payload(data, ttft_slo, tpot_slo, total_slo)

    # 提取数组
    input_lens = data.get('input_lens', [])
    output_lens = data.get('output_lens', [])
    ttfts = data.get('ttfts', [])
    itls_list = data.get('itls', [])
    tpots = data.get('tpots', data.get('tpot', []))

    total_requests = len(input_lens)
    # 确保所有数组长度一致
    assert len(output_lens) == total_requests, "output_lens length mismatch"
    assert len(ttfts) == total_requests, "ttfts length mismatch"
    assert len(itls_list) == total_requests, "itls length mismatch"

    ttft_slo_met = 0
    tpot_slo_met = 0
    total_slo_met = 0
    both_slo_met = 0

    lines = []

    for i in range(total_requests):
        prompt_len = input_lens[i]
        output_len = output_lens[i]
        ttft_time = ttfts[i]
        inter_token_latencies = itls_list[i]
        decoding_time = sum(inter_token_latencies) if inter_token_latencies else 0.0
        if tpots and len(tpots) == total_requests:
            tpot_time = tpots[i]
        else:
            tpot_time = decoding_time / len(inter_token_latencies) if inter_token_latencies else 0.0
        total_time = ttft_time + decoding_time

        ttft_ok = ttft_time <= ttft_slo
        tpot_ok = tpot_time <= tpot_slo
        total_ok = total_time <= total_slo
        both_ok = ttft_ok and tpot_ok

        if ttft_ok:
            ttft_slo_met += 1
        if tpot_ok:
            tpot_slo_met += 1
        if total_ok:
            total_slo_met += 1
        if both_ok:
            both_slo_met += 1

        lines.append(f"Request {i+1}:")
        lines.append(f"  Prompt length: {prompt_len}")
        lines.append(f"  Output length: {output_len}")
        lines.append(f"  TTFT: {ttft_time:.4f}s (SLO: {ttft_slo}s) {'✓' if ttft_ok else '✗'}")
        lines.append(f"  TPOT: {tpot_time:.4f}s (SLO: {tpot_slo}s) {'✓' if tpot_ok else '✗'}")
        lines.append(f"  Total time: {total_time:.4f}s (SLO: {total_slo}s) {'✓' if total_ok else '✗'}")
        lines.append(f"  Both TTFT and TPOT SLO met: {'✓' if both_ok else '✗'}")
        lines.append("")

    ttft_slo_rate = ttft_slo_met / total_requests * 100 if total_requests else 0
    tpot_slo_rate = tpot_slo_met / total_requests * 100 if total_requests else 0
    total_slo_rate = total_slo_met / total_requests * 100 if total_requests else 0
    both_slo_rate = both_slo_met / total_requests * 100 if total_requests else 0

    lines.append("=" * 60)
    lines.append("SLO Analysis Results")
    lines.append("=" * 60)
    lines.append(f"Total requests: {total_requests}")
    lines.append(f"TTFT SLO met: {ttft_slo_met}/{total_requests} ({ttft_slo_rate:.2f}%)")
    lines.append(f"TPOT SLO met: {tpot_slo_met}/{total_requests} ({tpot_slo_rate:.2f}%)")
    lines.append(f"Total SLO met: {total_slo_met}/{total_requests} ({total_slo_rate:.2f}%)")
    lines.append(f"Both TTFT and TPOT SLO met: {both_slo_met}/{total_requests} ({both_slo_rate:.2f}%)")
    lines.append("=" * 60)

    stats = {
        'total_requests': total_requests,
        'ttft_slo_met': ttft_slo_met,
        'tpot_slo_met': tpot_slo_met,
        'total_slo_met': total_slo_met,
        'both_slo_met': both_slo_met,
        'ttft_slo_rate': ttft_slo_rate,
        'tpot_slo_rate': tpot_slo_rate,
        'total_slo_rate': total_slo_rate,
        'both_slo_rate': both_slo_rate,
    }
    return stats, "\n".join(lines)

def analyze_csv(file_path, ttft_slo, tpot_slo, total_slo):
    """分析 .csv 文件，返回 (stats_dict, output_string)"""
    data = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['first_token_latency'] = float(row['first_token_latency'])
            row['total_latency'] = float(row['total_latency'])
            row['tpot'] = float(row['tpot'])
            data.append(row)

    total_requests = len(data)
    ttft_slo_met = 0
    tpot_slo_met = 0
    total_slo_met = 0
    both_slo_met = 0

    lines = []

    for i, request in enumerate(data):
        ttft_time = request['first_token_latency'] / 1000   # 秒
        tpot_time = request['tpot'] / 1000
        total_time = request['total_latency'] / 1000

        ttft_ok = ttft_time <= ttft_slo
        tpot_ok = tpot_time <= tpot_slo
        total_ok = total_time <= total_slo
        both_ok = ttft_ok and tpot_ok

        if ttft_ok:
            ttft_slo_met += 1
        if tpot_ok:
            tpot_slo_met += 1
        if total_ok:
            total_slo_met += 1
        if both_ok:
            both_slo_met += 1

        lines.append(f"Request {i+1}:")
        lines.append(f"  TTFT: {ttft_time:.4f}s (SLO: {ttft_slo}s) {'✓' if ttft_ok else '✗'}")
        lines.append(f"  TPOT: {tpot_time:.4f}s (SLO: {tpot_slo}s) {'✓' if tpot_ok else '✗'}")
        lines.append(f"  Total latency: {total_time:.4f}s (SLO: {total_slo}s) {'✓' if total_ok else '✗'}")
        lines.append(f"  Both TTFT and TPOT SLO met: {'✓' if both_ok else '✗'}")
        lines.append("")

    ttft_slo_rate = ttft_slo_met / total_requests * 100 if total_requests else 0
    tpot_slo_rate = tpot_slo_met / total_requests * 100 if total_requests else 0
    total_slo_rate = total_slo_met / total_requests * 100 if total_requests else 0
    both_slo_rate = both_slo_met / total_requests * 100 if total_requests else 0

    lines.append("=" * 60)
    lines.append("SLO Analysis Results")
    lines.append("=" * 60)
    lines.append(f"Total requests: {total_requests}")
    lines.append(f"TTFT SLO met: {ttft_slo_met}/{total_requests} ({ttft_slo_rate:.2f}%)")
    lines.append(f"TPOT SLO met: {tpot_slo_met}/{total_requests} ({tpot_slo_rate:.2f}%)")
    lines.append(f"Total SLO met: {total_slo_met}/{total_requests} ({total_slo_rate:.2f}%)")
    lines.append(f"Both TTFT and TPOT SLO met: {both_slo_met}/{total_requests} ({both_slo_rate:.2f}%)")
    lines.append("=" * 60)

    stats = {
        'total_requests': total_requests,
        'ttft_slo_met': ttft_slo_met,
        'tpot_slo_met': tpot_slo_met,
        'total_slo_met': total_slo_met,
        'both_slo_met': both_slo_met,
        'ttft_slo_rate': ttft_slo_rate,
        'tpot_slo_rate': tpot_slo_rate,
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
    lines.append(f"Unified SLO thresholds: TTFT={slo_values['ttft']}s, TPOT={slo_values['tpot']}s, Total={slo_values['total']}s")
    lines.append("")

    # 表头
    lines.append(f"{'Metric':<30} {'EXP (DistServe)':<25} {'CSV (simdistserve)':<25} {'Difference (Exp - CSV)':<20}")
    lines.append("-" * 100)

    # TTFT
    exp_ttft = f"{exp_stats['ttft_slo_met']}/{exp_stats['total_requests']} ({exp_stats['ttft_slo_rate']:.2f}%)"
    csv_ttft = f"{csv_stats['ttft_slo_met']}/{csv_stats['total_requests']} ({csv_stats['ttft_slo_rate']:.2f}%)"
    diff_ttft = f"{exp_stats['ttft_slo_rate'] - csv_stats['ttft_slo_rate']:+.2f}%"
    lines.append(f"{'TTFT SLO met rate':<30} {exp_ttft:<25} {csv_ttft:<25} {diff_ttft:<20}")

    # TPOT
    exp_tpot = f"{exp_stats['tpot_slo_met']}/{exp_stats['total_requests']} ({exp_stats['tpot_slo_rate']:.2f}%)"
    csv_tpot = f"{csv_stats['tpot_slo_met']}/{csv_stats['total_requests']} ({csv_stats['tpot_slo_rate']:.2f}%)"
    diff_tpot = f"{exp_stats['tpot_slo_rate'] - csv_stats['tpot_slo_rate']:+.2f}%"
    lines.append(f"{'TPOT SLO met rate':<30} {exp_tpot:<25} {csv_tpot:<25} {diff_tpot:<20}")

    # Total
    exp_total = f"{exp_stats['total_slo_met']}/{exp_stats['total_requests']} ({exp_stats['total_slo_rate']:.2f}%)"
    csv_total = f"{csv_stats['total_slo_met']}/{csv_stats['total_requests']} ({csv_stats['total_slo_rate']:.2f}%)"
    diff_total = f"{exp_stats['total_slo_rate'] - csv_stats['total_slo_rate']:+.2f}%"
    lines.append(f"{'Total SLO met rate':<30} {exp_total:<25} {csv_total:<25} {diff_total:<20}")

    # Both (TTFT+TPOT)
    exp_both = f"{exp_stats['both_slo_met']}/{exp_stats['total_requests']} ({exp_stats['both_slo_rate']:.2f}%)"
    csv_both = f"{csv_stats['both_slo_met']}/{csv_stats['total_requests']} ({csv_stats['both_slo_rate']:.2f}%)"
    diff_both = f"{exp_stats['both_slo_rate'] - csv_stats['both_slo_rate']:+.2f}%"
    lines.append(f"{'Both (TTFT+TPOT) SLO met rate':<30} {exp_both:<25} {csv_both:<25} {diff_both:<20}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare SLO compliance between DistServe .exp and simdistserve .csv results")
    parser.add_argument("--exp-file", required=True, help="Path to the .exp file (JSON)")
    parser.add_argument("--csv-file", required=True, help="Path to the .csv file")
    parser.add_argument("--ttft-slo", type=float, required=True, help="TTFT SLO in seconds (unified)")
    parser.add_argument("--tpot-slo", type=float, required=True, help="TPOT SLO in seconds (unified)")
    parser.add_argument("--total-slo", type=float, required=True, help="Total SLO in seconds (unified)")
    parser.add_argument("--exp-output", required=True, help="Output file for EXP raw analysis")
    parser.add_argument("--csv-output", required=True, help="Output file for CSV raw analysis")
    parser.add_argument("--compare-output", required=True, help="Output file for comparison report")

    args = parser.parse_args()

    exp_stats, exp_output = analyze_exp(args.exp_file, args.ttft_slo, args.tpot_slo, args.total_slo)
    os.makedirs(os.path.dirname(args.exp_output), exist_ok=True)
    with open(args.exp_output, 'w') as f:
        f.write(exp_output)

    # 分析 CSV
    csv_stats, csv_output = analyze_csv(args.csv_file, args.ttft_slo, args.tpot_slo, args.total_slo)
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    with open(args.csv_output, 'w') as f:
        f.write(csv_output)

    # 生成对比报告
    slo_values = {'ttft': args.ttft_slo, 'tpot': args.tpot_slo, 'total': args.total_slo}
    compare_report = generate_comparison(exp_stats, csv_stats, slo_values)
    os.makedirs(os.path.dirname(args.compare_output), exist_ok=True)
    with open(args.compare_output, 'w') as f:
        f.write(compare_report)

    print(f"Done. Raw EXP output written to: {args.exp_output}")
    print(f"Raw CSV output written to: {args.csv_output}")
    print(f"Comparison report written to: {args.compare_output}")


if __name__ == "__main__":
    main()

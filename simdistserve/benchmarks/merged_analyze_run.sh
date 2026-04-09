#!/bin/bash

MODELS=(
    # "llama_3B"
    "llama_1B"
    # "llama_8B"
    # "llama_7B"
)
RATES=(1 1.5 2 2.5 3 3.5 4)
TYPE="vllm_ascend"
# TYPE="distserve_cuda"

for model in "${MODELS[@]}"; do
    for rate in "${RATES[@]}"; do
        echo "=================================================="
        echo "Running: model = $model | rate = $rate"
        echo "=================================================="

        # 自动创建输出目录（防止目录不存在报错）
        mkdir -p /users/rh/DistServe/simdistserve/benchmarks/results/slo/$TYPE/raw/$model/rate_$rate/actual
        mkdir -p /users/rh/DistServe/simdistserve/benchmarks/results/slo/$TYPE/raw/$model/rate_$rate/sim
        mkdir -p /users/rh/DistServe/simdistserve/benchmarks/results/slo/$TYPE/compared/$model/rate_$rate

        # 运行你的 Python 命令
        python3 merged_analyze.py \
            --exp-file /users/rh/DistServe/evaluation/2-benchmark-serving/result/$model/distserve-100-$rate.exp \
            --csv-file /users/rh/DistServe/simdistserve/benchmarks/results/latency/$TYPE/organized_data/$model/rate_$rate/request_latency.csv \
            --prefill-slo 1.0 \
            --decode-slo 1.0 \
            --total-slo 1.0 \
            --exp-output /users/rh/DistServe/simdistserve/benchmarks/results/slo/$TYPE/raw/$model/rate_$rate/actual/exp_analysis.txt \
            --csv-output /users/rh/DistServe/simdistserve/benchmarks/results/slo/$TYPE/raw/$model/rate_$rate/sim/csv_analysis.txt \
            --compare-output /users/rh/DistServe/simdistserve/benchmarks/results/slo/$TYPE/compared/$model/rate_$rate/comparison.txt

        # 输出完成提示
        echo "✅ Finished: model = $model | rate = $rate"
        echo ""
    done
done

echo "🎉 All tasks completed successfully!"
#!/bin/bash

# 定义要循环的 rate 列表
rates=(1 1.5 2 2.5 3 3.5 4)

# 仅需修改这个变量来切换模型
MODEL="llama_1B"   # 可选: llama_1B, llama_3B, llama_7B, llama_8B
# TYPE="vllm_ascend"
TYPE="distserve_cuda"

# 使用关联数组映射模型 -> workload 和 model_path
declare -A WORKLOAD_MAP
declare -A MODEL_PATH_MAP

# 基础路径（可根据实际情况调整）
WORKLOAD_BASE="/users/rh/DistServe/evaluation/2-benchmark-serving/data"

# WORKLOAD_MAP["llama_1B"]="$WORKLOAD_BASE/sharegpt_1B.json"
# WORKLOAD_MAP["llama_3B"]="$WORKLOAD_BASE/sharegpt_3B.json"
# WORKLOAD_MAP["llama_7B"]="$WORKLOAD_BASE/sharegpt_7B.json"
# WORKLOAD_MAP["llama_8B"]="$WORKLOAD_BASE/sharegpt_8B.json"

WORKLOAD_MAP["llama_1B"]="$WORKLOAD_BASE/sampled_sharegpt_pure.jsonl"
WORKLOAD_MAP["llama_3B"]="$WORKLOAD_BASE/sampled_sharegpt_pure.jsonl"
WORKLOAD_MAP["llama_7B"]="$WORKLOAD_BASE/sampled_sharegpt_pure.jsonl"
WORKLOAD_MAP["llama_8B"]="$WORKLOAD_BASE/sampled_sharegpt_pure.jsonl"

MODEL_PATH_MAP["llama_1B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2"
MODEL_PATH_MAP["llama_3B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2"
MODEL_PATH_MAP["llama_7B"]="huggyllama/llama-7b"
MODEL_PATH_MAP["llama_8B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2"

# 检查模型是否在映射中
if [[ -z "${WORKLOAD_MAP[$MODEL]}" ]] || [[ -z "${MODEL_PATH_MAP[$MODEL]}" ]]; then
    echo "Error: Unknown model '$MODEL'"
    exit 1
fi

WORKLOAD_FILE="${WORKLOAD_MAP[$MODEL]}"
MODEL_PATH="${MODEL_PATH_MAP[$MODEL]}"

BACKEND="distserve"
if [[ "$TYPE" == "vllm_ascend" ]]; then
    BACKEND="vllm_ascend"
fi

for RATE in "${rates[@]}"; do
    echo "Running with model=$MODEL, rate=$RATE"

    OUTPUT_DIR="./results/latency/$TYPE/organized_data/${MODEL}/rate_${RATE}"
    mkdir -p "$OUTPUT_DIR"

    python simulate_dist.py \
        --backend "$BACKEND" \
        --model "$MODEL_PATH" \
        --seed 0 \
        --rate "$RATE" \
        --N 100 \
        --arrival poisson \
        --workload "$WORKLOAD_FILE" \
        --output "$OUTPUT_DIR/sharegpt.json.sim.csv" \
        --name "${MODEL}/rate_${RATE}" \
        --output-request-latency "$OUTPUT_DIR/request_latency.csv" \
        --slo-scales "[1.0]" \
        --output-request-event "$OUTPUT_DIR/request_event.csv" \
        --output-request-info "$OUTPUT_DIR/request_info.csv"
    
    if [ $? -ne 0 ]; then
        echo "Error: rate $RATE failed for model $MODEL"
        exit 1
    fi
done

echo "All rates completed for model $MODEL."

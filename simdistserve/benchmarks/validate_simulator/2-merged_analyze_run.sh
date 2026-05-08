#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TYPE="${TYPE:-ascend_vllm}"
MODE="${MODE:-val}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LATENCY_ROOT="${LATENCY_ROOT:-/users/rh/DistServe/simdistserve/benchmarks/validate_simulator/result/latency}"
SLO_ROOT="${SLO_ROOT:-/users/rh/DistServe/simdistserve/benchmarks/validate_simulator/result/slo}"
MODELS_STR="${MODELS:-llama_1B llama_3B llama_7B llama_8B}"
RATES_STR="${RATES:-1.0 1.5 2.0 2.5 3.0 3.5 4.0}"
SLO_SCALES_STR="${SLO_SCALES:-0.8 1.0 1.2}"
TTFT_TARGET="${TTFT_TARGET:-0.2}"
TPOT_TARGET="${TPOT_TARGET:-0.03}"
TOTAL_TARGET="${TOTAL_TARGET:-1}"

read -r -a MODELS <<< "$MODELS_STR"
read -r -a RATES <<< "$RATES_STR"
read -r -a SLO_SCALE_VALUES <<< "$SLO_SCALES_STR"

scale_slo() {
    local base_slo="$1"
    local scale="$2"
    awk -v base="$base_slo" -v scale="$scale" 'BEGIN { printf "%.10g", base * scale }'
}

scale_label() {
    local scale="$1"
    echo "${scale//./p}"
}

case "$TYPE" in
    cuda_distserve)
        EXP_BASE="${EXP_BASE:-/users/rh/DistServe/exp_data/cuda_distserve/${MODE}}"
        ;;
    ascend_vllm)
        EXP_BASE="${EXP_BASE:-/users/rh/DistServe/exp_data/ascend_vllm/${MODE}}"
        ;;
    *)
        echo "Error: Unsupported TYPE '$TYPE'. Use cuda_distserve or ascend_vllm."
        exit 1
        ;;
esac

for model in "${MODELS[@]}"; do
    for rate in "${RATES[@]}"; do
        echo "=================================================="
        echo "Running merged_analyze: type=$TYPE mode=$MODE model=$model rate=$rate"
        echo "=================================================="

        mkdir -p "$SLO_ROOT/$TYPE/raw/$model/rate_$rate/actual"
        mkdir -p "$SLO_ROOT/$TYPE/raw/$model/rate_$rate/sim"
        mkdir -p "$SLO_ROOT/$TYPE/compared/$model/rate_$rate"

        if [[ "$TYPE" == "ascend_vllm" ]]; then
            EXP_FILE="$EXP_BASE/$model/ascend-vllm-120-$rate.exp"
        elif [[ "$TYPE" == "cuda_distserve" ]]; then
            EXP_FILE="$EXP_BASE/$model/distserve-120-$rate.exp"
        else
            echo "Error: invalid TYPE" >&2
            exit 1
        fi

        CSV_FILE="$LATENCY_ROOT/$TYPE/$model/rate_$rate/request_latency.csv"

        for scale in "${SLO_SCALE_VALUES[@]}"; do
            scale_tag="$(scale_label "$scale")"
            ttft_slo="$(scale_slo "$TTFT_TARGET" "$scale")"
            tpot_slo="$(scale_slo "$TPOT_TARGET" "$scale")"
            total_slo="$(scale_slo "$TOTAL_TARGET" "$scale")"

            exp_output="$SLO_ROOT/$TYPE/raw/$model/rate_$rate/scale_${scale_tag}/actual/exp_analysis.txt"
            csv_output="$SLO_ROOT/$TYPE/raw/$model/rate_$rate/scale_${scale_tag}/sim/csv_analysis.txt"
            compare_output="$SLO_ROOT/$TYPE/compared/$model/rate_$rate/scale_${scale_tag}/comparison.txt"

            "$PYTHON_BIN" merged_analyze.py \
                --exp-file "$EXP_FILE" \
                --csv-file "$CSV_FILE" \
                --ttft-slo "$ttft_slo" \
                --tpot-slo "$tpot_slo" \
                --total-slo "$total_slo" \
                --exp-output "$exp_output" \
                --csv-output "$csv_output" \
                --compare-output "$compare_output"

            if [[ "$scale" == "1" || "$scale" == "1.0" ]]; then
                cp "$exp_output" "$SLO_ROOT/$TYPE/raw/$model/rate_$rate/actual/exp_analysis.txt"
                cp "$csv_output" "$SLO_ROOT/$TYPE/raw/$model/rate_$rate/sim/csv_analysis.txt"
                cp "$compare_output" "$SLO_ROOT/$TYPE/compared/$model/rate_$rate/comparison.txt"
            fi
        done

        echo "Finished: type=$TYPE mode=$MODE model=$model rate=$rate"
    done
done

echo "All merged analysis tasks completed for type=$TYPE mode=$MODE."

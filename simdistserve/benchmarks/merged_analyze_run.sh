#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TYPE="${TYPE:-vllm_ascend}"
MODE="val"
PYTHON_BIN="python3"
LATENCY_ROOT="/users/rh/DistServe/simdistserve/benchmarks/results/latency"
SLO_ROOT="/users/rh/DistServe/simdistserve/benchmarks/results/slo"
MODELS_STR="llama_1B llama_3B llama_7B llama_8B"
RATES_STR="1 1.5 2 2.5 3 3.5 4"
SLO_SCALES_STR="${SLO_SCALES:-0.4 0.6 0.8 1.0 1.2}"
PREFILL_SLO_BASE="${PREFILL_SLO:-1.0}"
DECODE_SLO_BASE="${DECODE_SLO:-1.0}"
TOTAL_SLO_BASE="${TOTAL_SLO:-1.0}"

read -r -a MODELS <<< "$MODELS_STR"
read -r -a RATES <<< "$RATES_STR"
read -r -a SLO_SCALE_VALUES <<< "$SLO_SCALES_STR"

declare -A ASCEND_REAL_DIR_MAP
ASCEND_REAL_DIR_MAP["llama_1B"]="llama1B"
ASCEND_REAL_DIR_MAP["llama_3B"]="llama3B"
ASCEND_REAL_DIR_MAP["llama_7B"]="llama7B"
ASCEND_REAL_DIR_MAP["llama_8B"]="llama8B"

normalize_exp_rate() {
    local rate="$1"
    local backend="$2"
    if [[ "$backend" == "vllm_ascend" && "$rate" == "1" ]]; then
        echo "1.0"
        return
    fi
    echo "$rate"
}

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
    distserve_cuda)
        EXP_BASE="${EXP_BASE:-/users/rh/DistServe/evaluation/2-benchmark-serving/result/${MODE}}"
        IS_VLLM_ARGS=()
        ;;
    vllm_ascend)
        EXP_BASE="${EXP_BASE:-/users/rh/ascend_data/ascend_vllm_holdout_${MODE}}"
        IS_VLLM_ARGS=(--is_vllm_ascend)
        ;;
    *)
        echo "Error: Unsupported TYPE '$TYPE'. Use distserve_cuda or vllm_ascend."
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

        if [[ "$TYPE" == "vllm_ascend" ]]; then
            real_dir="${ASCEND_REAL_DIR_MAP[$model]:-}"
            if [[ -z "$real_dir" ]]; then
                echo "Error: Unknown Ascend model alias '$model'"
                exit 1
            fi
            exp_rate="$(normalize_exp_rate "$rate" "$TYPE")"
            EXP_FILE="$EXP_BASE/$real_dir/ascend-vllm-120-$exp_rate.exp"
        else
            EXP_FILE="$EXP_BASE/$model/distserve-120-$rate.exp"
        fi

        CSV_FILE="$LATENCY_ROOT/$TYPE/organized_data/$model/rate_$rate/request_latency.csv"

        for scale in "${SLO_SCALE_VALUES[@]}"; do
            scale_tag="$(scale_label "$scale")"
            prefill_slo="$(scale_slo "$PREFILL_SLO_BASE" "$scale")"
            decode_slo="$(scale_slo "$DECODE_SLO_BASE" "$scale")"
            total_slo="$(scale_slo "$TOTAL_SLO_BASE" "$scale")"

            exp_output="$SLO_ROOT/$TYPE/raw/$model/rate_$rate/scale_${scale_tag}/actual/exp_analysis.txt"
            csv_output="$SLO_ROOT/$TYPE/raw/$model/rate_$rate/scale_${scale_tag}/sim/csv_analysis.txt"
            compare_output="$SLO_ROOT/$TYPE/compared/$model/rate_$rate/scale_${scale_tag}/comparison.txt"

            "$PYTHON_BIN" merged_analyze.py \
                --exp-file "$EXP_FILE" \
                --csv-file "$CSV_FILE" \
                --prefill-slo "$prefill_slo" \
                --decode-slo "$decode_slo" \
                --total-slo "$total_slo" \
                --exp-output "$exp_output" \
                --csv-output "$csv_output" \
                --compare-output "$compare_output" \
                "${IS_VLLM_ARGS[@]}"

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

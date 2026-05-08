#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TYPE="${TYPE:-ascend_vllm}" 
MODE="${MODE:-val}"
MODELS="${MODELS:-llama_1B llama_3B llama_7B llama_8B}"
SLO_SCALES="${SLO_SCALES:-0.4 0.6 0.8 1.0 1.2}"
NUM_REQUESTS="120"
SEED="0"
ARRIVAL="poisson"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RATES="${RATES:-1.0 1.5 2.0 2.5 3.0 3.5 4.0}"
WORKLOAD_BASE="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0"
LATENCY_ROOT="${LATENCY_ROOT:-/users/rh/DistServe/simdistserve/benchmarks/validate_simulator/result/latency}"
PREFILL_FIRST_TOKEN_VISIBLE_IMMEDIATELY="${PREFILL_FIRST_TOKEN_VISIBLE_IMMEDIATELY:-1}"

read -r -a RATES <<< "$RATES"
read -r -a MODELS <<< "$MODELS"
SLO_SCALES_PY="[${SLO_SCALES// /, }]"

EXTRA_SIM_ARGS=()

case "$PREFILL_FIRST_TOKEN_VISIBLE_IMMEDIATELY" in
    1|true|TRUE|yes|YES)
        EXTRA_SIM_ARGS+=(--prefill-first-token-visible-immediately)
        ;;
    0|false|FALSE|no|NO)
        EXTRA_SIM_ARGS+=(--no-prefill-first-token-visible-immediately)
        ;;
    *)
        echo "Error: invalid PREFILL_FIRST_TOKEN_VISIBLE_IMMEDIATELY='$PREFILL_FIRST_TOKEN_VISIBLE_IMMEDIATELY'" >&2
        exit 1
        ;;
esac

declare -A WORKLOAD_MAP
declare -A MODEL_PATH_MAP

WORKLOAD_MAP["llama_1B"]="$WORKLOAD_BASE/llama-3.2-1B/${MODE}.jsonl"
WORKLOAD_MAP["llama_3B"]="$WORKLOAD_BASE/llama-3.2-3B/${MODE}.jsonl"
WORKLOAD_MAP["llama_7B"]="$WORKLOAD_BASE/llama-2-7b/${MODE}.jsonl"
WORKLOAD_MAP["llama_8B"]="$WORKLOAD_BASE/llama-3.1-8B/${MODE}.jsonl"

MODEL_PATH_MAP["llama_1B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2"
MODEL_PATH_MAP["llama_3B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2"
MODEL_PATH_MAP["llama_7B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2"
MODEL_PATH_MAP["llama_8B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2"

case "$TYPE" in
    cuda_distserve)
        BACKEND="distserve"
        PROFILE_ENV="SIMDISTSERVE_DISTSERVE_PROFILE"
        PROFILE_PATH="${PROFILE_PATH:-/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_5p4d.json}"
        ;;
    ascend_vllm)
        BACKEND="vllm_ascend"
        PROFILE_ENV="SIMDISTSERVE_VLLM_ASCEND_PROFILE"
        PROFILE_PATH="${PROFILE_PATH:-/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_filtered.json}"
        ;;
    *)
        echo "Error: Unsupported TYPE '$TYPE'. Use distserve_cuda or vllm_ascend."
        exit 1
        ;;
esac

export "${PROFILE_ENV}=${PROFILE_PATH}"

for MODEL in "${MODELS[@]}"; do
    if [[ -z "${WORKLOAD_MAP[$MODEL]:-}" ]] || [[ -z "${MODEL_PATH_MAP[$MODEL]:-}" ]]; then
        echo "Error: Unknown model '$MODEL'"
        exit 1
    fi

    WORKLOAD_FILE="${WORKLOAD_MAP[$MODEL]}"
    MODEL_PATH="${MODEL_PATH_MAP[$MODEL]}"

    for RATE in "${RATES[@]}"; do
        echo "Running simulate_dist: type=$TYPE mode=$MODE model=$MODEL rate=$RATE profile=$PROFILE_PATH"

        OUTPUT_DIR="$LATENCY_ROOT/$TYPE/${MODEL}/rate_${RATE}"
        mkdir -p "$OUTPUT_DIR"

        "$PYTHON_BIN" simulate_dist.py \
            --backend "$BACKEND" \
            --model "$MODEL_PATH" \
            --seed "$SEED" \
            --rate "$RATE" \
            --N "$NUM_REQUESTS" \
            --arrival "$ARRIVAL" \
            --workload "$WORKLOAD_FILE" \
            --output "$OUTPUT_DIR/sharegpt.json.sim.csv" \
            --name "${MODEL}/rate_${RATE}" \
            --output-request-latency "$OUTPUT_DIR/request_latency.csv" \
            --slo-scales "$SLO_SCALES_PY" \
            --output-request-event "$OUTPUT_DIR/request_event.csv" \
            --output-request-info "$OUTPUT_DIR/request_info.csv" \
            "${EXTRA_SIM_ARGS[@]}"
    done
done

echo "All rates completed for type=$TYPE mode=$MODE models=${MODELS[*]}."

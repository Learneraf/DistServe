#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/users/rh/miniconda3/envs/distserve/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

ROOT="/users/rh/DistServe/simdistserve/benchmarks"
SLO_SCALES="${SLO_SCALES:-[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]}"
PREFILL_SLO="${PREFILL_SLO:-1.0}"
DECODE_SLO="${DECODE_SLO:-1.0}"
TOTAL_SLO="${TOTAL_SLO:-1.0}"

CUDA_VALIDATION_ROOT="${CUDA_VALIDATION_ROOT:-/users/rh/DistServe/simdistserve/benchmarks/results/validation/distserve_cuda_final_3param}"
ASCEND_VALIDATION_ROOT="${ASCEND_VALIDATION_ROOT:-/users/rh/ascend_data/validation/final_3param_profile}"
CUDA_OUT="${CUDA_OUT:-/users/rh/DistServe/simdistserve/benchmarks/results/slo_scale_plots_current/distserve_cuda_final_3param}"
ASCEND_OUT="${ASCEND_OUT:-/users/rh/DistServe/simdistserve/benchmarks/results/slo_scale_plots_current/vllm_ascend_final_3param}"
CUDA_DASHBOARD_OUT="${CUDA_DASHBOARD_OUT:-/users/rh/DistServe/simdistserve/benchmarks/results/slo/distserve_cuda_final_3param/plots}"
ASCEND_DASHBOARD_OUT="${ASCEND_DASHBOARD_OUT:-/users/rh/DistServe/simdistserve/benchmarks/results/slo/vllm_ascend_final_3param/plots}"

MODELS=(llama_1B llama_3B llama_7B llama_8B)
RATES=(1 1.5 2 2.5 3 3.5 4)

echo "Base SLOs:"
echo "  prefill=${PREFILL_SLO}s decode=${DECODE_SLO}s total=${TOTAL_SLO}s"
echo "  scales=${SLO_SCALES}"

echo
echo "Replotting CUDA validation dashboard..."
"$PYTHON_BIN" "$ROOT/plot_validation_summary.py" \
    --summary-csv "$CUDA_VALIDATION_ROOT/summary.csv" \
    --output-dir "$CUDA_DASHBOARD_OUT" \
    --sim-prefix new

echo
echo "Replotting Ascend validation dashboard..."
"$PYTHON_BIN" "$ROOT/plot_validation_summary.py" \
    --summary-csv "$ASCEND_VALIDATION_ROOT/summary.csv" \
    --output-dir "$ASCEND_DASHBOARD_OUT" \
    --sim-prefix new

echo
echo "Generating current CUDA SLO-scale plots..."
for MODEL in "${MODELS[@]}"; do
    for RATE in "${RATES[@]}"; do
        "$PYTHON_BIN" "$ROOT/plot_slo_scale_for_rate.py" \
            --backend distserve_cuda \
            --model "$MODEL" \
            --rate "$RATE" \
            --benchmark-file "/users/rh/DistServe/evaluation/2-benchmark-serving/result/${MODEL}/distserve-100-${RATE}.exp" \
            --benchmark-format exp \
            --simulator-file "$CUDA_VALIDATION_ROOT/${MODEL}/rate_${RATE}/request_latency.csv" \
            --prefill-slo "$PREFILL_SLO" \
            --decode-slo "$DECODE_SLO" \
            --total-slo "$TOTAL_SLO" \
            --slo-scales "$SLO_SCALES" \
            --output-dir "$CUDA_OUT"
    done
done

echo
echo "Generating current Ascend SLO-scale plots..."
for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
        llama_1B) REAL_MODEL=llama1B ;;
        llama_3B) REAL_MODEL=llama3B ;;
        llama_7B) REAL_MODEL=llama7B ;;
        llama_8B) REAL_MODEL=llama8B ;;
        *) echo "Unknown model alias: $MODEL" >&2; exit 1 ;;
    esac

    for RATE in "${RATES[@]}"; do
        REAL_RATE="$RATE"
        if [[ "$RATE" == "1" ]]; then
            REAL_RATE="1.0"
        fi

        "$PYTHON_BIN" "$ROOT/plot_slo_scale_for_rate.py" \
            --backend vllm_ascend \
            --model "$MODEL" \
            --rate "$RATE" \
            --benchmark-file "/users/rh/ascend_data/ascend_vllm_holdout/${REAL_MODEL}/ascend-vllm-120-${REAL_RATE}.exp" \
            --benchmark-format exp \
            --simulator-file "$ASCEND_VALIDATION_ROOT/${MODEL}/rate_${RATE}/request_latency.csv" \
            --prefill-slo "$PREFILL_SLO" \
            --decode-slo "$DECODE_SLO" \
            --total-slo "$TOTAL_SLO" \
            --slo-scales "$SLO_SCALES" \
            --output-dir "$ASCEND_OUT"
    done
done

echo
echo "Done."
echo "CUDA dashboard plots:   $CUDA_DASHBOARD_OUT"
echo "Ascend dashboard plots: $ASCEND_DASHBOARD_OUT"
echo "CUDA SLO-scale plots:   $CUDA_OUT"
echo "Ascend SLO-scale plots: $ASCEND_OUT"

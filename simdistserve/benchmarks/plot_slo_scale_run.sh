#!/bin/bash

set -euo pipefail

BACKEND="${BACKEND:-distserve_cuda}"
MODEL="${MODEL:-llama_3B}"
RATE="${RATE:-4}"
PREFILL_SLO="${PREFILL_SLO:-1.0}"
DECODE_SLO="${DECODE_SLO:-1.0}"
TOTAL_SLO="${TOTAL_SLO:-1.0}"
SLO_SCALES="${SLO_SCALES:-[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]}"
OUTPUT_DIR="${OUTPUT_DIR:-/users/rh/DistServe/simdistserve/benchmarks/results/slo_scale_plots}"

PYTHON_BIN="/users/rh/miniconda3/envs/distserve/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" /users/rh/DistServe/simdistserve/benchmarks/plot_slo_scale_for_rate.py \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --rate "$RATE" \
    --prefill-slo "$PREFILL_SLO" \
    --decode-slo "$DECODE_SLO" \
    --total-slo "$TOTAL_SLO" \
    --slo-scales "$SLO_SCALES" \
    --output-dir "$OUTPUT_DIR"

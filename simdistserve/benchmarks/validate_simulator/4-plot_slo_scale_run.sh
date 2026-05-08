#!/bin/bash
set -euo pipefail

BACKEND="${BACKEND:-cuda_distserve}"
MODEL="${MODEL:-llama_3B}"
RATE="${RATE:-4}"
TTFT_TARGET="${TTFT_TARGET:-0.2}"
TPOT_TARGET="${TPOT_TARGET:-0.03}"
TOTAL_TARGET="${TOTAL_TARGET:-1.0}"
SLO_SCALES="${SLO_SCALES:-[0.8, 1.0, 1.2]}"
OUTPUT_DIR="${OUTPUT_DIR:-result/slo_scale_plots}"

PYTHON_BIN="python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" ./plot_slo_scale_for_rate.py \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --rate "$RATE" \
    --ttft-slo "$TTFT_TARGET" \
    --tpot-slo "$TPOT_TARGET" \
    --total-slo "$TOTAL_TARGET" \
    --slo-scales "$SLO_SCALES" \
    --output-dir "$OUTPUT_DIR"

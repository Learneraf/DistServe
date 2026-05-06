#!/bin/bash

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/users/rh/cuda_data/sim/3p3d_fit_model_forward/slo_affine_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/plots}"
FORMATS="${FORMATS:-png}"
PYTHON_BIN="${PYTHON_BIN:-/users/rh/miniconda3/bin/python3.13}"

read -r -a FORMAT_ARGS <<< "$FORMATS"

"$PYTHON_BIN" /users/rh/DistServe/simdistserve/benchmarks/plot_cuda_slo_affine_eval.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --formats "${FORMAT_ARGS[@]}"

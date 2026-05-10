#!/bin/bash

set -euo pipefail

BACKENDS_STR="${BACKENDS:-cuda_distserve ascend_vllm cuda_vllm}"
MODELS_STR="${MODELS:-llama_1B llama_3B llama_7B llama_8B}"
RATES_STR="${RATES:-1.0 1.5 2.0 2.5 3.0 3.5 4.0}"

read -r -a BACKENDS <<< "$BACKENDS_STR"
read -r -a MODELS <<< "$MODELS_STR"
read -r -a RATES <<< "$RATES_STR"

for BACKEND in "${BACKENDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for RATE in "${RATES[@]}"; do
            BACKEND="$BACKEND" MODEL="$MODEL" RATE="$RATE" \
                ./4-plot_slo_scale_run.sh
        done
    done
done

#!/bin/bash

set -euo pipefail

BACKENDS_STR="${BACKENDS:-distserve_cuda vllm_ascend}"
MODELS_STR="${MODELS:-llama_1B llama_3B llama_7B llama_8B}"
RATES_STR="${RATES:-1 1.5 2 2.5 3 3.5 4}"

read -r -a BACKENDS <<< "$BACKENDS_STR"
read -r -a MODELS <<< "$MODELS_STR"
read -r -a RATES <<< "$RATES_STR"

for BACKEND in "${BACKENDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for RATE in "${RATES[@]}"; do
            BACKEND="$BACKEND" MODEL="$MODEL" RATE="$RATE" \
                /users/rh/DistServe/simdistserve/benchmarks/plot_slo_scale_run.sh
        done
    done
done

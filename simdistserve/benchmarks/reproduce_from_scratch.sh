#!/bin/bash
set -euo pipefail

ROOT="/users/rh/DistServe/simdistserve/benchmarks"

RUN_CUDA="${RUN_CUDA:-1}"
RUN_ASCEND="${RUN_ASCEND:-1}"
RUN_PLOTS="${RUN_PLOTS:-1}"

if [[ "$RUN_CUDA" == "1" ]]; then
    bash "$ROOT/refit_and_verify_distserve_cuda_existing.sh"
fi

if [[ "$RUN_ASCEND" == "1" ]]; then
    bash "$ROOT/refit_and_verify_vllm_ascend_existing.sh"
fi

if [[ "$RUN_PLOTS" == "1" ]]; then
    bash "$ROOT/reproduce_current_results.sh"
fi

echo "Full reproduction pipeline completed."

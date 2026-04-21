#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-/users/rh/miniconda3/envs/distserve/bin/python}"
REPO_ROOT="/users/rh/DistServe"

RESULTS_ROOT="${RESULTS_ROOT:-/users/rh/DistServe/evaluation/2-benchmark-serving/result}"
PROFILE_FINAL="${PROFILE_FINAL:-/users/rh/DistServe/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_cuda_data.json}"

VALIDATION_ROOT="${VALIDATION_ROOT:-/users/rh/DistServe/simdistserve/benchmarks/results/validation/distserve_cuda_final_3param}"
WORKLOAD="${WORKLOAD:-/users/rh/DistServe/evaluation/2-benchmark-serving/data/sampled_sharegpt_pure.jsonl}"
BENCH_ROOT="${BENCH_ROOT:-/users/rh/DistServe/evaluation/2-benchmark-serving/result}"

SEED="${SEED:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
ARRIVAL="${ARRIVAL:-poisson}"
CV="${CV:-1.0}"
PREFILL_SLO="${PREFILL_SLO:-1.0}"
DECODE_SLO="${DECODE_SLO:-1.0}"
TOTAL_SLO="${TOTAL_SLO:-1.0}"

mkdir -p "$(dirname "$PROFILE_FINAL")" "$VALIDATION_ROOT"

(
    cd "$REPO_ROOT"
    "$PYTHON" simdistserve/estimators/fit_params/retrain_distserve_cuda_live.py \
        --results-root "$RESULTS_ROOT" \
        --output "$PROFILE_FINAL"
)

(
    cd "$REPO_ROOT"
    "$PYTHON" simdistserve/benchmarks/validate_distserve_cuda_profile.py \
        --profile "$PROFILE_FINAL" \
        --output-root "$VALIDATION_ROOT" \
        --workload "$WORKLOAD" \
        --bench-root "$BENCH_ROOT" \
        --python-bin "$PYTHON" \
        --seed "$SEED" \
        --num-prompts "$NUM_PROMPTS" \
        --arrival "$ARRIVAL" \
        --cv "$CV" \
        --prefill-slo "$PREFILL_SLO" \
        --decode-slo "$DECODE_SLO" \
        --total-slo "$TOTAL_SLO"
)

echo "CUDA refit and verification completed."
echo "Profile file: $PROFILE_FINAL"
echo "Summary:      $VALIDATION_ROOT/summary.csv"

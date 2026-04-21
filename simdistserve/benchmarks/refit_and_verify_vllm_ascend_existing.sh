#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-/users/rh/miniconda3/envs/distserve/bin/python}"
REPO_ROOT="/users/rh/DistServe"

PREFILL_RESULTS_ROOT="${PREFILL_RESULTS_ROOT:-/users/rh/ascend_data/ascend_compute_grid}"
DECODE_RESULTS_ROOT="${DECODE_RESULTS_ROOT:-/users/rh/ascend_data/ascend_proxy_grid}"
PROFILE_BASE="${PROFILE_BASE:-/users/rh/DistServe/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_all.json}"
PROFILE_FINAL="${PROFILE_FINAL:-/users/rh/ascend_data/fitted_profiles/fit_params_live_ascend_data.json}"

VALIDATION_ROOT="${VALIDATION_ROOT:-/users/rh/ascend_data/validation/final_3param_profile}"
WORKLOAD_ROOT="${WORKLOAD_ROOT:-/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0}"
BENCH_ROOT="${BENCH_ROOT:-/users/rh/ascend_data/ascend_vllm_holdout}"
PREV_SUMMARY="${PREV_SUMMARY:-/users/rh/ascend_data/validation/refit_profile/summary.json}"

SEED="${SEED:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-120}"
ARRIVAL="${ARRIVAL:-poisson}"
CV="${CV:-1.0}"
PREFILL_SLO="${PREFILL_SLO:-1.0}"
DECODE_SLO="${DECODE_SLO:-1.0}"
TOTAL_SLO="${TOTAL_SLO:-1.0}"

mkdir -p "$(dirname "$PROFILE_FINAL")" "$VALIDATION_ROOT"

(
    cd "$REPO_ROOT"
    "$PYTHON" simdistserve/estimators/fit_params/retrain_vllm_ascend_live.py \
        --prefill-results-root "$PREFILL_RESULTS_ROOT" \
        --decode-results-root "$DECODE_RESULTS_ROOT" \
        --output "$PROFILE_FINAL"
)

(
    cd "$REPO_ROOT"
    "$PYTHON" simdistserve/benchmarks/validate_vllm_ascend_profile.py \
        --profile "$PROFILE_FINAL" \
        --output-root "$VALIDATION_ROOT" \
        --workload-root "$WORKLOAD_ROOT" \
        --bench-root "$BENCH_ROOT" \
        --prev-summary "$PREV_SUMMARY" \
        --python-bin "$PYTHON" \
        --seed "$SEED" \
        --num-prompts "$NUM_PROMPTS" \
        --arrival "$ARRIVAL" \
        --cv "$CV" \
        --prefill-slo "$PREFILL_SLO" \
        --decode-slo "$DECODE_SLO" \
        --total-slo "$TOTAL_SLO"
)

echo "Ascend refit and verification completed."
echo "Profile file: $PROFILE_FINAL"
echo "Summary:      $VALIDATION_ROOT/summary.csv"

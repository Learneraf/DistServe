#!/bin/bash
set -euo pipefail

PYTHON="${PYTHON:-/users/rh/miniconda3/envs/distserve/bin/python}"
REPO_ROOT="/users/rh/DistServe"

SEED_PROFILE="${SEED_PROFILE:-/users/rh/ascend_data/fitted_profiles/fit_params_live_ascend_data.json}"
WORKLOAD_ROOT="${WORKLOAD_ROOT:-/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0}"
REAL_FIT_ROOT="${REAL_FIT_ROOT:-/users/rh/ascend_data/ascend_vllm_split/fit}"
REAL_VALID_ROOT="${REAL_VALID_ROOT:-/users/rh/ascend_data/ascend_vllm_split/val}"
PROFILE_FINAL="${PROFILE_FINAL:-/users/rh/ascend_data/fitted_profiles/fit_params_split_component.json}"
FIT_WORK_DIR="${FIT_WORK_DIR:-/users/rh/ascend_data/validation/fit_work_split_component}"
VALIDATION_ROOT="${VALIDATION_ROOT:-/users/rh/ascend_data/validation/split_component_profile}"

SEED="${SEED:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-120}"
ARRIVAL="${ARRIVAL:-poisson}"
CV="${CV:-1.0}"
PREFILL_SLO="${PREFILL_SLO:-1.0}"
DECODE_SLO="${DECODE_SLO:-1.0}"
TOTAL_SLO="${TOTAL_SLO:-1.0}"

mkdir -p "$(dirname "$PROFILE_FINAL")" "$FIT_WORK_DIR" "$VALIDATION_ROOT"

(
    cd "$REPO_ROOT"
    "$PYTHON" simdistserve/estimators/fit_params/retrain_split_component_profile.py \
        --backend vllm_ascend \
        --seed-profile "$SEED_PROFILE" \
        --workload-root "$WORKLOAD_ROOT" \
        --real-fit-root "$REAL_FIT_ROOT" \
        --output "$PROFILE_FINAL" \
        --work-dir "$FIT_WORK_DIR" \
        --python-bin "$PYTHON" \
        --seed "$SEED" \
        --num-prompts "$NUM_PROMPTS" \
        --arrival "$ARRIVAL" \
        --cv "$CV"
)

(
    cd "$REPO_ROOT"
    "$PYTHON" simdistserve/benchmarks/validate_split_component_profile.py \
        --backend vllm_ascend \
        --profile "$PROFILE_FINAL" \
        --output-root "$VALIDATION_ROOT" \
        --workload-root "$WORKLOAD_ROOT" \
        --real-valid-root "$REAL_VALID_ROOT" \
        --python-bin "$PYTHON" \
        --seed "$SEED" \
        --num-prompts "$NUM_PROMPTS" \
        --arrival "$ARRIVAL" \
        --cv "$CV" \
        --prefill-slo "$PREFILL_SLO" \
        --decode-slo "$DECODE_SLO" \
        --total-slo "$TOTAL_SLO"
)

echo "vLLM Ascend split-component refit and validation completed."
echo "Profile file: $PROFILE_FINAL"
echo "Fit summary:  $FIT_WORK_DIR/fit_summary.json"
echo "Validation:   $VALIDATION_ROOT/summary.csv"

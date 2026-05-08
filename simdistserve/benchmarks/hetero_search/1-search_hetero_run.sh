#!/usr/bin/env bash
set -euo pipefail

# Run heterogeneous DistServe search for one or more model configs.
#
# Defaults:
#   - runs llama_1B, llama_3B, llama_7B, llama_8B
#   - evaluates milp, no_cross, and cuda_prefill_ascend_decode baselines
#   - writes results to simdistserve/hetero/results/search
#
# Optional overrides:
#   MODELS="llama_7B llama_8B" bash search_hetero_run.sh
#   CONFIG_NAME=example_search_config.json bash search_hetero_run.sh
#   OUTPUT_ROOT=/path/to/results bash search_hetero_run.sh
#   MU_CACHE_PATH=/path/to/mu_cache.json bash search_hetero_run.sh
#   DISTSERVE_PROFILE=/path/to/cuda_profile.json bash search_hetero_run.sh
#   VLLM_ASCEND_PROFILE=/path/to/ascend_profile.json bash search_hetero_run.sh

get_project_root() {
    local current="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 脚本所在目录的绝对路径
    while [[ "$current" != "/" ]]; do
        if [[ "$(basename "$current")" == "DistServe" ]]; then
            echo "$current"
            return 0
        fi
        current="$(dirname "$current")"
    done
    echo "错误：找不到项目根目录 DistServe" >&2
    return 1
}

REPO_ROOT="${REPO_ROOT:-$(get_project_root)}"
PYTHON="${PYTHON:-python}"

CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/simdistserve/hetero/configs}"
CONFIG_NAME="${CONFIG_NAME:-search_4nodes_high_affinity.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/simdistserve/benchmarks/hetero_search/result/search}"
MU_CACHE_PATH="${MU_CACHE_PATH:-${REPO_ROOT}/simdistserve/benchmarks/hetero_search/result/cache/single_instance_goodput_cache.json}"

CUDA_PROFILE="${CUDA_PROFILE:-${REPO_ROOT}/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_5p4d.json}"
ASCEND_PROFILE="${ASCEND_PROFILE:-${REPO_ROOT}/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_filtered.json}"

MODELS="${MODELS:-llama_1B llama_3B llama_7B llama_8B}"

cd "${REPO_ROOT}"

echo "repo=${REPO_ROOT}"
echo "python=${PYTHON}"
echo "config_root=${CONFIG_ROOT}"
echo "config_name=${CONFIG_NAME}"
echo "output_root=${OUTPUT_ROOT}"
echo "mu_cache_path=${MU_CACHE_PATH}"
echo "cuda_profile=${CUDA_PROFILE}"
echo "ascend_profile=${ASCEND_PROFILE}"
echo "models=${MODELS}"

for model in ${MODELS}; do
  config="${CONFIG_ROOT}/${model}/${CONFIG_NAME}"
  if [[ ! -f "${config}" ]]; then
    echo "Missing config for ${model}: ${config}" >&2
    exit 1
  fi

  echo "=================================================================="
  echo "Running heterogeneous search for ${model}"
  "${PYTHON}" -m simdistserve.benchmarks.hetero_search.run_hetero_search_suite \
    --config "${config}" \
    --output-root "${OUTPUT_ROOT}" \
    --mu-cache-path "${MU_CACHE_PATH}" \
    --cuda-profile "${CUDA_PROFILE}" \
    --ascend-profile "${ASCEND_PROFILE}"
done

echo "=================================================================="
echo "Done. Results written under ${OUTPUT_ROOT}"

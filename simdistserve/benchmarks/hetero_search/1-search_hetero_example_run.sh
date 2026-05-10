#!/usr/bin/env bash
set -euo pipefail

# Run the high-bandwidth example heterogeneous search configs for all Llama models.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    exit 1
}

REPO_ROOT="${REPO_ROOT:-$(get_project_root)}"

CONFIG_NAME="${CONFIG_NAME:-example_search_config_distserve.json}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/simdistserve/benchmarks/hetero_search/result/search/example_configs_distserve}" \
CUDA_PROFILE="${REPO_ROOT}/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_live_5p4d.json" \
ASCEND_PROFILE="${REPO_ROOT}/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_filtered.json" \
MU_CACHE_PATH="${REPO_ROOT}/simdistserve/benchmarks/hetero_search/result/cache/distserve_0509.json" \
bash "${SCRIPT_DIR}/1-search_hetero_run.sh"

CONFIG_NAME="${CONFIG_NAME:-example_search_config_vllm.json}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/simdistserve/benchmarks/hetero_search/result/search/example_configs_vllm}" \
CUDA_PROFILE="${REPO_ROOT}/simdistserve/estimators/profiled_data/distserve-cuda/fit_params_cuda_data_fit_5p4d_infer_batch.json" \
ASCEND_PROFILE="${REPO_ROOT}/simdistserve/estimators/profiled_data/vllm-ascend/fit_params_live_5p4d_filtered.json" \
MU_CACHE_PATH="${REPO_ROOT}/simdistserve/benchmarks/hetero_search/result/cache/vllm_0509.json" \
bash "${SCRIPT_DIR}/1-search_hetero_run.sh"
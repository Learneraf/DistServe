#!/usr/bin/env bash
set -euo pipefail

VLLM_PYTHON="${VLLM_PYTHON:-/tmp/vllm014/bin/python}"
MODEL="${MODEL:-llama_1B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
RESULT_ROOT="${RESULT_ROOT:-/users/rh/DistServe/evaluation/2-benchmark-serving/result/vllm_fit_with_batches}"
TRACE_PATH="${TRACE_PATH:-${RESULT_ROOT}/${MODEL}/vllm_batch_trace.jsonl}"

declare -A MODEL_PATH_MAP=(
  ["llama_1B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B/converted_bin_v2"
  ["llama_3B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B/converted_bin_v2"
  ["llama_7B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2"
  ["llama_8B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B/converted_bin_v2"
)

MODEL_PATH="${MODEL_PATH_MAP[${MODEL}]:-${MODEL}}"
mkdir -p "$(dirname "${TRACE_PATH}")"
rm -f "${TRACE_PATH}"

export VLLM_BATCH_TRACE_PATH="${TRACE_PATH}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TMPDIR="${TMPDIR:-/tmp}"

echo "Starting vLLM server"
echo "Model: ${MODEL_PATH}"
echo "Trace: ${VLLM_BATCH_TRACE_PATH}"

exec "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --disable-log-requests

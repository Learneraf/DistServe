#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="/users/rh/DistServe/evaluation/2-benchmark-serving"
VLLM_PYTHON="${VLLM_PYTHON:-/tmp/vllm014/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/users/rh/cdua_data}"
TRACE_ROOT="${TRACE_ROOT:-/tmp/cdua_data_batch_traces}"
MODELS="${MODELS:-llama_1B llama_3B llama_7B llama_8B}"
RATES="${RATES:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-120}"
DATASET_MODE="${DATASET_MODE:-fit}"
HOST="${HOST:-127.0.0.1}"
KV_HOST="${KV_HOST:-$(hostname -I | awk '{print $1}')}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-18000}"
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-18001}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-19000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-19001}"
TOKENIZER_ROOT="${TOKENIZER_ROOT:-/tmp/vllm_tokenizers}"
KV_MEM_POOL_SIZE_GB="${KV_MEM_POOL_SIZE_GB:-1}"
KV_BUFFER_SIZE_BYTES="${KV_BUFFER_SIZE_BYTES:-1000000000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

declare -A MODEL_PATH_MAP=(
  ["llama_1B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B/converted_bin_v2"
  ["llama_3B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-3B/converted_bin_v2"
  ["llama_7B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/llama-2-7b/converted_bin_v2"
  ["llama_8B"]="/users/rh/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3___1-8B/converted_bin_v2"
)

declare -A DATASET_MAP=(
  ["llama_1B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.2-1B/${DATASET_MODE}.jsonl"
  ["llama_3B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.2-3B/${DATASET_MODE}.jsonl"
  ["llama_7B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-2-7b/${DATASET_MODE}.jsonl"
  ["llama_8B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.1-8B/${DATASET_MODE}.jsonl"
)

wait_for_health() {
  local port="$1"
  local name="$2"
  for _ in $(seq 1 900); do
    if curl -fsS "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      echo "${name} is healthy on port ${port}"
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${name} on port ${port}" >&2
  return 1
}

prepare_tokenizer() {
  local model="$1"
  local model_path="$2"
  local tokenizer_dir="${TOKENIZER_ROOT}/${model}"
  mkdir -p "${tokenizer_dir}"
  python3 - "${model_path}" "${tokenizer_dir}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
for name in (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
):
    src_file = src / name
    if src_file.exists():
        shutil.copy2(src_file, dst / name)

cfg_file = dst / "tokenizer_config.json"
if cfg_file.exists():
    cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    if cfg.get("tokenizer_class") == "TokenizersBackend":
        cfg.pop("tokenizer_class", None)
    cfg_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
PY
  echo "${tokenizer_dir}"
}

cleanup_servers() {
  if [[ -n "${PREFILL_PID:-}" ]]; then
    kill "${PREFILL_PID}" >/dev/null 2>&1 || true
    wait "${PREFILL_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${DECODE_PID:-}" ]]; then
    kill "${DECODE_PID}" >/dev/null 2>&1 || true
    wait "${DECODE_PID}" >/dev/null 2>&1 || true
  fi
  PREFILL_PID=""
  DECODE_PID=""
}

trap cleanup_servers EXIT

cd "${BENCH_DIR}"
mkdir -p "${RESULT_ROOT}"
mkdir -p "${TRACE_ROOT}"

for model in ${MODELS}; do
  model_path="${MODEL_PATH_MAP[${model}]:-}"
  dataset="${DATASET_MAP[${model}]:-}"
  if [[ -z "${model_path}" || ! -d "${model_path}" ]]; then
    echo "Model path not found for ${model}: ${model_path}" >&2
    exit 1
  fi
  if [[ -z "${dataset}" || ! -f "${dataset}" ]]; then
    echo "Dataset not found for ${model}: ${dataset}" >&2
    exit 1
  fi

  model_dir="${RESULT_ROOT}/${model}"
  tokenizer_dir="$(prepare_tokenizer "${model}" "${model_path}")"
  trace_dir="${TRACE_ROOT}/${model}"
  mkdir -p "${model_dir}"
  rm -rf "${trace_dir}"
  mkdir -p "${trace_dir}"
  rm -f \
    "${model_dir}/prefill_batch_trace.jsonl" \
    "${model_dir}/decode_batch_trace.jsonl" \
    "${model_dir}/prefill_batch_trace.jsonl.gz" \
    "${model_dir}/decode_batch_trace.jsonl.gz"

  prefill_kv_config=$(python3 - <<PY
import json
print(json.dumps({
    "kv_connector": "P2pNcclConnector",
    "kv_role": "kv_producer",
    "kv_rank": 0,
    "kv_parallel_size": 2,
    "kv_ip": "${KV_HOST}",
    "kv_port": ${PREFILL_KV_PORT},
    "kv_buffer_size": ${KV_BUFFER_SIZE_BYTES},
    "kv_connector_extra_config": {"mem_pool_size_gb": ${KV_MEM_POOL_SIZE_GB}},
}))
PY
)
  decode_kv_config=$(python3 - <<PY
import json
print(json.dumps({
    "kv_connector": "P2pNcclConnector",
    "kv_role": "kv_consumer",
    "kv_rank": 1,
    "kv_parallel_size": 2,
    "kv_ip": "${KV_HOST}",
    "kv_port": ${DECODE_KV_PORT},
    "kv_buffer_size": ${KV_BUFFER_SIZE_BYTES},
    "kv_connector_extra_config": {"mem_pool_size_gb": ${KV_MEM_POOL_SIZE_GB}},
}))
PY
)

  echo "=================================================================="
  echo "Starting ${model}: prefill GPU ${PREFILL_GPU}, decode GPU ${DECODE_GPU}"
  echo "HTTP host: ${HOST}, KV host: ${KV_HOST}"
  echo "Results: ${model_dir}"

  CUDA_VISIBLE_DEVICES="${PREFILL_GPU}" \
  VLLM_BATCH_TRACE_PATH="${trace_dir}/prefill_batch_trace.jsonl" \
  VLLM_PRESERVE_REQUEST_ID=1 \
  HF_HOME="${HF_HOME:-/tmp/hf_cache}" \
  VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}" \
  TMPDIR="${TMPDIR:-/tmp}" \
  "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --tokenizer "${tokenizer_dir}" \
    --served-model-name "${model}" \
    --host "${HOST}" \
    --port "${PREFILL_HTTP_PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 2048 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --disable-log-requests \
    --enable-request-id-headers \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config "${prefill_kv_config}" \
    > "${model_dir}/prefill_server.log" 2>&1 &
  PREFILL_PID="$!"

  CUDA_VISIBLE_DEVICES="${DECODE_GPU}" \
  VLLM_BATCH_TRACE_PATH="${trace_dir}/decode_batch_trace.jsonl" \
  VLLM_PRESERVE_REQUEST_ID=1 \
  HF_HOME="${HF_HOME:-/tmp/hf_cache}" \
  VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}" \
  TMPDIR="${TMPDIR:-/tmp}" \
  "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --tokenizer "${tokenizer_dir}" \
    --served-model-name "${model}" \
    --host "${HOST}" \
    --port "${DECODE_HTTP_PORT}" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --max-model-len 2048 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --disable-log-requests \
    --enable-request-id-headers \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config "${decode_kv_config}" \
    > "${model_dir}/decode_server.log" 2>&1 &
  DECODE_PID="$!"

  wait_for_health "${PREFILL_HTTP_PORT}" "${model} prefill"
  wait_for_health "${DECODE_HTTP_PORT}" "${model} decode"

  for rate in ${RATES}; do
    echo "Running ${model}, request_rate=${rate}"
    "${VLLM_PYTHON}" "${BENCH_DIR}/run_vllm_pd_benchmark.py" \
      --model-alias "${model}" \
      --tokenizer "${tokenizer_dir}" \
      --dataset "${dataset}" \
      --output "${model_dir}/vllm-pd-${NUM_PROMPTS}-${rate}.exp" \
      --host "${HOST}" \
      --prefill-port "${PREFILL_HTTP_PORT}" \
      --decode-port "${DECODE_HTTP_PORT}" \
      --kv-host "${KV_HOST}" \
      --prefill-kv-port "${PREFILL_KV_PORT}" \
      --decode-kv-port "${DECODE_KV_PORT}" \
      --num-prompts "${NUM_PROMPTS}" \
      --request-rate "${rate}" \
      --request-id-prefix "${model}-rate-${rate}-bench" \
      --seed 0
    sleep 1
  done

  cleanup_servers
  if [[ -f "${trace_dir}/prefill_batch_trace.jsonl" ]]; then
    gzip -c "${trace_dir}/prefill_batch_trace.jsonl" > "${model_dir}/prefill_batch_trace.jsonl.gz"
  fi
  if [[ -f "${trace_dir}/decode_batch_trace.jsonl" ]]; then
    gzip -c "${trace_dir}/decode_batch_trace.jsonl" > "${model_dir}/decode_batch_trace.jsonl.gz"
  fi
done

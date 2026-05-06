#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="/users/rh/DistServe/evaluation/2-benchmark-serving"
PYTHON_BIN="${PYTHON_BIN:-/users/rh/miniconda3/envs/distserve/bin/python}"
MODEL="${MODEL:-llama_1B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RESULT_ROOT="${RESULT_ROOT:-${BENCH_DIR}/result/vllm_fit_with_batches}"
RATES="${RATES:-[(120, 1), (120, 1.5), (120, 2), (120, 2.5), (120, 3), (120, 3.5), (120, 4)]}"

declare -A DATASET_MAP=(
  ["llama_1B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.2-1B/fit.jsonl"
  ["llama_3B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.2-3B/fit.jsonl"
  ["llama_7B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-2-7b/fit.jsonl"
  ["llama_8B"]="/users/rh/DistServe/simdistserve/dataset/splits/sharegpt_four_models_common_ascend1900_seed0/llama-3.1-8B/fit.jsonl"
)

DATASET="${DATASET_MAP[${MODEL}]:-}"
if [[ -z "${DATASET}" || ! -f "${DATASET}" ]]; then
  echo "Dataset not found for ${MODEL}: ${DATASET}" >&2
  exit 1
fi

cd "${BENCH_DIR}"

"${PYTHON_BIN}" ./2-benchmark-serving.py \
  --backend openai \
  --dataset "${DATASET}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-model "${MODEL}" \
  --seed 0 \
  --num-prompts-req-rates "${RATES}" \
  --exp-result-root "${RESULT_ROOT}" \
  --exp-result-dir "${MODEL}" \
  --exp-result-prefix "vllm"

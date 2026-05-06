#!/usr/bin/env bash
set -euo pipefail

# Run the high-bandwidth example heterogeneous search configs for all Llama models.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_NAME="${CONFIG_NAME:-example_search_config.json}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/users/rh/DistServe/simdistserve/hetero/results/search/example_configs}" \
bash "${SCRIPT_DIR}/search_hetero_run.sh"

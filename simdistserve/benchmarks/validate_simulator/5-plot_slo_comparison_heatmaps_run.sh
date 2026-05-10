#!/bin/bash
set -euo pipefail

python ./plot_slo_comparison_heatmaps.py \
    --slo-root ./result/slo \
    --backends cuda_distserve ascend_vllm cuda_vllm
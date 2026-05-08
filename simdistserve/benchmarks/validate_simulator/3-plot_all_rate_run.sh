#!/bin/bash
set -euo pipefail

python ./plot_all_rate.py \
    --input_dir "./result/slo/ascend_vllm/compared/" \
    --output_dir "./result/slo/ascend_vllm/plots/"

python ./plot_all_rate.py \
    --input_dir "./result/slo/cuda_distserve/compared/" \
    --output_dir "./result/slo/cuda_distserve/plots/"
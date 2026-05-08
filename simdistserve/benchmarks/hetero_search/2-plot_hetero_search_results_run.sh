#!/usr/bin/env bash
set -euo pipefail

python ./plot_hetero_search_results.py \
    --result-root ./result/search/example_configs_distserve \
    --output-dir ./result/plots/example_configs_distserve \
    --thesis-figures-dir ./result/plots/example_configs_distserve \
    --font-path ../../../docs/fonts/TimesSimSunRegular.ttf \
    --models llama_1B llama_3B llama_7B llama_8B

python ./plot_hetero_search_results.py \
    --result-root ./result/search/example_configs_vllm \
    --output-dir ./result/plots/example_configs_vllm \
    --thesis-figures-dir ./result/plots/example_configs_vllm \
    --font-path ../../../docs/fonts/TimesSimSunRegular.ttf \
    --models llama_1B llama_3B llama_7B llama_8B
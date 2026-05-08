#!/bin/bash
set -euo pipefail

python ./plot_backend_slo_mean_delta.py \
    --slo-root ./result/slo \
    --output-dir ./result/slo \
    --font-path ../../../docs/fonts/TimesSimSunRegular.ttf
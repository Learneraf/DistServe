# DistServe Live Fit Ablation

- Benchmark files: 28
- Decode split threshold: 95

### Prefill

| Variant | CV mean abs % | CV RMSE % | CV max abs % | Fit-all mean abs % | Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 15.29 | 23.18 | 93.37 | 15.11 | 2426 |
| no_bs | 15.46 | 23.31 | 93.38 | 15.30 | 2426 |
| min_len | 15.99 | 23.99 | 178.64 | 15.77 | 2426 |
| mid_len | 15.70 | 23.59 | 128.10 | 15.51 | 2426 |
| no_max_len | 16.67 | 24.49 | 93.44 | 16.47 | 2426 |

### Decode (small batch)

| Variant | CV mean abs % | CV RMSE % | CV max abs % | Fit-all mean abs % | Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 9.22 | 17.93 | 1445.85 | 9.11 | 68140 |
| min_len | 11.90 | 19.57 | 1473.04 | 11.62 | 68140 |
| mid_len | 11.62 | 19.30 | 1461.68 | 11.39 | 68140 |
| no_max_len | 11.94 | 19.55 | 1480.96 | 11.72 | 68140 |

### Decode (large batch)

| Variant | CV mean abs % | CV RMSE % | CV max abs % | Fit-all mean abs % | Count |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| min_len | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| mid_len | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| no_max_len | 0.00 | 0.00 | 0.00 | 0.00 | 0 |

#### Prefill per-model: baseline

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 15.92 | 20.61 | 85.23 | 684 |
| llama_3B | 7.25 | 9.64 | 80.67 | 570 |
| llama_7B | 27.31 | 36.85 | 93.37 | 667 |
| llama_8B | 7.65 | 10.43 | 44.66 | 505 |

#### Prefill per-model: no_bs

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 16.12 | 20.86 | 85.17 | 684 |
| llama_3B | 7.60 | 10.14 | 82.89 | 570 |
| llama_7B | 27.30 | 36.84 | 93.38 | 667 |
| llama_8B | 7.81 | 10.65 | 44.00 | 505 |

#### Prefill per-model: min_len

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 16.15 | 21.78 | 178.64 | 684 |
| llama_3B | 8.70 | 12.29 | 85.34 | 570 |
| llama_7B | 27.24 | 36.81 | 93.35 | 667 |
| llama_8B | 9.16 | 12.77 | 45.25 | 505 |

#### Prefill per-model: mid_len

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 16.11 | 21.32 | 128.10 | 684 |
| llama_3B | 8.04 | 10.87 | 85.33 | 570 |
| llama_7B | 27.21 | 36.79 | 93.35 | 667 |
| llama_8B | 8.61 | 11.72 | 46.73 | 505 |

#### Prefill per-model: no_max_len

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 15.87 | 20.81 | 86.63 | 684 |
| llama_3B | 10.45 | 15.65 | 87.43 | 570 |
| llama_7B | 27.39 | 36.87 | 93.44 | 667 |
| llama_8B | 10.65 | 14.94 | 67.41 | 505 |

#### Decode small per-model: baseline

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 6.88 | 12.00 | 611.52 | 28415 |
| llama_3B | 5.86 | 10.24 | 115.92 | 14659 |
| llama_7B | 19.27 | 30.57 | 131.02 | 13404 |
| llama_8B | 7.58 | 17.97 | 1445.85 | 11662 |

#### Decode small per-model: min_len

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 9.91 | 14.95 | 597.23 | 28415 |
| llama_3B | 10.80 | 14.37 | 154.60 | 14659 |
| llama_7B | 19.36 | 30.61 | 137.86 | 13404 |
| llama_8B | 9.55 | 18.88 | 1473.04 | 11662 |

#### Decode small per-model: mid_len

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 9.67 | 14.58 | 644.57 | 28415 |
| llama_3B | 10.07 | 13.55 | 145.28 | 14659 |
| llama_7B | 19.33 | 30.60 | 134.94 | 13404 |
| llama_8B | 9.47 | 18.74 | 1461.68 | 11662 |

#### Decode small per-model: no_max_len

| Model | CV mean abs % | CV RMSE % | CV max abs % | Count |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 9.98 | 14.91 | 590.73 | 28415 |
| llama_3B | 10.63 | 14.20 | 153.72 | 14659 |
| llama_7B | 19.32 | 30.60 | 135.00 | 13404 |
| llama_8B | 9.89 | 19.03 | 1480.96 | 11662 |


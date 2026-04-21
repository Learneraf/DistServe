# Matched Length Distribution Analysis

This analysis asks a narrower question than the fit ablations:
holding total tokens fixed inside a batch, does a more extreme within-batch length distribution run slower than a more uniform one?

## Coverage

| Stage | Matched key | Total groups | Multi-sample groups | Varying-spread groups | Samples in varying-spread groups |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill exact | `(model, rate, batch_size, sum_prompt_len)` | 208 | 0 | 0 | 0 |
| Decode exact | `(model, rate, batch_size, sum_context_len)` | 55723 | 3218 | 3053 | 6546 |

Exact prefill duplicates do not occur in this corpus, so the strict same-sum experiment is only available for decode.
A small prefill near-match addendum is reported separately and should be treated as suggestive rather than decisive.

## Decode Exact Match

| Metric | Value |
| --- | ---: |
| usable matched groups | 3053 |
| high-spread slower groups % | 71.86 |
| median delta ms | 1.055 |
| mean delta ms | 0.507 |
| median slowdown % | 7.65 |
| mean slowdown % | 10.78 |

### Decode Per Model

| Model | Groups | High-spread slower % | Median delta ms | Median slowdown % |
| --- | ---: | ---: | ---: | ---: |
| llama_1B | 2155 | 73.09 | 0.995 | 9.31 |
| llama_3B | 421 | 72.45 | 1.173 | 4.74 |
| llama_7B | 266 | 51.88 | 0.190 | 0.97 |
| llama_8B | 211 | 83.41 | 2.628 | 7.64 |

### Decode By Minimum Spread Difference

| Min spread diff | Groups | High-spread slower % | Median delta ms | Median slowdown % |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3053 | 71.86 | 1.055 | 7.65 |
| 32 | 2982 | 72.50 | 1.075 | 7.88 |
| 64 | 2915 | 72.52 | 1.096 | 8.09 |
| 128 | 2738 | 74.18 | 1.173 | 8.85 |
| 256 | 2045 | 78.34 | 1.458 | 11.67 |
| 512 | 1321 | 85.84 | 1.814 | 15.86 |
| 1024 | 251 | 96.41 | 2.677 | 25.75 |

### Decode Strongest Positive Examples

| Model | Rate | Batch | Sum len | Spread low->high | Runtime low->high ms | Slowdown % |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| llama_7B | 2.0 | 3 | 1841 | 480 -> 890 | 16.82 -> 216.97 | 1190.28 |
| llama_7B | 1.0 | 7 | 6391 | 612 -> 1626 | 18.59 -> 208.12 | 1019.70 |
| llama_1B | 1.0 | 2 | 2307 | 39 -> 1571 | 10.50 -> 81.50 | 675.98 |
| llama_7B | 1.0 | 4 | 4418 | 243 -> 724 | 17.13 -> 103.86 | 506.24 |
| llama_7B | 1.0 | 7 | 6398 | 612 -> 1626 | 18.96 -> 110.22 | 481.31 |
| llama_7B | 1.5 | 8 | 8059 | 811 -> 1921 | 19.66 -> 114.01 | 479.80 |
| llama_7B | 1.0 | 4 | 3593 | 724 -> 1101 | 18.43 -> 85.48 | 363.87 |
| llama_7B | 1.0 | 5 | 4519 | 970 -> 977 | 19.21 -> 71.37 | 271.43 |

### Decode Strongest Negative Examples

| Model | Rate | Batch | Sum len | Spread low->high | Runtime low->high ms | Slowdown % |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| llama_1B | 1.0 | 4 | 4095 | 1033 -> 1441 | 171.00 -> 13.23 | -92.27 |
| llama_7B | 1.0 | 8 | 6932 | 1229 -> 1238 | 244.90 -> 20.52 | -91.62 |
| llama_7B | 1.5 | 2 | 1463 | 169 -> 257 | 170.61 -> 15.60 | -90.85 |
| llama_7B | 1.0 | 8 | 6924 | 1229 -> 1238 | 210.37 -> 19.52 | -90.72 |
| llama_1B | 1.0 | 4 | 2998 | 651 -> 769 | 108.82 -> 10.47 | -90.38 |
| llama_7B | 1.0 | 8 | 6940 | 1229 -> 1238 | 211.05 -> 20.75 | -90.17 |
| llama_7B | 1.0 | 5 | 5156 | 927 -> 970 | 179.76 -> 17.81 | -90.09 |
| llama_7B | 1.5 | 8 | 7189 | 1063 -> 1625 | 201.42 -> 20.38 | -89.88 |

## Prefill Near-Match Addendum

Near-match rule: same `(model, rate, batch_size)` and total prompt length difference <= 32 tokens.

| Metric | Value |
| --- | ---: |
| near-matched pairs | 7 |
| high-spread slower pairs % | 85.71 |
| median delta ms | 83.027 |
| mean delta ms | 75.558 |
| median slowdown % | 77.97 |
| mean slowdown % | 62.68 |

Because this section only has a handful of pairs, it is not strong enough to carry the main argument on its own.

| Model | Rate | Batch | Sum len low->high | Spread low->high | Runtime low->high ms | Slowdown % |
| --- | ---: | ---: | --- | --- | --- | ---: |
| llama_3B | 3.5 | 2 | 1338 -> 1366 | 378 -> 972 | 189.19 -> 324.30 | 71.41 |
| llama_3B | 3.5 | 3 | 2135 -> 2106 | 585 -> 743 | 241.26 -> 316.33 | 31.12 |
| llama_3B | 4.0 | 2 | 767 -> 747 | 117 -> 727 | 70.30 -> 142.78 | 103.09 |
| llama_8B | 3.5 | 2 | 767 -> 747 | 117 -> 727 | 106.49 -> 189.52 | 77.97 |
| llama_8B | 3.0 | 2 | 767 -> 747 | 117 -> 727 | 106.84 -> 191.09 | 78.86 |
| llama_8B | 4.0 | 2 | 767 -> 747 | 117 -> 727 | 106.36 -> 189.74 | 78.40 |
| llama_8B | 4.0 | 2 | 975 -> 972 | 541 -> 608 | 210.81 -> 206.39 | -2.10 |

## Interpretation

For decode, the same total context tokens do not imply the same runtime.
When the spread difference is large, the extreme distribution is more often slower, not just different.
That is direct evidence that total-work terms alone miss a real within-batch critical-path effect.

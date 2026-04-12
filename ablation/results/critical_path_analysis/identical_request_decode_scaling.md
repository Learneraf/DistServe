# Identical-Request Decode Scaling

For the current 4-term decode model:

`T = a0 + a1 * batch_size + a2 * sum_context_len + a3 * max_context_len`

if all requests in the batch have the same current context length `L`, then:

`sum_context_len = batch_size * L`

`max_context_len = L`

so the model becomes:

`T(b, L) = a0 + a1 * b + a2 * b * L + a3 * L`

This is not purely serial (`~ b * L`) and not purely parallel (`~ L`).
It is a mixture of:

- a pressure term that scales with total work: `a2 * b * L`
- a critical-path term that does not scale with batch size: `a3 * L`

Below are the predicted decode times from the current small-batch fit in
`fit_params_live_decode.json`, normalized against `batch_size = 1`.

## llama_1B

| Context L | bs=1 | bs=2 | bs=4 | bs=8 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1.000 | 1.020 | 1.059 | 1.138 |
| 1024 | 1.000 | 1.018 | 1.054 | 1.125 |
| 2048 | 1.000 | 1.016 | 1.047 | 1.110 |

## llama_3B

| Context L | bs=1 | bs=2 | bs=4 | bs=8 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1.000 | 1.009 | 1.026 | 1.060 |
| 1024 | 1.000 | 1.018 | 1.054 | 1.127 |
| 2048 | 1.000 | 1.028 | 1.085 | 1.197 |

## llama_7B

| Context L | bs=1 | bs=2 | bs=4 | bs=8 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1.000 | 1.023 | 1.070 | 1.163 |
| 1024 | 1.000 | 1.032 | 1.096 | 1.223 |
| 2048 | 1.000 | 1.047 | 1.142 | 1.330 |

## llama_8B

| Context L | bs=1 | bs=2 | bs=4 | bs=8 |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1.000 | 1.017 | 1.051 | 1.118 |
| 1024 | 1.000 | 1.020 | 1.061 | 1.142 |
| 2048 | 1.000 | 1.025 | 1.074 | 1.174 |

These ratios are far from linear-in-batch-size growth.
For example, at `L = 1024`, moving from `bs=1` to `bs=8` gives:

- `llama_1B`: `1.125x`
- `llama_3B`: `1.127x`
- `llama_7B`: `1.223x`
- `llama_8B`: `1.142x`

So the current `sum + max` structure already behaves like:

- critical path + shared pressure

rather than:

- pure sum of all request workloads.

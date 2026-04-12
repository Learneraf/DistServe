# DistServe CUDA Completion Pathology

This analysis focuses on two non-compute indicators:

- `post_last_token_ms = end_time - token_timestamps[-1]`
- observed request-start gaps compared with the seeded Poisson schedule used by the benchmark

If these values spike while server-side decode does not, the anomaly is more consistent with client/response-path delay than with model compute.

## llama_1B

| Rate | total p99 ms | decode p99 ms | post-last-token p99 ms | post-last-token max ms | observed max start gap ms | expected max gap ms | max concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13848.56 | 13709.44 | 5.98 | 42.22 | 4456.90 | 4454.50 | 7 |
| 1.5 | 10449.95 | 10310.28 | 5.32 | 7.17 | 2971.95 | 2969.66 | 9 |
| 2 | 10523.07 | 10388.91 | 5.70 | 5.97 | 2228.41 | 2227.25 | 11 |
| 2.5 | 10748.85 | 10608.44 | 5.14 | 6.36 | 1783.13 | 1781.80 | 13 |
| 3 | 10829.13 | 10744.66 | 4.83 | 5.14 | 1485.88 | 1484.83 | 14 |
| 3.5 | 11145.90 | 11061.66 | 4.71 | 4.84 | 1273.99 | 1272.71 | 16 |
| 4 | 12429.26 | 12369.42 | 5.64 | 5.86 | 1114.98 | 1113.62 | 20 |

## llama_3B

| Rate | total p99 ms | decode p99 ms | post-last-token p99 ms | post-last-token max ms | observed max start gap ms | expected max gap ms | max concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23122.64 | 22892.11 | 6.49 | 8.90 | 4455.82 | 4454.50 | 11 |
| 1.5 | 23933.26 | 23722.39 | 6.56 | 8.79 | 2971.42 | 2969.66 | 15 |
| 2 | 25810.87 | 25698.81 | 6.88 | 12.34 | 2227.92 | 2227.25 | 21 |
| 2.5 | 26984.83 | 26875.24 | 6.02 | 6.59 | 1784.12 | 1781.80 | 25 |
| 3 | 27890.15 | 27791.64 | 6.29 | 6.34 | 1486.49 | 1484.83 | 30 |
| 3.5 | 29736.81 | 29506.80 | 6.00 | 8.74 | 1273.87 | 1272.71 | 33 |
| 4 | 30551.04 | 30319.46 | 6.90 | 42.00 | 1115.26 | 1113.62 | 37 |

## llama_7B

| Rate | total p99 ms | decode p99 ms | post-last-token p99 ms | post-last-token max ms | observed max start gap ms | expected max gap ms | max concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 23990.64 | 23827.19 | 10.24 | 10.97 | 4456.06 | 4454.50 | 13 |
| 1.5 | 27265.00 | 27100.06 | 10.55 | 12.21 | 2970.38 | 2969.66 | 16 |
| 2 | 46779.20 | 46534.10 | 8.65 | 8.92 | 2229.75 | 2227.25 | 32 |
| 2.5 | 36584.67 | 36522.54 | 7.11 | 7.12 | 1784.27 | 1781.80 | 30 |
| 3 | 43864.82 | 43488.82 | 7.68 | 7.83 | 1486.38 | 1484.83 | 43 |
| 3.5 | 115730.78 | 24421.24 | 101435.16 | 101752.67 | 102001.64 | 1272.71 | 26 |
| 4 | 39903.90 | 39582.93 | 11.52 | 15.33 | 1119.28 | 1113.62 | 44 |

## llama_8B

| Rate | total p99 ms | decode p99 ms | post-last-token p99 ms | post-last-token max ms | observed max start gap ms | expected max gap ms | max concurrency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32841.10 | 32458.52 | 6.07 | 6.63 | 4459.38 | 4454.50 | 14 |
| 1.5 | 35283.33 | 35144.97 | 5.30 | 6.19 | 2971.06 | 2969.66 | 21 |
| 2 | 39758.59 | 39629.04 | 5.79 | 7.28 | 2228.26 | 2227.25 | 29 |
| 2.5 | 41129.73 | 40859.72 | 5.69 | 7.14 | 1783.87 | 1781.80 | 33 |
| 3 | 43125.22 | 42856.27 | 5.96 | 6.52 | 1486.36 | 1484.83 | 40 |
| 3.5 | 44256.00 | 43848.11 | 6.52 | 8.87 | 1274.60 | 1272.71 | 43 |
| 4 | 45600.62 | 45029.89 | 6.78 | 10.60 | 1114.51 | 1113.62 | 47 |


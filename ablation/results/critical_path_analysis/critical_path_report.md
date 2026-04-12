# Critical Path Term Analysis

## Prefill

### all

- sample count: 2426

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 16.67 | 24.49 | 93.44 |
| pressure_plus_max | 15.29 | 23.18 | 93.37 |
| pressure_plus_mid | 15.70 | 23.59 | 128.10 |
| pressure_plus_min | 15.99 | 23.99 | 178.64 |

### multi_request

- sample count: 208

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 15.62 | 22.48 | 146.11 |
| pressure_plus_max | 12.00 | 26.92 | 309.20 |
| pressure_plus_mid | 14.85 | 19.86 | 84.34 |
| pressure_plus_min | 14.78 | 20.12 | 83.14 |

### heterogeneous_multi

- sample count: 208

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 15.62 | 22.48 | 146.11 |
| pressure_plus_max | 12.00 | 26.92 | 309.20 |
| pressure_plus_mid | 14.85 | 19.86 | 84.34 |
| pressure_plus_min | 14.78 | 20.12 | 83.14 |

### high_spread_multi

- sample count: 106

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 13.64 | 19.35 | 79.98 |
| pressure_plus_max | 10.03 | 16.29 | 81.86 |
| pressure_plus_mid | 13.81 | 19.66 | 80.59 |
| pressure_plus_min | 14.14 | 19.63 | 79.94 |

## Decode

### all_small

- sample count: 68140

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 11.94 | 19.55 | 1480.96 |
| critical_only | 9.30 | 17.91 | 1395.70 |
| pressure_plus_max | 9.22 | 17.93 | 1445.85 |
| pressure_plus_mid | 11.62 | 19.30 | 1461.68 |
| pressure_plus_min | 11.90 | 19.57 | 1473.04 |

### multi_request_small

- sample count: 59382

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 11.38 | 19.18 | 1475.73 |
| critical_only | 9.26 | 18.05 | 1410.24 |
| pressure_plus_max | 9.19 | 18.08 | 1453.21 |
| pressure_plus_mid | 11.38 | 19.21 | 1467.34 |
| pressure_plus_min | 11.26 | 19.13 | 1463.56 |

### heterogeneous_small

- sample count: 59382

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 11.38 | 19.18 | 1475.73 |
| critical_only | 9.26 | 18.05 | 1410.24 |
| pressure_plus_max | 9.19 | 18.08 | 1453.21 |
| pressure_plus_mid | 11.38 | 19.21 | 1467.34 |
| pressure_plus_min | 11.26 | 19.13 | 1463.56 |

### high_spread_small

- sample count: 29716

| Variant | CV mean abs % | CV RMSE % | CV max abs % |
| --- | ---: | ---: | ---: |
| pressure_only | 11.52 | 20.72 | 1455.37 |
| critical_only | 10.23 | 20.08 | 1423.88 |
| pressure_plus_max | 10.25 | 20.18 | 1465.56 |
| pressure_plus_mid | 11.44 | 20.70 | 1459.44 |
| pressure_plus_min | 11.44 | 20.65 | 1444.14 |


# Drift batch blame -- stream audit

Pooled drift score: 0.0934 (max feature ratio 3.64 x null, restored: False)

## Per-feature drift (pooled)

| feature | statistic | null | ratio |
|---|---|---|---|
| age | 0.0187 | 0.0224 | 0.83 |
| systolic_bp | 0.0120 | 0.0224 | 0.54 |
| glucose_mg_dl | 0.0757 | 0.0224 | 3.38 |
| creatinine | 0.0267 | 0.0224 | 1.19 |
| bmi | 0.0337 | 0.0224 | 1.50 |
| device_model | 0.0095 | 0.0143 | 0.66 |
| readmitted | 0.0339 | 0.0093 | 3.64 |

## Batch blame

| batch | source | rows | LOBO blame | Shapley blame |
|---|---|---|---|---|
| b04 | device-hub | 300 | 0.0468 | 0.1872 |
| b08 | south-feed | 300 | 0.0158 | 0.0746 |
| b01 | north-feed | 300 | -0.0002 | -0.0029 |
| b09 | partner-api | 300 | -0.0046 | -0.0241 |
| b07 | partner-api | 300 | -0.0057 | -0.0098 |
| b02 | south-feed | 300 | -0.0065 | -0.0096 |
| b06 | device-hub | 300 | -0.0094 | -0.0300 |
| b10 | north-feed | 300 | -0.0105 | -0.0281 |
| b05 | north-feed | 300 | -0.0109 | -0.0230 |
| b03 | partner-api | 300 | -0.0131 | -0.0411 |

## Minimal blame set (greedy removal)

Removed in order: b04, b08

After removal: score 0.0150, max ratio 1.60, restored: True

Planted culprits (ground truth): b04, b08

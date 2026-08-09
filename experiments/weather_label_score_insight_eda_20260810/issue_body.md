# Weather-label score insight EDA: residual, stability, direction, FiCR, and 2025 exposure

이 이슈는 #31의 `5. Weather-Label Correlations` 후속 분석입니다. #31에서는 LDAPS `wind50max_speed`, GFS `wind850_speed`가 label ratio와 강하게 연결된다는 것을 확인했습니다. 이번 분석의 목적은 한 단계 더 좁혀서 **그 wind signal이 현재 best anchor의 residual과 score surface까지 설명하는지**를 검증하는 것입니다.

기준 anchor는 open issue 기준 current best인 `phase154_lower16`을 W4에서 재구성한 `phase98_w4_phase60_lineage + Lower16`입니다.

Artifacts:

- script: `experiments/weather_label_score_insight_eda_20260810/run_weather_label_score_insight_eda.py`
- results: `experiments/weather_label_score_insight_eda_20260810/results/`
- figures: `experiments/weather_label_score_insight_eda_20260810/results/figures/`

---

## 1. Best-anchor residual vs top wind bins

![Residual bias wind bins](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/01_residual_bias_wind_bins.png)

### What this tests

단순 label correlation이 아니라, top wind feature가 **현재 best anchor의 residual**까지 설명하는지 봅니다. LDAPS grid13 `wind50max_speed`를 decile bin으로 나누고, eligible row에서 signed error, absolute error, fail8 rate를 계산했습니다.

### Key table

| target      |   wind_mid |   mean_abs_error_ratio |   mean_signed_error_ratio |   fail8_rate |
|:------------|-----------:|-----------------------:|--------------------------:|-------------:|
| kpx_group_1 |  10.001658 |               0.144418 |                  0.033231 |     0.684426 |
| kpx_group_2 |  10.004524 |               0.159045 |                  0.044670 |     0.705521 |
| kpx_group_3 |  19.813607 |               0.155445 |                 -0.050808 |     0.723577 |

### Interpretation

- wind bin에 따라 residual magnitude와 sign이 달라집니다. 따라서 wind feature는 label만 설명하는 것이 아니라 current best의 error surface에도 일부 연결됩니다.
- 그러나 signed error가 모든 high-wind 구간에서 한 방향으로 고정되지는 않습니다. 즉 naive high-wind up/down correction은 위험합니다.
- public probe 실패와 일관되게, 큰 overlay보다 **bin-local, group-local, FiCR-aware correction**만 후보가 될 수 있습니다.

### Validity discussion

이 분석은 current best anchor의 실제 W4 residual을 사용하므로 score 개선과 직접 연결됩니다. 다만 W4 단일 split이므로 이 패턴이 2025 public에 그대로 유지된다고 가정하면 안 됩니다.

---

## 2. Monthly/seasonal correlation stability

![Monthly correlation stability](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/02_monthly_correlation_stability.png)

### What this tests

전체 기간 Spearman correlation 상위 feature가 월별/연도별로 안정적인지 검증합니다. `LDAPS grid13 wind50max`와 `GFS grid1 wind850`을 2022-2024 month 단위로 분해했습니다.

### Key table

| target      |   ldaps_min |   ldaps_median |   gfs_min |   gfs_median |
|:------------|------------:|---------------:|----------:|-------------:|
| kpx_group_1 |    0.558282 |       0.764652 |  0.547925 |     0.756257 |
| kpx_group_2 |    0.561365 |       0.765144 |  0.588217 |     0.770259 |
| kpx_group_3 |    0.589220 |       0.770475 |  0.577555 |     0.767256 |

### Interpretation

- median correlation은 높지만, month별 minimum은 훨씬 낮습니다. 즉 “전체 기간 top feature”는 안정적인 global rule이 아닙니다.
- month gate 없이 feature를 강하게 적용하면 특정 월에서 local gain이 public loss로 바뀔 수 있습니다.
- 제출 후보는 월별 correlation stability와 2025 exposure를 같이 통과해야 합니다.

### Validity discussion

Spearman correlation은 monotonic relation만 보는 단변량 진단입니다. label relevance를 빠르게 확인하는 데는 타당하지만, collinearity와 model residual 설명력은 따로 봐야 합니다. 그래서 이 분석은 후보 feature를 채택하는 증거가 아니라 **월별 gate/risk를 정하는 증거**로만 사용해야 합니다.

---

## 3. Wind direction-conditioned grid

![Direction conditioned grid](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/03_direction_conditioned_grid_corr.png)

### What this tests

전체 top grid 하나가 모든 풍향에서 유효한지 검증합니다. LDAPS `wind50max_speed` correlation을 direction sector별로 다시 계산해 sector별 top grid를 비교했습니다.

### Key table

| target      |   unique_top_grids |
|:------------|-------------------:|
| kpx_group_1 |                  6 |
| kpx_group_2 |                  4 |
| kpx_group_3 |                  6 |

### Interpretation

- target별 top grid가 direction sector에 따라 바뀝니다. 이는 fixed top-grid feature보다 direction-conditioned spatial aggregation이 더 타당하다는 뜻입니다.
- 특히 group3도 단순히 data-poor group이 아니라 다른 spatial/wind-response structure를 가질 가능성이 큽니다.

### Validity discussion

direction sector별 sample 수가 작아지는 구간이 있으므로 selection noise가 있습니다. 따라서 hard sector top-grid 하나를 그대로 모델에 넣기보다, nearest/upstream weighting 또는 sector-smoothed feature로 구현해야 합니다.

---

## 4. LDAPS-GFS disagreement and residual risk

![Disagreement residual risk](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/04_disagreement_residual_risk.png)

### What this tests

LDAPS local wind와 GFS large-scale wind가 불일치할 때 current best error가 커지는지 봅니다. `LDAPS grid13 wind50max - GFS grid1 wind850`을 binning하고 abs error/fail8 rate를 집계했습니다.

### Key table

| target      |   max_fail8 |   min_fail8 |   max_abs_error |
|:------------|------------:|------------:|----------------:|
| kpx_group_1 |    0.580165 |    0.473934 |        0.131425 |
| kpx_group_2 |    0.618812 |    0.466882 |        0.160622 |
| kpx_group_3 |    0.684818 |    0.532228 |        0.142789 |

### Interpretation

- disagreement regime별 fail8/abs error가 달라집니다. 이는 NWP source disagreement가 uncertainty/risk proxy가 될 수 있음을 시사합니다.
- 다만 disagreement 방향 하나만으로 보정 방향을 정하기엔 충분하지 않습니다. correction feature라기보다 **risk gate 또는 shrink gate**로 쓰는 쪽이 더 안전합니다.

### Validity discussion

두 source는 vertical level과 grid scale이 다르므로 절대 차이의 물리 단위가 완전히 동일하지 않습니다. 따라서 이 분석은 “바람 예보 불일치가 있다”는 risk proxy로 해석해야 하며, 직접적인 physical correction 값으로 해석하면 안 됩니다.

---

## 5. Group-specific wind-to-power curve

![Group power curve](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/05_group_power_curve_comparison.png)

### What this tests

같은 wind bin에서 group1/2/3의 actual/capacity response가 같은지 비교합니다. group3가 단순 data-poor group인지, 별도 power curve group인지 확인하는 분석입니다.

### Key table

| target      |   high_wind_mean_ratio |
|:------------|-----------------------:|
| kpx_group_1 |               0.702480 |
| kpx_group_2 |               0.749877 |
| kpx_group_3 |               0.674207 |

### Interpretation

- group별 wind-to-power curve가 동일하지 않습니다. 같은 wind speed에서도 평균 발전 비율과 분산이 다르게 나타납니다.
- group3는 top wind-label correlation이 강하지만, group1/2 정책을 그대로 전이할 대상은 아닙니다.
- score 개선 후보는 group-shared feature보다 group-specific calibration 또는 group3 specialist로 가야 합니다.

### Validity discussion

이 curve는 observed label과 single grid wind로 만든 empirical curve입니다. curtailment, outage, turbine availability를 분리하지 못하므로 “물리 power curve”로 부르면 안 됩니다. 모델링에서는 operating-regime curve로 해석해야 합니다.

---

## 6. FiCR-sensitive rows by wind regime

![FiCR boundary by wind](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/06_ficr_boundary_by_wind.png)

### What this tests

score가 nMAE만으로 결정되지 않기 때문에, wind bin별로 abs error 6-10% boundary row, fail8 row, actual 10-16% support-edge row가 어디에 집중되는지 봅니다.

### Key table

| target      |   max_boundary_rate |   max_fail8_rate |   max_near_support |
|:------------|--------------------:|-----------------:|-------------------:|
| kpx_group_1 |            0.206250 |         0.684426 |           0.472696 |
| kpx_group_2 |            0.202083 |         0.705521 |           0.419469 |
| kpx_group_3 |            0.198582 |         0.728953 |           0.518033 |

### Interpretation

- FiCR-sensitive row 비중은 wind regime별로 다릅니다. 즉 wind feature가 score-relevant하려면 residual뿐 아니라 FiCR boundary row를 설명해야 합니다.
- correction은 abs error 평균을 줄이는 것만으로 충분하지 않습니다. fail-to-pass8보다 pass-to-fail8을 더 많이 만들면 public probe처럼 total score가 하락합니다.
- 따라서 앞으로의 postprocessing은 `wind regime x FiCR margin x group` 단위의 pass/fail audit이 필요합니다.

### Validity discussion

FiCR label은 W4 actual이 있어야 계산할 수 있으므로 public에는 직접 적용할 수 없습니다. 하지만 validation에서 correction 후보를 폐기하는 guardrail로는 매우 타당합니다.

---

## 7. 2025 label-relevant wind exposure

![2025 wind exposure](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/weather_label_score_insight_eda_20260810/results/figures/07_2025_wind_exposure_risk.png)

### What this tests

2025 test가 top wind feature 기준으로 2024 validation과 얼마나 다른지, label relevance로 가중한 exposure risk를 봅니다.

### Key table

| source   | feature         |   month |   train2024_mean |   test2025_mean |   mean_delta |   pooled_std |   z_delta_vs_2024 |   test_above_train_p90_rate |   spearman_corr |   weighted_abs_shift |   weighted_shift_risk |
|:---------|:----------------|--------:|-----------------:|----------------:|-------------:|-------------:|------------------:|----------------------------:|----------------:|---------------------:|----------------------:|
| gfs      | wind850_speed   |       4 |         5.716444 |       10.727530 |     5.011086 |     4.075546 |          1.229549 |                    0.417593 |        0.812383 |             0.998865 |              0.998865 |
| ldaps    | wind50max_speed |       4 |         6.022475 |        8.978159 |     2.955684 |     3.054599 |          0.967618 |                    0.414551 |        0.826312 |             0.799554 |              0.799554 |
| ldaps    | wind50max_speed |       2 |         6.915823 |       10.458182 |     3.542358 |     3.956725 |          0.895275 |                    0.325614 |        0.826312 |             0.739777 |              0.739777 |
| gfs      | wind850_speed   |       2 |         6.982975 |       11.212619 |     4.229644 |     5.365881 |          0.788248 |                    0.271164 |        0.812383 |             0.640359 |              0.640359 |
| gfs      | wind850_speed   |       7 |        11.724161 |        6.525402 |    -5.198759 |     7.168182 |         -0.725255 |                    0.000000 |        0.812383 |             0.589185 |              0.589185 |
| ldaps    | wind50max_speed |       7 |        10.190609 |        6.883743 |    -3.306866 |     5.285491 |         -0.625650 |                    0.002524 |        0.826312 |             0.516982 |              0.516982 |
| ldaps    | wind50max_speed |       8 |         5.009896 |        6.958957 |     1.949061 |     3.198070 |          0.609449 |                    0.270245 |        0.826312 |             0.503595 |              0.503595 |
| ldaps    | wind50max_speed |       9 |         4.779808 |        6.048366 |     1.268559 |     2.290282 |          0.553888 |                    0.214062 |        0.826312 |             0.457684 |              0.457684 |
| gfs      | wind850_speed   |       6 |         5.657445 |        8.106788 |     2.449343 |     4.397886 |          0.556936 |                    0.226235 |        0.812383 |             0.452446 |              0.452446 |
| gfs      | wind850_speed   |       9 |         4.543866 |        6.070527 |     1.526662 |     3.188391 |          0.478819 |                    0.205093 |        0.812383 |             0.388984 |              0.388984 |

### Interpretation

- 2025 exposure risk는 월별로 매우 다릅니다. 특히 특정 wind feature/month에서 shift risk가 커집니다.
- W4에서 좋아 보인 month-gated overlay가 public에서 실패한 이유를 설명합니다. local score-positive slice가 public exposure에서는 같은 의미가 아닐 수 있습니다.

### Validity discussion

2025 label은 없으므로 exposure는 weather-only shift입니다. 따라서 score를 직접 예측하지는 못하지만, public submission 후보의 위험 월을 사전에 표시하는 데는 필요합니다.

---

## Final conclusion

이번 EDA로 증명된 것:

1. top wind feature는 label뿐 아니라 current best residual과도 일부 연결됩니다.
2. 하지만 그 연결은 월별, 풍향별, group별로 불안정합니다.
3. fixed top grid 또는 high-alpha overlay는 위험합니다.
4. group3는 별도 wind-to-power/operating-regime을 가진 group으로 다뤄야 합니다.
5. score 개선은 wind feature 자체보다 **wind-conditioned residual + FiCR transition + 2025 exposure risk**를 동시에 통과해야 합니다.

이번 EDA로 반증된 것:

- “전체 기간 상관이 높은 wind feature를 추가하면 score가 오른다.”
- “LDAPS grid13 같은 global top grid 하나를 고정하면 충분하다.”
- “group3는 label 부족만 해결하면 group1/2 정책을 그대로 전이할 수 있다.”
- “local W4 FiCR gain은 public에서도 안전하다.”

## Recommended next experiments

1. W3/W4 모두에서 `wind bin x group x FiCR margin` correction 후보를 만들고, 월별 손실이 있는 후보는 폐기.
2. direction-conditioned grid는 hard top-grid가 아니라 soft upstream weighting으로 구현.
3. LDAPS-GFS disagreement는 correction이 아니라 uncertainty shrink/gate로 사용.
4. group3는 shared model postprocessing이 아니라 specialist calibration으로 분리.
5. public submission 후보는 반드시 2025 exposure-risk month audit을 통과한 것만 제출.

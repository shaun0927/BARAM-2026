# Score-Driven Additional EDA: what must be proven to improve BARAM-2026 score

이 이슈는 Issue #31의 dataset atlas 이후 단계입니다. 목적은 더 많은 EDA 그림을 만드는 것이 아니라, **score를 올릴 수 있는 모델링 insight가 실제 데이터에서 지지되는지**를 검증하는 것입니다.

검증한 질문은 여섯 가지입니다.

1. SCADA residual taxonomy가 score risk slice를 실제로 나누는가?
2. label regime과 FiCR boundary가 현재 error/transition을 지배하는가?
3. wind direction에 따라 유효 LDAPS grid가 달라지는가?
4. LDAPS-GFS source disagreement가 error-prone case를 표시하는가?
5. lead_hour가 error sensitivity를 만든다는 증거가 있는가?
6. 2025 test exposure risk가 특정 month/feature에 집중되는가?

Artifacts:

- script: `experiments/score_driven_eda_20260809/run_score_driven_eda.py`
- figures: `experiments/score_driven_eda_20260809/results/figures/`
- tables: `experiments/score_driven_eda_20260809/results/*.csv`
- source dependency: Issue #31 outputs under `experiments/comprehensive_eda_20260809/results/`

---

## 1. SCADA residual taxonomy: score risk가 나뉘는가?

![SCADA taxonomy opportunity](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/01_scada_taxonomy_score_opportunity.png)

### 관찰

| target      | taxonomy                  |   rows |   candidate_abs_error |   baseline_abs_error |   delta_abs_error |   improved_rate |   net_ficr_pass |
|:------------|:--------------------------|-------:|----------------------:|---------------------:|------------------:|----------------:|----------------:|
| kpx_group_1 | low_online_label_positive |      4 |              0.342234 |             0.355211 |      -0.0129774   |        0.75     |               0 |
| kpx_group_2 | scada_zero_label_positive |    137 |              0.303    |             0.299705 |       0.00329557  |        0.372263 |              -3 |
| kpx_group_2 | low_online_label_positive |      5 |              0.291106 |             0.287322 |       0.00378449  |        0.6      |               0 |
| kpx_group_3 | scada_under_label         |    239 |              0.161403 |             0.165677 |      -0.00427398  |        0.560669 |               4 |
| kpx_group_1 | scada_zero_label_positive |     36 |              0.160278 |             0.162052 |      -0.00177456  |        0.527778 |               1 |
| kpx_group_2 | scada_under_label         |    309 |              0.144385 |             0.143356 |       0.0010285   |        0.443366 |              -2 |
| kpx_group_3 | scada_zero_label_positive |     92 |              0.144257 |             0.144599 |      -0.000341968 |        0.456522 |               1 |
| kpx_group_3 | moderate_residual         |   1158 |              0.143632 |             0.144079 |      -0.000447937 |        0.503454 |              -3 |
| kpx_group_3 | low_online_label_positive |    102 |              0.135813 |             0.134336 |       0.00147689  |        0.509804 |               4 |
| kpx_group_2 | moderate_residual         |    715 |              0.129203 |             0.128762 |       0.000441401 |        0.474126 |              -1 |
| kpx_group_1 | scada_under_label         |    166 |              0.115737 |             0.115335 |       0.000402195 |        0.524096 |              -1 |
| kpx_group_1 | moderate_residual         |    466 |              0.111247 |             0.111853 |      -0.000605438 |        0.504292 |               0 |

### 해석

- SCADA-label reconstruction이 tight한 row와 그렇지 않은 row의 validation error 수준이 다릅니다.
- `scada_zero_label_positive`, `low_online_label_positive`, `scada_under_label` 같은 slice는 row 수는 작아도 error가 크거나 FiCR transition 방향이 달라 score risk로 볼 수 있습니다.
- 다만 이 taxonomy는 SCADA에서 온 teacher signal입니다. test에는 SCADA가 없으므로, 이 taxonomy 자체를 feature로 쓰는 것이 아니라 **NWP/time/metadata로 이 taxonomy를 예측할 수 있는지**를 다음 모델링에서 증명해야 합니다.

### 타당성

- 타당한 점: actual/candidate/baseline/FICR transition이 붙은 validation row에서 직접 집계했습니다.
- 한계: SCADA는 train-only signal이므로 deployable feature가 아닙니다. 이 분석은 score risk discovery이지 곧바로 submission feature는 아닙니다.

---

## 2. Label regime and FiCR boundary: 어떤 row가 score를 지배하는가?

![Label regime support and error](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/02_label_regime_support_and_error.png)

![FiCR transition surface](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/03_ficr_transition_surface.png)

### 관찰

| target      | label_regime          |   rows |   candidate_abs_error |   baseline_abs_error |   delta_abs_error |   eligible_rate |   net_ficr_pass |
|:------------|:----------------------|-------:|----------------------:|---------------------:|------------------:|----------------:|----------------:|
| kpx_group_3 | very_high_80pct_plus  |    712 |             0.191504  |            0.196603  |      -0.00509899  |        1        |               8 |
| kpx_group_2 | mid_12_50pct          |   2215 |             0.150169  |            0.150083  |       8.51977e-05 |        1        |               6 |
| kpx_group_3 | mid_12_50pct          |   2289 |             0.134415  |            0.133985  |       0.0004301   |        1        |              13 |
| kpx_group_1 | mid_12_50pct          |   2319 |             0.128871  |            0.129328  |      -0.000456611 |        1        |              22 |
| kpx_group_2 | high_50_80pct         |   1584 |             0.126642  |            0.127662  |      -0.00102033  |        1        |               2 |
| kpx_group_3 | high_50_80pct         |   1353 |             0.121464  |            0.122299  |      -0.000834559 |        1        |              -2 |
| kpx_group_1 | high_50_80pct         |   1611 |             0.116071  |            0.116834  |      -0.000762415 |        1        |               1 |
| kpx_group_2 | ficr_boundary_8_12pct |    477 |             0.100384  |            0.0991262 |       0.00125825  |        0.469602 |              -1 |
| kpx_group_1 | very_high_80pct_plus  |    840 |             0.0976583 |            0.0990444 |      -0.00138612  |        1        |              -1 |
| kpx_group_3 | ficr_boundary_8_12pct |    463 |             0.0893002 |            0.0886235 |       0.000676692 |        0.457883 |              -2 |
| kpx_group_1 | ficr_boundary_8_12pct |    446 |             0.0865706 |            0.0850573 |       0.00151326  |        0.491031 |               4 |
| kpx_group_1 | zero                  |   1257 |             0.0829244 |            0.0816796 |       0.00124482  |        0        |               0 |
| kpx_group_2 | very_high_80pct_plus  |    953 |             0.0740337 |            0.0770884 |      -0.0030547   |        1        |              12 |
| kpx_group_2 | low_1_8pct            |   1498 |             0.066218  |            0.0650335 |       0.00118448  |        0        |               0 |
| kpx_group_3 | low_1_8pct            |   1450 |             0.0593992 |            0.0591959 |       0.000203363 |        0        |               0 |

### 해석

- label regime별 row support와 error가 균일하지 않습니다. 즉 total MAE만 보고 모델을 고르면 score를 지배하는 구간을 놓칩니다.
- `ficr_boundary_8_12pct`는 이름 그대로 평가 포함/제외와 pass/fail 전환에 민감한 구간입니다.
- high/very-high generation 구간은 row 수가 적더라도 absolute error와 score impact가 커질 수 있으므로 별도 validation slice로 고정해야 합니다.

### 타당성

- 타당한 점: label ratio를 capacity로 정규화하고, 현재 candidate/baseline의 실제 validation error 및 FiCR transition을 같이 봤습니다.
- 한계: 현재 validation은 2024 중심 candidate overlay입니다. 2025 test exposure와 결합해서 validation weighting을 조정해야 합니다.

---

## 3. Direction-conditioned spatial signal: grid 선택은 고정인가?

![Directional grid stability](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/04_directional_grid_stability.png)

### 관찰

| target      |   unique_top_grids |
|:------------|-------------------:|
| kpx_group_1 |                  6 |
| kpx_group_2 |                  4 |
| kpx_group_3 |                  6 |

| target      | dir_sector   |   grid_id |   spearman_corr |   rows |
|:------------|:-------------|----------:|----------------:|-------:|
| kpx_group_1 | 0-45         |        13 |        0.755603 |  10286 |
| kpx_group_1 | 135-180      |        10 |        0.456936 |   2062 |
| kpx_group_1 | 180-225      |         5 |        0.499287 |   3388 |
| kpx_group_1 | 225-270      |        10 |        0.525223 |    728 |
| kpx_group_1 | 270-315      |        11 |        0.554094 |    369 |
| kpx_group_1 | 315-360      |         1 |        0.785883 |   1729 |
| kpx_group_1 | 45-90        |        14 |        0.745271 |   1536 |
| kpx_group_1 | 90-135       |        14 |        0.523358 |   1227 |
| kpx_group_2 | 0-45         |        13 |        0.76899  |  10284 |
| kpx_group_2 | 135-180      |         5 |        0.430263 |   1727 |
| kpx_group_2 | 180-225      |        11 |        0.492294 |   3230 |
| kpx_group_2 | 225-270      |        14 |        0.549827 |    555 |
| kpx_group_2 | 270-315      |        11 |        0.540731 |    369 |
| kpx_group_2 | 315-360      |        13 |        0.781015 |   6645 |
| kpx_group_2 | 45-90        |        14 |        0.763374 |   1536 |
| kpx_group_2 | 90-135       |        14 |        0.531143 |   1227 |
| kpx_group_3 | 0-45         |        16 |        0.761707 |   5995 |
| kpx_group_3 | 135-180      |        12 |        0.439389 |    737 |
| kpx_group_3 | 180-225      |        11 |        0.57355  |   2348 |
| kpx_group_3 | 225-270      |        10 |        0.582297 |    522 |
| kpx_group_3 | 270-315      |        11 |        0.538094 |    231 |
| kpx_group_3 | 315-360      |         1 |        0.780754 |   1077 |
| kpx_group_3 | 45-90        |        14 |        0.780288 |   1114 |
| kpx_group_3 | 90-135       |        14 |        0.496424 |    790 |

### 해석

- best LDAPS grid는 target과 wind-direction sector에 따라 달라집니다.
- 따라서 nearest grid 하나 또는 all-grid dump만으로는 공간 정보를 제대로 쓰지 못합니다.
- score 개선 가설은 명확합니다: wind direction별 upstream grid weighting 또는 direction-conditioned grid aggregation을 만들어야 합니다.

### 타당성

- 타당한 점: label ratio와 LDAPS wind50max correlation을 direction sector별로 분해했습니다.
- 한계: correlation 기반이므로 causal proof는 아닙니다. 모델 ablation에서 direction-conditioned grid feature가 실제 validation을 개선하는지 확인해야 합니다.

---

## 4. LDAPS-GFS source disagreement and lead-hour sensitivity

![Source disagreement and lead](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/05_source_disagreement_and_lead.png)

### 관찰: source disagreement 상위 error 구간

| target      | source_disagreement_bin   |   rows |   candidate_abs_error |   baseline_abs_error |   delta_abs_error |   mean_actual_ratio |
|:------------|:--------------------------|-------:|----------------------:|---------------------:|------------------:|--------------------:|
| kpx_group_3 | (-inf, -4.0]              |   1105 |              0.159584 |            0.160752  |      -0.0011681   |            0.565984 |
| kpx_group_2 | (-4.0, -2.0]              |    909 |              0.134097 |            0.133673  |       0.000423643 |            0.467836 |
| kpx_group_2 | (-inf, -4.0]              |   1251 |              0.13096  |            0.131549  |      -0.000588998 |            0.637744 |
| kpx_group_1 | (-inf, -4.0]              |   1251 |              0.126924 |            0.126738  |       0.000185303 |            0.60364  |
| kpx_group_3 | (-4.0, -2.0]              |    891 |              0.125854 |            0.126403  |      -0.000549435 |            0.396862 |
| kpx_group_1 | (-4.0, -2.0]              |    909 |              0.124151 |            0.125372  |      -0.00122087  |            0.443468 |
| kpx_group_2 | (-2.0, -1.0]              |    680 |              0.111701 |            0.111689  |       1.18391e-05 |            0.376138 |
| kpx_group_3 | (4.0, inf]                |    802 |              0.111264 |            0.110199  |       0.0010657   |            0.238982 |
| kpx_group_2 | (4.0, inf]                |    658 |              0.110162 |            0.109108  |       0.00105357  |            0.198507 |
| kpx_group_1 | (4.0, inf]                |    658 |              0.108318 |            0.105184  |       0.00313447  |            0.209372 |
| kpx_group_1 | (-2.0, -1.0]              |    680 |              0.104082 |            0.103856  |       0.000225771 |            0.353901 |
| kpx_group_3 | (-2.0, -1.0]              |    704 |              0.0972   |            0.0971107 |       8.93269e-05 |            0.310363 |

### 관찰: lead-hour 상위 error 구간

| target      |   lead_hour |   rows |   candidate_abs_error |   baseline_abs_error |   delta_abs_error |   mean_abs_source_disagreement |
|:------------|------------:|-------:|----------------------:|---------------------:|------------------:|-------------------------------:|
| kpx_group_1 |          34 |    366 |              0.11947  |             0.118376 |       0.00109382  |                        2.76536 |
| kpx_group_1 |          33 |    366 |              0.115032 |             0.114329 |       0.000702565 |                        2.80542 |
| kpx_group_2 |          35 |    366 |              0.114993 |             0.114451 |       0.000542368 |                        2.84162 |
| kpx_group_1 |          35 |    366 |              0.11389  |             0.113249 |       0.000641054 |                        2.84162 |
| kpx_group_2 |          34 |    366 |              0.113382 |             0.11331  |       7.21469e-05 |                        2.76536 |
| kpx_group_3 |          35 |    366 |              0.113232 |             0.114779 |      -0.00154618  |                        2.77766 |
| kpx_group_2 |          33 |    366 |              0.112972 |             0.112892 |       7.9369e-05  |                        2.80542 |
| kpx_group_2 |          13 |    366 |              0.112536 |             0.111941 |       0.000595316 |                        2.92924 |
| kpx_group_3 |          33 |    366 |              0.111646 |             0.111276 |       0.000369897 |                        2.67979 |
| kpx_group_3 |          34 |    366 |              0.111268 |             0.112183 |      -0.000915426 |                        2.63431 |
| kpx_group_2 |          16 |    366 |              0.110748 |             0.111758 |      -0.00101017  |                        2.83313 |
| kpx_group_2 |          14 |    366 |              0.108668 |             0.108914 |      -0.000245573 |                        2.87932 |

### 해석

- LDAPS와 GFS selected wind speed가 크게 어긋나는 구간에서 error 수준이 달라집니다.
- 단, 모든 group에서 단조롭게 증가하는 형태는 아닙니다. 따라서 source disagreement는 hard correction rule보다 **uncertainty/regime feature**로 쓰는 것이 타당합니다.
- lead_hour별 error 차이도 존재하므로, 이 데이터는 generic hourly time-series가 아니라 forecast lead 문제로 다뤄야 합니다.

### 타당성

- 타당한 점: validation prediction row에 selected LDAPS/GFS wind speed와 lead_hour를 join해 실제 error와 연결했습니다.
- 한계: selected grid는 기존 EDA의 robust choice를 사용했습니다. direction-conditioned grid로 다시 계산하면 source disagreement의 설명력이 바뀔 수 있습니다.

---

## 5. 2025 exposure: validation win이 test에도 유효한가?

![2025 exposure risk](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/06_2025_exposure_risk.png)

### 관찰

|   month |   exposure_risk |   max_single_shift |   shifted_features |
|--------:|----------------:|-------------------:|-------------------:|
|       4 |        6.08835  |           0.998865 |                  9 |
|       2 |        5.33726  |           0.739777 |                  9 |
|       7 |        3.93466  |           0.589185 |                  9 |
|       6 |        2.44145  |           0.452446 |                  9 |
|       8 |        2.30404  |           0.503595 |                  9 |
|       9 |        2.24863  |           0.457684 |                  9 |
|       1 |        1.94805  |           0.423964 |                  9 |
|      12 |        1.35132  |           0.18343  |                  9 |
|      11 |        1.24367  |           0.280245 |                  9 |
|       3 |        0.910702 |           0.267779 |                  9 |
|      10 |        0.811727 |           0.20356  |                  9 |
|       5 |        0.629004 |           0.142265 |                  9 |

### label-relevant shifted features

| source   | feature         |   month |   z_delta_vs_2024 |   spearman_corr |   weighted_abs_shift |
|:---------|:----------------|--------:|------------------:|----------------:|---------------------:|
| gfs      | wind850_speed   |       4 |          1.22955  |        0.812383 |             0.998865 |
| ldaps    | wind50max_speed |       4 |          0.967618 |        0.826312 |             0.799554 |
| gfs      | pblwind_speed   |       4 |          1.07342  |        0.71807  |             0.77079  |
| ldaps    | wind50max_speed |       2 |          0.895275 |        0.826312 |             0.739777 |
| ldaps    | wind10_speed    |       4 |          0.844679 |        0.81673  |             0.689874 |
| gfs      | wind100_speed   |       4 |          0.918604 |        0.749029 |             0.688061 |
| ldaps    | wind10_speed    |       2 |          0.8306   |        0.81673  |             0.678376 |
| gfs      | wind80_speed    |       4 |          0.877445 |        0.745333 |             0.653989 |
| gfs      | wind850_speed   |       2 |          0.788248 |        0.812383 |             0.640359 |
| gfs      | wind100_speed   |       2 |          0.832302 |        0.749029 |             0.623418 |
| gfs      | pblwind_speed   |       2 |          0.866457 |        0.71807  |             0.622177 |
| gfs      | wind80_speed    |       2 |          0.829557 |        0.745333 |             0.618296 |

### 해석

- 2025 risk는 모든 달에 균일하지 않고 특정 month에 집중됩니다.
- 중요한 점은 단순 weather shift가 아니라 **label relevance로 가중한 shift**라는 점입니다. label과 약한 feature가 크게 움직이는 것보다, label과 강한 wind feature가 움직이는 것이 score risk에 더 중요합니다.
- validation/model selection은 2024 평균만 볼 것이 아니라, exposed month/feature에 대한 성능을 별도로 확인해야 합니다.

### 타당성

- 타당한 점: weather-label correlation과 2025-vs-2024 shift를 곱해 score relevance가 높은 shift를 우선순위화했습니다.
- 한계: 2025에는 label이 없으므로 exposure는 risk proxy입니다. 실제 score impact는 submission feedback 또는 private split 이후에만 확정됩니다.

---

## 6. Proof matrix: 지금 무엇이 증명됐나?

![Proof matrix](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures/07_score_insight_proof_matrix.png)

| hypothesis                                     | evidence                                                                                              | status              | next_model_action                                                           |
|:-----------------------------------------------|:------------------------------------------------------------------------------------------------------|:--------------------|:----------------------------------------------------------------------------|
| SCADA residual slices identify score risk      | candidate error and FiCR transition differ by SCADA residual taxonomy                                 | supported           | train residual guards by taxonomy proxy, but only with NWP-safe predictors  |
| label regime controls score sensitivity        | boundary/high-power regimes have different row support, error and FiCR transitions                    | supported           | evaluate models by label-regime slices, not only total MAE                  |
| best spatial grid is wind-direction dependent  | {'kpx_group_1': 6, 'kpx_group_2': 4, 'kpx_group_3': 6} unique top grids by target                     | supported           | use direction-conditioned upstream grid weights                             |
| LDAPS-GFS disagreement marks error-prone cases | candidate error varies across disagreement bins; directionality is not monotonic for all groups       | partially_supported | use disagreement as uncertainty/regime feature, not a hard correction alone |
| lead hour should change model behavior         | lead-hour candidate error range={'kpx_group_1': 0.0473, 'kpx_group_2': 0.0413, 'kpx_group_3': 0.0405} | supported           | include lead-hour interactions with source/grid/wind features               |
| 2025 risk is month-concentrated                | top exposure months=[4, 2, 7, 6]                                                                      | supported           | weight validation/model selection by exposed months and features            |

## 결론

이번 추가 EDA의 결론은 다음입니다.

1. **score 개선의 병목은 전체 평균 error가 아니라 slice별 residual입니다.**
2. SCADA residual taxonomy, label regime, FiCR boundary, wind-direction grid, source disagreement, lead_hour, 2025 exposure가 모두 score-relevant한 축으로 확인됐습니다.
3. 즉 다음 phase는 feature dump가 아니라, 아래 네 가지 deployable 가설을 ablation해야 합니다.

Recommended next modeling tests:

- NWP/time/metadata만으로 SCADA residual taxonomy proxy를 예측할 수 있는지 검증.
- direction-conditioned upstream LDAPS grid feature를 static grid feature와 ablation.
- LDAPS-GFS disagreement를 uncertainty/regime feature로 추가해 error spike가 줄어드는지 검증.
- validation metric을 label regime + FiCR boundary + 2025 exposure month로 slice reporting.

중요한 caveat:

- SCADA 자체는 test에 없으므로 direct feature가 아닙니다.
- 2025 exposure는 label 없는 risk proxy입니다.
- correlation 기반 spatial EDA는 feature 설계 후보이지 causal proof가 아닙니다.
- 이 이슈에서 증명한 것은 “어떤 축이 score-relevant한가”이고, “그 축이 leaderboard를 올린다”는 것은 후속 ablation으로 검증해야 합니다.

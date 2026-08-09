# Forecast-Time Contract Follow-up EDA: lead, issue-cycle, and score-relevant proof matrix

이 이슈는 #31의 `3. Forecast-Time Contract`에서 나온 결론을 score 개선 관점으로 확장한 추가 EDA입니다. #32가 SCADA taxonomy, label regime, 방향 grid, source disagreement, 2025 exposure를 넓게 다뤘다면, 이 이슈는 **forecast-time contract 자체에서 나오는 모델링 제약과 기회**에 집중합니다.

핵심 질문:

1. `lead_hour`, `hour`, `data_available_kst_dtm`는 정말 독립 feature가 아닌가?
2. lead-hour별 error와 FiCR transition이 score-relevant한가?
3. issue-cycle 단위로 공통 over/under bias가 생기는가?
4. lag/ramp feature 중 무엇이 test-time available이고 무엇이 leakage인가?
5. 2025 test shift는 month만이 아니라 lead-hour까지 분해해야 하는가?

Artifacts:

- script: `experiments/forecast_time_contract_score_eda_20260809/run_forecast_time_contract_score_eda.py`
- figures: `experiments/forecast_time_contract_score_eda_20260809/results/figures/`
- tables: `experiments/forecast_time_contract_score_eda_20260809/results/*.csv`

---

## 1. Time Identity Contract

![time identity](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/01_time_identity_contract.png)

### 관찰

| dataset     |   rows |   unique_hour_to_lead_pairs |   cycle_hour_values |   lead_min |   lead_max |   violations_of_deterministic_hour_to_lead |
|:------------|-------:|----------------------------:|--------------------:|-----------:|-----------:|-------------------------------------------:|
| ldaps_train |  26304 |                          24 |                  13 |         12 |         35 |                                          0 |
| gfs_train   |  26304 |                          24 |                  13 |         12 |         35 |                                          0 |

### 해석

    - train weather에서 forecast target hour와 `lead_hour`는 1:1로 결정됩니다. 01-23시는 lead 12-34에 대응하고, 00시는 lead 35에 대응합니다.
- 따라서 `hour`와 `lead_hour`를 독립적인 시간 feature처럼 해석하면 안 됩니다.
- 이것은 모델에 둘 다 넣지 말라는 뜻이 아니라, 둘의 중요도를 독립 원인으로 해석하면 안 된다는 뜻입니다.

### 타당성 논의

- 타당한 점: raw LDAPS/GFS train의 timestamp에서 직접 계산했습니다.
- 한계: test도 동일 contract인 것은 #31에서 이미 확인했습니다. 여기서는 score overlay가 가능한 train/valid 중심으로 시각화했습니다.

---

## 2. Lead-Hour Error Surface

![lead error](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/02_lead_hour_error_surface.png)

### 관찰

| target      |   lead_hour |   rows |   candidate_abs_error |   baseline_abs_error |   mean_signed_error |   mean_abs_source_disagreement |   net_ficr_pass |
|:------------|------------:|-------:|----------------------:|---------------------:|--------------------:|-------------------------------:|----------------:|
| kpx_group_1 |          34 |    366 |              0.11947  |             0.118376 |         0.026439    |                        2.76536 |               2 |
| kpx_group_1 |          33 |    366 |              0.115032 |             0.114329 |         0.0318161   |                        2.80542 |               3 |
| kpx_group_2 |          35 |    366 |              0.114993 |             0.114451 |         0.0460494   |                        2.84162 |               1 |
| kpx_group_1 |          35 |    366 |              0.11389  |             0.113249 |         0.0362043   |                        2.84162 |               3 |
| kpx_group_2 |          34 |    366 |              0.113382 |             0.11331  |         0.0340208   |                        2.76536 |               4 |
| kpx_group_3 |          35 |    366 |              0.113232 |             0.114779 |         0.00576496  |                        2.77766 |               3 |
| kpx_group_2 |          33 |    366 |              0.112972 |             0.112892 |         0.0426971   |                        2.80542 |              -1 |
| kpx_group_2 |          13 |    366 |              0.112536 |             0.111941 |         0.0601586   |                        2.92924 |              -3 |
| kpx_group_3 |          33 |    366 |              0.111646 |             0.111276 |         0.000228096 |                        2.67979 |               0 |
| kpx_group_3 |          34 |    366 |              0.111268 |             0.112183 |        -0.00519105  |                        2.63431 |               2 |
| kpx_group_2 |          16 |    366 |              0.110748 |             0.111758 |         0.0455555   |                        2.83313 |               6 |
| kpx_group_2 |          14 |    366 |              0.108668 |             0.108914 |         0.0524932   |                        2.87932 |              -3 |
| kpx_group_1 |          13 |    366 |              0.107998 |             0.107754 |         0.0514801   |                        2.92924 |              -3 |
| kpx_group_2 |          15 |    366 |              0.107596 |             0.107433 |         0.047446    |                        2.92831 |               6 |
| kpx_group_2 |          17 |    366 |              0.107559 |             0.107878 |         0.0452283   |                        2.67633 |               4 |

### 해석

- lead-hour별 candidate error 차이가 존재합니다.
- 특히 긴 lead 쪽에서 error가 커지는 target이 있고, FiCR net transition도 lead별로 달라집니다.
- score 개선 관점에서는 lead별 calibration이나 `lead_hour x wind/source/grid` interaction이 ablation 대상입니다.

### 타당성 논의

- 타당한 점: 2024 validation prediction row에 raw forecast lead를 join해 실제 candidate/baseline error를 집계했습니다.
- 한계: `lead_hour`와 `hour`가 동일 축에 묶여 있으므로, lead 효과와 diurnal generation effect를 완전히 분리했다고 보면 안 됩니다.

---

## 3. Issue-Cycle Bias

![cycle bias](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/03_issue_cycle_bias.png)

### 관찰

| target      | data_available_kst_dtm   |   rows |   mean_signed_error |   mean_abs_error |   mean_actual_ratio |   mean_ldaps_speed |   mean_source_disagreement |   net_ficr_pass |
|:------------|:-------------------------|-------:|--------------------:|-----------------:|--------------------:|-------------------:|---------------------------:|----------------:|
| kpx_group_2 | 2024-01-21 13:00:00      |     24 |            0.811788 |         0.811788 |         0.0356557   |           14.2437  |                  -2.49634  |               0 |
| kpx_group_1 | 2024-01-22 13:00:00      |     24 |            0.804775 |         0.804775 |         0.00217617  |           12.2472  |                  -2.40834  |               0 |
| kpx_group_2 | 2024-01-22 13:00:00      |     24 |            0.770506 |         0.770506 |         0.121698    |           12.2472  |                  -2.40834  |               0 |
| kpx_group_1 | 2024-01-21 13:00:00      |     24 |            0.759077 |         0.759077 |         0           |           14.2437  |                  -2.49634  |               0 |
| kpx_group_1 | 2024-01-23 13:00:00      |     24 |            0.554939 |         0.554939 |         0.0154006   |            8.85928 |                   1.7007   |               0 |
| kpx_group_2 | 2024-01-23 13:00:00      |     24 |            0.465021 |         0.465021 |         0.127194    |            8.85928 |                   1.7007   |               0 |
| kpx_group_3 | 2024-01-19 13:00:00      |     24 |            0.409189 |         0.409189 |         2.86964e-05 |            8.18806 |                  -4.26667  |               0 |
| kpx_group_3 | 2024-01-21 13:00:00      |     24 |            0.383356 |         0.383356 |         0.320858    |           15.5167  |                  -1.22336  |               0 |
| kpx_group_2 | 2024-02-29 13:00:00      |     24 |            0.37923  |         0.384581 |         0.224397    |           13.5913  |                  -3.16655  |               1 |
| kpx_group_2 | 2024-01-19 13:00:00      |     24 |            0.354835 |         0.354835 |         0           |            9.092   |                  -3.36273  |               0 |
| kpx_group_3 | 2024-02-20 13:00:00      |     24 |            0.329135 |         0.329135 |         0.0158693   |            8.44068 |                   0.587972 |               0 |
| kpx_group_1 | 2024-01-19 13:00:00      |     24 |            0.319277 |         0.319277 |         0           |            9.092   |                  -3.36273  |               0 |
| kpx_group_2 | 2024-01-24 13:00:00      |     24 |            0.300021 |         0.300021 |         0.192898    |            8.09064 |                   2.71248  |               0 |
| kpx_group_3 | 2024-03-28 13:00:00      |     24 |           -0.28513  |         0.286398 |         0.909629    |           17.5632  |                  -6.47883  |               0 |
| kpx_group_2 | 2024-07-03 13:00:00      |     24 |            0.284955 |         0.284955 |         0.427366    |           15.758   |                   1.14801  |               0 |

### 해석

- 같은 `data_available_kst_dtm`에서 발행된 24개 target hour가 함께 over/under 되는 cycle-level bias가 있습니다.
- 이건 row-level 후처리보다 issue-cycle regime correction이 더 맞는 후보가 있음을 의미합니다.
- 단, test label이 없으므로 cycle bias를 직접 보정하려면 NWP-only cycle descriptors로 proxy를 만들어야 합니다.

### 타당성 논의

- 타당한 점: 동일 issue cycle 내 24개 forecast row를 묶어 signed error를 계산했습니다.
- 한계: validation label을 쓴 사후 진단입니다. deployable correction은 cycle 평균 풍속, source disagreement, ramp shape 같은 forecast-only 변수로만 가능해야 합니다.

---

## 4. Forecast-Safe Ramp Features

![ramp proxy](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/04_available_ramp_proxy_correlations.png)

### 관찰

| target      | feature               |   rows |   spearman_with_actual_ramp |   spearman_with_signed_error |   spearman_with_abs_error |
|:------------|:----------------------|-------:|----------------------------:|-----------------------------:|--------------------------:|
| kpx_group_1 | same_cycle_ldaps_ramp |   8409 |                   0.13876   |                  -0.00674485 |                0.0237626  |
| kpx_group_1 | same_cycle_gfs_ramp   |   8409 |                   0.170546  |                   0.0184225  |                0.00167704 |
| kpx_group_1 | same_lead_issue_delta |   8752 |                   0.0265043 |                   0.0362019  |                0.143394   |
| kpx_group_1 | source_disagreement   |   8775 |                  -0.0493149 |                  -0.00248302 |               -0.145634   |
| kpx_group_2 | same_cycle_ldaps_ramp |   8409 |                   0.126629  |                  -0.00786396 |                0.0302165  |
| kpx_group_2 | same_cycle_gfs_ramp   |   8409 |                   0.168902  |                   0.025138   |                0.0178133  |
| kpx_group_2 | same_lead_issue_delta |   8752 |                   0.0194001 |                   0.0832496  |                0.150069   |
| kpx_group_2 | source_disagreement   |   8775 |                  -0.0566196 |                  -0.045027   |               -0.147894   |
| kpx_group_3 | same_cycle_ldaps_ramp |   8409 |                   0.13878   |                   0.00221107 |                0.0541377  |
| kpx_group_3 | same_cycle_gfs_ramp   |   8409 |                   0.157152  |                   0.00876328 |                0.0103439  |
| kpx_group_3 | same_lead_issue_delta |   8752 |                   0.0212639 |                  -0.0173328  |                0.25858    |
| kpx_group_3 | source_disagreement   |   8775 |                  -0.0537127 |                   0.0551707  |               -0.172157   |

### 해석

- 같은 issue cycle 안의 LDAPS/GFS wind ramp, 같은 lead에서 issue-to-issue wind delta는 test-time에 만들 수 있는 feature입니다.
- 실제 발전 ramp나 current residual과의 correlation은 강하지 않지만 0이 아닙니다.
- 따라서 ramp는 단독 해결책이 아니라 regime/uncertainty interaction feature로 쓰는 것이 타당합니다.

### 타당성 논의

- 타당한 점: SCADA나 label 미래값 없이 weather forecast table에서 만들 수 있는 ramp만 사용했습니다.
- 한계: actual ramp와의 상관이 약하므로, 단순 ramp feature 추가만으로 큰 score 개선을 기대하면 안 됩니다.

---

## 5. Leakage Contrast

![leakage contrast](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/05_leakage_contrast.png)

### 관찰

| target      | feature                  | test_time_available   |   rows |   spearman_with_actual_ratio |
|:------------|:-------------------------|:----------------------|-------:|-----------------------------:|
| kpx_group_1 | target_time_actual_lag1  | False                 |  26193 |                    0.961655  |
| kpx_group_1 | target_time_actual_lag24 | False                 |  26130 |                    0.439375  |
| kpx_group_1 | ldaps_speed              | True                  |   8778 |                    0.758748  |
| kpx_group_1 | gfs_speed                | True                  |   8778 |                    0.760232  |
| kpx_group_1 | same_cycle_ldaps_ramp    | True                  |   8411 |                    0.0361941 |
| kpx_group_1 | same_lead_issue_delta    | True                  |   8754 |                    0.333482  |
| kpx_group_2 | target_time_actual_lag1  | False                 |  26194 |                    0.964348  |
| kpx_group_2 | target_time_actual_lag24 | False                 |  26132 |                    0.434468  |
| kpx_group_2 | ldaps_speed              | True                  |   8778 |                    0.775256  |
| kpx_group_2 | gfs_speed                | True                  |   8778 |                    0.78615   |
| kpx_group_2 | same_cycle_ldaps_ramp    | True                  |   8411 |                    0.0454485 |
| kpx_group_2 | same_lead_issue_delta    | True                  |   8754 |                    0.343491  |
| kpx_group_3 | target_time_actual_lag1  | False                 |  17535 |                    0.957674  |
| kpx_group_3 | target_time_actual_lag24 | False                 |  17508 |                    0.410229  |
| kpx_group_3 | ldaps_speed              | True                  |   8778 |                    0.831696  |
| kpx_group_3 | gfs_speed                | True                  |   8778 |                    0.796973  |
| kpx_group_3 | same_cycle_ldaps_ramp    | True                  |   8411 |                    0.058976  |
| kpx_group_3 | same_lead_issue_delta    | True                  |   8754 |                    0.366431  |

### 해석

- target-time actual lag는 label과 강하게 연결되지만 test 제출 시점에는 사용할 수 없습니다.
- forecast-safe weather features는 상대적으로 약하지만 deployable합니다.
- 앞으로 lag/ramp 실험은 반드시 `target-time actual lag`와 `forecast issue feature`를 분리해서 기록해야 합니다.

### 타당성 논의

- 타당한 점: 같은 target에 대해 non-deployable actual lag와 deployable forecast feature를 같은 상관 척도로 비교했습니다.
- 한계: correlation 비교는 feature value의 정보량만 보여줍니다. 모델 안에서 interaction으로 쓰일 때의 효과는 별도 ablation이 필요합니다.

---

## 6. 2025 Lead-Month Exposure

![lead month exposure](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/06_2025_lead_month_exposure.png)

### 관찰

| source   | feature            |   month |   lead_hour |   train2024_mean |   test2025_mean |   z_delta_vs_2024 |   test_above_train_p90_rate |
|:---------|:-------------------|--------:|------------:|-----------------:|----------------:|------------------:|----------------------------:|
| gfs      | wind_primary_speed |       4 |          34 |          5.34522 |        12.5627  |           2.0244  |                    0.551852 |
| gfs      | wind_primary_speed |       4 |          33 |          5.48273 |        12.4192  |           1.8858  |                    0.551852 |
| gfs      | wind_primary_speed |       4 |          35 |          5.71298 |        12.6165  |           1.79905 |                    0.474074 |
| gfs      | wind_primary_speed |       4 |          32 |          5.59858 |        12.2871  |           1.77751 |                    0.511111 |
| gfs      | wind_primary_speed |       4 |          31 |          5.81129 |        11.8801  |           1.59715 |                    0.462963 |
| gfs      | wind_primary_speed |       4 |          12 |          5.87337 |        11.5031  |           1.52054 |                    0.492593 |
| gfs      | wind_primary_speed |       4 |          13 |          5.92449 |        11.3349  |           1.40305 |                    0.392593 |
| gfs      | wind_primary_speed |       4 |          30 |          5.99389 |        11.1722  |           1.33535 |                    0.437037 |
| gfs      | wind_primary_speed |       4 |          14 |          5.89718 |        11.0705  |           1.23737 |                    0.381481 |
| ldaps    | wind_primary_speed |       4 |          27 |          6.83691 |        10.5921  |           1.22509 |                    0.391667 |
| gfs      | wind_primary_speed |       4 |          16 |          5.69522 |        10.7346  |           1.20219 |                    0.422222 |
| gfs      | wind_primary_speed |       4 |          29 |          6.05539 |        10.7667  |           1.17609 |                    0.425926 |
| gfs      | wind_primary_speed |       4 |          17 |          5.69308 |        10.6433  |           1.16755 |                    0.385185 |
| ldaps    | wind_primary_speed |       4 |          25 |          6.52738 |        10.5587  |           1.16629 |                    0.485417 |
| gfs      | wind_primary_speed |       4 |          15 |          5.82403 |        10.8336  |           1.16575 |                    0.392593 |
| ldaps    | wind_primary_speed |       4 |          24 |          6.39521 |        10.2277  |           1.14332 |                    0.470833 |
| ldaps    | wind_primary_speed |       4 |          35 |          5.7474  |         8.62129 |           1.1433  |                    0.45     |
| ldaps    | wind_primary_speed |       2 |          28 |          7.22079 |        11.7398  |           1.13414 |                    0.46875  |
| ldaps    | wind_primary_speed |       4 |          26 |          6.8573  |        10.6499  |           1.11982 |                    0.483333 |
| ldaps    | wind_primary_speed |       4 |          34 |          5.79297 |         8.7107  |           1.11657 |                    0.427083 |

### 해석

- 2025 shift는 월 단위로만 보는 것보다 lead-hour까지 분해했을 때 더 구체적인 위험 셀이 보입니다.
- exposed month/lead cell에서 모델 selection과 postprocessing이 흔들릴 수 있습니다.
- validation report는 최소한 `month x lead_hour x target` slice를 포함해야 합니다.

### 타당성 논의

- 타당한 점: train 2024와 test 2025의 동일 month/lead cell을 비교했습니다.
- 한계: 2025 label이 없으므로 이것은 score impact가 아니라 exposure risk입니다.

---

## 7. Proof Matrix

![proof matrix](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures/07_forecast_time_score_proof_matrix.png)

| claim                                                  | evidence                                                                                                                                                                                 | status              | model_consequence                                              |
|:-------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:---------------------------------------------------------------|
| lead_hour is not independent from hour                 | violations=0; 24 target hours map deterministically to leads 12-35                                                                                                                       | proved              | avoid interpreting hour and lead as independent causal signals |
| lead_hour has score-relevant error structure           | candidate abs-error range by target={'kpx_group_1': 0.0473, 'kpx_group_2': 0.0413, 'kpx_group_3': 0.0405}                                                                                | supported           | use lead interactions and report validation by lead            |
| issue-cycle bias exists                                | 95th percentile absolute cycle signed error={'kpx_group_1': 0.1592, 'kpx_group_2': 0.188, 'kpx_group_3': 0.1607}                                                                         | supported           | consider cycle-level calibration/regime features               |
| test-time-available ramps are usable but weak proxies  | best Spearman with actual ramp={'kpx_group_1': 0.171, 'kpx_group_2': 0.169, 'kpx_group_3': 0.157}                                                                                        | partially_supported | use as interactions/uncertainty features, not standalone fixes |
| target-time label lags are tempting but non-deployable | max corr by availability={False: 0.964, True: 0.832}                                                                                                                                     | proved              | separate target lag experiments from forecast-safe features    |
| 2025 shift is lead-month specific                      | top shifted month/lead cells=[{'month': 4, 'lead_hour': 34}, {'month': 4, 'lead_hour': 33}, {'month': 4, 'lead_hour': 35}, {'month': 4, 'lead_hour': 32}, {'month': 4, 'lead_hour': 31}] | supported           | stress-test exposed lead-month cells, not only full-year 2024  |

## 결론

이번 추가 EDA에서 얻은 score-oriented lesson learned는 다음입니다.

1. **`lead_hour`와 `hour`는 독립 feature가 아니라 같은 forecast contract에서 나온 결정적 매핑입니다.**
2. **lead-hour별 error surface와 issue-cycle bias가 존재하므로, row-level 평균 MAE만으로는 모델을 고르면 안 됩니다.**
3. **target actual lag는 강력하지만 leakage입니다. deployable feature는 forecast issue 안의 ramp/source/grid/lead 구조에서 만들어야 합니다.**
4. **2025 exposure는 month뿐 아니라 lead-hour까지 분해해야 validation risk가 보입니다.**

다음에 증명해야 할 모델링 가설:

- `lead_hour x wind/source/grid` interaction이 static weather feature보다 validation slice error를 낮추는가?
- issue-cycle descriptor로 cycle-level signed bias를 줄일 수 있는가?
- forecast-safe ramp feature가 FiCR boundary row의 pass-to-fail을 줄이는가?
- 2025 exposed month/lead cell을 가중한 validation selection이 public/private transfer를 개선하는가?

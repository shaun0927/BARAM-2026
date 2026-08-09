# Comprehensive EDA Atlas: dataset contract, SCADA-label generation, spatial topology, FiCR surfaces, and 2025 shift

이 이슈는 모델 후보를 promote하기 위한 이슈가 아니라, 이후 모든 모델링/후처리 판단의 기준이 될 **EDA 전용 atlas**입니다.

기존 작업은 EDA가 전혀 없었던 것은 아니지만, 컬럼 계약, 시간 구조, 공간 구조, SCADA-label 생성 과정, FiCR 민감 row, 2025 test shift가 하나의 문서로 연결되어 있지 않았습니다. 그래서 이번 이슈는 원본 데이터에서 직접 다시 계산한 표와 이미지를 기준으로 “무엇을 알고 있고, 무엇이 아직 미완인지”를 고정합니다.

Artifacts:

- script: `experiments/comprehensive_eda_20260809/run_comprehensive_eda.py`
- report: `experiments/comprehensive_eda_20260809/results/eda_atlas_report.md`
- figures: `experiments/comprehensive_eda_20260809/results/figures/`
- tables: `experiments/comprehensive_eda_20260809/results/*.csv`

---

## 1. Column Contract Audit

![Column contract missingness](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/01_column_contract_missingness.png)

### 해석

- 가장 큰 구조적 결측은 `kpx_group_3`의 2022 label 부재입니다. 이것은 단순 missing value가 아니라 validation 설계와 group별 학습 정책을 바꾸는 제약입니다.
- column-level audit은 아직 “결측률/범위/분포” 수준입니다. 각 weather column의 실제 단위, 물리적 변환 가능성, train/test parity까지 포함한 data dictionary는 별도 후속이 필요합니다.

### 다음 검증

- 모든 weather column에 대해 `unit`, `valid range`, `physical transform`, `train/test support`, `model-use policy`를 붙인 `column_dictionary.csv` 생성.
- 결측이 0인 컬럼도 out-of-range, constant-by-grid, train/test coordinate drift를 별도로 검사.

---

## 2. Label Structure

![Label monthly structure](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/02_label_monthly_structure.png)

![Label ratio distribution](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/03_label_ratio_distribution.png)

### 해석

- 발전량 분포는 group/year/month별로 강하게 다릅니다. 같은 capacity normalization을 해도 group3는 2022 label이 없고, 2023-2024에서도 zero/near-zero 비중이 상대적으로 큽니다.
- 평가 대상 row는 `actual >= 10% capacity` 조건에 의해 계절적으로 크게 바뀝니다. 따라서 평균 MAE 중심 EDA만으로는 점수 지배 row를 설명하기 어렵습니다.
- group1/group2는 같은 VESTAS 계열이지만 monthly surface가 완전히 동일하지 않습니다. group-shared model을 쓰더라도 group-specific residual/guardrail이 필요합니다.

### 다음 검증

- group별 label regime taxonomy: zero, near-zero, mid-power, high-power, capacity-edge, ramp, flatline.
- group3는 별도 problem으로 취급하고, group1/2에서 배운 정책을 자동 전이하지 않기.

---

## 3. Forecast-Time Contract

![Forecast time contract](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/04_forecast_time_contract.png)

### 검증 결과

| dataset | rows | forecast_times | expected_grids | bad_grid_count_times | lead_min | lead_max |
|---|---:|---:|---:|---:|---:|---:|
| ldaps_train | 420864 | 26304 | 16 | 0 | 12 | 35 |
| gfs_train | 236736 | 26304 | 9 | 0 | 12 | 35 |
| ldaps_test | 140160 | 8760 | 16 | 0 | 12 | 35 |
| gfs_test | 78840 | 8760 | 9 | 0 | 12 | 35 |

### 해석

- train/test 모두 forecast timestamp별 grid completeness는 정상입니다.
- lead hour는 12-35로 완전히 규칙적입니다. 즉 이 데이터는 generic hourly time series가 아니라 “하루 1개 issue cycle에서 24개 target hour를 제공하는 forecast table”입니다.
- `hour`, `lead_hour`, `data_available date`는 강하게 묶여 있으므로 독립 feature처럼 해석하면 안 됩니다.

### 다음 검증

- 모든 join은 `forecast_kst_dtm` 기준으로 하되, feature engineering은 `data_available_kst_dtm`와 issue-cycle 구조를 보존해야 함.
- lag/ramp feature를 만들 때 target-time lag인지 forecast-issue lag인지 명시.

---

## 4. Spatial Topology

![Spatial topology](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/05_spatial_topology_turbines_grids.png)

### 해석

- GFS 9개 grid는 발전단지 주변을 매우 coarse하게 감싸는 구조입니다.
- LDAPS 16개 grid는 터빈 cluster와 같은 작은 영역에 밀집해 있습니다.
- nearest grid와 highest-correlation grid가 항상 같은지는 아직 별도 검증 대상입니다. 현재 broad spatial feature dump가 실패했다고 해서 spatial structure 자체가 무의미하다고 결론내리면 안 됩니다.

### 다음 검증

- turbine별 nearest LDAPS/GFS grid와 label-correlation top grid 비교.
- 풍향별 upstream grid selection.
- group1/2 VESTAS split과 group3 UNISON split을 반영한 group-specific spatial feature.

---

## 5. Weather-Label Correlations

![Weather grid label correlations](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/06_weather_grid_label_correlations.png)

### 해석

- LDAPS `wind50max_speed`와 `wind10_speed`, GFS `wind850_speed` 계열이 label ratio와 높은 rank correlation을 가집니다.
- group3의 top wind correlation이 가장 높게 나오는 구간이 있으며, 이는 group3가 단순히 “데이터가 부족한 group”이 아니라 별도 power-curve/operating-regime을 가진 group일 가능성을 강화합니다.
- correlation은 단변량 진단입니다. 이것만으로 feature 채택을 결정하면 collinearity와 selection noise에 취약합니다.

### 다음 검증

- top correlated wind feature가 model residual도 설명하는지 확인.
- monthly/seasonal correlation stability.
- wind direction-conditioned correlation.

---

## 6. SCADA-to-Label Reconstruction

![SCADA label reconstruction](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/07_scada_label_reconstruction.png)

![SCADA power curve clouds](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/08_scada_power_curve_clouds.png)

### 검증 결과

| maker | target | kind | rows | pearson_corr | median_scada_over_label | mae_ratio |
|---|---|---|---:|---:|---:|---:|
| VESTAS | kpx_group_1 | raw | 26200 | 0.0010 | 1.0148 | 9.4268 |
| VESTAS | kpx_group_1 | clean | 26200 | 0.9491 | 1.0030 | 0.0332 |
| VESTAS | kpx_group_2 | raw | 26201 | -0.0003 | 1.0110 | 11.0892 |
| VESTAS | kpx_group_2 | clean | 26201 | 0.9486 | 0.9984 | 0.0354 |
| UNISON | kpx_group_3 | raw | 17538 | 0.9790 | 0.9961 | 0.0175 |
| UNISON | kpx_group_3 | clean | 17538 | 0.9751 | 0.9929 | 0.0209 |

### 해석

- `power_kw10m`는 hourly mean이 아니라 10분 값의 hourly sum으로 읽어야 합니다.
- VESTAS raw power는 극단 outlier 때문에 전체 correlation이 거의 0으로 붕괴합니다. 간단한 물리 범위 clean 후 group1/2 label reconstruction corr이 약 0.949로 복구됩니다.
- UNISON은 raw에서도 높게 재구성되며 VESTAS와 outlier 구조가 다릅니다.
- SCADA-label residual은 이 대회에서 가장 중요한 EDA 대상입니다. test에는 SCADA가 없지만, train에서 label 생성과 operating state를 가장 직접적으로 설명합니다.

### 다음 검증

- SCADA-label residual timeline을 event taxonomy로 분해: outage, curtailment, forecast miss, aggregation mismatch, reporting residual.
- turbine dropout signature를 NWP/metadata만으로 예측 가능한지 검증.
- SCADA wind -> NWP wind bias를 월/풍향/풍속대별로 분해.

---

## 7. 2025 Test Shift

![2025 weather shift](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/09_test2025_weather_shift.png)

### 해석

- 2025 test는 2024 validation과 동일한 weather regime이 아닙니다.
- 특히 2월/4월/7월/9월의 GFS upper-level and near-surface wind 계열이 크게 움직입니다.
- 단순히 “2024 holdout에서 좋다”는 이유만으로 2025 submission-safe라고 볼 수 없습니다.

### 다음 검증

- label-relevant feature만 사용한 exposure risk score 생성.
- 2025 각 월이 2022/2023/2024 중 어느 해와 가까운지 비교.
- public LB가 특정 month/regime에 치우쳤을 가능성 점검.

---

## 8. FiCR-Sensitive Label Surface

![FiCR evaluation support](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/10_ficr_evaluation_support_surface.png)

![FiCR threshold slices](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/11_ficr_threshold_sensitive_slices.png)

### 해석

- 평가 대상 support 자체가 group/month/hour별로 크게 다릅니다.
- `actual/capacity`가 10% 근처인 row는 평가 포함/제외의 경계에 있으므로, postprocessing과 residual correction이 평균 MAE보다 더 복잡하게 score에 작용합니다.
- FiCR는 pass/fail boundary metric이므로 smoothing이나 spread shrink가 항상 좋은 방향이 아닙니다.

### 다음 검증

- 현재 best model residual을 FiCR surface에 overlay.
- fail-to-pass / pass-to-fail row의 weather/SCADA residual profile 분석.
- training loss에 evaluator row support와 FiCR band를 어떻게 반영할지 재검토.

---

## 결론

이번 EDA로 확인된 것은 “EDA가 없었다”가 아니라, **EDA가 흩어져 있었고 모델링 판단을 지배할 만큼 통합되지 않았다**는 점입니다.

다음 모델링 전 고정해야 할 우선순위:

1. column dictionary 완성
2. SCADA-label residual taxonomy
3. spatial topology 기반 feature 재설계
4. 2025 exposure risk score
5. best-model residual을 이 EDA atlas 위에 재매핑
6. group3 독립 정책 유지

이 이슈는 이후 phase가 작은 local delta를 반복하기 전에, 데이터셋 생성 구조와 scoring surface를 먼저 합의하기 위한 기준 문서로 사용해야 합니다.

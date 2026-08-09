# Comprehensive EDA Atlas: BARAM-2026 dataset walkthrough and diagnostics

이 이슈는 모델 성능을 홍보하기 위한 기록이 아니라, BARAM-2026 데이터를 처음 보는 사람이 **데이터셋의 구조, 컬럼 의미, 시간 계약, 라벨 생성 방식, SCADA/NWP의 한계, test shift**를 순서대로 이해할 수 있게 만든 EDA 기준 문서입니다.

핵심 결론부터 말하면, 이 데이터는 단순한 hourly time-series 문제가 아닙니다. `forecast_kst_dtm` 기준의 발전량을 예측하되, 입력 weather는 `data_available_kst_dtm`에 발행된 예보 cycle에서 나온 12-35시간 lead forecast이고, train에는 SCADA가 있지만 test에는 SCADA가 없습니다. 따라서 “train에서 SCADA로 라벨을 잘 복원했다”는 사실과 “test에서 NWP만으로 잘 예측할 수 있다”는 사실은 같은 말이 아닙니다.

## 먼저 읽을 순서

1. **파일 구조와 역할**을 먼저 봅니다. 어떤 파일이 target이고, 어떤 파일이 feature이며, 어떤 파일은 train-only 보조 정보인지 구분해야 합니다.
2. **시간 구조**를 봅니다. `forecast_kst_dtm`, `data_available_kst_dtm`, `lead_hour`를 구분하지 못하면 lag feature와 validation이 쉽게 잘못됩니다.
3. **라벨 구조**를 봅니다. group별, 월별, 연도별 label support가 다르고 특히 `kpx_group_3`은 2022 label이 없습니다.
4. **weather grid 구조**를 봅니다. LDAPS는 16개 grid, GFS는 9개 grid이며, 방향/계절에 따라 유효한 grid가 달라집니다.
5. **SCADA의 의미**를 봅니다. SCADA는 라벨 생성 과정을 설명하는 teacher signal이지만 test-time feature가 아닙니다.
6. **FiCR 평가 구조**를 봅니다. 실제 점수는 `actual >= 10% capacity` support에 민감하므로 전체 평균 오차만 보면 잘못된 판단을 합니다.
7. 마지막으로 **2025 test shift와 residual blind spot**을 봅니다. phase를 계속 돌려도 개선이 작았던 이유가 여기에서 드러납니다.

## 데이터셋 한 줄 요약

BARAM-2026은 제주 풍력 발전량을 예측하는 문제로, 각 `kpx_group`의 시간별 발전량 label을 예측해야 합니다. 입력은 예보 시점별 NWP weather grid이며, train에는 SCADA turbine 관측치가 함께 제공됩니다. 하지만 test 제출에서는 SCADA를 사용할 수 없으므로, 모델은 결국 NWP weather와 시간/공간 metadata만으로 발전량을 추정해야 합니다.

## 파일별 역할

| 파일/폴더 | 역할 | 모델링 관점에서의 의미 |
|---|---|---|
| `train/train_labels.csv` | group별 발전량 정답 label | 최종 예측 target. group/month/year별 분포가 다르므로 가장 먼저 EDA해야 하는 파일 |
| `sample_submission.csv` | 제출해야 할 timestamp/group 구조 | test의 시간 범위와 row support를 확인하는 기준 |
| `train/ldaps_train.csv`, `test/ldaps_test.csv` | LDAPS 예보 weather grid | 고해상도 지역 예보. 16개 grid, forecast lead 12-35h |
| `train/gfs_train.csv`, `test/gfs_test.csv` | GFS 예보 weather grid | 저해상도 전지구 예보. 9개 grid, forecast lead 12-35h |
| `train/scada_vestas_train.csv`, `train/scada_unison_train.csv` | turbine 단위 SCADA 관측 | 라벨 생성/품질을 이해하는 teacher signal. test에는 없으므로 직접 feature로 쓰면 안 됨 |
| `info.xlsx` | turbine/group/grid 위치 metadata | grid 선택, 거리 기반 feature, group별 공간 해석에 필요 |
| `data_description.md` | 대회 제공 데이터 설명 | 컬럼명과 기본 정의를 확인하는 원본 설명 문서 |

## 핵심 컬럼 이해

| 컬럼/컬럼군 | 의미 | 주의점 |
|---|---|---|
| `forecast_kst_dtm` | 예측 대상 시간 | label과 submission이 맞춰지는 시간축 |
| `data_available_kst_dtm` | 예보가 사용 가능해진 시간 | forecast issue cycle. target time과 섞으면 leakage 또는 잘못된 lag가 생김 |
| `lead_hour` | 예보 발행 후 target까지의 시간 | 이 데이터에서는 12-35h로 규칙적이며, 단순 hour feature와 독립이 아님 |
| `grid_id` | NWP 격자 지점 | LDAPS 16개, GFS 9개. 모든 grid가 항상 같은 중요도를 갖지 않음 |
| weather 변수들 | wind, temperature, pressure, humidity 등 예보값 | 변수 단위/범위는 column dictionary와 train/test parity로 검증해야 함 |
| SCADA 변수들 | turbine별 풍속, 방향, 출력 등 관측값 | train-only. label reconstruction과 이상치 판별에는 유용하지만 test 직접 사용 불가 |
| target 발전량 | group별 실제 발전량 | capacity normalization 후에도 group/month별 regime 차이가 큼 |

## EDA로 확정한 데이터셋의 큰 구조

- **시간 계약은 정상입니다.** LDAPS/GFS train/test 모두 forecast timestamp별 grid completeness가 맞고, lead hour는 12-35로 규칙적입니다.
- **라벨 support는 균일하지 않습니다.** `kpx_group_3`의 2022 label 부재, group별 월별 zero/near-zero 비중, `actual >= 10% capacity` 조건 때문에 평가에 기여하는 row가 계절적으로 달라집니다.
- **SCADA는 label을 강하게 설명하지만 test feature가 아닙니다.** 특히 VESTAS 계열은 clean SCADA로 label을 잘 복원할 수 있으나, 이는 NWP-only test 예측 성능을 보장하지 않습니다.
- **weather grid는 방향 조건부로 봐야 합니다.** 평균 correlation 하나로 best grid를 고정하면 wind direction regime에서 중요한 공간 신호를 놓칩니다.
- **2025 test exposure는 train 평균과 다릅니다.** label-relevance로 가중하면 특정 월과 기상 regime에서 shift risk가 집중됩니다.
- **모델 family만 바꿔서는 shared residual이 남습니다.** LGBM/XGBoost가 상대적으로 강하지만 residual correlation이 높아, 같은 blind spot을 공유합니다.

## 모델링 전에 반드시 지켜야 할 규칙

- validation split은 group/year/month support를 보존해야 합니다. random row split은 이 문제의 난이도를 과소평가합니다.
- SCADA를 직접 feature로 쓰는 실험과 NWP-only 실험을 분리해야 합니다. SCADA 성능은 teacher ceiling 또는 label QA로 해석해야 합니다.
- `forecast_kst_dtm` 기준 join과 `data_available_kst_dtm` 기준 feature engineering을 분리해서 기록해야 합니다.
- FiCR에 민감한 row, 즉 `actual >= 10% capacity` 근처와 high-power/ramp regime을 별도 평가해야 합니다.
- group3는 2022 label이 없으므로 group1/2와 같은 방식으로 transfer된다고 가정하면 안 됩니다.

## 이슈 안의 그림을 읽는 법

- **그림 01-04**: 데이터의 기본 구조입니다. 결측, label support, 시간 계약을 확인합니다.
- **그림 05-07**: 공간 구조와 SCADA-label 관계입니다. turbine/grid 위치와 SCADA teacher signal의 한계를 봅니다.
- **그림 08-11**: 2025 shift와 FiCR 민감 row입니다. test에서 무엇이 달라지는지 봅니다.
- **그림 12-21**: deep-dive diagnostics입니다. SCADA residual, NWP-SCADA bias, 방향 조건부 grid, 2025 exposure를 봅니다.
- **그림 22-29**: 최종 closure입니다. raw column dictionary, residual event timeline, model-family residual atlas를 봅니다.

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

- 이 그림은 **기하학적 topology만 보여주는 그림**입니다. 즉 터빈, LDAPS grid, GFS grid가 어디에 있는지는 보여주지만, 어떤 grid가 가장 예측력이 좋은지는 증명하지 않습니다.
- GFS 9개 grid는 발전단지 주변을 매우 coarse하게 감싸는 구조이고, LDAPS 16개 grid는 터빈 cluster 주변의 작은 영역에 더 촘촘히 걸쳐 있습니다.
- 하지만 nearest grid와 highest-correlation grid는 같은 개념이 아닙니다. 실제 보완 분석에서는 modal nearest LDAPS grid가 group1 `5`, group2 `6`, group3 `12`인 반면, static highest-correlation LDAPS grid는 세 group 모두 `13`으로 나옵니다.
- 더 중요한 것은 풍향 조건부로 top grid가 계속 바뀐다는 점입니다. group1은 8개 방향 sector에서 6개 grid, group2는 4개 grid, group3는 6개 grid가 top으로 등장합니다.
- 따라서 이 섹션의 결론은 “가까운 grid 하나를 쓰면 된다”가 아니라, **spatial signal은 direction-conditioned / upstream-like feature로 설계해야 한다**입니다.
- broad spatial feature dump가 실패했다고 해서 spatial structure 자체가 무의미하다고 결론내리면 안 됩니다. 실패한 것은 topology 해석이 아니라, topology를 모델 feature로 옮긴 방식일 수 있습니다.

![Spatial topology reinterpretation](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/30_spatial_topology_reinterpretation.png)

### 정량 재해석

| target | modal nearest LDAPS | static corr-top LDAPS | corr gap top-nearest | unique direction-top grids | direction sectors equal nearest |
|---|---:|---:|---:|---:|---:|
| `kpx_group_1` | 5 | 13 | +0.148 | 6 | 1/8 |
| `kpx_group_2` | 6 | 13 | +0.027 | 4 | 0/8 |
| `kpx_group_3` | 12 | 13 | +0.002 | 6 | 1/8 |

### 모델링 관점의 결론

- `nearest_grid_value`는 geometry prior일 뿐이고, 단독 feature 정책으로는 부족합니다.
- `static top grid`도 전체 평균 correlation 기준일 뿐이라 direction regime을 놓칩니다.
- 다음 spatial feature는 turbine centroid와 wind vector를 이용한 soft upstream weighting, 또는 direction-conditioned top-grid pooling이어야 합니다.
- spatial EDA의 최종 검증은 label correlation이 아니라 current-anchor residual이 줄어드는지로 해야 합니다.

### 다음 검증

- direction-conditioned top-grid feature와 soft-upstream feature를 같은 validation contract에서 비교.
- group1/2 VESTAS split과 group3 UNISON split을 반영한 group-specific spatial pooling.
- spatial feature가 FiCR boundary 및 current-anchor residual을 실제로 개선하는지 검증.

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

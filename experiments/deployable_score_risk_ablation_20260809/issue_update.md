# Deployable score-risk ablation from Issue #32 insights

이 실험은 Issue #32에서 얻은 인사이트가 실제 validation score 개선으로 이어지는지 검증합니다.

검증한 후보:

- `B1_aggregate`: 기존 aggregate weather baseline
- `B1_plus_direction_grid`: 풍향 sector별 best LDAPS grid feature
- `B1_plus_source_disagreement`: LDAPS-GFS disagreement feature
- `B1_plus_lead_interactions`: lead_hour 및 lead-hour interaction feature
- `B1_plus_all_score_insights`: 위 세 feature family를 모두 추가
- `B1_all_plus_2025_exposure_weight`: exposed month에 더 큰 sample weight를 준 모델

![Ablation summary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/deployable_score_risk_ablation_20260809/results/figures/01_ablation_score_and_proxy_summary.png)

## Main Result

| experiment                       |    score |   one_minus_nmae |   avg_nmae |     ficr |   worst_month |   high_generation_score |   eligible_hours |   feature_count_group1 |   feature_count_group2 |   feature_count_group3 |   delta_score_vs_baseline |   delta_ficr_vs_baseline |   delta_nmae_vs_baseline |
|:---------------------------------|---------:|-----------------:|-----------:|---------:|--------------:|------------------------:|-----------------:|-----------------------:|-----------------------:|-----------------------:|--------------------------:|-------------------------:|-------------------------:|
| B1_plus_lead_interactions        | 0.611835 |         0.869948 |   0.130052 | 0.353722 |      0.573404 |                0.600124 |            14531 |                    293 |                    293 |                    293 |               6.77525e-05 |              0.000215724 |              8.0219e-05  |
| B1_aggregate                     | 0.611767 |         0.870029 |   0.129971 | 0.353506 |      0.591715 |                0.600197 |            14531 |                    286 |                    286 |                    286 |               0           |              0           |              0           |
| B1_plus_direction_grid           | 0.61141  |         0.869536 |   0.130464 | 0.353285 |      0.583219 |                0.598989 |            14531 |                    299 |                    299 |                    299 |              -0.00035702  |             -0.000221437 |              0.000492602 |
| B1_plus_source_disagreement      | 0.610672 |         0.869846 |   0.130154 | 0.351498 |      0.582219 |                0.598622 |            14531 |                    302 |                    302 |                    302 |              -0.00109527  |             -0.00200831  |              0.000182227 |
| B1_plus_all_score_insights       | 0.608509 |         0.869372 |   0.130628 | 0.347645 |      0.585815 |                0.596309 |            14531 |                    321 |                    321 |                    321 |              -0.00325856  |             -0.00586079  |              0.00065632  |
| B1_all_plus_2025_exposure_weight | 0.608113 |         0.868638 |   0.131362 | 0.347587 |      0.571532 |                0.59622  |            14531 |                    321 |                    321 |                    321 |              -0.00365483  |             -0.00591869  |              0.00139098  |

해석:

- 전체 최고 점수는 `B1_plus_lead_interactions`입니다.
- baseline 제외 최고 후보는 `B1_plus_lead_interactions`이고, baseline 대비 score delta는 `0.000068`입니다.
- `lead_hour` interaction은 baseline 대비 `+0.000068`로 아주 작게 개선됐습니다. 다만 개선 폭이 작고 `worst_month`와 high-generation score는 오히려 나빠져, 안정적인 score 개선으로 보기에는 약합니다.
- direction grid, source disagreement, all-insight feature union, 2025 exposure-weighted model은 baseline을 넘지 못했습니다.
- 따라서 Issue #32의 인사이트는 **score-relevant slice를 찾는 데는 유효했지만, 현재 fixed-LGBM/B1 protocol에서 단순 feature addition만으로는 충분하지 않다**고 해석해야 합니다.

## Regime-Level Delta

| experiment                       | target      | label_regime          |   rows |   baseline_abs_error |   candidate_abs_error |   delta_abs_error |   impact_delta |
|:---------------------------------|:------------|:----------------------|-------:|---------------------:|----------------------:|------------------:|---------------:|
| B1_plus_all_score_insights       | kpx_group_3 | very_high_80pct_plus  |    712 |            0.25972   |             0.251945  |      -0.00777512  |      -5.53589  |
| B1_plus_source_disagreement      | kpx_group_2 | mid_12_50pct          |   2215 |            0.132063  |             0.130547  |      -0.0015163   |      -3.3586   |
| B1_plus_source_disagreement      | kpx_group_3 | very_high_80pct_plus  |    712 |            0.25972   |             0.255755  |      -0.0039654   |      -2.82336  |
| B1_plus_lead_interactions        | kpx_group_3 | high_50_80pct         |   1353 |            0.14148   |             0.139433  |      -0.00204703  |      -2.76963  |
| B1_plus_lead_interactions        | kpx_group_3 | very_high_80pct_plus  |    712 |            0.25972   |             0.256264  |      -0.00345572  |      -2.46047  |
| B1_plus_direction_grid           | kpx_group_3 | very_high_80pct_plus  |    712 |            0.25972   |             0.256483  |      -0.00323737  |      -2.30501  |
| B1_plus_direction_grid           | kpx_group_2 | very_high_80pct_plus  |    953 |            0.0996076 |             0.097367  |      -0.00224061  |      -2.1353   |
| B1_plus_lead_interactions        | kpx_group_2 | mid_12_50pct          |   2215 |            0.132063  |             0.131107  |      -0.000956382 |      -2.11839  |
| B1_all_plus_2025_exposure_weight | kpx_group_2 | very_high_80pct_plus  |    953 |            0.0996076 |             0.0974802 |      -0.00212742  |      -2.02743  |
| B1_plus_source_disagreement      | kpx_group_3 | high_50_80pct         |   1353 |            0.14148   |             0.140284  |      -0.00119578  |      -1.6179   |
| B1_plus_direction_grid           | kpx_group_2 | ficr_boundary_8_12pct |    224 |            0.103425  |             0.0978908 |      -0.005534    |      -1.23962  |
| B1_plus_direction_grid           | kpx_group_1 | mid_12_50pct          |   2319 |            0.114219  |             0.113762  |      -0.00045762  |      -1.06122  |
| B1_plus_all_score_insights       | kpx_group_2 | ficr_boundary_8_12pct |    224 |            0.103425  |             0.0994063 |      -0.00401842  |      -0.900127 |
| B1_plus_all_score_insights       | kpx_group_1 | ficr_boundary_8_12pct |    219 |            0.0708553 |             0.0679758 |      -0.0028795   |      -0.630612 |
| B1_all_plus_2025_exposure_weight | kpx_group_2 | ficr_boundary_8_12pct |    224 |            0.103425  |             0.100617  |      -0.0028076   |      -0.628902 |
| B1_plus_source_disagreement      | kpx_group_2 | ficr_boundary_8_12pct |    224 |            0.103425  |             0.100887  |      -0.00253766  |      -0.568436 |
| B1_plus_all_score_insights       | kpx_group_2 | very_high_80pct_plus  |    953 |            0.0996076 |             0.0990375 |      -0.000570136 |      -0.543339 |
| B1_plus_lead_interactions        | kpx_group_1 | very_high_80pct_plus  |    840 |            0.135326  |             0.134748  |      -0.00057804  |      -0.485553 |
| B1_plus_source_disagreement      | kpx_group_3 | ficr_boundary_8_12pct |    212 |            0.0792275 |             0.0769444 |      -0.00228306  |      -0.48401  |
| B1_plus_all_score_insights       | kpx_group_1 | mid_12_50pct          |   2319 |            0.114219  |             0.114041  |      -0.000177759 |      -0.412222 |

해석:

- 일부 regime에서는 개선이 존재하지만, 다른 regime에서 악화되어 total score가 상쇄됩니다.
- 즉 #32의 insight는 “어디를 봐야 하는가”를 맞혔지만, 현재 feature 구현은 그 slice를 안정적으로 개선하지 못했습니다.

## FiCR Transition Delta

| experiment                       | target      | label_regime          |   fail_to_pass8 |   pass_to_fail8 |   net_pass8 |   delta_abs_error |
|:---------------------------------|:------------|:----------------------|----------------:|----------------:|------------:|------------------:|
| B1_plus_direction_grid           | kpx_group_2 | mid_12_50pct          |             187 |             155 |          32 |       0.000557844 |
| B1_all_plus_2025_exposure_weight | kpx_group_2 | mid_12_50pct          |             214 |             182 |          32 |       0.000507902 |
| B1_plus_all_score_insights       | kpx_group_2 | mid_12_50pct          |             205 |             177 |          28 |       0.000600462 |
| B1_plus_source_disagreement      | kpx_group_3 | mid_12_50pct          |             149 |             128 |          21 |       0.000894158 |
| B1_plus_direction_grid           | kpx_group_1 | mid_12_50pct          |             137 |             117 |          20 |      -0.00045762  |
| B1_plus_source_disagreement      | kpx_group_2 | mid_12_50pct          |             147 |             129 |          18 |      -0.0015163   |
| B1_plus_direction_grid           | kpx_group_2 | very_high_80pct_plus  |              52 |              36 |          16 |      -0.00224061  |
| B1_all_plus_2025_exposure_weight | kpx_group_3 | high_50_80pct         |             102 |              89 |          13 |       0.000279652 |
| B1_plus_lead_interactions        | kpx_group_3 | very_high_80pct_plus  |              19 |               7 |          12 |      -0.00345572  |
| B1_plus_direction_grid           | kpx_group_3 | mid_12_50pct          |             185 |             173 |          12 |       0.0015542   |
| B1_plus_lead_interactions        | kpx_group_3 | mid_12_50pct          |             123 |             112 |          11 |       0.000348456 |
| B1_plus_direction_grid           | kpx_group_3 | high_50_80pct         |              91 |              81 |          10 |       0.000370522 |
| B1_plus_all_score_insights       | kpx_group_3 | mid_12_50pct          |             177 |             170 |           7 |       0.00179469  |
| B1_plus_lead_interactions        | kpx_group_1 | very_high_80pct_plus  |              34 |              27 |           7 |      -0.00057804  |
| B1_plus_direction_grid           | kpx_group_2 | ficr_boundary_8_12pct |              14 |               8 |           6 |      -0.005534    |
| B1_plus_all_score_insights       | kpx_group_1 | ficr_boundary_8_12pct |              14 |              10 |           4 |      -0.0028795   |
| B1_plus_all_score_insights       | kpx_group_2 | ficr_boundary_8_12pct |              19 |              16 |           3 |      -0.00401842  |
| B1_plus_source_disagreement      | kpx_group_2 | ficr_boundary_8_12pct |              15 |              12 |           3 |      -0.00253766  |
| B1_plus_lead_interactions        | kpx_group_1 | high_50_80pct         |              74 |              71 |           3 |       0.00171577  |
| B1_all_plus_2025_exposure_weight | kpx_group_3 | mid_12_50pct          |             180 |             177 |           3 |       0.00284789  |

해석:

- FiCR boundary에서 fail-to-pass와 pass-to-fail이 동시에 발생합니다.
- score를 올리려면 feature만 추가하는 것보다 boundary-specific calibration/postprocessing을 별도 검증해야 합니다.

## 2025 Exposure Month Delta

| experiment                       | target      |   month |   rows |   baseline_abs_error |   candidate_abs_error |   delta_abs_error |   impact_delta |
|:---------------------------------|:------------|--------:|-------:|---------------------:|----------------------:|------------------:|---------------:|
| B1_plus_all_score_insights       | kpx_group_2 |       2 |    262 |             0.143009 |              0.132521 |       -0.0104879  |       -2.74783 |
| B1_plus_lead_interactions        | kpx_group_3 |       5 |    485 |             0.146599 |              0.142585 |       -0.00401477 |       -1.94717 |
| B1_plus_direction_grid           | kpx_group_2 |       2 |    262 |             0.143009 |              0.135605 |       -0.00740383 |       -1.9398  |
| B1_plus_source_disagreement      | kpx_group_3 |       5 |    485 |             0.146599 |              0.142774 |       -0.0038256  |       -1.85542 |
| B1_plus_source_disagreement      | kpx_group_3 |       1 |    519 |             0.158466 |              0.154918 |       -0.0035487  |       -1.84178 |
| B1_plus_all_score_insights       | kpx_group_3 |       1 |    519 |             0.158466 |              0.154965 |       -0.00350144 |       -1.81725 |
| B1_plus_direction_grid           | kpx_group_3 |       5 |    485 |             0.146599 |              0.142858 |       -0.00374109 |       -1.81443 |
| B1_plus_lead_interactions        | kpx_group_3 |       1 |    519 |             0.158466 |              0.155041 |       -0.00342503 |       -1.77759 |
| B1_plus_all_score_insights       | kpx_group_3 |       2 |    256 |             0.164454 |              0.157797 |       -0.00665684 |       -1.70415 |
| B1_plus_direction_grid           | kpx_group_3 |       1 |    519 |             0.158466 |              0.155424 |       -0.00304192 |       -1.57876 |
| B1_all_plus_2025_exposure_weight | kpx_group_2 |      12 |    686 |             0.121881 |              0.119625 |       -0.00225583 |       -1.5475  |
| B1_plus_all_score_insights       | kpx_group_3 |       7 |    515 |             0.166453 |              0.163554 |       -0.00289937 |       -1.49318 |
| B1_plus_lead_interactions        | kpx_group_3 |       8 |    199 |             0.145143 |              0.138461 |       -0.00668151 |       -1.32962 |
| B1_plus_source_disagreement      | kpx_group_3 |       2 |    256 |             0.164454 |              0.159261 |       -0.00519231 |       -1.32923 |
| B1_plus_direction_grid           | kpx_group_2 |       9 |    303 |             0.107577 |              0.103522 |       -0.00405447 |       -1.22851 |
| B1_all_plus_2025_exposure_weight | kpx_group_2 |      10 |    315 |             0.123056 |              0.119228 |       -0.00382781 |       -1.20576 |
| B1_plus_source_disagreement      | kpx_group_2 |      10 |    315 |             0.123056 |              0.119426 |       -0.00363016 |       -1.1435  |
| B1_plus_direction_grid           | kpx_group_3 |       2 |    256 |             0.164454 |              0.160089 |       -0.00436471 |       -1.11737 |
| B1_all_plus_2025_exposure_weight | kpx_group_3 |       5 |    485 |             0.146599 |              0.144302 |       -0.00229725 |       -1.11416 |
| B1_plus_direction_grid           | kpx_group_2 |      10 |    315 |             0.123056 |              0.119572 |       -0.00348471 |       -1.09768 |

해석:

- exposure-weighted training은 month별 risk를 반영했지만, 현재 protocol에서는 전체 score 개선으로 연결되지 않았습니다.
- 2025 exposure는 model selection/reporting weight로는 유용하지만, naive sample weighting은 충분하지 않습니다.

## Can NWP/time/metadata reproduce SCADA score-risk slices?

| target      |   roc_auc |   average_precision |   positive_rate_valid |   n_valid |
|:------------|----------:|--------------------:|----------------------:|----------:|
| kpx_group_1 |  0.785876 |           0.0831078 |             0.0234517 |      8784 |
| kpx_group_2 |  0.794812 |           0.158024  |             0.0513434 |      8784 |
| kpx_group_3 |  0.853519 |           0.255823  |             0.0492942 |      8784 |

해석:

- SCADA-derived risk slice를 NWP/time/metadata로 분류할 수 있는지 2023 train -> 2024 valid classifier로 봤습니다.
- AUC가 0.5 근처면 test-safe proxy가 약하다는 뜻이고, 높으면 SCADA risk를 NWP feature로 일부 재현할 수 있다는 뜻입니다.
- 이 값이 낮은 group에서는 SCADA taxonomy를 모델 feature로 전환하는 전략이 위험합니다.

## Conclusion

Issue #32에서 얻은 score-relevant 축은 유효했습니다. 그러나 이번 ablation의 결론은 더 엄격합니다.

1. `lead_hour` interaction만 미세하게 baseline을 이겼고, 나머지 feature family는 현재 형태로 baseline을 이기지 못했습니다.
2. 따라서 score 개선 insight는 “단순 feature 추가”가 아니라 **slice-aware modeling/reporting/calibration**으로 사용해야 합니다.
3. 다음으로 할 일은:
   - label-regime별 calibration
   - FiCR boundary-specific postprocessing
   - group3 별도 모델 정책
   - stronger model family 또는 ensemble에서 같은 feature family 재검증
   - direction-grid feature를 hard sector 선택이 아니라 soft upstream weighting으로 재설계

즉 #32의 insight는 폐기할 것이 아니라, 현재 실험에서 “feature dump로는 부족하다”는 것이 증명된 상태입니다.

# Forecast-Time Score Ablation Results

이 실험은 #34에서 제안한 score 개선 가설을 실제 W4 validation score로 검증한 결과입니다.

Protocol:

- Train: 2022-2023 labels
- Validation: 2024 labels
- Metric: repo official local proxy, `0.5 * (1 - avg NMAE) + 0.5 * FiCR`
- Model: fixed LightGBM L1 settings, no HPO
- Seeds: `[11, 42, 77]`
- Baseline: `B1_aggregate`

![summary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_score_ablation_20260809/results/figures/01_forecast_time_ablation_summary.png)

## Main Result

| experiment                      |   score_mean |   score_std |   delta_score_vs_baseline |   delta_score_min |   delta_score_max |   ficr_mean |   avg_nmae_mean |   worst_month_mean |   high_generation_mean |   feature_count |
|:--------------------------------|-------------:|------------:|--------------------------:|------------------:|------------------:|------------:|----------------:|-------------------:|-----------------------:|----------------:|
| lead_basic                      |     0.611644 | 0.000302525 |               7.73901e-05 |      -0.000703057 |       0.00134934  |    0.353359 |        0.130072 |           0.577539 |               0.59988  |             291 |
| B1_aggregate                    |     0.611566 | 0.00120687  |               0           |       0           |       0           |    0.353022 |        0.129889 |           0.578534 |               0.599186 |             286 |
| lead_interactions               |     0.610611 | 0.000591898 |              -0.000955812 |      -0.00180459  |      -0.000336109 |    0.351392 |        0.130171 |           0.575032 |               0.597922 |             296 |
| source_disagreement             |     0.610458 | 0.000447829 |              -0.00110861  |      -0.0017407   |       0.000154322 |    0.351279 |        0.130363 |           0.584139 |               0.598359 |             289 |
| forecast_safe_ramp              |     0.609371 | 0.00092319  |              -0.00219569  |      -0.00248134  |      -0.0017789   |    0.349232 |        0.13049  |           0.574156 |               0.596689 |             290 |
| issue_cycle                     |     0.607094 | 0.000534231 |              -0.0044725   |      -0.00549931  |      -0.00374227  |    0.345589 |        0.131401 |           0.579126 |               0.595033 |             302 |
| all_forecast_time_features      |     0.605927 | 0.00043147  |              -0.00563974  |      -0.00644209  |      -0.00414069  |    0.343648 |        0.131795 |           0.575089 |               0.593325 |             319 |
| all_plus_2025_lead_month_weight |     0.604356 | 0.00156965  |              -0.00721054  |      -0.00874227  |      -0.00417365  |    0.340917 |        0.132205 |           0.563181 |               0.591005 |             319 |

Interpretation:

- Best mean score: `lead_basic` = `0.611643719`.
- Baseline mean score: `B1_aggregate` = `0.611566329`.
- Best delta vs baseline: `0.000077390`.
- Materiality 기준을 `+0.001` 이상으로 보면, 이번 forecast-time feature family 중 **확신 있게 score를 올린 후보는 없습니다**.

## Raw Seed Results

| experiment                      |   seed |    score |   one_minus_nmae |   avg_nmae |     ficr |   worst_month |   high_generation_score |   eligible_hours |   feature_count |   baseline_score |   baseline_ficr |   baseline_avg_nmae |   delta_score_vs_baseline |   delta_ficr_vs_baseline |   delta_nmae_vs_baseline |
|:--------------------------------|-------:|---------:|-----------------:|-----------:|---------:|--------------:|------------------------:|-----------------:|----------------:|-----------------:|----------------:|--------------------:|--------------------------:|-------------------------:|-------------------------:|
| B1_aggregate                    |     11 | 0.61266  |         0.870042 |   0.129958 | 0.355278 |      0.581028 |                0.599797 |            14531 |             286 |         0.61266  |        0.355278 |            0.129958 |               0           |              0           |              0           |
| B1_aggregate                    |     42 | 0.611767 |         0.870029 |   0.129971 | 0.353506 |      0.591715 |                0.600197 |            14531 |             286 |         0.611767 |        0.353506 |            0.129971 |               0           |              0           |              0           |
| B1_aggregate                    |     77 | 0.610272 |         0.87026  |   0.12974  | 0.350283 |      0.562861 |                0.597565 |            14531 |             286 |         0.610272 |        0.350283 |            0.12974  |               0           |              0           |              0           |
| all_forecast_time_features      |     11 | 0.606218 |         0.868386 |   0.131614 | 0.344049 |      0.575157 |                0.593454 |            14531 |             319 |         0.61266  |        0.355278 |            0.129958 |              -0.00644209  |             -0.0112282   |              0.00165598  |
| all_forecast_time_features      |     42 | 0.605431 |         0.868216 |   0.131784 | 0.342646 |      0.579193 |                0.593183 |            14531 |             319 |         0.611767 |        0.353506 |            0.129971 |              -0.00633644  |             -0.0108598   |              0.0018131   |
| all_forecast_time_features      |     77 | 0.606131 |         0.868014 |   0.131986 | 0.344248 |      0.570918 |                0.593337 |            14531 |             319 |         0.610272 |        0.350283 |            0.12974  |              -0.00414069  |             -0.00603471  |              0.00224668  |
| all_plus_2025_lead_month_weight |     11 | 0.603918 |         0.867941 |   0.132059 | 0.339895 |      0.565727 |                0.591247 |            14531 |             319 |         0.61266  |        0.355278 |            0.129958 |              -0.00874227  |             -0.0153829   |              0.00210158  |
| all_plus_2025_lead_month_weight |     42 | 0.603052 |         0.867345 |   0.132655 | 0.338758 |      0.554015 |                0.588955 |            14531 |             319 |         0.611767 |        0.353506 |            0.129971 |              -0.0087157   |             -0.0147476   |              0.00268378  |
| all_plus_2025_lead_month_weight |     77 | 0.606098 |         0.868099 |   0.131901 | 0.344096 |      0.5698   |                0.592813 |            14531 |             319 |         0.610272 |        0.350283 |            0.12974  |              -0.00417365  |             -0.00618623  |              0.00216108  |
| forecast_safe_ramp              |     11 | 0.610333 |         0.869654 |   0.130346 | 0.351013 |      0.572949 |                0.598001 |            14531 |             290 |         0.61266  |        0.355278 |            0.129958 |              -0.00232683  |             -0.0042648   |              0.000388867 |
| forecast_safe_ramp              |     42 | 0.609286 |         0.869618 |   0.130382 | 0.348954 |      0.578569 |                0.596162 |            14531 |             290 |         0.611767 |        0.353506 |            0.129971 |              -0.00248134  |             -0.00455181  |              0.00041087  |
| forecast_safe_ramp              |     77 | 0.608493 |         0.869258 |   0.130742 | 0.347728 |      0.57095  |                0.595903 |            14531 |             290 |         0.610272 |        0.350283 |            0.12974  |              -0.0017789   |             -0.00255514  |              0.00100266  |
| issue_cycle                     |     11 | 0.607161 |         0.868572 |   0.131428 | 0.34575  |      0.586477 |                0.594966 |            14531 |             302 |         0.61266  |        0.355278 |            0.129958 |              -0.00549931  |             -0.00952779  |              0.00147084  |
| issue_cycle                     |     42 | 0.607591 |         0.868843 |   0.131157 | 0.34634  |      0.58011  |                0.595063 |            14531 |             302 |         0.611767 |        0.353506 |            0.129971 |              -0.0041759   |             -0.00716643  |              0.00118538  |
| issue_cycle                     |     77 | 0.606529 |         0.868382 |   0.131618 | 0.344676 |      0.570792 |                0.59507  |            14531 |             302 |         0.610272 |        0.350283 |            0.12974  |              -0.00374227  |             -0.0056062   |              0.00187835  |
| lead_basic                      |     11 | 0.611957 |         0.8701   |   0.1299   | 0.353814 |      0.583046 |                0.599916 |            14531 |             291 |         0.61266  |        0.355278 |            0.129958 |              -0.000703057 |             -0.00146352  |             -5.74094e-05 |
| lead_basic                      |     42 | 0.611353 |         0.869898 |   0.130102 | 0.352809 |      0.571409 |                0.59986  |            14531 |             291 |         0.611767 |        0.353506 |            0.129971 |              -0.000414111 |             -0.000697138 |              0.000131083 |
| lead_basic                      |     77 | 0.611621 |         0.869787 |   0.130213 | 0.353455 |      0.578161 |                0.599862 |            14531 |             291 |         0.610272 |        0.350283 |            0.12974  |               0.00134934  |              0.00317219  |              0.000473511 |
| lead_interactions               |     11 | 0.610855 |         0.870005 |   0.129995 | 0.351706 |      0.578337 |                0.598592 |            14531 |             296 |         0.61266  |        0.355278 |            0.129958 |              -0.00180459  |             -0.00357151  |              3.76661e-05 |
| lead_interactions               |     42 | 0.611041 |         0.869737 |   0.130263 | 0.352344 |      0.580673 |                0.598934 |            14531 |             296 |         0.611767 |        0.353506 |            0.129971 |              -0.000726742 |             -0.00116162  |              0.000291862 |
| lead_interactions               |     77 | 0.609935 |         0.869746 |   0.130254 | 0.350125 |      0.566086 |                0.59624  |            14531 |             296 |         0.610272 |        0.350283 |            0.12974  |              -0.000336109 |             -0.000157443 |              0.000514776 |
| source_disagreement             |     11 | 0.610921 |         0.869877 |   0.130123 | 0.351964 |      0.585508 |                0.598426 |            14531 |             289 |         0.61266  |        0.355278 |            0.129958 |              -0.00173945  |             -0.00331327  |              0.00016563  |
| source_disagreement             |     42 | 0.610027 |         0.869261 |   0.130739 | 0.350792 |      0.583538 |                0.598075 |            14531 |             289 |         0.611767 |        0.353506 |            0.129971 |              -0.0017407   |             -0.0027138   |              0.000767611 |
| source_disagreement             |     77 | 0.610426 |         0.869772 |   0.130228 | 0.35108  |      0.583372 |                0.598576 |            14531 |             289 |         0.610272 |        0.350283 |            0.12974  |               0.000154322 |              0.000797207 |              0.000488562 |

## Best Improvements by Regime

| experiment                      | target      | label_regime         |   rows |   delta_abs_error |   impact_delta |   net_pass8 |
|:--------------------------------|:------------|:---------------------|-------:|------------------:|---------------:|------------:|
| source_disagreement             | kpx_group_3 | high_50_80pct        |   1353 |      -0.00283161  |       -3.83116 |           6 |
| lead_basic                      | kpx_group_1 | high_50_80pct        |   1611 |      -0.00236053  |       -3.80282 |          45 |
| lead_interactions               | kpx_group_1 | high_50_80pct        |   1611 |      -0.00198983  |       -3.20562 |          36 |
| lead_basic                      | kpx_group_3 | high_50_80pct        |   1353 |      -0.00221876  |       -3.00198 |          -4 |
| all_plus_2025_lead_month_weight | kpx_group_1 | high_50_80pct        |   1611 |      -0.00178036  |       -2.86817 |          24 |
| lead_basic                      | kpx_group_3 | high_50_80pct        |   1353 |      -0.0021099   |       -2.8547  |          46 |
| lead_interactions               | kpx_group_2 | mid_12_50pct         |   2215 |      -0.00118606  |       -2.62713 |           9 |
| lead_interactions               | kpx_group_3 | very_high_80pct_plus |    712 |      -0.00361798  |       -2.576   |           5 |
| forecast_safe_ramp              | kpx_group_3 | high_50_80pct        |   1353 |      -0.00172091  |       -2.32839 |          -7 |
| lead_basic                      | kpx_group_1 | very_high_80pct_plus |    840 |      -0.00269385  |       -2.26284 |           5 |
| forecast_safe_ramp              | kpx_group_1 | high_50_80pct        |   1611 |      -0.00131268  |       -2.11473 |          22 |
| forecast_safe_ramp              | kpx_group_3 | very_high_80pct_plus |    712 |      -0.00291303  |       -2.07408 |           6 |
| source_disagreement             | kpx_group_3 | very_high_80pct_plus |    712 |      -0.00264117  |       -1.88051 |          -2 |
| lead_interactions               | kpx_group_2 | mid_12_50pct         |   2215 |      -0.000836107 |       -1.85198 |          16 |
| lead_interactions               | kpx_group_3 | very_high_80pct_plus |    712 |      -0.00249271  |       -1.77481 |           1 |
| all_forecast_time_features      | kpx_group_3 | high_50_80pct        |   1353 |      -0.00124235  |       -1.6809  |          -5 |
| source_disagreement             | kpx_group_3 | high_50_80pct        |   1353 |      -0.00122127  |       -1.65238 |           3 |
| source_disagreement             | kpx_group_3 | very_high_80pct_plus |    712 |      -0.00230715  |       -1.64269 |          -5 |
| all_forecast_time_features      | kpx_group_2 | very_high_80pct_plus |    953 |      -0.00168914  |       -1.60975 |          -1 |
| forecast_safe_ramp              | kpx_group_2 | mid_12_50pct         |   2215 |      -0.000681859 |       -1.51032 |          20 |

## Worst Regressions by Regime

| experiment                      | target      | label_regime         |   rows |   delta_abs_error |   impact_delta |   net_pass8 |
|:--------------------------------|:------------|:---------------------|-------:|------------------:|---------------:|------------:|
| all_plus_2025_lead_month_weight | kpx_group_1 | high_50_80pct        |   1611 |        0.00609206 |        9.81431 |         -28 |
| issue_cycle                     | kpx_group_3 | mid_12_50pct         |   2289 |        0.00362875 |        8.30621 |          -7 |
| all_forecast_time_features      | kpx_group_1 | mid_12_50pct         |   2319 |        0.00344012 |        7.97765 |         -44 |
| all_forecast_time_features      | kpx_group_1 | high_50_80pct        |   1611 |        0.00471862 |        7.6017  |         -29 |
| all_forecast_time_features      | kpx_group_1 | mid_12_50pct         |   2319 |        0.00322886 |        7.48772 |         -56 |
| issue_cycle                     | kpx_group_1 | high_50_80pct        |   1611 |        0.00463261 |        7.46314 |         -25 |
| all_plus_2025_lead_month_weight | kpx_group_1 | very_high_80pct_plus |    840 |        0.00848925 |        7.13097 |         -56 |
| all_forecast_time_features      | kpx_group_1 | mid_12_50pct         |   2319 |        0.00297389 |        6.89646 |         -36 |
| all_plus_2025_lead_month_weight | kpx_group_1 | mid_12_50pct         |   2319 |        0.00293841 |        6.81416 |         -41 |
| all_plus_2025_lead_month_weight | kpx_group_2 | very_high_80pct_plus |    953 |        0.00671189 |        6.39643 |         -56 |
| all_plus_2025_lead_month_weight | kpx_group_1 | mid_12_50pct         |   2319 |        0.00272939 |        6.32946 |         -64 |
| issue_cycle                     | kpx_group_2 | high_50_80pct        |   1584 |        0.00360649 |        5.71268 |         -24 |
| all_forecast_time_features      | kpx_group_2 | mid_12_50pct         |   2215 |        0.00256204 |        5.67492 |         -26 |
| lead_interactions               | kpx_group_1 | high_50_80pct        |   1611 |        0.00344912 |        5.55654 |         -23 |
| issue_cycle                     | kpx_group_1 | mid_12_50pct         |   2319 |        0.00239525 |        5.5546  |         -39 |
| source_disagreement             | kpx_group_1 | high_50_80pct        |   1611 |        0.00336438 |        5.42002 |         -13 |
| all_plus_2025_lead_month_weight | kpx_group_1 | mid_12_50pct         |   2319 |        0.00227256 |        5.27006 |         -42 |
| all_forecast_time_features      | kpx_group_2 | high_50_80pct        |   1584 |        0.00329383 |        5.21742 |         -19 |
| all_plus_2025_lead_month_weight | kpx_group_3 | very_high_80pct_plus |    712 |        0.00722295 |        5.14274 |          -3 |
| issue_cycle                     | kpx_group_1 | very_high_80pct_plus |    840 |        0.00590508 |        4.96027 |         -47 |

## Month-Lead Cells Where Candidates Helped Most

| experiment                      |   month |   lead_hour |   rows |   delta_abs_error |   impact_delta |   net_pass8 |
|:--------------------------------|--------:|------------:|-------:|------------------:|---------------:|------------:|
| all_forecast_time_features      |       3 |          35 |     79 |       -0.0154249  |      -1.21857  |           7 |
| issue_cycle                     |       3 |          35 |     79 |       -0.0145319  |      -1.14802  |          10 |
| all_plus_2025_lead_month_weight |       3 |          35 |     79 |       -0.0143569  |      -1.1342   |           8 |
| all_forecast_time_features      |       7 |          17 |     68 |       -0.0144158  |      -0.980276 |           0 |
| issue_cycle                     |       3 |          28 |     58 |       -0.0167626  |      -0.972229 |           7 |
| issue_cycle                     |       7 |          18 |     69 |       -0.013957   |      -0.963031 |           6 |
| all_plus_2025_lead_month_weight |       3 |          35 |     79 |       -0.0121686  |      -0.96132  |           4 |
| all_forecast_time_features      |       1 |          18 |     61 |       -0.0150069  |      -0.915421 |           5 |
| all_forecast_time_features      |       7 |          18 |     69 |       -0.0130786  |      -0.902426 |           5 |
| all_forecast_time_features      |       3 |          32 |     73 |       -0.0118727  |      -0.866708 |           4 |
| all_forecast_time_features      |      12 |          12 |     90 |       -0.00935677 |      -0.842109 |           1 |
| all_plus_2025_lead_month_weight |      11 |          13 |     45 |       -0.0185219  |      -0.833484 |           4 |
| issue_cycle                     |       7 |          18 |     69 |       -0.0119782  |      -0.826498 |           2 |
| issue_cycle                     |       2 |          24 |     36 |       -0.0225421  |      -0.811516 |           5 |
| all_forecast_time_features      |       3 |          35 |     79 |       -0.010174   |      -0.803748 |           1 |
| issue_cycle                     |       7 |          17 |     68 |       -0.0117779  |      -0.800897 |           0 |
| issue_cycle                     |       3 |          25 |     63 |       -0.0127074  |      -0.800566 |           1 |
| issue_cycle                     |       3 |          25 |     63 |       -0.0125385  |      -0.789928 |           4 |
| all_plus_2025_lead_month_weight |      12 |          18 |     89 |       -0.00886622 |      -0.789094 |          -1 |
| all_forecast_time_features      |       7 |          19 |     69 |       -0.0113894  |      -0.785869 |           4 |

## Conclusion

What is proven:

1. `lead_basic` / `lead_interactions` can produce tiny local changes, but the gain is below materiality.
2. `issue_cycle`, `ramp`, and `source_disagreement` contain slice-level signal, but simple feature addition creates offsetting regressions.
3. `2025 lead-month exposure weighting` is not a reliable training trick in this fixed-LGBM protocol.
4. The EDA insight is still useful as a diagnostic/reporting axis, but not yet as a deployable score-raising feature family.

Decision:

- Do not promote a new submission from these experiments.
- Keep `lead_hour`, issue-cycle, ramp, and exposure as validation slices and possible calibration axes.
- Next experiment should be **postprocessing/calibration constrained to proven-positive slices**, not another broad feature dump.

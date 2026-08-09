# Spatial topology score-insight EDA: does direction/upstream geometry explain anchor residual and FiCR risk?

## Why this issue exists

Issue #31 clarified that spatial topology should not be read as “nearest grid wins.” The remaining question is whether the spatial signal is merely descriptive, or whether it can become a score-improving modeling lever.

This issue tests that explicitly against the current W4 anchor:

- anchor panel: `phase98_w4_phase60_lineage + Lower16`
- anchor W4 score: `0.6457168408`
- objective: determine whether `nearest`, `static corr-top`, `direction-conditioned top`, or `soft upstream` spatial features explain **current-anchor residual**, FiCR boundary risk, and 2025 exposure.

No model is promoted here. This is EDA for deciding whether spatial feature engineering is worth a controlled modeling phase.

## Figures

![Spatial residual signal summary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/spatial_score_insight_eda_20260810/results/figures/01_spatial_residual_signal_summary.png)

![Spatial stability and FiCR boundary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/spatial_score_insight_eda_20260810/results/figures/02_spatial_stability_and_ficr_boundary.png)

## 1. Label signal is real, but residual signal is much weaker

Top label correlations:

| target      | feature_family            |   speed_actual_corr |   speed_abs_residual_corr |   pass8_rate |   selected_grid_unique |
|:------------|:--------------------------|--------------------:|--------------------------:|-------------:|-----------------------:|
| kpx_group_1 | static_corr_top           |            0.749106 |                 0.0846396 |     0.485468 |                      1 |
| kpx_group_1 | direction_conditional_top |            0.725461 |                 0.0559908 |     0.485468 |                      6 |
| kpx_group_2 | static_corr_top           |            0.722028 |                 0.0422722 |     0.482918 |                      1 |
| kpx_group_2 | direction_conditional_top |            0.719138 |                 0.0388094 |     0.482918 |                      4 |
| kpx_group_3 | static_corr_top           |            0.714899 |                 0.282811  |     0.404512 |                      1 |
| kpx_group_3 | nearest_modal             |            0.699943 |                 0.278592  |     0.404512 |                      1 |
| kpx_group_3 | direction_conditional_top |            0.693502 |                 0.267969  |     0.404512 |                      6 |
| kpx_group_2 | nearest_modal             |            0.662618 |                 0.0442846 |     0.482918 |                      1 |
| kpx_group_3 | soft_upstream             |            0.646934 |                 0.249765  |     0.404512 |                      0 |
| kpx_group_2 | soft_upstream             |            0.639503 |                 0.0414447 |     0.482918 |                      0 |
| kpx_group_1 | soft_upstream             |            0.615608 |                 0.0536556 |     0.485468 |                      0 |
| kpx_group_1 | nearest_modal             |            0.508579 |                 0.016609  |     0.485468 |                      1 |

Top residual associations:

| target      | feature_family            |   speed_actual_corr |   speed_abs_residual_corr |   speed_signed_residual_corr |   pass8_rate |   selected_grid_unique |
|:------------|:--------------------------|--------------------:|--------------------------:|-----------------------------:|-------------:|-----------------------:|
| kpx_group_3 | static_corr_top           |            0.714899 |                 0.282811  |                  0.000726554 |     0.404512 |                      1 |
| kpx_group_3 | nearest_modal             |            0.699943 |                 0.278592  |                 -0.00172093  |     0.404512 |                      1 |
| kpx_group_3 | direction_conditional_top |            0.693502 |                 0.267969  |                 -0.00850977  |     0.404512 |                      6 |
| kpx_group_3 | soft_upstream             |            0.646934 |                 0.249765  |                 -0.00329614  |     0.404512 |                      0 |
| kpx_group_1 | static_corr_top           |            0.749106 |                 0.0846396 |                  0.13988     |     0.485468 |                      1 |
| kpx_group_1 | direction_conditional_top |            0.725461 |                 0.0559908 |                  0.121033    |     0.485468 |                      6 |
| kpx_group_1 | soft_upstream             |            0.615608 |                 0.0536556 |                  0.112841    |     0.485468 |                      0 |
| kpx_group_2 | nearest_modal             |            0.662618 |                 0.0442846 |                  0.170887    |     0.482918 |                      1 |
| kpx_group_2 | static_corr_top           |            0.722028 |                 0.0422722 |                  0.198034    |     0.482918 |                      1 |
| kpx_group_2 | soft_upstream             |            0.639503 |                 0.0414447 |                  0.164244    |     0.482918 |                      0 |
| kpx_group_2 | direction_conditional_top |            0.719138 |                 0.0388094 |                  0.188083    |     0.482918 |                      4 |
| kpx_group_1 | nearest_modal             |            0.508579 |                 0.016609  |                  0.0885542   |     0.485468 |                      1 |

Interpretation:

- Spatial speed features strongly explain the label surface, especially for `wind50max`-based LDAPS features.
- But the correlation with **anchor absolute residual** is much smaller. This means the current anchor already uses much of the basic wind-speed signal.
- A spatial modeling phase should therefore target residual slices, not re-learn generic wind-to-power mapping.

## 2. Nearest, static corr-top, direction-top, and soft-upstream are different objects

Summary by feature family:

| target      | feature_family            |   rows |   speed_actual_corr |   speed_abs_residual_corr |   speed_signed_residual_corr |   pass8_rate |   selected_grid_unique |
|:------------|:--------------------------|-------:|--------------------:|--------------------------:|-----------------------------:|-------------:|-----------------------:|
| kpx_group_1 | direction_conditional_top |   4989 |            0.725461 |                 0.0559908 |                  0.121033    |     0.485468 |                      6 |
| kpx_group_1 | nearest_modal             |   4989 |            0.508579 |                 0.016609  |                  0.0885542   |     0.485468 |                      1 |
| kpx_group_1 | soft_upstream             |   4989 |            0.615608 |                 0.0536556 |                  0.112841    |     0.485468 |                      0 |
| kpx_group_1 | static_corr_top           |   4989 |            0.749106 |                 0.0846396 |                  0.13988     |     0.485468 |                      1 |
| kpx_group_2 | direction_conditional_top |   4976 |            0.719138 |                 0.0388094 |                  0.188083    |     0.482918 |                      4 |
| kpx_group_2 | nearest_modal             |   4976 |            0.662618 |                 0.0442846 |                  0.170887    |     0.482918 |                      1 |
| kpx_group_2 | soft_upstream             |   4976 |            0.639503 |                 0.0414447 |                  0.164244    |     0.482918 |                      0 |
| kpx_group_2 | static_corr_top           |   4976 |            0.722028 |                 0.0422722 |                  0.198034    |     0.482918 |                      1 |
| kpx_group_3 | direction_conditional_top |   4566 |            0.693502 |                 0.267969  |                 -0.00850977  |     0.404512 |                      6 |
| kpx_group_3 | nearest_modal             |   4566 |            0.699943 |                 0.278592  |                 -0.00172093  |     0.404512 |                      1 |
| kpx_group_3 | soft_upstream             |   4566 |            0.646934 |                 0.249765  |                 -0.00329614  |     0.404512 |                      0 |
| kpx_group_3 | static_corr_top           |   4566 |            0.714899 |                 0.282811  |                  0.000726554 |     0.404512 |                      1 |

Interpretation:

- `nearest_modal` is only a geometry prior.
- `static_corr_top` often has strong label correlation but still does not guarantee residual leverage.
- `direction_conditional_top` keeps the direction-dependent grid identity visible.
- `soft_upstream` is physically smoother but may dilute the sharp grid identity that correlation selected.

## 3. Decile profile: where residual risk concentrates

The decile profile table records how anchor error and FiCR pass rates change across spatial-speed deciles.

| target      | feature_family            |   speed_decile |   rows |   speed_mean |   actual_ratio_mean |   abs_error_ratio_mean |   signed_error_ratio_mean |   pass8_rate |
|:------------|:--------------------------|---------------:|-------:|-------------:|--------------------:|-----------------------:|--------------------------:|-------------:|
| kpx_group_1 | direction_conditional_top |              0 |    499 |      4.11681 |            0.234944 |              0.0869713 |              -0.0332662   |     0.683367 |
| kpx_group_1 | direction_conditional_top |              1 |    499 |      6.32873 |            0.270947 |              0.101641  |              -0.0199234   |     0.577154 |
| kpx_group_1 | direction_conditional_top |              2 |    499 |      7.55163 |            0.308028 |              0.0998128 |               0.000927718 |     0.517034 |
| kpx_group_1 | direction_conditional_top |              3 |    499 |      8.71613 |            0.362444 |              0.120641  |               0.0176829   |     0.410822 |
| kpx_group_1 | direction_conditional_top |              4 |    499 |      9.85891 |            0.444634 |              0.136164  |               0.0258319   |     0.376754 |
| kpx_group_1 | direction_conditional_top |              5 |    498 |     10.9773  |            0.53178  |              0.126967  |               0.0214896   |     0.393574 |
| kpx_group_1 | direction_conditional_top |              6 |    499 |     12.2344  |            0.608694 |              0.134452  |               0.03627     |     0.396794 |
| kpx_group_1 | direction_conditional_top |              7 |    499 |     13.7783  |            0.66536  |              0.127192  |               0.0499702   |     0.42485  |
| kpx_group_1 | direction_conditional_top |              8 |    499 |     15.8168  |            0.740008 |              0.106124  |               0.0455292   |     0.503006 |
| kpx_group_1 | direction_conditional_top |              9 |    499 |     19.7002  |            0.806235 |              0.0901929 |               0.0317115   |     0.571142 |
| kpx_group_1 | soft_upstream             |              0 |    499 |      3.54282 |            0.232514 |              0.0851116 |              -0.0360614   |     0.699399 |
| kpx_group_1 | soft_upstream             |              1 |    499 |      5.24953 |            0.311671 |              0.109918  |              -0.00871266  |     0.521042 |
| kpx_group_1 | soft_upstream             |              2 |    499 |      6.30792 |            0.356362 |              0.118411  |               0.00202515  |     0.450902 |
| kpx_group_1 | soft_upstream             |              3 |    499 |      7.18471 |            0.40353  |              0.112605  |               0.0184438   |     0.470942 |
| kpx_group_1 | soft_upstream             |              4 |    499 |      8.04992 |            0.495269 |              0.123554  |               0.0270706   |     0.42485  |
| kpx_group_1 | soft_upstream             |              5 |    498 |      9.00613 |            0.513712 |              0.121639  |               0.0327622   |     0.419679 |
| kpx_group_1 | soft_upstream             |              6 |    499 |     10.1359  |            0.587105 |              0.118505  |               0.0115058   |     0.428858 |
| kpx_group_1 | soft_upstream             |              7 |    499 |     11.5715  |            0.61922  |              0.130806  |               0.0403891   |     0.416834 |
| kpx_group_1 | soft_upstream             |              8 |    499 |     13.6887  |            0.685653 |              0.11046   |               0.0492597   |     0.490982 |
| kpx_group_1 | soft_upstream             |              9 |    499 |     17.4853  |            0.768003 |              0.0991382 |               0.0395638   |     0.531062 |
| kpx_group_2 | direction_conditional_top |              0 |    498 |      4.22782 |            0.220033 |              0.0743985 |              -0.0269763   |     0.708835 |
| kpx_group_2 | direction_conditional_top |              1 |    498 |      6.52351 |            0.271109 |              0.0948005 |              -0.0183689   |     0.572289 |
| kpx_group_2 | direction_conditional_top |              2 |    497 |      7.7209  |            0.327446 |              0.112524  |              -0.00569997  |     0.492958 |
| kpx_group_2 | direction_conditional_top |              3 |    498 |      8.81864 |            0.398554 |              0.13307   |               0.0102501   |     0.411647 |
| kpx_group_2 | direction_conditional_top |              4 |    497 |      9.98576 |            0.495492 |              0.150966  |               0.0423024   |     0.311871 |
| kpx_group_2 | direction_conditional_top |              5 |    498 |     11.161   |            0.562544 |              0.14891   |               0.0577794   |     0.34739  |
| kpx_group_2 | direction_conditional_top |              6 |    497 |     12.4733  |            0.644641 |              0.144416  |               0.0564995   |     0.380282 |
| kpx_group_2 | direction_conditional_top |              7 |    498 |     14.0339  |            0.684978 |              0.128251  |               0.0709498   |     0.447791 |
| kpx_group_2 | direction_conditional_top |              8 |    497 |     16.0011  |            0.727266 |              0.12724   |               0.0880175   |     0.509054 |
| kpx_group_2 | direction_conditional_top |              9 |    498 |     19.7253  |            0.801177 |              0.0880111 |               0.052956    |     0.646586 |

Interpretation:

- Spatial-speed deciles are useful as diagnostic strata.
- However, a monotonic residual correction is not automatically justified. Some deciles carry signed overprediction, others underprediction, and group behavior differs.
- Any correction has to be cross-fitted and group-specific; otherwise it becomes the same local-offset failure mode seen in prior phases.

## 4. Stability: label correlation survives better than residual correlation

Year-level label stability:

| target      | feature_family            |   years |   min_corr |   max_corr |
|:------------|:--------------------------|--------:|-----------:|-----------:|
| kpx_group_1 | direction_conditional_top |       3 |   0.658168 |   0.725461 |
| kpx_group_1 | soft_upstream             |       3 |   0.538878 |   0.615608 |
| kpx_group_2 | direction_conditional_top |       3 |   0.680767 |   0.719138 |
| kpx_group_2 | soft_upstream             |       3 |   0.575868 |   0.639503 |
| kpx_group_3 | direction_conditional_top |       2 |   0.575375 |   0.693502 |
| kpx_group_3 | soft_upstream             |       2 |   0.526684 |   0.646934 |

Monthly residual stability, largest absolute correlations:

| target      | feature_family            |   month |   rows |   speed_abs_residual_corr |   mean_abs_error_ratio |   pass8_rate |
|:------------|:--------------------------|--------:|-------:|--------------------------:|-----------------------:|-------------:|
| kpx_group_3 | static_corr_top           |      11 |    293 |                  0.481185 |              0.150301  |     0.354949 |
| kpx_group_3 | static_corr_top           |      10 |    241 |                  0.448732 |              0.118495  |     0.518672 |
| kpx_group_3 | nearest_modal             |      11 |    293 |                  0.446208 |              0.150301  |     0.354949 |
| kpx_group_3 | nearest_modal             |      10 |    241 |                  0.441737 |              0.118495  |     0.518672 |
| kpx_group_3 | direction_conditional_top |      11 |    293 |                  0.439375 |              0.150301  |     0.354949 |
| kpx_group_3 | soft_upstream             |      11 |    293 |                  0.424225 |              0.150301  |     0.354949 |
| kpx_group_3 | direction_conditional_top |       9 |    249 |                  0.41853  |              0.0838676 |     0.630522 |
| kpx_group_3 | soft_upstream             |      10 |    241 |                  0.411776 |              0.118495  |     0.518672 |
| kpx_group_3 | nearest_modal             |       9 |    249 |                  0.402933 |              0.0838676 |     0.630522 |
| kpx_group_3 | direction_conditional_top |      10 |    241 |                  0.398564 |              0.118495  |     0.518672 |
| kpx_group_3 | static_corr_top           |       9 |    249 |                  0.397924 |              0.0838676 |     0.630522 |
| kpx_group_3 | soft_upstream             |       9 |    249 |                  0.389947 |              0.0838676 |     0.630522 |
| kpx_group_3 | static_corr_top           |       4 |    356 |                  0.387667 |              0.115777  |     0.494382 |
| kpx_group_1 | static_corr_top           |       6 |    330 |                  0.364847 |              0.127441  |     0.427273 |
| kpx_group_1 | static_corr_top           |       9 |    285 |                  0.361087 |              0.0950939 |     0.589474 |
| kpx_group_1 | direction_conditional_top |       6 |    330 |                  0.355691 |              0.127441  |     0.427273 |
| kpx_group_2 | direction_conditional_top |       6 |    345 |                  0.353243 |              0.124909  |     0.457971 |
| kpx_group_3 | nearest_modal             |       4 |    356 |                  0.344978 |              0.115777  |     0.494382 |
| kpx_group_3 | static_corr_top           |       3 |    496 |                  0.343228 |              0.141741  |     0.370968 |
| kpx_group_2 | static_corr_top           |       6 |    345 |                  0.343049 |              0.124909  |     0.457971 |

Interpretation:

- The label relationship is stable enough to justify spatial features as candidate predictors.
- The residual relationship is month-sensitive. This is the key risk: score improvement requires correcting what the anchor misses, not what the label already expresses.
- A spatial phase must use W3/W4 or month-block validation. W4-only selection is not enough.

## 5. FiCR boundary: spatial regimes can identify risk mass, but not yet a safe correction

High near-loss slices:

| target      | feature_family            |   speed_decile |   rows |   settlement_weight_share |   in8_rate |   near_8_10_rate |   lost_10_15_rate |   mean_signed_error_ratio |   mean_abs_error_ratio |
|:------------|:--------------------------|---------------:|-------:|--------------------------:|-----------:|-----------------:|------------------:|--------------------------:|-----------------------:|
| kpx_group_3 | soft_upstream             |              5 |    457 |                 0.107124  |   0.310722 |        0.111597  |          0.159737 |               0.0245693   |              0.144773  |
| kpx_group_1 | soft_upstream             |              7 |    499 |                 0.124541  |   0.416834 |        0.104208  |          0.140281 |               0.0403891   |              0.130806  |
| kpx_group_1 | nearest_modal             |              7 |    499 |                 0.118888  |   0.436874 |        0.102204  |          0.168337 |               0.0215899   |              0.118267  |
| kpx_group_3 | nearest_modal             |              4 |    456 |                 0.0922355 |   0.289474 |        0.0964912 |          0.199561 |               0.0245295   |              0.146361  |
| kpx_group_1 | direction_conditional_top |              8 |    499 |                 0.148835  |   0.503006 |        0.0961924 |          0.142285 |               0.0455292   |              0.106124  |
| kpx_group_2 | soft_upstream             |              2 |    497 |                 0.0730076 |   0.440644 |        0.0945674 |          0.156942 |               9.00818e-06 |              0.121687  |
| kpx_group_3 | direction_conditional_top |              4 |    456 |                 0.0927901 |   0.304825 |        0.0942982 |          0.199561 |               0.0148194   |              0.141417  |
| kpx_group_1 | direction_conditional_top |              9 |    499 |                 0.162155  |   0.571142 |        0.0941884 |          0.160321 |               0.0317115   |              0.0901929 |
| kpx_group_1 | soft_upstream             |              9 |    499 |                 0.154465  |   0.531062 |        0.0921844 |          0.178357 |               0.0395638   |              0.0991382 |
| kpx_group_1 | static_corr_top           |              9 |    499 |                 0.160875  |   0.56513  |        0.0921844 |          0.166333 |               0.0346798   |              0.0912014 |
| kpx_group_3 | static_corr_top           |              4 |    456 |                 0.0928716 |   0.296053 |        0.0921053 |          0.184211 |               0.0104428   |              0.146594  |
| kpx_group_1 | static_corr_top           |              8 |    499 |                 0.14859   |   0.498998 |        0.0901804 |          0.154309 |               0.0496207   |              0.106216  |
| kpx_group_1 | soft_upstream             |              6 |    499 |                 0.118082  |   0.428858 |        0.0901804 |          0.182365 |               0.0115058   |              0.118505  |
| kpx_group_1 | nearest_modal             |              9 |    499 |                 0.153869  |   0.531062 |        0.0901804 |          0.178357 |               0.038328    |              0.100156  |
| kpx_group_1 | nearest_modal             |              4 |    499 |                 0.0917724 |   0.466934 |        0.0901804 |          0.136273 |               0.0290466   |              0.114803  |
| kpx_group_1 | direction_conditional_top |              6 |    499 |                 0.122424  |   0.396794 |        0.0901804 |          0.168337 |               0.03627     |              0.134452  |
| kpx_group_1 | static_corr_top           |              6 |    499 |                 0.125048  |   0.408818 |        0.0901804 |          0.164329 |               0.0351498   |              0.131998  |
| kpx_group_2 | soft_upstream             |              8 |    497 |                 0.137027  |   0.511066 |        0.0885312 |          0.16499  |               0.0727559   |              0.113692  |
| kpx_group_1 | static_corr_top           |              7 |    499 |                 0.134655  |   0.44489  |        0.0881764 |          0.162325 |               0.0486217   |              0.124493  |
| kpx_group_3 | static_corr_top           |              3 |    457 |                 0.0761887 |   0.350109 |        0.0875274 |          0.190372 |               0.00550559  |              0.137204  |

Interpretation:

- Spatial-speed strata do expose where FiCR near-loss mass sits.
- This is useful because a model phase can target `8-15%` error rows rather than average MAE.
- But EDA alone does not prove a correction direction. The sign of residuals is mixed; blind offsets would risk converting pass rows into fail rows.

## 6. 2025 exposure: the spatial regimes are present, but shifted by month

Largest 2025 spatial-speed shifts:

| target      | feature_family            |   month |   test_rows |   z_delta_vs_train |   test_above_train_p90_rate |   test_below_train_p10_rate |
|:------------|:--------------------------|--------:|------------:|-------------------:|----------------------------:|----------------------------:|
| kpx_group_2 | static_corr_top           |       2 |         672 |           0.81839  |                  0.22619    |                   0.0238095 |
| kpx_group_3 | static_corr_top           |       2 |         672 |           0.81839  |                  0.22619    |                   0.0238095 |
| kpx_group_1 | static_corr_top           |       2 |         672 |           0.81839  |                  0.22619    |                   0.0238095 |
| kpx_group_2 | direction_conditional_top |       2 |         672 |           0.795641 |                  0.223214   |                   0.0178571 |
| kpx_group_3 | nearest_modal             |       2 |         672 |           0.765735 |                  0.200893   |                   0.0282738 |
| kpx_group_1 | direction_conditional_top |       2 |         672 |           0.741145 |                  0.221726   |                   0.016369  |
| kpx_group_3 | direction_conditional_top |       2 |         672 |           0.740655 |                  0.19494    |                   0.016369  |
| kpx_group_3 | soft_upstream             |       2 |         672 |           0.702351 |                  0.19494    |                   0.0223214 |
| kpx_group_2 | soft_upstream             |       2 |         672 |           0.687792 |                  0.197917   |                   0.0208333 |
| kpx_group_2 | nearest_modal             |       2 |         672 |           0.678497 |                  0.183036   |                   0.0238095 |
| kpx_group_1 | soft_upstream             |       2 |         672 |           0.660597 |                  0.203869   |                   0.0178571 |
| kpx_group_3 | static_corr_top           |      12 |         744 |           0.607316 |                  0.200269   |                   0.0416667 |
| kpx_group_2 | static_corr_top           |      12 |         744 |           0.607316 |                  0.200269   |                   0.0416667 |
| kpx_group_1 | static_corr_top           |      12 |         744 |           0.607316 |                  0.200269   |                   0.0416667 |
| kpx_group_3 | nearest_modal             |      12 |         744 |           0.594478 |                  0.176075   |                   0.0416667 |
| kpx_group_2 | direction_conditional_top |      12 |         744 |           0.585934 |                  0.197581   |                   0.0389785 |
| kpx_group_3 | static_corr_top           |      10 |         744 |          -0.578301 |                  0.00672043 |                   0.138441  |
| kpx_group_2 | static_corr_top           |      10 |         744 |          -0.578301 |                  0.00672043 |                   0.138441  |
| kpx_group_1 | static_corr_top           |      10 |         744 |          -0.578301 |                  0.00672043 |                   0.138441  |
| kpx_group_3 | direction_conditional_top |      12 |         744 |           0.563826 |                  0.157258   |                   0.0416667 |
| kpx_group_3 | static_corr_top           |       1 |         744 |           0.562994 |                  0.21371    |                   0.0443548 |
| kpx_group_2 | static_corr_top           |       1 |         744 |           0.562994 |                  0.21371    |                   0.0443548 |
| kpx_group_1 | static_corr_top           |       1 |         744 |           0.562994 |                  0.21371    |                   0.0443548 |
| kpx_group_3 | nearest_modal             |      10 |         744 |          -0.55547  |                  0.0120968  |                   0.134409  |

Interpretation:

- The spatial/wind regimes are present in 2025, but month-level exposure is not identical to train.
- A spatial feature may matter publicly only if the target regime appears in the scored public subset.
- This argues for exposure-aware reporting, not for selecting a month guard from W4.

## Conclusion

Spatial topology gives a real modeling lead, but not yet a submission-ready rule.

What is now supported:

1. `nearest_grid` is insufficient.
2. `direction_conditional_top` and `soft_upstream` are the right feature families to test next.
3. The strongest EDA value is as residual/FICR stratification, not as a direct correction.
4. Any score-improvement claim must prove residual reduction under W3/W4 or month-block validation.

What must be proved in the next modeling phase:

- A direction/upstream spatial feature reduces current-anchor residual, not only label error.
- The reduction is positive on independent time blocks.
- FiCR `fail_to_pass8` gains exceed `pass_to_fail8` losses.
- 2025 exposure includes the same spatial regimes.

If those are not proved, spatial topology remains an explanatory EDA axis rather than a leaderboard lever.

## Artifacts

- `experiments/spatial_score_insight_eda_20260810/run_spatial_score_insight_eda.py`
- `results/spatial_feature_residual_summary.csv`
- `results/spatial_feature_decile_residual_profile.csv`
- `results/spatial_month_stability.csv`
- `results/spatial_ficr_boundary_profile.csv`
- `results/spatial_year_correlation_stability.csv`
- `results/spatial_2025_exposure.csv`
- `results/figures/01_spatial_residual_signal_summary.png`
- `results/figures/02_spatial_stability_and_ficr_boundary.png`

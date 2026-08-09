## Deep-Dive EDA Completion Update — 2026-08-09

This update extends the first atlas with the missing deep-dive EDA: SCADA residual taxonomy, NWP-to-SCADA transfer bias, direction-conditioned spatial signal, label-relevance-weighted 2025 exposure, and best-candidate residual overlay.

### New Figures

12. ![SCADA residual taxonomy](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/12_scada_residual_taxonomy.png)
13. ![SCADA residual monthly heatmap](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/13_scada_residual_monthly_heatmap.png)
14. ![NWP SCADA wind bias monthly](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/14_nwp_scada_wind_bias_monthly.png)
15. ![NWP SCADA wind scatter](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/15_nwp_scada_wind_scatter.png)
16a. ![Directional spatial corr group1](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/16_directional_spatial_corr_kpx_group_1.png)
16b. ![Directional spatial corr group2](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/16_directional_spatial_corr_kpx_group_2.png)
16c. ![Directional spatial corr group3](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/16_directional_spatial_corr_kpx_group_3.png)
17. ![Weighted 2025 exposure](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/17_label_relevance_weighted_2025_exposure.png)
18. ![Top weighted 2025 shifts](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/18_top_weighted_2025_shift_features.png)
19. ![Model residual by SCADA taxonomy](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/19_model_residual_by_scada_taxonomy.png)
20. ![Model FiCR transition overlay](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/20_model_ficr_transition_overlay.png)
21. ![Model residual by NWP SCADA bias](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/21_model_residual_by_nwp_scada_bias.png)

### SCADA Residual Taxonomy

| target      | taxonomy                  |   rows |   mean_abs_residual_ratio |   mean_label_ratio |   mean_scada_ratio |   mean_online_rate |
|:------------|:--------------------------|-------:|--------------------------:|-------------------:|-------------------:|-------------------:|
| kpx_group_1 | tight_reconstruction      |  22693 |                0.00674433 |           0.292824 |         0.294693   |         0.697013   |
| kpx_group_1 | moderate_residual         |   1655 |                0.0635328  |           0.277386 |         0.216961   |         0.793907   |
| kpx_group_1 | scada_under_label         |   1296 |                0.290462   |           0.5322   |         0.241738   |         0.827203   |
| kpx_group_1 | scada_zero_label_positive |    529 |                0.435967   |           0.438027 |         0.00205997 |         0.706102   |
| kpx_group_1 | label_missing             |    104 |              nan          |         nan        |         0.0030471  |         0.00774573 |
| kpx_group_1 | low_online_label_positive |     27 |                0.166122   |           0.246062 |         0.07994    |         0.471193   |
| kpx_group_2 | tight_reconstruction      |  21962 |                0.00640439 |           0.320318 |         0.321434   |         0.702873   |
| kpx_group_2 | moderate_residual         |   2261 |                0.0636726  |           0.240797 |         0.179184   |         0.781918   |
| kpx_group_2 | scada_under_label         |   1098 |                0.261051   |           0.588623 |         0.327572   |         0.863616   |
| kpx_group_2 | scada_zero_label_positive |    868 |                0.408914   |           0.410616 |         0.00170208 |         0.727855   |
| kpx_group_2 | label_missing             |    103 |              nan          |         nan        |         0.00760698 |         0.00997843 |
| kpx_group_2 | low_online_label_positive |     12 |                0.118701   |           0.191001 |         0.0722994  |         0.518519   |
| kpx_group_3 | tight_reconstruction      |  14493 |                0.00755222 |           0.212574 |         0.212466   |         0.516415   |
| kpx_group_3 | moderate_residual         |   2345 |                0.0501136  |           0.485349 |         0.482012   |         0.910931   |
| kpx_group_3 | scada_under_label         |    341 |                0.188247   |           0.865348 |         0.6771     |         0.897752   |
| kpx_group_3 | low_online_label_positive |    260 |                0.0275803  |           0.243013 |         0.226488   |         0.436282   |
| kpx_group_3 | scada_zero_label_positive |     94 |                0.716432   |           0.716432 |         0          |         0.743262   |
| kpx_group_3 | label_missing             |      6 |              nan          |         nan        |         0          |         0          |

Interpretation:
- Clean SCADA reconstructs labels strongly, but the residual is structured enough to deserve a dedicated event taxonomy.
- `tight_reconstruction` dominates many rows, but `scada_under_label`, `scada_over_label`, and `scada_zero_label_positive` slices identify rows where simple power summation does not explain the label.
- These slices are the best current proxy for operating-state and reporting-state issues because SCADA is the closest observable teacher.

### NWP-to-SCADA Transfer Bias

| target      | source   |   month |   mae_speed_bias |   mean_speed_bias |   mean_abs_dir_diff |   mean_label_ratio |
|:------------|:---------|--------:|-----------------:|------------------:|--------------------:|-------------------:|
| kpx_group_3 | gfs      |      12 |          4.53678 |           3.68471 |             90.6408 |           0.451487 |
| kpx_group_1 | gfs      |      12 |          4.44759 |           2.82491 |             81.8512 |           0.524979 |
| kpx_group_3 | gfs      |      11 |          4.16731 |           3.24958 |            102.041  |           0.36205  |
| kpx_group_3 | gfs      |       3 |          4.11769 |           3.52396 |            102.471  |           0.314338 |
| kpx_group_2 | gfs      |      12 |          3.96041 |           2.1258  |             86.2823 |           0.578378 |
| kpx_group_1 | gfs      |       3 |          3.77315 |           2.51182 |             92.703  |           0.328637 |
| kpx_group_1 | gfs      |      11 |          3.66779 |           1.63364 |             91.9574 |           0.376958 |
| kpx_group_3 | gfs      |       1 |          3.64191 |           2.60028 |             96.4152 |           0.404695 |
| kpx_group_3 | gfs      |       7 |          3.62837 |           2.81765 |            116.302  |           0.358159 |
| kpx_group_1 | gfs      |       2 |          3.61292 |           1.62195 |             86.8407 |           0.316891 |
| kpx_group_2 | gfs      |       3 |          3.41806 |           1.87197 |            100.991  |           0.355631 |
| kpx_group_2 | gfs      |      11 |          3.39492 |           1.31776 |             94.7079 |           0.37826  |
| kpx_group_1 | gfs      |       1 |          3.32563 |           1.30431 |             82.7018 |           0.433114 |
| kpx_group_1 | gfs      |       5 |          3.16507 |           1.4315  |             98.9202 |           0.326506 |
| kpx_group_3 | ldaps    |      11 |          3.16314 |           2.90812 |            110.055  |           0.36205  |
| kpx_group_3 | gfs      |       4 |          3.15389 |           2.02867 |            103.499  |           0.261129 |
| kpx_group_3 | ldaps    |       3 |          3.10197 |           2.76338 |            108.168  |           0.314338 |
| kpx_group_3 | ldaps    |      12 |          3.08204 |           2.74142 |            102.433  |           0.451487 |

Interpretation:
- The deployable problem is not just label regression. NWP wind and turbine-measured SCADA wind differ by month/group/source.
- This explains why SCADA oracle headroom does not automatically transfer to test-time NWP.
- Future features should model NWP-to-SCADA bias regimes before using SCADA-derived power curves as if NWP wind were the same measurement.

### Direction-Conditional Spatial Signal

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

Interpretation:
- The best LDAPS grid is direction-dependent. A single nearest-grid or all-grid dump is too crude.
- Spatial feature engineering should use wind-direction conditioned upstream grids or soft weights, not a static grid list.

### 2025 Exposure Risk

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

Interpretation:
- 2025 exposure risk is concentrated by month, especially where label-relevant wind features shift away from 2024 validation.
- Public-submission decisions should carry this risk surface; a 2024-only win is not enough.

### Best Candidate Residual Overlay

| target      | taxonomy                  |   rows |   mean_delta_abs_error |   candidate_abs_error |   baseline_abs_error |   pass_to_fail |   fail_to_pass |   mean_nwp_scada_abs_speed_bias |   net_ficr_pass |
|:------------|:--------------------------|-------:|-----------------------:|----------------------:|---------------------:|---------------:|---------------:|--------------------------------:|----------------:|
| kpx_group_1 | low_online_label_positive |      4 |           -0.0129774   |             0.342234  |            0.355211  |              0 |              0 |                        2.80599  |               0 |
| kpx_group_1 | scada_zero_label_positive |     36 |           -0.00177456  |             0.160278  |            0.162052  |              0 |              1 |                        2.27052  |               1 |
| kpx_group_1 | moderate_residual         |    466 |           -0.000605438 |             0.111247  |            0.111853  |             15 |             15 |                        2.41617  |               0 |
| kpx_group_1 | tight_reconstruction      |   8106 |            0.000335211 |             0.093583  |            0.0932478 |             89 |            115 |                        1.88558  |              26 |
| kpx_group_1 | scada_under_label         |    166 |            0.000402195 |             0.115737  |            0.115335  |              5 |              4 |                        3.14794  |              -1 |
| kpx_group_1 | label_missing             |      6 |          nan           |           nan         |          nan         |              0 |              0 |                        6.84099  |               0 |
| kpx_group_2 | tight_reconstruction      |   7612 |           -7.36483e-05 |             0.0887857 |            0.0888594 |             80 |            105 |                        1.83648  |              25 |
| kpx_group_2 | moderate_residual         |    715 |            0.000441401 |             0.129203  |            0.128762  |             13 |             12 |                        2.13396  |              -1 |
| kpx_group_2 | scada_under_label         |    309 |            0.0010285   |             0.144385  |            0.143356  |              6 |              4 |                        2.67572  |              -2 |
| kpx_group_2 | scada_zero_label_positive |    137 |            0.00329557  |             0.303     |            0.299705  |              3 |              0 |                        2.50245  |              -3 |
| kpx_group_2 | low_online_label_positive |      5 |            0.00378449  |             0.291106  |            0.287322  |              0 |              0 |                        2.31341  |               0 |
| kpx_group_2 | label_missing             |      6 |          nan           |           nan         |          nan         |              0 |              0 |                        6.84099  |               0 |
| kpx_group_3 | scada_under_label         |    239 |           -0.00427398  |             0.161403  |            0.165677  |              2 |              6 |                        2.70899  |               4 |
| kpx_group_3 | moderate_residual         |   1158 |           -0.000447937 |             0.143632  |            0.144079  |             29 |             26 |                        3.30014  |              -3 |
| kpx_group_3 | scada_zero_label_positive |     92 |           -0.000341968 |             0.144257  |            0.144599  |              1 |              2 |                        3.19009  |               1 |
| kpx_group_3 | tight_reconstruction      |   7187 |            8.07298e-05 |             0.0825189 |            0.0824382 |             64 |             75 |                        2.33604  |              11 |
| kpx_group_3 | low_online_label_positive |    102 |            0.00147689  |             0.135813  |            0.134336  |              0 |              4 |                        3.22394  |               4 |
| kpx_group_3 | label_missing             |      6 |          nan           |           nan         |          nan         |              0 |              0 |                        0.575234 |               0 |

Interpretation:
- The current candidate must be judged by where it helps/hurts across SCADA residual taxonomy and FiCR transitions, not just total score.
- Error improvements are not uniform across operating-state proxy slices.
- The remaining modeling question is now concrete: identify which SCADA-residual/NWP-bias regimes are deployable from 2025 NWP alone.

### Updated EDA Completion Status

Completed in this issue:
- Column/label/time/spatial/weather/SCADA/FiCR/test-shift atlas.
- SCADA residual taxonomy.
- NWP-to-SCADA wind transfer gap.
- Direction-conditioned LDAPS spatial signal.
- Label-relevance-weighted 2025 exposure risk.
- Best-candidate residual overlay on SCADA taxonomy and NWP-SCADA bias.

Still not fully completed:
- Full human-readable data dictionary for every raw weather column.
- Causal labeling of SCADA residual events using external maintenance/curtailment data, which is not present in the provided dataset.
- A new model trained from this EDA. This issue intentionally stops at EDA and model-error diagnosis.

# Foundational EDA and Problem Framing

## Scope

- No model training, no feature engineering, no cleansing, no HPO, no submission.
- This report inventories dataset, target, and metric behavior to reset the ML optimization sequence.

## Key facts

- non-null label row-targets: 69,939
- missing label row-targets: 8,973
- official eligible row-targets (actual >= 10% capacity): 41,220
- official eligible rate among non-null labels: 0.5894
- negative label count: 0
- capacity exceed count: 38

## Dataset inventory

| file                         | exists   |     bytes |   rows |   columns | datetime_columns                        |   numeric_columns |   object_columns |
|:-----------------------------|:---------|----------:|-------:|----------:|:----------------------------------------|------------------:|-----------------:|
| train\train_labels.csv       | True     |   1138967 |  26304 |         4 | kst_dtm                                 |                 3 |                0 |
| train\ldaps_train.csv        | True     | 129687357 | 420864 |        35 | forecast_kst_dtm,data_available_kst_dtm |                33 |                0 |
| train\gfs_train.csv          | True     |  84315594 | 236736 |        40 | forecast_kst_dtm,data_available_kst_dtm |                38 |                0 |
| train\scada_unison_train.csv | True     |  16788646 | 105264 |        16 | kst_dtm                                 |                15 |                0 |
| train\scada_vestas_train.csv | True     |  33458140 | 157819 |        37 | kst_dtm                                 |                36 |                0 |
| test\ldaps_test.csv          | True     |  43122637 | 140160 |        35 | forecast_kst_dtm,data_available_kst_dtm |                33 |                0 |
| test\gfs_test.csv            | True     |  28037722 |  78840 |        40 | forecast_kst_dtm,data_available_kst_dtm |                38 |                0 |
| sample_submission.csv        | True     |    359229 |   8760 |         5 | forecast_kst_dtm                        |                 3 |                0 |
| data_description.md          | True     |     11694 |    nan |       nan |                                         |               nan |              nan |
| info.xlsx                    | True     |   3823422 |     20 |        12 | sheets=info                             |               nan |              nan |

## Label coverage

| target      |   year |   rows |   non_null |   missing |   eligible_count |   eligible_rate |   zero_count |   negative_count |   capacity_exceed_count |
|:------------|-------:|-------:|-----------:|----------:|-----------------:|----------------:|-------------:|-----------------:|------------------------:|
| kpx_group_1 |   2022 |   8759 |       8664 |        95 |             5416 |        0.618335 |          827 |                0 |                       0 |
| kpx_group_1 |   2023 |   8760 |       8757 |         3 |             5509 |        0.628881 |          917 |                0 |                       0 |
| kpx_group_1 |   2024 |   8784 |       8778 |         6 |             4989 |        0.567964 |         1257 |                0 |                       0 |
| kpx_group_1 |   2025 |      1 |          1 |         0 |                1 |        1        |            0 |                0 |                       0 |
| kpx_group_2 |   2022 |   8759 |       8664 |        95 |             5440 |        0.621075 |          869 |                0 |                       0 |
| kpx_group_2 |   2023 |   8760 |       8758 |         2 |             5474 |        0.624886 |          931 |                0 |                       0 |
| kpx_group_2 |   2024 |   8784 |       8778 |         6 |             4976 |        0.566485 |         1192 |                0 |                       0 |
| kpx_group_2 |   2025 |      1 |          1 |         0 |                1 |        1        |            0 |                0 |                       0 |
| kpx_group_3 |   2022 |   8759 |          0 |      8759 |                0 |        0        |            0 |                0 |                       0 |
| kpx_group_3 |   2023 |   8760 |       8759 |         1 |             4847 |        0.553311 |         1302 |                0 |                      11 |
| kpx_group_3 |   2024 |   8784 |       8778 |         6 |             4566 |        0.519809 |         1544 |                0 |                      27 |
| kpx_group_3 |   2025 |      1 |          1 |         0 |                1 |        1        |            0 |                0 |                       0 |

## Eligible generation-bin structure

| ratio_bin   |   eligible_count |   actual_sum |   eligible_share |   actual_sum_share |
|:------------|-----------------:|-------------:|-----------------:|-------------------:|
| 10-30%      |            12780 |  5.14212e+07 |        0.310044  |           0.116309 |
| 30-60%      |            12337 |  1.18499e+08 |        0.299296  |           0.26803  |
| 60-80%      |             8796 |  1.32386e+08 |        0.213392  |           0.299441 |
| 80-90%      |             4133 |  7.55616e+07 |        0.100267  |           0.170911 |
| 90-100%     |             3174 |  6.42429e+07 |        0.0770015 |           0.14531  |

## Year-level target shift signals

- kpx_group_1: 2024 mean ratio 0.2918, 2023 mean ratio 0.3241, delta -0.0323
- kpx_group_2: 2024 mean ratio 0.3004, 2023 mean ratio 0.3345, delta -0.0342
- kpx_group_3: 2024 mean ratio 0.2613, 2023 mean ratio 0.2685, delta -0.0072

## FICR boundaries

| target      |   capacity |   six_pct_boundary_kwh |   eight_pct_boundary_kwh |   eligible_count |   eligible_actual_sum |   eligible_mean_actual |   eligible_median_actual |   eligible_p10_actual |   eligible_p90_actual |   boundary_8pct_as_share_of_eligible_mean |   boundary_6pct_as_share_of_eligible_mean |
|:------------|-----------:|-----------------------:|-------------------------:|-----------------:|----------------------:|-----------------------:|-------------------------:|----------------------:|----------------------:|------------------------------------------:|------------------------------------------:|
| kpx_group_1 |      21600 |                   1296 |                     1728 |            15915 |           1.68086e+08 |               10561.5  |                 10370.1  |               3196.36 |               18556.2 |                                  0.163613 |                                  0.12271  |
| kpx_group_2 |      21600 |                   1296 |                     1728 |            15891 |           1.80042e+08 |               11329.8  |                 11440.4  |               3283.14 |               19207.1 |                                  0.152518 |                                  0.114388 |
| kpx_group_3 |      21000 |                   1260 |                     1680 |             9414 |           9.39812e+07 |                9983.13 |                  9270.88 |               2993.87 |               18339.3 |                                  0.168284 |                                  0.126213 |

## Initial interpretation

- The official metric excludes actual/capacity below 10%, so near-zero rows are primarily a training/data-behavior concern, not direct official-score mass.
- The eligible set is dominated by 10~80% ratio bins by count, while high-ratio bins carry high actual mass and stricter underprediction risk.
- Group/year coverage differs structurally because group3 has missing labels in 2022 by competition design.
- Before any more model HPO, the next decision should be based on target-year/time structure and feature coverage/drift, not on a single model's residuals alone.
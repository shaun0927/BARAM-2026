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

## Forecast availability and raw weather consistency

- forecast lead-hour range across weather files: 12.0 to 35.0
- forecasts with multiple data_available timestamps: 0
- duplicate forecast/grid rows: 0
- train/test weather schemas have identical ordered columns: True
- sample timestamps missing from test weather sources: 0

### Weather availability

| file                  |   rows |   unique_forecast_timestamps | forecast_min        | forecast_max        |   missing_forecast_hours |   unique_data_available_timestamps | data_available_min   | data_available_max   |   min_lead_hours |   p50_lead_hours |   max_lead_hours |   unique_lead_hours |   min_grid_count_per_forecast |   max_grid_count_per_forecast |   mode_grid_count_per_forecast |   forecasts_with_multiple_available_times |   duplicate_forecast_grid_rows |
|:----------------------|-------:|-----------------------------:|:--------------------|:--------------------|-------------------------:|-----------------------------------:|:---------------------|:---------------------|-----------------:|-----------------:|-----------------:|--------------------:|------------------------------:|------------------------------:|-------------------------------:|------------------------------------------:|-------------------------------:|
| train/ldaps_train.csv | 420864 |                        26304 | 2022-01-01 01:00:00 | 2025-01-01 00:00:00 |                        0 |                               1096 | 2021-12-31 13:00:00  | 2024-12-30 13:00:00  |               12 |             23.5 |               35 |                  24 |                            16 |                            16 |                             16 |                                         0 |                              0 |
| train/gfs_train.csv   | 236736 |                        26304 | 2022-01-01 01:00:00 | 2025-01-01 00:00:00 |                        0 |                               1096 | 2021-12-31 13:00:00  | 2024-12-30 13:00:00  |               12 |             23.5 |               35 |                  24 |                             9 |                             9 |                              9 |                                         0 |                              0 |
| test/ldaps_test.csv   | 140160 |                         8760 | 2025-01-01 01:00:00 | 2026-01-01 00:00:00 |                        0 |                                365 | 2024-12-31 13:00:00  | 2025-12-30 13:00:00  |               12 |             23.5 |               35 |                  24 |                            16 |                            16 |                             16 |                                         0 |                              0 |
| test/gfs_test.csv     |  78840 |                         8760 | 2025-01-01 01:00:00 | 2026-01-01 00:00:00 |                        0 |                                365 | 2024-12-31 13:00:00  | 2025-12-30 13:00:00  |               12 |             23.5 |               35 |                  24 |                             9 |                             9 |                              9 |                                         0 |                              0 |

### Weather schema consistency

| source   |   train_column_count |   test_column_count | same_ordered_columns   | train_only_columns   | test_only_columns   |
|:---------|---------------------:|--------------------:|:-----------------------|:---------------------|:--------------------|
| ldaps    |                   35 |                  35 | True                   |                      |                     |
| gfs      |                   40 |                  40 | True                   |                      |                     |

### Sample/test horizon alignment

| source            |   rows | min_ts              | max_ts              |   unique_timestamps |   missing_hour_gaps |   sample_timestamps_missing_from_source |   source_timestamps_not_in_sample |
|:------------------|-------:|:--------------------|:--------------------|--------------------:|--------------------:|----------------------------------------:|----------------------------------:|
| sample_submission |   8760 | 2025-01-01 01:00:00 | 2026-01-01 00:00:00 |                8760 |                   0 |                                     nan |                               nan |
| ldaps_test        | 140160 | 2025-01-01 01:00:00 | 2026-01-01 00:00:00 |                8760 |                   0 |                                       0 |                                 0 |
| gfs_test          |  78840 | 2025-01-01 01:00:00 | 2026-01-01 00:00:00 |                8760 |                   0 |                                       0 |                                 0 |

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

## Target temporal/extreme structure

- kpx_group_1: eligible_rate 0.6074, zero_rate 0.1145, high>=80% rate 0.0927, max_zero_run_hours 169, max_ratio 0.9850
- kpx_group_2: eligible_rate 0.6065, zero_rate 0.1142, high>=80% rate 0.1322, max_zero_run_hours 161, max_ratio 0.9890
- kpx_group_3: eligible_rate 0.5368, zero_rate 0.1623, high>=80% rate 0.0806, max_zero_run_hours 142, max_ratio 1.0062
- capacity-exceed details: 38 rows, max excess 130.674 kWh, max ratio 1.006223

### Target temporal/extreme summary

| target      | scope                  |   rows |   mean_ratio |   std_ratio |   zero_rate |   near_zero_rate_lt_1pct |   eligible_rate |   high_generation_rate_ge_80pct |   capacity_exceed_count |   max_zero_run_hours |   max_ratio |   best_hour |   worst_hour |   hourly_mean_ratio_range |   best_month |   worst_month |   monthly_mean_ratio_range |
|:------------|:-----------------------|-------:|-------------:|------------:|------------:|-------------------------:|----------------:|--------------------------------:|------------------------:|---------------------:|------------:|------------:|-------------:|--------------------------:|-------------:|--------------:|---------------------------:|
| kpx_group_1 | all_nonnull            |  26200 |     0.306573 |    0.304743 |   0.114542  |                 0.193664 |        0.607443 |                       0.0926718 |                       0 |                  169 |    0.984968 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_1 | year_2022              |   8664 |     0.303765 |    0.288478 |   0.0954524 |                 0.170822 |        0.625115 |                       0.0644044 |                       0 |                   38 |    0.984298 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_1 | year_2023              |   8757 |     0.324094 |    0.313933 |   0.104716  |                 0.173119 |        0.629097 |                       0.117506  |                       0 |                   85 |    0.984968 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_1 | year_2024              |   8778 |     0.291791 |    0.310111 |   0.143199  |                 0.236728 |        0.568353 |                       0.0956938 |                       0 |                  169 |    0.984298 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_1 | year_2025              |      1 |     0.966219 |  nan        |   0         |                 0        |        1        |                       1         |                       0 |                    0 |    0.966219 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_1 | eligible_hourly_range  |  15915 |     0.488959 |    0.259982 | nan         |               nan        |      nan        |                       0.15256   |                       0 |                  nan |    0.984968 |           2 |           11 |                 0.0938178 |          nan |           nan |                 nan        |
| kpx_group_1 | eligible_monthly_range |  15915 |     0.488959 |    0.259982 | nan         |               nan        |      nan        |                       0.15256   |                       0 |                  nan |    0.984968 |         nan |          nan |               nan         |           12 |             9 |                   0.283791 |
| kpx_group_2 | all_nonnull            |  26201 |     0.327632 |    0.324127 |   0.114194  |                 0.194344 |        0.606504 |                       0.132247  |                       0 |                  161 |    0.988985 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_2 | year_2022              |   8664 |     0.348173 |    0.33134  |   0.1003    |                 0.175323 |        0.627886 |                       0.15247   |                       0 |                   75 |    0.988985 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_2 | year_2023              |   8758 |     0.334542 |    0.322786 |   0.106303  |                 0.173784 |        0.625029 |                       0.135876  |                       0 |                   84 |    0.986977 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_2 | year_2024              |   8778 |     0.300387 |    0.316287 |   0.135794  |                 0.233652 |        0.566872 |                       0.108567  |                       0 |                  161 |    0.988985 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_2 | year_2025              |      1 |     0.986307 |  nan        |   0         |                 0        |        1        |                       1         |                       0 |                    0 |    0.986307 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_2 | eligible_hourly_range  |  15891 |     0.52453  |    0.272294 | nan         |               nan        |      nan        |                       0.218048  |                       0 |                  nan |    0.988985 |           3 |           16 |                 0.112693  |          nan |           nan |                 nan        |
| kpx_group_2 | eligible_monthly_range |  15891 |     0.52453  |    0.272294 | nan         |               nan        |      nan        |                       0.218048  |                       0 |                  nan |    0.988985 |         nan |          nan |               nan         |           12 |             9 |                   0.316637 |
| kpx_group_3 | all_nonnull            |  17538 |     0.264944 |    0.299742 |   0.162276  |                 0.26263  |        0.536777 |                       0.0806249 |                      38 |                  142 |    1.00622  |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_3 | year_2023              |   8759 |     0.268518 |    0.295427 |   0.148647  |                 0.239183 |        0.553374 |                       0.0801461 |                      11 |                   77 |    1.00416  |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_3 | year_2024              |   8778 |     0.261319 |    0.303928 |   0.175894  |                 0.286056 |        0.520164 |                       0.0811119 |                      27 |                  142 |    1.00622  |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_3 | year_2025              |      1 |     0.777567 |  nan        |   0         |                 0        |        1        |                       0         |                       0 |                    0 |    0.777567 |         nan |          nan |               nan         |          nan |           nan |                 nan        |
| kpx_group_3 | eligible_hourly_range  |   9414 |     0.475387 |    0.266602 | nan         |               nan        |      nan        |                       0.150202  |                      38 |                  nan |    1.00622  |           0 |           14 |                 0.0876344 |          nan |           nan |                 nan        |
| kpx_group_3 | eligible_monthly_range |   9414 |     0.475387 |    0.266602 | nan         |               nan        |      nan        |                       0.150202  |                      38 |                  nan |    1.00622  |         nan |          nan |               nan         |           11 |             9 |                   0.329391 |

### Capacity exceed context

| kst_dtm             | target      |   year |   month |   hour |   actual |   capacity |   actual_ratio |   excess_kwh |   same_year_month_hour_count |   same_year_month_hour_p95_ratio |   same_year_month_hour_max_ratio |
|:--------------------|:------------|-------:|--------:|-------:|---------:|-----------:|---------------:|-------------:|-----------------------------:|---------------------------------:|---------------------------------:|
| 2023-01-11 06:00:00 | kpx_group_3 |   2023 |       1 |      6 |  21058.4 |      21000 |        1.00278 |       58.358 |                           31 |                         0.974197 |                          1.00278 |
| 2023-02-28 02:00:00 | kpx_group_3 |   2023 |       2 |      2 |  21087.3 |      21000 |        1.00416 |       87.284 |                           28 |                         0.928879 |                          1.00416 |
| 2023-04-13 00:00:00 | kpx_group_3 |   2023 |       4 |      0 |  21058.4 |      21000 |        1.00278 |       58.358 |                           30 |                         0.945615 |                          1.00278 |
| 2023-04-13 02:00:00 | kpx_group_3 |   2023 |       4 |      2 |  21058.4 |      21000 |        1.00278 |       58.358 |                           30 |                         0.829359 |                          1.00278 |
| 2023-04-13 06:00:00 | kpx_group_3 |   2023 |       4 |      6 |  21000.5 |      21000 |        1.00002 |        0.505 |                           30 |                         0.925022 |                          1.00002 |
| 2023-07-14 18:00:00 | kpx_group_3 |   2023 |       7 |     18 |  21072.8 |      21000 |        1.00347 |       72.821 |                           31 |                         0.83783  |                          1.00347 |
| 2023-11-18 21:00:00 | kpx_group_3 |   2023 |      11 |     21 |  21043.9 |      21000 |        1.00209 |       43.895 |                           30 |                         0.964279 |                          1.00347 |
| 2023-11-21 21:00:00 | kpx_group_3 |   2023 |      11 |     21 |  21072.8 |      21000 |        1.00347 |       72.821 |                           30 |                         0.964279 |                          1.00347 |
| 2023-12-25 01:00:00 | kpx_group_3 |   2023 |      12 |      1 |  21000.5 |      21000 |        1.00002 |        0.505 |                           31 |                         0.955257 |                          1.00002 |
| 2023-12-26 01:00:00 | kpx_group_3 |   2023 |      12 |      1 |  21000.5 |      21000 |        1.00002 |        0.505 |                           31 |                         0.955257 |                          1.00002 |
| 2023-12-26 02:00:00 | kpx_group_3 |   2023 |      12 |      2 |  21058.4 |      21000 |        1.00278 |       58.358 |                           31 |                         0.970409 |                          1.00278 |
| 2024-01-02 11:00:00 | kpx_group_3 |   2024 |       1 |     11 |  21043.9 |      21000 |        1.00209 |       43.895 |                           31 |                         0.997958 |                          1.00209 |
| 2024-01-04 21:00:00 | kpx_group_3 |   2024 |       1 |     21 |  21087.3 |      21000 |        1.00416 |       87.284 |                           31 |                         0.977641 |                          1.00416 |
| 2024-01-04 22:00:00 | kpx_group_3 |   2024 |       1 |     22 |  21101.7 |      21000 |        1.00485 |      101.747 |                           31 |                         0.990382 |                          1.00485 |
| 2024-01-08 21:00:00 | kpx_group_3 |   2024 |       1 |     21 |  21072.8 |      21000 |        1.00347 |       72.821 |                           31 |                         0.977641 |                          1.00416 |
| 2024-01-08 22:00:00 | kpx_group_3 |   2024 |       1 |     22 |  21072.8 |      21000 |        1.00347 |       72.821 |                           31 |                         0.990382 |                          1.00485 |
| 2024-01-09 01:00:00 | kpx_group_3 |   2024 |       1 |      1 |  21087.3 |      21000 |        1.00416 |       87.284 |                           31 |                         1.00347  |                          1.00416 |
| 2024-01-11 03:00:00 | kpx_group_3 |   2024 |       1 |      3 |  21130.7 |      21000 |        1.00622 |      130.674 |                           31 |                         0.969032 |                          1.00622 |
| 2024-01-11 04:00:00 | kpx_group_3 |   2024 |       1 |      4 |  21043.9 |      21000 |        1.00209 |       43.895 |                           31 |                         0.970753 |                          1.00209 |
| 2024-01-11 05:00:00 | kpx_group_3 |   2024 |       1 |      5 |  21130.7 |      21000 |        1.00622 |      130.674 |                           31 |                         0.977985 |                          1.00622 |

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

## B1 feature inventory and drift

- B1 aggregate feature count: 286 across 3 source groups
- max missing rate train/valid/test: 0.0000 / 0.0000 / 0.0003
- near-constant feature count: 24
- features with train-valid KS > 0.30: 1 total, 0 excluding time features
- features with train-test KS > 0.30: 1 total, 0 excluding time features
- features with train-valid PSI > 0.25: 0
- features with train-test PSI > 0.25: 0
- SCADA files inventoried: 2
- info.xlsx turbine/meta rows: 17

### Feature source/stat composition

| source   | stat   |   feature_count |   mean_train_missing |   mean_valid_missing |   mean_test_missing |   near_constant_count |
|:---------|:-------|----------------:|---------------------:|---------------------:|--------------------:|----------------------:|
| gfs      | max    |              37 |                    0 |                    0 |         0           |                     2 |
| gfs      | mean   |              37 |                    0 |                    0 |         0           |                     2 |
| gfs      | min    |              37 |                    0 |                    0 |         0           |                     2 |
| gfs      | std    |              37 |                    0 |                    0 |         0           |                     2 |
| ldaps    | max    |              32 |                    0 |                    0 |         0.000167685 |                     4 |
| ldaps    | mean   |              32 |                    0 |                    0 |         0.000167685 |                     4 |
| ldaps    | min    |              32 |                    0 |                    0 |         0.000167685 |                     4 |
| ldaps    | std    |              32 |                    0 |                    0 |         0.000167685 |                     4 |
| time     | time   |              10 |                    0 |                    0 |         0           |                     0 |

### Top train-valid drift features

| feature                          | source   | stat   |   train_valid_ks |   train_test_ks |   valid_test_ks |   train_valid_psi |   train_test_psi |   valid_test_psi |   valid_mean_delta |   test_mean_delta_vs_train |
|:---------------------------------|:---------|:-------|-----------------:|----------------:|----------------:|------------------:|-----------------:|-----------------:|-------------------:|---------------------------:|
| year                             | time     | time   |        1         |       1         |       1         |         0         |        0         |       0          |           1.49997  |                   2.49997  |
| ldaps_heightAboveGround_2_r_min  | ldaps    | min    |        0.127852  |       0.0520544 |       0.0857376 |         0.115486  |        0.0156362 |       0.0560878  |           6.59406  |                   2.35218  |
| ldaps_heightAboveGround_2_r_mean | ldaps    | mean   |        0.126709  |       0.0504604 |       0.084726  |         0.121889  |        0.0132046 |       0.0558782  |           6.48333  |                   2.18887  |
| gfs_heightAboveGround_2_2r_min   | gfs      | min    |        0.124617  |       0.0479763 |       0.091006  |         0.0974697 |        0.0501122 |       0.0505983  |           4.40382  |                   1.08512  |
| ldaps_heightAboveGround_2_r_max  | ldaps    | max    |        0.118387  |       0.0371659 |       0.0886051 |         0.122466  |        0.0104343 |       0.0556651  |           5.90721  |                   1.79526  |
| gfs_heightAboveGround_2_2r_mean  | gfs      | mean   |        0.113471  |       0.0456969 |       0.0792795 |         0.0825347 |        0.0282658 |       0.0358651  |           4.97627  |                   1.72042  |
| gfs_heightAboveGround_2_2r_max   | gfs      | max    |        0.110845  |       0.0641445 |       0.054537  |         0.100874  |        0.0290078 |       0.0279885  |           4.58024  |                   2.04113  |
| gfs_isobaricInhPa_850_r_min      | gfs      | min    |        0.10104   |       0.076068  |       0.0510953 |         0.0417672 |        0.0379594 |       0.0163013  |           4.83026  |                   2.57365  |
| ldaps_surface_0_NDNLW_max        | ldaps    | max    |        0.100764  |       0.0338217 |       0.085858  |         0.0510546 |        0.0151478 |       0.040024   |           8.91383  |                   2.21515  |
| gfs_isobaricInhPa_850_r_mean     | gfs      | mean   |        0.0993735 |       0.0775575 |       0.0654038 |         0.0520103 |        0.0324803 |       0.0192307  |           5.53934  |                   2.97376  |
| ldaps_surface_0_NDNLW_min        | ldaps    | min    |        0.0983199 |       0.0210571 |       0.0939427 |         0.0466266 |        0.0101954 |       0.0446795  |           7.5435   |                   1.50551  |
| ldaps_surface_0_NDNLW_mean       | ldaps    | mean   |        0.0964541 |       0.027143  |       0.0910727 |         0.0484202 |        0.0126052 |       0.041027   |           8.51948  |                   1.89189  |
| gfs_heightAboveGround_2_2d_std   | gfs      | std    |        0.0955519 |       0.0843567 |       0.023029  |         0.0828569 |        0.0511854 |       0.00726663 |          -0.188159 |                  -0.151432 |
| gfs_isobaricInhPa_500_gh_max     | gfs      | max    |        0.0955093 |       0.0852328 |       0.0901149 |         0.0855839 |        0.122948  |       0.0722977  |          22.235    |                  -1.53049  |
| gfs_isobaricInhPa_500_gh_min     | gfs      | min    |        0.0953594 |       0.0862034 |       0.0930127 |         0.0895333 |        0.133401  |       0.070097   |          23.7697   |                  -1.86293  |

### Top train-test drift features

| feature                            | source   | stat   |   train_valid_ks |   train_test_ks |   valid_test_ks |   train_valid_psi |   train_test_psi |   valid_test_psi |   valid_mean_delta |   test_mean_delta_vs_train |
|:-----------------------------------|:---------|:-------|-----------------:|----------------:|----------------:|------------------:|-----------------:|-----------------:|-------------------:|---------------------------:|
| year                               | time     | time   |        1         |       1         |       1         |         0         |        0         |       0          |        1.49997     |                2.49997     |
| gfs_isobaricInhPa_500_gh_mean      | gfs      | mean   |        0.0948349 |       0.0867742 |       0.091469  |         0.0899089 |        0.126378  |       0.0692736  |       23.0278      |               -1.75636     |
| gfs_isobaricInhPa_500_gh_min       | gfs      | min    |        0.0953594 |       0.0862034 |       0.0930127 |         0.0895333 |        0.133401  |       0.070097   |       23.7697      |               -1.86293     |
| gfs_isobaricInhPa_500_gh_max       | gfs      | max    |        0.0955093 |       0.0852328 |       0.0901149 |         0.0855839 |        0.122948  |       0.0722977  |       22.235       |               -1.53049     |
| gfs_heightAboveGround_2_2d_std     | gfs      | std    |        0.0955519 |       0.0843567 |       0.023029  |         0.0828569 |        0.0511854 |       0.00726663 |       -0.188159    |               -0.151432    |
| gfs_isobaricInhPa_850_r_mean       | gfs      | mean   |        0.0993735 |       0.0775575 |       0.0654038 |         0.0520103 |        0.0324803 |       0.0192307  |        5.53934     |                2.97376     |
| gfs_isobaricInhPa_850_r_min        | gfs      | min    |        0.10104   |       0.076068  |       0.0510953 |         0.0417672 |        0.0379594 |       0.0163013  |        4.83026     |                2.57365     |
| gfs_isobaricInhPa_500_t_max        | gfs      | max    |        0.0859844 |       0.0753651 |       0.089206  |         0.111535  |        0.0703853 |       0.154296   |        1.19888     |                0.186865    |
| gfs_isobaricInhPa_500_t_mean       | gfs      | mean   |        0.0880584 |       0.0731956 |       0.0916236 |         0.106756  |        0.0707377 |       0.159887   |        1.24286     |                0.172564    |
| gfs_heightAboveGround_2_2r_std     | gfs      | std    |        0.0341085 |       0.0731623 |       0.0441582 |         0.0168312 |        0.0328894 |       0.0154577  |       -0.0124964   |                0.340971    |
| gfs_isobaricInhPa_850_r_max        | gfs      | max    |        0.0910807 |       0.0720797 |       0.0612294 |         0.0547308 |        0.0300514 |       0.0168275  |        5.53408     |                3.28644     |
| gfs_isobaricInhPa_500_t_min        | gfs      | min    |        0.0883199 |       0.0709706 |       0.0963133 |         0.099693  |        0.0725813 |       0.162071   |        1.28357     |                0.143566    |
| gfs_surface_0_sp_std               | gfs      | std    |        0.0363866 |       0.0679369 |       0.044131  |         0.0191569 |        0.0574494 |       0.0254494  |       -3.10954     |               -7.73361     |
| ldaps_heightAboveGround_2_dpt_mean | ldaps    | mean   |        0.0779372 |       0.066562  |       0.0739471 |         0.0941097 |        0.0469238 |       0.100439   |        2.09659     |                0.731095    |
| ldaps_heightAboveGround_2_q_max    | ldaps    | max    |        0.0790438 |       0.0665057 |       0.0738093 |         0.0920604 |        0.0433115 |       0.103016   |        0.000701933 |                0.000371009 |

## SCADA/meta inventory

| file                         |   rows |   columns | min_ts              | max_ts              |   unique_timestamps | mode_timestamp_gap   |   missing_values |   numeric_columns |
|:-----------------------------|-------:|----------:|:--------------------|:--------------------|--------------------:|:---------------------|-----------------:|------------------:|
| train/scada_unison_train.csv | 105264 |        16 | 2023-01-01 00:10:00 | 2025-01-01 00:00:00 |              105264 | 0 days 00:10:00      |             9511 |                15 |
| train/scada_vestas_train.csv | 157819 |        37 | 2022-01-01 01:00:00 | 2025-01-01 00:00:00 |              157819 | 0 days 00:10:00      |                0 |                36 |

## Initial interpretation

- The official metric excludes actual/capacity below 10%, so near-zero rows are primarily a training/data-behavior concern, not direct official-score mass.
- The eligible set is dominated by 10~80% ratio bins by count, while high-ratio bins carry high actual mass and stricter underprediction risk.
- Group/year coverage differs structurally because group3 has missing labels in 2022 by competition design.
- Raw weather availability and train/test schema are structurally usable for a fixed B1 feature baseline, but this does not prove feature sufficiency.
- B1 aggregate feature drift appears mild after excluding the deterministic `year` feature; this weakens the case for more model-only search as the next step.
- The correct next action is a focused audit issue, not another broad modeling run. The focused route must be selected from hard data-quality blockers, target/temporal shift, and B1 feature coverage/drift.
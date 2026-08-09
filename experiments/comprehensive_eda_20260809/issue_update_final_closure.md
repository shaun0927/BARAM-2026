## Final EDA Closure Update — 2026-08-09

This update closes the remaining EDA gaps that were explicitly left open after the atlas and deep-dive passes:

- full raw weather column dictionary,
- SCADA residual event timeline casebook,
- model-family residual atlas.

### New Figures

22. ![Column family shift](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/22_column_dictionary_family_shift.png)
23. ![Top raw column shifts](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/23_column_dictionary_top_shifts.png)
24. ![SCADA residual event casebook](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/24_scada_residual_event_casebook.png)
25. ![SCADA residual event detail](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/25_scada_residual_event_detail.png)
26. ![Model monthly residual heatmap](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/26_model_family_monthly_residual_heatmap.png)
27. ![Model actual-bin delta](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/27_model_family_actual_bin_delta.png)
28. ![Model residual correlation](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/28_model_family_residual_correlation.png)
29. ![Top-model monthly residual heatmap](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/29_model_family_monthly_residual_top_models.png)

### Full Weather Column Dictionary

Artifacts:

- `results/final_closure/full_raw_weather_column_dictionary.csv`
- `results/final_closure/weather_column_family_summary.csv`

Family summary:

| source   | family                 |   columns |   max_abs_shift_z |   max_label_relevance |   missing_columns |
|:---------|:-----------------------|----------:|------------------:|----------------------:|------------------:|
| gfs      | cloud                  |         3 |        0.019057   |            nan        |                 0 |
| gfs      | humidity               |         2 |        0.0393991  |            nan        |                 0 |
| gfs      | metadata               |         3 |        0.13849    |              0.749029 |                 0 |
| gfs      | precipitation_snow     |         1 |        0.0313194  |            nan        |                 0 |
| gfs      | pressure               |         2 |        0.0802384  |            nan        |                 0 |
| gfs      | radiation              |         2 |        0.00891509 |            nan        |                 0 |
| gfs      | spatial_contract       |         3 |        0          |            nan        |                 0 |
| gfs      | static_or_geopotential |         1 |        0.0595014  |            nan        |                 0 |
| gfs      | temperature_dewpoint   |         7 |        0.0534229  |            nan        |                 0 |
| gfs      | time_contract          |         2 |      nan          |            nan        |                 0 |
| gfs      | wind_or_motion         |         2 |        0.0899179  |            nan        |                 0 |
| gfs      | wind_vector            |        12 |        0.137386   |              0.812383 |                 0 |
| ldaps    | boundary_layer         |         1 |        0.0597876  |            nan        |                 0 |
| ldaps    | cloud                  |         4 |        0.0474617  |            nan        |                 0 |
| ldaps    | humidity               |         2 |        0.0280537  |            nan        |                 0 |
| ldaps    | precipitation_snow     |         4 |        0.0405332  |            nan        |                 0 |
| ldaps    | pressure               |         3 |        0.0836822  |            nan        |                 0 |
| ldaps    | radiation              |         4 |        0.0273561  |            nan        |                 0 |
| ldaps    | spatial_contract       |         3 |        0          |            nan        |                 0 |
| ldaps    | static_or_geopotential |         2 |        0          |            nan        |                 0 |
| ldaps    | temperature_dewpoint   |         2 |        0.00238131 |            nan        |                 0 |
| ldaps    | time_contract          |         2 |      nan          |            nan        |                 0 |
| ldaps    | wind_vector            |         8 |        0.127677   |              0.826312 |                 0 |

Interpretation:

- The raw LDAPS/GFS columns are now grouped by physical family, height/level, component, recommended transform, train/test parity, missingness, train/test shift, and derived wind-feature relevance where applicable.
- This closes the previous gap where the EDA had column statistics but no usable modeling policy per raw column.
- The highest-risk columns for test exposure are no longer hidden in raw CSV names; they are visible by physical family and shift magnitude.

### SCADA Residual Event Timeline Casebook

Artifact:

- `results/final_closure/scada_residual_event_casebook.csv`

Top events:

| target      | taxonomy                  | start               | end                 |   duration_hours |   mean_abs_residual_ratio |   mean_label_ratio |   mean_scada_ratio |   mean_online_rate |
|:------------|:--------------------------|:--------------------|:--------------------|-----------------:|--------------------------:|-------------------:|-------------------:|-------------------:|
| kpx_group_2 | scada_zero_label_positive | 2022-01-31 13:00:00 | 2022-02-07 11:00:00 |              167 |                 0.61555   |           0.61555  |        0           |           0.738689 |
| kpx_group_3 | scada_zero_label_positive | 2024-02-12 14:00:00 | 2024-02-15 09:00:00 |               68 |                 0.812631  |           0.812631 |        0           |           0.758333 |
| kpx_group_2 | scada_zero_label_positive | 2022-02-14 21:00:00 | 2022-02-17 10:00:00 |               62 |                 0.654449  |           0.654449 |        0           |           0.822581 |
| kpx_group_2 | scada_zero_label_positive | 2022-02-20 16:00:00 | 2022-02-23 00:00:00 |               57 |                 0.554738  |           0.554738 |        0           |           0.810916 |
| kpx_group_1 | scada_zero_label_positive | 2022-12-17 16:00:00 | 2022-12-19 20:00:00 |               53 |                 0.474816  |           0.474816 |        0           |           0.643606 |
| kpx_group_1 | scada_zero_label_positive | 2022-01-16 15:00:00 | 2022-01-18 13:00:00 |               47 |                 0.573137  |           0.573455 |        0.000317179 |           0.83156  |
| kpx_group_1 | scada_zero_label_positive | 2022-12-14 04:00:00 | 2022-12-15 11:00:00 |               32 |                 0.495309  |           0.495309 |        0           |           0.652778 |
| kpx_group_1 | scada_under_label         | 2022-08-03 07:00:00 | 2022-08-04 10:00:00 |               28 |                 0.405676  |           0.573767 |        0.168092    |           0.833333 |
| kpx_group_1 | scada_under_label         | 2023-01-20 12:00:00 | 2023-01-21 12:00:00 |               25 |                 0.394255  |           0.654485 |        0.26023     |           0.833333 |
| kpx_group_1 | scada_zero_label_positive | 2022-03-04 17:00:00 | 2022-03-05 16:00:00 |               24 |                 0.438414  |           0.438414 |        0           |           0.5      |
| kpx_group_2 | scada_under_label         | 2023-01-28 17:00:00 | 2023-01-29 14:00:00 |               22 |                 0.380743  |           0.630937 |        0.250194    |           0.839646 |
| kpx_group_1 | scada_under_label         | 2022-12-06 10:00:00 | 2022-12-07 07:00:00 |               22 |                 0.276065  |           0.6772   |        0.401134    |           0.82702  |
| kpx_group_2 | scada_zero_label_positive | 2024-01-23 08:00:00 | 2024-01-24 04:00:00 |               21 |                 0.143946  |           0.147788 |        0.00384259  |           0.166667 |
| kpx_group_2 | scada_under_label         | 2023-01-03 19:00:00 | 2023-01-04 14:00:00 |               20 |                 0.375868  |           0.741639 |        0.365771    |           0.8375   |
| kpx_group_3 | low_online_label_positive | 2023-01-27 18:00:00 | 2023-01-28 13:00:00 |               20 |                 0.0113142 |           0.330242 |        0.331755    |           0.415    |

Interpretation:

- The SCADA-label residual problem is now event-level, not just row-level. Long contiguous non-tight intervals exist and should be treated as operating/reporting regimes.
- These events are not automatically removable label errors. They are the correct casebook for deciding whether a future model can detect the same regime from NWP-only 2025 inputs.
- Without external maintenance/curtailment logs, the EDA can classify observable signatures but cannot prove final causal labels.

### Model-Family Residual Atlas

Artifacts:

- `results/final_closure/model_family_residual_summary.csv`
- `results/final_closure/model_family_monthly_residual.csv`
- `results/final_closure/model_family_actual_bin_residual.csv`
- `results/final_closure/model_family_residual_correlation.csv`

Top model-family summary:

| model            |   mae_ratio |   ficr_pass_rate |   bias_ratio |
|:-----------------|------------:|-----------------:|-------------:|
| lgbm_l1_baseline |   0.0925344 |         0.413242 |  0.00284478  |
| xgboost_l1       |   0.0928503 |         0.412627 |  0.000283345 |
| catboost_mae     |   0.0936206 |         0.398016 |  0.00311543  |
| hist_gbdt_l2     |   0.0981259 |         0.40661  |  0.0107447   |
| extra_trees      |   0.0982818 |         0.411045 |  0.00766369  |
| lgbm_l2          |   0.0985697 |         0.403931 |  0.0113844   |
| random_forest    |   0.0991815 |         0.40269  |  0.0076098   |
| ridge            |   0.11969   |         0.371363 |  0.0269992   |
| lgbm_huber       |   0.276638  |         0.180307 |  0.0325224   |

Interpretation:

- Model differences are not just global scores. The residual atlas shows where families differ by month, group, and actual-power bin.
- Residual correlations are high enough that most families share the same blind spots; ensembling alone cannot solve the SCADA/NWP transfer and FiCR-boundary issues.
- LGBM remains a strong baseline, but the atlas makes the failure regimes explicit instead of treating the score table as the EDA.

### Final EDA Completion Statement

Completed:

- raw file/column contract,
- full raw weather column dictionary,
- label and evaluation support structure,
- forecast-time and grid completeness contract,
- spatial topology,
- weather-label correlation,
- direction-conditioned spatial signal,
- SCADA hourly reconstruction,
- SCADA residual taxonomy,
- SCADA residual event casebook,
- NWP-to-SCADA transfer bias,
- 2025 exposure risk weighted by label relevance,
- FiCR-sensitive surfaces,
- current best candidate residual overlay,
- model-family residual atlas.

Remaining boundaries, not missing EDA:

- True causal labels for maintenance/curtailment require external operations data that is not in the provided dataset.
- A new model trained from these findings is modeling work, not EDA.
- Future public/private split inference cannot be proved from provided files; only exposure-risk proxies can be audited.

Conclusion:

The EDA is now complete enough to stop exploratory diagnosis and move to hypothesis-driven modeling. Any next phase should cite this atlas and state exactly which discovered regime it targets.

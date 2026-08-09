# SCADA Residual Deployability EDA: can the #31 teacher signal become a test-time score action?

This issue follows #31 section 6, **SCADA-to-Label Reconstruction**.

#31 proved that clean SCADA reconstructs labels strongly, but SCADA is train-only. The open modeling question was therefore not "can SCADA explain the label?" but:

> Can the SCADA residual regimes be detected from 2025-available NWP/time features, and can they be converted into score-positive actions?

This EDA uses SCADA only as a train-time teacher label. All detectability and 2025 exposure analyses use LDAPS/GFS/time features only.

Artifacts:

- script: `experiments/scada_residual_deployability_eda_20260810/run_scada_residual_deployability_eda.py`
- report: `experiments/scada_residual_deployability_eda_20260810/results/conclusion.md`
- figures: `experiments/scada_residual_deployability_eda_20260810/results/figures/`
- tables: `experiments/scada_residual_deployability_eda_20260810/results/*.csv`

---

## 1. NWP-only Detectability

![NWP-only detectability](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/01_nwp_only_regime_detectability.png)

| target      |   balanced_accuracy |      auc |       f1 |   positive_rate_valid |
|:------------|--------------------:|---------:|---------:|----------------------:|
| kpx_group_1 |            0.548089 | 0.712741 | 0.166014 |              0.076555 |
| kpx_group_2 |            0.629903 | 0.737892 | 0.354839 |              0.132832 |
| kpx_group_3 |            0.720348 | 0.813824 | 0.496597 |              0.181249 |

### Interpretation

The SCADA residual regime is not invisible to NWP. Chronological validation shows real binary separation of `non-tight` vs `tight_reconstruction` regimes.

- group1 is weak but above random: AUC `0.713`, balanced accuracy `0.548`.
- group2 is usable: AUC `0.738`, balanced accuracy `0.630`.
- group3 is strongest on 2024 binary validation: AUC `0.814`, balanced accuracy `0.720`.

### Validity Discussion

This is not leakage: SCADA is used only to define train labels for the taxonomy. The classifier receives only LDAPS/GFS/time features. The split is chronological: train on 2022-2023, validate on 2024. For group3, the train side is 2023-only because 2022 labels do not exist.

The caveat is class imbalance. F1 is much lower than AUC, especially group1, so a hard classifier threshold is not submission-ready. The signal should be used as a probability gate, not a deterministic taxonomy replacement.

---

## 2. 2025 Exposure Surface

![2025 exposure](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/02_test2025_predicted_scada_regime_exposure.png)

Top predicted non-tight exposure months:

| target      |   month |   non_tight_probability |
|:------------|--------:|------------------------:|
| kpx_group_3 |       2 |                0.550459 |
| kpx_group_3 |      12 |                0.524983 |
| kpx_group_2 |       2 |                0.515233 |
| kpx_group_3 |       1 |                0.471364 |
| kpx_group_3 |      11 |                0.431216 |
| kpx_group_1 |      12 |                0.407676 |
| kpx_group_2 |      12 |                0.393864 |
| kpx_group_3 |       4 |                0.389482 |
| kpx_group_2 |       1 |                0.388732 |
| kpx_group_2 |      11 |                0.377826 |
| kpx_group_3 |       3 |                0.354859 |
| kpx_group_2 |       5 |                0.341899 |

### Interpretation

If a model uses SCADA-residual gates, the risk is not uniform across 2025. It concentrates in winter/early-year months and differs by group. This matters because a local W4 win can be misleading if its winning regime is under-exposed or its losing regime is over-exposed in 2025.

### Validity Discussion

These are inferred probabilities, not observed 2025 SCADA states. They are valid as submission-risk triage and candidate targeting, but they do not prove public score.

---

## 3. Feature Evidence For Deployability

![Feature importance](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/03_nwp_only_regime_feature_importance.png)

Top permutation features include:

| target      | feature                    |   importance_mean |   importance_std |
|:------------|:---------------------------|------------------:|-----------------:|
| kpx_group_1 | month_cos                  |          0.006512 |         0.002320 |
| kpx_group_1 | doy_sin                    |          0.006077 |         0.002045 |
| kpx_group_1 | ldaps_wind50max_speed_mean |          0.005774 |         0.001656 |
| kpx_group_2 | gfs_wind850_dir_deg_min    |          0.005619 |         0.001464 |
| kpx_group_2 | gfs_wind850_speed_mean     |          0.005124 |         0.001801 |
| kpx_group_3 | gfs_wind850_u_mean         |          0.005863 |         0.000683 |
| kpx_group_3 | gfs_wind850_speed_mean     |          0.005093 |         0.001193 |

### Interpretation

The signal is not only a date artifact. Calendar terms matter, but wind magnitude and direction features also carry separability, especially GFS 850hPa wind for group2/group3 and LDAPS 50m wind for group1.

### Validity Discussion

Permutation importance is model-dependent and correlated features share credit. It is sufficient to show that NWP/time axes carry deployable information, but not sufficient to pick a final feature set.

---

## 4. Taxonomy Weather Surface

![Taxonomy weather surface](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/04_taxonomy_nwp_wind_surface.png)

| target      | taxonomy                  |   rows |   ldaps_wind50max_speed_mean_mean |   gfs_wind850_speed_mean_mean |
|:------------|:--------------------------|-------:|----------------------------------:|------------------------------:|
| kpx_group_1 | scada_under_label         |   1296 |                          11.3927  |                      12.9186  |
| kpx_group_1 | scada_zero_label_positive |    529 |                          11.9943  |                      14.7168  |
| kpx_group_1 | tight_reconstruction      |  22693 |                           7.14211 |                       7.58434 |
| kpx_group_2 | scada_under_label         |   1098 |                          11.1967  |                      13.1309  |
| kpx_group_2 | scada_zero_label_positive |    868 |                           9.92605 |                      11.6338  |
| kpx_group_2 | tight_reconstruction      |  21962 |                           7.24263 |                       7.68104 |
| kpx_group_3 | scada_under_label         |    341 |                          13.4467  |                      16.6758  |
| kpx_group_3 | scada_zero_label_positive |     94 |                          13.1708  |                      16.4603  |
| kpx_group_3 | tight_reconstruction      |  14493 |                           6.70013 |                       6.97632 |

### Interpretation

Non-tight SCADA regimes live in much higher NWP wind surfaces than tight reconstruction. That explains why a smooth power curve is not enough: high-wind rows include multiple operating/reporting regimes.

### Validity Discussion

This table alone is descriptive. Its validity comes from being consistent with the classifier and power-curve plots. There is still overlap, so hard wind-threshold rules would be brittle.

---

## 5. Event-level Structure

![Event duration](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/05_event_level_residual_duration.png)

| target      | taxonomy                  | start               | end                 |   duration_hours |   mean_abs_residual_ratio |   ldaps_event_minus_context |   gfs_event_minus_context |
|:------------|:--------------------------|:--------------------|:--------------------|-----------------:|--------------------------:|----------------------------:|--------------------------:|
| kpx_group_2 | scada_zero_label_positive | 2022-01-31 13:00:00 | 2022-02-07 11:00:00 |              167 |                  0.61555  |                    6.36268  |                   9.55934 |
| kpx_group_3 | scada_zero_label_positive | 2024-02-12 14:00:00 | 2024-02-15 09:00:00 |               68 |                  0.812631 |                    6.1653   |                  10.0282  |
| kpx_group_2 | scada_zero_label_positive | 2022-02-14 21:00:00 | 2022-02-17 10:00:00 |               62 |                  0.654449 |                   10.0328   |                  11.513   |
| kpx_group_2 | scada_zero_label_positive | 2022-02-20 16:00:00 | 2022-02-23 00:00:00 |               57 |                  0.554738 |                    2.63126  |                   6.89571 |
| kpx_group_1 | scada_zero_label_positive | 2022-12-17 16:00:00 | 2022-12-19 20:00:00 |               53 |                  0.474816 |                    3.96189  |                   9.2016  |

### Interpretation

The residual is event-level, not row-level noise. Long contiguous non-tight intervals exist and often coincide with high NWP wind context shifts relative to the 24h before/after window.

### Validity Discussion

The EDA does not prove causal labels like maintenance or curtailment. Without external operations logs, these are observable residual signatures only. That is enough for model gating, not enough for causal claims.

---

## 6. NWP-to-SCADA Bias

![NWP SCADA bias](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/06_nwp_scada_bias_taxonomy_heatmap.png)

The largest residual bins are dominated by `scada_zero_label_positive`, especially when NWP wind is several m/s above SCADA wind.

### Interpretation

This is the most actionable bridge from SCADA teacher knowledge to test-time modeling. We cannot observe SCADA in 2025, but we can build NWP-only proxies for high NWP-vs-actual-wind-bias regimes and only act when the direction is stable.

### Validity Discussion

The bias table itself uses train-only SCADA, so it cannot be directly applied to test. It defines what a deployable proxy must learn.

---

## 7. Power-curve Regime

![Power curve taxonomy](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/07_power_curve_taxonomy_by_nwp_wind.png)

### Interpretation

At comparable NWP wind speeds, taxonomy changes label response. `tight_reconstruction` follows a smooth curve, while `scada_under_label` and `scada_zero_label_positive` sit far away from that curve. A single NWP power curve cannot represent all regimes.

### Validity Discussion

The qcut wind bins stabilize the EDA view, but this is not a calibrated postprocessor. It proves that the next model should be regime-aware.

---

## 8. FiCR Boundary

![FiCR boundary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/scada_residual_deployability_eda_20260810/results/figures/08_ficr_boundary_by_scada_taxonomy.png)

Worst W4 anchor FiCR pass surfaces:

| target      | taxonomy                  |   rows |   eligible_rows |   ficr_pass_rate |   ficr_gold_rate |   near_10_actual_rows |   near_16_pred_rows |   mean_abs_err_norm |
|:------------|:--------------------------|-------:|----------------:|-----------------:|-----------------:|----------------------:|--------------------:|--------------------:|
| kpx_group_1 | low_online_label_positive |      4 |               4 |         0        |         0        |                     2 |                   0 |           0.347573  |
| kpx_group_3 | tight_reconstruction      |   7187 |            3015 |         0.172116 |         0.135801 |                   401 |                 407 |           0.0915074 |
| kpx_group_2 | moderate_residual         |    715 |             430 |         0.248951 |         0.188811 |                   105 |                  51 |           0.127736  |
| kpx_group_1 | scada_zero_label_positive |     36 |              36 |         0.25     |         0.194444 |                     5 |                   1 |           0.162031  |
| kpx_group_2 | tight_reconstruction      |   7612 |            4095 |         0.252102 |         0.195744 |                   335 |                 262 |           0.0894372 |

### Interpretation

Score action must be evaluated on FiCR transitions, not only MAE. Some SCADA taxonomies have high residual but different FiCR sensitivity. A good correction is one that creates more FiCR pass/gold transitions than it destroys while keeping NMAE cost controlled.

### Validity Discussion

This uses the W4 diagnostic anchor, so it is a local score surface. It is enough to prioritize experiments, not enough to authorize public submission.

---

## Final Conclusion

The SCADA-to-label lesson from #31 is **partially deployable**:

1. SCADA residual regimes are detectable from NWP/time features above random.
2. 2025 exposure is concentrated by month and group, so submission candidates need exposure-aware risk checks.
3. NWP-to-SCADA wind bias and power-curve regime are the best bridges from train-only SCADA to test-time features.
4. The next score experiment should be a probabilistic gated residual/bias model, not direct SCADA reconstruction and not broad feature dumping.

What remains to prove before promotion:

- **Detectability:** the regime gate remains valid on W3/W4 or another independent split.
- **Directionality:** the correction direction is stable inside the inferred regime.
- **Score payoff:** FiCR gain exceeds NMAE cost under the official metric.

# CV Protocol Results

## Environment

```json
{
  "python": "3.12.10",
  "pandas": "3.0.5",
  "numpy": "2.5.1",
  "sklearn": "1.9.0",
  "lightgbm": "4.7.0"
}
```

## Top Experiments

| experiment_id                        | model         | feature_set     | window                       |    score |   avg_nmae |   ficr_proxy |   worst_month_score |   high_generation_score |
|:-------------------------------------|:--------------|:----------------|:-----------------------------|---------:|-----------:|-------------:|--------------------:|------------------------:|
| B1_lgbm_W4_2022_2023_to_2024         | B1_lgbm       | B1_aggregate    | W4_2022_2023_to_2024         | 0.62567  |   0.129971 |     0.381311 |            0.60141  |                0.602028 |
| B1_lgbm_W5_2022_2023_recency_to_2024 | B1_lgbm       | B1_aggregate    | W5_2022_2023_recency_to_2024 | 0.623897 |   0.13074  |     0.378535 |            0.587616 |                0.600094 |
| B2_lgbm_W4_2022_2023_to_2024         | B2_lgbm       | B2_wind_physics | W4_2022_2023_to_2024         | 0.623247 |   0.131746 |     0.37824  |            0.598116 |                0.596269 |
| B2_lgbm_W5_2022_2023_recency_to_2024 | B2_lgbm       | B2_wind_physics | W5_2022_2023_recency_to_2024 | 0.621381 |   0.132283 |     0.375045 |            0.589034 |                0.591368 |
| B2_lgbm_W3_2023_to_2024              | B2_lgbm       | B2_wind_physics | W3_2023_to_2024              | 0.619951 |   0.132583 |     0.372486 |            0.588267 |                0.587137 |
| B1_lgbm_W3_2023_to_2024              | B1_lgbm       | B1_aggregate    | W3_2023_to_2024              | 0.61964  |   0.132399 |     0.371679 |            0.578591 |                0.591071 |
| B0_W5_2022_2023_recency_to_2024      | B0_month_hour | B0_time_mean    | W5_2022_2023_recency_to_2024 | 0.472203 |   0.251135 |     0.195542 |            0.363723 |                0.330715 |
| B0_W4_2022_2023_to_2024              | B0_month_hour | B0_time_mean    | W4_2022_2023_to_2024         | 0.471886 |   0.251619 |     0.195391 |            0.347671 |                0.329352 |
| B0_W3_2023_to_2024                   | B0_month_hour | B0_time_mean    | W3_2023_to_2024              | 0.47088  |   0.251958 |     0.193718 |            0.386854 |                0.33396  |

## Window Verdicts

| window                       |    score |   avg_nmae |   ficr_proxy |   worst_month_score |   delta_score_vs_W3 |   delta_nmae_vs_W3 |   months_improved_vs_W3 |   groups_improved_vs_W3 | verdict    | reason                                                            |
|:-----------------------------|---------:|-----------:|-------------:|--------------------:|--------------------:|-------------------:|------------------------:|------------------------:|:-----------|:------------------------------------------------------------------|
| W3_2023_to_2024              | 0.619796 |   0.132491 |     0.372083 |            0.583429 |          0          |        0           |                     nan |                     nan | Diagnostic | kept for comparison                                               |
| W4_2022_2023_to_2024         | 0.624459 |   0.130858 |     0.379776 |            0.599763 |          0.00466293 |       -0.00163288  |                       7 |                       2 | Support    | best mean B1/B2 score, but does not pass all stability criteria   |
| W5_2022_2023_recency_to_2024 | 0.622639 |   0.131511 |     0.37679  |            0.588325 |          0.00284354 |       -0.000979832 |                       8 |                       2 | Support    | stable positive alternative, but gain is below material threshold |

## Recommended Train Policy

Provisional T1/T6 candidate: full-history is supported for group 1/2, but needs public-LB calibration; group 3 remains recent-window because 2022 labels are absent.

## Caveats

- The official page confirms the group FICR structure, but the exact per-hour settlement table remains in a DACON code-download attachment that was not accessible in this run.
- FICR therefore uses a 6%/8% threshold proxy until the official code attachment is obtained.
- Public/private LB calibration still needs actual submissions.
- Group 3 has only 2023-2024 usable labels, so its train-window evidence is weaker.
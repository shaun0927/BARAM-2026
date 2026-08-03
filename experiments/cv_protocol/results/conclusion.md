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

| experiment_id                        | model         | feature_set     | window                       |    score |   avg_nmae |     ficr |   worst_month_score |   high_generation_score |
|:-------------------------------------|:--------------|:----------------|:-----------------------------|---------:|-----------:|---------:|--------------------:|------------------------:|
| B1_lgbm_W4_2022_2023_to_2024         | B1_lgbm       | B1_aggregate    | W4_2022_2023_to_2024         | 0.611767 |   0.129971 | 0.353506 |            0.591715 |                0.600197 |
| B1_lgbm_W5_2022_2023_recency_to_2024 | B1_lgbm       | B1_aggregate    | W5_2022_2023_recency_to_2024 | 0.609705 |   0.13074  | 0.350149 |            0.574091 |                0.59778  |
| B2_lgbm_W4_2022_2023_to_2024         | B2_lgbm       | B2_wind_physics | W4_2022_2023_to_2024         | 0.607669 |   0.131746 | 0.347084 |            0.585863 |                0.594585 |
| B2_lgbm_W5_2022_2023_recency_to_2024 | B2_lgbm       | B2_wind_physics | W5_2022_2023_recency_to_2024 | 0.60434  |   0.132283 | 0.340964 |            0.577188 |                0.589636 |
| B1_lgbm_W3_2023_to_2024              | B1_lgbm       | B1_aggregate    | W3_2023_to_2024              | 0.603268 |   0.132399 | 0.338935 |            0.558331 |                0.58845  |
| B2_lgbm_W3_2023_to_2024              | B2_lgbm       | B2_wind_physics | W3_2023_to_2024              | 0.600645 |   0.132583 | 0.333874 |            0.569492 |                0.584089 |
| B0_W5_2022_2023_recency_to_2024      | B0_month_hour | B0_time_mean    | W5_2022_2023_recency_to_2024 | 0.43369  |   0.251135 | 0.118515 |            0.329566 |                0.326112 |
| B0_W4_2022_2023_to_2024              | B0_month_hour | B0_time_mean    | W4_2022_2023_to_2024         | 0.433655 |   0.251619 | 0.118929 |            0.316567 |                0.32481  |
| B0_W3_2023_to_2024                   | B0_month_hour | B0_time_mean    | W3_2023_to_2024              | 0.43271  |   0.251958 | 0.117378 |            0.352742 |                0.329482 |

## Window Verdicts

| window                       |    score |   avg_nmae |     ficr |   worst_month_score |   delta_score_vs_W3 |   delta_nmae_vs_W3 |   months_improved_vs_W3 |   groups_improved_vs_W3 | verdict    | reason                                                             |
|:-----------------------------|---------:|-----------:|---------:|--------------------:|--------------------:|-------------------:|------------------------:|------------------------:|:-----------|:-------------------------------------------------------------------|
| W3_2023_to_2024              | 0.601957 |   0.132491 | 0.336405 |            0.563911 |          0          |        0           |                     nan |                     nan | Diagnostic | kept for comparison                                                |
| W4_2022_2023_to_2024         | 0.609718 |   0.130858 | 0.350295 |            0.588789 |          0.00776177 |       -0.00163288  |                       8 |                       2 | Adopt      | best mean B1/B2 score and passes material/stability/group criteria |
| W5_2022_2023_recency_to_2024 | 0.607023 |   0.131511 | 0.345557 |            0.575639 |          0.00506588 |       -0.000979832 |                       8 |                       2 | Support    | materially improves over W3 on mean B1/B2 metrics                  |

## Recommended Train Policy

Adopted T1/T6 candidate: full-history is supported for group 1/2, but needs public-LB calibration; group 3 remains recent-window because 2022 labels are absent.

## Caveats

- FICR now follows the official DACON code-share notebook: error-rate thresholds 6%/8%, unit prices 4/3/0, settlement weighted by actual generation.
- Public/private LB calibration still needs actual submissions.
- Group 3 has only 2023-2024 usable labels, so its train-window evidence is weaker.
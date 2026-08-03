# FICR-aware Postprocessing Results

## Baseline

{
  "experiment_id": "baseline_w4",
  "score": 0.6117673564727266,
  "one_minus_nmae": 0.8700286771288893,
  "avg_nmae": 0.12997132287111068,
  "ficr": 0.3535060358165638,
  "worst_month_score": 0.5917149198256562,
  "high_generation_score": 0.6001969855349586,
  "alphas": {
    "kpx_group_1": 1.0,
    "kpx_group_2": 1.0,
    "kpx_group_3": 1.0
  }
}

## Top Experiments

| experiment_id              |    score |   delta_score |   one_minus_nmae |   delta_one_minus_nmae |     ficr |   delta_ficr |   worst_month_score |   delta_worst_month_score | verdict   |
|:---------------------------|---------:|--------------:|-----------------:|-----------------------:|---------:|-------------:|--------------------:|--------------------------:|:----------|
| A2_group_1.030_1.030_1.030 | 0.621355 |    0.00958789 |         0.871202 |            0.00117347  | 0.371508 |    0.0180023 |            0.602684 |                 0.0109689 | Reject    |
| A1_global_1.0300           | 0.621355 |    0.00958789 |         0.871202 |            0.00117347  | 0.371508 |    0.0180023 |            0.602684 |                 0.0109689 | Reject    |
| A3_shrink_1.00             | 0.621355 |    0.00958789 |         0.871202 |            0.00117347  | 0.371508 |    0.0180023 |            0.602684 |                 0.0109689 | Reject    |
| A2_group_1.030_1.030_1.025 | 0.621168 |    0.00940075 |         0.871108 |            0.00107932  | 0.371228 |    0.0177222 |            0.602589 |                 0.0108746 | Reject    |
| A2_group_1.030_1.020_1.030 | 0.621144 |    0.00937617 |         0.87133  |            0.00130112  | 0.370957 |    0.0174512 |            0.602876 |                 0.011161  | Reject    |
| A2_group_1.030_1.025_1.030 | 0.621117 |    0.00934933 |         0.871273 |            0.00124428  | 0.37096  |    0.0174544 |            0.602617 |                 0.0109021 | Reject    |
| A2_group_1.030_1.020_1.025 | 0.620956 |    0.00918903 |         0.871236 |            0.00120696  | 0.370677 |    0.0171711 |            0.602782 |                 0.0110666 | Reject    |
| A2_group_1.030_1.025_1.025 | 0.62093  |    0.00916219 |         0.871179 |            0.00115012  | 0.37068  |    0.0171743 |            0.602523 |                 0.0108078 | Reject    |
| A2_group_1.030_1.015_1.030 | 0.620805 |    0.00903755 |         0.87137  |            0.00134175  | 0.370239 |    0.0167333 |            0.604499 |                 0.0127841 | Reject    |
| A2_group_1.030_1.030_1.020 | 0.620761 |    0.00899397 |         0.871008 |            0.000979603 | 0.370514 |    0.0170083 |            0.602051 |                 0.0103357 | Reject    |

## Best Params

{
  "best_by_score": {
    "experiment_id": "A1_global_1.0275",
    "method": "A1_global",
    "verdict": "Strong accept",
    "reason": "passes strong score/FiCR/nMAE/worst-month criteria",
    "score": 0.6207259749431353,
    "delta_score": 0.008958618470408686,
    "one_minus_nmae": 0.8711552359099702,
    "delta_one_minus_nmae": 0.0011265587810809219,
    "ficr": 0.3702967139763003,
    "delta_ficr": 0.016790678159736505,
    "worst_month_score": 0.6030160002282019,
    "delta_worst_month_score": 0.011301080402545738,
    "alphas": {
      "kpx_group_1": 1.0275,
      "kpx_group_2": 1.0275,
      "kpx_group_3": 1.0275
    }
  },
  "phase_1_submission_candidate_created": true
}

## Submission

{
  "path": "C:\\Users\\USER\\Desktop\\jh0927\\BARAM-2026\\experiments\\ficr_postprocessing\\submissions\\submission_w4_postprocessed_A1_global.csv",
  "rows": 8760,
  "missing_predictions": 0,
  "min_prediction": 0.0,
  "max_prediction": 21600.0,
  "columns": [
    "forecast_id",
    "forecast_kst_dtm",
    "kpx_group_1",
    "kpx_group_2",
    "kpx_group_3"
  ]
}

## Interpretation

- Phase 1 is deliberately low-parameter: global, group-wise, and shrinked group-wise alpha only.
- A candidate is submission-worthy only if it improves local score and FiCR without meaningful nMAE/worst-month damage.
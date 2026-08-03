# Model Screening Results

## Baseline

- local baseline score: 0.6207259749
- Public LB baseline: 0.6220908318
- fixed alpha baseline: 1.0275

## Top Models

| model_id         | model_family                 |   raw_score |   best_alpha |   alpha_score |   delta_vs_baseline |   alpha_ficr |   worst_month_score |   high_generation_score |   months_better_than_lgbm |   groups_better_than_lgbm | status   |
|:-----------------|:-----------------------------|------------:|-------------:|--------------:|--------------------:|-------------:|--------------------:|------------------------:|--------------------------:|--------------------------:|:---------|
| lgbm_l1_baseline | LightGBM                     |    0.611767 |       1.03   |      0.621355 |         0.000629272 |     0.371508 |            0.602684 |                0.615929 |                         0 |                         0 | ok       |
| xgboost_l1       | XGBoost                      |    0.608979 |       1.0275 |      0.61764  |        -0.00308568  |     0.365071 |            0.586263 |                0.607879 |                         3 |                         1 | ok       |
| catboost_mae     | CatBoost                     |    0.607359 |       1.03   |      0.614037 |        -0.0066894   |     0.35952  |            0.593512 |                0.611968 |                         1 |                         1 | ok       |
| hist_gbdt_l2     | sklearn HistGradientBoosting |    0.603296 |       1.03   |      0.613442 |        -0.00728362  |     0.357727 |            0.582279 |                0.605391 |                         4 |                         0 | ok       |
| lgbm_l2          | LightGBM                     |    0.601581 |       1.03   |      0.610883 |        -0.00984262  |     0.353107 |            0.576897 |                0.602171 |                         3 |                         0 | ok       |
| extra_trees      | sklearn ExtraTrees           |    0.593241 |       1.03   |      0.608998 |        -0.0117276   |     0.346795 |            0.566407 |                0.592101 |                         2 |                         0 | ok       |
| random_forest    | sklearn RandomForest         |    0.59278  |       1.03   |      0.60601  |        -0.0147163   |     0.343097 |            0.58278  |                0.588892 |                         1 |                         0 | ok       |
| ridge            | linear                       |    0.58253  |       1.03   |      0.590379 |        -0.0303466   |     0.31785  |            0.552896 |                0.565389 |                         0 |                         0 | ok       |
| lgbm_huber       | LightGBM                     |    0.419151 |       1.03   |      0.421778 |        -0.198948    |     0.105759 |            0.353173 |                0.2883   |                         0 |                         0 | ok       |
| elastic_net      | linear                       |  nan        |     nan      |    nan        |       nan           |   nan        |          nan        |              nan        |                       nan |                       nan | skipped  |
| huber            | linear                       |  nan        |     nan      |    nan        |       nan           |   nan        |          nan        |              nan        |                       nan |                       nan | skipped  |
| tabm_feasibility | TabM                         |  nan        |     nan      |    nan        |       nan           |   nan        |          nan        |              nan        |                       nan |                       nan | skipped  |

## Top Ensembles

| ensemble_id                                               | members                                    |   best_alpha |   alpha_score |   delta_vs_baseline |   alpha_ficr |   worst_month_score |   high_generation_score |
|:----------------------------------------------------------|:-------------------------------------------|-------------:|--------------:|--------------------:|-------------:|--------------------:|------------------------:|
| w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.7_0.1_0.2 | lgbm_l1_baseline+catboost_mae+hist_gbdt_l2 |         1.03 |      0.622063 |         0.00133684  |     0.371998 |            0.597979 |                0.616313 |
| w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.7_0.2_0.1 | lgbm_l1_baseline+catboost_mae+hist_gbdt_l2 |         1.03 |      0.621882 |         0.00115597  |     0.371788 |            0.601027 |                0.61691  |
| w3_lgbm_l1_baseline_catboost_mae_lgbm_l2_0.6_0.2_0.2      | lgbm_l1_baseline+catboost_mae+lgbm_l2      |         1.03 |      0.621794 |         0.00106767  |     0.371571 |            0.603982 |                0.616327 |
| w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.6_0.2_0.2 | lgbm_l1_baseline+catboost_mae+hist_gbdt_l2 |         1.03 |      0.621683 |         0.000956989 |     0.371166 |            0.600814 |                0.616052 |
| w2_lgbm_l1_baseline_xgboost_l1_0.95                       | lgbm_l1_baseline+xgboost_l1                |         1.03 |      0.621617 |         0.000891248 |     0.371962 |            0.602911 |                0.615913 |
| w3_lgbm_l1_baseline_catboost_mae_lgbm_l2_0.8_0.1_0.1      | lgbm_l1_baseline+catboost_mae+lgbm_l2      |         1.03 |      0.621568 |         0.000841871 |     0.371368 |            0.602577 |                0.61628  |
| w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.6_0.1_0.3 | lgbm_l1_baseline+catboost_mae+hist_gbdt_l2 |         1.03 |      0.621498 |         0.000771884 |     0.370753 |            0.599251 |                0.615397 |
| w2_lgbm_l1_baseline_extra_trees_0.70                      | lgbm_l1_baseline+extra_trees               |         1.03 |      0.621456 |         0.000729748 |     0.370047 |            0.602504 |                0.612493 |
| w2_lgbm_l1_baseline_extra_trees_0.90                      | lgbm_l1_baseline+extra_trees               |         1.03 |      0.621437 |         0.000711433 |     0.370955 |            0.605344 |                0.615092 |
| w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.5_0.2_0.3 | lgbm_l1_baseline+catboost_mae+hist_gbdt_l2 |         1.03 |      0.621419 |         0.000693126 |     0.370557 |            0.600062 |                0.615273 |
| w2_lgbm_l1_baseline_hist_gbdt_l2_0.85                     | lgbm_l1_baseline+hist_gbdt_l2              |         1.03 |      0.62141  |         0.000684276 |     0.371019 |            0.602582 |                0.615179 |
| w3_lgbm_l1_baseline_xgboost_l1_catboost_mae_0.8_0.1_0.1   | lgbm_l1_baseline+xgboost_l1+catboost_mae   |         1.03 |      0.621362 |         0.000635588 |     0.371144 |            0.600755 |                0.615744 |

## Selected Candidate

{
  "kind": "ensemble",
  "candidate": {
    "ensemble_id": "w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.7_0.1_0.2",
    "model_id": "w3_lgbm_l1_baseline_catboost_mae_hist_gbdt_l2_0.7_0.1_0.2",
    "members": "lgbm_l1_baseline+catboost_mae+hist_gbdt_l2",
    "weights": "{\"lgbm_l1_baseline\": 0.7, \"catboost_mae\": 0.1, \"hist_gbdt_l2\": 0.2}",
    "best_alpha": 1.03,
    "alpha_score": 0.6220628143285927,
    "alpha_1_nmae": 0.8721273947713679,
    "alpha_ficr": 0.37199823388581743,
    "avg_nmae": 0.12787260522863209,
    "worst_month_score": 0.5979793105155076,
    "high_generation_score": 0.6163130623492746,
    "delta_vs_baseline": 0.00133683942859264
  },
  "submission": null
}
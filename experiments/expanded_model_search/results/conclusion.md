# Expanded Model Search Results

## Fixed protocol

- scope: GitHub issue #4 expanded model-family / AutoML first-pass screening
- local validation: train 2022-2023, valid 2024
- feature baseline: B1 aggregate only
- metric: official DACON score
- alpha grid: 0.9700 to 1.0300, step 0.0025
- baseline to beat: W4 + LightGBM L1 + global alpha 1.0275
- local baseline score: 0.6207259749
- Public LB baseline: 0.6220908318
- acceptance score threshold: 0.6222259749

## Execution coverage

- successful model predictions: 25
- failed/skipped metadata rows: 1
- deep Optuna tuning: intentionally not performed; this run is broad model exploration.
- feature engineering expansion: intentionally not performed.

## Top single models

| model_id                   | model_family                 |   raw_score |   best_alpha |   alpha_score |   delta_vs_baseline |   alpha_ficr |   worst_month_score |   high_generation_score |   months_better_than_lgbm |   groups_better_than_lgbm | status   | notes                                                                                                                                                                                                                                                                          |
|:---------------------------|:-----------------------------|------------:|-------------:|--------------:|--------------------:|-------------:|--------------------:|------------------------:|--------------------------:|--------------------------:|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| catboost_quantile_055      | CatBoost                     |    0.618851 |        1.025 |      0.622655 |         0.00192913  |     0.374994 |            0.593283 |                0.629268 |                         6 |                         1 | ok       | CatBoost Quantile q=0.55 first-pass; loaded cached prediction                                                                                                                                                                                                                  |
| lgbm_l1_baseline           | LightGBM                     |    0.611767 |        1.03  |      0.621355 |         0.000629272 |     0.371508 |            0.602684 |                0.615929 |                         0 |                         0 | ok       | current B1/W4 baseline model; included as anchor; loaded cached prediction                                                                                                                                                                                                     |
| hist_gbdt_absolute         | sklearn HistGradientBoosting |    0.610621 |        1.03  |      0.618716 |        -0.0020104   |     0.366612 |            0.598786 |                0.612131 |                         4 |                         1 | ok       | sklearn boosting absolute-error variant; loaded cached prediction                                                                                                                                                                                                              |
| catboost_rmse              | CatBoost                     |    0.603653 |        1.03  |      0.613299 |        -0.00742733  |     0.355291 |            0.579511 |                0.603821 |                         5 |                         1 | ok       | CatBoost RMSE first-pass diversity; loaded cached prediction                                                                                                                                                                                                                   |
| gradient_boosting_quantile | sklearn GradientBoosting     |    0.602959 |        1.03  |      0.612811 |        -0.00791506  |     0.356054 |            0.58569  |                0.61112  |                         3 |                         1 | ok       | quantile boosting q=0.55 for underprediction-bias diversity; loaded cached prediction                                                                                                                                                                                          |
| xgboost_squarederror       | XGBoost                      |    0.603165 |        1.03  |      0.612186 |        -0.00854043  |     0.354811 |            0.584744 |                0.601634 |                         3 |                         0 | ok       | XGBoost squared-error diversity baseline; loaded cached prediction                                                                                                                                                                                                             |
| h2o_automl                 | H2O AutoML                   |    0.603701 |        1.03  |      0.61102  |        -0.0097058   |     0.353858 |            0.585629 |                0.604983 |                         2 |                         0 | ok       | loaded cached prediction                                                                                                                                                                                                                                                       |
| extra_trees_wide           | sklearn ExtraTrees           |    0.593617 |        1.03  |      0.608834 |        -0.0118924   |     0.346406 |            0.56642  |                0.592182 |                         2 |                         0 | ok       | bagging diversity; wider than #4 first run; loaded cached prediction                                                                                                                                                                                                           |
| autogluon_tabular          | AutoGluon Tabular            |    0.596442 |        1.03  |      0.606223 |        -0.0145027   |     0.343935 |            0.582579 |                0.590323 |                         2 |                         0 | ok       | AutoGluon first-pass; time_limit_per_target=60; leaders={'kpx_group_1': 'WeightedEnsemble_L2', 'kpx_group_2': 'WeightedEnsemble_L2', 'kpx_group_3': 'WeightedEnsemble_L2'}                                                                                                     |
| random_forest_wide         | sklearn RandomForest         |    0.591936 |        1.03  |      0.605499 |        -0.0152267   |     0.342628 |            0.579614 |                0.587739 |                         2 |                         0 | ok       | bagging diversity; wider than #4 first run; loaded cached prediction                                                                                                                                                                                                           |
| pycaret_compare            | PyCaret                      |    0.593383 |        1.03  |      0.604458 |        -0.0162682   |     0.339967 |            0.57507  |                0.586389 |                         2 |                         0 | ok       | PyCaret compare_models first-pass; budget_time_minutes=0.5; leaders={'kpx_group_1': 'RandomForestRegressor(n_jobs=-1, random_state=42)', 'kpx_group_2': 'RandomForestRegressor(n_jobs=-1, random_state=42)', 'kpx_group_3': 'ExtraTreesRegressor(n_jobs=-1, random_state=42)'} |
| ngboost_normal             | NGBoost                      |    0.591864 |        1.03  |      0.601892 |        -0.0188337   |     0.337286 |            0.551372 |                0.593473 |                         0 |                         0 | ok       | NGBoost Normal distribution first-pass sampled                                                                                                                                                                                                                                 |
| explainable_boosting       | ExplainableBoostingMachine   |    0.592907 |        1.03  |      0.599764 |        -0.020962    |     0.335127 |            0.581229 |                0.58553  |                         2 |                         0 | ok       | EBM first-pass sampled                                                                                                                                                                                                                                                         |
| flaml_automl               | FLAML AutoML                 |    0.586097 |        1.03  |      0.593964 |        -0.0267618   |     0.330428 |            0.569678 |                0.583565 |                         1 |                         0 | ok       | loaded cached prediction                                                                                                                                                                                                                                                       |
| huber_regressor            | linear                       |    0.583677 |        1.03  |      0.59097  |        -0.0297565   |     0.320153 |            0.565562 |                0.567053 |                         0 |                         1 | ok       | previously deferred; robust linear baseline; loaded cached prediction                                                                                                                                                                                                          |
| elastic_net                | linear                       |    0.577547 |        1.03  |      0.584239 |        -0.036487    |     0.308356 |            0.552306 |                0.556693 |                         0 |                         0 | ok       | previously deferred; first-pass linear sparse baseline; loaded cached prediction                                                                                                                                                                                               |
| tabm_torch                 | TabM                         |    0.551007 |        1.03  |      0.559015 |        -0.0617112   |     0.274757 |            0.506754 |                0.528658 |                         0 |                         0 | ok       | TabM torch wrapper first-pass; epochs=8; sampled rows                                                                                                                                                                                                                          |
| knn_distance               | neighbors                    |    0.52099  |        1.03  |      0.532531 |        -0.0881946   |     0.233233 |            0.456218 |                0.483464 |                         0 |                         0 | ok       | distance-weighted KNN with robust scaling; loaded cached prediction                                                                                                                                                                                                            |
| mlp_regressor              | neural_sklearn               |    0.479114 |        1.03  |      0.485895 |        -0.134831    |     0.180944 |            0.429264 |                0.439114 |                         0 |                         0 | ok       | sklearn MLP first-pass tabular neural baseline; loaded cached prediction                                                                                                                                                                                                       |
| nystroem_ridge             | kernel                       |    0.451638 |        1.03  |      0.455788 |        -0.164938    |     0.135367 |            0.379049 |                0.327942 |                         1 |                         0 | ok       | kernel approximation + ridge; nonlinear low-rank diversity; loaded cached prediction                                                                                                                                                                                           |

## Top ensembles

| ensemble_id                                                                         | members                                                           |   best_alpha |   alpha_score |   delta_vs_baseline |   alpha_ficr |   worst_month_score |   high_generation_score |
|:------------------------------------------------------------------------------------|:------------------------------------------------------------------|-------------:|--------------:|--------------------:|-------------:|--------------------:|------------------------:|
| w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.60_0.20_0.20           | catboost_quantile_055+lgbm_l1_baseline+extra_trees_wide           |       1.03   |      0.62609  |          0.00536411 |     0.379096 |            0.601247 |                0.626865 |
| w3_catboost_quantile_055_lgbm_l1_baseline_h2o_automl_0.70_0.20_0.10                 | catboost_quantile_055+lgbm_l1_baseline+h2o_automl                 |       1.03   |      0.625978 |          0.00525224 |     0.379813 |            0.603766 |                0.629529 |
| w2_lgbm_catboost_quantile_055_0.50                                                  | lgbm_l1_baseline+catboost_quantile_055                            |       1.03   |      0.625848 |          0.00512234 |     0.379269 |            0.602986 |                0.627006 |
| w3_catboost_quantile_055_lgbm_l1_baseline_h2o_automl_0.60_0.20_0.20                 | catboost_quantile_055+lgbm_l1_baseline+h2o_automl                 |       1.03   |      0.625708 |          0.00498244 |     0.378852 |            0.605976 |                0.62778  |
| w3_catboost_quantile_055_lgbm_l1_baseline_h2o_automl_0.70_0.10_0.20                 | catboost_quantile_055+lgbm_l1_baseline+h2o_automl                 |       1.03   |      0.625587 |          0.004861   |     0.379017 |            0.605643 |                0.628736 |
| w3_catboost_quantile_055_lgbm_l1_baseline_xgboost_squarederror_0.60_0.20_0.20       | catboost_quantile_055+lgbm_l1_baseline+xgboost_squarederror       |       1.03   |      0.625512 |          0.00478626 |     0.37861  |            0.604798 |                0.626744 |
| w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.70_0.20_0.10           | catboost_quantile_055+lgbm_l1_baseline+extra_trees_wide           |       1.03   |      0.625489 |          0.00476287 |     0.378591 |            0.602923 |                0.628554 |
| w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.70_0.10_0.20           | catboost_quantile_055+lgbm_l1_baseline+extra_trees_wide           |       1.03   |      0.625413 |          0.00468727 |     0.378141 |            0.600991 |                0.627924 |
| w3_catboost_quantile_055_lgbm_l1_baseline_hist_gbdt_absolute_0.60_0.20_0.20         | catboost_quantile_055+lgbm_l1_baseline+hist_gbdt_absolute         |       1.0275 |      0.625318 |          0.00459177 |     0.378332 |            0.601336 |                0.626756 |
| w3_catboost_quantile_055_lgbm_l1_baseline_xgboost_squarederror_0.70_0.20_0.10       | catboost_quantile_055+lgbm_l1_baseline+xgboost_squarederror       |       1.03   |      0.62521  |          0.00448389 |     0.378383 |            0.604305 |                0.628362 |
| w2_lgbm_catboost_quantile_055_0.55                                                  | lgbm_l1_baseline+catboost_quantile_055                            |       1.03   |      0.624997 |          0.00427138 |     0.377532 |            0.601131 |                0.625039 |
| w3_catboost_quantile_055_lgbm_l1_baseline_gradient_boosting_quantile_0.60_0.20_0.20 | catboost_quantile_055+lgbm_l1_baseline+gradient_boosting_quantile |       1.03   |      0.624938 |          0.0042122  |     0.377316 |            0.604359 |                0.627954 |
| w3_catboost_quantile_055_lgbm_l1_baseline_h2o_automl_0.33_0.33_0.33                 | catboost_quantile_055+lgbm_l1_baseline+h2o_automl                 |       1.03   |      0.624912 |          0.00418587 |     0.376931 |            0.602471 |                0.623275 |
| w2_lgbm_catboost_quantile_055_0.60                                                  | lgbm_l1_baseline+catboost_quantile_055                            |       1.03   |      0.624907 |          0.00418139 |     0.377356 |            0.603027 |                0.624288 |
| w3_catboost_quantile_055_lgbm_l1_baseline_hist_gbdt_absolute_0.70_0.20_0.10         | catboost_quantile_055+lgbm_l1_baseline+hist_gbdt_absolute         |       1.03   |      0.624816 |          0.00409013 |     0.377691 |            0.601063 |                0.627746 |
| w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.33_0.33_0.33           | catboost_quantile_055+lgbm_l1_baseline+extra_trees_wide           |       1.03   |      0.624791 |          0.00406479 |     0.375826 |            0.602996 |                0.620789 |
| w3_catboost_quantile_055_lgbm_l1_baseline_gradient_boosting_quantile_0.70_0.20_0.10 | catboost_quantile_055+lgbm_l1_baseline+gradient_boosting_quantile |       1.03   |      0.624693 |          0.00396657 |     0.377279 |            0.602162 |                0.62879  |
| w3_catboost_quantile_055_lgbm_l1_baseline_h2o_automl_0.80_0.10_0.10                 | catboost_quantile_055+lgbm_l1_baseline+h2o_automl                 |       1.0275 |      0.624669 |          0.0039427  |     0.377641 |            0.602137 |                0.628984 |
| w3_catboost_quantile_055_lgbm_l1_baseline_catboost_rmse_0.70_0.20_0.10              | catboost_quantile_055+lgbm_l1_baseline+catboost_rmse              |       1.03   |      0.624558 |          0.00383201 |     0.377135 |            0.602385 |                0.627898 |
| w3_catboost_quantile_055_lgbm_l1_baseline_xgboost_squarederror_0.70_0.10_0.20       | catboost_quantile_055+lgbm_l1_baseline+xgboost_squarederror       |       1.03   |      0.624544 |          0.00381785 |     0.377072 |            0.604236 |                0.627056 |

## Lowest residual-correlation candidates

| model_id            |   mean_pearson |   better_abs_error_rate |
|:--------------------|---------------:|------------------------:|
| radius_neighbors    |       0.391207 |               0.0446464 |
| xgboost_pseudohuber |       0.394565 |               0.0447984 |
| sgd_huber           |       0.436927 |               0.0551789 |
| svr_rbf_sampled     |       0.445539 |               0.274504  |
| nystroem_ridge      |       0.484064 |               0.32242   |
| linear_svr          |       0.622022 |               0.119781  |
| mlp_regressor       |       0.632949 |               0.295706  |
| knn_distance        |       0.677745 |               0.397024  |
| tabm_torch          |       0.762435 |               0.41557   |
| elastic_net         |       0.826932 |               0.457065  |
| huber_regressor     |       0.829014 |               0.463802  |
| flaml_automl        |       0.876601 |               0.448966  |

## Selected candidate

{
  "kind": "ensemble",
  "candidate": {
    "ensemble_id": "w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.60_0.20_0.20",
    "model_id": "w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.60_0.20_0.20",
    "members": "catboost_quantile_055+lgbm_l1_baseline+extra_trees_wide",
    "weights": "{\"catboost_quantile_055\": 0.6, \"lgbm_l1_baseline\": 0.2, \"extra_trees_wide\": 0.2}",
    "best_alpha": 1.03,
    "alpha_score": 0.6260900871556871,
    "alpha_1_nmae": 0.8730840482283418,
    "alpha_ficr": 0.37909612608303256,
    "avg_nmae": 0.12691595177165818,
    "delta_vs_baseline": 0.005364112255687115,
    "worst_month_score": 0.6012469332924494,
    "high_generation_score": 0.626865165838502
  },
  "accepted": true,
  "submission": {
    "created": true,
    "path": "C:\\Users\\USER\\Desktop\\jh0927\\BARAM-2026\\experiments\\expanded_model_search\\submissions\\submission_w3_catboost_quantile_055_lgbm_l1_baseline_extra_trees_wide_0.60_0.20_0.20_alpha.csv",
    "rows": 8760,
    "missing_predictions": 0,
    "min_prediction": 2.9187148148148148,
    "max_prediction": 21600.0,
    "columns_match": true,
    "ids_match": true,
    "dt_match": true
  }
}

## Answer: is model exploration complete enough?

This run completes the practical #4 model-family exploration pass under the fixed B1/W4/official-metric/alpha-grid protocol. It covers manual first-pass baselines, AutoML systems, tabular foundation/deep candidates, linear/robust/kernel/neighbor/neural alternatives, NGBoost, and EBM where the environment supports them.

A model-family branch should only remain open if it failed for a concrete runtime/API reason or if a successful model shows strong residual diversity but needs candidate-specific reproduction. Hyperparameter tuning is a follow-up only for families that show standalone or ensemble value.

## Failed or skipped models

| model_id         | model_family   | status   | notes                                                                                                                                                   |
|:-----------------|:---------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| tabpfn_regressor | TabPFN         | failed   | TabPFN requires one-time Prior Labs license/API-key authentication before local weight download; non-interactive run failed with OSError WinError 10038 |
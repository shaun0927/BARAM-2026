# Next Audit Routing

## Anchor reconstruction

- score: 0.6314912599 (expected 0.6314912599)
- FiCR: 0.3908188625 (expected 0.3908188625)
- worst_month: 0.6038740233 (expected 0.6038740233)
- high_generation_score: 0.6425350509

## Main routing decision

- recommended next issue: **generation-regime feature sufficiency audit with q-policy diagnostic**
- priority: high
- evidence: Mid-generation bins dominate total impact, but high-generation bins have severe underprediction and q=0.60 slice wins. Audit whether B1 lacks features to separate mid-generation overprediction from high-generation underprediction.
- what not to do yet: do not start broad HPO, do not apply data cleansing, and do not add feature families before the routed audit confirms the intervention.

## H1: systematic underprediction

- weighted target-level underprediction rate: 0.3638
- target-level normalized bias spread: 0.0239
- mid-generation underprediction rate (10~80%): 0.4568
- high-generation underprediction rate (80%+): 0.7908
- interpretation: there is no global underprediction if target-level underprediction is below 0.5, but high-generation-specific underprediction can still justify q/policy diagnostics.

## H2: quantile policy slices

| target      | actual_bin   | abs_error_winner              |   winner_count |   slice_count |   winner_rate |
|:------------|:-------------|:------------------------------|---------------:|--------------:|--------------:|
| kpx_group_3 | 90-100%      | catboost_quantile_060_depth6  |            237 |           438 |      0.541096 |
| kpx_group_3 | 80-90%       | catboost_quantile_060_depth6  |            141 |           274 |      0.514599 |
| kpx_group_1 | 90-100%      | catboost_quantile_060_depth6  |            155 |           392 |      0.395408 |
| kpx_group_1 | 80-90%       | catboost_quantile_060_depth6  |            166 |           448 |      0.370536 |
| kpx_group_2 | 90-100%      | catboost_quantile_060_depth6  |            125 |           355 |      0.352113 |
| kpx_group_3 | 60-80%       | catboost_quantile_060_depth6  |            282 |           899 |      0.313682 |
| kpx_group_2 | 80-90%       | catboost_quantile_060_depth6  |            178 |           598 |      0.297659 |
| kpx_group_1 | 60-80%       | catboost_quantile_060_depth6  |            306 |          1111 |      0.275428 |
| kpx_group_2 | 60-80%       | catboost_quantile_060_depth6  |            284 |          1125 |      0.252444 |
| kpx_group_3 | 90-100%      | catboost_quantile_0575_depth6 |            106 |           438 |      0.242009 |

## H3/H4: top remaining problem slices

| target      | actual_bin   |   count |   eligible_count |   normalized_abs_error |   ficr_fail_rate |   underprediction_rate |   impact_proxy |
|:------------|:-------------|--------:|-----------------:|-----------------------:|-----------------:|-----------------------:|---------------:|
| kpx_group_2 | 30-60%       |    1462 |             1462 |              0.156493  |         0.699042 |               0.422709 |       228.793  |
| kpx_group_1 | 30-60%       |    1484 |             1484 |              0.140057  |         0.659704 |               0.435984 |       207.845  |
| kpx_group_3 | 30-60%       |    1448 |             1448 |              0.142959  |         0.688536 |               0.421961 |       207.005  |
| kpx_group_2 | 10-30%       |    1436 |             1436 |              0.137659  |         0.549443 |               0.431755 |       197.678  |
| kpx_group_3 | 10-30%       |    1507 |             1507 |              0.118847  |         0.526875 |               0.445255 |       179.102  |
| kpx_group_1 | 10-30%       |    1554 |             1554 |              0.112445  |         0.521879 |               0.460746 |       174.739  |
| kpx_group_2 | 60-80%       |    1125 |             1125 |              0.118417  |         0.596444 |               0.426667 |       133.22   |
| kpx_group_1 | 60-80%       |    1111 |             1111 |              0.11068   |         0.544554 |               0.522952 |       122.965  |
| kpx_group_3 | 60-80%       |     899 |              899 |              0.117017  |         0.523915 |               0.611791 |       105.198  |
| kpx_group_3 | 90-100%      |     438 |              438 |              0.217881  |         0.974886 |               1        |        95.4321 |
| kpx_group_2 | 80-90%       |     598 |              598 |              0.0790522 |         0.314381 |               0.553512 |        47.2732 |
| kpx_group_1 | 80-90%       |     448 |              448 |              0.103977  |         0.450893 |               0.785714 |        46.5817 |

## Highest-impact months

|   month |   count |   eligible_count |   normalized_abs_error |   ficr_fail_rate |   underprediction_rate |   impact_proxy |
|--------:|--------:|-----------------:|-----------------------:|-----------------:|-----------------------:|---------------:|
|       1 |    2232 |             1492 |              0.172157  |         0.398297 |               0.326613 |        256.859 |
|      12 |    2232 |             2038 |              0.112714  |         0.508961 |               0.414427 |        229.711 |
|       3 |    2232 |             1587 |              0.117296  |         0.436828 |               0.392025 |        186.149 |
|       7 |    2232 |             1577 |              0.111294  |         0.446237 |               0.357975 |        175.511 |
|       5 |    2232 |             1545 |              0.101599  |         0.396953 |               0.417563 |        156.971 |
|       4 |    2160 |             1166 |              0.0866006 |         0.289352 |               0.385648 |        100.976 |

## High vs low generation impact

- mid-generation impact proxy (10~80%): 1556.5453
- high-generation impact proxy (80%+): 296.6439
- low/near-zero impact proxy (<10%, official eligible count is usually zero): 0.0000

## H5: weak/low-correlation model slice hints

| slice_type                  | model_id             | target      | actual_bin   |   count |   candidate_better_abs_error_rate |   anchor_fail_candidate_success_count |   anchor_fail_candidate_success_rate |
|:----------------------------|:---------------------|:------------|:-------------|--------:|----------------------------------:|--------------------------------------:|-------------------------------------:|
| actual_bin                  | tabm_torch           | nan         | 30-60%       |    4394 |                          0.507055 |                                   819 |                            0.186391  |
| target+near_zero_flag       | mlp_regressor_medium | kpx_group_2 | nan          |    4982 |                          0.467884 |                                   774 |                            0.155359  |
| target                      | mlp_regressor_medium | kpx_group_2 | nan          |    8784 |                          0.4449   |                                   774 |                            0.0881148 |
| actual_bin                  | tabm_torch           | nan         | 10-30%       |    4497 |                          0.539915 |                                   748 |                            0.166333  |
| target+near_zero_flag       | tabm_torch           | kpx_group_2 | nan          |    4982 |                          0.420112 |                                   734 |                            0.14733   |
| target                      | tabm_torch           | kpx_group_2 | nan          |    8784 |                          0.378985 |                                   734 |                            0.083561  |
| target+high_generation_flag | mlp_regressor_medium | kpx_group_2 | nan          |    7831 |                          0.472609 |                                   732 |                            0.0934747 |
| target+high_generation_flag | tabm_torch           | kpx_group_2 | nan          |    7831 |                          0.401737 |                                   702 |                            0.0896437 |
| target+near_zero_flag       | tabm_torch           | kpx_group_3 | nan          |    4572 |                          0.432415 |                                   680 |                            0.148731  |
| target                      | tabm_torch           | kpx_group_3 | nan          |    8784 |                          0.407787 |                                   680 |                            0.0774135 |

## Required next-step discipline

This issue only routes the next focused audit. The next issue should test the routed cause directly and should not combine data cleansing, feature engineering, HPO, and postprocessing in one scope.

## Row-target table

- stored as: `C:\Users\USER\Desktop\jh0927\BARAM-2026\experiments\error_decomposition\results\row_target_errors.parquet`
- format: `parquet`
- rows: 26352
- comparison models loaded: ['catboost_quantile_055', 'catboost_quantile_0575_depth6', 'catboost_quantile_060_depth6', 'explainable_boosting', 'extra_trees_wide', 'hist_gbdt_quantile_055', 'lgbm_l1_baseline', 'mlp_regressor_medium', 'ngboost_normal_260_lr03', 'tabm_torch']

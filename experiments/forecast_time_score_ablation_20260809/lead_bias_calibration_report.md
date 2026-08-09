# Lead-Bias Calibration Transfer Result

This is the constrained postprocessing experiment requested after the broad feature ablation failed.

Protocol:

- Learn lead-hour median signed residual from a 2022 -> 2023 B1 diagnostic model.
- Apply that correction to cached 2024 `B1_aggregate_seed42` predictions.
- Group 3 is left uncorrected because there are no 2022 labels for a clean 2022 -> 2023 calibration.
- No 2024 labels are used to learn correction values.

## Summary

| experiment                       |    score |   one_minus_nmae |   avg_nmae |     ficr |   worst_month |   shrink |   delta_score_vs_baseline |   delta_ficr_vs_baseline |   delta_nmae_vs_baseline |
|:---------------------------------|---------:|-----------------:|-----------:|---------:|--------------:|---------:|--------------------------:|-------------------------:|-------------------------:|
| baseline_no_correction           | 0.611767 |         0.870029 |   0.129971 | 0.353506 |      0.591715 |   nan    |               0           |               0          |              0           |
| lead_bias_correction_shrink_0.10 | 0.61238  |         0.870211 |   0.129789 | 0.35455  |      0.591952 |     0.1  |               0.000612985 |               0.00104391 |             -0.000182063 |
| lead_bias_correction_shrink_0.20 | 0.613079 |         0.870377 |   0.129623 | 0.35578  |      0.59294  |     0.2  |               0.00131119  |               0.00227393 |             -0.000348453 |
| lead_bias_correction_shrink_0.35 | 0.61419  |         0.870599 |   0.129401 | 0.35778  |      0.59505  |     0.35 |               0.00242224  |               0.00427401 |             -0.000570464 |
| lead_bias_correction_shrink_0.50 | 0.615002 |         0.870783 |   0.129217 | 0.359221 |      0.595609 |     0.5  |               0.00323463  |               0.00571519 |             -0.00075407  |
| lead_bias_correction_shrink_0.75 | 0.616286 |         0.871006 |   0.128994 | 0.361567 |      0.595595 |     0.75 |               0.00451911  |               0.00806055 |             -0.000977669 |
| lead_bias_correction_shrink_1.00 | 0.616906 |         0.87112  |   0.12888  | 0.362691 |      0.595741 |     1    |               0.00513818  |               0.00918502 |             -0.00109135  |

## Best Group/Month Details

Best experiment: `lead_bias_correction_shrink_1.00`

| experiment                       | slice_type   | target      |       nmae |     ficr |   eligible_hours |   month |      score |   one_minus_nmae |   avg_nmae |
|:---------------------------------|:-------------|:------------|-----------:|---------:|-----------------:|--------:|-----------:|-----------------:|-----------:|
| lead_bias_correction_shrink_1.00 | group        | kpx_group_1 |   0.115952 | 0.406156 |             4989 |     nan | nan        |       nan        | nan        |
| lead_bias_correction_shrink_1.00 | group        | kpx_group_2 |   0.12475  | 0.418546 |             4976 |     nan | nan        |       nan        | nan        |
| lead_bias_correction_shrink_1.00 | group        | kpx_group_3 |   0.145938 | 0.26337  |             4566 |     nan | nan        |       nan        | nan        |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.365491 |             1492 |       1 |   0.609173 |         0.852855 |   0.147145 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.338918 |              772 |       2 |   0.598967 |         0.859017 |   0.140983 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.390065 |             1587 |       3 |   0.627364 |         0.864663 |   0.135337 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.339901 |             1166 |       4 |   0.611444 |         0.882986 |   0.117014 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.383871 |             1545 |       5 |   0.629661 |         0.875451 |   0.124549 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.335885 |              970 |       6 |   0.605588 |         0.875291 |   0.124709 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.35823  |             1577 |       7 |   0.610472 |         0.862713 |   0.137287 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.351905 |              689 |       8 |   0.609895 |         0.867884 |   0.132116 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.362037 |              837 |       9 |   0.62717  |         0.892303 |   0.107697 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.321241 |              895 |      10 |   0.598109 |         0.874977 |   0.125023 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.324879 |              963 |      11 |   0.595741 |         0.866602 |   0.133398 |
| lead_bias_correction_shrink_1.00 | month        | nan         | nan        | 0.401447 |             2038 |      12 |   0.641273 |         0.881098 |   0.118902 |

## Interpretation

- Lead-bias calibration is the first forecast-time experiment here with material validation gain.
- Best shrink `1.00` improves score by `0.005138`, FiCR by `0.009185`, and avg NMAE by `0.001091`.
- The result supports #34's claim that lead-hour is useful as a calibration axis.
- It is not yet a submission decision for the current best ensemble, because this calibration was tested on B1 LightGBM only.

Next proof required:

- Recompute the same lead-bias correction for the current ensemble/current-best prediction family.
- Validate with a second temporal split or bootstrap because correction strength increases monotonically up to shrink 1.00.
- Check public-LB transfer risk before replacing the current submission.

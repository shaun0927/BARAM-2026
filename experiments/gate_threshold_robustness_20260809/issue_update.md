# Gate threshold robustness check

이 업데이트는 previous best gate가 2024 validation threshold search에 과적합된 것인지 확인합니다.

검증 방식:

- base: `B1_plus_lead_interactions`
- fixed candidate: `specialist_pred >= 0.65 :: B1_plus_fundamental_all`
- calibration/holdout split을 월 단위로 나눔
- calibration에서 threshold/specialist를 고른 뒤 holdout month에서 평가
- fixed 0.65 rule도 같은 holdout에서 별도 평가

![Gate threshold robustness](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/gate_threshold_robustness_20260809/results/figures/01_gate_threshold_robustness.png)

## Full 2024 Gate Grid

| candidate                                          |    score |   delta_score_vs_base |     ficr |   avg_nmae | gate            |   threshold | specialist                |
|:---------------------------------------------------|---------:|----------------------:|---------:|-----------:|:----------------|------------:|:--------------------------|
| specialist_pred_ge_0.55::B1_plus_fundamental_all   | 0.614517 |           0.00268205  | 0.360154 |   0.13112  | specialist_pred |        0.55 | B1_plus_fundamental_all   |
| specialist_pred_ge_0.55::B1_plus_scada_wind_proxy  | 0.614269 |           0.00243408  | 0.360199 |   0.13166  | specialist_pred |        0.55 | B1_plus_scada_wind_proxy  |
| specialist_pred_ge_0.55::B1_plus_proxy_power_curve | 0.614253 |           0.00241742  | 0.359912 |   0.131407 | specialist_pred |        0.55 | B1_plus_proxy_power_curve |
| specialist_pred_ge_0.65::B1_plus_fundamental_all   | 0.614138 |           0.00230289  | 0.358909 |   0.130633 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| specialist_pred_ge_0.60::B1_plus_fundamental_all   | 0.613854 |           0.00201935  | 0.35872  |   0.131011 | specialist_pred |        0.6  | B1_plus_fundamental_all   |
| specialist_pred_ge_0.60::B1_plus_proxy_power_curve | 0.613592 |           0.00175733  | 0.358438 |   0.131254 | specialist_pred |        0.6  | B1_plus_proxy_power_curve |
| specialist_pred_ge_0.60::B1_plus_scada_wind_proxy  | 0.613322 |           0.00148644  | 0.358131 |   0.131488 | specialist_pred |        0.6  | B1_plus_scada_wind_proxy  |
| specialist_pred_ge_0.90::B1_plus_scada_wind_proxy  | 0.613258 |           0.00142262  | 0.356567 |   0.130051 | specialist_pred |        0.9  | B1_plus_scada_wind_proxy  |
| specialist_pred_ge_0.65::B1_plus_scada_wind_proxy  | 0.613092 |           0.00125665  | 0.357391 |   0.131208 | specialist_pred |        0.65 | B1_plus_scada_wind_proxy  |
| specialist_pred_ge_0.90::B1_plus_fundamental_all   | 0.612891 |           0.00105569  | 0.355779 |   0.129997 | specialist_pred |        0.9  | B1_plus_fundamental_all   |
| specialist_pred_ge_0.90::B1_plus_proxy_power_curve | 0.612883 |           0.00104762  | 0.355846 |   0.130081 | specialist_pred |        0.9  | B1_plus_proxy_power_curve |
| specialist_pred_ge_0.85::B1_plus_scada_wind_proxy  | 0.612865 |           0.00103009  | 0.356289 |   0.130558 | specialist_pred |        0.85 | B1_plus_scada_wind_proxy  |
| lead_pred_ge_0.80::B1_plus_scada_wind_proxy        | 0.612829 |           0.000994016 | 0.355868 |   0.13021  | lead_pred       |        0.8  | B1_plus_scada_wind_proxy  |
| specialist_pred_ge_0.65::B1_plus_proxy_power_curve | 0.612759 |           0.000923967 | 0.356494 |   0.130976 | specialist_pred |        0.65 | B1_plus_proxy_power_curve |
| specialist_pred_ge_0.70::B1_plus_fundamental_all   | 0.612732 |           0.000896825 | 0.35629  |   0.130826 | specialist_pred |        0.7  | B1_plus_fundamental_all   |

Fixed gate:

- candidate: `specialist_pred_ge_0.65::B1_plus_fundamental_all`
- full 2024 score: `0.614138`
- delta vs base: `0.002303`

## Split Robustness

| split              | policy                  | candidate                                          |   eval_score |   eval_delta_vs_base |   eval_ficr |   eval_avg_nmae | gate            |   threshold | specialist                |
|:-------------------|:------------------------|:---------------------------------------------------|-------------:|---------------------:|------------:|----------------:|:----------------|------------:|:--------------------------|
| cal_h1_eval_h2     | selected_on_calibration | specialist_pred_ge_0.85::B1_plus_scada_wind_proxy  |     0.614919 |         -0.00140716  |    0.357508 |        0.127671 | specialist_pred |        0.85 | B1_plus_scada_wind_proxy  |
| cal_h1_eval_h2     | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.618217 |          0.00189161  |    0.363892 |        0.127457 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_h2_eval_h1     | selected_on_calibration | specialist_pred_ge_0.55::B1_plus_fundamental_all   |     0.61085  |          0.00300374  |    0.355699 |        0.133999 | specialist_pred |        0.55 | B1_plus_fundamental_all   |
| cal_h2_eval_h1     | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.610224 |          0.00237801  |    0.354034 |        0.133586 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_odd_eval_even  | selected_on_calibration | specialist_pred_ge_0.55::B1_plus_scada_wind_proxy  |     0.614366 |         -0.00121453  |    0.35693  |        0.128199 | specialist_pred |        0.55 | B1_plus_scada_wind_proxy  |
| cal_odd_eval_even  | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.615193 |         -0.000387532 |    0.357487 |        0.127102 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_even_eval_odd  | selected_on_calibration | lead_pred_ge_0.80::B1_plus_scada_wind_proxy        |     0.609933 |         -0.00019956  |    0.353894 |        0.134027 | lead_pred       |        0.8  | B1_plus_scada_wind_proxy  |
| cal_even_eval_odd  | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.613795 |          0.00366225  |    0.36109  |        0.133499 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_not_q1_eval_q1 | selected_on_calibration | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.608487 |          0.00472935  |    0.358879 |        0.141904 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_not_q1_eval_q1 | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.608487 |          0.00472935  |    0.358879 |        0.141904 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_not_q2_eval_q2 | selected_on_calibration | specialist_pred_ge_0.55::B1_plus_fundamental_all   |     0.610179 |         -0.00225699  |    0.346008 |        0.125649 | specialist_pred |        0.55 | B1_plus_fundamental_all   |
| cal_not_q2_eval_q2 | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.611948 |         -0.000488193 |    0.348673 |        0.124777 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_not_q3_eval_q3 | selected_on_calibration | specialist_pred_ge_0.55::B1_plus_proxy_power_curve |     0.612716 |          0.00264473  |    0.354877 |        0.129445 | specialist_pred |        0.55 | B1_plus_proxy_power_curve |
| cal_not_q3_eval_q3 | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.614558 |          0.00448631  |    0.357674 |        0.128559 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| cal_not_q4_eval_q4 | selected_on_calibration | specialist_pred_ge_0.55::B1_plus_fundamental_all   |     0.620245 |         -0.00106208  |    0.367446 |        0.126956 | specialist_pred |        0.55 | B1_plus_fundamental_all   |
| cal_not_q4_eval_q4 | fixed_0p65_fundamental  | specialist_pred_ge_0.65::B1_plus_fundamental_all   |     0.620546 |         -0.000760782 |    0.367717 |        0.126624 | specialist_pred |        0.65 | B1_plus_fundamental_all   |

Summary:

- fixed 0.65 mean holdout delta: `0.001939`
- fixed 0.65 median holdout delta: `0.002135`
- fixed 0.65 positive splits: `5/8`
- calibration-selected mean holdout delta: `0.000530`
- calibration-selected positive splits: `3/8`

## Monthly Delta For Fixed Gate

|   month |   base_score |   candidate_score |   delta_score |   base_ficr |   candidate_ficr |   delta_ficr |   base_avg_nmae |   candidate_avg_nmae |   delta_avg_nmae |   eligible_hours |
|--------:|-------------:|------------------:|--------------:|------------:|-----------------:|-------------:|----------------:|---------------------:|-----------------:|-----------------:|
|       1 |     0.61045  |          0.62332  |   0.0128703   |    0.367073 |         0.391339 |   0.0242655  |        0.146173 |             0.144698 |     -0.00147513  |             1492 |
|       2 |     0.573404 |          0.578818 |   0.00541371  |    0.291645 |         0.298025 |   0.00637955 |        0.144836 |             0.140389 |     -0.00444787  |              772 |
|       3 |     0.614551 |          0.61003  |  -0.00452042  |    0.366899 |         0.360311 |  -0.00658836 |        0.137798 |             0.14025  |      0.00245248  |             1587 |
|       4 |     0.605956 |          0.605568 |  -0.000388492 |    0.33162  |         0.332769 |   0.00114823 |        0.119708 |             0.121633 |      0.00192521  |             1166 |
|       5 |     0.619142 |          0.620045 |   0.000902997 |    0.364015 |         0.367469 |   0.00345341 |        0.125732 |             0.127379 |      0.00164742  |             1545 |
|       6 |     0.605551 |          0.601849 |  -0.00370201  |    0.334933 |         0.327989 |  -0.00694451 |        0.123832 |             0.124291 |      0.00045951  |              970 |
|       7 |     0.608071 |          0.615378 |   0.00730651  |    0.35376  |         0.366223 |   0.0124633  |        0.137618 |             0.135468 |     -0.00214975  |             1577 |
|       8 |     0.614761 |          0.612917 |  -0.00184422  |    0.360456 |         0.359449 |  -0.00100675 |        0.130933 |             0.133615 |      0.00268169  |              689 |
|       9 |     0.61466  |          0.614478 |  -0.000182498 |    0.339475 |         0.339475 |   0          |        0.110154 |             0.110519 |      0.000364997 |              837 |
|      10 |     0.598993 |          0.594649 |  -0.0043439   |    0.327113 |         0.319513 |  -0.0076001  |        0.129128 |             0.130215 |      0.0010877   |              895 |
|      11 |     0.590168 |          0.590665 |   0.000497704 |    0.315242 |         0.31284  |  -0.00240207 |        0.134907 |             0.131509 |     -0.00339748  |              963 |
|      12 |     0.641984 |          0.641009 |  -0.000974768 |    0.402936 |         0.405044 |   0.00210838 |        0.118969 |             0.123026 |      0.00405791  |             2038 |

## Interpretation

- If the fixed gate remains positive across most held-out month splits, the gate is robust enough to promote.
- If full-2024 is positive but split holdouts are mixed, the gate is promising but threshold selection is unstable.
- If calibration-selected gates do not transfer, threshold search itself is overfitting and needs a stronger OOF protocol.

## Conclusion

- Full-2024 기준 최고 threshold는 `0.55`였지만, holdout robustness까지 보면 `0.65` fixed gate가 더 보수적이고 안정적입니다.
- `specialist_pred >= 0.65 :: B1_plus_fundamental_all`은 full 2024에서 `+0.002303`, split holdout 평균 `+0.001939`, positive split `5/8`입니다.
- 즉 이 gate는 **promote 후보**입니다. 다만 3개 split과 6개 월에서는 음수이므로, 아직 “안전한 최종 제출 규칙”이라고 단정하면 안 됩니다.
- calibration-selected threshold는 holdout positive split이 `3/8`에 그쳐 threshold search 자체가 과적합될 수 있음을 보여줍니다.
- 다음 단계는 threshold를 더 고르는 것이 아니라, OOF prediction으로 gate를 만들고 month-risk guardrail을 추가하는 것입니다.

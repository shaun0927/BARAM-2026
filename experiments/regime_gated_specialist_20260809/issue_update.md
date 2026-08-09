# Regime-gated specialist ablation

이 업데이트는 “SCADA-like/proxy feature를 전체 row에 뿌리면 score가 떨어진다”는 이전 결과 이후, 해당 specialist를 특정 regime에서만 켜는 방식이 가능한지 검증합니다.

Base model:

- `B1_plus_lead_interactions`: previous best validation score `0.611835`

Tested gates:

- oracle true very-high gate: 실제 `actual/capacity >= 80%`를 아는 비배포 상한선
- oracle true boundary gate: 실제 FiCR boundary를 아는 비배포 상한선
- deployable high gate: baseline 또는 specialist prediction ratio가 threshold 이상일 때만 specialist 사용
- deployable boundary gate: prediction ratio가 6-14%, 8-12%, 8-16% 구간일 때만 specialist 사용

![Regime gated specialist summary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/regime_gated_specialist_20260809/results/figures/01_regime_gated_specialist_summary.png)

## Main Result

| experiment                                                     |    score |   one_minus_nmae |   avg_nmae |     ficr |   worst_month |   high_generation_score |   delta_score_vs_lead_base |   delta_ficr_vs_lead_base |   delta_nmae_vs_lead_base |
|:---------------------------------------------------------------|---------:|-----------------:|-----------:|---------:|--------------:|------------------------:|---------------------------:|--------------------------:|--------------------------:|
| oracle_boundary_or_very_high::B1_plus_fundamental_all          | 0.619408 |         0.871899 |   0.128101 | 0.366918 |      0.571214 |                0.611064 |                0.00757338  |               0.0131963   |              -0.00195043  |
| oracle_boundary_or_very_high::B1_plus_scada_wind_proxy         | 0.619286 |         0.871446 |   0.128554 | 0.367127 |      0.576336 |                0.610673 |                0.00745105  |               0.013405    |              -0.00149712  |
| oracle_very_high::B1_plus_fundamental_all                      | 0.619264 |         0.871841 |   0.128159 | 0.366687 |      0.57121  |                0.611064 |                0.00742894  |               0.0129649   |              -0.00189302  |
| oracle_very_high::B1_plus_scada_wind_proxy                     | 0.61918  |         0.871372 |   0.128628 | 0.366988 |      0.576499 |                0.610673 |                0.00734453  |               0.0132658   |              -0.00142325  |
| oracle_boundary_or_very_high::B1_plus_proxy_power_curve        | 0.618987 |         0.871555 |   0.128445 | 0.366419 |      0.569559 |                0.610329 |                0.00715173  |               0.012697    |              -0.00160641  |
| oracle_very_high::B1_plus_proxy_power_curve                    | 0.618847 |         0.871487 |   0.128513 | 0.366207 |      0.56948  |                0.610329 |                0.00701221  |               0.0124857   |              -0.00153869  |
| deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all   | 0.614138 |         0.869367 |   0.130633 | 0.358909 |      0.578818 |                0.605874 |                0.00230289  |               0.00518729  |               0.00058152  |
| deploy_high_specialist_pred_ge_0.60::B1_plus_fundamental_all   | 0.613854 |         0.868989 |   0.131011 | 0.35872  |      0.581209 |                0.607276 |                0.00201935  |               0.00499821  |               0.000959512 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_proxy_power_curve | 0.613592 |         0.868746 |   0.131254 | 0.358438 |      0.580153 |                0.606931 |                0.00175733  |               0.00471672  |               0.00120206  |
| deploy_high_specialist_pred_ge_0.60::B1_plus_scada_wind_proxy  | 0.613322 |         0.868512 |   0.131488 | 0.358131 |      0.582792 |                0.606477 |                0.00148644  |               0.00440954  |               0.00143666  |
| deploy_high_specialist_pred_ge_0.90::B1_plus_scada_wind_proxy  | 0.613258 |         0.869949 |   0.130051 | 0.356567 |      0.578371 |                0.602052 |                0.00142262  |               0.00284497  |              -2.72139e-07 |
| deploy_high_specialist_pred_ge_0.65::B1_plus_scada_wind_proxy  | 0.613092 |         0.868792 |   0.131208 | 0.357391 |      0.580453 |                0.604222 |                0.00125665  |               0.00366954  |               0.00115624  |
| deploy_high_specialist_pred_ge_0.90::B1_plus_fundamental_all   | 0.612891 |         0.870003 |   0.129997 | 0.355779 |      0.572847 |                0.601573 |                0.00105569  |               0.00205706  |              -5.43189e-05 |
| deploy_high_specialist_pred_ge_0.90::B1_plus_proxy_power_curve | 0.612883 |         0.869919 |   0.130081 | 0.355846 |      0.573313 |                0.601525 |                0.00104762  |               0.00212436  |               2.91319e-05 |
| deploy_high_specialist_pred_ge_0.85::B1_plus_scada_wind_proxy  | 0.612865 |         0.869442 |   0.130558 | 0.356289 |      0.58492  |                0.601287 |                0.00103009  |               0.00256682  |               0.000506634 |
| deploy_high_lead_pred_ge_0.80::B1_plus_scada_wind_proxy        | 0.612829 |         0.86979  |   0.13021  | 0.355868 |      0.583483 |                0.601319 |                0.000994016 |               0.00214664  |               0.000158612 |
| oracle_boundary::B1_plus_fundamental_all                       | 0.61198  |         0.870006 |   0.129994 | 0.353953 |      0.573409 |                0.600124 |                0.000144442 |               0.000231472 |              -5.74118e-05 |
| oracle_boundary::B1_plus_proxy_power_curve                     | 0.611975 |         0.870016 |   0.129984 | 0.353933 |      0.573484 |                0.600124 |                0.000139514 |               0.000211309 |              -6.77184e-05 |

해석:

- 전체 최고 후보는 `oracle_boundary_or_very_high::B1_plus_fundamental_all`이고 score delta vs lead-base는 `0.007573`입니다.
- 최고 deployable gate는 `deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all`이고 score는 `0.614138`입니다.
- oracle gate와 deployable gate 차이는 “신호는 있는데 실제 gate가 충분히 정확하지 않은지”를 판단하는 핵심입니다.

## Oracle Upper Bounds

| experiment                                              |    score |   delta_score_vs_lead_base |   delta_ficr_vs_lead_base |   delta_nmae_vs_lead_base |   high_generation_score |
|:--------------------------------------------------------|---------:|---------------------------:|--------------------------:|--------------------------:|------------------------:|
| oracle_boundary_or_very_high::B1_plus_fundamental_all   | 0.619408 |                0.00757338  |               0.0131963   |              -0.00195043  |                0.611064 |
| oracle_boundary_or_very_high::B1_plus_scada_wind_proxy  | 0.619286 |                0.00745105  |               0.013405    |              -0.00149712  |                0.610673 |
| oracle_very_high::B1_plus_fundamental_all               | 0.619264 |                0.00742894  |               0.0129649   |              -0.00189302  |                0.611064 |
| oracle_very_high::B1_plus_scada_wind_proxy              | 0.61918  |                0.00734453  |               0.0132658   |              -0.00142325  |                0.610673 |
| oracle_boundary_or_very_high::B1_plus_proxy_power_curve | 0.618987 |                0.00715173  |               0.012697    |              -0.00160641  |                0.610329 |
| oracle_very_high::B1_plus_proxy_power_curve             | 0.618847 |                0.00701221  |               0.0124857   |              -0.00153869  |                0.610329 |
| oracle_boundary::B1_plus_fundamental_all                | 0.61198  |                0.000144442 |               0.000231472 |              -5.74118e-05 |                0.600124 |
| oracle_boundary::B1_plus_proxy_power_curve              | 0.611975 |                0.000139514 |               0.000211309 |              -6.77184e-05 |                0.600124 |

해석:

- oracle very-high gate가 좋아지면 specialist 자체는 올바른 구간에서 가치가 있다는 뜻입니다.
- deployable gate가 못 따라오면 문제는 specialist가 아니라 gate quality입니다.

## Top Deployable Gate Grid

| experiment                                                     |    score |   delta_score_vs_lead_base |     ficr |   avg_nmae |   worst_month |   high_generation_score | gate            |   threshold | specialist                |
|:---------------------------------------------------------------|---------:|---------------------------:|---------:|-----------:|--------------:|------------------------:|:----------------|------------:|:--------------------------|
| deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all   | 0.614138 |                0.00230289  | 0.358909 |   0.130633 |      0.578818 |                0.605874 | specialist_pred |        0.65 | B1_plus_fundamental_all   |
| deploy_high_specialist_pred_ge_0.60::B1_plus_fundamental_all   | 0.613854 |                0.00201935  | 0.35872  |   0.131011 |      0.581209 |                0.607276 | specialist_pred |        0.6  | B1_plus_fundamental_all   |
| deploy_high_specialist_pred_ge_0.60::B1_plus_proxy_power_curve | 0.613592 |                0.00175733  | 0.358438 |   0.131254 |      0.580153 |                0.606931 | specialist_pred |        0.6  | B1_plus_proxy_power_curve |
| deploy_high_specialist_pred_ge_0.60::B1_plus_scada_wind_proxy  | 0.613322 |                0.00148644  | 0.358131 |   0.131488 |      0.582792 |                0.606477 | specialist_pred |        0.6  | B1_plus_scada_wind_proxy  |
| deploy_high_specialist_pred_ge_0.90::B1_plus_scada_wind_proxy  | 0.613258 |                0.00142262  | 0.356567 |   0.130051 |      0.578371 |                0.602052 | specialist_pred |        0.9  | B1_plus_scada_wind_proxy  |
| deploy_high_specialist_pred_ge_0.65::B1_plus_scada_wind_proxy  | 0.613092 |                0.00125665  | 0.357391 |   0.131208 |      0.580453 |                0.604222 | specialist_pred |        0.65 | B1_plus_scada_wind_proxy  |
| deploy_high_specialist_pred_ge_0.90::B1_plus_fundamental_all   | 0.612891 |                0.00105569  | 0.355779 |   0.129997 |      0.572847 |                0.601573 | specialist_pred |        0.9  | B1_plus_fundamental_all   |
| deploy_high_specialist_pred_ge_0.90::B1_plus_proxy_power_curve | 0.612883 |                0.00104762  | 0.355846 |   0.130081 |      0.573313 |                0.601525 | specialist_pred |        0.9  | B1_plus_proxy_power_curve |
| deploy_high_specialist_pred_ge_0.85::B1_plus_scada_wind_proxy  | 0.612865 |                0.00103009  | 0.356289 |   0.130558 |      0.58492  |                0.601287 | specialist_pred |        0.85 | B1_plus_scada_wind_proxy  |
| deploy_high_lead_pred_ge_0.80::B1_plus_scada_wind_proxy        | 0.612829 |                0.000994016 | 0.355868 |   0.13021  |      0.583483 |                0.601319 | lead_pred       |        0.8  | B1_plus_scada_wind_proxy  |
| deploy_high_specialist_pred_ge_0.65::B1_plus_proxy_power_curve | 0.612759 |                0.000923967 | 0.356494 |   0.130976 |      0.575462 |                0.603723 | specialist_pred |        0.65 | B1_plus_proxy_power_curve |
| deploy_high_specialist_pred_ge_0.70::B1_plus_fundamental_all   | 0.612732 |                0.000896825 | 0.35629  |   0.130826 |      0.574355 |                0.602728 | specialist_pred |        0.7  | B1_plus_fundamental_all   |
| deploy_high_specialist_pred_ge_0.85::B1_plus_fundamental_all   | 0.612731 |                0.00089567  | 0.355836 |   0.130374 |      0.578134 |                0.601244 | specialist_pred |        0.85 | B1_plus_fundamental_all   |
| deploy_high_specialist_pred_ge_0.75::B1_plus_fundamental_all   | 0.612628 |                0.000793218 | 0.355888 |   0.130631 |      0.578363 |                0.601755 | specialist_pred |        0.75 | B1_plus_fundamental_all   |
| deploy_high_specialist_pred_ge_0.85::B1_plus_proxy_power_curve | 0.612576 |                0.000740794 | 0.355702 |   0.13055  |      0.578408 |                0.601038 | specialist_pred |        0.85 | B1_plus_proxy_power_curve |

## Regime Delta For Top Candidates

| experiment                                                     | target      | label_regime         |   rows |   delta_abs_error |   impact_delta |   net_pass8 |
|:---------------------------------------------------------------|:------------|:---------------------|-------:|------------------:|---------------:|------------:|
| deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all   | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0330833  |      -23.5553  |          34 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_fundamental_all   | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0317976  |      -22.6399  |          32 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_proxy_power_curve | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0294634  |      -20.9779  |          27 |
| deploy_high_specialist_pred_ge_0.65::B1_plus_scada_wind_proxy  | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0245646  |      -17.49    |          18 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_scada_wind_proxy  | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0235578  |      -16.7732  |          17 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_scada_wind_proxy  | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0143555  |      -13.6807  |          56 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_fundamental_all   | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0139986  |      -13.3407  |          38 |
| deploy_high_specialist_pred_ge_0.65::B1_plus_scada_wind_proxy  | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0136626  |      -13.0204  |          56 |
| oracle_boundary_or_very_high::B1_plus_scada_wind_proxy         | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0135911  |      -12.9523  |          55 |
| oracle_very_high::B1_plus_scada_wind_proxy                     | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0135911  |      -12.9523  |          55 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_proxy_power_curve | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0135762  |      -12.9381  |          36 |
| oracle_very_high::B1_plus_fundamental_all                      | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0134716  |      -12.8384  |          37 |
| oracle_boundary_or_very_high::B1_plus_fundamental_all          | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0134716  |      -12.8384  |          37 |
| deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all   | kpx_group_2 | very_high_80pct_plus |    953 |       -0.013116   |      -12.4995  |          38 |
| oracle_boundary_or_very_high::B1_plus_proxy_power_curve        | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0130809  |      -12.4661  |          35 |
| oracle_very_high::B1_plus_proxy_power_curve                    | kpx_group_2 | very_high_80pct_plus |    953 |       -0.0130809  |      -12.4661  |          35 |
| oracle_boundary_or_very_high::B1_plus_fundamental_all          | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0128708  |       -9.16403 |          29 |
| oracle_very_high::B1_plus_fundamental_all                      | kpx_group_3 | very_high_80pct_plus |    712 |       -0.0128708  |       -9.16403 |          29 |
| deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all   | kpx_group_1 | very_high_80pct_plus |    840 |       -0.0104213  |       -8.75388 |          38 |
| deploy_high_specialist_pred_ge_0.65::B1_plus_scada_wind_proxy  | kpx_group_1 | very_high_80pct_plus |    840 |       -0.0101691  |       -8.54205 |          43 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_fundamental_all   | kpx_group_1 | very_high_80pct_plus |    840 |       -0.00881377 |       -7.40357 |          38 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_proxy_power_curve | kpx_group_1 | very_high_80pct_plus |    840 |       -0.0084924  |       -7.13362 |          40 |
| deploy_high_specialist_pred_ge_0.60::B1_plus_scada_wind_proxy  | kpx_group_1 | very_high_80pct_plus |    840 |       -0.00830832 |       -6.97899 |          43 |
| oracle_very_high::B1_plus_fundamental_all                      | kpx_group_1 | very_high_80pct_plus |    840 |       -0.00648556 |       -5.44787 |          38 |

## Conclusion

이 실험의 핵심 질문은 “#32 신호를 전체 row가 아니라 필요한 regime에서만 쓰면 score가 오르는가?”였습니다.

결론:

- **오릅니다.** 최고 deployable gate는 `deploy_high_specialist_pred_ge_0.65::B1_plus_fundamental_all`이고, lead-base 대비 `+0.002303` score를 냈습니다.
- oracle upper bound는 `+0.007573`까지 열려 있습니다. 즉 specialist 자체의 잠재력은 더 큽니다.
- 개선은 주로 `very_high_80pct_plus` regime에서 나옵니다. 이 구간에서 deployable gate가 group별로 큰 `delta_abs_error` 개선과 positive net pass8을 만듭니다.
- 반대로 mid/high regime에서는 여전히 악화가 생깁니다. 그래서 specialist는 전체 row에 쓰면 안 되고, high-generation gate로 제한해야 합니다.

Important caveat:

- 이번 deployable threshold grid는 2024 validation에서 고른 값입니다. 제출 후보로 쓰려면 threshold를 out-of-fold 또는 다른 holdout에서 다시 고정해야 합니다.
- 그래도 이번 결과는 이전 실패의 원인을 명확히 뒤집습니다. #32 인사이트는 실패한 것이 아니라, **regime gate 없이 global feature로 넣은 방식이 실패한 것**입니다.

Next candidate:

- `B1_plus_lead_interactions`를 base로 두고,
- `B1_plus_fundamental_all` specialist를
- specialist prediction ratio `>= 0.65`인 row에서만 적용하는 gated blend.

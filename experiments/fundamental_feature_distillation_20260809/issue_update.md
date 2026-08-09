# Fundamental feature/model ablation: SCADA-like NWP proxy, soft upstream grids, group power curve

이 업데이트는 postprocessing/calibration이 아니라 더 근본적인 feature/model 가설을 검증합니다.

검증한 후보:

- `B1_aggregate`: 기존 aggregate weather baseline
- `B1_plus_scada_wind_proxy`: NWP/time feature로 SCADA group wind를 distill한 proxy
- `B1_plus_soft_upstream`: 터빈 centroid와 wind vector를 이용한 soft upstream LDAPS aggregation
- `B1_plus_proxy_power_curve`: SCADA-like wind proxy 기반 group-specific power curve feature
- `B1_plus_fundamental_all`: 위 세 feature family를 모두 추가

![Fundamental ablation summary](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/fundamental_feature_distillation_20260809/results/figures/01_fundamental_feature_ablation_summary.png)

## Main Result

| experiment                |    score |   one_minus_nmae |   avg_nmae |     ficr |   worst_month |   high_generation_score |   eligible_hours |   feature_count_group1 |   feature_count_group2 |   feature_count_group3 |   delta_score_vs_baseline |   delta_ficr_vs_baseline |   delta_nmae_vs_baseline |
|:--------------------------|---------:|-----------------:|-----------:|---------:|--------------:|------------------------:|-----------------:|-----------------------:|-----------------------:|-----------------------:|--------------------------:|-------------------------:|-------------------------:|
| B1_aggregate              | 0.611767 |         0.870029 |   0.129971 | 0.353506 |      0.591715 |                0.600197 |            14531 |                    286 |                    286 |                    286 |                0          |              0           |              0           |
| B1_plus_soft_upstream     | 0.610322 |         0.869212 |   0.130788 | 0.351431 |      0.578857 |                0.599046 |            14531 |                    293 |                    293 |                    293 |               -0.00144551 |             -0.0020746   |              0.000816422 |
| B1_plus_fundamental_all   | 0.60973  |         0.865595 |   0.134405 | 0.353866 |      0.58425  |                0.603479 |            14531 |                    298 |                    298 |                    298 |               -0.00203698 |              0.00035977  |              0.00443373  |
| B1_plus_proxy_power_curve | 0.609178 |         0.865066 |   0.134934 | 0.35329  |      0.588194 |                0.602026 |            14531 |                    291 |                    291 |                    291 |               -0.00258924 |             -0.000215893 |              0.00496259  |
| B1_plus_scada_wind_proxy  | 0.609062 |         0.865003 |   0.134997 | 0.35312  |      0.587028 |                0.602345 |            14531 |                    289 |                    289 |                    289 |               -0.00270576 |             -0.000386231 |              0.00502528  |

해석:

- 최고 실험은 `B1_aggregate`입니다.
- baseline 대비 score delta는 `0.000000`입니다.
- SCADA-like wind proxy, soft upstream grid, group power-curve feature 모두 현재 fixed-LGBM/B1 protocol에서는 baseline을 넘지 못했습니다.
- 다만 `B1_plus_fundamental_all`은 FiCR `+0.000360`, high-generation score `+0.003282`를 만들었습니다. 문제는 avg NMAE가 `+0.004434` 악화되어 총점이 떨어졌다는 점입니다.
- 이 실험은 “근본 feature를 만들면 바로 오른다”가 아니라, **근본 신호는 존재하지만 현재 downstream 연결 방식이 score objective와 맞지 않는다**는 것을 보여줍니다.

## SCADA-like wind proxy quality

| target      |   valid_rows |   mae_ws |    r2_ws |   teacher_mean_ws_valid |
|:------------|-------------:|---------:|---------:|------------------------:|
| kpx_group_1 |         8784 |  1.06382 | 0.826496 |                 6.75358 |
| kpx_group_2 |         8784 |  1.1575  | 0.838325 |                 7.10443 |
| kpx_group_3 |         8778 |  1.09633 | 0.833677 |                 5.78396 |

해석:

- proxy R2가 높으면 NWP/time만으로 SCADA wind를 어느 정도 재현할 수 있다는 뜻입니다.
- 하지만 proxy가 좋아도 발전량 score가 오르지 않으면, proxy를 power model에 넣는 방식 또는 downstream model capacity가 문제입니다.
- 실제로 세 group 모두 SCADA wind proxy R2가 약 `0.83`입니다. 즉 NWP→SCADA wind distillation 자체는 됩니다. 실패한 부분은 proxy 생성이 아니라, 그 proxy를 발전량 예측과 FiCR/NMAE objective에 연결하는 방식입니다.

## Regime delta vs baseline

| experiment                | target      | label_regime          |   rows |   delta_abs_error |   impact_delta |   fail_to_pass8 |   pass_to_fail8 |   net_pass8 |
|:--------------------------|:------------|:----------------------|-------:|------------------:|---------------:|----------------:|----------------:|------------:|
| B1_plus_fundamental_all   | kpx_group_3 | very_high_80pct_plus  |    712 |      -0.0163266   |    -11.6245    |              51 |              10 |          41 |
| B1_plus_scada_wind_proxy  | kpx_group_2 | very_high_80pct_plus  |    953 |      -0.0114219   |    -10.8851    |             103 |              54 |          49 |
| B1_plus_fundamental_all   | kpx_group_2 | very_high_80pct_plus  |    953 |      -0.0113024   |    -10.7712    |             100 |              69 |          31 |
| B1_plus_proxy_power_curve | kpx_group_2 | very_high_80pct_plus  |    953 |      -0.0109117   |    -10.3989    |              98 |              69 |          29 |
| B1_plus_proxy_power_curve | kpx_group_3 | very_high_80pct_plus  |    712 |      -0.0105616   |     -7.51989   |              47 |              11 |          36 |
| B1_plus_scada_wind_proxy  | kpx_group_3 | very_high_80pct_plus  |    712 |      -0.00854536  |     -6.0843    |              38 |              12 |          26 |
| B1_plus_fundamental_all   | kpx_group_1 | very_high_80pct_plus  |    840 |      -0.0070636   |     -5.93342   |             101 |              56 |          45 |
| B1_plus_proxy_power_curve | kpx_group_1 | very_high_80pct_plus  |    840 |      -0.00653377  |     -5.48837   |             103 |              56 |          47 |
| B1_plus_scada_wind_proxy  | kpx_group_1 | very_high_80pct_plus  |    840 |      -0.00576392  |     -4.8417    |             103 |              53 |          50 |
| B1_plus_soft_upstream     | kpx_group_1 | very_high_80pct_plus  |    840 |      -0.00225777  |     -1.89652   |              33 |              33 |           0 |
| B1_plus_scada_wind_proxy  | kpx_group_2 | ficr_boundary_8_12pct |    224 |      -0.00345987  |     -0.775012  |              24 |              19 |           5 |
| B1_plus_fundamental_all   | kpx_group_2 | ficr_boundary_8_12pct |    224 |      -0.00323668  |     -0.725016  |              24 |              17 |           7 |
| B1_plus_scada_wind_proxy  | kpx_group_3 | ficr_boundary_8_12pct |    212 |      -0.00333427  |     -0.706866  |              16 |              12 |           4 |
| B1_plus_proxy_power_curve | kpx_group_3 | ficr_boundary_8_12pct |    212 |      -0.00267438  |     -0.566969  |              17 |              11 |           6 |
| B1_plus_proxy_power_curve | kpx_group_2 | ficr_boundary_8_12pct |    224 |      -0.00246618  |     -0.552423  |              24 |              20 |           4 |
| B1_plus_soft_upstream     | kpx_group_3 | high_50_80pct         |   1353 |      -0.00037176  |     -0.502991  |              62 |              73 |         -11 |
| B1_plus_fundamental_all   | kpx_group_3 | ficr_boundary_8_12pct |    212 |      -0.00184048  |     -0.390181  |              17 |              11 |           6 |
| B1_plus_soft_upstream     | kpx_group_2 | ficr_boundary_8_12pct |    224 |      -0.000972391 |     -0.217815  |              16 |              15 |           1 |
| B1_plus_soft_upstream     | kpx_group_1 | ficr_boundary_8_12pct |    219 |      -0.000697836 |     -0.152826  |               7 |              12 |          -5 |
| B1_plus_soft_upstream     | kpx_group_3 | ficr_boundary_8_12pct |    212 |      -0.000283746 |     -0.0601542 |              11 |               8 |           3 |

해석:

- 특정 label regime에서는 개선이 생길 수 있지만, total score는 다른 regime 악화와 FiCR 변화에 의해 쉽게 상쇄됩니다.
- 따라서 이 결과는 후속으로 group/regime-specific model 또는 soft calibration이 필요한지 판단하는 근거입니다.
- 특히 very-high generation regime에서는 proxy/power-curve 계열이 크게 개선됩니다. 반대로 mid/high regime에서는 악화가 커서 전체 score를 잃습니다.
- 따라서 이 feature들은 global model feature로 넣기보다 high-generation specialist, regime-gated model, 또는 group-specific expert로 써야 합니다.

## Conclusion

이번 실험의 판단 기준:

- SCADA-like wind proxy는 성공했습니다. R2가 약 0.83으로 높습니다.
- 그러나 proxy를 global LGBM feature로 넣는 방식은 실패했습니다.
- very-high generation에서는 강한 개선이 있으므로 feature 자체를 폐기하면 안 됩니다.
- 다음 단계는 global feature addition이 아니라 `very_high`/high-generation expert, group-specific power curve, regime gate를 붙인 mixture-of-experts 구조입니다.

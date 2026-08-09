## Spatial Topology Reinterpretation — geometry is not the same as predictive grid choice

This update clarifies section 4 of the EDA atlas. The original spatial topology figure is useful, but it should be read as **geometry only**. It does not prove that nearest-grid features are sufficient.

![Spatial topology reinterpretation](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/30_spatial_topology_reinterpretation.png)

### What the original topology plot can and cannot prove

It can prove:

- GFS is a coarse 9-grid envelope around the wind farms.
- LDAPS is a denser 16-grid local field.
- The three KPX groups sit across multiple nearby LDAPS cells, not inside one obvious universal cell.

It cannot prove:

- that the nearest grid is the most predictive grid,
- that one static grid per group is sufficient,
- that all-grid feature dumping captures the spatial structure,
- or that spatial topology is useless just because broad spatial feature additions failed.

### Nearest grid vs static correlation grid

| target      |   group |   modal_nearest_ldaps_grid |   modal_nearest_turbines |   static_top_corr_grid |   static_top_corr |   modal_nearest_grid_corr |   corr_gap_static_minus_nearest |   direction_sector_count |   unique_direction_top_grids |   sectors_where_direction_top_equals_nearest |   sectors_where_direction_top_equals_static_top |
|:------------|--------:|---------------------------:|-------------------------:|-----------------------:|------------------:|--------------------------:|--------------------------------:|-------------------------:|-----------------------------:|---------------------------------------------:|------------------------------------------------:|
| kpx_group_1 |       1 |                          5 |                        3 |                     13 |          0.8075   |                  0.659954 |                      0.147547   |                        8 |                            6 |                                            1 |                                               1 |
| kpx_group_2 |       2 |                          6 |                        3 |                     13 |          0.812794 |                  0.786104 |                      0.0266893  |                        8 |                            4 |                                            0 |                                               2 |
| kpx_group_3 |       3 |                         12 |                        4 |                     13 |          0.826312 |                  0.823962 |                      0.00235026 |                        8 |                            6 |                                            1 |                                               0 |

Interpretation:

- The modal nearest LDAPS grid and the static highest-correlation grid are not the same object.
- The static correlation advantage over the modal nearest grid is small in absolute correlation terms, but the identity mismatch matters for feature design.
- A nearest-grid rule is therefore a geometry prior, not an empirical proof.

### Direction-conditioned result

| target      | dir_sector   |   grid_id |   rows |   spearman_corr |
|:------------|:-------------|----------:|-------:|----------------:|
| kpx_group_1 | 0-45         |        13 |  10286 |        0.755603 |
| kpx_group_1 | 135-180      |        10 |   2062 |        0.456936 |
| kpx_group_1 | 180-225      |         5 |   3388 |        0.499287 |
| kpx_group_1 | 225-270      |        10 |    728 |        0.525223 |
| kpx_group_1 | 270-315      |        11 |    369 |        0.554094 |
| kpx_group_1 | 315-360      |         1 |   1729 |        0.785883 |
| kpx_group_1 | 45-90        |        14 |   1536 |        0.745271 |
| kpx_group_1 | 90-135       |        14 |   1227 |        0.523358 |
| kpx_group_2 | 0-45         |        13 |  10284 |        0.76899  |
| kpx_group_2 | 135-180      |         5 |   1727 |        0.430263 |
| kpx_group_2 | 180-225      |        11 |   3230 |        0.492294 |
| kpx_group_2 | 225-270      |        14 |    555 |        0.549827 |
| kpx_group_2 | 270-315      |        11 |    369 |        0.540731 |
| kpx_group_2 | 315-360      |        13 |   6645 |        0.781015 |
| kpx_group_2 | 45-90        |        14 |   1536 |        0.763374 |
| kpx_group_2 | 90-135       |        14 |   1227 |        0.531143 |
| kpx_group_3 | 0-45         |        16 |   5995 |        0.761707 |
| kpx_group_3 | 135-180      |        12 |    737 |        0.439389 |
| kpx_group_3 | 180-225      |        11 |   2348 |        0.57355  |
| kpx_group_3 | 225-270      |        10 |    522 |        0.582297 |
| kpx_group_3 | 270-315      |        11 |    231 |        0.538094 |
| kpx_group_3 | 315-360      |         1 |   1077 |        0.780754 |
| kpx_group_3 | 45-90        |        14 |   1114 |        0.780288 |
| kpx_group_3 | 90-135       |        14 |    790 |        0.496424 |

Interpretation:

- The best LDAPS grid changes by wind-direction sector.
- Group 1 uses 6 distinct top grids across 8 sectors, group 2 uses 4, and group 3 uses 6.
- This is the strongest spatial EDA conclusion: the usable spatial signal is **direction-conditioned / upstream-like**, not a single nearest-grid lookup.

### Modeling consequence

The correct next spatial feature is not `nearest_grid_value` and not a blind dump of every grid column. It should be one of:

1. direction-conditioned top-grid features,
2. soft upstream weighting from turbine centroids and wind vector,
3. group-specific spatial pooling that keeps VESTAS group 1/2 and UNISON group 3 separate,
4. residual tests showing whether the spatial feature reduces current-anchor error, not only label correlation.

This update narrows the EDA conclusion: section 4 should be read as **spatial feature design guidance**, not as proof that topology alone explains score movement.

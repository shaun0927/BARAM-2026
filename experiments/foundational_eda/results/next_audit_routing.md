# Next Audit Routing

- recommended next issue: **data quality / cleansing audit**
- evidence: Range violations exist in labels and must be explained before modeling changes.
- priority: high
- what not to do yet: do not resume CatBoost HPO, do not apply cleansing, and do not add feature families until the next focused audit explains whether the issue is drift, feature coverage, or data quality.

## Why not model HPO now

- The project now has model hints, but the foundational inventory shows the next uncertainty is problem structure: target regime, year/time distribution, metric eligibility, and feature coverage.
- HPO should follow after data/target/metric and feature inventory identify stable intervention targets.

## Recommended next issue scope

If no hard data-quality blocker is present, open a focused `Temporal drift and feature inventory audit` issue:

- compare train years 2022/2023 vs validation 2024 target distributions by group/month/hour/bin
- inspect B1 feature inventory and train-valid/test feature coverage
- quantify feature drift and missingness drift
- connect drift/feature coverage to the generation regimes identified here
- do not train new models until this audit routes an intervention

## Supporting tables

- `dataset_inventory.csv`
- `timestamp_coverage.csv`
- `label_coverage.csv`
- `target_distribution_summary.csv`
- `target_ratio_bin_summary.csv`
- `target_by_year_month_hour.csv`
- `target_group_correlation.csv`
- `metric_eligible_distribution.csv`
- `metric_generation_bin_summary.csv`
- `ficr_boundary_distribution.csv`
- `data_quality_inventory.csv`
- `label_range_violations.csv`
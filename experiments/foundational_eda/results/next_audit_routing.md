# Next Audit Routing

- recommended next issue: **data quality / cleansing audit**
- evidence: Capacity-exceed label values exist and must be explained before any cleansing rule, model HPO, or feature expansion.
- priority: high
- what not to do yet: do not resume CatBoost HPO, do not apply cleansing, and do not add feature families until the next focused audit explains whether the issue is drift, feature coverage, or data quality.

## Why not model HPO now

- The project now has model hints, but the foundational inventory shows the next uncertainty is problem structure: target regime, year/time distribution, metric eligibility, and feature coverage.
- HPO should follow after data/target/metric and feature inventory identify stable intervention targets.

## Recommended next issue scope

Open a focused `Data quality / cleansing audit` issue first, because a hard label-range blocker is present:

- enumerate all capacity-exceed rows and their timestamp/target/month/hour/regime
- verify whether exceedance is true production, capacity metadata mismatch, rounding/noise, aggregation issue, or label anomaly
- test candidate cleansing rules diagnostically only: clip-to-capacity, remove rows, keep-as-is, target-specific rule
- replay historical validation without adding features or HPO to measure whether each rule changes metric-facing behavior
- decide one explicit policy: no cleansing, clipping, exclusion, or metadata correction
- only after this is resolved, move to temporal drift / feature sufficiency audit

## Supporting tables

- `dataset_inventory.csv`
- `timestamp_coverage.csv`
- `weather_availability_inventory.csv`
- `weather_schema_inventory.csv`
- `weather_schema_consistency.csv`
- `sample_horizon_alignment.csv`
- `label_coverage.csv`
- `target_temporal_extreme_summary.csv`
- `target_distribution_summary.csv`
- `target_ratio_bin_summary.csv`
- `target_by_year_month_hour.csv`
- `target_group_correlation.csv`
- `metric_eligible_distribution.csv`
- `metric_generation_bin_summary.csv`
- `ficr_boundary_distribution.csv`
- `data_quality_inventory.csv`
- `label_range_violations.csv`
- `capacity_exceed_context.csv`
- `feature_inventory.csv`
- `feature_drift_summary.csv`
- `feature_composition_summary.csv`
- `scada_inventory.csv`
- `meta_inventory.csv`
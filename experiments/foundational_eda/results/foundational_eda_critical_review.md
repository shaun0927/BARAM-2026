# Foundational EDA Critical Review

## Completed foundational checks

- Dataset and timestamp inventory
- Train/test/sample horizon alignment
- Raw weather forecast availability and lead-time inventory
- Raw weather schema consistency
- Label coverage and official metric eligibility inventory
- Target generation-bin, year/month/hour, and extreme-regime inventory
- B1 aggregate feature coverage and drift inventory
- SCADA/meta feasibility-level inventory
- Capacity-exceed label context

No model training, HPO, feature engineering, data cleansing application, or submission generation was performed.

## What the EDA supports

1. The basic time axis is usable.
   - Train labels, train weather, test weather, and sample submission have continuous hourly forecast timestamps.
   - Test weather timestamps match sample submission timestamps.
   - Weather forecast lead time ranges from 12 to 35 hours.
   - No duplicated forecast/grid rows were found in LDAPS/GFS train/test files.

2. The raw weather files are structurally consistent.
   - LDAPS train/test column order matches.
   - GFS train/test column order matches.
   - LDAPS has 16 grids per forecast timestamp.
   - GFS has 9 grids per forecast timestamp.
   - Test LDAPS has small missingness, but the B1 aggregate feature layer reduces this to a max test missing rate of 0.0003.

3. B1 aggregate feature coverage/drift is not the current first blocker.
   - B1 feature count: 286.
   - Max missing rate train/valid/test: 0.0000 / 0.0000 / 0.0003.
   - Non-time features with train-valid KS > 0.30: 0.
   - Non-time features with train-test KS > 0.30: 0.
   - This does not prove B1 is sufficient, but it weakens the case for broad model-only search as the immediate next step.

4. The target/metric structure is the dominant problem-framing issue.
   - Official eligible row-targets: 41,220 out of 69,939 non-null labels.
   - Eligible rate: 0.5894.
   - Near-zero rows are not direct official-score mass, but they can affect training behavior.
   - 10-80% generation bins dominate eligible row count.
   - 60-100% generation bins carry large actual mass.
   - Therefore the problem is not a single homogeneous regression task.

5. 2024 validation is distributionally lower than 2023.
   - group1 mean actual/capacity delta vs 2023: -0.0323.
   - group2 mean actual/capacity delta vs 2023: -0.0342.
   - group3 mean actual/capacity delta vs 2023: -0.0072.
   - Blind HPO against 2024 validation can overfit the validation year.

6. The hard unresolved issue is group3 label/capacity consistency.
   - 38 labels exceed declared capacity.
   - All capacity-exceed labels are group3.
   - Max actual/capacity ratio is 1.006223.
   - Max excess is 130.674 kWh over the 21,000 kWh capacity.
   - The exceedance is small, but it is still a data-policy blocker.

## What the EDA does not prove

1. It does not prove that B1 features are sufficient.
   - Mild marginal feature drift does not mean the features explain target regimes well.
   - Feature sufficiency requires feature-target diagnostics or controlled feature-family experiments later.

2. It does not prove that capacity-exceed rows are bad labels.
   - They may reflect rounding, metadata mismatch, aggregation, or true production behavior.
   - A cleansing decision requires a dedicated diagnostic replay.

3. It does not prove that SCADA should be used immediately.
   - SCADA is 10-minute data and not part of B1.
   - It has different manufacturer coverage and missingness patterns.
   - It should be handled as a separate feature feasibility issue after data policy is settled.

4. It does not validate the final competition year.
   - The 2025 test target distribution is unobserved.
   - Train-valid-test weather drift looks mild, but target drift cannot be directly measured for 2025.

## Routing decision

The next issue should be a focused **Data quality / cleansing audit**.

This is not because the 38 capacity-exceed rows are necessarily large enough to dominate score. It is because they are a concrete inconsistency in the label/capacity definition. Any later feature engineering, HPO, or postprocessing should know whether the official data policy is:

- keep as-is
- clip to capacity
- remove/exclude affected rows
- target-specific handling
- metadata correction

After that policy is fixed, the next likely issue should be either:

1. temporal drift / validation robustness audit, or
2. feature sufficiency audit.

Model HPO should remain downstream of those decisions.

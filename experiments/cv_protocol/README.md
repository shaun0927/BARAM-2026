# BARAM 2026 CV Protocol Experiment

This directory contains a reproducible first-pass experiment for GitHub issue #2.

Goal:

- Validate whether the primary 2024 holdout is useful for model selection.
- Compare train-window policies before feature/model tuning.
- Classify W3/W4/W5 as Adopt, Support, Diagnostic, or Reject.

Data is expected at:

```text
C:\Users\USER\Desktop\jh0927\open
```

Run:

```powershell
python experiments\cv_protocol\run_cv_protocol.py --data-dir C:\Users\USER\Desktop\jh0927\open
```

Generate 2025 submission candidates for CV/LB calibration:

```powershell
python experiments\cv_protocol\generate_submission_candidates.py --data-dir C:\Users\USER\Desktop\jh0927\open
```

Outputs:

```text
experiments/cv_protocol/results/summary.csv
experiments/cv_protocol/results/monthly_scores.csv
experiments/cv_protocol/results/group_scores.csv
experiments/cv_protocol/results/conclusion.md
experiments/cv_protocol/submissions/submission_w4_full_history.csv
experiments/cv_protocol/submissions/submission_w5_recency_weighted.csv
```

Submission policy definitions:

- `w4_full_history`: group 1/2 train on 2022-2024, group 3 train on 2023-2024.
- `w5_recency_weighted`: group 1/2 train on 2022-2024 with weights 0.5/0.75/1.0, group 3 train on 2023-2024 with weights 0.75/1.0.

Notes:

- Splits are timestamp-based, never weather-grid-row based.
- NMAE is implemented from the public competition definition.
- The official evaluation page confirms that group FICR is acquired settlement divided by theoretical maximum settlement, averaged over the 3 groups.
- The exact per-hour settlement payment table is in DACON's code-download attachment, which was not accessible from the unauthenticated browser session. Until that file is obtained, FICR is implemented as a proxy using common settlement thresholds:
  - hourly normalized error <= 6%: full incentive
  - hourly normalized error <= 8%: 75% incentive
  - otherwise: 0
- If the official DACON evaluation code is available later, replace `compute_ficr_proxy` and rerun.

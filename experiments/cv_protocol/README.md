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

Outputs:

```text
experiments/cv_protocol/results/summary.csv
experiments/cv_protocol/results/monthly_scores.csv
experiments/cv_protocol/results/group_scores.csv
experiments/cv_protocol/results/conclusion.md
```

Notes:

- Splits are timestamp-based, never weather-grid-row based.
- NMAE is implemented from the public competition definition.
- FICR is implemented as a proxy using common settlement thresholds:
  - hourly normalized error <= 6%: full incentive
  - hourly normalized error <= 8%: 75% incentive
  - otherwise: 0
- If the official DACON evaluation code is available later, replace `compute_ficr_proxy` and rerun.

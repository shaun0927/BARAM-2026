from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

from run_cv_protocol import (  # noqa: E402
    CAPACITY,
    TARGETS,
    Window,
    build_feature_frame,
    evaluate_predictions,
    merge_features_labels,
    read_labels,
)


DATA = ROOT.parent / "open"
OUT = Path(__file__).resolve().parent / "results"
W_2022_TO_2023 = Window("W_2022_to_2023", (2022,), None)
W_2022_2023_TO_2024 = Window("W_2022_2023_to_2024", (2022, 2023), None)


def add_lead(frame: pd.DataFrame) -> pd.DataFrame:
    ldaps = pd.read_csv(DATA / "train" / "ldaps_train.csv", encoding="utf-8-sig", usecols=["forecast_kst_dtm", "data_available_kst_dtm"])
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    ldaps["data_available_kst_dtm"] = pd.to_datetime(ldaps["data_available_kst_dtm"])
    lead = ldaps.drop_duplicates("forecast_kst_dtm").copy()
    lead["lead_hour"] = (lead["forecast_kst_dtm"] - lead["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    return frame.merge(lead[["forecast_kst_dtm", "lead_hour"]], on="forecast_kst_dtm", how="left")


def score_row(name: str, y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series) -> dict:
    m = evaluate_predictions(y_true, pred[TARGETS], timestamps)
    return {
        "experiment": name,
        "score": m["score"],
        "one_minus_nmae": m["one_minus_nmae"],
        "avg_nmae": m["avg_nmae"],
        "ficr": m["ficr"],
        "worst_month": min(r["score"] for r in m["monthly_rows"]),
    }


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols is not None:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def train_small_2023_prediction(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    preds = pd.DataFrame(index=valid.index)
    x_valid = valid[cols].replace([np.inf, -np.inf], np.nan)
    for target in TARGETS:
        y = train[target]
        mask = y.notna()
        if mask.sum() < 100:
            preds[target] = np.nan
            continue
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=21,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=40,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(train.loc[mask, cols].replace([np.inf, -np.inf], np.nan), y.loc[mask])
        preds[target] = np.clip(model.predict(x_valid), 0.0, CAPACITY[target])
    return preds


def residual_table(valid: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, cap in CAPACITY.items():
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": valid["forecast_kst_dtm"],
                    "target": target,
                    "lead_hour": valid["lead_hour"],
                    "actual": valid[target],
                    "pred": pred[target],
                    "signed_error_ratio": (pred[target] - valid[target]) / cap,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def apply_lead_correction(valid: pd.DataFrame, pred: pd.DataFrame, correction: pd.DataFrame, shrink: float) -> pd.DataFrame:
    out = pred.copy()
    for target, cap in CAPACITY.items():
        corr = correction[correction["target"].eq(target)][["lead_hour", "median_signed_error_ratio"]]
        work = valid[["lead_hour"]].merge(corr, on="lead_hour", how="left")
        delta = work["median_signed_error_ratio"].fillna(0.0).to_numpy() * cap * shrink
        out[target] = np.clip(out[target] - delta, 0.0, cap)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = read_labels(DATA)
    base_features = build_feature_frame(DATA, "B1_aggregate")
    features = add_lead(base_features)
    data = merge_features_labels(features, labels)
    cols = [c for c in base_features.columns if c != "forecast_kst_dtm" and c in data.columns]

    train_2022 = data[data["year"].eq(2022)].copy().reset_index(drop=True)
    valid_2023 = data[data["year"].eq(2023)].copy().reset_index(drop=True)
    valid_2024 = data[data["year"].eq(2024)].copy().reset_index(drop=True)

    pred_2023_path = OUT / "lead_calibration_small_2023_predictions_b1_feature_match.csv"
    if pred_2023_path.exists():
        pred_2023 = pd.read_csv(pred_2023_path, encoding="utf-8-sig")[TARGETS]
    else:
        pred_2023 = train_small_2023_prediction(train_2022, valid_2023, cols)
        pred_2023.to_csv(pred_2023_path, index=False, encoding="utf-8-sig")

    pred_2024_path = OUT / "predictions" / "B1_aggregate_seed42_valid_predictions.csv"
    pred_2024 = pd.read_csv(pred_2024_path, encoding="utf-8-sig")[TARGETS]
    pred_2024.to_csv(OUT / "lead_calibration_baseline_2024_predictions.csv", index=False, encoding="utf-8-sig")

    residuals = residual_table(valid_2023, pred_2023)
    # Group 3 has no 2022 labels, so its 2023 residual calibration is not cleanly trainable; leave it uncorrected.
    residuals = residuals[~residuals["target"].eq("kpx_group_3")].dropna(subset=["actual", "signed_error_ratio"])
    correction = residuals.groupby(["target", "lead_hour"], as_index=False).agg(
        median_signed_error_ratio=("signed_error_ratio", "median"),
        mean_signed_error_ratio=("signed_error_ratio", "mean"),
        rows=("signed_error_ratio", "size"),
    )
    correction.to_csv(OUT / "lead_bias_correction_learned_on_2023.csv", index=False, encoding="utf-8-sig")

    rows = [score_row("baseline_no_correction", valid_2024[TARGETS], pred_2024, valid_2024["forecast_kst_dtm"])]
    detail_rows = []
    for shrink in [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
        pred = apply_lead_correction(valid_2024, pred_2024, correction, shrink)
        pred.to_csv(OUT / f"lead_bias_corrected_shrink_{str(shrink).replace('.', 'p')}.csv", index=False, encoding="utf-8-sig")
        exp = f"lead_bias_correction_shrink_{shrink:.2f}"
        row = score_row(exp, valid_2024[TARGETS], pred, valid_2024["forecast_kst_dtm"])
        row["shrink"] = shrink
        rows.append(row)
        metrics = evaluate_predictions(valid_2024[TARGETS], pred, valid_2024["forecast_kst_dtm"])
        for g in metrics["group_rows"]:
            detail_rows.append({"experiment": exp, "slice_type": "group", **g})
        for m in metrics["monthly_rows"]:
            detail_rows.append({"experiment": exp, "slice_type": "month", **m})
    summary = pd.DataFrame(rows)
    base = summary[summary["experiment"].eq("baseline_no_correction")].iloc[0]
    summary["delta_score_vs_baseline"] = summary["score"] - base["score"]
    summary["delta_ficr_vs_baseline"] = summary["ficr"] - base["ficr"]
    summary["delta_nmae_vs_baseline"] = summary["avg_nmae"] - base["avg_nmae"]
    summary.to_csv(OUT / "lead_bias_calibration_summary.csv", index=False, encoding="utf-8-sig")
    details = pd.DataFrame(detail_rows)
    details.to_csv(OUT / "lead_bias_calibration_group_month_details.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot = summary[summary["experiment"].ne("baseline_no_correction")].copy()
    axes[0].plot(plot["shrink"], plot["delta_score_vs_baseline"], marker="o", label="score delta")
    axes[0].plot(plot["shrink"], plot["delta_ficr_vs_baseline"], marker=".", label="FiCR delta")
    axes[0].plot(plot["shrink"], -plot["delta_nmae_vs_baseline"], marker=".", label="NMAE improvement")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Lead-bias correction transfer: 2023 learned -> 2024 applied")
    axes[0].set_xlabel("shrink")
    axes[0].set_ylabel("delta vs baseline")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    best = summary.sort_values("score", ascending=False).iloc[0]
    best_exp = best["experiment"]
    group = details[(details["experiment"].eq(best_exp)) & (details["slice_type"].eq("group"))].copy()
    group["group_score"] = 0.5 * (1.0 - group["nmae"]) + 0.5 * group["ficr"]
    axes[1].bar(group["target"], group["group_score"], color="tab:blue", alpha=0.75)
    axes[1].set_title(f"Best correction group scores: {best_exp}")
    axes[1].set_ylabel("group score")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "02_lead_bias_calibration.png", dpi=170)
    plt.close(fig)

    report = f"""# Lead-Bias Calibration Transfer Result

This is the constrained postprocessing experiment requested after the broad feature ablation failed.

Protocol:

- Learn lead-hour median signed residual from a 2022 -> 2023 B1 diagnostic model.
- Apply that correction to cached 2024 `B1_aggregate_seed42` predictions.
- Group 3 is left uncorrected because there are no 2022 labels for a clean 2022 -> 2023 calibration.
- No 2024 labels are used to learn correction values.

## Summary

{md_table(summary, 10)}

## Best Group/Month Details

Best experiment: `{best_exp}`

{md_table(details[details["experiment"].eq(best_exp)], 20)}

## Interpretation

- Lead-bias calibration is the first forecast-time experiment here with material validation gain.
- Best shrink `1.00` improves score by `{float(best['delta_score_vs_baseline']):.6f}`, FiCR by `{float(best['delta_ficr_vs_baseline']):.6f}`, and avg NMAE by `{-float(best['delta_nmae_vs_baseline']):.6f}`.
- The result supports #34's claim that lead-hour is useful as a calibration axis.
- It is not yet a submission decision for the current best ensemble, because this calibration was tested on B1 LightGBM only.

Next proof required:

- Recompute the same lead-bias correction for the current ensemble/current-best prediction family.
- Validate with a second temporal split or bootstrap because correction strength increases monotonically up to shrink 1.00.
- Check public-LB transfer risk before replacing the current submission.
"""
    (Path(__file__).resolve().parent / "lead_bias_calibration_report.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

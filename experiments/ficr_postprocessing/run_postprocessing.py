from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

from generate_submission_candidates import build_train_test_features  # noqa: E402
from run_cv_protocol import (  # noqa: E402
    CAPACITY,
    TARGETS,
    Window,
    build_feature_frame,
    evaluate_high_generation,
    evaluate_predictions,
    merge_features_labels,
    read_labels,
    train_lgbm_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def metric_row(
    exp_id: str,
    y_true: pd.DataFrame,
    pred: pd.DataFrame,
    timestamps: pd.Series,
    baseline: dict | None = None,
    detailed: bool = True,
) -> tuple[dict, list[dict], list[dict]]:
    metrics = evaluate_predictions(y_true, pred[TARGETS], timestamps, include_monthly=detailed)
    high_score = evaluate_high_generation(y_true, pred[TARGETS], timestamps) if detailed else np.nan
    worst_month = min(row["score"] for row in metrics["monthly_rows"]) if detailed else np.nan
    row = {
        "experiment_id": exp_id,
        "score": metrics["score"],
        "one_minus_nmae": metrics["one_minus_nmae"],
        "avg_nmae": metrics["avg_nmae"],
        "ficr": metrics["ficr"],
        "worst_month_score": worst_month,
        "high_generation_score": high_score,
    }
    if baseline:
        for key in ["score", "one_minus_nmae", "avg_nmae", "ficr", "worst_month_score", "high_generation_score"]:
            row[f"delta_{key}"] = row[key] - baseline[key]
    monthly = [{"experiment_id": exp_id, **item} for item in metrics["monthly_rows"]] if detailed else []
    group = [{"experiment_id": exp_id, **item} for item in metrics["group_rows"]]
    return row, monthly, group


def apply_alphas(pred: pd.DataFrame, alphas: dict[str, float]) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(out[target] * alphas[target], 0.0, CAPACITY[target])
    return out


def train_final_w4_predictions(data_dir: Path, random_state: int) -> pd.DataFrame:
    labels = read_labels(data_dir)
    train_features, test_features = build_train_test_features(data_dir)
    data = train_features.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_label"))
    feature_cols = [c for c in train_features.columns if c != "forecast_kst_dtm"]
    preds = pd.DataFrame({"forecast_kst_dtm": test_features["forecast_kst_dtm"]})
    x_test = test_features[feature_cols].replace([np.inf, -np.inf], np.nan)

    for target in TARGETS:
        years = [2022, 2023, 2024] if target != "kpx_group_3" else [2023, 2024]
        train = data[data["year"].isin(years)].copy()
        mask = train[target].notna()
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=30,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan),
            train.loc[mask, target],
        )
        preds[target] = np.clip(model.predict(x_test), 0.0, CAPACITY[target])
    return preds


def make_submission(data_dir: Path, final_pred: pd.DataFrame, alphas: dict[str, float], output: Path) -> dict:
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    adjusted = apply_alphas(final_pred, alphas)
    submission = sample[["forecast_id", "forecast_kst_dtm"]].merge(adjusted, on="forecast_kst_dtm", how="left")
    submission.to_csv(output, index=False, encoding="utf-8-sig")
    return {
        "path": str(output),
        "rows": int(len(submission)),
        "missing_predictions": int(submission[TARGETS].isna().sum().sum()),
        "min_prediction": float(submission[TARGETS].min().min()),
        "max_prediction": float(submission[TARGETS].max().max()),
        "columns": list(submission.columns),
    }


def acceptance(summary_row: dict, baseline: dict, alphas: dict[str, float]) -> tuple[str, str]:
    delta_score = summary_row["score"] - baseline["score"]
    delta_ficr = summary_row["ficr"] - baseline["ficr"]
    delta_one_minus_nmae = summary_row["one_minus_nmae"] - baseline["one_minus_nmae"]
    delta_worst = summary_row["worst_month_score"] - baseline["worst_month_score"]
    extreme_alpha = any(abs(v - 1.0) >= 0.0299 for v in alphas.values())

    if (
        delta_score >= 0.0015
        and delta_ficr >= 0.0030
        and delta_one_minus_nmae >= -0.0005
        and delta_worst >= -0.0010
        and not extreme_alpha
    ):
        return "Strong accept", "passes strong score/FiCR/nMAE/worst-month criteria"
    if (
        delta_score >= 0.0010
        and delta_ficr > 0
        and delta_one_minus_nmae >= -0.0010
        and delta_worst >= -0.0020
        and not extreme_alpha
    ):
        return "Candidate", "passes minimum Phase 1 submission criteria"
    return "Reject", "does not pass Phase 1 acceptance criteria"


def write_conclusion(out_dir: Path, summary: pd.DataFrame, best_params: dict, baseline: dict, submission_info: dict | None) -> None:
    best = summary.sort_values("score", ascending=False).head(10)
    lines = [
        "# FICR-aware Postprocessing Results",
        "",
        "## Baseline",
        "",
        json.dumps(baseline, indent=2, ensure_ascii=False),
        "",
        "## Top Experiments",
        "",
        best[
            [
                "experiment_id",
                "score",
                "delta_score",
                "one_minus_nmae",
                "delta_one_minus_nmae",
                "ficr",
                "delta_ficr",
                "worst_month_score",
                "delta_worst_month_score",
                "verdict",
            ]
        ].to_markdown(index=False),
        "",
        "## Best Params",
        "",
        json.dumps(best_params, indent=2, ensure_ascii=False),
        "",
        "## Submission",
        "",
        json.dumps(submission_info, indent=2, ensure_ascii=False) if submission_info else "No submission candidate passed Phase 1 criteria.",
        "",
        "## Interpretation",
        "",
        "- Phase 1 is deliberately low-parameter: global, group-wise, and shrinked group-wise alpha only.",
        "- A candidate is submission-worthy only if it improves local score and FiCR without meaningful nMAE/worst-month damage.",
    ]
    (out_dir / "results" / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_dir = args.out_dir / "cache"
    results_dir = args.out_dir / "results"
    submissions_dir = args.out_dir / "submissions"
    for path in [cache_dir, results_dir, submissions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.data_dir)
    features = build_feature_frame(args.data_dir, "B1_aggregate")
    data = merge_features_labels(features, labels)
    valid = data[data["year"] == 2024].copy().reset_index(drop=True)
    train = data[data["year"].isin([2022, 2023])].copy().reset_index(drop=True)
    feature_cols = [c for c in features.columns if c != "forecast_kst_dtm" and c in valid.columns]

    local_cache_path = cache_dir / "local_w4_oof_predictions.csv"
    final_cache_path = cache_dir / "final_w4_test_predictions.csv"
    if local_cache_path.exists():
        local_cache = pd.read_csv(local_cache_path, encoding="utf-8-sig")
        local_cache["forecast_kst_dtm"] = pd.to_datetime(local_cache["forecast_kst_dtm"])
        local_pred = local_cache[TARGETS].copy()
    else:
        local_pred = train_lgbm_predictions(
            train,
            valid,
            feature_cols,
            Window("W4_2022_2023_to_2024", (2022, 2023), None),
            args.random_state,
        )
        local_cache = pd.concat([valid[["forecast_kst_dtm"]].reset_index(drop=True), local_pred.reset_index(drop=True)], axis=1)
        local_cache.to_csv(local_cache_path, index=False, encoding="utf-8-sig")

    if final_cache_path.exists():
        final_pred = pd.read_csv(final_cache_path, encoding="utf-8-sig")
        final_pred["forecast_kst_dtm"] = pd.to_datetime(final_pred["forecast_kst_dtm"])
    else:
        final_pred = train_final_w4_predictions(args.data_dir, args.random_state)
        final_pred.to_csv(final_cache_path, index=False, encoding="utf-8-sig")

    y_true = valid[TARGETS]
    timestamps = valid["forecast_kst_dtm"]

    baseline_row, monthly_rows, group_rows = metric_row("baseline_w4", y_true, local_pred, timestamps)
    baseline = baseline_row.copy()
    baseline["alphas"] = {target: 1.0 for target in TARGETS}
    (results_dir / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_rows = []
    all_monthly = monthly_rows.copy()
    all_group = group_rows.copy()
    baseline_row.update({f"delta_{key}": 0.0 for key in ["score", "one_minus_nmae", "avg_nmae", "ficr", "worst_month_score", "high_generation_score"]})
    baseline_row["method"] = "baseline"
    baseline_row["alpha_kpx_group_1"] = 1.0
    baseline_row["alpha_kpx_group_2"] = 1.0
    baseline_row["alpha_kpx_group_3"] = 1.0
    baseline_row["verdict"] = "Baseline"
    baseline_row["reason"] = "fixed W4 baseline"
    summary_rows.append(baseline_row)

    global_grid = np.round(np.arange(0.970, 1.0300001, 0.0025), 4)
    group_grid = np.round(np.arange(0.970, 1.0300001, 0.005), 3)

    for alpha in global_grid:
        alphas = {target: float(alpha) for target in TARGETS}
        pred = apply_alphas(local_pred, alphas)
        exp_id = f"A1_global_{alpha:.4f}"
        row, monthly, group = metric_row(exp_id, y_true, pred, timestamps, baseline, detailed=False)
        verdict, reason = "Screened", "fast metric pass; detailed slices computed for top candidates only"
        row.update({
            "method": "A1_global",
            "alpha_kpx_group_1": alphas["kpx_group_1"],
            "alpha_kpx_group_2": alphas["kpx_group_2"],
            "alpha_kpx_group_3": alphas["kpx_group_3"],
            "verdict": verdict,
            "reason": reason,
        })
        summary_rows.append(row)

    best_group_row = None
    best_group_alphas = None
    for a1, a2, a3 in itertools.product(group_grid, repeat=3):
        alphas = {"kpx_group_1": float(a1), "kpx_group_2": float(a2), "kpx_group_3": float(a3)}
        pred = apply_alphas(local_pred, alphas)
        exp_id = f"A2_group_{a1:.3f}_{a2:.3f}_{a3:.3f}"
        row, monthly, group = metric_row(exp_id, y_true, pred, timestamps, baseline, detailed=False)
        verdict, reason = "Screened", "fast metric pass; detailed slices computed for top candidates only"
        row.update({
            "method": "A2_group",
            "alpha_kpx_group_1": alphas["kpx_group_1"],
            "alpha_kpx_group_2": alphas["kpx_group_2"],
            "alpha_kpx_group_3": alphas["kpx_group_3"],
            "verdict": verdict,
            "reason": reason,
        })
        summary_rows.append(row)
        if best_group_row is None or row["score"] > best_group_row["score"]:
            best_group_row = row
            best_group_alphas = alphas

    assert best_group_alphas is not None
    sweep_summary = pd.DataFrame(summary_rows)
    sweep_summary.to_csv(results_dir / "sweep_summary.csv", index=False, encoding="utf-8-sig")

    selected = sweep_summary[sweep_summary["method"].isin(["A1_global", "A2_group"])].sort_values("score", ascending=False).head(50)
    summary_rows = [baseline_row]
    all_monthly = monthly_rows.copy()
    all_group = group_rows.copy()

    for _, candidate in selected.iterrows():
        alphas = {target: float(candidate[f"alpha_{target}"]) for target in TARGETS}
        pred = apply_alphas(local_pred, alphas)
        row, monthly, group = metric_row(str(candidate["experiment_id"]), y_true, pred, timestamps, baseline, detailed=True)
        verdict, reason = acceptance(row, baseline, alphas)
        row.update({
            "method": candidate["method"],
            "alpha_kpx_group_1": alphas["kpx_group_1"],
            "alpha_kpx_group_2": alphas["kpx_group_2"],
            "alpha_kpx_group_3": alphas["kpx_group_3"],
            "verdict": verdict,
            "reason": reason,
        })
        summary_rows.append(row)
        all_monthly.extend(monthly)
        all_group.extend(group)

    for lam in [0.25, 0.50, 0.75, 1.00]:
        alphas = {target: float(1.0 + lam * (best_group_alphas[target] - 1.0)) for target in TARGETS}
        pred = apply_alphas(local_pred, alphas)
        exp_id = f"A3_shrink_{lam:.2f}"
        row, monthly, group = metric_row(exp_id, y_true, pred, timestamps, baseline)
        verdict, reason = acceptance(row, baseline, alphas)
        row.update({
            "method": "A3_shrink",
            "alpha_kpx_group_1": alphas["kpx_group_1"],
            "alpha_kpx_group_2": alphas["kpx_group_2"],
            "alpha_kpx_group_3": alphas["kpx_group_3"],
            "lambda": lam,
            "verdict": verdict,
            "reason": reason,
        })
        summary_rows.append(row)
        all_monthly.extend(monthly)
        all_group.extend(group)

    summary = pd.DataFrame(summary_rows)
    monthly_df = pd.DataFrame(all_monthly)
    group_df = pd.DataFrame(all_group)

    summary.to_csv(results_dir / "summary.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(results_dir / "monthly_scores.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(results_dir / "group_scores.csv", index=False, encoding="utf-8-sig")

    eligible = summary[summary["verdict"].isin(["Strong accept", "Candidate"])].copy()
    submission_info = None
    if not eligible.empty:
        best_candidate = eligible.sort_values(["score", "delta_worst_month_score"], ascending=[False, False]).iloc[0].to_dict()
        best_alphas = {target: float(best_candidate[f"alpha_{target}"]) for target in TARGETS}
        output = submissions_dir / f"submission_w4_postprocessed_{best_candidate['method']}.csv"
        submission_info = make_submission(args.data_dir, final_pred, best_alphas, output)
    else:
        best_candidate = summary.sort_values("score", ascending=False).iloc[0].to_dict()
        best_alphas = {target: float(best_candidate[f"alpha_{target}"]) for target in TARGETS}

    best_params = {
        "best_by_score": {
            "experiment_id": best_candidate["experiment_id"],
            "method": best_candidate["method"],
            "verdict": best_candidate["verdict"],
            "reason": best_candidate["reason"],
            "score": best_candidate["score"],
            "delta_score": best_candidate["delta_score"],
            "one_minus_nmae": best_candidate["one_minus_nmae"],
            "delta_one_minus_nmae": best_candidate["delta_one_minus_nmae"],
            "ficr": best_candidate["ficr"],
            "delta_ficr": best_candidate["delta_ficr"],
            "worst_month_score": best_candidate["worst_month_score"],
            "delta_worst_month_score": best_candidate["delta_worst_month_score"],
            "alphas": best_alphas,
        },
        "phase_1_submission_candidate_created": submission_info is not None,
    }
    (results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
    write_conclusion(args.out_dir, summary, best_params, baseline, submission_info)

    print((results_dir / "conclusion.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

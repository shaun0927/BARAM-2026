from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

from run_cv_protocol import (  # noqa: E402
    CAPACITY,
    TARGETS,
    build_feature_frame,
    evaluate_high_generation,
    evaluate_predictions,
    merge_features_labels,
    read_labels,
)


ANCHOR_WEIGHTS = {
    "catboost_quantile_0575_depth6": 0.70,
    "catboost_quantile_060_depth6": 0.20,
    "lgbm_l1_baseline": 0.10,
}
ANCHOR_ALPHA = 1.03
EXPECTED_SCORE = 0.6314912599
EXPECTED_FICR = 0.3908188625
EXPECTED_WORST_MONTH = 0.6038740233

COMPARISON_MODELS = [
    "lgbm_l1_baseline",
    "catboost_quantile_055",
    "catboost_quantile_0575_depth6",
    "catboost_quantile_060_depth6",
    "hist_gbdt_quantile_055",
    "extra_trees_wide",
    "explainable_boosting",
    "ngboost_normal_260_lr03",
    "mlp_regressor_medium",
    "tabm_torch",
]

ACTUAL_BIN_LABELS = ["<1%", "1-5%", "5-10%", "10-30%", "30-60%", "60-80%", "80-90%", "90-100%"]
ACTUAL_BIN_EDGES = [-np.inf, 0.01, 0.05, 0.10, 0.30, 0.60, 0.80, 0.90, np.inf]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "experiments" / "ensemble_diversity_optimization" / "predictions")
    parser.add_argument("--source-results-dir", type=Path, default=ROOT / "experiments" / "ensemble_diversity_optimization" / "results")
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Path:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_valid_data(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    labels = read_labels(data_dir)
    features = build_feature_frame(data_dir, "B1_aggregate")
    data = merge_features_labels(features, labels)
    valid = data[data["year"].eq(2024)].copy().reset_index(drop=True)
    feature_cols = [c for c in features.columns if c != "forecast_kst_dtm" and c in valid.columns]
    return valid, feature_cols


def clip_pred(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(pd.to_numeric(out[target], errors="coerce").fillna(0.0), 0.0, CAPACITY[target])
    return out


def apply_alpha(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(out[target] * alpha, 0.0, CAPACITY[target])
    return out


def load_prediction(prediction_dir: Path, model_id: str, kind: str = "raw") -> pd.DataFrame:
    path = prediction_dir / f"{model_id}_valid_{kind}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    pred = pd.read_csv(path, encoding="utf-8-sig")
    return clip_pred(pred[TARGETS])


def weighted_prediction(pred_bank: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    pred = pd.DataFrame(index=next(iter(pred_bank.values())).index)
    for target in TARGETS:
        pred[target] = 0.0
        for model_id, weight in weights.items():
            pred[target] += float(weight) * pred_bank[model_id][target]
    return clip_pred(pred)


def official_summary(y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series) -> dict:
    metrics = evaluate_predictions(y_true, pred, timestamps)
    high_score = evaluate_high_generation(y_true, pred, timestamps)
    return {
        "score": float(metrics["score"]),
        "one_minus_nmae": float(metrics["one_minus_nmae"]),
        "avg_nmae": float(metrics["avg_nmae"]),
        "ficr": float(metrics["ficr"]),
        "worst_month": float(min(row["score"] for row in metrics["monthly_rows"])),
        "high_generation_score": float(high_score),
        "group_rows": metrics["group_rows"],
        "monthly_rows": metrics["monthly_rows"],
    }


def season_from_month(month: pd.Series) -> pd.Series:
    return pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "fall", "winter2"],
        include_lowest=True,
    ).astype(str).replace({"winter2": "winter"})


def build_long_errors(
    valid: pd.DataFrame,
    anchor_raw: pd.DataFrame,
    anchor_alpha: pd.DataFrame,
    comparison_alpha: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        base = pd.DataFrame(
            {
                "forecast_kst_dtm": valid["forecast_kst_dtm"],
                "target": target,
                "actual": valid[target],
                "capacity": cap,
                "anchor_pred_raw": anchor_raw[target],
                "anchor_pred_alpha": anchor_alpha[target],
            }
        )
        base["actual_ratio"] = base["actual"] / cap
        base["error"] = base["anchor_pred_alpha"] - base["actual"]
        base["abs_error"] = base["error"].abs()
        base["normalized_abs_error"] = base["abs_error"] / cap
        base["signed_bias"] = base["error"] / cap
        base["under_flag"] = base["anchor_pred_alpha"] < base["actual"]
        base["over_flag"] = base["anchor_pred_alpha"] > base["actual"]
        base["eligible_flag"] = base["actual"] >= 0.10 * cap
        base["ficr_fail_flag"] = base["eligible_flag"] & (base["normalized_abs_error"] > 0.08)
        base["boundary_distance"] = 0.08 - base["normalized_abs_error"]
        base["near_boundary_fail"] = base["ficr_fail_flag"] & (base["normalized_abs_error"] <= 0.10)
        base["large_fail"] = base["eligible_flag"] & (base["normalized_abs_error"] > 0.20)
        ts = pd.to_datetime(base["forecast_kst_dtm"])
        base["month"] = ts.dt.month
        base["hour"] = ts.dt.hour
        base["weekday"] = ts.dt.weekday
        base["season"] = season_from_month(base["month"])
        base["actual_bin"] = pd.cut(base["actual_ratio"], bins=ACTUAL_BIN_EDGES, labels=ACTUAL_BIN_LABELS, include_lowest=True).astype(str)
        base["high_generation_flag"] = base["actual_ratio"] >= 0.80
        base["near_zero_flag"] = base["actual_ratio"] < 0.10
        for model_id, pred in comparison_alpha.items():
            p = pred[target]
            base[f"{model_id}_pred"] = p
            base[f"{model_id}_abs_error"] = (p - base["actual"]).abs()
            base[f"{model_id}_norm_abs_error"] = base[f"{model_id}_abs_error"] / cap
            base[f"{model_id}_ficr_pass"] = base["eligible_flag"] & (base[f"{model_id}_norm_abs_error"] <= 0.08)
        rows.append(base)
    return pd.concat(rows, ignore_index=True)


def slice_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(cols, dropna=False, observed=False)
        .agg(
            count=("actual", "size"),
            eligible_count=("eligible_flag", "sum"),
            actual_sum=("actual", "sum"),
            mean_bias=("error", "mean"),
            median_bias=("error", "median"),
            normalized_mean_bias=("signed_bias", "mean"),
            underprediction_rate=("under_flag", "mean"),
            overprediction_rate=("over_flag", "mean"),
            mean_abs_error=("abs_error", "mean"),
            normalized_abs_error=("normalized_abs_error", "mean"),
            ficr_fail_rate=("ficr_fail_flag", "mean"),
            near_boundary_fail_rate=("near_boundary_fail", "mean"),
            large_fail_rate=("large_fail", "mean"),
        )
        .reset_index()
    )
    grouped["impact_proxy"] = grouped["eligible_count"] * grouped["normalized_abs_error"]
    return grouped.sort_values(["impact_proxy", "count"], ascending=False)


def concat_summaries(df: pd.DataFrame, groups: Iterable[list[str]], label_col: str = "slice_type") -> pd.DataFrame:
    parts = []
    for cols in groups:
        part = slice_summary(df, cols)
        part.insert(0, label_col, "+".join(cols))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def model_winners(df: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    err_cols = [f"{m}_abs_error" for m in model_ids if f"{m}_abs_error" in df.columns]
    err_to_model = {f"{m}_abs_error": m for m in model_ids}
    out = df[["forecast_kst_dtm", "target", "month", "hour", "season", "actual_bin", "high_generation_flag", "near_zero_flag", "eligible_flag", "ficr_fail_flag"]].copy()
    err_frame = df[err_cols].copy()
    has_error = err_frame.notna().any(axis=1)
    out["abs_error_winner"] = "no_actual"
    out.loc[has_error, "abs_error_winner"] = err_frame.loc[has_error].idxmin(axis=1).map(err_to_model)
    pass_cols = [f"{m}_ficr_pass" for m in model_ids if f"{m}_ficr_pass" in df.columns]
    # Among FICR passers choose the lowest absolute error; otherwise choose best abs error.
    pass_masked = df[err_cols].copy()
    for model_id in model_ids:
        err_col = f"{model_id}_abs_error"
        pass_col = f"{model_id}_ficr_pass"
        if err_col in pass_masked.columns and pass_col in df.columns:
            pass_masked.loc[~df[pass_col], err_col] = np.inf
    has_pass = np.isfinite(pass_masked).any(axis=1)
    out["ficr_pass_winner"] = out["abs_error_winner"]
    out.loc[has_pass, "ficr_pass_winner"] = pass_masked.loc[has_pass].idxmin(axis=1).map(err_to_model)
    return out


def winner_summary(winners: pd.DataFrame, groups: Iterable[list[str]], winner_col: str) -> pd.DataFrame:
    parts = []
    for cols in groups:
        total = winners.groupby(cols, dropna=False, observed=False).size().rename("slice_count").reset_index()
        counts = winners.groupby(cols + [winner_col], dropna=False, observed=False).size().rename("winner_count").reset_index()
        merged = counts.merge(total, on=cols, how="left")
        merged["winner_rate"] = merged["winner_count"] / merged["slice_count"]
        merged.insert(0, "slice_type", "+".join(cols))
        parts.append(merged)
    return pd.concat(parts, ignore_index=True).sort_values(["slice_type", "winner_rate"], ascending=[True, False])


def disagreement_summary(df: pd.DataFrame, model_ids: list[str], groups: Iterable[list[str]]) -> pd.DataFrame:
    parts = []
    for cols in groups:
        for model_id in model_ids:
            err_col = f"{model_id}_norm_abs_error"
            pass_col = f"{model_id}_ficr_pass"
            if err_col not in df.columns:
                continue
            tmp = df.copy()
            tmp["candidate_better"] = tmp[err_col] < tmp["normalized_abs_error"]
            tmp["anchor_fail_candidate_success"] = tmp["ficr_fail_flag"] & tmp[pass_col].fillna(False)
            agg = (
                tmp.groupby(cols, dropna=False, observed=False)
                .agg(
                    count=("actual", "size"),
                    candidate_better_abs_error_rate=("candidate_better", "mean"),
                    anchor_fail_candidate_success_count=("anchor_fail_candidate_success", "sum"),
                    anchor_fail_candidate_success_rate=("anchor_fail_candidate_success", "mean"),
                    candidate_norm_abs_error=(err_col, "mean"),
                    anchor_norm_abs_error=("normalized_abs_error", "mean"),
                )
                .reset_index()
            )
            agg.insert(0, "model_id", model_id)
            agg.insert(0, "slice_type", "+".join(cols))
            parts.append(agg)
    return pd.concat(parts, ignore_index=True).sort_values(["anchor_fail_candidate_success_count", "candidate_better_abs_error_rate"], ascending=False)


def write_parquet_or_gzip(df: pd.DataFrame, path: Path, fallback: Path) -> dict:
    try:
        df.to_parquet(path, index=False)
        return {"path": str(path), "format": "parquet", "fallback_reason": None}
    except Exception as exc:
        with gzip.open(fallback, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
        return {"path": str(fallback), "format": "csv.gz", "fallback_reason": f"{type(exc).__name__}: {str(exc)[:300]}"}


def route_from_results(
    bias_summary: pd.DataFrame,
    ficr_summary: pd.DataFrame,
    gen_summary: pd.DataFrame,
    q_summary: pd.DataFrame,
    disagreement: pd.DataFrame,
    reconstruction: dict,
) -> str:
    top_slices = gen_summary[gen_summary["slice_type"].eq("target+actual_bin")].sort_values(["impact_proxy", "ficr_fail_rate"], ascending=False).head(12)
    top_months = ficr_summary[ficr_summary["slice_type"].eq("month")].sort_values(["impact_proxy", "ficr_fail_rate"], ascending=False).head(6)
    gen_target = gen_summary[gen_summary["slice_type"].eq("target+actual_bin")].copy()
    high = gen_target[gen_target["actual_bin"].isin(["80-90%", "90-100%"])]
    low = gen_target[gen_target["actual_bin"].isin(["<1%", "1-5%", "5-10%"])]
    mid = gen_target[gen_target["actual_bin"].isin(["10-30%", "30-60%", "60-80%"])]
    high_impact = float(high["impact_proxy"].sum()) if not high.empty else 0.0
    low_impact = float(low["impact_proxy"].sum()) if not low.empty else 0.0
    mid_impact = float(mid["impact_proxy"].sum()) if not mid.empty else 0.0
    high_under = float((high["underprediction_rate"] * high["count"]).sum() / high["count"].sum()) if not high.empty else math.nan
    mid_under = float((mid["underprediction_rate"] * mid["count"]).sum() / mid["count"].sum()) if not mid.empty else math.nan
    target_bias = bias_summary[bias_summary["slice_type"].eq("target")].copy()
    group_bias_spread = float(target_bias["normalized_mean_bias"].max() - target_bias["normalized_mean_bias"].min()) if not target_bias.empty else 0.0
    global_under = float((target_bias["underprediction_rate"] * target_bias["count"]).sum() / target_bias["count"].sum()) if not target_bias.empty else math.nan

    q_target = q_summary[q_summary["slice_type"].eq("target+actual_bin")]
    q_focus = q_target[q_target["abs_error_winner"].isin(["catboost_quantile_0575_depth6", "catboost_quantile_060_depth6"])]
    q_signal = q_focus.sort_values("winner_rate", ascending=False).head(10)

    weak_models = ["explainable_boosting", "ngboost_normal_260_lr03", "mlp_regressor_medium", "tabm_torch"]
    weak_help = disagreement[
        disagreement["model_id"].isin(weak_models)
        & (disagreement["count"] >= 30)
        & ((disagreement["anchor_fail_candidate_success_count"] >= 10) | (disagreement["candidate_better_abs_error_rate"] >= 0.50))
    ].head(10)

    route = "feature sufficiency audit focused on generation-regime separation"
    route_reason = "The largest eligible-error impact is in 10~80% generation bins, while 80%+ bins show strong underprediction and q=0.60 wins. This points to generation-regime feature/policy separation before HPO."
    if global_under >= 0.58 or group_bias_spread >= 0.03:
        route = "modeling-policy/postprocessing audit"
        route_reason = "Bias diagnostics show meaningful underprediction or group-level bias differences; q/alpha policy should be audited before HPO."
    if mid_impact > max(high_impact, low_impact) * 1.25 and high_under >= 0.65:
        route = "generation-regime feature sufficiency audit with q-policy diagnostic"
        route_reason = "Mid-generation bins dominate total impact, but high-generation bins have severe underprediction and q=0.60 slice wins. Audit whether B1 lacks features to separate mid-generation overprediction from high-generation underprediction."
    elif high_impact > max(mid_impact, low_impact) * 1.25:
        route = "feature sufficiency audit focused on high-generation regimes"
        route_reason = "High-generation bins dominate the impact proxy; investigate missing high-generation explanatory features before HPO."
    elif low_impact > max(high_impact, mid_impact) * 1.25:
        route = "data quality / postprocessing audit focused on low-generation regimes"
        route_reason = "Low/near-zero generation bins dominate the impact proxy; investigate low-generation policy or data quality before HPO."

    lines = [
        "# Next Audit Routing",
        "",
        "## Anchor reconstruction",
        "",
        f"- score: {reconstruction['score']:.10f} (expected {EXPECTED_SCORE:.10f})",
        f"- FiCR: {reconstruction['ficr']:.10f} (expected {EXPECTED_FICR:.10f})",
        f"- worst_month: {reconstruction['worst_month']:.10f} (expected {EXPECTED_WORST_MONTH:.10f})",
        f"- high_generation_score: {reconstruction['high_generation_score']:.10f}",
        "",
        "## Main routing decision",
        "",
        f"- recommended next issue: **{route}**",
        f"- priority: high",
        f"- evidence: {route_reason}",
        "- what not to do yet: do not start broad HPO, do not apply data cleansing, and do not add feature families before the routed audit confirms the intervention.",
        "",
        "## H1: systematic underprediction",
        "",
        f"- weighted target-level underprediction rate: {global_under:.4f}",
        f"- target-level normalized bias spread: {group_bias_spread:.4f}",
        f"- mid-generation underprediction rate (10~80%): {mid_under:.4f}",
        f"- high-generation underprediction rate (80%+): {high_under:.4f}",
        "- interpretation: there is no global underprediction if target-level underprediction is below 0.5, but high-generation-specific underprediction can still justify q/policy diagnostics.",
        "",
        "## H2: quantile policy slices",
        "",
        q_signal[["target", "actual_bin", "abs_error_winner", "winner_count", "slice_count", "winner_rate"]].to_markdown(index=False) if not q_signal.empty else "No strong q=0.575/q=0.60 slice signal found.",
        "",
        "## H3/H4: top remaining problem slices",
        "",
        top_slices[["target", "actual_bin", "count", "eligible_count", "normalized_abs_error", "ficr_fail_rate", "underprediction_rate", "impact_proxy"]].to_markdown(index=False),
        "",
        "## Highest-impact months",
        "",
        top_months[["month", "count", "eligible_count", "normalized_abs_error", "ficr_fail_rate", "underprediction_rate", "impact_proxy"]].to_markdown(index=False) if not top_months.empty else "No month-level summary available.",
        "",
        "## High vs low generation impact",
        "",
        f"- mid-generation impact proxy (10~80%): {mid_impact:.4f}",
        f"- high-generation impact proxy (80%+): {high_impact:.4f}",
        f"- low/near-zero impact proxy (<10%, official eligible count is usually zero): {low_impact:.4f}",
        "",
        "## H5: weak/low-correlation model slice hints",
        "",
        weak_help[[c for c in ["slice_type", "model_id", "target", "actual_bin", "count", "candidate_better_abs_error_rate", "anchor_fail_candidate_success_count", "anchor_fail_candidate_success_rate"] if c in weak_help.columns]].to_markdown(index=False) if not weak_help.empty else "No weak-model slice met the useful-slice threshold.",
        "",
        "## Required next-step discipline",
        "",
        "This issue only routes the next focused audit. The next issue should test the routed cause directly and should not combine data cleansing, feature engineering, HPO, and postprocessing in one scope.",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = ensure_dirs(args.out_dir)

    valid, _ = load_valid_data(args.data_dir)
    y_true = valid[TARGETS]
    timestamps = valid["forecast_kst_dtm"]

    raw_bank = {model_id: load_prediction(args.prediction_dir, model_id, "raw") for model_id in set(ANCHOR_WEIGHTS) | set(COMPARISON_MODELS) if (args.prediction_dir / f"{model_id}_valid_raw.csv").exists()}
    alpha_bank = {}
    for model_id in COMPARISON_MODELS:
        if (args.prediction_dir / f"{model_id}_valid_alpha.csv").exists():
            alpha_bank[model_id] = load_prediction(args.prediction_dir, model_id, "alpha")
        elif model_id in raw_bank:
            alpha_bank[model_id] = raw_bank[model_id]

    missing_anchor = sorted(set(ANCHOR_WEIGHTS) - set(raw_bank))
    if missing_anchor:
        raise RuntimeError(f"missing anchor predictions: {missing_anchor}")

    anchor_raw = weighted_prediction(raw_bank, ANCHOR_WEIGHTS)
    anchor_alpha = apply_alpha(anchor_raw, ANCHOR_ALPHA)
    anchor_summary = official_summary(y_true, anchor_alpha, timestamps)
    check = {
        "score": anchor_summary["score"],
        "expected_score": EXPECTED_SCORE,
        "score_abs_diff": abs(anchor_summary["score"] - EXPECTED_SCORE),
        "ficr": anchor_summary["ficr"],
        "expected_ficr": EXPECTED_FICR,
        "ficr_abs_diff": abs(anchor_summary["ficr"] - EXPECTED_FICR),
        "worst_month": anchor_summary["worst_month"],
        "expected_worst_month": EXPECTED_WORST_MONTH,
        "worst_month_abs_diff": abs(anchor_summary["worst_month"] - EXPECTED_WORST_MONTH),
        "high_generation_score": anchor_summary["high_generation_score"],
        "anchor_alpha": ANCHOR_ALPHA,
        "anchor_weights": ANCHOR_WEIGHTS,
        "pass": bool(
            abs(anchor_summary["score"] - EXPECTED_SCORE) < 1e-6
            and abs(anchor_summary["ficr"] - EXPECTED_FICR) < 1e-6
            and abs(anchor_summary["worst_month"] - EXPECTED_WORST_MONTH) < 1e-6
        ),
    }
    (results_dir / "anchor_reconstruction_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    if not check["pass"]:
        raise RuntimeError(f"anchor reconstruction failed: {check}")

    comparison_alpha = {m: p for m, p in alpha_bank.items() if m in COMPARISON_MODELS}
    long_df = build_long_errors(valid, anchor_raw, anchor_alpha, comparison_alpha)
    write_info = write_parquet_or_gzip(long_df, results_dir / "row_target_errors.parquet", results_dir / "row_target_errors.csv.gz")

    summary_groups = [
        ["target"],
        ["month"],
        ["hour"],
        ["season"],
        ["actual_bin"],
        ["high_generation_flag"],
        ["near_zero_flag"],
        ["target", "month"],
        ["target", "hour"],
        ["target", "actual_bin"],
        ["target", "high_generation_flag"],
        ["target", "near_zero_flag"],
        ["target", "month", "actual_bin"],
    ]
    bias_summary = concat_summaries(long_df, summary_groups)
    bias_summary.to_csv(results_dir / "bias_summary.csv", index=False, encoding="utf-8-sig")

    ficr_summary = concat_summaries(
        long_df,
        [
            ["target"],
            ["month"],
            ["hour"],
            ["actual_bin"],
            ["high_generation_flag"],
            ["near_zero_flag"],
            ["target", "month"],
            ["target", "actual_bin"],
            ["target", "hour"],
            ["target", "month", "actual_bin"],
        ],
    )
    ficr_summary.to_csv(results_dir / "ficr_failure_summary.csv", index=False, encoding="utf-8-sig")

    generation_summary = concat_summaries(
        long_df,
        [
            ["actual_bin"],
            ["target", "actual_bin"],
            ["month", "actual_bin"],
            ["hour", "actual_bin"],
            ["target", "month", "actual_bin"],
        ],
    )
    generation_summary.to_csv(results_dir / "generation_bin_scores.csv", index=False, encoding="utf-8-sig")

    q_models = [m for m in ["lgbm_l1_baseline", "catboost_quantile_055", "catboost_quantile_0575_depth6", "catboost_quantile_060_depth6"] if m in comparison_alpha]
    # Treat the current anchor as an additional comparison model.
    long_df["current_anchor_abs_error"] = long_df["abs_error"]
    long_df["current_anchor_norm_abs_error"] = long_df["normalized_abs_error"]
    long_df["current_anchor_ficr_pass"] = long_df["eligible_flag"] & (long_df["normalized_abs_error"] <= 0.08)
    q_models.append("current_anchor")
    winners = model_winners(long_df, q_models)
    q_summary_abs = winner_summary(winners, [["target"], ["actual_bin"], ["target", "actual_bin"], ["target", "month"], ["target", "hour"], ["target", "high_generation_flag"]], "abs_error_winner")
    q_summary_ficr = winner_summary(winners, [["target"], ["actual_bin"], ["target", "actual_bin"], ["target", "month"], ["target", "hour"], ["target", "high_generation_flag"]], "ficr_pass_winner")
    q_summary_abs.to_csv(results_dir / "quantile_win_loss_summary.csv", index=False, encoding="utf-8-sig")
    q_summary_ficr.to_csv(results_dir / "quantile_ficr_pass_winner_summary.csv", index=False, encoding="utf-8-sig")

    disagreement = disagreement_summary(
        long_df,
        [m for m in COMPARISON_MODELS if m in comparison_alpha],
        [["target"], ["actual_bin"], ["target", "actual_bin"], ["target", "month"], ["target", "hour"], ["target", "high_generation_flag"], ["target", "near_zero_flag"]],
    )
    disagreement.to_csv(results_dir / "model_disagreement_summary.csv", index=False, encoding="utf-8-sig")

    error_slice = ficr_summary.copy()
    error_slice["recommended_next_audit_route"] = np.select(
        [
            error_slice["near_zero_flag"].eq(True) if "near_zero_flag" in error_slice.columns else pd.Series(False, index=error_slice.index),
            error_slice["high_generation_flag"].eq(True) if "high_generation_flag" in error_slice.columns else pd.Series(False, index=error_slice.index),
            error_slice["underprediction_rate"] >= 0.60,
            error_slice["near_boundary_fail_rate"] >= 0.25,
        ],
        [
            "data quality / postprocessing audit",
            "feature sufficiency audit focused on high generation",
            "modeling-policy/postprocessing audit",
            "postprocessing audit",
        ],
        default="feature sufficiency audit",
    )
    error_slice.to_csv(results_dir / "error_slice_summary.csv", index=False, encoding="utf-8-sig")

    routing = route_from_results(bias_summary, ficr_summary, generation_summary, q_summary_abs, disagreement, check)
    routing += "\n\n## Row-target table\n\n"
    routing += f"- stored as: `{write_info['path']}`\n"
    routing += f"- format: `{write_info['format']}`\n"
    if write_info["fallback_reason"]:
        routing += f"- fallback reason: {write_info['fallback_reason']}\n"
    routing += f"- rows: {len(long_df)}\n"
    routing += f"- comparison models loaded: {sorted(comparison_alpha)}\n"
    (results_dir / "next_audit_routing.md").write_text(routing, encoding="utf-8")

    sanity = {
        "row_target_rows": int(len(long_df)),
        "expected_rows": int(len(valid) * len(TARGETS)),
        "missing_values_core": {c: int(long_df[c].isna().sum()) for c in ["actual", "anchor_pred_raw", "anchor_pred_alpha", "normalized_abs_error"]},
        "inf_values_core": {c: int(np.isinf(long_df[c].to_numpy(dtype=float)).sum()) for c in ["actual", "anchor_pred_raw", "anchor_pred_alpha", "normalized_abs_error"]},
        "result_files": sorted(path.name for path in results_dir.glob("*")),
    }
    (results_dir / "sanity_check.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")

    print(routing)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}


@dataclass(frozen=True)
class Window:
    name: str
    train_years: tuple[int, ...]
    recency_weights: dict[int, float] | None = None


WINDOWS = [
    Window("W3_2023_to_2024", (2023,), None),
    Window("W4_2022_2023_to_2024", (2022, 2023), None),
    Window("W5_2022_2023_recency_to_2024", (2022, 2023), {2022: 0.5, 2023: 1.0}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_labels(data_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(data_dir / "train" / "train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels["year"] = labels["kst_dtm"].dt.year
    labels["month"] = labels["kst_dtm"].dt.month
    labels["hour"] = labels["kst_dtm"].dt.hour
    return labels


def add_wind_physics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    pairs = []
    cols = set(out.columns)
    candidate_pairs = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind10"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "wind80"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "wind100"),
        ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "wind50max"),
        ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "wind50min"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "pblwind"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "wind850"),
        ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "wind700"),
        ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "wind500"),
    ]
    for u_col, v_col, name in candidate_pairs:
        if u_col in cols and v_col in cols:
            pairs.append((u_col, v_col, name))

    for u_col, v_col, name in pairs:
        u = pd.to_numeric(out[u_col], errors="coerce")
        v = pd.to_numeric(out[v_col], errors="coerce")
        speed = np.sqrt(u * u + v * v)
        out[f"{prefix}_{name}_speed"] = speed
        out[f"{prefix}_{name}_speed2"] = speed * speed
        out[f"{prefix}_{name}_speed3"] = speed * speed * speed
        angle = np.arctan2(v, u)
        out[f"{prefix}_{name}_dir_sin"] = np.sin(angle)
        out[f"{prefix}_{name}_dir_cos"] = np.cos(angle)
    return out


def aggregate_weather(path: Path, source: str, include_physics: bool) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    if include_physics:
        df = add_wind_physics(df, source)
    drop_cols = {"data_available_kst_dtm", "grid_id"}
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"forecast_kst_dtm"} | drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df.groupby("forecast_kst_dtm", sort=True)[numeric_cols].agg(["mean", "min", "max", "std"])
    grouped.columns = [f"{source}_{col}_{stat}" for col, stat in grouped.columns]
    return grouped.reset_index()


def build_feature_frame(data_dir: Path, feature_set: str) -> pd.DataFrame:
    include_physics = feature_set == "B2_wind_physics"
    ldaps = aggregate_weather(data_dir / "train" / "ldaps_train.csv", "ldaps", include_physics)
    gfs = aggregate_weather(data_dir / "train" / "gfs_train.csv", "gfs", include_physics)
    feat = ldaps.merge(gfs, on="forecast_kst_dtm", how="inner")
    feat["year"] = feat["forecast_kst_dtm"].dt.year
    feat["month"] = feat["forecast_kst_dtm"].dt.month
    feat["hour"] = feat["forecast_kst_dtm"].dt.hour
    feat["dayofyear"] = feat["forecast_kst_dtm"].dt.dayofyear
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12.0)
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24.0)
    feat["doy_sin"] = np.sin(2 * np.pi * feat["dayofyear"] / 366.0)
    feat["doy_cos"] = np.cos(2 * np.pi * feat["dayofyear"] / 366.0)
    return feat


def merge_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return features.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_label"))


def compute_ficr_proxy(abs_err_norm: pd.Series) -> float:
    incentive = np.select(
        [abs_err_norm <= 0.06, abs_err_norm <= 0.08],
        [1.0, 0.75],
        default=0.0,
    )
    return float(np.mean(incentive)) if len(incentive) else np.nan


def evaluate_predictions(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    timestamps: pd.Series,
    threshold: float = 0.10,
    include_monthly: bool = True,
) -> dict:
    group_rows = []
    nmaes = []
    ficrs = []
    for target in TARGETS:
        cap = CAPACITY[target]
        actual = y_true[target]
        pred = y_pred[target].clip(0.0, cap)
        mask = actual.notna() & (actual >= threshold * cap)
        if not mask.any():
            group_rows.append({"target": target, "nmae": np.nan, "ficr": np.nan, "eligible_hours": 0})
            continue
        abs_err_norm = (pred[mask] - actual[mask]).abs() / cap
        nmae = float(abs_err_norm.mean())
        ficr = compute_ficr_proxy(abs_err_norm)
        nmaes.append(nmae)
        ficrs.append(ficr)
        group_rows.append({"target": target, "nmae": nmae, "ficr": ficr, "eligible_hours": int(mask.sum())})

    avg_nmae = float(np.nanmean(nmaes))
    one_minus_nmae = 1.0 - avg_nmae
    ficr = float(np.nanmean(ficrs))
    score = 0.5 * one_minus_nmae + 0.5 * ficr

    monthly_rows = []
    if include_monthly:
        ts = pd.to_datetime(timestamps)
        for month in range(1, 13):
            month_mask = ts.dt.month == month
            if month_mask.any():
                m = evaluate_predictions(
                    y_true.loc[month_mask],
                    y_pred.loc[month_mask],
                    ts.loc[month_mask],
                    threshold,
                    include_monthly=False,
                )
                monthly_rows.append(
                    {
                        "month": month,
                        "score": m["score"],
                        "one_minus_nmae": m["one_minus_nmae"],
                        "ficr": m["ficr"],
                        "avg_nmae": m["avg_nmae"],
                        "eligible_hours": m["eligible_hours"],
                    }
                )

    return {
        "score": score,
        "one_minus_nmae": one_minus_nmae,
        "ficr": ficr,
        "avg_nmae": avg_nmae,
        "eligible_hours": int(sum(r["eligible_hours"] for r in group_rows)),
        "group_rows": group_rows,
        "monthly_rows": monthly_rows,
    }


def evaluate_high_generation(y_true: pd.DataFrame, y_pred: pd.DataFrame, timestamps: pd.Series) -> float:
    return evaluate_predictions(y_true, y_pred, timestamps, threshold=0.50)["score"]


def make_b0_predictions(train: pd.DataFrame, valid: pd.DataFrame, window: Window) -> pd.DataFrame:
    preds = pd.DataFrame(index=valid.index)
    for target in TARGETS:
        source = train[["month", "hour", "year", target]].dropna()
        if window.recency_weights:
            source = source.copy()
            source["_w"] = source["year"].map(window.recency_weights).fillna(1.0)
            table = (
                source.groupby(["month", "hour"])
                .apply(lambda g: np.average(g[target], weights=g["_w"]), include_groups=False)
                .rename("pred")
                .reset_index()
            )
        else:
            table = source.groupby(["month", "hour"], as_index=False)[target].mean().rename(columns={target: "pred"})
        fallback = float(source[target].mean())
        merged = valid[["month", "hour"]].merge(table, on=["month", "hour"], how="left")
        preds[target] = merged["pred"].fillna(fallback).to_numpy()
    return preds


def train_lgbm_predictions(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str], window: Window, random_state: int) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    preds = pd.DataFrame(index=valid.index)
    x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    for target in TARGETS:
        y = train[target]
        mask = y.notna()
        x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = y.loc[mask]
        sample_weight = None
        if window.recency_weights:
            sample_weight = train.loc[mask, "year"].map(window.recency_weights).fillna(1.0).to_numpy()
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
        model.fit(x_train, y_train, sample_weight=sample_weight)
        preds[target] = model.predict(x_valid)
    return preds


def summarize_experiment(exp_id: str, model_name: str, feature_set: str, window: Window, metrics: dict, high_score: float) -> dict:
    row = {
        "experiment_id": exp_id,
        "model": model_name,
        "feature_set": feature_set,
        "window": window.name,
        "score": metrics["score"],
        "one_minus_nmae": metrics["one_minus_nmae"],
        "ficr_proxy": metrics["ficr"],
        "avg_nmae": metrics["avg_nmae"],
        "eligible_hours": metrics["eligible_hours"],
        "worst_month_score": min(m["score"] for m in metrics["monthly_rows"]),
        "high_generation_score": high_score,
    }
    for g in metrics["group_rows"]:
        row[f"{g['target']}_nmae"] = g["nmae"]
        row[f"{g['target']}_ficr_proxy"] = g["ficr"]
        row[f"{g['target']}_eligible_hours"] = g["eligible_hours"]
    return row


def classify_windows(summary: pd.DataFrame, monthly: pd.DataFrame, group: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    focus = summary[summary["model"].isin(["B1_lgbm", "B2_lgbm"])].copy()
    by_window = focus.groupby("window", as_index=False).agg(
        score=("score", "mean"),
        avg_nmae=("avg_nmae", "mean"),
        ficr_proxy=("ficr_proxy", "mean"),
        worst_month_score=("worst_month_score", "mean"),
        high_generation_score=("high_generation_score", "mean"),
    )
    best = by_window.sort_values(["score", "avg_nmae"], ascending=[False, True]).iloc[0]
    w3 = by_window[by_window["window"] == "W3_2023_to_2024"].iloc[0]

    rows = []
    for _, row in by_window.iterrows():
        delta_score = row["score"] - w3["score"]
        delta_nmae = row["avg_nmae"] - w3["avg_nmae"]
        months_improved = np.nan
        groups_improved = np.nan
        if row["window"] != "W3_2023_to_2024":
            focus_monthly = monthly[monthly["experiment_id"].str.startswith(("B1_lgbm_", "B2_lgbm_"))].copy()
            focus_monthly["window"] = focus_monthly["experiment_id"].str.replace(r"^B[12]_lgbm_", "", regex=True)
            monthly_mean = focus_monthly.groupby(["window", "month"], as_index=False)["score"].mean()
            base_month = monthly_mean[monthly_mean["window"] == "W3_2023_to_2024"][["month", "score"]].rename(columns={"score": "base_score"})
            cur_month = monthly_mean[monthly_mean["window"] == row["window"]][["month", "score"]]
            month_delta = cur_month.merge(base_month, on="month", how="inner")
            months_improved = int((month_delta["score"] > month_delta["base_score"]).sum())

            focus_group = group[group["experiment_id"].str.startswith(("B1_lgbm_", "B2_lgbm_"))].copy()
            focus_group["window"] = focus_group["experiment_id"].str.replace(r"^B[12]_lgbm_", "", regex=True)
            group_mean = focus_group.groupby(["window", "target"], as_index=False)["nmae"].mean()
            base_group = group_mean[group_mean["window"] == "W3_2023_to_2024"][["target", "nmae"]].rename(columns={"nmae": "base_nmae"})
            cur_group = group_mean[group_mean["window"] == row["window"]][["target", "nmae"]]
            group_delta = cur_group.merge(base_group, on="target", how="inner")
            groups_improved = int((group_delta["nmae"] < group_delta["base_nmae"]).sum())
        verdict = "Diagnostic"
        reason = "kept for comparison"
        material_gain = delta_score >= 0.003 or delta_nmae <= -0.002
        stability_ok = (not pd.isna(months_improved)) and months_improved >= 8
        group_ok = (not pd.isna(groups_improved)) and groups_improved >= 2
        if row["window"] == best["window"] and material_gain and stability_ok and group_ok:
            verdict = "Adopt"
            reason = "best mean B1/B2 score and passes material/stability/group criteria"
        elif row["window"] == best["window"] and material_gain:
            verdict = "Support"
            reason = "best mean B1/B2 score, but does not pass all stability criteria"
        elif material_gain:
            verdict = "Support"
            reason = "materially improves over W3 on mean B1/B2 metrics"
        elif stability_ok and group_ok and delta_score > 0:
            verdict = "Support"
            reason = "stable positive alternative, but gain is below material threshold"
        elif delta_score < -0.003 and delta_nmae > 0.002:
            verdict = "Reject"
            reason = "materially worse than W3 on mean B1/B2 metrics"
        rows.append({
            **row.to_dict(),
            "delta_score_vs_W3": delta_score,
            "delta_nmae_vs_W3": delta_nmae,
            "months_improved_vs_W3": months_improved,
            "groups_improved_vs_W3": groups_improved,
            "verdict": verdict,
            "reason": reason,
        })

    verdicts = pd.DataFrame(rows)
    if (verdicts["verdict"] == "Adopt").any():
        selected = verdicts[verdicts["verdict"] == "Adopt"].iloc[0]["window"]
        prefix = "Adopted"
    else:
        selected = verdicts.sort_values(["score", "avg_nmae"], ascending=[False, True]).iloc[0]["window"]
        prefix = "Provisional"
    if selected == "W4_2022_2023_to_2024":
        policy = f"{prefix} T1/T6 candidate: full-history is supported for group 1/2, but needs public-LB calibration; group 3 remains recent-window because 2022 labels are absent."
    elif selected == "W5_2022_2023_recency_to_2024":
        policy = f"{prefix} T4/T6 candidate: recency-weighted full history is preferred; keep group-specific handling for group 3."
    else:
        policy = f"{prefix} T2/T6 candidate: recent-window policy is preferred over adding 2022 uniformly."
    return verdicts, policy


def write_conclusion(out_dir: Path, summary: pd.DataFrame, verdicts: pd.DataFrame, policy: str, package_versions: dict) -> None:
    best_rows = summary.sort_values("score", ascending=False).head(10)
    lines = []
    lines.append("# CV Protocol Results\n")
    lines.append("## Environment\n")
    lines.append("```json")
    lines.append(json.dumps(package_versions, indent=2, ensure_ascii=False))
    lines.append("```\n")
    lines.append("## Top Experiments\n")
    lines.append(best_rows[["experiment_id", "model", "feature_set", "window", "score", "avg_nmae", "ficr_proxy", "worst_month_score", "high_generation_score"]].to_markdown(index=False))
    lines.append("\n## Window Verdicts\n")
    lines.append(verdicts[["window", "score", "avg_nmae", "ficr_proxy", "worst_month_score", "delta_score_vs_W3", "delta_nmae_vs_W3", "months_improved_vs_W3", "groups_improved_vs_W3", "verdict", "reason"]].to_markdown(index=False))
    lines.append("\n## Recommended Train Policy\n")
    lines.append(policy)
    lines.append("\n## Caveats\n")
    lines.append("- The official page confirms the group FICR structure, but the exact per-hour settlement table remains in a DACON code-download attachment that was not accessible in this run.")
    lines.append("- FICR therefore uses a 6%/8% threshold proxy until the official code attachment is obtained.")
    lines.append("- Public/private LB calibration still needs actual submissions.")
    lines.append("- Group 3 has only 2023-2024 usable labels, so its train-window evidence is weaker.")
    (out_dir / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.data_dir)
    valid_labels = labels[labels["year"] == 2024].copy().reset_index(drop=True)
    y_valid = valid_labels[TARGETS]
    timestamps_valid = valid_labels["kst_dtm"]

    all_summary = []
    all_monthly = []
    all_group = []

    # B0 does not need weather features.
    for window in WINDOWS:
        train_labels = labels[labels["year"].isin(window.train_years)].copy()
        preds = make_b0_predictions(train_labels, valid_labels, window)
        metrics = evaluate_predictions(y_valid, preds, timestamps_valid)
        high_score = evaluate_high_generation(y_valid, preds, timestamps_valid)
        exp_id = f"B0_{window.name}"
        all_summary.append(summarize_experiment(exp_id, "B0_month_hour", "B0_time_mean", window, metrics, high_score))
        for row in metrics["monthly_rows"]:
            all_monthly.append({"experiment_id": exp_id, **row})
        for row in metrics["group_rows"]:
            all_group.append({"experiment_id": exp_id, **row})

    for feature_set, model_name in [("B1_aggregate", "B1_lgbm"), ("B2_wind_physics", "B2_lgbm")]:
        features = build_feature_frame(args.data_dir, feature_set)
        data = merge_features_labels(features, labels)
        valid = data[data["year"] == 2024].copy().reset_index(drop=True)
        feature_cols = [c for c in features.columns if c != "forecast_kst_dtm"]
        feature_cols = [c for c in feature_cols if c in valid.columns]
        for window in WINDOWS:
            train = data[data["year"].isin(window.train_years)].copy().reset_index(drop=True)
            preds = train_lgbm_predictions(train, valid, feature_cols, window, args.random_state)
            y_true = valid[TARGETS]
            timestamps = valid["forecast_kst_dtm"]
            metrics = evaluate_predictions(y_true, preds, timestamps)
            high_score = evaluate_high_generation(y_true, preds, timestamps)
            exp_id = f"{model_name}_{window.name}"
            all_summary.append(summarize_experiment(exp_id, model_name, feature_set, window, metrics, high_score))
            for row in metrics["monthly_rows"]:
                all_monthly.append({"experiment_id": exp_id, **row})
            for row in metrics["group_rows"]:
                all_group.append({"experiment_id": exp_id, **row})

    summary = pd.DataFrame(all_summary)
    monthly = pd.DataFrame(all_monthly)
    group = pd.DataFrame(all_group)
    verdicts, policy = classify_windows(summary, monthly, group)

    summary.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.out_dir / "monthly_scores.csv", index=False, encoding="utf-8-sig")
    group.to_csv(args.out_dir / "group_scores.csv", index=False, encoding="utf-8-sig")
    verdicts.to_csv(args.out_dir / "window_verdicts.csv", index=False, encoding="utf-8-sig")

    import lightgbm
    import sklearn

    package_versions = {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }
    write_conclusion(args.out_dir, summary, verdicts, policy, package_versions)

    print((args.out_dir / "conclusion.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

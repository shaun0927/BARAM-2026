from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
SOURCE = ROOT / "experiments" / "comprehensive_eda_20260809" / "results"
DEEP = SOURCE / "deep_dive"
FINAL = SOURCE / "final_closure"
CV_DIR = ROOT / "experiments" / "cv_protocol"
OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"

sys.path.insert(0, str(CV_DIR))
from run_cv_protocol import CAPACITY, TARGETS, evaluate_predictions  # noqa: E402


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)


def wind_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id"]].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out[f"{source}_lead_hour"] = (
        out["forecast_kst_dtm"] - out["data_available_kst_dtm"]
    ).dt.total_seconds() / 3600.0

    pairs = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind10"),
        ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "wind50max"),
        ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "wind50min"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "wind80"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "wind100"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "wind850"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "pblwind"),
    ]
    for u_col, v_col, name in pairs:
        if u_col not in df.columns or v_col not in df.columns:
            continue
        u = pd.to_numeric(df[u_col], errors="coerce")
        v = pd.to_numeric(df[v_col], errors="coerce")
        speed = np.sqrt(u * u + v * v)
        direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
        out[f"{source}_{name}_speed"] = speed
        out[f"{source}_{name}_speed2"] = speed * speed
        out[f"{source}_{name}_u"] = u
        out[f"{source}_{name}_v"] = v
        out[f"{source}_{name}_dir_sin"] = np.sin(np.radians(direction))
        out[f"{source}_{name}_dir_cos"] = np.cos(np.radians(direction))
        out[f"{source}_{name}_dir_deg"] = direction
    return out


def aggregate_weather(split: str) -> pd.DataFrame:
    suffix = "test" if split == "test" else "train"
    frames = []
    for source in ["ldaps", "gfs"]:
        raw = read_csv(DATA / split / f"{source}_{suffix}.csv")
        wind = wind_features(raw, source)
        drop = {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id"}
        value_cols = [c for c in wind.columns if c not in drop]
        grouped = wind.groupby("forecast_kst_dtm", sort=True)[value_cols].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        frames.append(grouped.reset_index())
    out = frames[0].merge(frames[1], on="forecast_kst_dtm", how="inner")
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour
    out["dayofyear"] = out["forecast_kst_dtm"].dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 366.0)
    return out


def read_taxonomy() -> pd.DataFrame:
    tax = read_csv(DEEP / "scada_label_residual_taxonomy_rows.csv")
    tax["kst_dtm"] = pd.to_datetime(tax["kst_dtm"])
    tax["year"] = tax["kst_dtm"].dt.year
    tax["month"] = tax["kst_dtm"].dt.month
    tax["hour"] = tax["kst_dtm"].dt.hour
    tax["is_non_tight"] = ~tax["taxonomy"].isin(["tight_reconstruction", "label_missing"])
    return tax


def feature_cols(df: pd.DataFrame) -> list[str]:
    forbidden = {"forecast_kst_dtm"}
    cols = [c for c in df.columns if c not in forbidden and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def train_target_classifier(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str]) -> tuple[RandomForestClassifier, dict]:
    cols = [c for c in cols if c in train.columns and c in valid.columns]
    x_train = train[cols].replace([np.inf, -np.inf], np.nan).fillna(train[cols].median(numeric_only=True))
    x_valid = valid[cols].replace([np.inf, -np.inf], np.nan).fillna(train[cols].median(numeric_only=True))
    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_train, train["binary_label"])
    prob = clf.predict_proba(x_valid)[:, 1]
    pred = prob >= 0.5
    metrics = {
        "rows_train": int(len(train)),
        "rows_valid": int(len(valid)),
        "positive_rate_train": float(train["binary_label"].mean()),
        "positive_rate_valid": float(valid["binary_label"].mean()),
        "balanced_accuracy": float(balanced_accuracy_score(valid["binary_label"], pred)),
        "f1": float(f1_score(valid["binary_label"], pred, zero_division=0)),
        "auc": float(roc_auc_score(valid["binary_label"], prob)) if valid["binary_label"].nunique() == 2 else np.nan,
        "mean_prob_valid": float(prob.mean()),
    }
    return clf, metrics


def separability_eda(features: pd.DataFrame, test_features: pd.DataFrame, tax: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = feature_cols(features)
    rows = []
    exposure_rows = []
    importance_rows = []
    for target in TARGETS:
        t = tax[tax["target"].eq(target)].copy()
        data = features.merge(t, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_tax"))
        data = data[~data["taxonomy"].eq("label_missing")].copy()
        data["binary_label"] = data["is_non_tight"].astype(int)
        train = data[data["year_tax"].isin([2022, 2023])].copy()
        valid = data[data["year_tax"].eq(2024)].copy()
        if target == "kpx_group_3":
            train = data[data["year_tax"].eq(2023)].copy()
        clf, metrics = train_target_classifier(train, valid, cols)
        metrics["target"] = target
        rows.append(metrics)

        x_valid = valid[cols].replace([np.inf, -np.inf], np.nan).fillna(train[cols].median(numeric_only=True))
        perm = permutation_importance(
            clf,
            x_valid,
            valid["binary_label"],
            scoring="balanced_accuracy",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        top_idx = np.argsort(perm.importances_mean)[::-1][:12]
        for idx in top_idx:
            importance_rows.append(
                {
                    "target": target,
                    "feature": cols[idx],
                    "importance_mean": float(perm.importances_mean[idx]),
                    "importance_std": float(perm.importances_std[idx]),
                }
            )

        x_test = test_features[cols].replace([np.inf, -np.inf], np.nan).fillna(train[cols].median(numeric_only=True))
        test_prob = clf.predict_proba(x_test)[:, 1]
        tmp = test_features[["forecast_kst_dtm", "month"]].copy()
        tmp["target"] = target
        tmp["non_tight_probability"] = test_prob
        exposure_rows.append(tmp)

    sep = pd.DataFrame(rows)
    exposure = pd.concat(exposure_rows, ignore_index=True)
    importance = pd.DataFrame(importance_rows)
    sep.to_csv(OUT / "nwp_only_scada_regime_separability.csv", index=False, encoding="utf-8-sig")
    exposure.to_csv(OUT / "test2025_scada_regime_exposure.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(OUT / "nwp_only_scada_regime_feature_importance.csv", index=False, encoding="utf-8-sig")
    return sep, exposure, importance


def taxonomy_weather_summary(features: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    focus_cols = [
        "ldaps_wind50max_speed_mean",
        "ldaps_wind50max_speed_std",
        "ldaps_wind50max_dir_deg_mean",
        "gfs_wind850_speed_mean",
        "gfs_wind100_speed_mean",
        "ldaps_lead_hour_mean",
        "month",
        "hour",
    ]
    rows = []
    for target in TARGETS:
        data = features.merge(
            tax[tax["target"].eq(target)],
            left_on="forecast_kst_dtm",
            right_on="kst_dtm",
            how="inner",
            suffixes=("", "_tax"),
        )
        for taxonomy, g in data.groupby("taxonomy"):
            if taxonomy == "label_missing":
                continue
            row = {"target": target, "taxonomy": taxonomy, "rows": int(len(g))}
            for col in focus_cols:
                if col in g.columns:
                    row[f"{col}_mean"] = float(g[col].mean())
                    row[f"{col}_std"] = float(g[col].std())
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "taxonomy_nwp_weather_summary.csv", index=False, encoding="utf-8-sig")
    return out


def event_level_eda(features: pd.DataFrame) -> pd.DataFrame:
    events = read_csv(FINAL / "scada_residual_event_casebook.csv")
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])
    rows = []
    for event in events.itertuples(index=False):
        m = (features["forecast_kst_dtm"] >= event.start) & (features["forecast_kst_dtm"] <= event.end)
        before = (features["forecast_kst_dtm"] >= event.start - pd.Timedelta(hours=24)) & (
            features["forecast_kst_dtm"] < event.start
        )
        after = (features["forecast_kst_dtm"] > event.end) & (
            features["forecast_kst_dtm"] <= event.end + pd.Timedelta(hours=24)
        )
        f_event = features.loc[m]
        f_before = features.loc[before]
        f_after = features.loc[after]
        rows.append(
            {
                "target": event.target,
                "taxonomy": event.taxonomy,
                "start": event.start,
                "end": event.end,
                "duration_hours": int(event.duration_hours),
                "mean_abs_residual_ratio": float(event.mean_abs_residual_ratio),
                "event_ldaps_wind50_mean": float(f_event["ldaps_wind50max_speed_mean"].mean()),
                "before_ldaps_wind50_mean": float(f_before["ldaps_wind50max_speed_mean"].mean()),
                "after_ldaps_wind50_mean": float(f_after["ldaps_wind50max_speed_mean"].mean()),
                "event_gfs_wind850_mean": float(f_event["gfs_wind850_speed_mean"].mean()),
                "before_gfs_wind850_mean": float(f_before["gfs_wind850_speed_mean"].mean()),
                "after_gfs_wind850_mean": float(f_after["gfs_wind850_speed_mean"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["ldaps_event_minus_context"] = out["event_ldaps_wind50_mean"] - 0.5 * (
        out["before_ldaps_wind50_mean"] + out["after_ldaps_wind50_mean"]
    )
    out["gfs_event_minus_context"] = out["event_gfs_wind850_mean"] - 0.5 * (
        out["before_gfs_wind850_mean"] + out["after_gfs_wind850_mean"]
    )
    out.to_csv(OUT / "event_level_nwp_context.csv", index=False, encoding="utf-8-sig")
    return out


def nwp_scada_bias_eda(tax: pd.DataFrame) -> pd.DataFrame:
    bias = read_csv(DEEP / "nwp_scada_wind_bias_rows.csv")
    bias["kst_dtm"] = pd.to_datetime(bias["kst_dtm"])
    rows = []
    for target in TARGETS:
        joined = bias[bias["target"].eq(target)].merge(
            tax[tax["target"].eq(target)][["kst_dtm", "taxonomy", "abs_residual_ratio"]],
            on="kst_dtm",
            how="inner",
        )
        if "speed_bias_nwp_minus_scada" in joined.columns:
            bias_col = "speed_bias_nwp_minus_scada"
        elif "ldaps_speed_bias" in joined.columns:
            bias_col = "ldaps_speed_bias"
        else:
            candidates = [c for c in joined.columns if "bias" in c and "ldaps" in c]
            bias_col = candidates[0]
        joined["bias_bin"] = pd.cut(
            joined[bias_col],
            bins=[-np.inf, -4, -2, -1, 1, 2, 4, np.inf],
            labels=["<=-4", "-4..-2", "-2..-1", "-1..1", "1..2", "2..4", ">=4"],
        )
        for (taxonomy, bias_bin), g in joined.groupby(["taxonomy", "bias_bin"], observed=True):
            rows.append(
                {
                    "target": target,
                    "taxonomy": taxonomy,
                    "bias_bin": str(bias_bin),
                    "rows": int(len(g)),
                    "mean_abs_residual_ratio": float(g["abs_residual_ratio"].mean()),
                    "mean_ldaps_speed_bias": float(g[bias_col].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "nwp_scada_bias_by_taxonomy.csv", index=False, encoding="utf-8-sig")
    return out


def power_curve_eda(features: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        joined = features.merge(
            tax[tax["target"].eq(target)],
            left_on="forecast_kst_dtm",
            right_on="kst_dtm",
            how="inner",
            suffixes=("", "_tax"),
        )
        joined["nwp_wind_bin"] = pd.qcut(joined["ldaps_wind50max_speed_mean"], q=12, duplicates="drop")
        for (taxonomy, wind_bin), g in joined.groupby(["taxonomy", "nwp_wind_bin"], observed=True):
            if taxonomy == "label_missing":
                continue
            rows.append(
                {
                    "target": target,
                    "taxonomy": taxonomy,
                    "nwp_wind_bin": str(wind_bin),
                    "wind_center": float(g["ldaps_wind50max_speed_mean"].mean()),
                    "rows": int(len(g)),
                    "mean_label_ratio": float(g["label_ratio"].mean()),
                    "mean_scada_ratio": float(g["scada_ratio"].mean()),
                    "mean_residual_ratio": float(g["scada_minus_label_ratio"].mean()),
                    "capacity": cap,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "power_curve_taxonomy_by_nwp_wind.csv", index=False, encoding="utf-8-sig")
    return out


def ficr_boundary_eda(tax: pd.DataFrame) -> pd.DataFrame:
    pred_path = (
        ROOT
        / "experiments"
        / "autoresearch_harness"
        / "results"
        / "phase98_source_sister_oof_stack"
        / "diagnostic_predictions"
        / "phase98_w4_phase60_lineage.csv"
    )
    if not pred_path.exists():
        pred_path = ROOT / "experiments" / "local_best_anchor_score_risk_transfer_20260809" / "results" / "best_w4_valid_predictions.csv"
    pred = read_csv(pred_path)
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    labels = read_csv(DATA / "train" / "train_labels.csv")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    valid = labels[labels["kst_dtm"].dt.year.eq(2024)].copy()
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        joined = (
            valid[["kst_dtm", target]]
            .merge(pred[["forecast_kst_dtm", target]], left_on="kst_dtm", right_on="forecast_kst_dtm", how="inner", suffixes=("_actual", "_pred"))
            .merge(tax[tax["target"].eq(target)][["kst_dtm", "taxonomy"]], on="kst_dtm", how="left")
        )
        actual = joined[f"{target}_actual"]
        predv = joined[f"{target}_pred"].clip(0, cap)
        eligible = actual >= 0.10 * cap
        abs_err_norm = (predv - actual).abs() / cap
        joined["ficr_pass"] = eligible & (abs_err_norm <= 0.08)
        joined["ficr_gold"] = eligible & (abs_err_norm <= 0.06)
        joined["near_10_actual"] = actual.between(0.08 * cap, 0.12 * cap)
        joined["near_16_pred"] = predv.between(0.14 * cap, 0.18 * cap)
        joined["abs_err_norm"] = abs_err_norm
        for taxonomy, g in joined.groupby("taxonomy"):
            if taxonomy == "label_missing":
                continue
            rows.append(
                {
                    "target": target,
                    "taxonomy": taxonomy,
                    "rows": int(len(g)),
                    "eligible_rows": int((g[f"{target}_actual"] >= 0.10 * cap).sum()),
                    "ficr_pass_rate": float(g["ficr_pass"].mean()),
                    "ficr_gold_rate": float(g["ficr_gold"].mean()),
                    "near_10_actual_rows": int(g["near_10_actual"].sum()),
                    "near_16_pred_rows": int(g["near_16_pred"].sum()),
                    "mean_abs_err_norm": float(g["abs_err_norm"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ficr_boundary_by_scada_taxonomy.csv", index=False, encoding="utf-8-sig")
    return out


def plot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str, path: Path, title: str, fmt: str = ".2f") -> None:
    pivot = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def make_figures(
    sep: pd.DataFrame,
    exposure: pd.DataFrame,
    importance: pd.DataFrame,
    weather: pd.DataFrame,
    events: pd.DataFrame,
    bias: pd.DataFrame,
    power: pd.DataFrame,
    ficr: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    metrics = ["balanced_accuracy", "auc", "f1"]
    for ax, metric in zip(axes, metrics):
        ax.bar(sep["target"], sep[metric], color=["#4c78a8", "#f58518", "#54a24b"])
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("NWP-only detectability of non-tight SCADA-label regimes")
    fig.tight_layout()
    fig.savefig(FIG / "01_nwp_only_regime_detectability.png", dpi=170)
    plt.close(fig)

    exp_month = exposure.groupby(["target", "month"], as_index=False)["non_tight_probability"].mean()
    plot_heatmap(
        exp_month,
        "target",
        "month",
        "non_tight_probability",
        FIG / "02_test2025_predicted_scada_regime_exposure.png",
        "2025 exposure: predicted probability of non-tight SCADA-label regime",
    )

    top_imp = importance.sort_values(["target", "importance_mean"], ascending=[True, False]).groupby("target").head(8)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, target in zip(axes, TARGETS):
        g = top_imp[top_imp["target"].eq(target)].sort_values("importance_mean")
        ax.barh(g["feature"], g["importance_mean"], color="#4c78a8")
        ax.set_title(target)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Permutation importance for SCADA-regime detectability")
    fig.tight_layout()
    fig.savefig(FIG / "03_nwp_only_regime_feature_importance.png", dpi=170)
    plt.close(fig)

    weather_focus = weather[weather["taxonomy"].isin(["tight_reconstruction", "scada_zero_label_positive", "scada_under_label", "low_online_label_positive"])].copy()
    plot_heatmap(
        weather_focus,
        "taxonomy",
        "target",
        "ldaps_wind50max_speed_mean_mean",
        FIG / "04_taxonomy_nwp_wind_surface.png",
        "LDAPS wind50max mean by SCADA residual taxonomy",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    order = events.sort_values("duration_hours", ascending=False).head(25)
    colors = {"scada_zero_label_positive": "#e45756", "scada_under_label": "#f58518", "low_online_label_positive": "#72b7b2", "moderate_residual": "#4c78a8"}
    ax.barh(
        [f"{r.target}|{r.taxonomy}" for r in order.itertuples()],
        order["duration_hours"],
        color=[colors.get(t, "#999999") for t in order["taxonomy"]],
    )
    ax.invert_yaxis()
    ax.set_xlabel("duration hours")
    ax.set_title("Event-level residual regimes: longest contiguous non-tight intervals")
    fig.tight_layout()
    fig.savefig(FIG / "05_event_level_residual_duration.png", dpi=170)
    plt.close(fig)

    plot_heatmap(
        bias[bias["taxonomy"].isin(["tight_reconstruction", "scada_zero_label_positive", "scada_under_label", "low_online_label_positive"])],
        "taxonomy",
        "bias_bin",
        "mean_abs_residual_ratio",
        FIG / "06_nwp_scada_bias_taxonomy_heatmap.png",
        "SCADA residual size by LDAPS-vs-SCADA wind bias bin",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        g = power[power["target"].eq(target)]
        for taxonomy in ["tight_reconstruction", "scada_under_label", "scada_zero_label_positive", "low_online_label_positive"]:
            h = g[g["taxonomy"].eq(taxonomy)].sort_values("wind_center")
            if len(h):
                ax.plot(h["wind_center"], h["mean_label_ratio"], marker="o", linewidth=1.3, label=taxonomy)
        ax.set_title(target)
        ax.set_xlabel("LDAPS wind50max mean")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("label / capacity")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Power-curve view: taxonomy changes label response at the same NWP wind")
    fig.tight_layout()
    fig.savefig(FIG / "07_power_curve_taxonomy_by_nwp_wind.png", dpi=170)
    plt.close(fig)

    plot_heatmap(
        ficr[ficr["taxonomy"].isin(["tight_reconstruction", "scada_zero_label_positive", "scada_under_label", "low_online_label_positive", "moderate_residual"])],
        "taxonomy",
        "target",
        "ficr_pass_rate",
        FIG / "08_ficr_boundary_by_scada_taxonomy.png",
        "W4 anchor FiCR pass rate by SCADA taxonomy",
    )


def table_md(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    view = df[cols].copy()
    if n is not None:
        view = view.head(n)
    return view.to_markdown(index=False)


def write_report(
    sep: pd.DataFrame,
    exposure: pd.DataFrame,
    importance: pd.DataFrame,
    weather: pd.DataFrame,
    events: pd.DataFrame,
    bias: pd.DataFrame,
    power: pd.DataFrame,
    ficr: pd.DataFrame,
) -> None:
    exp_month = exposure.groupby(["target", "month"], as_index=False)["non_tight_probability"].mean()
    exp_top = exp_month.sort_values("non_tight_probability", ascending=False).head(12)
    event_top = events.sort_values("duration_hours", ascending=False).head(10)
    ficr_focus = ficr.sort_values("ficr_pass_rate").head(12)
    imp_top = importance.sort_values(["target", "importance_mean"], ascending=[True, False]).groupby("target").head(5)
    non_tight_score = sep[["target", "balanced_accuracy", "auc", "f1", "positive_rate_valid"]].copy()

    lines = []
    lines.append("# SCADA Residual Deployability EDA\n")
    lines.append("This EDA tests whether the SCADA-to-label lessons from issue #31 are deployable from test-time available NWP/time features.")
    lines.append("SCADA is used only as a teacher label for train-time taxonomy; the classifier and 2025 exposure use LDAPS/GFS/time features only.\n")
    lines.append("## Figures\n")
    for i, name in enumerate(
        [
            "01_nwp_only_regime_detectability.png",
            "02_test2025_predicted_scada_regime_exposure.png",
            "03_nwp_only_regime_feature_importance.png",
            "04_taxonomy_nwp_wind_surface.png",
            "05_event_level_residual_duration.png",
            "06_nwp_scada_bias_taxonomy_heatmap.png",
            "07_power_curve_taxonomy_by_nwp_wind.png",
            "08_ficr_boundary_by_scada_taxonomy.png",
        ],
        start=1,
    ):
        lines.append(f"{i}. ![{name}](figures/{name})")
    lines.append("\n## 1. NWP-only Detectability\n")
    lines.append(table_md(non_tight_score, ["target", "balanced_accuracy", "auc", "f1", "positive_rate_valid"]))
    lines.append(
        "\nInterpretation: detectability is real but uneven. Group2/group3 show stronger binary separation than group1; group3's headline score is high, but it rests on 2023-only training because 2022 labels are absent. This supports using SCADA taxonomy as a teacher, but not as a direct submission rule."
    )
    lines.append("\nValidity discussion: the split is chronological, training on 2022-2023 and validating on 2024. For group3, training is 2023 only because 2022 labels are absent. The target is binary non-tight vs tight reconstruction to avoid overclaiming fine-grained causal labels.")
    lines.append("\n\nTop NWP-only features by permutation importance:\n")
    lines.append(table_md(imp_top, ["target", "feature", "importance_mean", "importance_std"]))

    lines.append("\n## 2. 2025 Exposure\n")
    lines.append(table_md(exp_top, ["target", "month", "non_tight_probability"], 12))
    lines.append(
        "\nInterpretation: the deployable risk is month- and group-concentrated. A candidate that adjusts SCADA-derived residual regimes should carry this exposure surface; a 2024-only gain is not enough if the target months are over- or under-exposed in 2025."
    )
    lines.append("\nValidity discussion: these are classifier probabilities, not observed 2025 SCADA states. They are suitable for submission-risk triage, not for proving public score.")

    lines.append("\n## 3. Taxonomy Weather Surface\n")
    weather_view = weather[weather["taxonomy"].isin(["tight_reconstruction", "scada_zero_label_positive", "scada_under_label", "low_online_label_positive"])]
    lines.append(table_md(weather_view.sort_values(["target", "taxonomy"]), ["target", "taxonomy", "rows", "ldaps_wind50max_speed_mean_mean", "gfs_wind850_speed_mean_mean"], 16))
    lines.append(
        "\nInterpretation: SCADA residual regimes occupy different NWP wind surfaces, but overlap remains substantial. This justifies gated/probabilistic actions rather than hard taxonomy replacement."
    )
    lines.append("\nValidity discussion: weather summaries are descriptive and do not prove separability by themselves. They are paired with the chronological classifier above.")

    lines.append("\n## 4. Event-level Structure\n")
    lines.append(table_md(event_top, ["target", "taxonomy", "start", "end", "duration_hours", "mean_abs_residual_ratio", "ldaps_event_minus_context", "gfs_event_minus_context"], 10))
    lines.append(
        "\nInterpretation: residual regimes include long contiguous events, so they are not just independent row noise. Some events show clear NWP context shifts; others do not, which is exactly why any deployable correction must be guarded."
    )
    lines.append("\nValidity discussion: without external operation logs, event labels remain observable signatures, not causal outage/curtailment proof.")

    lines.append("\n## 5. NWP-to-SCADA Bias\n")
    bias_view = bias.sort_values("mean_abs_residual_ratio", ascending=False).head(15)
    lines.append(table_md(bias_view, ["target", "taxonomy", "bias_bin", "rows", "mean_abs_residual_ratio", "mean_ldaps_speed_bias"], 15))
    lines.append(
        "\nInterpretation: large SCADA-label residuals concentrate in specific NWP-vs-SCADA wind-bias bins. This is the strongest route for converting SCADA teacher knowledge into test-time features: estimate NWP measurement bias, then act only where the score direction is stable."
    )
    lines.append("\nValidity discussion: the bias itself is measured with train-only SCADA, so test deployment must use NWP-only proxies for the bias regime.")

    lines.append("\n## 6. Power-curve Regime\n")
    pc_view = power[(power["rows"] >= 20) & power["taxonomy"].isin(["tight_reconstruction", "scada_zero_label_positive", "scada_under_label", "low_online_label_positive"])].sort_values(["target", "wind_center"]).head(20)
    lines.append(table_md(pc_view, ["target", "taxonomy", "wind_center", "rows", "mean_label_ratio", "mean_scada_ratio", "mean_residual_ratio"], 20))
    lines.append(
        "\nInterpretation: at comparable NWP wind speeds, taxonomy changes the label response. That means a single smooth NWP power curve is structurally insufficient for all regimes."
    )
    lines.append("\nValidity discussion: qcut bins make the curve stable enough for EDA, but the figure should be read as regime diagnosis, not a calibrated postprocessor.")

    lines.append("\n## 7. FiCR Boundary\n")
    lines.append(table_md(ficr_focus, ["target", "taxonomy", "rows", "eligible_rows", "ficr_pass_rate", "ficr_gold_rate", "near_10_actual_rows", "near_16_pred_rows", "mean_abs_err_norm"], 12))
    lines.append(
        "\nInterpretation: the score-relevant question is not only whether a taxonomy has large MAE. It is whether it changes FiCR pass/gold rates near eligible rows and prediction floors. This keeps the SCADA lesson connected to the actual competition score."
    )
    lines.append("\nValidity discussion: this uses the W4 diagnostic anchor, so it is a local score surface. It is valid for deciding what to test next, not enough by itself to authorize public submission.")

    lines.append("\n## Conclusion\n")
    lines.append(
        "The additional EDA confirms the central lesson: SCADA residual regimes are partially deployable, not directly deployable. The evidence supports NWP-only gated residual/bias models, but group3 needs special caution despite its high 2024 binary detectability because its training evidence is 2023-only and its UNISON residual structure differs from VESTAS group1/2. Every score action still needs W3/cross-year confirmation."
    )
    lines.append("\nWhat must be proven before a scoring experiment is promoted:")
    lines.append("- Detectability: the SCADA residual regime can be inferred from NWP/time features on a chronological split.")
    lines.append("- Directionality: the required prediction move is stable within that inferred regime.")
    lines.append("- Score payoff: FiCR gain exceeds NMAE cost on W3/W4 or another independent split.")

    (OUT / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    tax = read_taxonomy()
    features = aggregate_weather("train")
    test_features = aggregate_weather("test")

    sep, exposure, importance = separability_eda(features, test_features, tax)
    weather = taxonomy_weather_summary(features, tax)
    events = event_level_eda(features)
    bias = nwp_scada_bias_eda(tax)
    power = power_curve_eda(features, tax)
    ficr = ficr_boundary_eda(tax)
    make_figures(sep, exposure, importance, weather, events, bias, power, ficr)
    write_report(sep, exposure, importance, weather, events, bias, power, ficr)

    manifest = {
        "experiment": "scada_residual_deployability_eda_20260810",
        "figures": sorted(p.name for p in FIG.glob("*.png")),
        "tables": sorted(p.name for p in OUT.glob("*.csv")),
        "report": "conclusion.md",
        "method_boundary": "SCADA is used only to define train-time teacher taxonomy; detectability and 2025 exposure use NWP/time features.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print((OUT / "conclusion.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

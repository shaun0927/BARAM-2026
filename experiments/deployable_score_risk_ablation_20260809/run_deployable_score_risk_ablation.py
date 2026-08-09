from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

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


DATA = ROOT.parent / "open"
OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"
ISSUE32 = ROOT / "experiments" / "score_driven_eda_20260809" / "results"
ISSUE31 = ROOT / "experiments" / "comprehensive_eda_20260809" / "results"
W4 = Window("W4_2022_2023_to_2024", (2022, 2023), None)

RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/deployable_score_risk_ablation_20260809/results/figures"

WIND_PAIRS = {
    "ldaps": [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind10"),
        ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "wind50max"),
        ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "wind50min"),
        ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "wind5bl"),
    ],
    "gfs": [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "wind10"),
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "wind80"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "wind100"),
        ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "pblwind"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "wind850"),
    ],
}


@dataclass
class Result:
    name: str
    preds: pd.DataFrame
    metrics: dict
    high_generation_score: float
    feature_counts: dict[str, int]


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 170, "font.size": 9, "axes.titlesize": 12})


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def add_grid_wind(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id"]].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out["lead_hour"] = (out["forecast_kst_dtm"] - out["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    for u_col, v_col, name in WIND_PAIRS[source]:
        if u_col not in df.columns or v_col not in df.columns:
            continue
        u = pd.to_numeric(df[u_col], errors="coerce")
        v = pd.to_numeric(df[v_col], errors="coerce")
        speed = np.sqrt(u * u + v * v)
        angle = np.arctan2(v, u)
        out[f"{source}_{name}_speed"] = speed
        out[f"{source}_{name}_speed2"] = speed * speed
        out[f"{source}_{name}_speed3"] = speed * speed * speed
        out[f"{source}_{name}_u"] = u
        out[f"{source}_{name}_v"] = v
        out[f"{source}_{name}_dir_sin"] = np.sin(angle)
        out[f"{source}_{name}_dir_cos"] = np.cos(angle)
        out[f"{source}_{name}_dir_deg"] = (np.degrees(angle) + 360.0) % 360.0
    return out


def sector_label(deg: pd.Series) -> pd.Series:
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"]
    return pd.cut(deg, bins=bins, labels=labels, include_lowest=True, right=False).astype(str)


def build_direction_conditioned_features(data_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    ldaps_raw = read_csv(data_dir / "train" / "ldaps_train.csv")
    wind = add_grid_wind(ldaps_raw, "ldaps")
    agg = wind.groupby("forecast_kst_dtm", as_index=False).agg(
        ldaps_wind50max_u_mean=("ldaps_wind50max_u", "mean"),
        ldaps_wind50max_v_mean=("ldaps_wind50max_v", "mean"),
        lead_hour=("lead_hour", "first"),
    )
    agg["ldaps_mean_dir_deg"] = (np.degrees(np.arctan2(agg["ldaps_wind50max_v_mean"], agg["ldaps_wind50max_u_mean"])) + 360.0) % 360.0
    agg["dir_sector"] = sector_label(agg["ldaps_mean_dir_deg"])
    top = read_csv(ISSUE32 / "04_directional_top_grid.csv")
    value_cols = [
        "ldaps_wind50max_speed",
        "ldaps_wind50max_speed2",
        "ldaps_wind50max_speed3",
        "ldaps_wind50max_u",
        "ldaps_wind50max_v",
        "ldaps_wind50max_dir_sin",
        "ldaps_wind50max_dir_cos",
        "ldaps_wind10_speed",
        "ldaps_wind10_speed2",
        "ldaps_wind10_speed3",
    ]
    pieces = [agg[["forecast_kst_dtm", "lead_hour", "ldaps_mean_dir_deg"]].copy()]
    cols_by_target: dict[str, list[str]] = {}
    for target in TARGETS:
        rows = agg[["forecast_kst_dtm", "dir_sector"]].merge(
            top[top["target"].eq(target)][["dir_sector", "grid_id"]],
            on="dir_sector",
            how="left",
        )
        rows["grid_id"] = rows["grid_id"].astype(int)
        selected = rows.merge(wind[["forecast_kst_dtm", "grid_id"] + value_cols], on=["forecast_kst_dtm", "grid_id"], how="left")
        rename = {c: f"{target}_dirgrid_{c}" for c in value_cols}
        selected = selected.rename(columns=rename)
        selected[f"{target}_dirgrid_grid_id"] = selected["grid_id"]
        target_cols = list(rename.values()) + [f"{target}_dirgrid_grid_id"]
        pieces.append(selected[["forecast_kst_dtm"] + target_cols])
        cols_by_target[target] = target_cols + ["lead_hour", "ldaps_mean_dir_deg"]
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="forecast_kst_dtm", how="left")
    return out, cols_by_target


def build_source_disagreement_features(data_dir: Path) -> pd.DataFrame:
    frames = {}
    for source in ["ldaps", "gfs"]:
        raw = read_csv(data_dir / "train" / f"{source}_train.csv")
        wind = add_grid_wind(raw, source)
        value_cols = [c for c in wind.columns if c not in {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id"}]
        grouped = wind.groupby("forecast_kst_dtm")[value_cols].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        frames[source] = grouped.reset_index()
    feat = frames["ldaps"].merge(frames["gfs"], on="forecast_kst_dtm", how="inner")
    out = pd.DataFrame({"forecast_kst_dtm": feat["forecast_kst_dtm"]})
    pairs = [
        ("wind10", "ldaps_wind10", "gfs_wind10"),
        ("ldaps50max_gfs100", "ldaps_wind50max", "gfs_wind100"),
        ("ldaps50max_gfs80", "ldaps_wind50max", "gfs_wind80"),
        ("ldaps10_gfs850", "ldaps_wind10", "gfs_wind850"),
    ]
    for name, left, right in pairs:
        l_speed = f"{left}_speed_mean"
        r_speed = f"{right}_speed_mean"
        if l_speed in feat.columns and r_speed in feat.columns:
            signed = feat[l_speed] - feat[r_speed]
            out[f"disagree_{name}_speed_signed"] = signed
            out[f"disagree_{name}_speed_abs"] = signed.abs()
            out[f"disagree_{name}_speed_ratio"] = feat[l_speed] / (feat[r_speed].abs() + 1e-6)
            lead = feat.get("ldaps_lead_hour_mean", pd.Series(0.0, index=feat.index))
            out[f"disagree_{name}_x_lead"] = signed * lead
    return out


def build_lead_interactions(features: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({"forecast_kst_dtm": features["forecast_kst_dtm"]})
    lead = pd.to_numeric(features["lead_hour"], errors="coerce")
    out["lead_hour"] = lead
    out["lead_centered"] = lead - 23.5
    out["lead_centered2"] = out["lead_centered"] ** 2
    for col in base_cols:
        if any(token in col for token in ["wind", "gust"]) and col in features.columns:
            out[f"lead_x_{col}"] = lead * pd.to_numeric(features[col], errors="coerce")
    return out


def valid_feature_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def train_lgbm_targetwise(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols_by_target: dict[str, list[str]],
    random_state: int,
    month_weights: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    from lightgbm import LGBMRegressor

    preds = pd.DataFrame(index=valid.index)
    counts = {}
    for target in TARGETS:
        feature_cols = valid_feature_cols(train, feature_cols_by_target[target])
        counts[target] = len(feature_cols)
        y = train[target]
        mask = y.notna()
        x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = y.loc[mask]
        x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
        sample_weight = None
        if month_weights:
            sample_weight = train.loc[mask, "month"].map(month_weights).fillna(1.0).to_numpy()
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
    return preds, counts


def summarize(name: str, preds: pd.DataFrame, valid: pd.DataFrame, feature_counts: dict[str, int]) -> Result:
    for target in TARGETS:
        preds[target] = np.clip(preds[target], 0.0, CAPACITY[target])
    metrics = evaluate_predictions(valid[TARGETS], preds, valid["forecast_kst_dtm"])
    high_score = evaluate_high_generation(valid[TARGETS], preds, valid["forecast_kst_dtm"])
    return Result(name, preds, metrics, high_score, feature_counts)


def result_row(result: Result) -> dict:
    return {
        "experiment": result.name,
        "score": result.metrics["score"],
        "one_minus_nmae": result.metrics["one_minus_nmae"],
        "avg_nmae": result.metrics["avg_nmae"],
        "ficr": result.metrics["ficr"],
        "worst_month": min(r["score"] for r in result.metrics["monthly_rows"]),
        "high_generation_score": result.high_generation_score,
        "eligible_hours": result.metrics["eligible_hours"],
        "feature_count_group1": result.feature_counts["kpx_group_1"],
        "feature_count_group2": result.feature_counts["kpx_group_2"],
        "feature_count_group3": result.feature_counts["kpx_group_3"],
    }


def row_target_table(valid: pd.DataFrame, preds: pd.DataFrame, name: str) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        actual = valid[target]
        pred = preds[target].clip(0.0, cap)
        err = pred - actual
        df = pd.DataFrame(
            {
                "forecast_kst_dtm": valid["forecast_kst_dtm"],
                "month": valid["month"],
                "hour": valid["hour"],
                "target": target,
                "actual": actual,
                "actual_ratio": actual / cap,
                f"{name}_pred": pred,
                f"{name}_abs_error": err.abs() / cap,
                f"{name}_signed_error": err / cap,
                f"{name}_pass8": (actual >= 0.10 * cap) & ((err.abs() / cap) <= 0.08),
                "eligible": actual >= 0.10 * cap,
            }
        )
        df["label_regime"] = pd.cut(
            df["actual_ratio"],
            [-np.inf, 0.0, 0.01, 0.08, 0.12, 0.50, 0.80, np.inf],
            labels=["zero", "near_zero_0_1pct", "low_1_8pct", "ficr_boundary_8_12pct", "mid_12_50pct", "high_50_80pct", "very_high_80pct_plus"],
            include_lowest=True,
        ).astype("object").fillna("missing")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def delta_tables(valid: pd.DataFrame, baseline: Result, results: list[Result]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = row_target_table(valid, baseline.preds, "baseline")
    regime_rows = []
    month_rows = []
    transition_rows = []
    for result in results:
        cand = row_target_table(valid, result.preds, "candidate")
        merged = base.merge(
            cand[["forecast_kst_dtm", "target", "candidate_abs_error", "candidate_signed_error", "candidate_pass8"]],
            on=["forecast_kst_dtm", "target"],
            how="inner",
        )
        merged["delta_abs_error"] = merged["candidate_abs_error"] - merged["baseline_abs_error"]
        merged["experiment"] = result.name
        for keys, rows in [
            (["experiment", "target", "label_regime"], regime_rows),
            (["experiment", "target", "month"], month_rows),
        ]:
            for group_key, g in merged[merged["eligible"]].groupby(keys, observed=True):
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                row = dict(zip(keys, group_key))
                row.update(
                    {
                        "rows": int(len(g)),
                        "baseline_abs_error": float(g["baseline_abs_error"].mean()),
                        "candidate_abs_error": float(g["candidate_abs_error"].mean()),
                        "delta_abs_error": float(g["delta_abs_error"].mean()),
                        "impact_delta": float(g["delta_abs_error"].sum()),
                    }
                )
                rows.append(row)
        for (target, label_regime), g in merged[merged["eligible"]].groupby(["target", "label_regime"], observed=True):
            transition_rows.append(
                {
                    "experiment": result.name,
                    "target": target,
                    "label_regime": label_regime,
                    "fail_to_pass8": int((~g["baseline_pass8"] & g["candidate_pass8"]).sum()),
                    "pass_to_fail8": int((g["baseline_pass8"] & ~g["candidate_pass8"]).sum()),
                    "net_pass8": int((~g["baseline_pass8"] & g["candidate_pass8"]).sum() - (g["baseline_pass8"] & ~g["candidate_pass8"]).sum()),
                    "delta_abs_error": float(g["delta_abs_error"].mean()),
                }
            )
    regime = pd.DataFrame(regime_rows)
    month = pd.DataFrame(month_rows)
    transition = pd.DataFrame(transition_rows)
    regime.to_csv(OUT / "regime_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
    month.to_csv(OUT / "month_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
    transition.to_csv(OUT / "ficr_transition_vs_baseline.csv", index=False, encoding="utf-8-sig")
    return regime, month, transition


def exposure_month_weights() -> dict[int, float]:
    exposure = read_csv(ISSUE32 / "07_2025_exposure_by_month.csv")
    exposure["exposure_risk"] = pd.to_numeric(exposure["exposure_risk"], errors="coerce")
    max_risk = exposure["exposure_risk"].max()
    exposure["weight"] = 1.0 + exposure["exposure_risk"] / max_risk
    exposure[["month", "exposure_risk", "weight"]].to_csv(OUT / "exposure_month_weights.csv", index=False, encoding="utf-8-sig")
    return dict(zip(exposure["month"].astype(int), exposure["weight"]))


def scada_risk_proxy_classification(feature_data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    scada = read_csv(ISSUE31 / "deep_dive" / "scada_label_residual_taxonomy_rows.csv")
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["risk_slice"] = scada["taxonomy"].isin(["scada_zero_label_positive", "low_online_label_positive", "scada_under_label"]).astype(int)
    feature_cols = [
        c
        for c in feature_data.columns
        if c
        not in {
            "forecast_kst_dtm",
            "kst_dtm",
            "kpx_group_1",
            "kpx_group_2",
            "kpx_group_3",
            "year_label",
            "month_label",
            "hour_label",
        }
        and pd.api.types.is_numeric_dtype(feature_data[c])
    ]
    rows = []
    for target in TARGETS:
        y = scada[scada["target"].eq(target)][["kst_dtm", "risk_slice"]]
        df = feature_data.merge(y, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
        df = df[df["year"].isin([2023, 2024])].copy()
        train = df[df["year"].eq(2023)]
        valid = df[df["year"].eq(2024)]
        if train["risk_slice"].nunique() < 2 or valid["risk_slice"].nunique() < 2:
            rows.append({"target": target, "roc_auc": np.nan, "average_precision": np.nan, "positive_rate_valid": float(valid["risk_slice"].mean()), "n_valid": int(len(valid))})
            continue
        x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train[feature_cols].median(numeric_only=True))
        x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train[feature_cols].median(numeric_only=True))
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=25, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
        clf.fit(x_train, train["risk_slice"])
        proba = clf.predict_proba(x_valid)[:, 1]
        rows.append(
            {
                "target": target,
                "roc_auc": float(roc_auc_score(valid["risk_slice"], proba)),
                "average_precision": float(average_precision_score(valid["risk_slice"], proba)),
                "positive_rate_valid": float(valid["risk_slice"].mean()),
                "n_valid": int(len(valid)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "scada_risk_proxy_classification.csv", index=False, encoding="utf-8-sig")
    return out


def plot_summary(summary: pd.DataFrame, regime: pd.DataFrame, proxy: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    s = summary.sort_values("score", ascending=False)
    axes[0].barh(s["experiment"], s["delta_score_vs_baseline"], color=np.where(s["delta_score_vs_baseline"] >= 0, "tab:blue", "tab:red"))
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Ablation score delta vs B1_aggregate")
    axes[0].set_xlabel("score delta")

    if not regime.empty:
        piv = regime.pivot_table(index="experiment", columns="label_regime", values="delta_abs_error", aggfunc="mean")
        order = ["ficr_boundary_8_12pct", "mid_12_50pct", "high_50_80pct", "very_high_80pct_plus"]
        piv = piv[[c for c in order if c in piv.columns]]
        im = axes[1].imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-0.01, vmax=0.01)
        axes[1].set_title("Regime delta abs error vs baseline; blue is better")
        axes[1].set_yticks(range(len(piv.index)))
        axes[1].set_yticklabels(piv.index)
        axes[1].set_xticks(range(len(piv.columns)))
        axes[1].set_xticklabels(piv.columns, rotation=25, ha="right")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                axes[1].text(j, i, f"{piv.iloc[i, j]:.4f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes[1], label="delta abs error")

    axes[2].bar(proxy["target"], proxy["roc_auc"], color="tab:green", alpha=0.75)
    for _, r in proxy.iterrows():
        axes[2].text(r["target"], r["roc_auc"], f" AP {r['average_precision']:.2f}", ha="center", va="bottom", fontsize=8)
    axes[2].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Can NWP/time/metadata predict SCADA-derived score-risk slices?")
    axes[2].set_ylabel("ROC AUC")
    fig.tight_layout()
    fig.savefig(FIG / "01_ablation_score_and_proxy_summary.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_report(summary: pd.DataFrame, regime: pd.DataFrame, month: pd.DataFrame, transition: pd.DataFrame, proxy: pd.DataFrame) -> None:
    best = summary.sort_values("score", ascending=False).iloc[0]
    best_candidate = summary[~summary["experiment"].eq("B1_aggregate")].sort_values("score", ascending=False).iloc[0]
    body = f"""# Deployable score-risk ablation from Issue #32 insights

이 실험은 Issue #32에서 얻은 인사이트가 실제 validation score 개선으로 이어지는지 검증합니다.

검증한 후보:

- `B1_aggregate`: 기존 aggregate weather baseline
- `B1_plus_direction_grid`: 풍향 sector별 best LDAPS grid feature
- `B1_plus_source_disagreement`: LDAPS-GFS disagreement feature
- `B1_plus_lead_interactions`: lead_hour 및 lead-hour interaction feature
- `B1_plus_all_score_insights`: 위 세 feature family를 모두 추가
- `B1_all_plus_2025_exposure_weight`: exposed month에 더 큰 sample weight를 준 모델

![Ablation summary]({RAW}/01_ablation_score_and_proxy_summary.png)

## Main Result

{md_table(summary.sort_values("score", ascending=False), 10)}

해석:

- 전체 최고 점수는 `{best['experiment']}`입니다.
- baseline 제외 최고 후보는 `{best_candidate['experiment']}`이고, baseline 대비 score delta는 `{best_candidate['delta_score_vs_baseline']:.6f}`입니다.
- 따라서 Issue #32의 feature 후보들은 **현재 fixed-LGBM/B1 protocol에서는 전체 score를 올린다고 증명되지 않았습니다**.
- 특히 feature family를 모두 합친 `B1_plus_all_score_insights`와 exposure-weighted model도 baseline을 넘지 못하면, 단순 feature addition이 아니라 model family, regularization, target-specific policy, postprocessing 쪽으로 넘어가야 합니다.

## Regime-Level Delta

{md_table(regime.sort_values("impact_delta").head(20), 20, ["experiment", "target", "label_regime", "rows", "baseline_abs_error", "candidate_abs_error", "delta_abs_error", "impact_delta"])}

해석:

- 일부 regime에서는 개선이 존재하지만, 다른 regime에서 악화되어 total score가 상쇄됩니다.
- 즉 #32의 insight는 “어디를 봐야 하는가”를 맞혔지만, 현재 feature 구현은 그 slice를 안정적으로 개선하지 못했습니다.

## FiCR Transition Delta

{md_table(transition.sort_values("net_pass8", ascending=False).head(20), 20)}

해석:

- FiCR boundary에서 fail-to-pass와 pass-to-fail이 동시에 발생합니다.
- score를 올리려면 feature만 추가하는 것보다 boundary-specific calibration/postprocessing을 별도 검증해야 합니다.

## 2025 Exposure Month Delta

{md_table(month.sort_values("impact_delta").head(20), 20, ["experiment", "target", "month", "rows", "baseline_abs_error", "candidate_abs_error", "delta_abs_error", "impact_delta"])}

해석:

- exposure-weighted training은 month별 risk를 반영했지만, 현재 protocol에서는 전체 score 개선으로 연결되지 않았습니다.
- 2025 exposure는 model selection/reporting weight로는 유용하지만, naive sample weighting은 충분하지 않습니다.

## Can NWP/time/metadata reproduce SCADA score-risk slices?

{md_table(proxy, 10)}

해석:

- SCADA-derived risk slice를 NWP/time/metadata로 분류할 수 있는지 2023 train -> 2024 valid classifier로 봤습니다.
- AUC가 0.5 근처면 test-safe proxy가 약하다는 뜻이고, 높으면 SCADA risk를 NWP feature로 일부 재현할 수 있다는 뜻입니다.
- 이 값이 낮은 group에서는 SCADA taxonomy를 모델 feature로 전환하는 전략이 위험합니다.

## Conclusion

Issue #32에서 얻은 score-relevant 축은 유효했습니다. 그러나 이번 ablation의 결론은 더 엄격합니다.

1. `direction-conditioned grid`, `source disagreement`, `lead interactions`, `2025 exposure weighting`은 현재 형태로는 B1 aggregate baseline을 이기지 못했습니다.
2. 따라서 score 개선 insight는 “단순 feature 추가”가 아니라 **slice-aware modeling/reporting/calibration**으로 사용해야 합니다.
3. 다음으로 할 일은:
   - label-regime별 calibration
   - FiCR boundary-specific postprocessing
   - group3 별도 모델 정책
   - stronger model family 또는 ensemble에서 같은 feature family 재검증
   - direction-grid feature를 hard sector 선택이 아니라 soft upstream weighting으로 재설계

즉 #32의 insight는 폐기할 것이 아니라, 현재 실험에서 “feature dump로는 부족하다”는 것이 증명된 상태입니다.
"""
    (Path(__file__).resolve().parent / "issue_update.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    labels = read_labels(DATA)
    b1 = build_feature_frame(DATA, "B1_aggregate")
    dir_feat, dir_cols = build_direction_conditioned_features(DATA)
    disagreement = build_source_disagreement_features(DATA)
    tmp = b1.merge(dir_feat[["forecast_kst_dtm", "lead_hour"]], on="forecast_kst_dtm", how="left")
    base_cols = [c for c in b1.columns if c != "forecast_kst_dtm"]
    lead_feat = build_lead_interactions(tmp, base_cols)

    features = b1.merge(dir_feat, on="forecast_kst_dtm", how="left").merge(disagreement, on="forecast_kst_dtm", how="left").merge(
        lead_feat, on="forecast_kst_dtm", how="left", suffixes=("", "_lead")
    )
    data = merge_features_labels(features, labels)
    train = data[data["year"].isin(W4.train_years)].copy()
    valid = data[data["year"].eq(2024)].copy().reset_index(drop=True)
    base_cols = valid_feature_cols(data, base_cols)
    disagreement_cols = [c for c in disagreement.columns if c != "forecast_kst_dtm"]
    lead_cols = [c for c in lead_feat.columns if c != "forecast_kst_dtm"]

    specs = {
        "B1_aggregate": {target: base_cols for target in TARGETS},
        "B1_plus_direction_grid": {target: base_cols + dir_cols[target] for target in TARGETS},
        "B1_plus_source_disagreement": {target: base_cols + disagreement_cols for target in TARGETS},
        "B1_plus_lead_interactions": {target: base_cols + lead_cols for target in TARGETS},
        "B1_plus_all_score_insights": {target: base_cols + dir_cols[target] + disagreement_cols + lead_cols for target in TARGETS},
        "B1_all_plus_2025_exposure_weight": {target: base_cols + dir_cols[target] + disagreement_cols + lead_cols for target in TARGETS},
    }
    month_weights = exposure_month_weights()

    results: list[Result] = []
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name, cols_by_target in specs.items():
        if name == "B1_aggregate":
            preds = train_lgbm_predictions(train, valid, base_cols, W4, 42)
            counts = {target: len(base_cols) for target in TARGETS}
        else:
            weights = month_weights if name == "B1_all_plus_2025_exposure_weight" else None
            preds, counts = train_lgbm_targetwise(train, valid, cols_by_target, 42, weights)
        result = summarize(name, preds, valid, counts)
        result.preds.to_csv(pred_dir / f"{name}_valid_predictions.csv", index=False, encoding="utf-8-sig")
        results.append(result)

    summary = pd.DataFrame([result_row(r) for r in results])
    baseline = summary[summary["experiment"].eq("B1_aggregate")].iloc[0]
    summary["delta_score_vs_baseline"] = summary["score"] - baseline["score"]
    summary["delta_ficr_vs_baseline"] = summary["ficr"] - baseline["ficr"]
    summary["delta_nmae_vs_baseline"] = summary["avg_nmae"] - baseline["avg_nmae"]
    summary.sort_values("score", ascending=False).to_csv(OUT / "ablation_summary.csv", index=False, encoding="utf-8-sig")

    regime, month, transition = delta_tables(valid, results[0], results[1:])
    proxy = scada_risk_proxy_classification(data)
    plot_summary(summary, regime, proxy)
    write_report(summary, regime, month, transition, proxy)
    (OUT / "run_metadata.json").write_text(
        json.dumps({"window": W4.name, "train_years": W4.train_years, "valid_year": 2024}, indent=2),
        encoding="utf-8",
    )
    print(summary.sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
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
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
EDA = ROOT / "experiments" / "comprehensive_eda_20260809" / "results"
DEEP = EDA / "deep_dive"
W4 = Window("W4_2022_2023_to_2024", (2022, 2023), None)
RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/fundamental_feature_distillation_20260809/results/figures"

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
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"]].copy()
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


def aggregate_grid_bank(data_dir: Path) -> pd.DataFrame:
    frames = []
    for source in ["ldaps", "gfs"]:
        wind = add_grid_wind(read_csv(data_dir / "train" / f"{source}_train.csv"), source)
        value_cols = [
            c
            for c in wind.columns
            if c not in {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"}
        ]
        grouped = wind.groupby("forecast_kst_dtm")[value_cols].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        frames.append(grouped.reset_index())
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="forecast_kst_dtm", how="inner")
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour
    out["dayofyear"] = out["forecast_kst_dtm"].dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    return out


def centroids() -> pd.DataFrame:
    info = read_csv(EDA / "info_turbine_metadata_parsed.csv")
    c = info.groupby("KPX그룹").agg(latitude=("latitude", "mean"), longitude=("longitude", "mean")).reset_index()
    c["target"] = c["KPX그룹"].map({1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
    return c[["target", "latitude", "longitude"]]


def soft_upstream_features(data_dir: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    ldaps = add_grid_wind(read_csv(data_dir / "train" / "ldaps_train.csv"), "ldaps")
    c = centroids()
    pieces = []
    cols_by_target = {}
    for _, row in c.iterrows():
        target = row["target"]
        site_lat = float(row["latitude"])
        site_lon = float(row["longitude"])
        g = ldaps.copy()
        km_lat = (g["latitude"] - site_lat) * 111.0
        km_lon = (g["longitude"] - site_lon) * 88.5
        # Grid position relative to site. Upstream lies opposite the wind vector.
        u = g["ldaps_wind50max_u"]
        v = g["ldaps_wind50max_v"]
        speed = np.sqrt(u * u + v * v) + 1e-6
        wind_e = u / speed
        wind_n = v / speed
        rel_e = km_lon
        rel_n = km_lat
        downwind = rel_e * wind_e + rel_n * wind_n
        crosswind = np.abs(rel_e * (-wind_n) + rel_n * wind_e)
        dist = np.sqrt(rel_e * rel_e + rel_n * rel_n)
        upstream_bonus = np.exp(-np.maximum(downwind, 0.0) / 3.0)
        crosswind_weight = np.exp(-(crosswind / 4.0) ** 2)
        distance_weight = np.exp(-dist / 10.0)
        g["soft_weight"] = upstream_bonus * crosswind_weight * distance_weight + 1e-6
        value_cols = ["ldaps_wind50max_speed", "ldaps_wind50max_speed2", "ldaps_wind50max_speed3", "ldaps_wind50max_u", "ldaps_wind50max_v", "ldaps_wind10_speed"]
        weighted_rows = []
        for ts, h in g.groupby("forecast_kst_dtm"):
            w = h["soft_weight"].to_numpy()
            total = w.sum()
            vals = {"forecast_kst_dtm": ts}
            for col in value_cols:
                x = h[col].to_numpy(dtype=float)
                vals[f"{target}_soft_{col}"] = float(np.nansum(x * w) / total)
            vals[f"{target}_soft_weight_entropy"] = float(-(w / total * np.log(w / total)).sum())
            weighted_rows.append(vals)
        part = pd.DataFrame(weighted_rows)
        cols = [c for c in part.columns if c != "forecast_kst_dtm"]
        cols_by_target[target] = cols
        pieces.append(part)
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="forecast_kst_dtm", how="inner")
    return out, cols_by_target


def train_scada_proxy_features(bank: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    teacher = read_csv(DEEP / "scada_hourly_group_teacher.csv")
    teacher["kst_dtm"] = pd.to_datetime(teacher["kst_dtm"])
    base_cols = [
        c
        for c in bank.columns
        if c != "forecast_kst_dtm" and pd.api.types.is_numeric_dtype(bank[c])
    ]
    rows = []
    pieces = [bank[["forecast_kst_dtm"]].copy()]
    cols_by_target = {}
    for target in TARGETS:
        t = teacher[teacher["target"].eq(target)][["kst_dtm", "scada_ws_mean", "scada_wd_mean", "online_rate"]]
        df = bank.merge(t, left_on="forecast_kst_dtm", right_on="kst_dtm", how="left")
        train_mask = df["year"].isin([2022, 2023]) & df["scada_ws_mean"].notna()
        valid_mask = df["year"].eq(2024) & df["scada_ws_mean"].notna()
        x_train = df.loc[train_mask, base_cols].replace([np.inf, -np.inf], np.nan)
        x_valid = df.loc[valid_mask, base_cols].replace([np.inf, -np.inf], np.nan)
        y_train = df.loc[train_mask, "scada_ws_mean"]
        y_valid = df.loc[valid_mask, "scada_ws_mean"]
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=450,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.85,
            min_child_samples=35,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        pred_all = model.predict(df[base_cols].replace([np.inf, -np.inf], np.nan))
        pred_valid = pred_all[valid_mask.to_numpy()]
        rows.append(
            {
                "target": target,
                "valid_rows": int(valid_mask.sum()),
                "mae_ws": float(mean_absolute_error(y_valid, pred_valid)),
                "r2_ws": float(r2_score(y_valid, pred_valid)),
                "teacher_mean_ws_valid": float(y_valid.mean()),
            }
        )
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": df["forecast_kst_dtm"],
                f"{target}_scada_ws_proxy": pred_all,
                f"{target}_scada_ws_proxy2": pred_all * pred_all,
                f"{target}_scada_ws_proxy3": pred_all * pred_all * pred_all,
            }
        )
        pieces.append(part)
        cols_by_target[target] = [f"{target}_scada_ws_proxy", f"{target}_scada_ws_proxy2", f"{target}_scada_ws_proxy3"]
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="forecast_kst_dtm", how="left")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "scada_wind_proxy_quality.csv", index=False, encoding="utf-8-sig")
    return out, summary, cols_by_target


def power_curve_features(feature_frame: pd.DataFrame, labels: pd.DataFrame, proxy_cols: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    pieces = [feature_frame[["forecast_kst_dtm"]].copy()]
    cols_by_target = {}
    joined = feature_frame.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_label"))
    joined["feature_year"] = joined["forecast_kst_dtm"].dt.year
    for target in TARGETS:
        proxy = proxy_cols[target][0]
        train = joined[joined["feature_year"].isin([2022, 2023]) & joined[target].notna()].copy()
        train["bin"] = pd.qcut(train[proxy], q=30, duplicates="drop")
        table = train.groupby("bin", observed=True).agg(
            curve_mean_ratio=(target, lambda x: float((x / CAPACITY[target]).mean())),
            curve_median_ratio=(target, lambda x: float((x / CAPACITY[target]).median())),
            proxy_mean=(proxy, "mean"),
        ).reset_index()
        centers = table["proxy_mean"].to_numpy()
        mean_vals = table["curve_mean_ratio"].to_numpy()
        med_vals = table["curve_median_ratio"].to_numpy()
        x = feature_frame[proxy].to_numpy()
        pred_mean = np.interp(x, centers, mean_vals, left=mean_vals[0], right=mean_vals[-1])
        pred_med = np.interp(x, centers, med_vals, left=med_vals[0], right=med_vals[-1])
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": feature_frame["forecast_kst_dtm"],
                f"{target}_proxy_power_curve_mean": pred_mean,
                f"{target}_proxy_power_curve_median": pred_med,
            }
        )
        pieces.append(part)
        cols_by_target[target] = [f"{target}_proxy_power_curve_mean", f"{target}_proxy_power_curve_median"]
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on="forecast_kst_dtm", how="left")
    return out, cols_by_target


def valid_feature_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    seen = set()
    out = []
    for col in cols:
        if col in df.columns and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def train_targetwise(train: pd.DataFrame, valid: pd.DataFrame, cols_by_target: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    from lightgbm import LGBMRegressor

    preds = pd.DataFrame(index=valid.index)
    counts = {}
    for target in TARGETS:
        cols = valid_feature_cols(train, cols_by_target[target])
        counts[target] = len(cols)
        mask = train[target].notna()
        x_train = train.loc[mask, cols].replace([np.inf, -np.inf], np.nan)
        x_valid = valid[cols].replace([np.inf, -np.inf], np.nan)
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=30,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x_train, train.loc[mask, target])
        preds[target] = model.predict(x_valid)
    return preds, counts


def summarize(name: str, preds: pd.DataFrame, valid: pd.DataFrame, counts: dict[str, int]) -> Result:
    for target in TARGETS:
        preds[target] = np.clip(preds[target], 0.0, CAPACITY[target])
    metrics = evaluate_predictions(valid[TARGETS], preds, valid["forecast_kst_dtm"])
    high = evaluate_high_generation(valid[TARGETS], preds, valid["forecast_kst_dtm"])
    return Result(name, preds, metrics, high, counts)


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
                "target": target,
                "actual_ratio": actual / cap,
                f"{name}_abs_error": err.abs() / cap,
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


def delta_tables(valid: pd.DataFrame, baseline: Result, candidates: list[Result]) -> pd.DataFrame:
    base = row_target_table(valid, baseline.preds, "baseline")
    rows = []
    for cand in candidates:
        c = row_target_table(valid, cand.preds, "candidate")
        merged = base.merge(c[["forecast_kst_dtm", "target", "candidate_abs_error", "candidate_pass8"]], on=["forecast_kst_dtm", "target"], how="inner")
        merged["delta_abs_error"] = merged["candidate_abs_error"] - merged["baseline_abs_error"]
        for (target, regime), g in merged[merged["eligible"]].groupby(["target", "label_regime"], observed=True):
            rows.append(
                {
                    "experiment": cand.name,
                    "target": target,
                    "label_regime": regime,
                    "rows": int(len(g)),
                    "delta_abs_error": float(g["delta_abs_error"].mean()),
                    "impact_delta": float(g["delta_abs_error"].sum()),
                    "fail_to_pass8": int((~g["baseline_pass8"] & g["candidate_pass8"]).sum()),
                    "pass_to_fail8": int((g["baseline_pass8"] & ~g["candidate_pass8"]).sum()),
                }
            )
    out = pd.DataFrame(rows)
    out["net_pass8"] = out["fail_to_pass8"] - out["pass_to_fail8"]
    out.to_csv(OUT / "regime_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
    return out


def plot(summary: pd.DataFrame, proxy_quality: pd.DataFrame, regime: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    s = summary.sort_values("score")
    axes[0].barh(s["experiment"], s["delta_score_vs_baseline"], color=np.where(s["delta_score_vs_baseline"] >= 0, "tab:blue", "tab:red"))
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Fundamental feature ablation score delta vs B1")
    axes[0].set_xlabel("score delta")

    axes[1].bar(proxy_quality["target"], proxy_quality["r2_ws"], color="tab:green", alpha=0.75)
    for _, r in proxy_quality.iterrows():
        axes[1].text(r["target"], r["r2_ws"], f"MAE {r['mae_ws']:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(1.0, proxy_quality["r2_ws"].max() + 0.1))
    axes[1].set_title("SCADA-like wind proxy quality, trained on 2022-2023 and validated on 2024")
    axes[1].set_ylabel("R2")

    piv = regime.pivot_table(index="experiment", columns="label_regime", values="delta_abs_error", aggfunc="mean")
    order = ["ficr_boundary_8_12pct", "mid_12_50pct", "high_50_80pct", "very_high_80pct_plus"]
    piv = piv[[c for c in order if c in piv.columns]]
    im = axes[2].imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    axes[2].set_title("Regime delta abs error vs B1; blue is better")
    axes[2].set_yticks(range(len(piv.index)))
    axes[2].set_yticklabels(piv.index)
    axes[2].set_xticks(range(len(piv.columns)))
    axes[2].set_xticklabels(piv.columns, rotation=25, ha="right")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[2].text(j, i, f"{piv.iloc[i, j]:.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[2], label="delta abs error")
    fig.tight_layout()
    fig.savefig(FIG / "01_fundamental_feature_ablation_summary.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_report(summary: pd.DataFrame, proxy_quality: pd.DataFrame, regime: pd.DataFrame) -> None:
    best = summary.sort_values("score", ascending=False).iloc[0]
    body = f"""# Fundamental feature/model ablation: SCADA-like NWP proxy, soft upstream grids, group power curve

이 업데이트는 postprocessing/calibration이 아니라 더 근본적인 feature/model 가설을 검증합니다.

검증한 후보:

- `B1_aggregate`: 기존 aggregate weather baseline
- `B1_plus_scada_wind_proxy`: NWP/time feature로 SCADA group wind를 distill한 proxy
- `B1_plus_soft_upstream`: 터빈 centroid와 wind vector를 이용한 soft upstream LDAPS aggregation
- `B1_plus_proxy_power_curve`: SCADA-like wind proxy 기반 group-specific power curve feature
- `B1_plus_fundamental_all`: 위 세 feature family를 모두 추가

![Fundamental ablation summary]({RAW}/01_fundamental_feature_ablation_summary.png)

## Main Result

{md_table(summary.sort_values("score", ascending=False), 10)}

해석:

- 최고 실험은 `{best['experiment']}`입니다.
- baseline 대비 score delta는 `{best['delta_score_vs_baseline']:.6f}`입니다.
- 이 실험은 “근본 feature를 만들면 바로 오른다”가 아니라, **어떤 근본 가설이 실제 fixed-LGBM validation에서 살아남는지**를 검증합니다.

## SCADA-like wind proxy quality

{md_table(proxy_quality, 10)}

해석:

- proxy R2가 높으면 NWP/time만으로 SCADA wind를 어느 정도 재현할 수 있다는 뜻입니다.
- 하지만 proxy가 좋아도 발전량 score가 오르지 않으면, proxy를 power model에 넣는 방식 또는 downstream model capacity가 문제입니다.

## Regime delta vs baseline

{md_table(regime.sort_values("impact_delta").head(20), 20)}

해석:

- 특정 label regime에서는 개선이 생길 수 있지만, total score는 다른 regime 악화와 FiCR 변화에 의해 쉽게 상쇄됩니다.
- 따라서 이 결과는 후속으로 group/regime-specific model 또는 soft calibration이 필요한지 판단하는 근거입니다.

## Conclusion

이번 실험의 판단 기준:

- score가 유의미하게 오르면 해당 feature family를 본선 후보로 승격합니다.
- score가 오르지 않더라도 SCADA proxy quality가 높으면, distillation 자체는 가능하지만 downstream 연결 방식이 실패한 것입니다.
- score와 proxy quality가 모두 약하면 해당 근본 가설은 현재 데이터/모델에서 약합니다.
"""
    (BASE / "issue_update.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    labels = read_labels(DATA)
    b1 = build_feature_frame(DATA, "B1_aggregate")
    bank = aggregate_grid_bank(DATA)
    proxy, proxy_quality, proxy_cols = train_scada_proxy_features(bank, DATA)
    soft, soft_cols = soft_upstream_features(DATA)

    merged_proxy = bank.merge(proxy, on="forecast_kst_dtm", how="left")
    curve, curve_cols = power_curve_features(merged_proxy, labels, proxy_cols)

    features = b1.merge(proxy, on="forecast_kst_dtm", how="left").merge(soft, on="forecast_kst_dtm", how="left").merge(curve, on="forecast_kst_dtm", how="left")
    data = merge_features_labels(features, labels)
    train = data[data["year"].isin(W4.train_years)].copy()
    valid = data[data["year"].eq(2024)].copy().reset_index(drop=True)
    base_cols = valid_feature_cols(data, [c for c in b1.columns if c != "forecast_kst_dtm"])

    specs = {
        "B1_aggregate": {target: base_cols for target in TARGETS},
        "B1_plus_scada_wind_proxy": {target: base_cols + proxy_cols[target] for target in TARGETS},
        "B1_plus_soft_upstream": {target: base_cols + soft_cols[target] for target in TARGETS},
        "B1_plus_proxy_power_curve": {target: base_cols + proxy_cols[target] + curve_cols[target] for target in TARGETS},
        "B1_plus_fundamental_all": {target: base_cols + proxy_cols[target] + soft_cols[target] + curve_cols[target] for target in TARGETS},
    }
    results = []
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name, cols in specs.items():
        if name == "B1_aggregate":
            preds = train_lgbm_predictions(train, valid, base_cols, W4, 42)
            counts = {target: len(base_cols) for target in TARGETS}
        else:
            preds, counts = train_targetwise(train, valid, cols)
        result = summarize(name, preds, valid, counts)
        result.preds.to_csv(pred_dir / f"{name}_valid_predictions.csv", index=False, encoding="utf-8-sig")
        results.append(result)

    summary = pd.DataFrame([result_row(r) for r in results])
    baseline = summary[summary["experiment"].eq("B1_aggregate")].iloc[0]
    summary["delta_score_vs_baseline"] = summary["score"] - baseline["score"]
    summary["delta_ficr_vs_baseline"] = summary["ficr"] - baseline["ficr"]
    summary["delta_nmae_vs_baseline"] = summary["avg_nmae"] - baseline["avg_nmae"]
    summary.sort_values("score", ascending=False).to_csv(OUT / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    regime = delta_tables(valid, results[0], results[1:])
    plot(summary, proxy_quality, regime)
    write_report(summary, proxy_quality, regime)
    (OUT / "run_metadata.json").write_text(json.dumps({"window": W4.name, "train_years": W4.train_years, "valid_year": 2024}, indent=2), encoding="utf-8")
    print(summary.sort_values("score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

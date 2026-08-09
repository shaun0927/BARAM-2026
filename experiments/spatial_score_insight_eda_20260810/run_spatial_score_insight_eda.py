from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

from run_cv_protocol import CAPACITY, TARGETS, evaluate_predictions, read_labels  # noqa: E402


DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
COMP = ROOT / "experiments" / "comprehensive_eda_20260809" / "results"
SPATIAL = COMP / "spatial_reinterpretation"
DEEP = COMP / "deep_dive"
ANCHOR_W4 = (
    ROOT
    / "experiments"
    / "autoresearch_harness"
    / "results"
    / "phase98_source_sister_oof_stack"
    / "diagnostic_predictions"
    / "phase98_w4_phase60_lineage.csv"
)

TARGET_GROUP = {"kpx_group_1": 1, "kpx_group_2": 2, "kpx_group_3": 3}
TARGET_MAKER = {"kpx_group_1": "VESTAS", "kpx_group_2": "VESTAS", "kpx_group_3": "UNISON"}
SECTOR_ORDER = ["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def lower16(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.maximum(out[target].to_numpy(float), 0.16 * CAPACITY[target])
    return out


def add_ldaps_wind(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude"]].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    u = pd.to_numeric(df["heightAboveGround_50_50MUmax"], errors="coerce")
    v = pd.to_numeric(df["heightAboveGround_50_50MVmax"], errors="coerce")
    speed = np.sqrt(u * u + v * v)
    direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    out["wind50max_u"] = u
    out["wind50max_v"] = v
    out["wind50max_speed"] = speed
    out["wind50max_dir_deg"] = direction
    return out


def sector_label(deg: pd.Series) -> pd.Series:
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    return pd.cut(deg, bins=bins, labels=SECTOR_ORDER, include_lowest=True, right=False).astype(str)


def group_centroids() -> pd.DataFrame:
    info = read_csv(COMP / "info_turbine_metadata_parsed.csv")
    cent = (
        info.groupby("KPX그룹")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), turbines=("호기", "count"))
        .reset_index()
    )
    cent["target"] = cent["KPX그룹"].map({1: "kpx_group_1", 2: "kpx_group_2", 3: "kpx_group_3"})
    return cent[["target", "latitude", "longitude", "turbines"]]


def soft_upstream_for_target(wind: pd.DataFrame, target: str, centroid: pd.Series) -> pd.DataFrame:
    g = wind.copy()
    km_lat = (g["latitude"] - float(centroid["latitude"])) * 111.0
    km_lon = (g["longitude"] - float(centroid["longitude"])) * 88.5
    speed = np.sqrt(g["wind50max_u"] ** 2 + g["wind50max_v"] ** 2) + 1e-6
    wind_e = g["wind50max_u"] / speed
    wind_n = g["wind50max_v"] / speed
    downwind = km_lon * wind_e + km_lat * wind_n
    crosswind = np.abs(km_lon * (-wind_n) + km_lat * wind_e)
    dist = np.sqrt(km_lon * km_lon + km_lat * km_lat)
    g["soft_weight"] = np.exp(-np.maximum(downwind, 0.0) / 3.0) * np.exp(-(crosswind / 4.0) ** 2) * np.exp(-dist / 10.0) + 1e-6

    rows = []
    for ts, h in g.groupby("forecast_kst_dtm", sort=True):
        w = h["soft_weight"].to_numpy(float)
        total = w.sum()
        rows.append(
            {
                "forecast_kst_dtm": ts,
                "target": target,
                "feature_family": "soft_upstream",
                "spatial_speed": float(np.nansum(h["wind50max_speed"].to_numpy(float) * w) / total),
                "selected_grid": np.nan,
                "direction_sector": None,
                "weight_entropy": float(-(w / total * np.log(w / total)).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_spatial_feature_rows(wind: pd.DataFrame) -> pd.DataFrame:
    reinterpret = read_csv(SPATIAL / "spatial_topology_reinterpretation_summary.csv").set_index("target")
    dir_top = read_csv(DEEP / "direction_conditional_ldaps_top_grid.csv")
    cent = group_centroids().set_index("target")

    mean_dir = (
        wind.groupby("forecast_kst_dtm", as_index=False)
        .agg(wind50max_u=("wind50max_u", "mean"), wind50max_v=("wind50max_v", "mean"))
    )
    mean_dir["mean_dir_deg"] = (np.degrees(np.arctan2(mean_dir["wind50max_v"], mean_dir["wind50max_u"])) + 360.0) % 360.0
    mean_dir["direction_sector"] = sector_label(mean_dir["mean_dir_deg"])

    grid_speed = wind[["forecast_kst_dtm", "grid_id", "wind50max_speed"]].copy()
    rows = []
    for target in TARGETS:
        nearest_grid = int(reinterpret.loc[target, "modal_nearest_ldaps_grid"])
        static_grid = int(reinterpret.loc[target, "static_top_corr_grid"])
        for family, grid_id in [("nearest_modal", nearest_grid), ("static_corr_top", static_grid)]:
            part = grid_speed[grid_speed["grid_id"].eq(grid_id)][["forecast_kst_dtm", "wind50max_speed"]].copy()
            part["target"] = target
            part["feature_family"] = family
            part["selected_grid"] = grid_id
            part["direction_sector"] = None
            part = part.rename(columns={"wind50max_speed": "spatial_speed"})
            rows.append(part)

        target_top = dir_top[dir_top["target"].eq(target)][["dir_sector", "grid_id"]].copy()
        part = mean_dir[["forecast_kst_dtm", "direction_sector"]].merge(
            target_top, left_on="direction_sector", right_on="dir_sector", how="left"
        )
        part = part.merge(grid_speed, on=["forecast_kst_dtm", "grid_id"], how="left")
        part["target"] = target
        part["feature_family"] = "direction_conditional_top"
        part = part.rename(columns={"wind50max_speed": "spatial_speed", "grid_id": "selected_grid"})
        rows.append(part[["forecast_kst_dtm", "target", "feature_family", "spatial_speed", "selected_grid", "direction_sector"]])

        rows.append(soft_upstream_for_target(wind, target, cent.loc[target]))

    out = pd.concat(rows, ignore_index=True)
    return out


def anchor_residual_rows() -> pd.DataFrame:
    labels = read_labels(DATA)
    valid = labels[labels["year"].eq(2024)].copy().reset_index(drop=True)
    valid = valid.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    anchor = read_csv(ANCHOR_W4)
    anchor["forecast_kst_dtm"] = pd.to_datetime(anchor["forecast_kst_dtm"])
    merged = valid[["forecast_kst_dtm", *TARGETS]].merge(anchor, on="forecast_kst_dtm", suffixes=("_actual", "_pred"))
    pred = lower16(merged[[f"{t}_pred" for t in TARGETS]].rename(columns={f"{t}_pred": t for t in TARGETS}))
    metrics = evaluate_predictions(
        merged[[f"{t}_actual" for t in TARGETS]].rename(columns={f"{t}_actual": t for t in TARGETS}),
        pred,
        merged["forecast_kst_dtm"],
    )

    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        actual = merged[f"{target}_actual"].to_numpy(float)
        forecast = pred[target].to_numpy(float)
        eligible = actual >= 0.10 * cap
        err_ratio = (forecast - actual) / cap
        abs_err_ratio = np.abs(err_ratio)
        pass8 = eligible & (abs_err_ratio <= 0.08)
        pass6 = eligible & (abs_err_ratio <= 0.06)
        actual_ratio = actual / cap
        pred_ratio = forecast / cap
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": merged["forecast_kst_dtm"],
                    "target": target,
                    "actual": actual,
                    "pred": forecast,
                    "actual_ratio": actual_ratio,
                    "pred_ratio": pred_ratio,
                    "signed_error_ratio": err_ratio,
                    "abs_error_ratio": abs_err_ratio,
                    "eligible": eligible,
                    "pass8": pass8,
                    "pass6": pass6,
                    "month": merged["forecast_kst_dtm"].dt.month,
                    "hour": merged["forecast_kst_dtm"].dt.hour,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    return out, metrics


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def feature_summary(spatial: pd.DataFrame, residual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = residual.merge(spatial, on=["forecast_kst_dtm", "target"], how="inner")
    df = df[df["eligible"]].copy()

    rows = []
    for (target, family), g in df.groupby(["target", "feature_family"], observed=True):
        rows.append(
            {
                "target": target,
                "feature_family": family,
                "rows": int(len(g)),
                "speed_actual_corr": safe_spearman(g["spatial_speed"], g["actual_ratio"]),
                "speed_abs_residual_corr": safe_spearman(g["spatial_speed"], g["abs_error_ratio"]),
                "speed_signed_residual_corr": safe_spearman(g["spatial_speed"], g["signed_error_ratio"]),
                "mean_abs_error_ratio": float(g["abs_error_ratio"].mean()),
                "pass8_rate": float(g["pass8"].mean()),
                "pass6_rate": float(g["pass6"].mean()),
                "selected_grid_unique": int(g["selected_grid"].nunique(dropna=True)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "spatial_feature_residual_summary.csv", index=False, encoding="utf-8-sig")

    decile_rows = []
    for (target, family), g in df.groupby(["target", "feature_family"], observed=True):
        work = g.copy()
        work["speed_decile"] = pd.qcut(work["spatial_speed"], 10, labels=False, duplicates="drop")
        for decile, h in work.groupby("speed_decile", observed=True):
            decile_rows.append(
                {
                    "target": target,
                    "feature_family": family,
                    "speed_decile": int(decile),
                    "rows": int(len(h)),
                    "speed_mean": float(h["spatial_speed"].mean()),
                    "actual_ratio_mean": float(h["actual_ratio"].mean()),
                    "abs_error_ratio_mean": float(h["abs_error_ratio"].mean()),
                    "signed_error_ratio_mean": float(h["signed_error_ratio"].mean()),
                    "pass8_rate": float(h["pass8"].mean()),
                    "pass6_rate": float(h["pass6"].mean()),
                }
            )
    deciles = pd.DataFrame(decile_rows)
    deciles.to_csv(OUT / "spatial_feature_decile_residual_profile.csv", index=False, encoding="utf-8-sig")

    month_rows = []
    for (target, family, month), g in df.groupby(["target", "feature_family", "month"], observed=True):
        month_rows.append(
            {
                "target": target,
                "feature_family": family,
                "month": int(month),
                "rows": int(len(g)),
                "speed_actual_corr": safe_spearman(g["spatial_speed"], g["actual_ratio"]),
                "speed_abs_residual_corr": safe_spearman(g["spatial_speed"], g["abs_error_ratio"]),
                "mean_abs_error_ratio": float(g["abs_error_ratio"].mean()),
                "pass8_rate": float(g["pass8"].mean()),
            }
        )
    month = pd.DataFrame(month_rows)
    month.to_csv(OUT / "spatial_month_stability.csv", index=False, encoding="utf-8-sig")

    direction_rows = []
    directional = df[df["feature_family"].eq("direction_conditional_top")].copy()
    for (target, sector, grid_id), g in directional.groupby(["target", "direction_sector", "selected_grid"], observed=True):
        direction_rows.append(
            {
                "target": target,
                "direction_sector": sector,
                "selected_grid": int(grid_id),
                "rows": int(len(g)),
                "speed_actual_corr": safe_spearman(g["spatial_speed"], g["actual_ratio"]),
                "speed_abs_residual_corr": safe_spearman(g["spatial_speed"], g["abs_error_ratio"]),
                "mean_abs_error_ratio": float(g["abs_error_ratio"].mean()),
                "pass8_rate": float(g["pass8"].mean()),
            }
        )
    direction = pd.DataFrame(direction_rows)
    direction.to_csv(OUT / "direction_sector_residual_summary.csv", index=False, encoding="utf-8-sig")
    return summary, deciles, month, direction


def ficr_transition_proxy(spatial: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    df = residual.merge(spatial, on=["forecast_kst_dtm", "target"], how="inner")
    df = df[df["eligible"]].copy()
    df["ficr_band"] = pd.cut(
        df["abs_error_ratio"],
        [-np.inf, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, np.inf],
        labels=["in6", "in8", "near_8_10", "lost_10_15", "lost_15_20", "lost_20_30", "lost_30_plus"],
    )
    rows = []
    for (target, family), g in df.groupby(["target", "feature_family"], observed=True):
        work = g.copy()
        work["speed_decile"] = pd.qcut(work["spatial_speed"], 10, labels=False, duplicates="drop")
        for decile, h in work.groupby("speed_decile", observed=True):
            rows.append(
                {
                    "target": target,
                    "feature_family": family,
                    "speed_decile": int(decile),
                    "rows": int(len(h)),
                    "settlement_weight_share": float(h["actual"].sum() / g["actual"].sum()),
                    "in8_rate": float(h["pass8"].mean()),
                    "near_8_10_rate": float(h["ficr_band"].eq("near_8_10").mean()),
                    "lost_10_15_rate": float(h["ficr_band"].eq("lost_10_15").mean()),
                    "mean_signed_error_ratio": float(h["signed_error_ratio"].mean()),
                    "mean_abs_error_ratio": float(h["abs_error_ratio"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "spatial_ficr_boundary_profile.csv", index=False, encoding="utf-8-sig")
    return out


def year_correlation_stability(spatial_train: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}).copy()
    lab = lab[lab["year"].isin([2022, 2023, 2024])].copy()
    rows = []
    for target in TARGETS:
        target_sp = spatial_train[spatial_train["target"].eq(target)].copy()
        df = lab[["forecast_kst_dtm", "year", "month", target]].merge(target_sp, on="forecast_kst_dtm", how="inner")
        df["actual_ratio"] = df[target] / CAPACITY[target]
        df = df[df[target].notna() & (df[target] >= 0.10 * CAPACITY[target])]
        for (family, year), g in df.groupby(["feature_family", "year"], observed=True):
            rows.append(
                {
                    "target": target,
                    "feature_family": family,
                    "year": int(year),
                    "rows": int(len(g)),
                    "speed_actual_corr": safe_spearman(g["spatial_speed"], g["actual_ratio"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "spatial_year_correlation_stability.csv", index=False, encoding="utf-8-sig")
    return out


def test_exposure(spatial_train: pd.DataFrame, spatial_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, family), train in spatial_train.groupby(["target", "feature_family"], observed=True):
        test = spatial_test[(spatial_test["target"].eq(target)) & (spatial_test["feature_family"].eq(family))]
        if len(test) == 0:
            continue
        q10, q90 = train["spatial_speed"].quantile([0.10, 0.90])
        train_mean = float(train["spatial_speed"].mean())
        train_std = float(train["spatial_speed"].std())
        for month, h in test.groupby(test["forecast_kst_dtm"].dt.month):
            rows.append(
                {
                    "target": target,
                    "feature_family": family,
                    "month": int(month),
                    "test_rows": int(len(h)),
                    "train_mean": train_mean,
                    "test_mean": float(h["spatial_speed"].mean()),
                    "z_delta_vs_train": float((h["spatial_speed"].mean() - train_mean) / train_std) if train_std > 0 else np.nan,
                    "test_above_train_p90_rate": float((h["spatial_speed"] > q90).mean()),
                    "test_below_train_p10_rate": float((h["spatial_speed"] < q10).mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "spatial_2025_exposure.csv", index=False, encoding="utf-8-sig")
    return out


def plot_all(summary: pd.DataFrame, deciles: pd.DataFrame, month: pd.DataFrame, ficr: pd.DataFrame, year_stability: pd.DataFrame, exposure: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 9, "axes.titlesize": 11})

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    piv = summary.pivot(index="feature_family", columns="target", values="speed_abs_residual_corr").loc[
        ["nearest_modal", "static_corr_top", "direction_conditional_top", "soft_upstream"]
    ]
    im = axes[0, 0].imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-0.25, vmax=0.25)
    axes[0, 0].set_title("Spatial speed vs anchor absolute residual; nonzero means possible residual lever")
    axes[0, 0].set_xticks(range(len(piv.columns)))
    axes[0, 0].set_xticklabels(piv.columns, rotation=20, ha="right")
    axes[0, 0].set_yticks(range(len(piv.index)))
    axes[0, 0].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[0, 0].text(j, i, f"{piv.iloc[i, j]:+.3f}", ha="center", va="center")
    fig.colorbar(im, ax=axes[0, 0])

    piv2 = summary.pivot(index="feature_family", columns="target", values="speed_actual_corr").loc[piv.index]
    im2 = axes[0, 1].imshow(piv2.values, aspect="auto", cmap="viridis", vmin=0.35, vmax=0.85)
    axes[0, 1].set_title("Spatial speed vs label; label signal can be strong even when residual signal is weak")
    axes[0, 1].set_xticks(range(len(piv2.columns)))
    axes[0, 1].set_xticklabels(piv2.columns, rotation=20, ha="right")
    axes[0, 1].set_yticks(range(len(piv2.index)))
    axes[0, 1].set_yticklabels(piv2.index)
    for i in range(piv2.shape[0]):
        for j in range(piv2.shape[1]):
            axes[0, 1].text(j, i, f"{piv2.iloc[i, j]:.3f}", ha="center", va="center", color="white" if piv2.iloc[i, j] > 0.65 else "black")
    fig.colorbar(im2, ax=axes[0, 1])

    show = deciles[deciles["feature_family"].isin(["direction_conditional_top", "soft_upstream"])].copy()
    for target in TARGETS:
        for family, linestyle in [("direction_conditional_top", "-"), ("soft_upstream", "--")]:
            d = show[(show["target"].eq(target)) & (show["feature_family"].eq(family))]
            axes[1, 0].plot(d["speed_decile"], d["abs_error_ratio_mean"], linestyle=linestyle, marker="o", label=f"{target}:{family}")
    axes[1, 0].set_title("Anchor abs residual by spatial-speed decile")
    axes[1, 0].set_xlabel("spatial speed decile")
    axes[1, 0].set_ylabel("mean abs error / capacity")
    axes[1, 0].legend(fontsize=7, ncol=2)

    exp = exposure[exposure["feature_family"].eq("direction_conditional_top")].copy()
    exp_piv = exp.pivot_table(index="target", columns="month", values="z_delta_vs_train", aggfunc="mean").loc[TARGETS]
    im3 = axes[1, 1].imshow(exp_piv.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    axes[1, 1].set_title("2025 exposure for direction-conditioned spatial speed")
    axes[1, 1].set_xticks(range(len(exp_piv.columns)))
    axes[1, 1].set_xticklabels(exp_piv.columns)
    axes[1, 1].set_yticks(range(len(exp_piv.index)))
    axes[1, 1].set_yticklabels(exp_piv.index)
    for i in range(exp_piv.shape[0]):
        for j in range(exp_piv.shape[1]):
            axes[1, 1].text(j, i, f"{exp_piv.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im3, ax=axes[1, 1], label="z delta")
    fig.tight_layout()
    fig.savefig(FIG / "01_spatial_residual_signal_summary.png")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    m = month[month["feature_family"].isin(["direction_conditional_top", "soft_upstream"])].copy()
    for target in TARGETS:
        for family, linestyle in [("direction_conditional_top", "-"), ("soft_upstream", "--")]:
            d = m[(m["target"].eq(target)) & (m["feature_family"].eq(family))]
            axes[0].plot(d["month"], d["speed_abs_residual_corr"], marker="o", linestyle=linestyle, label=f"{target}:{family}")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Monthly stability of spatial residual correlation")
    axes[0].set_ylabel("Spearman with abs residual")
    axes[0].legend(fontsize=7, ncol=2)

    y = year_stability[year_stability["feature_family"].isin(["direction_conditional_top", "soft_upstream"])].copy()
    for target in TARGETS:
        for family, linestyle in [("direction_conditional_top", "-"), ("soft_upstream", "--")]:
            d = y[(y["target"].eq(target)) & (y["feature_family"].eq(family))]
            axes[1].plot(d["year"], d["speed_actual_corr"], marker="o", linestyle=linestyle, label=f"{target}:{family}")
    axes[1].set_title("Year stability of label correlation")
    axes[1].set_ylabel("Spearman with label ratio")
    axes[1].set_xticks([2022, 2023, 2024])

    f = ficr[ficr["feature_family"].eq("direction_conditional_top")].copy()
    for target in TARGETS:
        d = f[f["target"].eq(target)]
        axes[2].plot(d["speed_decile"], d["near_8_10_rate"] + d["lost_10_15_rate"], marker="o", label=target)
    axes[2].set_title("FiCR-near-loss mass by direction-conditioned spatial-speed decile")
    axes[2].set_xlabel("spatial speed decile")
    axes[2].set_ylabel("near 8-15% loss rate")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_spatial_stability_and_ficr_boundary.png")
    plt.close(fig)


def table(df: pd.DataFrame, n: int = 20, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_issue(metrics: dict, summary: pd.DataFrame, deciles: pd.DataFrame, month: pd.DataFrame, ficr: pd.DataFrame, year_stability: pd.DataFrame, exposure: pd.DataFrame) -> None:
    raw_base = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/spatial_score_insight_eda_20260810/results/figures"
    residual_rank = summary.sort_values("speed_abs_residual_corr", key=lambda s: s.abs(), ascending=False)
    label_rank = summary.sort_values("speed_actual_corr", ascending=False)
    exp_rank = exposure.reindex(exposure["z_delta_vs_train"].abs().sort_values(ascending=False).index)
    ficr_rank = ficr.sort_values("near_8_10_rate", ascending=False)

    stable = (
        year_stability[year_stability["feature_family"].isin(["direction_conditional_top", "soft_upstream"])]
        .groupby(["target", "feature_family"])
        .agg(years=("year", "nunique"), min_corr=("speed_actual_corr", "min"), max_corr=("speed_actual_corr", "max"))
        .reset_index()
    )

    body = f"""# Spatial topology score-insight EDA: does direction/upstream geometry explain anchor residual and FiCR risk?

## Why this issue exists

Issue #31 clarified that spatial topology should not be read as “nearest grid wins.” The remaining question is whether the spatial signal is merely descriptive, or whether it can become a score-improving modeling lever.

This issue tests that explicitly against the current W4 anchor:

- anchor panel: `phase98_w4_phase60_lineage + Lower16`
- anchor W4 score: `{metrics['score']:.10f}`
- objective: determine whether `nearest`, `static corr-top`, `direction-conditioned top`, or `soft upstream` spatial features explain **current-anchor residual**, FiCR boundary risk, and 2025 exposure.

No model is promoted here. This is EDA for deciding whether spatial feature engineering is worth a controlled modeling phase.

## Figures

![Spatial residual signal summary]({raw_base}/01_spatial_residual_signal_summary.png)

![Spatial stability and FiCR boundary]({raw_base}/02_spatial_stability_and_ficr_boundary.png)

## 1. Label signal is real, but residual signal is much weaker

Top label correlations:

{table(label_rank, 12, ['target', 'feature_family', 'speed_actual_corr', 'speed_abs_residual_corr', 'pass8_rate', 'selected_grid_unique'])}

Top residual associations:

{table(residual_rank, 12, ['target', 'feature_family', 'speed_actual_corr', 'speed_abs_residual_corr', 'speed_signed_residual_corr', 'pass8_rate', 'selected_grid_unique'])}

Interpretation:

- Spatial speed features strongly explain the label surface, especially for `wind50max`-based LDAPS features.
- But the correlation with **anchor absolute residual** is much smaller. This means the current anchor already uses much of the basic wind-speed signal.
- A spatial modeling phase should therefore target residual slices, not re-learn generic wind-to-power mapping.

## 2. Nearest, static corr-top, direction-top, and soft-upstream are different objects

Summary by feature family:

{table(summary.sort_values(['target', 'feature_family']), 20, ['target', 'feature_family', 'rows', 'speed_actual_corr', 'speed_abs_residual_corr', 'speed_signed_residual_corr', 'pass8_rate', 'selected_grid_unique'])}

Interpretation:

- `nearest_modal` is only a geometry prior.
- `static_corr_top` often has strong label correlation but still does not guarantee residual leverage.
- `direction_conditional_top` keeps the direction-dependent grid identity visible.
- `soft_upstream` is physically smoother but may dilute the sharp grid identity that correlation selected.

## 3. Decile profile: where residual risk concentrates

The decile profile table records how anchor error and FiCR pass rates change across spatial-speed deciles.

{table(deciles[deciles['feature_family'].isin(['direction_conditional_top', 'soft_upstream'])].sort_values(['target', 'feature_family', 'speed_decile']), 30, ['target', 'feature_family', 'speed_decile', 'rows', 'speed_mean', 'actual_ratio_mean', 'abs_error_ratio_mean', 'signed_error_ratio_mean', 'pass8_rate'])}

Interpretation:

- Spatial-speed deciles are useful as diagnostic strata.
- However, a monotonic residual correction is not automatically justified. Some deciles carry signed overprediction, others underprediction, and group behavior differs.
- Any correction has to be cross-fitted and group-specific; otherwise it becomes the same local-offset failure mode seen in prior phases.

## 4. Stability: label correlation survives better than residual correlation

Year-level label stability:

{table(stable, 20)}

Monthly residual stability, largest absolute correlations:

{table(month.reindex(month['speed_abs_residual_corr'].abs().sort_values(ascending=False).index), 20, ['target', 'feature_family', 'month', 'rows', 'speed_abs_residual_corr', 'mean_abs_error_ratio', 'pass8_rate'])}

Interpretation:

- The label relationship is stable enough to justify spatial features as candidate predictors.
- The residual relationship is month-sensitive. This is the key risk: score improvement requires correcting what the anchor misses, not what the label already expresses.
- A spatial phase must use W3/W4 or month-block validation. W4-only selection is not enough.

## 5. FiCR boundary: spatial regimes can identify risk mass, but not yet a safe correction

High near-loss slices:

{table(ficr_rank, 20, ['target', 'feature_family', 'speed_decile', 'rows', 'settlement_weight_share', 'in8_rate', 'near_8_10_rate', 'lost_10_15_rate', 'mean_signed_error_ratio', 'mean_abs_error_ratio'])}

Interpretation:

- Spatial-speed strata do expose where FiCR near-loss mass sits.
- This is useful because a model phase can target `8-15%` error rows rather than average MAE.
- But EDA alone does not prove a correction direction. The sign of residuals is mixed; blind offsets would risk converting pass rows into fail rows.

## 6. 2025 exposure: the spatial regimes are present, but shifted by month

Largest 2025 spatial-speed shifts:

{table(exp_rank, 24, ['target', 'feature_family', 'month', 'test_rows', 'z_delta_vs_train', 'test_above_train_p90_rate', 'test_below_train_p10_rate'])}

Interpretation:

- The spatial/wind regimes are present in 2025, but month-level exposure is not identical to train.
- A spatial feature may matter publicly only if the target regime appears in the scored public subset.
- This argues for exposure-aware reporting, not for selecting a month guard from W4.

## Conclusion

Spatial topology gives a real modeling lead, but not yet a submission-ready rule.

What is now supported:

1. `nearest_grid` is insufficient.
2. `direction_conditional_top` and `soft_upstream` are the right feature families to test next.
3. The strongest EDA value is as residual/FICR stratification, not as a direct correction.
4. Any score-improvement claim must prove residual reduction under W3/W4 or month-block validation.

What must be proved in the next modeling phase:

- A direction/upstream spatial feature reduces current-anchor residual, not only label error.
- The reduction is positive on independent time blocks.
- FiCR `fail_to_pass8` gains exceed `pass_to_fail8` losses.
- 2025 exposure includes the same spatial regimes.

If those are not proved, spatial topology remains an explanatory EDA axis rather than a leaderboard lever.

## Artifacts

- `experiments/spatial_score_insight_eda_20260810/run_spatial_score_insight_eda.py`
- `results/spatial_feature_residual_summary.csv`
- `results/spatial_feature_decile_residual_profile.csv`
- `results/spatial_month_stability.csv`
- `results/spatial_ficr_boundary_profile.csv`
- `results/spatial_year_correlation_stability.csv`
- `results/spatial_2025_exposure.csv`
- `results/figures/01_spatial_residual_signal_summary.png`
- `results/figures/02_spatial_stability_and_ficr_boundary.png`
"""
    (BASE / "issue_body.md").write_text(body, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    labels = read_labels(DATA)
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    residual, metrics = anchor_residual_rows()

    ldaps_train = add_ldaps_wind(read_csv(DATA / "train" / "ldaps_train.csv"))
    ldaps_test = add_ldaps_wind(read_csv(DATA / "test" / "ldaps_test.csv"))
    spatial_train = build_spatial_feature_rows(ldaps_train)
    spatial_test = build_spatial_feature_rows(ldaps_test)
    summary, deciles, month, direction = feature_summary(spatial_train, residual)
    ficr = ficr_transition_proxy(spatial_train, residual)
    year_stability = year_correlation_stability(spatial_train, labels)
    exposure = test_exposure(spatial_train, spatial_test)
    plot_all(summary, deciles, month, ficr, year_stability, exposure)
    write_issue(metrics, summary, deciles, month, ficr, year_stability, exposure)

    manifest = {
        "experiment": "spatial_score_insight_eda_20260810",
        "anchor": "phase98_w4_phase60_lineage + Lower16",
        "anchor_score": metrics["score"],
        "anchor_one_minus_nmae": metrics["one_minus_nmae"],
        "anchor_ficr": metrics["ficr"],
        "feature_families": ["nearest_modal", "static_corr_top", "direction_conditional_top", "soft_upstream"],
        "verdict": "SPATIAL_SIGNAL_REAL_BUT_RESIDUAL_LEVER_UNPROVED",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

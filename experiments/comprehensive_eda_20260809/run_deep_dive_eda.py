from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
DEEP = OUT / "deep_dive"

CAPACITY = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}
TARGETS = list(CAPACITY)


def setup() -> None:
    DEEP.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def add_weather_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour
    u10 = out["heightAboveGround_10_10u"]
    v10 = out["heightAboveGround_10_10v"]
    out["wind10_speed"] = np.hypot(u10, v10)
    out["wind10_dir_math"] = (np.degrees(np.arctan2(v10, u10)) + 360.0) % 360.0
    if source == "ldaps":
        umax = out["heightAboveGround_50_50MUmax"]
        vmax = out["heightAboveGround_50_50MVmax"]
        out["wind_primary_speed"] = np.hypot(umax, vmax)
        out["wind_primary_dir_math"] = (np.degrees(np.arctan2(vmax, umax)) + 360.0) % 360.0
    else:
        u850 = out["isobaricInhPa_850_u"]
        v850 = out["isobaricInhPa_850_v"]
        out["wind_primary_speed"] = np.hypot(u850, v850)
        out["wind_primary_dir_math"] = (np.degrees(np.arctan2(v850, u850)) + 360.0) % 360.0
    return out


def labels() -> pd.DataFrame:
    df = read_csv(DATA / "train" / "train_labels.csv")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    df["year"] = df["kst_dtm"].dt.year
    df["month"] = df["kst_dtm"].dt.month
    df["hour"] = df["kst_dtm"].dt.hour
    return df


def scada_hourly() -> pd.DataFrame:
    vestas = read_csv(DATA / "train" / "scada_vestas_train.csv")
    unison = read_csv(DATA / "train" / "scada_unison_train.csv")
    rows = []

    def one(df: pd.DataFrame, target: str, nums: list[int], prefix: str, cap_kw10m: float) -> None:
        work = df.copy()
        work["kst_dtm"] = pd.to_datetime(work["kst_dtm"])
        work["hour_end"] = work["kst_dtm"].dt.ceil("h")
        power_cols = [f"{prefix}{i:02d}_power_kw10m" for i in nums]
        ws_cols = [f"{prefix}{i:02d}_ws" for i in nums]
        wd_cols = [f"{prefix}{i:02d}_wd" for i in nums]
        clean_power = []
        online = []
        for c in power_cols:
            valid = work[c].between(0, cap_kw10m * 1.10)
            clean_power.append(work[c].where(valid))
            online.append((work[c].where(valid).fillna(0) > cap_kw10m * 0.02).astype(float))
        power = pd.concat(clean_power, axis=1).sum(axis=1, skipna=False)
        ws = work[ws_cols].mean(axis=1, skipna=True)
        wd_rad = np.deg2rad(work[wd_cols])
        wd_x = np.cos(wd_rad).mean(axis=1, skipna=True)
        wd_y = np.sin(wd_rad).mean(axis=1, skipna=True)
        online_rate = pd.concat(online, axis=1).mean(axis=1)
        h = pd.DataFrame(
            {
                "kst_dtm": work["hour_end"],
                "target": target,
                "scada_power_clean": power,
                "scada_ws_mean": ws,
                "wd_x": wd_x,
                "wd_y": wd_y,
                "online_rate": online_rate,
            }
        ).groupby(["kst_dtm", "target"], as_index=False).agg(
            scada_power_clean=("scada_power_clean", "sum"),
            scada_ws_mean=("scada_ws_mean", "mean"),
            wd_x=("wd_x", "mean"),
            wd_y=("wd_y", "mean"),
            online_rate=("online_rate", "mean"),
        )
        h["scada_wd_mean"] = (np.degrees(np.arctan2(h["wd_y"], h["wd_x"])) + 360.0) % 360.0
        rows.append(h.drop(columns=["wd_x", "wd_y"]))

    one(vestas, "kpx_group_1", list(range(1, 7)), "vestas_wtg", 600.0)
    one(vestas, "kpx_group_2", list(range(7, 13)), "vestas_wtg", 600.0)
    one(unison, "kpx_group_3", list(range(1, 6)), "unison_wtg", 700.0)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DEEP / "scada_hourly_group_teacher.csv", index=False, encoding="utf-8-sig")
    return out


def scada_residual_taxonomy(label_df: pd.DataFrame, scada: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, cap in CAPACITY.items():
        y = label_df[["kst_dtm", "year", "month", "hour", target]].rename(columns={target: "label"})
        g = scada[scada["target"].eq(target)].merge(y, on="kst_dtm", how="inner")
        g["target"] = target
        g["label_ratio"] = g["label"] / cap
        g["scada_ratio"] = g["scada_power_clean"] / cap
        g["scada_minus_label_ratio"] = g["scada_ratio"] - g["label_ratio"]
        g["abs_residual_ratio"] = g["scada_minus_label_ratio"].abs()
        conds = [
            g["label"].isna(),
            g["scada_ratio"].lt(0.03) & g["label_ratio"].ge(0.10),
            g["online_rate"].lt(0.60) & g["label_ratio"].ge(0.10),
            g["scada_minus_label_ratio"].lt(-0.12),
            g["scada_minus_label_ratio"].gt(0.12),
            g["abs_residual_ratio"].le(0.03),
        ]
        labels_ = [
            "label_missing",
            "scada_zero_label_positive",
            "low_online_label_positive",
            "scada_under_label",
            "scada_over_label",
            "tight_reconstruction",
        ]
        g["taxonomy"] = np.select(conds, labels_, default="moderate_residual")
        rows.append(g)
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(DEEP / "scada_label_residual_taxonomy_rows.csv", index=False, encoding="utf-8-sig")
    summary = long.groupby(["target", "taxonomy"], as_index=False).agg(
        rows=("taxonomy", "size"),
        mean_abs_residual_ratio=("abs_residual_ratio", "mean"),
        mean_label_ratio=("label_ratio", "mean"),
        mean_scada_ratio=("scada_ratio", "mean"),
        mean_online_rate=("online_rate", "mean"),
    )
    summary.to_csv(DEEP / "scada_label_residual_taxonomy_summary.csv", index=False, encoding="utf-8-sig")

    pivot = summary.pivot(index="target", columns="taxonomy", values="rows").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    bottom = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col], bottom=bottom, label=col)
        bottom += pivot[col].values
    ax.set_title("SCADA-label residual taxonomy: share of hourly rows")
    ax.set_ylabel("row share")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "12_scada_residual_taxonomy.png")
    plt.close(fig)

    timeline = long.copy()
    timeline["ym"] = timeline["kst_dtm"].dt.to_period("M").astype(str)
    heat = timeline.groupby(["target", "ym"], as_index=False)["abs_residual_ratio"].mean()
    piv = heat.pivot(index="target", columns="ym", values="abs_residual_ratio")
    fig, ax = plt.subplots(figsize=(13, 3.8))
    im = ax.imshow(piv.fillna(0).values, aspect="auto", cmap="viridis")
    ax.set_title("SCADA-label residual by month: residual left after clean hourly reconstruction")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=70)
    fig.colorbar(im, ax=ax, label="mean abs residual / capacity")
    fig.tight_layout()
    fig.savefig(FIG / "13_scada_residual_monthly_heatmap.png")
    plt.close(fig)
    return long


def nwp_scada_bias(label_df: pd.DataFrame, scada: pd.DataFrame, ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    # Use simple, robust group-specific grid choices from nearest/correlation evidence.
    choices = {
        "kpx_group_1": {"ldaps": 6, "gfs": 5},
        "kpx_group_2": {"ldaps": 6, "gfs": 5},
        "kpx_group_3": {"ldaps": 12, "gfs": 5},
    }
    rows = []
    for target in TARGETS:
        s = scada[scada["target"].eq(target)][["kst_dtm", "scada_ws_mean", "scada_wd_mean", "online_rate"]].copy()
        s = s.merge(label_df[["kst_dtm", "year", "month", "hour", target]], on="kst_dtm", how="left")
        s["label_ratio"] = s[target] / CAPACITY[target]
        for source, df in [("ldaps", ldaps), ("gfs", gfs)]:
            grid = choices[target][source]
            w = df[df["grid_id"].eq(grid)][["forecast_kst_dtm", "wind_primary_speed", "wind_primary_dir_math"]].rename(
                columns={"forecast_kst_dtm": "kst_dtm", "wind_primary_speed": "nwp_speed", "wind_primary_dir_math": "nwp_dir"}
            )
            m = s.merge(w, on="kst_dtm", how="inner")
            m["target"] = target
            m["source"] = source
            m["grid_id"] = grid
            m["speed_bias_nwp_minus_scada"] = m["nwp_speed"] - m["scada_ws_mean"]
            # absolute circular direction difference.
            diff = ((m["nwp_dir"] - m["scada_wd_mean"] + 180.0) % 360.0) - 180.0
            m["abs_dir_diff"] = diff.abs()
            rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DEEP / "nwp_scada_wind_bias_rows.csv", index=False, encoding="utf-8-sig")
    summary = out.groupby(["target", "source", "month"], as_index=False).agg(
        rows=("target", "size"),
        mean_speed_bias=("speed_bias_nwp_minus_scada", "mean"),
        mae_speed_bias=("speed_bias_nwp_minus_scada", lambda x: x.abs().mean()),
        mean_abs_dir_diff=("abs_dir_diff", "mean"),
        mean_label_ratio=("label_ratio", "mean"),
    )
    summary.to_csv(DEEP / "nwp_scada_wind_bias_monthly.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, target in zip(axes, TARGETS):
        x = summary[summary["target"].eq(target)]
        for source, g in x.groupby("source"):
            ax.plot(g["month"], g["mean_speed_bias"], marker="o", label=f"{source} mean bias")
            ax.plot(g["month"], g["mae_speed_bias"], marker=".", linestyle="--", label=f"{source} abs bias")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_ylabel("m/s")
        ax.grid(alpha=0.25)
        ax.legend(ncols=2, loc="upper right")
    axes[-1].set_xlabel("month")
    fig.suptitle("NWP-to-SCADA wind speed bias by group and month")
    fig.tight_layout()
    fig.savefig(FIG / "14_nwp_scada_wind_bias_monthly.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        sample = out[out["target"].eq(target)].dropna(subset=["scada_ws_mean", "nwp_speed", "source"])
        if len(sample) > 18000:
            sample = sample.sample(18000, random_state=7)
        for source, g in sample.groupby("source"):
            ax.scatter(g["scada_ws_mean"], g["nwp_speed"], s=3, alpha=0.18, label=source)
        lim = max(25, sample[["scada_ws_mean", "nwp_speed"]].quantile(0.995).max().max())
        ax.plot([0, lim], [0, lim], color="black", linewidth=1)
        ax.set_title(target)
        ax.set_xlabel("SCADA wind speed")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("NWP selected-grid wind speed")
    axes[0].legend()
    fig.suptitle("NWP wind vs train-only SCADA wind: teacher/student transfer gap")
    fig.tight_layout()
    fig.savefig(FIG / "15_nwp_scada_wind_scatter.png")
    plt.close(fig)
    return out


def direction_conditional_spatial(label_df: pd.DataFrame, ldaps: pd.DataFrame) -> pd.DataFrame:
    labels_long = []
    for target, cap in CAPACITY.items():
        y = label_df[["kst_dtm", target]].rename(columns={"kst_dtm": "forecast_kst_dtm", target: "actual"})
        y["target"] = target
        y["actual_ratio"] = y["actual"] / cap
        labels_long.append(y)
    yall = pd.concat(labels_long, ignore_index=True)
    rows = []
    bins = np.arange(0, 361, 45)
    for grid_id, g in ldaps.groupby("grid_id"):
        w = g[["forecast_kst_dtm", "grid_id", "wind_primary_speed", "wind_primary_dir_math"]].copy()
        w["dir_sector"] = pd.cut(w["wind_primary_dir_math"], bins=bins, right=False, include_lowest=True, labels=[f"{i}-{i+45}" for i in bins[:-1]])
        m = yall.merge(w, on="forecast_kst_dtm", how="inner")
        for (target, sector), x in m.groupby(["target", "dir_sector"], observed=True):
            ok = x[["actual_ratio", "wind_primary_speed"]].dropna()
            rows.append(
                {
                    "target": target,
                    "grid_id": int(grid_id),
                    "dir_sector": str(sector),
                    "rows": int(len(ok)),
                    "spearman_corr": float(ok["actual_ratio"].corr(ok["wind_primary_speed"], method="spearman")) if len(ok) > 30 else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(DEEP / "direction_conditional_ldaps_grid_corr.csv", index=False, encoding="utf-8-sig")
    top = out.sort_values("spearman_corr", ascending=False).groupby(["target", "dir_sector"], as_index=False).first()
    top.to_csv(DEEP / "direction_conditional_ldaps_top_grid.csv", index=False, encoding="utf-8-sig")

    for target in TARGETS:
        piv = out[out["target"].eq(target)].pivot_table(index="grid_id", columns="dir_sector", values="spearman_corr", aggfunc="max")
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(piv.fillna(0).values, aspect="auto", vmin=0, vmax=0.9, cmap="viridis")
        ax.set_title(f"{target}: LDAPS wind50max correlation by wind-direction sector")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"grid {int(i)}" for i in piv.index])
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right")
        fig.colorbar(im, ax=ax, label="Spearman corr")
        fig.tight_layout()
        fig.savefig(FIG / f"16_directional_spatial_corr_{target}.png")
        plt.close(fig)
    return out


def exposure_risk(corr: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    # Combine label relevance (corr) with 2025-vs-2024 shift. Only features present in both tables enter.
    corr2 = corr.copy()
    corr2["feature"] = corr2["wind_feature"].replace(
        {
            "wind50max_speed": "wind50max_speed",
            "wind10_speed": "wind10_speed",
            "wind850_speed": "wind850_speed",
            "wind700_speed": "wind700_speed",
            "wind500_speed": "wind500_speed",
            "wind80_speed": "wind80_speed",
            "wind100_speed": "wind100_speed",
            "pblwind_speed": "pblwind_speed",
        }
    )
    rel = corr2.groupby(["source", "feature"], as_index=False)["spearman_corr"].max()
    risk = shift.merge(rel, on=["source", "feature"], how="inner")
    risk["weighted_abs_shift"] = risk["z_delta_vs_2024"].abs() * risk["spearman_corr"].clip(lower=0)
    risk_month = risk.groupby("month", as_index=False).agg(
        exposure_risk=("weighted_abs_shift", "sum"),
        max_single_shift=("weighted_abs_shift", "max"),
        shifted_features=("feature", "nunique"),
    )
    risk.to_csv(DEEP / "label_relevance_weighted_2025_shift.csv", index=False, encoding="utf-8-sig")
    risk_month.to_csv(DEEP / "label_relevance_weighted_2025_shift_by_month.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(risk_month["month"], risk_month["exposure_risk"], color="tab:red", alpha=0.75)
    ax.set_title("2025 exposure risk weighted by label-relevant wind correlations")
    ax.set_xlabel("month")
    ax.set_ylabel("sum(abs shift z) * max feature-label corr")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "17_label_relevance_weighted_2025_exposure.png")
    plt.close(fig)

    top = risk.sort_values("weighted_abs_shift", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = top["source"] + "::" + top["feature"] + "::m" + top["month"].astype(str)
    signed = top["z_delta_vs_2024"] * top["spearman_corr"].clip(lower=0)
    colors = np.where(signed >= 0, "tab:red", "tab:blue")
    ax.barh(labels[::-1], signed.iloc[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top label-relevance-weighted 2025 weather shifts")
    ax.set_xlabel("signed shift z * feature-label corr")
    fig.tight_layout()
    fig.savefig(FIG / "18_top_weighted_2025_shift_features.png")
    plt.close(fig)
    return risk


def model_residual_overlay(label_df: pd.DataFrame, scada_tax: pd.DataFrame, nwp_bias_rows: pd.DataFrame) -> pd.DataFrame:
    cand_path = ROOT / "experiments" / "pre_submission_robustness" / "cache" / "candidate_valid_predictions.csv"
    base_path = ROOT / "experiments" / "pre_submission_robustness" / "cache" / "baseline_anchor_valid_predictions.csv"
    if not cand_path.exists() or not base_path.exists():
        return pd.DataFrame()
    cand = read_csv(cand_path)
    base = read_csv(base_path)
    valid = label_df[label_df["year"].eq(2024)].copy().reset_index(drop=True)
    rows = []
    for target, cap in CAPACITY.items():
        x = pd.DataFrame(
            {
                "kst_dtm": valid["kst_dtm"],
                "target": target,
                "month": valid["month"],
                "hour": valid["hour"],
                "actual": valid[target],
                "candidate_pred": cand[target],
                "baseline_pred": base[target],
            }
        )
        x["actual_ratio"] = x["actual"] / cap
        x["candidate_abs_error"] = (x["candidate_pred"] - x["actual"]).abs() / cap
        x["baseline_abs_error"] = (x["baseline_pred"] - x["actual"]).abs() / cap
        x["delta_abs_error"] = x["candidate_abs_error"] - x["baseline_abs_error"]
        x["eligible"] = x["actual_ratio"] >= 0.10
        x["candidate_pass8"] = x["eligible"] & (x["candidate_abs_error"] <= 0.08)
        x["baseline_pass8"] = x["eligible"] & (x["baseline_abs_error"] <= 0.08)
        x["ficr_transition"] = np.select(
            [
                ~x["baseline_pass8"] & x["candidate_pass8"],
                x["baseline_pass8"] & ~x["candidate_pass8"],
                x["baseline_pass8"] & x["candidate_pass8"],
            ],
            ["fail_to_pass", "pass_to_fail", "pass_to_pass"],
            default="fail_to_fail",
        )
        rows.append(x)
    long = pd.concat(rows, ignore_index=True)
    tax = scada_tax[["kst_dtm", "target", "taxonomy", "abs_residual_ratio", "online_rate", "scada_ws_mean"]]
    long = long.merge(tax, on=["kst_dtm", "target"], how="left")
    nb = nwp_bias_rows[nwp_bias_rows["source"].eq("ldaps")][["kst_dtm", "target", "speed_bias_nwp_minus_scada", "abs_dir_diff"]]
    long = long.merge(nb, on=["kst_dtm", "target"], how="left")
    long.to_csv(DEEP / "best_model_residual_overlay_rows.csv", index=False, encoding="utf-8-sig")
    summary = long.groupby(["target", "taxonomy"], as_index=False).agg(
        rows=("target", "size"),
        mean_delta_abs_error=("delta_abs_error", "mean"),
        candidate_abs_error=("candidate_abs_error", "mean"),
        baseline_abs_error=("baseline_abs_error", "mean"),
        pass_to_fail=("ficr_transition", lambda x: int((x == "pass_to_fail").sum())),
        fail_to_pass=("ficr_transition", lambda x: int((x == "fail_to_pass").sum())),
        mean_nwp_scada_abs_speed_bias=("speed_bias_nwp_minus_scada", lambda x: x.abs().mean()),
    )
    summary["net_ficr_pass"] = summary["fail_to_pass"] - summary["pass_to_fail"]
    summary.to_csv(DEEP / "best_model_residual_overlay_by_scada_taxonomy.csv", index=False, encoding="utf-8-sig")

    piv = summary.pivot(index="target", columns="taxonomy", values="mean_delta_abs_error")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(piv.fillna(0).values, aspect="auto", cmap="coolwarm", vmin=-0.004, vmax=0.004)
    ax.set_title("Best candidate delta error by SCADA residual taxonomy")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=35, ha="right")
    fig.colorbar(im, ax=ax, label="candidate abs error - baseline abs error")
    fig.tight_layout()
    fig.savefig(FIG / "19_model_residual_by_scada_taxonomy.png")
    plt.close(fig)

    trans = long.groupby(["target", "ficr_transition"], as_index=False).size()
    piv2 = trans.pivot(index="target", columns="ficr_transition", values="size").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(piv2))
    for col in piv2.columns:
        ax.bar(piv2.index, piv2[col], bottom=bottom, label=col)
        bottom += piv2[col].values
    ax.set_title("Best candidate FiCR pass/fail transitions vs baseline anchor")
    ax.set_ylabel("rows")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "20_model_ficr_transition_overlay.png")
    plt.close(fig)

    long["nwp_bias_bin"] = pd.cut(long["speed_bias_nwp_minus_scada"], [-np.inf, -4, -2, 0, 2, 4, np.inf])
    nwp_summary = long.groupby(["target", "nwp_bias_bin"], observed=True, as_index=False).agg(
        rows=("target", "size"),
        mean_delta_abs_error=("delta_abs_error", "mean"),
        candidate_abs_error=("candidate_abs_error", "mean"),
    )
    nwp_summary.to_csv(DEEP / "best_model_residual_by_nwp_scada_bias.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, g in nwp_summary.groupby("target"):
        ax.plot(g["nwp_bias_bin"].astype(str), g["mean_delta_abs_error"], marker="o", label=target)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Candidate error delta by LDAPS-vs-SCADA wind bias")
    ax.set_xlabel("LDAPS selected wind speed - SCADA wind speed bin")
    ax.set_ylabel("candidate abs error - baseline")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "21_model_residual_by_nwp_scada_bias.png")
    plt.close(fig)
    return long


def write_deep_report(outputs: dict[str, pd.DataFrame]) -> None:
    def table(df: pd.DataFrame, n: int = 12) -> str:
        return df.head(n).to_markdown(index=False) if df is not None and not df.empty else "No rows."

    scada_summary = read_csv(DEEP / "scada_label_residual_taxonomy_summary.csv")
    bias = read_csv(DEEP / "nwp_scada_wind_bias_monthly.csv")
    top_bias = bias.sort_values("mae_speed_bias", ascending=False)
    exposure_month = read_csv(DEEP / "label_relevance_weighted_2025_shift_by_month.csv").sort_values("exposure_risk", ascending=False)
    overlay = read_csv(DEEP / "best_model_residual_overlay_by_scada_taxonomy.csv")
    directional = read_csv(DEEP / "direction_conditional_ldaps_top_grid.csv")
    body = f"""## Deep-Dive EDA Completion Update — 2026-08-09

This update extends the first atlas with the missing deep-dive EDA: SCADA residual taxonomy, NWP-to-SCADA transfer bias, direction-conditioned spatial signal, label-relevance-weighted 2025 exposure, and best-candidate residual overlay.

### New Figures

12. ![SCADA residual taxonomy](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/12_scada_residual_taxonomy.png)
13. ![SCADA residual monthly heatmap](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/13_scada_residual_monthly_heatmap.png)
14. ![NWP SCADA wind bias monthly](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/14_nwp_scada_wind_bias_monthly.png)
15. ![NWP SCADA wind scatter](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/15_nwp_scada_wind_scatter.png)
16a. ![Directional spatial corr group1](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/16_directional_spatial_corr_kpx_group_1.png)
16b. ![Directional spatial corr group2](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/16_directional_spatial_corr_kpx_group_2.png)
16c. ![Directional spatial corr group3](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/16_directional_spatial_corr_kpx_group_3.png)
17. ![Weighted 2025 exposure](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/17_label_relevance_weighted_2025_exposure.png)
18. ![Top weighted 2025 shifts](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/18_top_weighted_2025_shift_features.png)
19. ![Model residual by SCADA taxonomy](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/19_model_residual_by_scada_taxonomy.png)
20. ![Model FiCR transition overlay](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/20_model_ficr_transition_overlay.png)
21. ![Model residual by NWP SCADA bias](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/21_model_residual_by_nwp_scada_bias.png)

### SCADA Residual Taxonomy

{table(scada_summary.sort_values(["target", "rows"], ascending=[True, False]), 18)}

Interpretation:
- Clean SCADA reconstructs labels strongly, but the residual is structured enough to deserve a dedicated event taxonomy.
- `tight_reconstruction` dominates many rows, but `scada_under_label`, `scada_over_label`, and `scada_zero_label_positive` slices identify rows where simple power summation does not explain the label.
- These slices are the best current proxy for operating-state and reporting-state issues because SCADA is the closest observable teacher.

### NWP-to-SCADA Transfer Bias

{table(top_bias[["target", "source", "month", "mae_speed_bias", "mean_speed_bias", "mean_abs_dir_diff", "mean_label_ratio"]], 18)}

Interpretation:
- The deployable problem is not just label regression. NWP wind and turbine-measured SCADA wind differ by month/group/source.
- This explains why SCADA oracle headroom does not automatically transfer to test-time NWP.
- Future features should model NWP-to-SCADA bias regimes before using SCADA-derived power curves as if NWP wind were the same measurement.

### Direction-Conditional Spatial Signal

{table(directional[["target", "dir_sector", "grid_id", "spearman_corr", "rows"]], 24)}

Interpretation:
- The best LDAPS grid is direction-dependent. A single nearest-grid or all-grid dump is too crude.
- Spatial feature engineering should use wind-direction conditioned upstream grids or soft weights, not a static grid list.

### 2025 Exposure Risk

{table(exposure_month, 12)}

Interpretation:
- 2025 exposure risk is concentrated by month, especially where label-relevant wind features shift away from 2024 validation.
- Public-submission decisions should carry this risk surface; a 2024-only win is not enough.

### Best Candidate Residual Overlay

{table(overlay.sort_values(["target", "mean_delta_abs_error"]), 18)}

Interpretation:
- The current candidate must be judged by where it helps/hurts across SCADA residual taxonomy and FiCR transitions, not just total score.
- Error improvements are not uniform across operating-state proxy slices.
- The remaining modeling question is now concrete: identify which SCADA-residual/NWP-bias regimes are deployable from 2025 NWP alone.

### Updated EDA Completion Status

Completed in this issue:
- Column/label/time/spatial/weather/SCADA/FiCR/test-shift atlas.
- SCADA residual taxonomy.
- NWP-to-SCADA wind transfer gap.
- Direction-conditioned LDAPS spatial signal.
- Label-relevance-weighted 2025 exposure risk.
- Best-candidate residual overlay on SCADA taxonomy and NWP-SCADA bias.

Still not fully completed:
- Full human-readable data dictionary for every raw weather column.
- Causal labeling of SCADA residual events using external maintenance/curtailment data, which is not present in the provided dataset.
- A new model trained from this EDA. This issue intentionally stops at EDA and model-error diagnosis.
"""
    (BASE / "issue_update_deep_dive.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    label_df = labels()
    ldaps_tr = add_weather_features(read_csv(DATA / "train" / "ldaps_train.csv"), "ldaps")
    gfs_tr = add_weather_features(read_csv(DATA / "train" / "gfs_train.csv"), "gfs")
    scada = scada_hourly()
    scada_tax = scada_residual_taxonomy(label_df, scada)
    nwp_bias = nwp_scada_bias(label_df, scada, ldaps_tr, gfs_tr)
    directional = direction_conditional_spatial(label_df, ldaps_tr)
    corr = read_csv(OUT / "weather_grid_label_correlations.csv")
    shift = read_csv(OUT / "test2025_vs_valid2024_feature_shift.csv")
    exposure = exposure_risk(corr, shift)
    overlay = model_residual_overlay(label_df, scada_tax, nwp_bias)
    write_deep_report(
        {
            "scada_tax": scada_tax,
            "nwp_bias": nwp_bias,
            "directional": directional,
            "exposure": exposure,
            "overlay": overlay,
        }
    )
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    manifest["deep_dive_figures"] = sorted(p.name for p in FIG.glob("1*.png")) + sorted(p.name for p in FIG.glob("2*.png"))
    manifest["deep_dive_tables"] = sorted(p.name for p in DEEP.glob("*.csv"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"deep_dive": str(DEEP), "figures": manifest["deep_dive_figures"]}, indent=2))


if __name__ == "__main__":
    main()

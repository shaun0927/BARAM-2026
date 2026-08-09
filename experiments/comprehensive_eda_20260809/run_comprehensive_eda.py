from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
OUT = Path(__file__).resolve().parent / "results"
FIG = OUT / "figures"

CAPACITY = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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


def dms_to_decimal(value: object) -> tuple[float, float]:
    if not isinstance(value, str) or " " not in value:
        return (np.nan, np.nan)
    lat_s, lon_s = value.split(" ", 1)

    def one(part: str) -> float:
        deg, rest = part.split("°", 1)
        minute, rest = rest.split("'", 1)
        sec = rest.split('"', 1)[0]
        hemi = part[-1]
        out = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
        return -out if hemi in {"S", "W"} else out

    return (one(lat_s), one(lon_s))


def parse_info() -> pd.DataFrame:
    raw = pd.read_excel(DATA / "info.xlsx", sheet_name="info", header=3)
    raw = raw.drop(columns=[raw.columns[0]], errors="ignore").dropna(how="all")
    raw["KPX그룹"] = raw["KPX그룹"].ffill().astype(int)
    raw["그룹설비용량(MW)"] = raw["그룹설비용량(MW)"].ffill()
    coords = raw["좌표(Google)"].map(dms_to_decimal)
    raw["latitude"] = coords.map(lambda x: x[0])
    raw["longitude"] = coords.map(lambda x: x[1])
    raw.to_csv(OUT / "info_turbine_metadata_parsed.csv", index=False, encoding="utf-8-sig")
    return raw


def schema_audit(files: list[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        df = read_csv(path)
        for col in df.columns:
            s = df[col]
            row = {
                "file": str(path.relative_to(DATA)),
                "column": col,
                "dtype": str(s.dtype),
                "rows": len(df),
                "missing_rate": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(s):
                row.update(
                    {
                        "min": float(s.min(skipna=True)),
                        "p01": float(s.quantile(0.01)),
                        "p50": float(s.quantile(0.50)),
                        "p99": float(s.quantile(0.99)),
                        "max": float(s.max(skipna=True)),
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "column_contract_audit.csv", index=False, encoding="utf-8-sig")

    top = out.sort_values(["missing_rate", "file"], ascending=[False, True]).head(35)
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = top["file"].str.replace("\\\\", "/", regex=False) + "::" + top["column"]
    ax.barh(labels[::-1], top["missing_rate"].iloc[::-1])
    ax.set_title("Column contract audit: highest missing-rate columns")
    ax.set_xlabel("missing rate")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG / "01_column_contract_missingness.png")
    plt.close(fig)
    return out


def label_eda(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels["year"] = labels["kst_dtm"].dt.year
    labels["month"] = labels["kst_dtm"].dt.month
    labels["hour"] = labels["kst_dtm"].dt.hour
    rows = []
    for target, cap in CAPACITY.items():
        s = labels[target]
        ratio = s / cap
        for (year, month), idx in labels.groupby(["year", "month"]).groups.items():
            r = ratio.loc[idx]
            rows.append(
                {
                    "target": target,
                    "year": year,
                    "month": month,
                    "rows": len(r),
                    "missing_rate": float(r.isna().mean()),
                    "mean_ratio": float(r.mean(skipna=True)),
                    "p50_ratio": float(r.quantile(0.50)),
                    "p90_ratio": float(r.quantile(0.90)),
                    "zero_rate": float((r == 0).mean()),
                    "eligible_10pct_rate": float((r >= 0.10).mean()),
                    "high_80pct_rate": float((r >= 0.80).mean()),
                    "capacity_exceed": int((r > 1.0).sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "label_monthly_distribution.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, target in zip(axes, CAPACITY):
        x = summary[summary["target"].eq(target)].copy()
        x["ym"] = x["year"].astype(str) + "-" + x["month"].astype(str).str.zfill(2)
        ax.plot(x["ym"], x["mean_ratio"], marker="o", label="mean actual / capacity")
        ax.plot(x["ym"], x["eligible_10pct_rate"], marker=".", label="eligible >=10% rate")
        ax.plot(x["ym"], x["zero_rate"], marker=".", label="zero rate")
        ax.set_title(target)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].tick_params(axis="x", rotation=70)
    fig.suptitle("Label structure by month: generation level, eligible rows, zero rows")
    fig.tight_layout()
    fig.savefig(FIG / "02_label_monthly_structure.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    long = []
    for target, cap in CAPACITY.items():
        tmp = labels[["kst_dtm", target]].rename(columns={target: "actual"}).copy()
        tmp["target"] = target
        tmp["ratio"] = tmp["actual"] / cap
        long.append(tmp)
    long_df = pd.concat(long, ignore_index=True)
    for target, g in long_df.groupby("target"):
        ax.hist(g["ratio"].dropna().clip(0, 1.2), bins=80, alpha=0.45, density=True, label=target)
    ax.axvline(0.10, color="black", linestyle="--", linewidth=1, label="evaluation threshold 10%")
    ax.axvline(0.80, color="gray", linestyle=":", linewidth=1, label="high generation 80%")
    ax.set_title("Actual generation ratio distribution")
    ax.set_xlabel("actual / capacity")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_label_ratio_distribution.png")
    plt.close(fig)
    return summary


def add_wind_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out["lead_hour"] = (out["forecast_kst_dtm"] - out["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour
    out["wind10_speed"] = np.hypot(out["heightAboveGround_10_10u"], out["heightAboveGround_10_10v"])
    if source == "ldaps":
        out["wind50max_speed"] = np.hypot(out["heightAboveGround_50_50MUmax"], out["heightAboveGround_50_50MVmax"])
        out["wind50min_speed"] = np.hypot(out["heightAboveGround_50_50MUmin"], out["heightAboveGround_50_50MVmin"])
        out["blwind_speed"] = np.hypot(out["heightAboveGround_5_XBLWS"], out["heightAboveGround_5_YBLWS"])
    else:
        out["wind80_speed"] = np.hypot(out["heightAboveGround_80_u"], out["heightAboveGround_80_v"])
        out["wind100_speed"] = np.hypot(out["heightAboveGround_100_100u"], out["heightAboveGround_100_100v"])
        out["pblwind_speed"] = np.hypot(out["planetaryBoundaryLayer_0_u"], out["planetaryBoundaryLayer_0_v"])
        out["wind850_speed"] = np.hypot(out["isobaricInhPa_850_u"], out["isobaricInhPa_850_v"])
        out["wind700_speed"] = np.hypot(out["isobaricInhPa_700_u"], out["isobaricInhPa_700_v"])
        out["wind500_speed"] = np.hypot(out["isobaricInhPa_500_u"], out["isobaricInhPa_500_v"])
    return out


def forecast_contract(ldaps_tr: pd.DataFrame, gfs_tr: pd.DataFrame, ldaps_te: pd.DataFrame, gfs_te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df, expected_grids in [
        ("ldaps_train", ldaps_tr, 16),
        ("gfs_train", gfs_tr, 9),
        ("ldaps_test", ldaps_te, 16),
        ("gfs_test", gfs_te, 9),
    ]:
        per_time = df.groupby("forecast_kst_dtm")["grid_id"].nunique()
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "forecast_times": int(per_time.shape[0]),
                "expected_grids": expected_grids,
                "min_grids_per_time": int(per_time.min()),
                "max_grids_per_time": int(per_time.max()),
                "bad_grid_count_times": int((per_time != expected_grids).sum()),
                "lead_min": float(df["lead_hour"].min()),
                "lead_max": float(df["lead_hour"].max()),
                "lead_unique": ",".join(map(lambda x: str(int(x)), sorted(df["lead_hour"].dropna().unique()))),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "forecast_time_contract_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, df in [("LDAPS train", ldaps_tr), ("LDAPS test", ldaps_te), ("GFS train", gfs_tr), ("GFS test", gfs_te)]:
        vc = df.drop_duplicates(["forecast_kst_dtm"])["lead_hour"].value_counts().sort_index()
        axes[0].plot(vc.index, vc.values, marker="o", label=name)
    axes[0].set_title("Forecast lead-hour support")
    axes[0].set_xlabel("lead hours from data_available to forecast")
    axes[0].set_ylabel("forecast timestamps")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    bad = summary.set_index("dataset")["bad_grid_count_times"]
    axes[1].bar(bad.index, bad.values)
    axes[1].set_title("Grid completeness violations")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("forecast timestamps with wrong grid count")
    fig.tight_layout()
    fig.savefig(FIG / "04_forecast_time_contract.png")
    plt.close(fig)
    return summary


def spatial_eda(info: pd.DataFrame, ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    grids = []
    for source, df in [("ldaps", ldaps), ("gfs", gfs)]:
        g = df.groupby("grid_id")[["latitude", "longitude"]].first().reset_index()
        g["source"] = source
        grids.append(g)
    grid_df = pd.concat(grids, ignore_index=True)
    rows = []
    for _, t in info.iterrows():
        for _, g in grid_df.iterrows():
            km_lat = (float(t["latitude"]) - float(g["latitude"])) * 111.0
            km_lon = (float(t["longitude"]) - float(g["longitude"])) * 88.0
            rows.append(
                {
                    "turbine": f"{t['제작사']}_{int(t['호기']):02d}",
                    "group": int(t["KPX그룹"]),
                    "source": g["source"],
                    "grid_id": int(g["grid_id"]),
                    "distance_km_approx": math.hypot(km_lat, km_lon),
                    "bearing_proxy_dx_km": km_lon,
                    "bearing_proxy_dy_km": km_lat,
                }
            )
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / "turbine_grid_distance_matrix.csv", index=False, encoding="utf-8-sig")
    nearest = dist.sort_values("distance_km_approx").groupby(["turbine", "source"]).head(3)
    nearest.to_csv(OUT / "nearest_grids_by_turbine.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 7))
    for source, g in grid_df.groupby("source"):
        ax.scatter(g["longitude"], g["latitude"], s=90 if source == "ldaps" else 130, marker="s" if source == "ldaps" else "^", label=f"{source} grids")
        for _, r in g.iterrows():
            ax.text(r["longitude"], r["latitude"], f"{source[0].upper()}{int(r['grid_id'])}", fontsize=7)
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for group, g in info.groupby("KPX그룹"):
        ax.scatter(g["longitude"], g["latitude"], s=45, color=colors[int(group)], label=f"group {int(group)} turbines")
        for _, r in g.iterrows():
            ax.text(r["longitude"], r["latitude"], f"G{int(group)}-{int(r['호기'])}", fontsize=7)
    ax.set_title("Spatial topology: turbines vs LDAPS/GFS grids")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best", ncols=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "05_spatial_topology_turbines_grids.png")
    plt.close(fig)
    return nearest


def weather_label_corr(labels: pd.DataFrame, ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wind_cols = {
        "ldaps": ["wind10_speed", "wind50max_speed", "wind50min_speed", "blwind_speed"],
        "gfs": ["wind10_speed", "wind80_speed", "wind100_speed", "pblwind_speed", "wind850_speed", "wind700_speed", "wind500_speed"],
    }
    labels = labels.copy()
    labels["forecast_kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    for source, df in [("ldaps", ldaps), ("gfs", gfs)]:
        for grid_id, g in df.groupby("grid_id"):
            merged = labels.merge(g[["forecast_kst_dtm"] + wind_cols[source]], on="forecast_kst_dtm", how="inner")
            for target, cap in CAPACITY.items():
                actual = merged[target] / cap
                for col in wind_cols[source]:
                    rows.append(
                        {
                            "source": source,
                            "grid_id": int(grid_id),
                            "target": target,
                            "wind_feature": col,
                            "spearman_corr": float(actual.corr(merged[col], method="spearman")),
                            "pearson_corr": float(actual.corr(merged[col], method="pearson")),
                            "n": int(actual.notna().sum()),
                        }
                    )
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "weather_grid_label_correlations.csv", index=False, encoding="utf-8-sig")
    top = corr.sort_values("spearman_corr", ascending=False).groupby(["target", "source"]).head(12)
    top.to_csv(OUT / "top_weather_grid_label_correlations.csv", index=False, encoding="utf-8-sig")

    piv = top.pivot_table(index=["target", "source", "grid_id"], columns="wind_feature", values="spearman_corr", aggfunc="max")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(piv.fillna(0).values, aspect="auto", vmin=0, vmax=0.9, cmap="viridis")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{a}|{b}|g{c}" for a, b, c in piv.index])
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_title("Top weather-grid correlations with label ratio")
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(FIG / "06_weather_grid_label_correlations.png")
    plt.close(fig)
    return corr


def scada_eda(labels: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    vestas = read_csv(DATA / "train" / "scada_vestas_train.csv")
    unison = read_csv(DATA / "train" / "scada_unison_train.csv")
    outputs = []

    def one(df: pd.DataFrame, maker: str, power_prefix: str, groups: dict[str, list[int]], cap_per_turbine_kw10m: float) -> None:
        df = df.copy()
        df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
        df["hour_end"] = df["kst_dtm"].dt.ceil("h")
        for target, nums in groups.items():
            cols = [f"{power_prefix}{i:02d}_power_kw10m" for i in nums]
            raw = df[cols].sum(axis=1, skipna=False)
            clean_parts = []
            for c in cols:
                clean_parts.append(df[c].where(df[c].between(0, cap_per_turbine_kw10m * 1.10)))
            clean = pd.concat(clean_parts, axis=1).sum(axis=1, skipna=False)
            h = pd.DataFrame({"hour_end": df["hour_end"], "raw": raw, "clean": clean}).groupby("hour_end", as_index=False).sum()
            h = h.rename(columns={"hour_end": "kst_dtm"})
            merged = labels[["kst_dtm", target]].merge(h, on="kst_dtm", how="left")
            cap = CAPACITY[target]
            for kind in ["raw", "clean"]:
                ok = merged[[target, kind]].dropna()
                outputs.append(
                    {
                        "maker": maker,
                        "target": target,
                        "kind": kind,
                        "rows": int(len(ok)),
                        "pearson_corr": float(ok[target].corr(ok[kind])),
                        "median_scada_over_label": float((ok[kind] / ok[target].replace(0, np.nan)).median()),
                        "mae_ratio": float(((ok[kind] - ok[target]).abs() / cap).mean()),
                    }
                )
            merged["residual_clean_ratio"] = (merged["clean"] - merged[target]) / cap
            merged.to_csv(OUT / f"scada_reconstruction_{target}.csv", index=False, encoding="utf-8-sig")

    one(vestas, "VESTAS", "vestas_wtg", {"kpx_group_1": list(range(1, 7)), "kpx_group_2": list(range(7, 13))}, 600.0)
    one(unison, "UNISON", "unison_wtg", {"kpx_group_3": list(range(1, 6))}, 700.0)
    summary = pd.DataFrame(outputs)
    summary.to_csv(OUT / "scada_label_reconstruction_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False, sharey=False)
    for ax, target in zip(axes, CAPACITY):
        df = read_csv(OUT / f"scada_reconstruction_{target}.csv")
        ax.scatter(df[target] / CAPACITY[target], df["clean"] / CAPACITY[target], s=4, alpha=0.2)
        ax.plot([0, 1.1], [0, 1.1], color="black", linewidth=1)
        ax.set_title(target)
        ax.set_xlabel("label / capacity")
        ax.set_ylabel("clean SCADA hourly sum / capacity")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.2)
    fig.suptitle("SCADA-to-label reconstruction after 10-minute hourly summation")
    fig.tight_layout()
    fig.savefig(FIG / "07_scada_label_reconstruction.png")
    plt.close(fig)

    # Power curve sample.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    curve_specs = [
        (vestas, "vestas_wtg01", "group1 sample turbine"),
        (vestas, "vestas_wtg07", "group2 sample turbine"),
        (unison, "unison_wtg01", "group3 sample turbine"),
    ]
    for ax, (df, prefix, title) in zip(axes, curve_specs):
        sample = df[[f"{prefix}_ws", f"{prefix}_power_kw10m"]].dropna().sample(min(18000, len(df)), random_state=17)
        ax.scatter(sample[f"{prefix}_ws"], sample[f"{prefix}_power_kw10m"].clip(-100, 900), s=2, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("SCADA wind speed")
        ax.set_ylabel("power_kw10m clipped to [-100,900]")
        ax.grid(alpha=0.2)
    fig.suptitle("SCADA power-curve clouds expose operating-state and outlier regimes")
    fig.tight_layout()
    fig.savefig(FIG / "08_scada_power_curve_clouds.png")
    plt.close(fig)
    return summary


def shift_eda(ldaps_tr: pd.DataFrame, gfs_tr: pd.DataFrame, ldaps_te: pd.DataFrame, gfs_te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_map = {
        "ldaps": ["wind10_speed", "wind50max_speed", "blwind_speed", "etc_0_blh"],
        "gfs": ["wind10_speed", "wind80_speed", "wind100_speed", "pblwind_speed", "wind850_speed", "wind700_speed", "wind500_speed", "surface_0_gust"],
    }
    for source, train, test in [("ldaps", ldaps_tr, ldaps_te), ("gfs", gfs_tr, gfs_te)]:
        train_2024 = train[train["year"].eq(2024)]
        for feat in feature_map[source]:
            if feat not in train.columns:
                continue
            for month in range(1, 13):
                a = train_2024.loc[train_2024["month"].eq(month), feat].dropna()
                b = test.loc[test["month"].eq(month), feat].dropna()
                if len(a) == 0 or len(b) == 0:
                    continue
                rows.append(
                    {
                        "source": source,
                        "feature": feat,
                        "month": month,
                        "train2024_mean": float(a.mean()),
                        "test2025_mean": float(b.mean()),
                        "mean_delta": float(b.mean() - a.mean()),
                        "pooled_std": float(a.std(ddof=0)),
                        "z_delta_vs_2024": float((b.mean() - a.mean()) / (a.std(ddof=0) + 1e-9)),
                        "test_above_train_p90_rate": float((b > a.quantile(0.90)).mean()),
                    }
                )
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT / "test2025_vs_valid2024_feature_shift.csv", index=False, encoding="utf-8-sig")
    top = shift.reindex(shift["z_delta_vs_2024"].abs().sort_values(ascending=False).index).head(30)

    fig, ax = plt.subplots(figsize=(12, 8))
    labels = top["source"] + "::" + top["feature"] + "::m" + top["month"].astype(str)
    colors = np.where(top["z_delta_vs_2024"] >= 0, "tab:red", "tab:blue")
    ax.barh(labels[::-1], top["z_delta_vs_2024"].iloc[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Largest 2025 test vs 2024 validation weather shifts")
    ax.set_xlabel("monthly mean delta in 2024 standard deviations")
    fig.tight_layout()
    fig.savefig(FIG / "09_test2025_weather_shift.png")
    plt.close(fig)
    return shift


def ficr_surface_eda(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels["month"] = labels["kst_dtm"].dt.month
    labels["hour"] = labels["kst_dtm"].dt.hour
    rows = []
    for target, cap in CAPACITY.items():
        actual = labels[target]
        ratio = actual / cap
        for (month, hour), idx in labels.groupby(["month", "hour"]).groups.items():
            r = ratio.loc[idx]
            rows.append(
                {
                    "target": target,
                    "month": month,
                    "hour": hour,
                    "rows": int(len(r)),
                    "eligible_rate": float((r >= 0.10).mean()),
                    "near_eval_threshold_rate": float(r.between(0.08, 0.12).mean()),
                    "near_low_ficr_band_rate": float(r.between(0.02, 0.18).mean()),
                    "mid_power_rate": float(r.between(0.20, 0.80).mean()),
                    "high_power_rate": float((r >= 0.80).mean()),
                    "mean_ratio": float(r.mean(skipna=True)),
                }
            )
    surface = pd.DataFrame(rows)
    surface.to_csv(OUT / "ficr_sensitive_label_surface.csv", index=False, encoding="utf-8-sig")

    piv = surface.pivot_table(index=["target", "month"], columns="hour", values="eligible_rate")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(piv.fillna(0).values, aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_title("Evaluation support surface: actual >= 10% capacity rate")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{t}|m{m}" for t, m in piv.index])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("hour")
    fig.colorbar(im, ax=ax, label="eligible rate")
    fig.tight_layout()
    fig.savefig(FIG / "10_ficr_evaluation_support_surface.png")
    plt.close(fig)

    risk = surface.sort_values("near_eval_threshold_rate", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = risk["target"] + "::m" + risk["month"].astype(str) + "::h" + risk["hour"].astype(str)
    ax.barh(labels[::-1], risk["near_eval_threshold_rate"].iloc[::-1])
    ax.set_title("Rows near the 10% evaluation threshold")
    ax.set_xlabel("rate of actual/capacity in [0.08, 0.12]")
    fig.tight_layout()
    fig.savefig(FIG / "11_ficr_threshold_sensitive_slices.png")
    plt.close(fig)
    return surface


def write_report(tables: dict[str, pd.DataFrame]) -> None:
    def md_table(df: pd.DataFrame, n: int = 10) -> str:
        if df.empty:
            return "No rows."
        return df.head(n).to_markdown(index=False)

    label = tables["label"]
    contract = tables["contract"]
    scada = tables["scada"]
    shift = tables["shift"].reindex(tables["shift"]["z_delta_vs_2024"].abs().sort_values(ascending=False).index)
    corr = tables["corr"].sort_values("spearman_corr", ascending=False)
    nearest = tables["nearest"]
    issue = f"""# Comprehensive EDA Atlas — 2026-08-09

This EDA pass is intentionally dataset-first. It does not promote a model candidate. It documents the column contract, forecast-time contract, spatial topology, label structure, SCADA label-generation evidence, weather-label links, FiCR-sensitive surfaces, and 2025 test shift.

## Figures

1. ![Column contract missingness](figures/01_column_contract_missingness.png)
2. ![Label monthly structure](figures/02_label_monthly_structure.png)
3. ![Label ratio distribution](figures/03_label_ratio_distribution.png)
4. ![Forecast time contract](figures/04_forecast_time_contract.png)
5. ![Spatial topology](figures/05_spatial_topology_turbines_grids.png)
6. ![Weather grid label correlations](figures/06_weather_grid_label_correlations.png)
7. ![SCADA label reconstruction](figures/07_scada_label_reconstruction.png)
8. ![SCADA power curve clouds](figures/08_scada_power_curve_clouds.png)
9. ![2025 weather shift](figures/09_test2025_weather_shift.png)
10. ![FiCR evaluation support](figures/10_ficr_evaluation_support_surface.png)
11. ![FiCR threshold slices](figures/11_ficr_threshold_sensitive_slices.png)

## Main Findings

- EDA was not absent, but it was fragmented. This run produces a single atlas that can be used before further modeling.
- The label surface is highly group/year/month dependent. Group 3 has no 2022 labels and a higher zero/near-zero share; it should not inherit group 1/2 assumptions by default.
- The forecast-time contract is mechanically regular and must be treated as an issue-cycle/lead-hour problem, not a generic row-level time series.
- Spatial topology is a first-class axis: nearest grid, highest-correlation grid, and source-specific grid resolution need to be audited before feature dumping.
- SCADA hourly summation strongly reconstructs labels after treating `power_kw10m` as a 10-minute value. The remaining residual should be the next EDA target because it is the closest observable proxy for label generation and operating state.
- 2025 weather shift is not uniform; some monthly wind regimes are far from 2024 validation. Any public-submission decision should carry a 2025 exposure audit.
- FiCR-sensitive rows are concentrated around eligibility and band boundaries. Model error analysis should be routed through these surfaces rather than through average MAE alone.

## Forecast-Time Contract

{md_table(contract, 20)}

## Label Monthly Distribution: Largest Missing/Edge Slices

{md_table(label.sort_values(["missing_rate", "capacity_exceed"], ascending=False), 12)}

## Top Weather-Label Correlations

{md_table(corr[["target", "source", "grid_id", "wind_feature", "spearman_corr", "n"]], 18)}

## Nearest Grid Samples

{md_table(nearest[["turbine", "group", "source", "grid_id", "distance_km_approx"]], 18)}

## SCADA Reconstruction Summary

{md_table(scada, 10)}

## Largest 2025-vs-2024 Weather Shifts

{md_table(shift[["source", "feature", "month", "train2024_mean", "test2025_mean", "z_delta_vs_2024", "test_above_train_p90_rate"]], 18)}

## Open EDA Work Items

1. Build a column-level data dictionary artifact for every numeric weather column: unit/range/missing/train-test parity/physical transform.
2. Convert SCADA-label reconstruction residuals into event taxonomy: outage, curtailment, forecast miss, aggregation mismatch, and group-level reporting residual.
3. Redo spatial features from topology: group/turbine distance, wind-direction conditioned upstream grids, and source-specific height choices.
4. Create a 2025 exposure risk score using only label-relevant features, not all weather columns equally.
5. Map current best model residuals onto this atlas: label regime, SCADA residual regime, spatial regime, lead-hour, and FiCR boundary.
6. Treat group 3 as a separate data problem until evidence supports shared modeling.

## Files

- `column_contract_audit.csv`
- `forecast_time_contract_summary.csv`
- `info_turbine_metadata_parsed.csv`
- `turbine_grid_distance_matrix.csv`
- `weather_grid_label_correlations.csv`
- `scada_label_reconstruction_summary.csv`
- `test2025_vs_valid2024_feature_shift.csv`
- `ficr_sensitive_label_surface.csv`
"""
    (OUT / "eda_atlas_report.md").write_text(issue, encoding="utf-8")


def main() -> None:
    setup()
    files = [
        DATA / "train" / "train_labels.csv",
        DATA / "train" / "ldaps_train.csv",
        DATA / "train" / "gfs_train.csv",
        DATA / "train" / "scada_vestas_train.csv",
        DATA / "train" / "scada_unison_train.csv",
        DATA / "test" / "ldaps_test.csv",
        DATA / "test" / "gfs_test.csv",
        DATA / "sample_submission.csv",
    ]
    schema = schema_audit(files)
    labels = read_csv(DATA / "train" / "train_labels.csv")
    info = parse_info()
    label_summary = label_eda(labels)

    ldaps_tr = add_wind_features(read_csv(DATA / "train" / "ldaps_train.csv"), "ldaps")
    gfs_tr = add_wind_features(read_csv(DATA / "train" / "gfs_train.csv"), "gfs")
    ldaps_te = add_wind_features(read_csv(DATA / "test" / "ldaps_test.csv"), "ldaps")
    gfs_te = add_wind_features(read_csv(DATA / "test" / "gfs_test.csv"), "gfs")

    contract = forecast_contract(ldaps_tr, gfs_tr, ldaps_te, gfs_te)
    nearest = spatial_eda(info, ldaps_tr, gfs_tr)
    corr = weather_label_corr(labels, ldaps_tr, gfs_tr)
    scada = scada_eda(labels, info)
    shift = shift_eda(ldaps_tr, gfs_tr, ldaps_te, gfs_te)
    ficr = ficr_surface_eda(labels)

    write_report(
        {
            "schema": schema,
            "label": label_summary,
            "contract": contract,
            "nearest": nearest,
            "corr": corr,
            "scada": scada,
            "shift": shift,
            "ficr": ficr,
        }
    )
    manifest = {
        "result_dir": str(OUT),
        "figures": sorted(p.name for p in FIG.glob("*.png")),
        "tables": sorted(p.name for p in OUT.glob("*.csv")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

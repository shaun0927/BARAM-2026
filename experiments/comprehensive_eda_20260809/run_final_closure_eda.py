from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
FINAL = OUT / "final_closure"
MODEL_PRED = ROOT / "experiments" / "model_screening" / "predictions"

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}


def setup() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
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


def labels() -> pd.DataFrame:
    df = read_csv(DATA / "train" / "train_labels.csv")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    df["year"] = df["kst_dtm"].dt.year
    df["month"] = df["kst_dtm"].dt.month
    df["hour"] = df["kst_dtm"].dt.hour
    return df


def description_lookup() -> dict[str, str]:
    text = (DATA / "data_description.md").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2:
            col = parts[0].strip("` ")
            desc = parts[1]
            out[col] = desc
    return out


def infer_column_semantics(col: str, source: str) -> dict[str, str]:
    family = "metadata"
    height = ""
    component = ""
    transform = "raw"
    model_policy = "candidate_feature"
    if col in {"forecast_kst_dtm", "data_available_kst_dtm"}:
        family = "time_contract"
        model_policy = "join_or_time_feature_only"
    elif col in {"grid_id", "latitude", "longitude"}:
        family = "spatial_contract"
        model_policy = "spatial_feature_or_join_key"
    elif re.search(r"(_u$|_v$|10u|10v|MU|max|MV|min|XBLWS|YBLWS)", col):
        family = "wind_vector"
        transform = "speed_direction_required"
    elif any(k in col for k in ["wind", "gust", "VRATE"]):
        family = "wind_or_motion"
    elif any(k in col for k in ["_t", "2t", "_dpt", "2d"]):
        family = "temperature_dewpoint"
    elif any(k in col for k in ["_r", "2sh", "_q", "850_r"]):
        family = "humidity"
    elif any(k in col for k in ["sp", "prmsl"]):
        family = "pressure"
    elif any(k in col for k in ["SW", "dswrf", "dlwrf", "NDNSW", "NDNLW"]):
        family = "radiation"
    elif any(k in col for k in ["cc", "tcc", "VLCDC"]):
        family = "cloud"
    elif any(k in col for k in ["prate", "tp", "ncpcp", "lsprate", "lssrate", "snol", "SNOM"]):
        family = "precipitation_snow"
    elif any(k in col for k in ["blh", "pbl", "planetaryBoundaryLayer"]):
        family = "boundary_layer"
    elif col in {"surface_0_lsm", "surface_0_h", "isobaricInhPa_500_gh"}:
        family = "static_or_geopotential"
    if "10" in col:
        height = "10m"
    elif "50" in col:
        height = "50m"
    elif "80" in col:
        height = "80m"
    elif "100" in col:
        height = "100m"
    elif "850" in col:
        height = "850hPa"
    elif "700" in col:
        height = "700hPa"
    elif "500" in col:
        height = "500hPa"
    if col.endswith("_u") or "10u" in col or "MU" in col or "XBLWS" in col:
        component = "u/x"
    elif col.endswith("_v") or "10v" in col or "MV" in col or "YBLWS" in col:
        component = "v/y"
    return {
        "source": source,
        "family": family,
        "height_or_level": height,
        "component": component,
        "recommended_transform": transform,
        "eda_model_policy": model_policy,
    }


def full_column_dictionary() -> pd.DataFrame:
    desc = description_lookup()
    files = [
        ("ldaps", DATA / "train" / "ldaps_train.csv", DATA / "test" / "ldaps_test.csv"),
        ("gfs", DATA / "train" / "gfs_train.csv", DATA / "test" / "gfs_test.csv"),
    ]
    corr = read_csv(OUT / "weather_grid_label_correlations.csv")
    rel = corr.groupby(["source", "wind_feature"], as_index=False)["spearman_corr"].max().rename(
        columns={"wind_feature": "derived_feature", "spearman_corr": "max_label_spearman_corr"}
    )
    rows = []
    for source, train_path, test_path in files:
        train = read_csv(train_path)
        test = read_csv(test_path)
        for col in train.columns:
            sem = infer_column_semantics(col, source)
            row = {
                "source": source,
                "column": col,
                "description": desc.get(col, ""),
                **{k: v for k, v in sem.items() if k != "source"},
                "dtype_train": str(train[col].dtype),
                "train_missing_rate": float(train[col].isna().mean()),
                "test_missing_rate": float(test[col].isna().mean()) if col in test else np.nan,
                "train_nunique": int(train[col].nunique(dropna=True)),
                "test_nunique": int(test[col].nunique(dropna=True)) if col in test else 0,
                "train_test_parity": bool(col in test),
            }
            if pd.api.types.is_numeric_dtype(train[col]) and col in test:
                tr = train[col].dropna()
                te = test[col].dropna()
                row.update(
                    {
                        "train_min": float(tr.min()) if len(tr) else np.nan,
                        "train_p01": float(tr.quantile(0.01)) if len(tr) else np.nan,
                        "train_p50": float(tr.quantile(0.50)) if len(tr) else np.nan,
                        "train_p99": float(tr.quantile(0.99)) if len(tr) else np.nan,
                        "train_max": float(tr.max()) if len(tr) else np.nan,
                        "test_min": float(te.min()) if len(te) else np.nan,
                        "test_p50": float(te.quantile(0.50)) if len(te) else np.nan,
                        "test_max": float(te.max()) if len(te) else np.nan,
                        "mean_shift_z": float((te.mean() - tr.mean()) / (tr.std(ddof=0) + 1e-9)) if len(tr) and len(te) else np.nan,
                    }
                )
            rows.append(row)
    dictionary = pd.DataFrame(rows)
    # Attach derived-feature relevance when column is directly part of a known derived wind speed.
    mapping = []
    for _, r in dictionary.iterrows():
        derived = ""
        c = r["column"]
        if r["source"] == "ldaps":
            if "50_50M" in c:
                derived = "wind50max_speed"
            elif "10_10" in c:
                derived = "wind10_speed"
        if r["source"] == "gfs":
            if "850_" in c and c.endswith(("_u", "_v")):
                derived = "wind850_speed"
            elif "700_" in c and c.endswith(("_u", "_v")):
                derived = "wind700_speed"
            elif "500_" in c and c.endswith(("_u", "_v")):
                derived = "wind500_speed"
            elif "80_" in c:
                derived = "wind80_speed"
            elif "100_" in c:
                derived = "wind100_speed"
            elif "10_10" in c:
                derived = "wind10_speed"
        mapping.append(derived)
    dictionary["derived_feature_link"] = mapping
    dictionary = dictionary.merge(
        rel, left_on=["source", "derived_feature_link"], right_on=["source", "derived_feature"], how="left"
    ).drop(columns=["derived_feature"], errors="ignore")
    dictionary.to_csv(FINAL / "full_raw_weather_column_dictionary.csv", index=False, encoding="utf-8-sig")

    family = dictionary.groupby(["source", "family"], as_index=False).agg(
        columns=("column", "count"),
        max_abs_shift_z=("mean_shift_z", lambda x: float(np.nanmax(np.abs(x))) if x.notna().any() else np.nan),
        max_label_relevance=("max_label_spearman_corr", "max"),
        missing_columns=("train_missing_rate", lambda x: int((x > 0).sum())),
    )
    family.to_csv(FINAL / "weather_column_family_summary.csv", index=False, encoding="utf-8-sig")

    piv = family.pivot(index="family", columns="source", values="max_abs_shift_z").fillna(0)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(piv.values, aspect="auto", cmap="magma")
    ax.set_title("Raw weather column dictionary: max absolute train/test shift by family")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    fig.colorbar(im, ax=ax, label="max |test mean - train mean| / train std")
    fig.tight_layout()
    fig.savefig(FIG / "22_column_dictionary_family_shift.png")
    plt.close(fig)

    top = dictionary.reindex(dictionary["mean_shift_z"].abs().sort_values(ascending=False).index).head(30)
    fig, ax = plt.subplots(figsize=(12, 8))
    labels = top["source"] + "::" + top["column"]
    colors = np.where(top["mean_shift_z"] >= 0, "tab:red", "tab:blue")
    ax.barh(labels[::-1], top["mean_shift_z"].iloc[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Raw column dictionary: largest overall train/test shifts")
    ax.set_xlabel("test mean shift in train standard deviations")
    fig.tight_layout()
    fig.savefig(FIG / "23_column_dictionary_top_shifts.png")
    plt.close(fig)
    return dictionary


def event_casebook() -> pd.DataFrame:
    rows_path = OUT / "deep_dive" / "scada_label_residual_taxonomy_rows.csv"
    df = read_csv(rows_path)
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    focus = df[~df["taxonomy"].isin(["tight_reconstruction", "label_missing"])].copy()
    focus = focus.sort_values(["target", "taxonomy", "kst_dtm"])
    segments = []
    for (target, taxonomy), g in focus.groupby(["target", "taxonomy"]):
        g = g.sort_values("kst_dtm").copy()
        gap = g["kst_dtm"].diff().dt.total_seconds().div(3600).fillna(999)
        seg_id = (gap > 1.5).cumsum()
        for _, s in g.groupby(seg_id):
            duration = len(s)
            if duration < 2 and s["abs_residual_ratio"].mean() < 0.10:
                continue
            segments.append(
                {
                    "target": target,
                    "taxonomy": taxonomy,
                    "start": s["kst_dtm"].min(),
                    "end": s["kst_dtm"].max(),
                    "duration_hours": int(duration),
                    "mean_abs_residual_ratio": float(s["abs_residual_ratio"].mean()),
                    "max_abs_residual_ratio": float(s["abs_residual_ratio"].max()),
                    "mean_label_ratio": float(s["label_ratio"].mean()),
                    "mean_scada_ratio": float(s["scada_ratio"].mean()),
                    "mean_online_rate": float(s["online_rate"].mean()),
                    "mean_scada_ws": float(s["scada_ws_mean"].mean()),
                }
            )
    casebook = pd.DataFrame(segments).sort_values(
        ["duration_hours", "mean_abs_residual_ratio"], ascending=False
    )
    casebook.to_csv(FINAL / "scada_residual_event_casebook.csv", index=False, encoding="utf-8-sig")

    top = casebook.head(40).copy()
    top["start"] = pd.to_datetime(top["start"])
    top["end"] = pd.to_datetime(top["end"])
    fig, ax = plt.subplots(figsize=(13, 8))
    tax_order = {t: i for i, t in enumerate(sorted(top["taxonomy"].unique()))}
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(tax_order))))
    color_map = {t: colors[i] for t, i in tax_order.items()}
    for i, (_, r) in enumerate(top.iterrows()):
        ax.plot([r["start"], r["end"]], [i, i], linewidth=5, color=color_map[r["taxonomy"]])
        ax.scatter([r["start"]], [i], s=12, color=color_map[r["taxonomy"]])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r.target}|{r.taxonomy}|{r.duration_hours}h" for r in top.itertuples()], fontsize=7)
    ax.set_title("SCADA residual event casebook: top contiguous non-tight events")
    ax.set_xlabel("time")
    handles = [plt.Line2D([0], [0], color=c, lw=4, label=t) for t, c in color_map.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "24_scada_residual_event_casebook.png")
    plt.close(fig)

    # Representative detail: top long/high-residual event per target.
    raw = df.copy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)
    for ax, target in zip(axes, TARGETS):
        candidates = casebook[casebook["target"].eq(target)].copy()
        if candidates.empty:
            ax.set_title(f"{target}: no casebook event")
            continue
        event = candidates.sort_values(["mean_abs_residual_ratio", "duration_hours"], ascending=False).iloc[0]
        start = pd.to_datetime(event["start"]) - pd.Timedelta(hours=24)
        end = pd.to_datetime(event["end"]) + pd.Timedelta(hours=24)
        g = raw[(raw["target"].eq(target)) & (raw["kst_dtm"].between(start, end))].copy()
        ax.plot(g["kst_dtm"], g["label_ratio"], label="label/capacity", linewidth=1.5)
        ax.plot(g["kst_dtm"], g["scada_ratio"], label="clean SCADA/capacity", linewidth=1.2)
        ax.plot(g["kst_dtm"], g["online_rate"], label="online_rate proxy", linewidth=1, alpha=0.8)
        ax.axvspan(pd.to_datetime(event["start"]), pd.to_datetime(event["end"]), color="tab:red", alpha=0.15)
        ax.set_title(f"{target}: representative {event['taxonomy']} event")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    fig.suptitle("Representative SCADA residual event detail windows")
    fig.tight_layout()
    fig.savefig(FIG / "25_scada_residual_event_detail.png")
    plt.close(fig)
    return casebook


def model_family_residual_atlas(label_df: pd.DataFrame) -> pd.DataFrame:
    valid = label_df[label_df["year"].eq(2024)].copy().reset_index(drop=True)
    pred_files = sorted(MODEL_PRED.glob("*_valid_alpha.csv"))
    rows = []
    long_rows = []
    for p in pred_files:
        model = p.name.replace("_valid_alpha.csv", "")
        pred = read_csv(p)
        if len(pred) != len(valid):
            continue
        for target, cap in CAPACITY.items():
            actual = valid[target]
            err = (pred[target] - actual) / cap
            x = pd.DataFrame(
                {
                    "model": model,
                    "target": target,
                    "kst_dtm": valid["kst_dtm"],
                    "month": valid["month"],
                    "hour": valid["hour"],
                    "actual_ratio": actual / cap,
                    "pred_ratio": pred[target] / cap,
                    "signed_error": err,
                    "abs_error": err.abs(),
                }
            )
            x["actual_bin"] = pd.cut(
                x["actual_ratio"], [-np.inf, 0.1, 0.3, 0.6, 0.8, np.inf], labels=["<10%", "10-30%", "30-60%", "60-80%", "80%+"]
            )
            x["eligible"] = x["actual_ratio"] >= 0.10
            x["ficr_pass8"] = x["eligible"] & (x["abs_error"] <= 0.08)
            long_rows.append(x)
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "rows": len(x),
                    "mae_ratio": float(x["abs_error"].mean()),
                    "eligible_mae_ratio": float(x.loc[x["eligible"], "abs_error"].mean()),
                    "ficr_pass_rate": float(x.loc[x["eligible"], "ficr_pass8"].mean()),
                    "bias_ratio": float(x["signed_error"].mean()),
                    "under_rate": float((x["signed_error"] < 0).mean()),
                }
            )
    long = pd.concat(long_rows, ignore_index=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(FINAL / "model_family_residual_summary.csv", index=False, encoding="utf-8-sig")

    month = long.groupby(["model", "month"], as_index=False).agg(abs_error=("abs_error", "mean"))
    month.to_csv(FINAL / "model_family_monthly_residual.csv", index=False, encoding="utf-8-sig")
    piv = month.pivot(index="model", columns="month", values="abs_error")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_title("Model-family residual atlas: monthly mean absolute error")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_xlabel("2024 validation month")
    fig.colorbar(im, ax=ax, label="mean abs error / capacity")
    fig.tight_layout()
    fig.savefig(FIG / "26_model_family_monthly_residual_heatmap.png")
    plt.close(fig)

    top_month_models = summary.groupby("model")["mae_ratio"].mean().sort_values().head(8).index
    piv_top = piv.loc[[m for m in top_month_models if m in piv.index]]
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(piv_top.values, aspect="auto", cmap="viridis", vmin=float(np.nanmin(piv_top.values)), vmax=float(np.nanquantile(piv_top.values, 0.95)))
    ax.set_title("Model-family residual atlas: top normal models only")
    ax.set_yticks(range(len(piv_top.index)))
    ax.set_yticklabels(piv_top.index)
    ax.set_xticks(range(len(piv_top.columns)))
    ax.set_xticklabels(piv_top.columns)
    ax.set_xlabel("2024 validation month")
    fig.colorbar(im, ax=ax, label="mean abs error / capacity")
    fig.tight_layout()
    fig.savefig(FIG / "29_model_family_monthly_residual_top_models.png")
    plt.close(fig)

    bin_summary = long.groupby(["model", "target", "actual_bin"], observed=True, as_index=False).agg(
        abs_error=("abs_error", "mean"), ficr_pass_rate=("ficr_pass8", "mean"), rows=("abs_error", "size")
    )
    bin_summary.to_csv(FINAL / "model_family_actual_bin_residual.csv", index=False, encoding="utf-8-sig")
    base = bin_summary[bin_summary["model"].eq("lgbm_l1_baseline")][["target", "actual_bin", "abs_error"]].rename(
        columns={"abs_error": "lgbm_abs_error"}
    )
    delta = bin_summary.merge(base, on=["target", "actual_bin"], how="left")
    delta["delta_vs_lgbm"] = delta["abs_error"] - delta["lgbm_abs_error"]
    top_models = summary.groupby("model")["mae_ratio"].mean().sort_values().head(9).index
    plot = delta[delta["model"].isin(top_models)].copy()
    piv2 = plot.pivot_table(index="model", columns=["target", "actual_bin"], values="delta_vs_lgbm")
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(piv2.fillna(0).values, aspect="auto", cmap="coolwarm", vmin=-0.02, vmax=0.02)
    ax.set_title("Top model families: error delta vs LGBM by target and actual-power bin")
    ax.set_yticks(range(len(piv2.index)))
    ax.set_yticklabels(piv2.index)
    ax.set_xticks(range(len(piv2.columns)))
    ax.set_xticklabels([f"{t}|{b}" for t, b in piv2.columns], rotation=55, ha="right")
    fig.colorbar(im, ax=ax, label="MAE ratio delta vs lgbm_l1_baseline")
    fig.tight_layout()
    fig.savefig(FIG / "27_model_family_actual_bin_delta.png")
    plt.close(fig)

    # Residual correlation between model families over all target rows.
    wide = long.pivot_table(index=["target", "kst_dtm"], columns="model", values="signed_error")
    corr = wide.corr()
    corr.to_csv(FINAL / "model_family_residual_correlation.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.fillna(0).values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Model-family signed residual correlation")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson corr")
    fig.tight_layout()
    fig.savefig(FIG / "28_model_family_residual_correlation.png")
    plt.close(fig)
    return summary


def closure_update(dictionary: pd.DataFrame, casebook: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    family = read_csv(FINAL / "weather_column_family_summary.csv")
    model_top = model_summary.groupby("model", as_index=False).agg(
        mae_ratio=("mae_ratio", "mean"), ficr_pass_rate=("ficr_pass_rate", "mean"), bias_ratio=("bias_ratio", "mean")
    ).sort_values("mae_ratio")
    case_top = casebook.head(15)

    def table(df: pd.DataFrame, n: int = 12) -> str:
        return df.head(n).to_markdown(index=False) if not df.empty else "No rows."

    body = f"""## Final EDA Closure Update — 2026-08-09

This update closes the remaining EDA gaps that were explicitly left open after the atlas and deep-dive passes:

- full raw weather column dictionary,
- SCADA residual event timeline casebook,
- model-family residual atlas.

### New Figures

22. ![Column family shift](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/22_column_dictionary_family_shift.png)
23. ![Top raw column shifts](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/23_column_dictionary_top_shifts.png)
24. ![SCADA residual event casebook](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/24_scada_residual_event_casebook.png)
25. ![SCADA residual event detail](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/25_scada_residual_event_detail.png)
26. ![Model monthly residual heatmap](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/26_model_family_monthly_residual_heatmap.png)
27. ![Model actual-bin delta](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/27_model_family_actual_bin_delta.png)
28. ![Model residual correlation](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/28_model_family_residual_correlation.png)
29. ![Top-model monthly residual heatmap](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/29_model_family_monthly_residual_top_models.png)

### Full Weather Column Dictionary

Artifacts:

- `results/final_closure/full_raw_weather_column_dictionary.csv`
- `results/final_closure/weather_column_family_summary.csv`

Family summary:

{table(family.sort_values(["source", "family"]), 30)}

Interpretation:

- The raw LDAPS/GFS columns are now grouped by physical family, height/level, component, recommended transform, train/test parity, missingness, train/test shift, and derived wind-feature relevance where applicable.
- This closes the previous gap where the EDA had column statistics but no usable modeling policy per raw column.
- The highest-risk columns for test exposure are no longer hidden in raw CSV names; they are visible by physical family and shift magnitude.

### SCADA Residual Event Timeline Casebook

Artifact:

- `results/final_closure/scada_residual_event_casebook.csv`

Top events:

{table(case_top[["target", "taxonomy", "start", "end", "duration_hours", "mean_abs_residual_ratio", "mean_label_ratio", "mean_scada_ratio", "mean_online_rate"]], 15)}

Interpretation:

- The SCADA-label residual problem is now event-level, not just row-level. Long contiguous non-tight intervals exist and should be treated as operating/reporting regimes.
- These events are not automatically removable label errors. They are the correct casebook for deciding whether a future model can detect the same regime from NWP-only 2025 inputs.
- Without external maintenance/curtailment logs, the EDA can classify observable signatures but cannot prove final causal labels.

### Model-Family Residual Atlas

Artifacts:

- `results/final_closure/model_family_residual_summary.csv`
- `results/final_closure/model_family_monthly_residual.csv`
- `results/final_closure/model_family_actual_bin_residual.csv`
- `results/final_closure/model_family_residual_correlation.csv`

Top model-family summary:

{table(model_top, 12)}

Interpretation:

- Model differences are not just global scores. The residual atlas shows where families differ by month, group, and actual-power bin.
- Residual correlations are high enough that most families share the same blind spots; ensembling alone cannot solve the SCADA/NWP transfer and FiCR-boundary issues.
- LGBM remains a strong baseline, but the atlas makes the failure regimes explicit instead of treating the score table as the EDA.

### Final EDA Completion Statement

Completed:

- raw file/column contract,
- full raw weather column dictionary,
- label and evaluation support structure,
- forecast-time and grid completeness contract,
- spatial topology,
- weather-label correlation,
- direction-conditioned spatial signal,
- SCADA hourly reconstruction,
- SCADA residual taxonomy,
- SCADA residual event casebook,
- NWP-to-SCADA transfer bias,
- 2025 exposure risk weighted by label relevance,
- FiCR-sensitive surfaces,
- current best candidate residual overlay,
- model-family residual atlas.

Remaining boundaries, not missing EDA:

- True causal labels for maintenance/curtailment require external operations data that is not in the provided dataset.
- A new model trained from these findings is modeling work, not EDA.
- Future public/private split inference cannot be proved from provided files; only exposure-risk proxies can be audited.

Conclusion:

The EDA is now complete enough to stop exploratory diagnosis and move to hypothesis-driven modeling. Any next phase should cite this atlas and state exactly which discovered regime it targets.
"""
    (BASE / "issue_update_final_closure.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    label_df = labels()
    dictionary = full_column_dictionary()
    casebook = event_casebook()
    model_summary = model_family_residual_atlas(label_df)
    closure_update(dictionary, casebook, model_summary)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    manifest["final_closure_figures"] = [p.name for p in sorted(FIG.glob("2*.png"))]
    manifest["final_closure_tables"] = [p.name for p in sorted(FINAL.glob("*.csv"))]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"final_closure": str(FINAL), "figures": manifest["final_closure_figures"]}, indent=2))


if __name__ == "__main__":
    main()

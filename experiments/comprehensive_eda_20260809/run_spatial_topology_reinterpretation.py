from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
RES = BASE / "results"
DEEP = RES / "deep_dive"
FIG = RES / "figures"

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
TARGET_GROUP = {"kpx_group_1": 1, "kpx_group_2": 2, "kpx_group_3": 3}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out_dir = RES / "spatial_reinterpretation"
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_csv(RES / "info_turbine_metadata_parsed.csv")
    nearest = read_csv(RES / "nearest_grids_by_turbine.csv")
    corr = read_csv(RES / "weather_grid_label_correlations.csv")
    dir_top = read_csv(DEEP / "direction_conditional_ldaps_top_grid.csv")
    ldaps = read_csv(DATA / "train" / "ldaps_train.csv")
    gfs = read_csv(DATA / "train" / "gfs_train.csv")

    for frame in [ldaps, gfs]:
        frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])

    ldaps_grid = ldaps.groupby("grid_id", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
    gfs_grid = gfs.groupby("grid_id", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first"))

    nearest_ranked = nearest.sort_values(["turbine", "source", "distance_km_approx"]).copy()
    nearest_ranked["nearest_rank"] = nearest_ranked.groupby(["turbine", "source"]).cumcount() + 1
    nearest_rank1 = nearest_ranked[nearest_ranked["nearest_rank"].eq(1)].copy()
    nearest_ldaps = nearest_rank1[nearest_rank1["source"].eq("ldaps")].copy()

    nearest_mode = (
        nearest_ldaps.groupby(["group", "grid_id"])
        .size()
        .rename("nearest_turbines")
        .reset_index()
        .sort_values(["group", "nearest_turbines", "grid_id"], ascending=[True, False, True])
    )
    nearest_mode.to_csv(out_dir / "nearest_ldaps_grid_counts_by_group.csv", index=False, encoding="utf-8-sig")

    corr_ldaps = corr[corr["source"].eq("ldaps")].copy()
    corr_ldaps = corr_ldaps[corr_ldaps["wind_feature"].eq("wind50max_speed")].copy()
    static_top = (
        corr_ldaps.sort_values(["target", "spearman_corr"], ascending=[True, False])
        .groupby("target")
        .head(5)
        .copy()
    )
    static_top.to_csv(out_dir / "static_top_ldaps_wind50max_corr_by_target.csv", index=False, encoding="utf-8-sig")

    nearest_summary_rows = []
    for target, group in TARGET_GROUP.items():
        group_counts = nearest_mode[nearest_mode["group"].eq(group)].copy()
        nearest_grid = int(group_counts.iloc[0]["grid_id"])
        nearest_turbines = int(group_counts.iloc[0]["nearest_turbines"])
        top_row = static_top[static_top["target"].eq(target)].iloc[0]
        top_grid = int(top_row["grid_id"])
        top_corr = float(top_row["spearman_corr"])
        nearest_corr_rows = corr_ldaps[(corr_ldaps["target"].eq(target)) & (corr_ldaps["grid_id"].eq(nearest_grid))]
        nearest_corr = float(nearest_corr_rows["spearman_corr"].iloc[0]) if len(nearest_corr_rows) else np.nan
        direction_rows = dir_top[dir_top["target"].eq(target)]
        sector_count = int(direction_rows["dir_sector"].nunique())
        unique_direction_top = int(direction_rows["grid_id"].nunique())
        sectors_equal_nearest = int(direction_rows["grid_id"].eq(nearest_grid).sum())
        sectors_equal_static_top = int(direction_rows["grid_id"].eq(top_grid).sum())
        nearest_summary_rows.append(
            {
                "target": target,
                "group": group,
                "modal_nearest_ldaps_grid": nearest_grid,
                "modal_nearest_turbines": nearest_turbines,
                "static_top_corr_grid": top_grid,
                "static_top_corr": top_corr,
                "modal_nearest_grid_corr": nearest_corr,
                "corr_gap_static_minus_nearest": top_corr - nearest_corr,
                "direction_sector_count": sector_count,
                "unique_direction_top_grids": unique_direction_top,
                "sectors_where_direction_top_equals_nearest": sectors_equal_nearest,
                "sectors_where_direction_top_equals_static_top": sectors_equal_static_top,
            }
        )
    summary = pd.DataFrame(nearest_summary_rows)
    summary.to_csv(out_dir / "spatial_topology_reinterpretation_summary.csv", index=False, encoding="utf-8-sig")

    dir_counts = (
        dir_top.groupby(["target", "grid_id"])
        .agg(sectors=("dir_sector", "count"), rows=("rows", "sum"), mean_spearman=("spearman_corr", "mean"))
        .reset_index()
        .sort_values(["target", "sectors", "mean_spearman"], ascending=[True, False, False])
    )
    dir_counts.to_csv(out_dir / "directional_top_grid_counts_by_target.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 9, "axes.titlesize": 11})
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.05, 1.05], hspace=0.34, wspace=0.34)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_dir = fig.add_subplot(gs[1, 0])
    ax_sum = fig.add_subplot(gs[1, 1])

    group_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    for group, g in info.groupby("KPX그룹"):
        ax_map.scatter(g["longitude"], g["latitude"], s=42, label=f"group {group} turbines", color=group_colors[group], alpha=0.9)
    ax_map.scatter(ldaps_grid["longitude"], ldaps_grid["latitude"], marker="s", s=95, facecolors="none", edgecolors="black", label="LDAPS grid")
    ax_map.scatter(gfs_grid["longitude"], gfs_grid["latitude"], marker="^", s=95, facecolors="none", edgecolors="gray", label="GFS grid")
    for _, r in ldaps_grid.iterrows():
        ax_map.text(r["longitude"], r["latitude"], str(int(r["grid_id"])), fontsize=8, ha="center", va="center")
    for _, r in gfs_grid.iterrows():
        ax_map.text(r["longitude"], r["latitude"], f"G{int(r['grid_id'])}", fontsize=7, ha="center", va="center", color="gray")
    for target, group in TARGET_GROUP.items():
        row = summary[summary["target"].eq(target)].iloc[0]
        grid_id = int(row["static_top_corr_grid"])
        grid = ldaps_grid[ldaps_grid["grid_id"].eq(grid_id)].iloc[0]
        centroid = info[info["KPX그룹"].eq(group)][["longitude", "latitude"]].mean()
        ax_map.plot([centroid["longitude"], grid["longitude"]], [centroid["latitude"], grid["latitude"]], color=group_colors[group], linestyle=":", linewidth=1.6)
        ax_map.text(grid["longitude"], grid["latitude"] + 0.003, f"{target} corr-top", color=group_colors[group], fontsize=7, ha="center")
    ax_map.set_title("Geometry only: turbines, NWP grids, and static corr-top grid")
    ax_map.set_xlabel("longitude")
    ax_map.set_ylabel("latitude")
    ax_map.legend(loc="best", fontsize=7)

    pivot_corr = corr_ldaps.pivot_table(index="target", columns="grid_id", values="spearman_corr", aggfunc="max").loc[TARGETS]
    im = ax_corr.imshow(pivot_corr.values, aspect="auto", cmap="viridis", vmin=0.65, vmax=0.84)
    ax_corr.set_xticks(range(len(pivot_corr.columns)))
    ax_corr.set_xticklabels([str(int(c)) for c in pivot_corr.columns])
    ax_corr.set_yticks(range(len(pivot_corr.index)))
    ax_corr.set_yticklabels(pivot_corr.index)
    ax_corr.set_xlabel("LDAPS grid_id")
    ax_corr.set_title("Static label correlation: nearest grid is not necessarily best")
    for i, target in enumerate(pivot_corr.index):
        group = TARGET_GROUP[target]
        nearest_grid = int(summary.loc[summary["target"].eq(target), "modal_nearest_ldaps_grid"].iloc[0])
        top_grid = int(summary.loc[summary["target"].eq(target), "static_top_corr_grid"].iloc[0])
        for j, grid_id in enumerate(pivot_corr.columns):
            value = pivot_corr.iloc[i, j]
            mark = ""
            if int(grid_id) == nearest_grid:
                mark += "N"
            if int(grid_id) == top_grid:
                mark += "T"
            ax_corr.text(j, i, f"{value:.3f}\n{mark}", ha="center", va="center", fontsize=7, color="white" if value > 0.75 else "black")
    fig.colorbar(im, ax=ax_corr, label="Spearman corr")

    sector_order = ["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"]
    dir_mat = dir_top.copy()
    dir_mat["dir_sector"] = pd.Categorical(dir_mat["dir_sector"], categories=sector_order, ordered=True)
    grid_piv = dir_mat.pivot(index="target", columns="dir_sector", values="grid_id").loc[TARGETS, sector_order]
    corr_piv = dir_mat.pivot(index="target", columns="dir_sector", values="spearman_corr").loc[TARGETS, sector_order]
    im2 = ax_dir.imshow(grid_piv.values.astype(float), aspect="auto", cmap="tab20", vmin=1, vmax=16)
    ax_dir.set_xticks(range(len(sector_order)))
    ax_dir.set_xticklabels(sector_order, rotation=30, ha="right")
    ax_dir.set_yticks(range(len(TARGETS)))
    ax_dir.set_yticklabels(TARGETS)
    ax_dir.set_title("Direction-conditioned top LDAPS grid: static grid assumption breaks")
    for i in range(grid_piv.shape[0]):
        for j in range(grid_piv.shape[1]):
            ax_dir.text(j, i, f"G{int(grid_piv.iloc[i, j])}\n{corr_piv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im2, ax=ax_dir, label="top grid_id")

    ax_sum.axis("off")
    display = summary[
        [
            "target",
            "modal_nearest_ldaps_grid",
            "static_top_corr_grid",
            "corr_gap_static_minus_nearest",
            "unique_direction_top_grids",
            "sectors_where_direction_top_equals_nearest",
        ]
    ].copy()
    display["corr_gap_static_minus_nearest"] = display["corr_gap_static_minus_nearest"].map(lambda x: f"{x:+.3f}")
    display = display.rename(
        columns={
            "target": "target",
            "modal_nearest_ldaps_grid": "nearest\nmode",
            "static_top_corr_grid": "static\ncorr top",
            "corr_gap_static_minus_nearest": "corr gap\nT-N",
            "unique_direction_top_grids": "dir-top\nunique",
            "sectors_where_direction_top_equals_nearest": "dir sectors\n=nearest",
        }
    )
    table = ax_sum.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.14, 0.16, 0.14, 0.16, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    ax_sum.set_title("Interpretation guardrail: topology is candidate geometry, not a fixed-grid proof", pad=16)

    fig.suptitle("Spatial Topology Reinterpretation: separate geometry, static correlation, and directional signal", fontsize=14)
    fig.savefig(FIG / "30_spatial_topology_reinterpretation.png", bbox_inches="tight")
    plt.close(fig)

    body = f"""## Spatial Topology Reinterpretation — geometry is not the same as predictive grid choice

This update clarifies section 4 of the EDA atlas. The original spatial topology figure is useful, but it should be read as **geometry only**. It does not prove that nearest-grid features are sufficient.

![Spatial topology reinterpretation](https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/comprehensive_eda_20260809/results/figures/30_spatial_topology_reinterpretation.png)

### What the original topology plot can and cannot prove

It can prove:

- GFS is a coarse 9-grid envelope around the wind farms.
- LDAPS is a denser 16-grid local field.
- The three KPX groups sit across multiple nearby LDAPS cells, not inside one obvious universal cell.

It cannot prove:

- that the nearest grid is the most predictive grid,
- that one static grid per group is sufficient,
- that all-grid feature dumping captures the spatial structure,
- or that spatial topology is useless just because broad spatial feature additions failed.

### Nearest grid vs static correlation grid

{summary.to_markdown(index=False)}

Interpretation:

- The modal nearest LDAPS grid and the static highest-correlation grid are not the same object.
- The static correlation advantage over the modal nearest grid is small in absolute correlation terms, but the identity mismatch matters for feature design.
- A nearest-grid rule is therefore a geometry prior, not an empirical proof.

### Direction-conditioned result

{dir_top[["target", "dir_sector", "grid_id", "rows", "spearman_corr"]].to_markdown(index=False)}

Interpretation:

- The best LDAPS grid changes by wind-direction sector.
- Group 1 uses 6 distinct top grids across 8 sectors, group 2 uses 4, and group 3 uses 6.
- This is the strongest spatial EDA conclusion: the usable spatial signal is **direction-conditioned / upstream-like**, not a single nearest-grid lookup.

### Modeling consequence

The correct next spatial feature is not `nearest_grid_value` and not a blind dump of every grid column. It should be one of:

1. direction-conditioned top-grid features,
2. soft upstream weighting from turbine centroids and wind vector,
3. group-specific spatial pooling that keeps VESTAS group 1/2 and UNISON group 3 separate,
4. residual tests showing whether the spatial feature reduces current-anchor error, not only label correlation.

This update narrows the EDA conclusion: section 4 should be read as **spatial feature design guidance**, not as proof that topology alone explains score movement.
"""
    (BASE / "issue_update_spatial_reinterpretation.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()

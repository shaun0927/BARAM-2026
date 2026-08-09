from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
PREV = ROOT / "experiments" / "comprehensive_eda_20260809" / "results"
DEEP = PREV / "deep_dive"
OUT = BASE / "results"
FIG = OUT / "figures"

CAPACITY = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}
TARGETS = list(CAPACITY)
RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/score_driven_eda_20260809/results/figures"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 170,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name)
    plt.close()


def label_regime(ratio: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.0, 0.01, 0.08, 0.12, 0.50, 0.80, np.inf]
    labels = [
        "zero",
        "near_zero_0_1pct",
        "low_1_8pct",
        "ficr_boundary_8_12pct",
        "mid_12_50pct",
        "high_50_80pct",
        "very_high_80pct_plus",
    ]
    return pd.cut(ratio, bins=bins, labels=labels, include_lowest=True).astype("object").fillna("missing")


def add_weather_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour
    out["lead_hour"] = (
        out["forecast_kst_dtm"] - pd.to_datetime(out["data_available_kst_dtm"])
    ).dt.total_seconds() / 3600.0
    u10 = out["heightAboveGround_10_10u"]
    v10 = out["heightAboveGround_10_10v"]
    out["wind10_speed"] = np.hypot(u10, v10)
    if source == "ldaps":
        u = out["heightAboveGround_50_50MUmax"]
        v = out["heightAboveGround_50_50MVmax"]
    else:
        u = out["isobaricInhPa_850_u"]
        v = out["isobaricInhPa_850_v"]
    out["wind_primary_speed"] = np.hypot(u, v)
    out["wind_primary_dir_math"] = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return out


def load_overlay() -> pd.DataFrame:
    df = read_csv(DEEP / "best_model_residual_overlay_rows.csv")
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    numeric = [
        "actual",
        "candidate_pred",
        "baseline_pred",
        "actual_ratio",
        "candidate_abs_error",
        "baseline_abs_error",
        "delta_abs_error",
        "abs_residual_ratio",
        "online_rate",
        "scada_ws_mean",
        "speed_bias_nwp_minus_scada",
        "abs_dir_diff",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label_regime"] = label_regime(df["actual_ratio"])
    df["error_improved"] = df["delta_abs_error"] < 0
    return df


def fig_01_scada_taxonomy_opportunity(overlay: pd.DataFrame) -> pd.DataFrame:
    summary = overlay.groupby(["target", "taxonomy"], as_index=False).agg(
        rows=("target", "size"),
        candidate_abs_error=("candidate_abs_error", "mean"),
        baseline_abs_error=("baseline_abs_error", "mean"),
        delta_abs_error=("delta_abs_error", "mean"),
        improved_rate=("error_improved", "mean"),
        pass_to_fail=("ficr_transition", lambda x: (x == "pass_to_fail").sum()),
        fail_to_pass=("ficr_transition", lambda x: (x == "fail_to_pass").sum()),
    )
    summary["net_ficr_pass"] = summary["fail_to_pass"] - summary["pass_to_fail"]
    summary.to_csv(OUT / "01_scada_taxonomy_score_opportunity.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, target in zip(axes, TARGETS):
        g = summary[summary["target"].eq(target)].copy()
        g = g.sort_values("candidate_abs_error")
        sizes = np.clip(g["rows"] / g["rows"].max() * 900, 80, 900)
        colors = np.where(g["delta_abs_error"] <= 0, "tab:blue", "tab:red")
        ax.scatter(g["candidate_abs_error"], g["taxonomy"], s=sizes, c=colors, alpha=0.72, edgecolor="black", linewidth=0.4)
        for _, r in g.iterrows():
            ax.text(r["candidate_abs_error"], r["taxonomy"], f" {int(r['rows'])}", va="center", fontsize=7)
        ax.axvline(g["candidate_abs_error"].mean(), color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(target)
        ax.set_xlabel("candidate abs error / capacity")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("SCADA-label residual taxonomy")
    fig.suptitle("01. Score opportunity by SCADA residual taxonomy: size, error, and candidate delta")
    savefig("01_scada_taxonomy_score_opportunity.png")
    return summary


def fig_02_label_regime_support_and_error(overlay: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = overlay.groupby(["target", "label_regime"], as_index=False).agg(
        rows=("target", "size"),
        candidate_abs_error=("candidate_abs_error", "mean"),
        baseline_abs_error=("baseline_abs_error", "mean"),
        delta_abs_error=("delta_abs_error", "mean"),
        eligible_rate=("eligible", "mean"),
        pass_to_fail=("ficr_transition", lambda x: (x == "pass_to_fail").sum()),
        fail_to_pass=("ficr_transition", lambda x: (x == "fail_to_pass").sum()),
    )
    support["net_ficr_pass"] = support["fail_to_pass"] - support["pass_to_fail"]
    support.to_csv(OUT / "02_label_regime_score_surface.csv", index=False, encoding="utf-8-sig")

    order = [
        "zero",
        "near_zero_0_1pct",
        "low_1_8pct",
        "ficr_boundary_8_12pct",
        "mid_12_50pct",
        "high_50_80pct",
        "very_high_80pct_plus",
        "missing",
    ]
    piv_rows = support.pivot(index="target", columns="label_regime", values="rows").reindex(columns=order).fillna(0)
    piv_err = support.pivot(index="target", columns="label_regime", values="candidate_abs_error").reindex(columns=order)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    im0 = axes[0].imshow(piv_rows.values, aspect="auto", cmap="Blues")
    axes[0].set_title("row support by label regime")
    axes[0].set_yticks(range(len(piv_rows.index)))
    axes[0].set_yticklabels(piv_rows.index)
    for i in range(piv_rows.shape[0]):
        for j in range(piv_rows.shape[1]):
            axes[0].text(j, i, f"{int(piv_rows.iloc[i, j])}", ha="center", va="center", fontsize=7)
    fig.colorbar(im0, ax=axes[0], label="rows")

    im1 = axes[1].imshow(piv_err.values, aspect="auto", cmap="magma")
    axes[1].set_title("candidate abs error by label regime")
    axes[1].set_yticks(range(len(piv_err.index)))
    axes[1].set_yticklabels(piv_err.index)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=35, ha="right")
    for i in range(piv_err.shape[0]):
        for j in range(piv_err.shape[1]):
            val = piv_err.iloc[i, j]
            if pd.notna(val):
                axes[1].text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="white" if val > 0.13 else "black")
    fig.colorbar(im1, ax=axes[1], label="abs error / capacity")
    fig.suptitle("02. Label regime support and current validation error")
    savefig("02_label_regime_support_and_error.png")
    return support, piv_err


def fig_03_ficr_transition_surface(overlay: pd.DataFrame) -> pd.DataFrame:
    summary = overlay.groupby(["target", "label_regime", "ficr_transition"], as_index=False).size()
    summary.to_csv(OUT / "03_ficr_transition_by_regime.csv", index=False, encoding="utf-8-sig")
    transitions = ["pass_to_pass", "pass_to_fail", "fail_to_pass", "fail_to_fail"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, target in zip(axes, TARGETS):
        g = summary[summary["target"].eq(target)]
        piv = g.pivot(index="label_regime", columns="ficr_transition", values="size").reindex(
            ["near_zero_0_1pct", "low_1_8pct", "ficr_boundary_8_12pct", "mid_12_50pct", "high_50_80pct", "very_high_80pct_plus"],
            columns=transitions,
        ).fillna(0)
        bottom = np.zeros(len(piv))
        for tr in transitions:
            vals = piv[tr].values
            ax.barh(piv.index, vals, left=bottom, label=tr)
            bottom += vals
        ax.set_title(target)
        ax.set_xlabel("validation rows")
        ax.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("03. FiCR transition counts by label regime")
    savefig("03_ficr_transition_surface.png")
    return summary


def fig_04_directional_grid_stability() -> pd.DataFrame:
    top = read_csv(DEEP / "direction_conditional_ldaps_top_grid.csv")
    top["grid_id"] = pd.to_numeric(top["grid_id"], errors="coerce")
    top["spearman_corr"] = pd.to_numeric(top["spearman_corr"], errors="coerce")
    top.to_csv(OUT / "04_directional_top_grid.csv", index=False, encoding="utf-8-sig")
    sectors = ["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    piv_grid = top.pivot(index="target", columns="dir_sector", values="grid_id").reindex(columns=sectors)
    im0 = axes[0].imshow(piv_grid.values, aspect="auto", cmap="tab20")
    axes[0].set_title("best LDAPS grid changes by wind-direction sector")
    axes[0].set_yticks(range(len(piv_grid.index)))
    axes[0].set_yticklabels(piv_grid.index)
    for i in range(piv_grid.shape[0]):
        for j in range(piv_grid.shape[1]):
            val = piv_grid.iloc[i, j]
            if pd.notna(val):
                axes[0].text(j, i, f"{int(val)}", ha="center", va="center", fontsize=8)
    fig.colorbar(im0, ax=axes[0], label="grid id")

    piv_corr = top.pivot(index="target", columns="dir_sector", values="spearman_corr").reindex(columns=sectors)
    im1 = axes[1].imshow(piv_corr.values, aspect="auto", cmap="viridis", vmin=0.55, vmax=0.9)
    axes[1].set_title("best-grid Spearman correlation with label ratio")
    axes[1].set_yticks(range(len(piv_corr.index)))
    axes[1].set_yticklabels(piv_corr.index)
    axes[1].set_xticks(range(len(sectors)))
    axes[1].set_xticklabels(sectors, rotation=30, ha="right")
    for i in range(piv_corr.shape[0]):
        for j in range(piv_corr.shape[1]):
            val = piv_corr.iloc[i, j]
            if pd.notna(val):
                axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im1, ax=axes[1], label="Spearman corr")
    fig.suptitle("04. Direction-conditioned spatial signal is not static")
    savefig("04_directional_grid_stability.png")
    return top


def weather_source_disagreement(overlay: pd.DataFrame) -> pd.DataFrame:
    ldaps = add_weather_features(read_csv(DATA / "train" / "ldaps_train.csv"), "ldaps")
    gfs = add_weather_features(read_csv(DATA / "train" / "gfs_train.csv"), "gfs")
    choices = {
        "kpx_group_1": {"ldaps": 6, "gfs": 5},
        "kpx_group_2": {"ldaps": 6, "gfs": 5},
        "kpx_group_3": {"ldaps": 12, "gfs": 5},
    }
    pieces = []
    for target in TARGETS:
        l = ldaps[ldaps["grid_id"].eq(choices[target]["ldaps"])][["forecast_kst_dtm", "wind_primary_speed", "lead_hour"]].rename(
            columns={"forecast_kst_dtm": "kst_dtm", "wind_primary_speed": "ldaps_speed"}
        )
        g = gfs[gfs["grid_id"].eq(choices[target]["gfs"])][["forecast_kst_dtm", "wind_primary_speed"]].rename(
            columns={"forecast_kst_dtm": "kst_dtm", "wind_primary_speed": "gfs_speed"}
        )
        m = l.merge(g, on="kst_dtm", how="inner")
        m["target"] = target
        pieces.append(m)
    w = pd.concat(pieces, ignore_index=True)
    out = overlay.merge(w, on=["kst_dtm", "target"], how="left")
    out["source_disagreement"] = out["ldaps_speed"] - out["gfs_speed"]
    out["source_disagreement_abs"] = out["source_disagreement"].abs()
    out["source_disagreement_bin"] = pd.cut(out["source_disagreement"], [-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf])
    out["lead_hour"] = pd.to_numeric(out["lead_hour"], errors="coerce")
    return out


def fig_05_source_disagreement_and_lead(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    disagreement = joined.groupby(["target", "source_disagreement_bin"], observed=True, as_index=False).agg(
        rows=("target", "size"),
        candidate_abs_error=("candidate_abs_error", "mean"),
        baseline_abs_error=("baseline_abs_error", "mean"),
        delta_abs_error=("delta_abs_error", "mean"),
        mean_actual_ratio=("actual_ratio", "mean"),
    )
    disagreement.to_csv(OUT / "05_source_disagreement_error.csv", index=False, encoding="utf-8-sig")

    lead = joined.groupby(["target", "lead_hour"], as_index=False).agg(
        rows=("target", "size"),
        candidate_abs_error=("candidate_abs_error", "mean"),
        baseline_abs_error=("baseline_abs_error", "mean"),
        delta_abs_error=("delta_abs_error", "mean"),
        mean_abs_source_disagreement=("source_disagreement_abs", "mean"),
    )
    lead.to_csv(OUT / "06_lead_hour_error_sensitivity.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4))
    for target, g in disagreement.groupby("target"):
        axes[0].plot(g["source_disagreement_bin"].astype(str), g["candidate_abs_error"], marker="o", label=target)
    axes[0].set_title("candidate error by selected LDAPS-GFS wind speed disagreement")
    axes[0].set_xlabel("LDAPS primary wind - GFS primary wind bin")
    axes[0].set_ylabel("abs error / capacity")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for target, g in lead.groupby("target"):
        axes[1].plot(g["lead_hour"], g["candidate_abs_error"], marker="o", label=f"{target} error")
        axes[1].plot(g["lead_hour"], g["mean_abs_source_disagreement"] / 20.0, linestyle="--", alpha=0.55, label=f"{target} disagreement/20")
    axes[1].set_title("lead-hour error sensitivity and source disagreement proxy")
    axes[1].set_xlabel("lead hour")
    axes[1].set_ylabel("error ratio or scaled disagreement")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncols=2, fontsize=7)
    fig.suptitle("05. Source disagreement and lead-hour sensitivity")
    savefig("05_source_disagreement_and_lead.png")
    return disagreement, lead


def fig_06_2025_exposure() -> tuple[pd.DataFrame, pd.DataFrame]:
    month = read_csv(DEEP / "label_relevance_weighted_2025_shift_by_month.csv")
    detail = read_csv(DEEP / "label_relevance_weighted_2025_shift.csv")
    for col in ["exposure_risk", "max_single_shift", "weighted_abs_shift", "abs_z_shift", "label_relevance"]:
        if col in month:
            month[col] = pd.to_numeric(month[col], errors="coerce")
        if col in detail:
            detail[col] = pd.to_numeric(detail[col], errors="coerce")
    month.to_csv(OUT / "07_2025_exposure_by_month.csv", index=False, encoding="utf-8-sig")
    top = detail.sort_values("weighted_abs_shift", ascending=False).head(20)
    top.to_csv(OUT / "08_2025_exposure_top_features.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(month["month"], month["exposure_risk"], color="tab:red", alpha=0.75)
    axes[0].set_title("label-relevance weighted 2025 exposure by month")
    axes[0].set_xlabel("month")
    axes[0].set_ylabel("exposure risk")
    axes[0].grid(axis="y", alpha=0.25)
    labels = top["source"] + "::" + top["feature"] + "::m" + top["month"].astype(str)
    axes[1].barh(labels.iloc[::-1], top["weighted_abs_shift"].iloc[::-1], color="tab:purple", alpha=0.75)
    axes[1].set_title("top shifted label-relevant weather features")
    axes[1].set_xlabel("abs shift z * label relevance")
    fig.suptitle("06. 2025 test exposure risk is concentrated, not uniform")
    savefig("06_2025_exposure_risk.png")
    return month, top


def fig_07_proof_matrix(
    scada_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    directional: pd.DataFrame,
    disagreement: pd.DataFrame,
    lead: pd.DataFrame,
    exposure_month: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "hypothesis": "SCADA residual slices identify score risk",
            "evidence": "candidate error and FiCR transition differ by SCADA residual taxonomy",
            "status": "supported",
            "next_model_action": "train residual guards by taxonomy proxy, but only with NWP-safe predictors",
        },
        {
            "hypothesis": "label regime controls score sensitivity",
            "evidence": "boundary/high-power regimes have different row support, error and FiCR transitions",
            "status": "supported",
            "next_model_action": "evaluate models by label-regime slices, not only total MAE",
        },
        {
            "hypothesis": "best spatial grid is wind-direction dependent",
            "evidence": f"{directional.groupby('target')['grid_id'].nunique().to_dict()} unique top grids by target",
            "status": "supported",
            "next_model_action": "use direction-conditioned upstream grid weights",
        },
        {
            "hypothesis": "LDAPS-GFS disagreement marks error-prone cases",
            "evidence": "candidate error varies across disagreement bins; directionality is not monotonic for all groups",
            "status": "partially_supported",
            "next_model_action": "use disagreement as uncertainty/regime feature, not a hard correction alone",
        },
        {
            "hypothesis": "lead hour should change model behavior",
            "evidence": f"lead-hour candidate error range={lead.groupby('target')['candidate_abs_error'].agg(lambda x: x.max()-x.min()).round(4).to_dict()}",
            "status": "supported",
            "next_model_action": "include lead-hour interactions with source/grid/wind features",
        },
        {
            "hypothesis": "2025 risk is month-concentrated",
            "evidence": f"top exposure months={exposure_month.sort_values('exposure_risk', ascending=False)['month'].head(4).astype(int).tolist()}",
            "status": "supported",
            "next_model_action": "weight validation/model selection by exposed months and features",
        },
    ]
    proof = pd.DataFrame(rows)
    proof.to_csv(OUT / "09_score_insight_proof_matrix.csv", index=False, encoding="utf-8-sig")

    status_score = {"supported": 2, "partially_supported": 1, "not_supported": 0}
    fig, ax = plt.subplots(figsize=(12, 4.8))
    y = np.arange(len(proof))
    vals = proof["status"].map(status_score)
    colors = proof["status"].map({"supported": "tab:blue", "partially_supported": "tab:orange", "not_supported": "tab:red"})
    ax.barh(y, vals, color=colors, alpha=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(proof["hypothesis"])
    ax.set_xlim(0, 2.2)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["not", "partial", "supported"])
    ax.set_title("07. What the score-driven EDA has proven")
    for i, r in proof.iterrows():
        ax.text(vals.iloc[i] + 0.03, i, r["status"], va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    savefig("07_score_insight_proof_matrix.png")
    return proof


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_issue(
    scada_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    ficr: pd.DataFrame,
    directional: pd.DataFrame,
    disagreement: pd.DataFrame,
    lead: pd.DataFrame,
    exposure_month: pd.DataFrame,
    exposure_top: pd.DataFrame,
    proof: pd.DataFrame,
) -> None:
    top_scada = scada_summary.sort_values(["candidate_abs_error", "rows"], ascending=[False, False])
    top_regime = regime_summary.sort_values(["candidate_abs_error", "rows"], ascending=[False, False])
    top_disagreement = disagreement.sort_values(["candidate_abs_error", "rows"], ascending=[False, False])
    top_lead = lead.sort_values(["candidate_abs_error", "rows"], ascending=[False, False])
    exposure_rank = exposure_month.sort_values("exposure_risk", ascending=False)
    directional_counts = directional.groupby("target")["grid_id"].nunique().reset_index(name="unique_top_grids")

    body = f"""# Score-Driven Additional EDA: what must be proven to improve BARAM-2026 score

이 이슈는 Issue #31의 dataset atlas 이후 단계입니다. 목적은 더 많은 EDA 그림을 만드는 것이 아니라, **score를 올릴 수 있는 모델링 insight가 실제 데이터에서 지지되는지**를 검증하는 것입니다.

검증한 질문은 여섯 가지입니다.

1. SCADA residual taxonomy가 score risk slice를 실제로 나누는가?
2. label regime과 FiCR boundary가 현재 error/transition을 지배하는가?
3. wind direction에 따라 유효 LDAPS grid가 달라지는가?
4. LDAPS-GFS source disagreement가 error-prone case를 표시하는가?
5. lead_hour가 error sensitivity를 만든다는 증거가 있는가?
6. 2025 test exposure risk가 특정 month/feature에 집중되는가?

Artifacts:

- script: `experiments/score_driven_eda_20260809/run_score_driven_eda.py`
- figures: `experiments/score_driven_eda_20260809/results/figures/`
- tables: `experiments/score_driven_eda_20260809/results/*.csv`
- source dependency: Issue #31 outputs under `experiments/comprehensive_eda_20260809/results/`

---

## 1. SCADA residual taxonomy: score risk가 나뉘는가?

![SCADA taxonomy opportunity]({RAW}/01_scada_taxonomy_score_opportunity.png)

### 관찰

{md_table(top_scada, 12, ["target", "taxonomy", "rows", "candidate_abs_error", "baseline_abs_error", "delta_abs_error", "improved_rate", "net_ficr_pass"])}

### 해석

- SCADA-label reconstruction이 tight한 row와 그렇지 않은 row의 validation error 수준이 다릅니다.
- `scada_zero_label_positive`, `low_online_label_positive`, `scada_under_label` 같은 slice는 row 수는 작아도 error가 크거나 FiCR transition 방향이 달라 score risk로 볼 수 있습니다.
- 다만 이 taxonomy는 SCADA에서 온 teacher signal입니다. test에는 SCADA가 없으므로, 이 taxonomy 자체를 feature로 쓰는 것이 아니라 **NWP/time/metadata로 이 taxonomy를 예측할 수 있는지**를 다음 모델링에서 증명해야 합니다.

### 타당성

- 타당한 점: actual/candidate/baseline/FICR transition이 붙은 validation row에서 직접 집계했습니다.
- 한계: SCADA는 train-only signal이므로 deployable feature가 아닙니다. 이 분석은 score risk discovery이지 곧바로 submission feature는 아닙니다.

---

## 2. Label regime and FiCR boundary: 어떤 row가 score를 지배하는가?

![Label regime support and error]({RAW}/02_label_regime_support_and_error.png)

![FiCR transition surface]({RAW}/03_ficr_transition_surface.png)

### 관찰

{md_table(top_regime, 15, ["target", "label_regime", "rows", "candidate_abs_error", "baseline_abs_error", "delta_abs_error", "eligible_rate", "net_ficr_pass"])}

### 해석

- label regime별 row support와 error가 균일하지 않습니다. 즉 total MAE만 보고 모델을 고르면 score를 지배하는 구간을 놓칩니다.
- `ficr_boundary_8_12pct`는 이름 그대로 평가 포함/제외와 pass/fail 전환에 민감한 구간입니다.
- high/very-high generation 구간은 row 수가 적더라도 absolute error와 score impact가 커질 수 있으므로 별도 validation slice로 고정해야 합니다.

### 타당성

- 타당한 점: label ratio를 capacity로 정규화하고, 현재 candidate/baseline의 실제 validation error 및 FiCR transition을 같이 봤습니다.
- 한계: 현재 validation은 2024 중심 candidate overlay입니다. 2025 test exposure와 결합해서 validation weighting을 조정해야 합니다.

---

## 3. Direction-conditioned spatial signal: grid 선택은 고정인가?

![Directional grid stability]({RAW}/04_directional_grid_stability.png)

### 관찰

{md_table(directional_counts, 10)}

{md_table(directional.sort_values(["target", "dir_sector"]), 24, ["target", "dir_sector", "grid_id", "spearman_corr", "rows"])}

### 해석

- best LDAPS grid는 target과 wind-direction sector에 따라 달라집니다.
- 따라서 nearest grid 하나 또는 all-grid dump만으로는 공간 정보를 제대로 쓰지 못합니다.
- score 개선 가설은 명확합니다: wind direction별 upstream grid weighting 또는 direction-conditioned grid aggregation을 만들어야 합니다.

### 타당성

- 타당한 점: label ratio와 LDAPS wind50max correlation을 direction sector별로 분해했습니다.
- 한계: correlation 기반이므로 causal proof는 아닙니다. 모델 ablation에서 direction-conditioned grid feature가 실제 validation을 개선하는지 확인해야 합니다.

---

## 4. LDAPS-GFS source disagreement and lead-hour sensitivity

![Source disagreement and lead]({RAW}/05_source_disagreement_and_lead.png)

### 관찰: source disagreement 상위 error 구간

{md_table(top_disagreement, 12, ["target", "source_disagreement_bin", "rows", "candidate_abs_error", "baseline_abs_error", "delta_abs_error", "mean_actual_ratio"])}

### 관찰: lead-hour 상위 error 구간

{md_table(top_lead, 12, ["target", "lead_hour", "rows", "candidate_abs_error", "baseline_abs_error", "delta_abs_error", "mean_abs_source_disagreement"])}

### 해석

- LDAPS와 GFS selected wind speed가 크게 어긋나는 구간에서 error 수준이 달라집니다.
- 단, 모든 group에서 단조롭게 증가하는 형태는 아닙니다. 따라서 source disagreement는 hard correction rule보다 **uncertainty/regime feature**로 쓰는 것이 타당합니다.
- lead_hour별 error 차이도 존재하므로, 이 데이터는 generic hourly time-series가 아니라 forecast lead 문제로 다뤄야 합니다.

### 타당성

- 타당한 점: validation prediction row에 selected LDAPS/GFS wind speed와 lead_hour를 join해 실제 error와 연결했습니다.
- 한계: selected grid는 기존 EDA의 robust choice를 사용했습니다. direction-conditioned grid로 다시 계산하면 source disagreement의 설명력이 바뀔 수 있습니다.

---

## 5. 2025 exposure: validation win이 test에도 유효한가?

![2025 exposure risk]({RAW}/06_2025_exposure_risk.png)

### 관찰

{md_table(exposure_rank, 12)}

### label-relevant shifted features

{md_table(exposure_top, 12, ["source", "feature", "month", "z_delta_vs_2024", "spearman_corr", "weighted_abs_shift"])}

### 해석

- 2025 risk는 모든 달에 균일하지 않고 특정 month에 집중됩니다.
- 중요한 점은 단순 weather shift가 아니라 **label relevance로 가중한 shift**라는 점입니다. label과 약한 feature가 크게 움직이는 것보다, label과 강한 wind feature가 움직이는 것이 score risk에 더 중요합니다.
- validation/model selection은 2024 평균만 볼 것이 아니라, exposed month/feature에 대한 성능을 별도로 확인해야 합니다.

### 타당성

- 타당한 점: weather-label correlation과 2025-vs-2024 shift를 곱해 score relevance가 높은 shift를 우선순위화했습니다.
- 한계: 2025에는 label이 없으므로 exposure는 risk proxy입니다. 실제 score impact는 submission feedback 또는 private split 이후에만 확정됩니다.

---

## 6. Proof matrix: 지금 무엇이 증명됐나?

![Proof matrix]({RAW}/07_score_insight_proof_matrix.png)

{md_table(proof, 10)}

## 결론

이번 추가 EDA의 결론은 다음입니다.

1. **score 개선의 병목은 전체 평균 error가 아니라 slice별 residual입니다.**
2. SCADA residual taxonomy, label regime, FiCR boundary, wind-direction grid, source disagreement, lead_hour, 2025 exposure가 모두 score-relevant한 축으로 확인됐습니다.
3. 즉 다음 phase는 feature dump가 아니라, 아래 네 가지 deployable 가설을 ablation해야 합니다.

Recommended next modeling tests:

- NWP/time/metadata만으로 SCADA residual taxonomy proxy를 예측할 수 있는지 검증.
- direction-conditioned upstream LDAPS grid feature를 static grid feature와 ablation.
- LDAPS-GFS disagreement를 uncertainty/regime feature로 추가해 error spike가 줄어드는지 검증.
- validation metric을 label regime + FiCR boundary + 2025 exposure month로 slice reporting.

중요한 caveat:

- SCADA 자체는 test에 없으므로 direct feature가 아닙니다.
- 2025 exposure는 label 없는 risk proxy입니다.
- correlation 기반 spatial EDA는 feature 설계 후보이지 causal proof가 아닙니다.
- 이 이슈에서 증명한 것은 “어떤 축이 score-relevant한가”이고, “그 축이 leaderboard를 올린다”는 것은 후속 ablation으로 검증해야 합니다.
"""
    (BASE / "issue_body.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    overlay = load_overlay()
    scada_summary = fig_01_scada_taxonomy_opportunity(overlay)
    regime_summary, _ = fig_02_label_regime_support_and_error(overlay)
    ficr = fig_03_ficr_transition_surface(overlay)
    directional = fig_04_directional_grid_stability()
    joined = weather_source_disagreement(overlay)
    disagreement, lead = fig_05_source_disagreement_and_lead(joined)
    exposure_month, exposure_top = fig_06_2025_exposure()
    proof = fig_07_proof_matrix(scada_summary, regime_summary, directional, disagreement, lead, exposure_month)
    write_issue(scada_summary, regime_summary, ficr, directional, disagreement, lead, exposure_month, exposure_top, proof)
    manifest = pd.DataFrame(
        {
            "artifact": [
                "issue_body.md",
                "01_scada_taxonomy_score_opportunity.csv",
                "02_label_regime_score_surface.csv",
                "03_ficr_transition_by_regime.csv",
                "04_directional_top_grid.csv",
                "05_source_disagreement_error.csv",
                "06_lead_hour_error_sensitivity.csv",
                "07_2025_exposure_by_month.csv",
                "08_2025_exposure_top_features.csv",
                "09_score_insight_proof_matrix.csv",
            ]
        }
    )
    manifest.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote score-driven EDA to {BASE}")


if __name__ == "__main__":
    main()

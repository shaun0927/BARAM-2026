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
DEEP = COMP / "deep_dive"
PHASE98_W4 = (
    ROOT
    / "experiments"
    / "autoresearch_harness"
    / "results"
    / "phase98_source_sister_oof_stack"
    / "diagnostic_predictions"
    / "phase98_w4_phase60_lineage.csv"
)

RAW_BASE = (
    "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/"
    "experiments/weather_label_score_insight_eda_20260810/results/figures"
)


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def add_weather_wind(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id"]].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out["lead_hour"] = (out["forecast_kst_dtm"] - out["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    if source == "ldaps":
        u = pd.to_numeric(df["heightAboveGround_50_50MUmax"], errors="coerce")
        v = pd.to_numeric(df["heightAboveGround_50_50MVmax"], errors="coerce")
        u10 = pd.to_numeric(df["heightAboveGround_10_10u"], errors="coerce")
        v10 = pd.to_numeric(df["heightAboveGround_10_10v"], errors="coerce")
        out["wind50max_speed"] = np.hypot(u, v)
        out["wind10_speed"] = np.hypot(u10, v10)
        out["wind50max_dir_deg"] = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    else:
        u = pd.to_numeric(df["isobaricInhPa_850_u"], errors="coerce")
        v = pd.to_numeric(df["isobaricInhPa_850_v"], errors="coerce")
        out["wind850_speed"] = np.hypot(u, v)
    return out


def lower16(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = out[target].clip(0.16 * CAPACITY[target], CAPACITY[target])
    return out


def valid_anchor_frame() -> pd.DataFrame:
    labels = read_labels(DATA)
    valid = labels[labels["year"].eq(2024)].copy()
    valid = valid.rename(columns={"kst_dtm": "forecast_kst_dtm"}).reset_index(drop=True)
    pred = read_csv(PHASE98_W4)
    pred["forecast_kst_dtm"] = pd.to_datetime(pred["forecast_kst_dtm"])
    pred[TARGETS] = lower16(pred[TARGETS])
    merged = valid[["forecast_kst_dtm", "year", "month", "hour", *TARGETS]].merge(
        pred[["forecast_kst_dtm", *TARGETS]],
        on="forecast_kst_dtm",
        how="inner",
        suffixes=("_actual", "_pred"),
    )
    if len(merged) != len(valid):
        raise RuntimeError(f"W4 anchor alignment failed: {len(merged)} vs {len(valid)}")
    metrics = evaluate_predictions(
        merged[[f"{t}_actual" for t in TARGETS]].rename(columns={f"{t}_actual": t for t in TARGETS}),
        merged[[f"{t}_pred" for t in TARGETS]].rename(columns={f"{t}_pred": t for t in TARGETS}),
        merged["forecast_kst_dtm"],
    )
    (OUT / "anchor_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return merged


def weather_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ldaps = add_weather_wind(read_csv(DATA / "train" / "ldaps_train.csv"), "ldaps")
    gfs = add_weather_wind(read_csv(DATA / "train" / "gfs_train.csv"), "gfs")
    return ldaps, gfs


def anchor_long(anchor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": anchor["forecast_kst_dtm"],
                    "month": anchor["month"],
                    "hour": anchor["hour"],
                    "target": target,
                    "actual": anchor[f"{target}_actual"],
                    "pred": anchor[f"{target}_pred"],
                    "actual_ratio": anchor[f"{target}_actual"] / cap,
                    "pred_ratio": anchor[f"{target}_pred"] / cap,
                    "signed_error_ratio": (anchor[f"{target}_pred"] - anchor[f"{target}_actual"]) / cap,
                    "abs_error_ratio": (anchor[f"{target}_pred"] - anchor[f"{target}_actual"]).abs() / cap,
                    "eligible": anchor[f"{target}_actual"] >= 0.10 * cap,
                    "pass8": ((anchor[f"{target}_pred"] - anchor[f"{target}_actual"]).abs() / cap) <= 0.08,
                    "ficr_boundary": ((anchor[f"{target}_pred"] - anchor[f"{target}_actual"]).abs() / cap).between(0.06, 0.10),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def top_weather_by_time(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    ld13 = ldaps[ldaps["grid_id"].eq(13)][["forecast_kst_dtm", "wind50max_speed", "wind10_speed", "wind50max_dir_deg"]].copy()
    gf1 = gfs[gfs["grid_id"].eq(1)][["forecast_kst_dtm", "wind850_speed"]].copy()
    out = ld13.merge(gf1, on="forecast_kst_dtm", how="inner")
    out["ldaps_gfs_wind_disagreement"] = out["wind50max_speed"] - out["wind850_speed"]
    out["dir_sector"] = pd.cut(
        out["wind50max_dir_deg"],
        bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
        labels=["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"],
        include_lowest=True,
        right=False,
    ).astype(str)
    return out


def residual_by_wind_bins(long: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = long.merge(weather, on="forecast_kst_dtm", how="inner")
    df = df[df["eligible"]].copy()
    df["wind_bin"] = pd.qcut(df["wind50max_speed"], q=10, duplicates="drop")
    rows = []
    for (target, wind_bin), g in df.groupby(["target", "wind_bin"], observed=True):
        rows.append(
            {
                "target": target,
                "wind_bin": str(wind_bin),
                "wind_mid": float(g["wind50max_speed"].mean()),
                "rows": int(len(g)),
                "mean_signed_error_ratio": float(g["signed_error_ratio"].mean()),
                "mean_abs_error_ratio": float(g["abs_error_ratio"].mean()),
                "fail8_rate": float((~g["pass8"]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "01_residual_by_ldaps_wind50_bins.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        s = out[out["target"].eq(target)].sort_values("wind_mid")
        ax.bar(s["wind_mid"], s["mean_signed_error_ratio"], width=0.55, alpha=0.75, label="signed error")
        ax.plot(s["wind_mid"], s["mean_abs_error_ratio"], marker="o", color="black", linewidth=1.2, label="abs error")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_xlabel("LDAPS grid13 wind50max speed")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("capacity-normalized error")
    axes[0].legend()
    fig.suptitle("Best-anchor residual vs top correlated wind bins")
    fig.tight_layout()
    fig.savefig(FIG / "01_residual_bias_wind_bins.png")
    plt.close(fig)
    return out


def monthly_correlation_stability(labels: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["forecast_kst_dtm"] = pd.to_datetime(lab["kst_dtm"])
    df = lab.merge(weather[["forecast_kst_dtm", "wind50max_speed", "wind850_speed"]], on="forecast_kst_dtm", how="inner")
    rows = []
    for (year, month), g in df.groupby(["year", "month"]):
        for target in TARGETS:
            actual = g[target] / CAPACITY[target]
            rows.append(
                {
                    "year": int(year),
                    "month": int(month),
                    "target": target,
                    "ldaps13_wind50_spearman": float(actual.corr(g["wind50max_speed"], method="spearman")),
                    "gfs1_wind850_spearman": float(actual.corr(g["wind850_speed"], method="spearman")),
                    "rows": int(actual.notna().sum()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "02_monthly_correlation_stability.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, target in zip(axes, TARGETS):
        s = out[out["target"].eq(target)].copy()
        s["ym"] = s["year"].astype(str) + "-" + s["month"].astype(str).str.zfill(2)
        ax.plot(s["ym"], s["ldaps13_wind50_spearman"], marker="o", label="LDAPS grid13 wind50max")
        ax.plot(s["ym"], s["gfs1_wind850_spearman"], marker=".", label="GFS grid1 wind850")
        ax.axhline(0.75, color="gray", linestyle=":", linewidth=1)
        ax.set_title(target)
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower left")
    axes[-1].tick_params(axis="x", rotation=70)
    fig.suptitle("Monthly stability of top weather-label correlations")
    fig.tight_layout()
    fig.savefig(FIG / "02_monthly_correlation_stability.png")
    plt.close(fig)
    return out


def direction_conditioned_grid() -> pd.DataFrame:
    top = read_csv(DEEP / "direction_conditional_ldaps_top_grid.csv")
    top.to_csv(OUT / "03_direction_conditioned_top_grid.csv", index=False, encoding="utf-8-sig")
    order = ["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, target in zip(axes, TARGETS):
        s = top[top["target"].eq(target)].set_index("dir_sector").reindex(order).reset_index()
        colors = plt.cm.viridis(np.clip((s["spearman_corr"].fillna(0).to_numpy() - 0.4) / 0.45, 0, 1))
        ax.barh(s["dir_sector"], s["grid_id"], color=colors)
        for _, r in s.iterrows():
            if pd.notna(r["grid_id"]):
                ax.text(r["grid_id"] + 0.15, r["dir_sector"], f"g{int(r['grid_id'])} / {r['spearman_corr']:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, 18)
        ax.set_title(target)
        ax.set_xlabel("selected LDAPS grid id")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("wind direction sector")
    fig.suptitle("Direction-conditioned top LDAPS grid is not globally fixed")
    fig.tight_layout()
    fig.savefig(FIG / "03_direction_conditioned_grid_corr.png")
    plt.close(fig)
    return top


def disagreement_residual(long: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = long.merge(weather, on="forecast_kst_dtm", how="inner")
    df = df[df["eligible"]].copy()
    df["disagreement_bin"] = pd.qcut(df["ldaps_gfs_wind_disagreement"], q=8, duplicates="drop")
    rows = []
    for (target, b), g in df.groupby(["target", "disagreement_bin"], observed=True):
        rows.append(
            {
                "target": target,
                "disagreement_bin": str(b),
                "disagreement_mid": float(g["ldaps_gfs_wind_disagreement"].mean()),
                "rows": int(len(g)),
                "mean_abs_error_ratio": float(g["abs_error_ratio"].mean()),
                "mean_signed_error_ratio": float(g["signed_error_ratio"].mean()),
                "fail8_rate": float((~g["pass8"]).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "04_disagreement_residual_risk.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        s = out[out["target"].eq(target)].sort_values("disagreement_mid")
        ax.plot(s["disagreement_mid"], s["mean_abs_error_ratio"], marker="o", label="abs error")
        ax.plot(s["disagreement_mid"], s["fail8_rate"], marker="s", label="fail8 rate")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_xlabel("LDAPS wind50 - GFS wind850")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rate or capacity-normalized error")
    axes[0].legend()
    fig.suptitle("LDAPS-GFS wind disagreement vs anchor error risk")
    fig.tight_layout()
    fig.savefig(FIG / "04_disagreement_residual_risk.png")
    plt.close(fig)
    return out


def group_power_curve(labels: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["forecast_kst_dtm"] = pd.to_datetime(lab["kst_dtm"])
    df = lab.merge(weather[["forecast_kst_dtm", "wind50max_speed"]], on="forecast_kst_dtm", how="inner")
    rows = []
    for target in TARGETS:
        s = df[["forecast_kst_dtm", "wind50max_speed", target]].dropna().copy()
        s["actual_ratio"] = s[target] / CAPACITY[target]
        s["wind_bin"] = pd.qcut(s["wind50max_speed"], q=18, duplicates="drop")
        for b, g in s.groupby("wind_bin", observed=True):
            rows.append(
                {
                    "target": target,
                    "wind_bin": str(b),
                    "wind_mid": float(g["wind50max_speed"].mean()),
                    "rows": int(len(g)),
                    "mean_actual_ratio": float(g["actual_ratio"].mean()),
                    "p10_actual_ratio": float(g["actual_ratio"].quantile(0.10)),
                    "p90_actual_ratio": float(g["actual_ratio"].quantile(0.90)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "05_group_power_curve_by_ldaps_wind.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for target in TARGETS:
        s = out[out["target"].eq(target)].sort_values("wind_mid")
        ax.plot(s["wind_mid"], s["mean_actual_ratio"], marker="o", label=target)
        ax.fill_between(s["wind_mid"], s["p10_actual_ratio"], s["p90_actual_ratio"], alpha=0.12)
    ax.set_title("Group-specific wind-to-power curves differ")
    ax.set_xlabel("LDAPS grid13 wind50max speed")
    ax.set_ylabel("actual / capacity")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "05_group_power_curve_comparison.png")
    plt.close(fig)
    return out


def ficr_boundary_by_wind(long: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = long.merge(weather, on="forecast_kst_dtm", how="inner")
    df = df[df["eligible"]].copy()
    df["wind_bin"] = pd.qcut(df["wind50max_speed"], q=10, duplicates="drop")
    rows = []
    for (target, b), g in df.groupby(["target", "wind_bin"], observed=True):
        rows.append(
            {
                "target": target,
                "wind_bin": str(b),
                "wind_mid": float(g["wind50max_speed"].mean()),
                "rows": int(len(g)),
                "fail8_rate": float((~g["pass8"]).mean()),
                "boundary_6_10_rate": float(g["ficr_boundary"].mean()),
                "near_support_10_16_actual_rate": float(g["actual_ratio"].between(0.10, 0.16).mean()),
                "mean_actual_ratio": float(g["actual_ratio"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "06_ficr_boundary_by_wind.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, target in zip(axes, TARGETS):
        s = out[out["target"].eq(target)].sort_values("wind_mid")
        ax.plot(s["wind_mid"], s["boundary_6_10_rate"], marker="o", label="abs error 6-10%")
        ax.plot(s["wind_mid"], s["fail8_rate"], marker="s", label="fail >8%")
        ax.plot(s["wind_mid"], s["near_support_10_16_actual_rate"], marker=".", label="actual 10-16%")
        ax.set_title(target)
        ax.set_xlabel("LDAPS grid13 wind50max speed")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("row rate")
    axes[0].legend()
    fig.suptitle("FiCR-sensitive rows concentrate by wind regime")
    fig.tight_layout()
    fig.savefig(FIG / "06_ficr_boundary_by_wind.png")
    plt.close(fig)
    return out


def exposure_2025() -> pd.DataFrame:
    src = read_csv(DEEP / "label_relevance_weighted_2025_shift.csv")
    focus = src[src["feature"].isin(["wind50max_speed", "wind850_speed"])].copy()
    focus["weighted_shift_risk"] = focus["weighted_abs_shift"]
    focus.to_csv(OUT / "07_2025_label_relevant_weather_exposure.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, feature in zip(axes, ["wind50max_speed", "wind850_speed"]):
        s = focus[focus["feature"].eq(feature)].sort_values(["source", "month"])
        for source, g in s.groupby("source"):
            ax.plot(g["month"], g["weighted_shift_risk"], marker="o", label=source)
        ax.set_title(feature)
        ax.set_xlabel("2025 test month")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("label-relevance weighted shift risk")
    fig.suptitle("2025 exposure risk is month-specific on label-relevant wind features")
    fig.tight_layout()
    fig.savefig(FIG / "07_2025_wind_exposure_risk.png")
    plt.close(fig)
    return focus


def write_issue_body(tables: dict[str, pd.DataFrame]) -> None:
    residual = tables["residual"]
    corr = tables["monthly_corr"]
    direction = tables["direction"]
    disagreement = tables["disagreement"]
    curve = tables["curve"]
    ficr = tables["ficr"]
    exposure = tables["exposure"]

    high_bias = residual.loc[residual.groupby("target")["mean_abs_error_ratio"].idxmax()][
        ["target", "wind_mid", "mean_abs_error_ratio", "mean_signed_error_ratio", "fail8_rate"]
    ]
    corr_summary = corr.groupby("target").agg(
        ldaps_min=("ldaps13_wind50_spearman", "min"),
        ldaps_median=("ldaps13_wind50_spearman", "median"),
        gfs_min=("gfs1_wind850_spearman", "min"),
        gfs_median=("gfs1_wind850_spearman", "median"),
    ).reset_index()
    direction_switches = direction.groupby("target")["grid_id"].nunique().reset_index(name="unique_top_grids")
    disagreement_summary = disagreement.groupby("target").agg(
        max_fail8=("fail8_rate", "max"),
        min_fail8=("fail8_rate", "min"),
        max_abs_error=("mean_abs_error_ratio", "max"),
    ).reset_index()
    group_curve_tail = curve.sort_values("wind_mid").groupby("target").tail(3).groupby("target").agg(
        high_wind_mean_ratio=("mean_actual_ratio", "mean")
    ).reset_index()
    ficr_summary = ficr.groupby("target").agg(
        max_boundary_rate=("boundary_6_10_rate", "max"),
        max_fail8_rate=("fail8_rate", "max"),
        max_near_support=("near_support_10_16_actual_rate", "max"),
    ).reset_index()
    exposure_top = exposure.sort_values("weighted_shift_risk", ascending=False).head(10)

    def md(df: pd.DataFrame, n: int = 10) -> str:
        return df.head(n).to_markdown(index=False, floatfmt=".6f")

    body = f"""# Weather-label score insight EDA: residual, stability, direction, FiCR, and 2025 exposure

이 이슈는 #31의 `5. Weather-Label Correlations` 후속 분석입니다. #31에서는 LDAPS `wind50max_speed`, GFS `wind850_speed`가 label ratio와 강하게 연결된다는 것을 확인했습니다. 이번 분석의 목적은 한 단계 더 좁혀서 **그 wind signal이 현재 best anchor의 residual과 score surface까지 설명하는지**를 검증하는 것입니다.

기준 anchor는 open issue 기준 current best인 `phase154_lower16`을 W4에서 재구성한 `phase98_w4_phase60_lineage + Lower16`입니다.

Artifacts:

- script: `experiments/weather_label_score_insight_eda_20260810/run_weather_label_score_insight_eda.py`
- results: `experiments/weather_label_score_insight_eda_20260810/results/`
- figures: `experiments/weather_label_score_insight_eda_20260810/results/figures/`

---

## 1. Best-anchor residual vs top wind bins

![Residual bias wind bins]({RAW_BASE}/01_residual_bias_wind_bins.png)

### What this tests

단순 label correlation이 아니라, top wind feature가 **현재 best anchor의 residual**까지 설명하는지 봅니다. LDAPS grid13 `wind50max_speed`를 decile bin으로 나누고, eligible row에서 signed error, absolute error, fail8 rate를 계산했습니다.

### Key table

{md(high_bias)}

### Interpretation

- wind bin에 따라 residual magnitude와 sign이 달라집니다. 따라서 wind feature는 label만 설명하는 것이 아니라 current best의 error surface에도 일부 연결됩니다.
- 그러나 signed error가 모든 high-wind 구간에서 한 방향으로 고정되지는 않습니다. 즉 naive high-wind up/down correction은 위험합니다.
- public probe 실패와 일관되게, 큰 overlay보다 **bin-local, group-local, FiCR-aware correction**만 후보가 될 수 있습니다.

### Validity discussion

이 분석은 current best anchor의 실제 W4 residual을 사용하므로 score 개선과 직접 연결됩니다. 다만 W4 단일 split이므로 이 패턴이 2025 public에 그대로 유지된다고 가정하면 안 됩니다.

---

## 2. Monthly/seasonal correlation stability

![Monthly correlation stability]({RAW_BASE}/02_monthly_correlation_stability.png)

### What this tests

전체 기간 Spearman correlation 상위 feature가 월별/연도별로 안정적인지 검증합니다. `LDAPS grid13 wind50max`와 `GFS grid1 wind850`을 2022-2024 month 단위로 분해했습니다.

### Key table

{md(corr_summary)}

### Interpretation

- median correlation은 높지만, month별 minimum은 훨씬 낮습니다. 즉 “전체 기간 top feature”는 안정적인 global rule이 아닙니다.
- month gate 없이 feature를 강하게 적용하면 특정 월에서 local gain이 public loss로 바뀔 수 있습니다.
- 제출 후보는 월별 correlation stability와 2025 exposure를 같이 통과해야 합니다.

### Validity discussion

Spearman correlation은 monotonic relation만 보는 단변량 진단입니다. label relevance를 빠르게 확인하는 데는 타당하지만, collinearity와 model residual 설명력은 따로 봐야 합니다. 그래서 이 분석은 후보 feature를 채택하는 증거가 아니라 **월별 gate/risk를 정하는 증거**로만 사용해야 합니다.

---

## 3. Wind direction-conditioned grid

![Direction conditioned grid]({RAW_BASE}/03_direction_conditioned_grid_corr.png)

### What this tests

전체 top grid 하나가 모든 풍향에서 유효한지 검증합니다. LDAPS `wind50max_speed` correlation을 direction sector별로 다시 계산해 sector별 top grid를 비교했습니다.

### Key table

{md(direction_switches)}

### Interpretation

- target별 top grid가 direction sector에 따라 바뀝니다. 이는 fixed top-grid feature보다 direction-conditioned spatial aggregation이 더 타당하다는 뜻입니다.
- 특히 group3도 단순히 data-poor group이 아니라 다른 spatial/wind-response structure를 가질 가능성이 큽니다.

### Validity discussion

direction sector별 sample 수가 작아지는 구간이 있으므로 selection noise가 있습니다. 따라서 hard sector top-grid 하나를 그대로 모델에 넣기보다, nearest/upstream weighting 또는 sector-smoothed feature로 구현해야 합니다.

---

## 4. LDAPS-GFS disagreement and residual risk

![Disagreement residual risk]({RAW_BASE}/04_disagreement_residual_risk.png)

### What this tests

LDAPS local wind와 GFS large-scale wind가 불일치할 때 current best error가 커지는지 봅니다. `LDAPS grid13 wind50max - GFS grid1 wind850`을 binning하고 abs error/fail8 rate를 집계했습니다.

### Key table

{md(disagreement_summary)}

### Interpretation

- disagreement regime별 fail8/abs error가 달라집니다. 이는 NWP source disagreement가 uncertainty/risk proxy가 될 수 있음을 시사합니다.
- 다만 disagreement 방향 하나만으로 보정 방향을 정하기엔 충분하지 않습니다. correction feature라기보다 **risk gate 또는 shrink gate**로 쓰는 쪽이 더 안전합니다.

### Validity discussion

두 source는 vertical level과 grid scale이 다르므로 절대 차이의 물리 단위가 완전히 동일하지 않습니다. 따라서 이 분석은 “바람 예보 불일치가 있다”는 risk proxy로 해석해야 하며, 직접적인 physical correction 값으로 해석하면 안 됩니다.

---

## 5. Group-specific wind-to-power curve

![Group power curve]({RAW_BASE}/05_group_power_curve_comparison.png)

### What this tests

같은 wind bin에서 group1/2/3의 actual/capacity response가 같은지 비교합니다. group3가 단순 data-poor group인지, 별도 power curve group인지 확인하는 분석입니다.

### Key table

{md(group_curve_tail)}

### Interpretation

- group별 wind-to-power curve가 동일하지 않습니다. 같은 wind speed에서도 평균 발전 비율과 분산이 다르게 나타납니다.
- group3는 top wind-label correlation이 강하지만, group1/2 정책을 그대로 전이할 대상은 아닙니다.
- score 개선 후보는 group-shared feature보다 group-specific calibration 또는 group3 specialist로 가야 합니다.

### Validity discussion

이 curve는 observed label과 single grid wind로 만든 empirical curve입니다. curtailment, outage, turbine availability를 분리하지 못하므로 “물리 power curve”로 부르면 안 됩니다. 모델링에서는 operating-regime curve로 해석해야 합니다.

---

## 6. FiCR-sensitive rows by wind regime

![FiCR boundary by wind]({RAW_BASE}/06_ficr_boundary_by_wind.png)

### What this tests

score가 nMAE만으로 결정되지 않기 때문에, wind bin별로 abs error 6-10% boundary row, fail8 row, actual 10-16% support-edge row가 어디에 집중되는지 봅니다.

### Key table

{md(ficr_summary)}

### Interpretation

- FiCR-sensitive row 비중은 wind regime별로 다릅니다. 즉 wind feature가 score-relevant하려면 residual뿐 아니라 FiCR boundary row를 설명해야 합니다.
- correction은 abs error 평균을 줄이는 것만으로 충분하지 않습니다. fail-to-pass8보다 pass-to-fail8을 더 많이 만들면 public probe처럼 total score가 하락합니다.
- 따라서 앞으로의 postprocessing은 `wind regime x FiCR margin x group` 단위의 pass/fail audit이 필요합니다.

### Validity discussion

FiCR label은 W4 actual이 있어야 계산할 수 있으므로 public에는 직접 적용할 수 없습니다. 하지만 validation에서 correction 후보를 폐기하는 guardrail로는 매우 타당합니다.

---

## 7. 2025 label-relevant wind exposure

![2025 wind exposure]({RAW_BASE}/07_2025_wind_exposure_risk.png)

### What this tests

2025 test가 top wind feature 기준으로 2024 validation과 얼마나 다른지, label relevance로 가중한 exposure risk를 봅니다.

### Key table

{md(exposure_top)}

### Interpretation

- 2025 exposure risk는 월별로 매우 다릅니다. 특히 특정 wind feature/month에서 shift risk가 커집니다.
- W4에서 좋아 보인 month-gated overlay가 public에서 실패한 이유를 설명합니다. local score-positive slice가 public exposure에서는 같은 의미가 아닐 수 있습니다.

### Validity discussion

2025 label은 없으므로 exposure는 weather-only shift입니다. 따라서 score를 직접 예측하지는 못하지만, public submission 후보의 위험 월을 사전에 표시하는 데는 필요합니다.

---

## Final conclusion

이번 EDA로 증명된 것:

1. top wind feature는 label뿐 아니라 current best residual과도 일부 연결됩니다.
2. 하지만 그 연결은 월별, 풍향별, group별로 불안정합니다.
3. fixed top grid 또는 high-alpha overlay는 위험합니다.
4. group3는 별도 wind-to-power/operating-regime을 가진 group으로 다뤄야 합니다.
5. score 개선은 wind feature 자체보다 **wind-conditioned residual + FiCR transition + 2025 exposure risk**를 동시에 통과해야 합니다.

이번 EDA로 반증된 것:

- “전체 기간 상관이 높은 wind feature를 추가하면 score가 오른다.”
- “LDAPS grid13 같은 global top grid 하나를 고정하면 충분하다.”
- “group3는 label 부족만 해결하면 group1/2 정책을 그대로 전이할 수 있다.”
- “local W4 FiCR gain은 public에서도 안전하다.”

## Recommended next experiments

1. W3/W4 모두에서 `wind bin x group x FiCR margin` correction 후보를 만들고, 월별 손실이 있는 후보는 폐기.
2. direction-conditioned grid는 hard top-grid가 아니라 soft upstream weighting으로 구현.
3. LDAPS-GFS disagreement는 correction이 아니라 uncertainty shrink/gate로 사용.
4. group3는 shared model postprocessing이 아니라 specialist calibration으로 분리.
5. public submission 후보는 반드시 2025 exposure-risk month audit을 통과한 것만 제출.
"""
    (BASE / "issue_body.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    labels = read_labels(DATA)
    anchor = valid_anchor_frame()
    long = anchor_long(anchor)
    ldaps, gfs = weather_frames()
    weather = top_weather_by_time(ldaps, gfs)
    tables = {
        "residual": residual_by_wind_bins(long, weather),
        "monthly_corr": monthly_correlation_stability(labels, weather),
        "direction": direction_conditioned_grid(),
        "disagreement": disagreement_residual(long, weather),
        "curve": group_power_curve(labels, weather),
        "ficr": ficr_boundary_by_wind(long, weather),
        "exposure": exposure_2025(),
    }
    write_issue_body(tables)
    manifest = {
        "experiment": "weather_label_score_insight_eda_20260810",
        "anchor": "phase98_w4_phase60_lineage + lower16, proxy for phase154 local W4 anchor",
        "tables": [str(p.relative_to(BASE)) for p in OUT.glob("*.csv")],
        "figures": [str(p.relative_to(BASE)) for p in FIG.glob("*.png")],
        "issue_body": "issue_body.md",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

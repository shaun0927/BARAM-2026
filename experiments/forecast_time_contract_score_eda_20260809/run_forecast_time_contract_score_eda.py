from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_contract_score_eda_20260809/results/figures"

CAPACITY = {
    "kpx_group_1": 21600.0,
    "kpx_group_2": 21600.0,
    "kpx_group_3": 21000.0,
}
TARGETS = list(CAPACITY)


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


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name)
    plt.close()


def add_weather_features(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out["lead_hour"] = (out["forecast_kst_dtm"] - out["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    out["hour"] = out["forecast_kst_dtm"].dt.hour
    out["issue_date"] = out["data_available_kst_dtm"].dt.date.astype(str)
    out["cycle_hour"] = out["data_available_kst_dtm"].dt.hour
    out["wind10_speed"] = np.hypot(out["heightAboveGround_10_10u"], out["heightAboveGround_10_10v"])
    if source == "ldaps":
        u = out["heightAboveGround_50_50MUmax"]
        v = out["heightAboveGround_50_50MVmax"]
    else:
        u = out["isobaricInhPa_850_u"]
        v = out["isobaricInhPa_850_v"]
    out["wind_primary_speed"] = np.hypot(u, v)
    out["wind_primary_dir"] = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return out


def load_labels() -> pd.DataFrame:
    labels = read_csv(DATA / "train" / "train_labels.csv")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels["year"] = labels["kst_dtm"].dt.year
    labels["month"] = labels["kst_dtm"].dt.month
    labels["hour"] = labels["kst_dtm"].dt.hour
    return labels


def load_overlay() -> pd.DataFrame:
    pred_dir = ROOT / "experiments" / "pre_submission_robustness" / "cache"
    cand = read_csv(pred_dir / "candidate_valid_predictions.csv")
    base = read_csv(pred_dir / "baseline_anchor_valid_predictions.csv")
    labels = load_labels()
    valid = labels[labels["year"].eq(2024)].copy().reset_index(drop=True)
    rows = []
    for target, cap in CAPACITY.items():
        x = pd.DataFrame(
            {
                "forecast_kst_dtm": valid["kst_dtm"],
                "target": target,
                "month": valid["month"],
                "hour": valid["hour"],
                "actual": valid[target],
                "candidate_pred": cand[target],
                "baseline_pred": base[target],
            }
        )
        x["actual_ratio"] = x["actual"] / cap
        x["candidate_error"] = (x["candidate_pred"] - x["actual"]) / cap
        x["baseline_error"] = (x["baseline_pred"] - x["actual"]) / cap
        x["candidate_abs_error"] = x["candidate_error"].abs()
        x["baseline_abs_error"] = x["baseline_error"].abs()
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
    return pd.concat(rows, ignore_index=True)


def selected_weather(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    choices = {
        "kpx_group_1": {"ldaps": 6, "gfs": 5},
        "kpx_group_2": {"ldaps": 6, "gfs": 5},
        "kpx_group_3": {"ldaps": 12, "gfs": 5},
    }
    pieces = []
    for target, pick in choices.items():
        l = ldaps[ldaps["grid_id"].eq(pick["ldaps"])][
            [
                "forecast_kst_dtm",
                "data_available_kst_dtm",
                "lead_hour",
                "month",
                "hour",
                "issue_date",
                "cycle_hour",
                "wind_primary_speed",
                "wind10_speed",
            ]
        ].rename(columns={"wind_primary_speed": "ldaps_speed", "wind10_speed": "ldaps_wind10"})
        g = gfs[gfs["grid_id"].eq(pick["gfs"])][
            ["forecast_kst_dtm", "wind_primary_speed", "wind10_speed"]
        ].rename(columns={"wind_primary_speed": "gfs_speed", "wind10_speed": "gfs_wind10"})
        m = l.merge(g, on="forecast_kst_dtm", how="inner")
        m["target"] = target
        m["source_disagreement"] = m["ldaps_speed"] - m["gfs_speed"]
        m["source_disagreement_abs"] = m["source_disagreement"].abs()
        pieces.append(m)
    out = pd.concat(pieces, ignore_index=True)
    out["issue_key"] = out["target"] + "|" + out["data_available_kst_dtm"].astype(str)
    return out


def join_validation(overlay: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    joined = overlay.merge(weather, on=["forecast_kst_dtm", "target"], how="left", suffixes=("", "_weather"))
    joined["same_cycle_ldaps_ramp"] = joined.sort_values(["target", "data_available_kst_dtm", "lead_hour"]).groupby(
        ["target", "data_available_kst_dtm"]
    )["ldaps_speed"].diff()
    joined["same_cycle_gfs_ramp"] = joined.sort_values(["target", "data_available_kst_dtm", "lead_hour"]).groupby(
        ["target", "data_available_kst_dtm"]
    )["gfs_speed"].diff()
    joined["same_lead_issue_delta"] = joined.sort_values(["target", "lead_hour", "data_available_kst_dtm"]).groupby(
        ["target", "lead_hour"]
    )["ldaps_speed"].diff()
    joined["actual_ratio_ramp"] = joined.sort_values(["target", "forecast_kst_dtm"]).groupby("target")[
        "actual_ratio"
    ].diff()
    joined.to_csv(OUT / "validation_forecast_time_joined_rows.csv", index=False, encoding="utf-8-sig")
    return joined


def fig_01_time_identity(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in [("ldaps_train", ldaps), ("gfs_train", gfs)]:
        one = df.drop_duplicates("forecast_kst_dtm").copy()
        one["expected_lead_from_hour"] = np.where(one["hour"].eq(0), 35, one["hour"] + 11)
        rows.append(
            {
                "dataset": name,
                "rows": int(len(one)),
                "unique_hour_to_lead_pairs": int(one[["hour", "lead_hour"]].drop_duplicates().shape[0]),
                "cycle_hour_values": ",".join(map(str, sorted(one["cycle_hour"].unique()))),
                "lead_min": float(one["lead_hour"].min()),
                "lead_max": float(one["lead_hour"].max()),
                "violations_of_deterministic_hour_to_lead": int(
                    (one["lead_hour"] != one["expected_lead_from_hour"]).sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "01_time_identity_contract.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for name, df in [("LDAPS", ldaps), ("GFS", gfs)]:
        one = df.drop_duplicates("forecast_kst_dtm")
        axes[0].scatter(one["hour"], one["lead_hour"], s=5, alpha=0.25, label=name)
    hours = np.arange(24)
    axes[0].plot(hours, np.where(hours == 0, 35, hours + 11), color="black", linewidth=1, label="deterministic hour -> lead")
    axes[0].set_title("lead_hour is deterministic from target hour")
    axes[0].set_xlabel("forecast target hour")
    axes[0].set_ylabel("lead_hour")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    counts = ldaps.drop_duplicates("forecast_kst_dtm")["lead_hour"].value_counts().sort_index()
    axes[1].bar(counts.index, counts.values, color="tab:blue", alpha=0.75)
    axes[1].set_title("rectangular 24-lead issue cycle")
    axes[1].set_xlabel("lead_hour")
    axes[1].set_ylabel("forecast timestamps")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("01. Forecast-time contract: lead is determined by target hour")
    savefig("01_time_identity_contract.png")
    return summary


def fig_02_lead_error(joined: pd.DataFrame) -> pd.DataFrame:
    summary = joined.groupby(["target", "lead_hour"], as_index=False).agg(
        rows=("target", "size"),
        candidate_abs_error=("candidate_abs_error", "mean"),
        baseline_abs_error=("baseline_abs_error", "mean"),
        mean_signed_error=("candidate_error", "mean"),
        std_signed_error=("candidate_error", "std"),
        mean_actual_ratio=("actual_ratio", "mean"),
        mean_abs_source_disagreement=("source_disagreement_abs", "mean"),
        pass_to_fail=("ficr_transition", lambda x: int((x == "pass_to_fail").sum())),
        fail_to_pass=("ficr_transition", lambda x: int((x == "fail_to_pass").sum())),
    )
    summary["net_ficr_pass"] = summary["fail_to_pass"] - summary["pass_to_fail"]
    summary.to_csv(OUT / "02_lead_hour_error_surface.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    for ax, target in zip(axes, TARGETS):
        g = summary[summary["target"].eq(target)]
        ax.plot(g["lead_hour"], g["candidate_abs_error"], marker="o", label="candidate abs error")
        ax.plot(g["lead_hour"], g["baseline_abs_error"], marker=".", label="baseline abs error")
        ax2 = ax.twinx()
        ax2.bar(g["lead_hour"], g["net_ficr_pass"], alpha=0.18, color="tab:green", label="net FiCR pass")
        ax.set_title(target)
        ax.set_ylabel("abs error / capacity")
        ax2.set_ylabel("net pass rows")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("lead_hour")
    fig.suptitle("02. Lead-hour score surface: error and FiCR transitions")
    savefig("02_lead_hour_error_surface.png")
    return summary


def fig_03_issue_cycle_bias(joined: pd.DataFrame) -> pd.DataFrame:
    cycle = joined.groupby(["target", "data_available_kst_dtm"], as_index=False).agg(
        rows=("target", "size"),
        mean_signed_error=("candidate_error", "mean"),
        mean_abs_error=("candidate_abs_error", "mean"),
        mean_actual_ratio=("actual_ratio", "mean"),
        mean_ldaps_speed=("ldaps_speed", "mean"),
        mean_source_disagreement=("source_disagreement", "mean"),
        pass_to_fail=("ficr_transition", lambda x: int((x == "pass_to_fail").sum())),
        fail_to_pass=("ficr_transition", lambda x: int((x == "fail_to_pass").sum())),
    )
    cycle["net_ficr_pass"] = cycle["fail_to_pass"] - cycle["pass_to_fail"]
    cycle["abs_cycle_bias"] = cycle["mean_signed_error"].abs()
    cycle.to_csv(OUT / "03_issue_cycle_bias.csv", index=False, encoding="utf-8-sig")
    top = cycle.sort_values("abs_cycle_bias", ascending=False).groupby("target").head(15)
    top.to_csv(OUT / "03_issue_cycle_bias_top_events.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, target in zip(axes, TARGETS):
        g = cycle[cycle["target"].eq(target)]
        ax.scatter(g["mean_signed_error"], g["mean_abs_error"], s=14, alpha=0.55)
        q = g["abs_cycle_bias"].quantile(0.95)
        ax.axvline(q, color="tab:red", linestyle=":", linewidth=0.9)
        ax.axvline(-q, color="tab:red", linestyle=":", linewidth=0.9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(target)
        ax.set_xlabel("cycle mean signed error / capacity")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cycle mean abs error / capacity")
    fig.suptitle("03. Issue-cycle bias: errors cluster by forecast issue, not only by row")
    savefig("03_issue_cycle_bias.png")
    return cycle


def fig_04_available_ramp(joined: pd.DataFrame) -> pd.DataFrame:
    features = ["same_cycle_ldaps_ramp", "same_cycle_gfs_ramp", "same_lead_issue_delta", "source_disagreement"]
    rows = []
    for target, g in joined.groupby("target"):
        for feat in features:
            ok = g[[feat, "actual_ratio_ramp", "candidate_error", "candidate_abs_error"]].dropna()
            rows.append(
                {
                    "target": target,
                    "feature": feat,
                    "rows": int(len(ok)),
                    "spearman_with_actual_ramp": float(ok[feat].corr(ok["actual_ratio_ramp"], method="spearman")),
                    "spearman_with_signed_error": float(ok[feat].corr(ok["candidate_error"], method="spearman")),
                    "spearman_with_abs_error": float(ok[feat].corr(ok["candidate_abs_error"], method="spearman")),
                }
            )
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "04_available_ramp_feature_correlations.csv", index=False, encoding="utf-8-sig")

    piv = corr.pivot(index="target", columns="feature", values="spearman_with_actual_ramp")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    im = axes[0].imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-0.35, vmax=0.35)
    axes[0].set_title("available weather ramps vs actual target-time ramp")
    axes[0].set_yticks(range(len(piv.index)))
    axes[0].set_yticklabels(piv.index)
    axes[0].set_xticks(range(len(piv.columns)))
    axes[0].set_xticklabels(piv.columns, rotation=30, ha="right")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[0].text(j, i, f"{piv.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0], label="Spearman corr")

    piv2 = corr.pivot(index="target", columns="feature", values="spearman_with_abs_error")
    im2 = axes[1].imshow(piv2.values, aspect="auto", cmap="coolwarm", vmin=-0.20, vmax=0.20)
    axes[1].set_title("available weather ramps vs current abs error")
    axes[1].set_yticks(range(len(piv2.index)))
    axes[1].set_yticklabels(piv2.index)
    axes[1].set_xticks(range(len(piv2.columns)))
    axes[1].set_xticklabels(piv2.columns, rotation=30, ha="right")
    for i in range(piv2.shape[0]):
        for j in range(piv2.shape[1]):
            axes[1].text(j, i, f"{piv2.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im2, ax=axes[1], label="Spearman corr")
    fig.suptitle("04. Test-time-available ramp features are weak but measurable proxies")
    savefig("04_available_ramp_proxy_correlations.png")
    return corr


def fig_05_leakage_contrast(labels: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, cap in CAPACITY.items():
        y = labels[["kst_dtm", target]].copy().rename(columns={target: "actual"})
        y["actual_ratio"] = y["actual"] / cap
        y = y.sort_values("kst_dtm")
        y["target_time_actual_lag1"] = y["actual_ratio"].shift(1)
        y["target_time_actual_lag24"] = y["actual_ratio"].shift(24)
        for feat in ["target_time_actual_lag1", "target_time_actual_lag24"]:
            ok = y[["actual_ratio", feat]].dropna()
            rows.append(
                {
                    "target": target,
                    "feature": feat,
                    "test_time_available": False,
                    "rows": int(len(ok)),
                    "spearman_with_actual_ratio": float(ok["actual_ratio"].corr(ok[feat], method="spearman")),
                }
            )
        j = joined[joined["target"].eq(target)]
        for feat in ["ldaps_speed", "gfs_speed", "same_cycle_ldaps_ramp", "same_lead_issue_delta"]:
            ok = j[["actual_ratio", feat]].dropna()
            rows.append(
                {
                    "target": target,
                    "feature": feat,
                    "test_time_available": True,
                    "rows": int(len(ok)),
                    "spearman_with_actual_ratio": float(ok["actual_ratio"].corr(ok[feat], method="spearman")),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "05_leakage_contrast_actual_lag_vs_available_weather.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    plot = out.copy()
    plot["label"] = plot["target"] + "|" + plot["feature"]
    colors = np.where(plot["test_time_available"], "tab:blue", "tab:red")
    ax.barh(plot["label"], plot["spearman_with_actual_ratio"], color=colors, alpha=0.75)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("05. Leakage contrast: strong target lags are not submission features")
    ax.set_xlabel("Spearman correlation with actual ratio")
    savefig("05_leakage_contrast.png")
    return out


def fig_06_2025_lead_month_exposure(ldaps_tr: pd.DataFrame, gfs_tr: pd.DataFrame, ldaps_te: pd.DataFrame, gfs_te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = {
        "ldaps": ["wind_primary_speed", "wind10_speed"],
        "gfs": ["wind_primary_speed", "wind10_speed"],
    }
    for source, train, test in [("ldaps", ldaps_tr, ldaps_te), ("gfs", gfs_tr, gfs_te)]:
        tr = train[train["year"].eq(2024)]
        for feat in features[source]:
            for month in range(1, 13):
                for lead in range(12, 36):
                    a = tr.loc[tr["month"].eq(month) & tr["lead_hour"].eq(lead), feat].dropna()
                    b = test.loc[test["month"].eq(month) & test["lead_hour"].eq(lead), feat].dropna()
                    if len(a) < 20 or len(b) < 20:
                        continue
                    rows.append(
                        {
                            "source": source,
                            "feature": feat,
                            "month": month,
                            "lead_hour": lead,
                            "train2024_mean": float(a.mean()),
                            "test2025_mean": float(b.mean()),
                            "z_delta_vs_2024": float((b.mean() - a.mean()) / (a.std(ddof=0) + 1e-9)),
                            "test_above_train_p90_rate": float((b > a.quantile(0.90)).mean()),
                        }
                    )
    shift = pd.DataFrame(rows)
    shift["abs_z_delta"] = shift["z_delta_vs_2024"].abs()
    shift.to_csv(OUT / "06_2025_lead_month_exposure.csv", index=False, encoding="utf-8-sig")
    top = shift.sort_values("abs_z_delta", ascending=False).head(25)
    top.to_csv(OUT / "06_2025_lead_month_exposure_top.csv", index=False, encoding="utf-8-sig")

    heat = shift.groupby(["month", "lead_hour"], as_index=False)["abs_z_delta"].max()
    piv = heat.pivot(index="month", columns="lead_hour", values="abs_z_delta")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(piv.values, aspect="auto", cmap="magma")
    axes[0].set_title("max 2025-vs-2024 shift by month and lead")
    axes[0].set_yticks(range(len(piv.index)))
    axes[0].set_yticklabels(piv.index)
    axes[0].set_xticks(range(0, len(piv.columns), 2))
    axes[0].set_xticklabels(piv.columns[::2])
    axes[0].set_xlabel("lead_hour")
    axes[0].set_ylabel("month")
    fig.colorbar(im, ax=axes[0], label="max abs z-shift")

    labels = top["source"] + "|" + top["feature"] + "|m" + top["month"].astype(str) + "|L" + top["lead_hour"].astype(str)
    signed = top["z_delta_vs_2024"]
    axes[1].barh(labels.iloc[::-1], signed.iloc[::-1], color=np.where(signed.iloc[::-1] >= 0, "tab:red", "tab:blue"))
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("largest lead-month weather shifts")
    axes[1].set_xlabel("z_delta_vs_2024")
    fig.suptitle("06. 2025 exposure is lead-month specific")
    savefig("06_2025_lead_month_exposure.png")
    return shift


def fig_07_proof_matrix(
    time_identity: pd.DataFrame,
    lead_error: pd.DataFrame,
    cycle_bias: pd.DataFrame,
    ramp_corr: pd.DataFrame,
    leakage: pd.DataFrame,
    exposure: pd.DataFrame,
) -> pd.DataFrame:
    lead_ranges = lead_error.groupby("target")["candidate_abs_error"].agg(lambda x: x.max() - x.min()).round(4).to_dict()
    cycle_p95 = cycle_bias.groupby("target")["abs_cycle_bias"].quantile(0.95).round(4).to_dict()
    ramp_best = ramp_corr.groupby("target")["spearman_with_actual_ramp"].max().round(3).to_dict()
    lag_vs_available = leakage.groupby("test_time_available")["spearman_with_actual_ratio"].max().round(3).to_dict()
    top_exposure = exposure.sort_values("abs_z_delta", ascending=False).head(5)[["month", "lead_hour"]].astype(int).to_dict("records")
    rows = [
        {
            "claim": "lead_hour is not independent from hour",
            "evidence": f"violations={int(time_identity['violations_of_deterministic_hour_to_lead'].sum())}; 24 target hours map deterministically to leads 12-35",
            "status": "proved",
            "model_consequence": "avoid interpreting hour and lead as independent causal signals",
        },
        {
            "claim": "lead_hour has score-relevant error structure",
            "evidence": f"candidate abs-error range by target={lead_ranges}",
            "status": "supported",
            "model_consequence": "use lead interactions and report validation by lead",
        },
        {
            "claim": "issue-cycle bias exists",
            "evidence": f"95th percentile absolute cycle signed error={cycle_p95}",
            "status": "supported",
            "model_consequence": "consider cycle-level calibration/regime features",
        },
        {
            "claim": "test-time-available ramps are usable but weak proxies",
            "evidence": f"best Spearman with actual ramp={ramp_best}",
            "status": "partially_supported",
            "model_consequence": "use as interactions/uncertainty features, not standalone fixes",
        },
        {
            "claim": "target-time label lags are tempting but non-deployable",
            "evidence": f"max corr by availability={lag_vs_available}",
            "status": "proved",
            "model_consequence": "separate target lag experiments from forecast-safe features",
        },
        {
            "claim": "2025 shift is lead-month specific",
            "evidence": f"top shifted month/lead cells={top_exposure}",
            "status": "supported",
            "model_consequence": "stress-test exposed lead-month cells, not only full-year 2024",
        },
    ]
    proof = pd.DataFrame(rows)
    proof.to_csv(OUT / "07_forecast_time_score_proof_matrix.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    colors = proof["status"].map({"proved": "tab:blue", "supported": "tab:green", "partially_supported": "tab:orange"})
    values = proof["status"].map({"proved": 2, "supported": 1.6, "partially_supported": 1.0})
    ax.barh(proof["claim"], values, color=colors, alpha=0.78)
    ax.set_xlim(0, 2.2)
    ax.set_xticks([1, 1.6, 2])
    ax.set_xticklabels(["partial", "supported", "proved"])
    ax.set_title("07. Forecast-time contract proof matrix for score work")
    ax.grid(axis="x", alpha=0.25)
    savefig("07_forecast_time_score_proof_matrix.png")
    return proof


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols is not None:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_issue(
    time_identity: pd.DataFrame,
    lead_error: pd.DataFrame,
    cycle_bias: pd.DataFrame,
    ramp_corr: pd.DataFrame,
    leakage: pd.DataFrame,
    exposure: pd.DataFrame,
    proof: pd.DataFrame,
) -> None:
    top_lead = lead_error.sort_values("candidate_abs_error", ascending=False)
    top_cycle = cycle_bias.sort_values("abs_cycle_bias", ascending=False)
    top_exposure = exposure.sort_values("abs_z_delta", ascending=False)
    body = f"""# Forecast-Time Contract Follow-up EDA: lead, issue-cycle, and score-relevant proof matrix

이 이슈는 #31의 `3. Forecast-Time Contract`에서 나온 결론을 score 개선 관점으로 확장한 추가 EDA입니다. #32가 SCADA taxonomy, label regime, 방향 grid, source disagreement, 2025 exposure를 넓게 다뤘다면, 이 이슈는 **forecast-time contract 자체에서 나오는 모델링 제약과 기회**에 집중합니다.

핵심 질문:

1. `lead_hour`, `hour`, `data_available_kst_dtm`는 정말 독립 feature가 아닌가?
2. lead-hour별 error와 FiCR transition이 score-relevant한가?
3. issue-cycle 단위로 공통 over/under bias가 생기는가?
4. lag/ramp feature 중 무엇이 test-time available이고 무엇이 leakage인가?
5. 2025 test shift는 month만이 아니라 lead-hour까지 분해해야 하는가?

Artifacts:

- script: `experiments/forecast_time_contract_score_eda_20260809/run_forecast_time_contract_score_eda.py`
- figures: `experiments/forecast_time_contract_score_eda_20260809/results/figures/`
- tables: `experiments/forecast_time_contract_score_eda_20260809/results/*.csv`

---

## 1. Time Identity Contract

![time identity]({RAW}/01_time_identity_contract.png)

### 관찰

{md_table(time_identity, 10)}

### 해석

    - train weather에서 forecast target hour와 `lead_hour`는 1:1로 결정됩니다. 01-23시는 lead 12-34에 대응하고, 00시는 lead 35에 대응합니다.
- 따라서 `hour`와 `lead_hour`를 독립적인 시간 feature처럼 해석하면 안 됩니다.
- 이것은 모델에 둘 다 넣지 말라는 뜻이 아니라, 둘의 중요도를 독립 원인으로 해석하면 안 된다는 뜻입니다.

### 타당성 논의

- 타당한 점: raw LDAPS/GFS train의 timestamp에서 직접 계산했습니다.
- 한계: test도 동일 contract인 것은 #31에서 이미 확인했습니다. 여기서는 score overlay가 가능한 train/valid 중심으로 시각화했습니다.

---

## 2. Lead-Hour Error Surface

![lead error]({RAW}/02_lead_hour_error_surface.png)

### 관찰

{md_table(top_lead, 15, ["target", "lead_hour", "rows", "candidate_abs_error", "baseline_abs_error", "mean_signed_error", "mean_abs_source_disagreement", "net_ficr_pass"])}

### 해석

- lead-hour별 candidate error 차이가 존재합니다.
- 특히 긴 lead 쪽에서 error가 커지는 target이 있고, FiCR net transition도 lead별로 달라집니다.
- score 개선 관점에서는 lead별 calibration이나 `lead_hour x wind/source/grid` interaction이 ablation 대상입니다.

### 타당성 논의

- 타당한 점: 2024 validation prediction row에 raw forecast lead를 join해 실제 candidate/baseline error를 집계했습니다.
- 한계: `lead_hour`와 `hour`가 동일 축에 묶여 있으므로, lead 효과와 diurnal generation effect를 완전히 분리했다고 보면 안 됩니다.

---

## 3. Issue-Cycle Bias

![cycle bias]({RAW}/03_issue_cycle_bias.png)

### 관찰

{md_table(top_cycle, 15, ["target", "data_available_kst_dtm", "rows", "mean_signed_error", "mean_abs_error", "mean_actual_ratio", "mean_ldaps_speed", "mean_source_disagreement", "net_ficr_pass"])}

### 해석

- 같은 `data_available_kst_dtm`에서 발행된 24개 target hour가 함께 over/under 되는 cycle-level bias가 있습니다.
- 이건 row-level 후처리보다 issue-cycle regime correction이 더 맞는 후보가 있음을 의미합니다.
- 단, test label이 없으므로 cycle bias를 직접 보정하려면 NWP-only cycle descriptors로 proxy를 만들어야 합니다.

### 타당성 논의

- 타당한 점: 동일 issue cycle 내 24개 forecast row를 묶어 signed error를 계산했습니다.
- 한계: validation label을 쓴 사후 진단입니다. deployable correction은 cycle 평균 풍속, source disagreement, ramp shape 같은 forecast-only 변수로만 가능해야 합니다.

---

## 4. Forecast-Safe Ramp Features

![ramp proxy]({RAW}/04_available_ramp_proxy_correlations.png)

### 관찰

{md_table(ramp_corr, 20)}

### 해석

- 같은 issue cycle 안의 LDAPS/GFS wind ramp, 같은 lead에서 issue-to-issue wind delta는 test-time에 만들 수 있는 feature입니다.
- 실제 발전 ramp나 current residual과의 correlation은 강하지 않지만 0이 아닙니다.
- 따라서 ramp는 단독 해결책이 아니라 regime/uncertainty interaction feature로 쓰는 것이 타당합니다.

### 타당성 논의

- 타당한 점: SCADA나 label 미래값 없이 weather forecast table에서 만들 수 있는 ramp만 사용했습니다.
- 한계: actual ramp와의 상관이 약하므로, 단순 ramp feature 추가만으로 큰 score 개선을 기대하면 안 됩니다.

---

## 5. Leakage Contrast

![leakage contrast]({RAW}/05_leakage_contrast.png)

### 관찰

{md_table(leakage, 20)}

### 해석

- target-time actual lag는 label과 강하게 연결되지만 test 제출 시점에는 사용할 수 없습니다.
- forecast-safe weather features는 상대적으로 약하지만 deployable합니다.
- 앞으로 lag/ramp 실험은 반드시 `target-time actual lag`와 `forecast issue feature`를 분리해서 기록해야 합니다.

### 타당성 논의

- 타당한 점: 같은 target에 대해 non-deployable actual lag와 deployable forecast feature를 같은 상관 척도로 비교했습니다.
- 한계: correlation 비교는 feature value의 정보량만 보여줍니다. 모델 안에서 interaction으로 쓰일 때의 효과는 별도 ablation이 필요합니다.

---

## 6. 2025 Lead-Month Exposure

![lead month exposure]({RAW}/06_2025_lead_month_exposure.png)

### 관찰

{md_table(top_exposure, 20, ["source", "feature", "month", "lead_hour", "train2024_mean", "test2025_mean", "z_delta_vs_2024", "test_above_train_p90_rate"])}

### 해석

- 2025 shift는 월 단위로만 보는 것보다 lead-hour까지 분해했을 때 더 구체적인 위험 셀이 보입니다.
- exposed month/lead cell에서 모델 selection과 postprocessing이 흔들릴 수 있습니다.
- validation report는 최소한 `month x lead_hour x target` slice를 포함해야 합니다.

### 타당성 논의

- 타당한 점: train 2024와 test 2025의 동일 month/lead cell을 비교했습니다.
- 한계: 2025 label이 없으므로 이것은 score impact가 아니라 exposure risk입니다.

---

## 7. Proof Matrix

![proof matrix]({RAW}/07_forecast_time_score_proof_matrix.png)

{md_table(proof, 10)}

## 결론

이번 추가 EDA에서 얻은 score-oriented lesson learned는 다음입니다.

1. **`lead_hour`와 `hour`는 독립 feature가 아니라 같은 forecast contract에서 나온 결정적 매핑입니다.**
2. **lead-hour별 error surface와 issue-cycle bias가 존재하므로, row-level 평균 MAE만으로는 모델을 고르면 안 됩니다.**
3. **target actual lag는 강력하지만 leakage입니다. deployable feature는 forecast issue 안의 ramp/source/grid/lead 구조에서 만들어야 합니다.**
4. **2025 exposure는 month뿐 아니라 lead-hour까지 분해해야 validation risk가 보입니다.**

다음에 증명해야 할 모델링 가설:

- `lead_hour x wind/source/grid` interaction이 static weather feature보다 validation slice error를 낮추는가?
- issue-cycle descriptor로 cycle-level signed bias를 줄일 수 있는가?
- forecast-safe ramp feature가 FiCR boundary row의 pass-to-fail을 줄이는가?
- 2025 exposed month/lead cell을 가중한 validation selection이 public/private transfer를 개선하는가?
"""
    (BASE / "issue_body.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    labels = load_labels()
    ldaps_tr = add_weather_features(read_csv(DATA / "train" / "ldaps_train.csv"), "ldaps")
    gfs_tr = add_weather_features(read_csv(DATA / "train" / "gfs_train.csv"), "gfs")
    ldaps_te = add_weather_features(read_csv(DATA / "test" / "ldaps_test.csv"), "ldaps")
    gfs_te = add_weather_features(read_csv(DATA / "test" / "gfs_test.csv"), "gfs")
    overlay = load_overlay()
    weather = selected_weather(ldaps_tr, gfs_tr)
    joined = join_validation(overlay, weather)
    time_identity = fig_01_time_identity(ldaps_tr, gfs_tr)
    lead_error = fig_02_lead_error(joined)
    cycle_bias = fig_03_issue_cycle_bias(joined)
    ramp_corr = fig_04_available_ramp(joined)
    leakage = fig_05_leakage_contrast(labels, joined)
    exposure = fig_06_2025_lead_month_exposure(ldaps_tr, gfs_tr, ldaps_te, gfs_te)
    proof = fig_07_proof_matrix(time_identity, lead_error, cycle_bias, ramp_corr, leakage, exposure)
    write_issue(time_identity, lead_error, cycle_bias, ramp_corr, leakage, exposure, proof)
    manifest = pd.DataFrame(
        {
            "artifact": [
                "issue_body.md",
                "01_time_identity_contract.csv",
                "02_lead_hour_error_surface.csv",
                "03_issue_cycle_bias.csv",
                "03_issue_cycle_bias_top_events.csv",
                "04_available_ramp_feature_correlations.csv",
                "05_leakage_contrast_actual_lag_vs_available_weather.csv",
                "06_2025_lead_month_exposure.csv",
                "06_2025_lead_month_exposure_top.csv",
                "07_forecast_time_score_proof_matrix.csv",
            ]
        }
    )
    manifest.to_csv(OUT / "manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote forecast-time score EDA to {BASE}")


if __name__ == "__main__":
    main()

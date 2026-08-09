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
ISSUE34 = ROOT / "experiments" / "forecast_time_contract_score_eda_20260809" / "results"

W4 = Window("W4_2022_2023_to_2024", (2022, 2023), None)
SEEDS = [11, 42, 77]
RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/forecast_time_score_ablation_20260809/results/figures"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 170, "font.size": 9, "axes.titlesize": 12})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def add_raw_wind(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = df[["forecast_kst_dtm", "data_available_kst_dtm", "grid_id"]].copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out["lead_hour"] = (out["forecast_kst_dtm"] - out["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    out["cycle_hour"] = out["data_available_kst_dtm"].dt.hour
    if source == "ldaps":
        pairs = {
            "primary": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
            "wind10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
            "blwind": ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
        }
    else:
        pairs = {
            "primary": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
            "wind10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
            "wind100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
            "pblwind": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
        }
    for name, (u_col, v_col) in pairs.items():
        u = pd.to_numeric(df[u_col], errors="coerce")
        v = pd.to_numeric(df[v_col], errors="coerce")
        speed = np.hypot(u, v)
        out[f"{source}_{name}_speed"] = speed
        out[f"{source}_{name}_u"] = u
        out[f"{source}_{name}_v"] = v
    return out


def weather_contract_features() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    ldaps = add_raw_wind(read_csv(DATA / "train" / "ldaps_train.csv"), "ldaps")
    gfs = add_raw_wind(read_csv(DATA / "train" / "gfs_train.csv"), "gfs")

    def per_time(w: pd.DataFrame, source: str) -> pd.DataFrame:
        numeric = [c for c in w.columns if c not in {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id"}]
        grouped = w.groupby("forecast_kst_dtm", sort=True)[numeric].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
        base = grouped.reset_index()
        meta = w.groupby("forecast_kst_dtm", as_index=False).agg(
            data_available_kst_dtm=("data_available_kst_dtm", "first"),
            lead_hour=("lead_hour", "first"),
            cycle_hour=("cycle_hour", "first"),
        )
        base = base.merge(meta, on="forecast_kst_dtm", how="left")
        # Forecast-safe within-issue ramp and same-lead issue-to-issue delta.
        speed_col = f"{source}_primary_speed_mean"
        base = base.sort_values(["data_available_kst_dtm", "lead_hour"])
        base[f"{source}_issue_primary_ramp"] = base.groupby("data_available_kst_dtm")[speed_col].diff()
        base[f"{source}_issue_primary_ramp_abs"] = base[f"{source}_issue_primary_ramp"].abs()
        cycle = base.groupby("data_available_kst_dtm", as_index=False).agg(
            **{
                f"{source}_cycle_primary_mean": (speed_col, "mean"),
                f"{source}_cycle_primary_std": (speed_col, "std"),
                f"{source}_cycle_primary_min": (speed_col, "min"),
                f"{source}_cycle_primary_max": (speed_col, "max"),
                f"{source}_cycle_ramp_abs_mean": (f"{source}_issue_primary_ramp_abs", "mean"),
            }
        )
        cycle[f"{source}_cycle_primary_range"] = cycle[f"{source}_cycle_primary_max"] - cycle[f"{source}_cycle_primary_min"]
        base = base.merge(cycle, on="data_available_kst_dtm", how="left")
        base = base.sort_values(["lead_hour", "data_available_kst_dtm"])
        base[f"{source}_same_lead_issue_delta"] = base.groupby("lead_hour")[speed_col].diff()
        base[f"{source}_same_lead_issue_delta_abs"] = base[f"{source}_same_lead_issue_delta"].abs()
        return base

    l = per_time(ldaps, "ldaps")
    g = per_time(gfs, "gfs")
    feat = l.merge(
        g.drop(columns=["data_available_kst_dtm", "lead_hour", "cycle_hour"]),
        on="forecast_kst_dtm",
        how="inner",
    )
    feat["hour"] = feat["forecast_kst_dtm"].dt.hour
    feat["lead_centered"] = feat["lead_hour"] - 23.5
    feat["lead_centered2"] = feat["lead_centered"] ** 2
    feat["lead_sin"] = np.sin(2 * np.pi * feat["lead_hour"] / 24.0)
    feat["lead_cos"] = np.cos(2 * np.pi * feat["lead_hour"] / 24.0)
    feat["source_primary_disagree"] = feat["ldaps_primary_speed_mean"] - feat["gfs_primary_speed_mean"]
    feat["source_primary_disagree_abs"] = feat["source_primary_disagree"].abs()
    feat["source_primary_ratio"] = feat["ldaps_primary_speed_mean"] / (feat["gfs_primary_speed_mean"].abs() + 1e-6)
    for col in [
        "ldaps_primary_speed_mean",
        "ldaps_primary_speed_max",
        "gfs_primary_speed_mean",
        "gfs_wind100_speed_mean",
        "source_primary_disagree",
    ]:
        feat[f"lead_x_{col}"] = feat["lead_centered"] * feat[col]

    families = {
        "lead_basic": ["lead_hour", "lead_centered", "lead_centered2", "lead_sin", "lead_cos"],
        "lead_interactions": [c for c in feat.columns if c.startswith("lead_x_")],
        "source_disagreement": ["source_primary_disagree", "source_primary_disagree_abs", "source_primary_ratio"],
        "issue_cycle": [
            c
            for c in feat.columns
            if "_cycle_" in c
            or c.endswith("_issue_primary_ramp")
            or c.endswith("_issue_primary_ramp_abs")
        ],
        "ramp": [c for c in feat.columns if "same_lead_issue_delta" in c],
    }
    return feat, families


def base_feature_cols(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c != "forecast_kst_dtm" and pd.api.types.is_numeric_dtype(features[c])]


def train_targetwise(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    cols_by_target: dict[str, list[str]],
    seed: int,
    sample_weight: pd.Series | None = None,
) -> pd.DataFrame:
    from lightgbm import LGBMRegressor

    preds = pd.DataFrame(index=valid.index)
    for target in TARGETS:
        cols = [c for c in cols_by_target[target] if c in train.columns]
        x_valid = valid[cols].replace([np.inf, -np.inf], np.nan)
        y = train[target]
        mask = y.notna()
        x_train = train.loc[mask, cols].replace([np.inf, -np.inf], np.nan)
        weights = sample_weight.loc[mask].to_numpy() if sample_weight is not None else None
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=30,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x_train, y.loc[mask], sample_weight=weights)
        preds[target] = np.clip(model.predict(x_valid), 0.0, CAPACITY[target])
    return preds


def exposure_weights(frame: pd.DataFrame) -> pd.Series:
    exposure = read_csv(ISSUE34 / "06_2025_lead_month_exposure.csv")
    exposure = exposure.groupby(["month", "lead_hour"], as_index=False)["abs_z_delta"].max()
    exposure["weight"] = 1.0 + exposure["abs_z_delta"] / max(float(exposure["abs_z_delta"].max()), 1e-9)
    weight_map = {(int(r.month), int(r.lead_hour)): float(r.weight) for r in exposure.itertuples(index=False)}
    return frame.apply(lambda r: weight_map.get((int(r["month"]), int(r["lead_hour"])), 1.0), axis=1)


def row_table(valid: pd.DataFrame, preds: pd.DataFrame, experiment: str) -> pd.DataFrame:
    rows = []
    for target, cap in CAPACITY.items():
        actual = valid[target]
        pred = preds[target].clip(0.0, cap)
        err = (pred - actual) / cap
        df = pd.DataFrame(
            {
                "forecast_kst_dtm": valid["forecast_kst_dtm"],
                "target": target,
                "month": valid["month"],
                "lead_hour": valid["lead_hour"],
                "actual": actual,
                "actual_ratio": actual / cap,
                "abs_error": err.abs(),
                "signed_error": err,
                "eligible": actual >= 0.10 * cap,
                "pass8": (actual >= 0.10 * cap) & (err.abs() <= 0.08),
                "experiment": experiment,
            }
        )
        df["label_regime"] = pd.cut(
            df["actual_ratio"],
            [-np.inf, 0.0, 0.01, 0.08, 0.12, 0.50, 0.80, np.inf],
            labels=[
                "zero",
                "near_zero_0_1pct",
                "low_1_8pct",
                "ficr_boundary_8_12pct",
                "mid_12_50pct",
                "high_50_80pct",
                "very_high_80pct_plus",
            ],
            include_lowest=True,
        ).astype("object").fillna("missing")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def summarize_result(name: str, seed: int, valid: pd.DataFrame, preds: pd.DataFrame, feature_count: int) -> dict:
    metrics = evaluate_predictions(valid[TARGETS], preds[TARGETS], valid["forecast_kst_dtm"])
    return {
        "experiment": name,
        "seed": seed,
        "score": metrics["score"],
        "one_minus_nmae": metrics["one_minus_nmae"],
        "avg_nmae": metrics["avg_nmae"],
        "ficr": metrics["ficr"],
        "worst_month": min(r["score"] for r in metrics["monthly_rows"]),
        "high_generation_score": evaluate_high_generation(valid[TARGETS], preds[TARGETS], valid["forecast_kst_dtm"]),
        "eligible_hours": metrics["eligible_hours"],
        "feature_count": feature_count,
    }


def delta_slices(valid: pd.DataFrame, baseline_rows: pd.DataFrame, candidate_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = baseline_rows.drop(columns=["experiment"], errors="ignore").rename(
        columns={"abs_error": "baseline_abs_error", "pass8": "baseline_pass8"}
    )
    cand = candidate_rows[["forecast_kst_dtm", "target", "abs_error", "signed_error", "pass8", "experiment"]].rename(
        columns={
            "abs_error": "candidate_abs_error",
            "signed_error": "candidate_signed_error",
            "pass8": "candidate_pass8",
        }
    )
    merged = base.merge(
        cand,
        on=["forecast_kst_dtm", "target"],
        how="inner",
    )
    merged["delta_abs_error"] = merged["candidate_abs_error"] - merged["baseline_abs_error"]
    group_specs = {
        "lead": ["experiment", "target", "lead_hour"],
        "regime": ["experiment", "target", "label_regime"],
        "month_lead": ["experiment", "month", "lead_hour"],
    }
    outs = {}
    for name, keys in group_specs.items():
        out = (
            merged[merged["eligible"]]
            .groupby(keys, observed=True)
            .agg(
                rows=("actual", "size"),
                baseline_abs_error=("baseline_abs_error", "mean"),
                candidate_abs_error=("candidate_abs_error", "mean"),
                delta_abs_error=("delta_abs_error", "mean"),
                impact_delta=("delta_abs_error", "sum"),
                fail_to_pass8=("candidate_pass8", lambda s: 0),
            )
            .reset_index()
        )
        eligible = merged[merged["eligible"]].copy()
        eligible["fail_to_pass8"] = (~eligible["baseline_pass8"]) & eligible["candidate_pass8"]
        eligible["pass_to_fail8"] = eligible["baseline_pass8"] & (~eligible["candidate_pass8"])
        transition = (
            eligible
            .groupby(keys, observed=True)
            .agg(fail_to_pass8=("fail_to_pass8", "sum"), pass_to_fail8=("pass_to_fail8", "sum"))
            .reset_index()
        )
        out = out.drop(columns=["fail_to_pass8"]).merge(transition, on=keys, how="left")
        out["net_pass8"] = out["fail_to_pass8"] - out["pass_to_fail8"]
        outs[name] = out
    return outs["lead"], outs["regime"], outs["month_lead"]


def plot_summary(summary_mean: pd.DataFrame, lead_delta: pd.DataFrame, regime_delta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    s = summary_mean.sort_values("delta_score_vs_baseline")
    axes[0].barh(s["experiment"], s["delta_score_vs_baseline"], color=np.where(s["delta_score_vs_baseline"] >= 0, "tab:blue", "tab:red"))
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Mean score delta vs B1 aggregate across seeds")
    axes[0].set_xlabel("delta score")

    best_name = summary_mean.sort_values("score_mean", ascending=False).iloc[0]["experiment"]
    lead = lead_delta[lead_delta["experiment"].eq(best_name)]
    for target, g in lead.groupby("target"):
        axes[1].plot(g["lead_hour"], g["delta_abs_error"], marker="o", label=target)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{best_name}: lead-hour delta abs error vs baseline")
    axes[1].set_xlabel("lead_hour")
    axes[1].set_ylabel("delta abs error; negative is better")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    reg = regime_delta[regime_delta["experiment"].eq(best_name)]
    piv = reg.pivot_table(index="target", columns="label_regime", values="delta_abs_error", aggfunc="mean")
    order = ["ficr_boundary_8_12pct", "mid_12_50pct", "high_50_80pct", "very_high_80pct_plus"]
    piv = piv[[c for c in order if c in piv.columns]]
    im = axes[2].imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    axes[2].set_title(f"{best_name}: regime delta abs error")
    axes[2].set_yticks(range(len(piv.index)))
    axes[2].set_yticklabels(piv.index)
    axes[2].set_xticks(range(len(piv.columns)))
    axes[2].set_xticklabels(piv.columns, rotation=25, ha="right")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[2].text(j, i, f"{piv.iloc[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[2], label="delta abs error")
    fig.tight_layout()
    fig.savefig(FIG / "01_forecast_time_ablation_summary.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols is not None:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_report(summary: pd.DataFrame, summary_mean: pd.DataFrame, lead_delta: pd.DataFrame, regime_delta: pd.DataFrame, month_lead_delta: pd.DataFrame) -> None:
    best = summary_mean.sort_values("score_mean", ascending=False).iloc[0]
    baseline = summary_mean[summary_mean["experiment"].eq("B1_aggregate")].iloc[0]
    top_regime = regime_delta.sort_values("impact_delta")
    worst_regime = regime_delta.sort_values("impact_delta", ascending=False)
    top_month_lead = month_lead_delta.sort_values("impact_delta")
    body = f"""# Forecast-Time Score Ablation Results

이 실험은 #34에서 제안한 score 개선 가설을 실제 W4 validation score로 검증한 결과입니다.

Protocol:

- Train: 2022-2023 labels
- Validation: 2024 labels
- Metric: repo official local proxy, `0.5 * (1 - avg NMAE) + 0.5 * FiCR`
- Model: fixed LightGBM L1 settings, no HPO
- Seeds: `{SEEDS}`
- Baseline: `B1_aggregate`

![summary]({RAW}/01_forecast_time_ablation_summary.png)

## Main Result

{md_table(summary_mean.sort_values("score_mean", ascending=False), 20)}

Interpretation:

- Best mean score: `{best['experiment']}` = `{best['score_mean']:.9f}`.
- Baseline mean score: `B1_aggregate` = `{baseline['score_mean']:.9f}`.
- Best delta vs baseline: `{best['delta_score_vs_baseline']:.9f}`.
- Materiality 기준을 `+0.001` 이상으로 보면, 이번 forecast-time feature family 중 **확신 있게 score를 올린 후보는 없습니다**.

## Raw Seed Results

{md_table(summary.sort_values(["experiment", "seed"]), 30)}

## Best Improvements by Regime

{md_table(top_regime, 20, ["experiment", "target", "label_regime", "rows", "delta_abs_error", "impact_delta", "net_pass8"])}

## Worst Regressions by Regime

{md_table(worst_regime, 20, ["experiment", "target", "label_regime", "rows", "delta_abs_error", "impact_delta", "net_pass8"])}

## Month-Lead Cells Where Candidates Helped Most

{md_table(top_month_lead, 20, ["experiment", "month", "lead_hour", "rows", "delta_abs_error", "impact_delta", "net_pass8"])}

## Conclusion

What is proven:

1. `lead_basic` / `lead_interactions` can produce tiny local changes, but the gain is below materiality.
2. `issue_cycle`, `ramp`, and `source_disagreement` contain slice-level signal, but simple feature addition creates offsetting regressions.
3. `2025 lead-month exposure weighting` is not a reliable training trick in this fixed-LGBM protocol.
4. The EDA insight is still useful as a diagnostic/reporting axis, but not yet as a deployable score-raising feature family.

Decision:

- Do not promote a new submission from these experiments.
- Keep `lead_hour`, issue-cycle, ramp, and exposure as validation slices and possible calibration axes.
- Next experiment should be **postprocessing/calibration constrained to proven-positive slices**, not another broad feature dump.
"""
    (Path(__file__).resolve().parent / "report.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    labels = read_labels(DATA)
    b1 = build_feature_frame(DATA, "B1_aggregate")
    contract, families = weather_contract_features()
    features = b1.merge(contract, on="forecast_kst_dtm", how="left")
    data = merge_features_labels(features, labels)
    data["lead_hour"] = pd.to_numeric(data["lead_hour"], errors="coerce")
    data["month"] = data["forecast_kst_dtm"].dt.month
    train = data[data["year"].isin(W4.train_years)].copy().reset_index(drop=True)
    valid = data[data["year"].eq(2024)].copy().reset_index(drop=True)

    base_cols = base_feature_cols(b1)
    specs = {
        "B1_aggregate": base_cols,
        "lead_basic": base_cols + families["lead_basic"],
        "lead_interactions": base_cols + families["lead_basic"] + families["lead_interactions"],
        "issue_cycle": base_cols + families["issue_cycle"],
        "forecast_safe_ramp": base_cols + families["ramp"],
        "source_disagreement": base_cols + families["source_disagreement"],
        "all_forecast_time_features": base_cols
        + families["lead_basic"]
        + families["lead_interactions"]
        + families["issue_cycle"]
        + families["ramp"]
        + families["source_disagreement"],
        "all_plus_2025_lead_month_weight": base_cols
        + families["lead_basic"]
        + families["lead_interactions"]
        + families["issue_cycle"]
        + families["ramp"]
        + families["source_disagreement"],
    }
    specs = {name: list(dict.fromkeys(cols)) for name, cols in specs.items()}
    weights = exposure_weights(train)

    rows = []
    row_tables: dict[tuple[str, int], pd.DataFrame] = {}
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for seed in SEEDS:
        baseline_preds = None
        for name, cols in specs.items():
            cols_by_target = {target: cols for target in TARGETS}
            pred_path = pred_dir / f"{name}_seed{seed}_valid_predictions.csv"
            if pred_path.exists():
                preds = read_csv(pred_path)[TARGETS]
            elif name == "B1_aggregate":
                preds = train_lgbm_predictions(train, valid, cols, W4, seed)
                baseline_preds = preds.copy()
            else:
                sample_weight = weights if name == "all_plus_2025_lead_month_weight" else None
                preds = train_targetwise(train, valid, cols_by_target, seed, sample_weight=sample_weight)
            if name == "B1_aggregate":
                baseline_preds = preds.copy()
            preds.to_csv(pred_path, index=False, encoding="utf-8-sig")
            rows.append(summarize_result(name, seed, valid, preds, len(cols)))
            row_tables[(name, seed)] = row_table(valid, preds, name)
        assert baseline_preds is not None

    summary = pd.DataFrame(rows)
    baseline_by_seed = summary[summary["experiment"].eq("B1_aggregate")][["seed", "score", "ficr", "avg_nmae"]].rename(
        columns={"score": "baseline_score", "ficr": "baseline_ficr", "avg_nmae": "baseline_avg_nmae"}
    )
    summary = summary.merge(baseline_by_seed, on="seed", how="left")
    summary["delta_score_vs_baseline"] = summary["score"] - summary["baseline_score"]
    summary["delta_ficr_vs_baseline"] = summary["ficr"] - summary["baseline_ficr"]
    summary["delta_nmae_vs_baseline"] = summary["avg_nmae"] - summary["baseline_avg_nmae"]
    summary.to_csv(OUT / "seed_ablation_summary.csv", index=False, encoding="utf-8-sig")

    summary_mean = (
        summary.groupby("experiment", as_index=False)
        .agg(
            score_mean=("score", "mean"),
            score_std=("score", "std"),
            delta_score_vs_baseline=("delta_score_vs_baseline", "mean"),
            delta_score_min=("delta_score_vs_baseline", "min"),
            delta_score_max=("delta_score_vs_baseline", "max"),
            ficr_mean=("ficr", "mean"),
            avg_nmae_mean=("avg_nmae", "mean"),
            worst_month_mean=("worst_month", "mean"),
            high_generation_mean=("high_generation_score", "mean"),
            feature_count=("feature_count", "first"),
        )
        .sort_values("score_mean", ascending=False)
    )
    summary_mean.to_csv(OUT / "ablation_summary_mean.csv", index=False, encoding="utf-8-sig")

    lead_parts, regime_parts, month_lead_parts = [], [], []
    for seed in SEEDS:
        baseline_rows = row_tables[("B1_aggregate", seed)]
        for name in specs:
            if name == "B1_aggregate":
                continue
            lead, regime, month_lead = delta_slices(valid, baseline_rows, row_tables[(name, seed)])
            lead["seed"] = seed
            regime["seed"] = seed
            month_lead["seed"] = seed
            lead_parts.append(lead)
            regime_parts.append(regime)
            month_lead_parts.append(month_lead)
    lead_delta = pd.concat(lead_parts, ignore_index=True)
    regime_delta = pd.concat(regime_parts, ignore_index=True)
    month_lead_delta = pd.concat(month_lead_parts, ignore_index=True)
    for name, df in [("lead_delta_vs_baseline.csv", lead_delta), ("regime_delta_vs_baseline.csv", regime_delta), ("month_lead_delta_vs_baseline.csv", month_lead_delta)]:
        df.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    plot_summary(summary_mean, lead_delta, regime_delta)
    write_report(summary, summary_mean, lead_delta, regime_delta, month_lead_delta)
    print((Path(__file__).resolve().parent / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

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

from run_cv_protocol import CAPACITY, TARGETS, evaluate_high_generation, evaluate_predictions, read_labels  # noqa: E402


DATA = ROOT.parent / "open"
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
FIG = OUT / "figures"
FUND = ROOT / "experiments" / "fundamental_feature_distillation_20260809" / "results" / "predictions"
DEPLOY = ROOT / "experiments" / "deployable_score_risk_ablation_20260809" / "results" / "predictions"
RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/regime_gated_specialist_20260809/results/figures"


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 170, "font.size": 9, "axes.titlesize": 12})


def read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[TARGETS].copy()


def valid_labels() -> pd.DataFrame:
    labels = read_labels(DATA)
    valid = labels[labels["year"].eq(2024)].copy().reset_index(drop=True)
    valid = valid.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    return valid


def label_regime(actual_ratio: pd.Series) -> pd.Series:
    return pd.cut(
        actual_ratio,
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


def summarize(name: str, preds: pd.DataFrame, valid: pd.DataFrame) -> dict:
    clipped = preds.copy()
    for target in TARGETS:
        clipped[target] = clipped[target].clip(0.0, CAPACITY[target])
    metrics = evaluate_predictions(valid[TARGETS], clipped, valid["forecast_kst_dtm"])
    high = evaluate_high_generation(valid[TARGETS], clipped, valid["forecast_kst_dtm"])
    return {
        "experiment": name,
        "score": metrics["score"],
        "one_minus_nmae": metrics["one_minus_nmae"],
        "avg_nmae": metrics["avg_nmae"],
        "ficr": metrics["ficr"],
        "worst_month": min(r["score"] for r in metrics["monthly_rows"]),
        "high_generation_score": high,
        "eligible_hours": metrics["eligible_hours"],
    }


def blended(base: pd.DataFrame, specialist: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    out = base.copy()
    for target in TARGETS:
        m = masks[target].fillna(False).to_numpy()
        out.loc[m, target] = specialist.loc[m, target]
    return out


def actual_ratio(valid: pd.DataFrame, target: str) -> pd.Series:
    return valid[target] / CAPACITY[target]


def pred_ratio(preds: pd.DataFrame, target: str) -> pd.Series:
    return preds[target] / CAPACITY[target]


def oracle_masks(valid: pd.DataFrame, regime: str) -> dict[str, pd.Series]:
    masks = {}
    for target in TARGETS:
        masks[target] = label_regime(actual_ratio(valid, target)).eq(regime)
    return masks


def predicted_high_masks(gate_preds: pd.DataFrame, threshold: float) -> dict[str, pd.Series]:
    return {target: pred_ratio(gate_preds, target).ge(threshold) for target in TARGETS}


def predicted_boundary_masks(gate_preds: pd.DataFrame, low: float, high: float) -> dict[str, pd.Series]:
    return {target: pred_ratio(gate_preds, target).between(low, high, inclusive="both") for target in TARGETS}


def regime_delta_table(valid: pd.DataFrame, baseline: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, preds in candidates.items():
        for target in TARGETS:
            cap = CAPACITY[target]
            actual = valid[target]
            eligible = actual >= 0.10 * cap
            b_err = (baseline[target].clip(0.0, cap) - actual).abs() / cap
            c_err = (preds[target].clip(0.0, cap) - actual).abs() / cap
            b_pass = eligible & (b_err <= 0.08)
            c_pass = eligible & (c_err <= 0.08)
            reg = label_regime(actual / cap)
            for regime, idx in reg[eligible].groupby(reg[eligible], observed=True).groups.items():
                rows.append(
                    {
                        "experiment": name,
                        "target": target,
                        "label_regime": regime,
                        "rows": int(len(idx)),
                        "baseline_abs_error": float(b_err.loc[idx].mean()),
                        "candidate_abs_error": float(c_err.loc[idx].mean()),
                        "delta_abs_error": float((c_err.loc[idx] - b_err.loc[idx]).mean()),
                        "impact_delta": float((c_err.loc[idx] - b_err.loc[idx]).sum()),
                        "fail_to_pass8": int((~b_pass.loc[idx] & c_pass.loc[idx]).sum()),
                        "pass_to_fail8": int((b_pass.loc[idx] & ~c_pass.loc[idx]).sum()),
                    }
                )
    out = pd.DataFrame(rows)
    out["net_pass8"] = out["fail_to_pass8"] - out["pass_to_fail8"]
    out.to_csv(OUT / "regime_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
    return out


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    setup()
    valid = valid_labels()
    preds = {
        "B1_aggregate": read_pred(FUND / "B1_aggregate_valid_predictions.csv"),
        "B1_plus_lead_interactions": read_pred(DEPLOY / "B1_plus_lead_interactions_valid_predictions.csv"),
        "B1_plus_scada_wind_proxy": read_pred(FUND / "B1_plus_scada_wind_proxy_valid_predictions.csv"),
        "B1_plus_proxy_power_curve": read_pred(FUND / "B1_plus_proxy_power_curve_valid_predictions.csv"),
        "B1_plus_fundamental_all": read_pred(FUND / "B1_plus_fundamental_all_valid_predictions.csv"),
    }
    base = preds["B1_plus_lead_interactions"]

    candidates: dict[str, pd.DataFrame] = {
        "B1_aggregate": preds["B1_aggregate"],
        "B1_plus_lead_interactions": base,
    }

    # Non-deployable upper bounds: if true regime is known, how much can specialist help?
    for specialist_name in ["B1_plus_scada_wind_proxy", "B1_plus_proxy_power_curve", "B1_plus_fundamental_all"]:
        candidates[f"oracle_very_high::{specialist_name}"] = blended(base, preds[specialist_name], oracle_masks(valid, "very_high_80pct_plus"))
        candidates[f"oracle_boundary::{specialist_name}"] = blended(base, preds[specialist_name], oracle_masks(valid, "ficr_boundary_8_12pct"))
        combo_masks = {}
        vh = oracle_masks(valid, "very_high_80pct_plus")
        bd = oracle_masks(valid, "ficr_boundary_8_12pct")
        for target in TARGETS:
            combo_masks[target] = vh[target] | bd[target]
        candidates[f"oracle_boundary_or_very_high::{specialist_name}"] = blended(base, preds[specialist_name], combo_masks)

    # Deployable gates: no true labels, only prediction ratio from base or specialist prediction.
    rows = []
    deployable_preds: dict[str, pd.DataFrame] = {}
    for specialist_name in ["B1_plus_scada_wind_proxy", "B1_plus_proxy_power_curve", "B1_plus_fundamental_all"]:
        for gate_name, gate_pred in [("lead_pred", base), ("specialist_pred", preds[specialist_name])]:
            for threshold in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
                name = f"deploy_high_{gate_name}_ge_{threshold:.2f}::{specialist_name}"
                p = blended(base, preds[specialist_name], predicted_high_masks(gate_pred, threshold))
                deployable_preds[name] = p
                r = summarize(name, p, valid)
                r["gate"] = gate_name
                r["threshold"] = threshold
                r["specialist"] = specialist_name
                rows.append(r)
            for low, high in [(0.06, 0.14), (0.08, 0.12), (0.08, 0.16)]:
                name = f"deploy_boundary_{gate_name}_{low:.2f}_{high:.2f}::{specialist_name}"
                p = blended(base, preds[specialist_name], predicted_boundary_masks(gate_pred, low, high))
                deployable_preds[name] = p
                r = summarize(name, p, valid)
                r["gate"] = gate_name
                r["threshold"] = np.nan
                r["specialist"] = specialist_name
                rows.append(r)

    deployable_grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    lead_base_score = summarize("B1_plus_lead_interactions", base, valid)["score"]
    deployable_grid["delta_score_vs_lead_base"] = deployable_grid["score"] - lead_base_score
    deployable_grid.to_csv(OUT / "deployable_gate_grid.csv", index=False, encoding="utf-8-sig")

    # Add top deployable gates and best oracle gates to the main comparison.
    for name in deployable_grid.head(10)["experiment"]:
        candidates[name] = deployable_preds[name]

    summary = pd.DataFrame([summarize(name, p, valid) for name, p in candidates.items()])
    baseline_score = float(summary.loc[summary["experiment"].eq("B1_plus_lead_interactions"), "score"].iloc[0])
    baseline_ficr = float(summary.loc[summary["experiment"].eq("B1_plus_lead_interactions"), "ficr"].iloc[0])
    baseline_nmae = float(summary.loc[summary["experiment"].eq("B1_plus_lead_interactions"), "avg_nmae"].iloc[0])
    summary["delta_score_vs_lead_base"] = summary["score"] - baseline_score
    summary["delta_ficr_vs_lead_base"] = summary["ficr"] - baseline_ficr
    summary["delta_nmae_vs_lead_base"] = summary["avg_nmae"] - baseline_nmae
    summary = summary.sort_values("score", ascending=False)
    summary.to_csv(OUT / "gated_ablation_summary.csv", index=False, encoding="utf-8-sig")

    top_names = summary.head(12)["experiment"].tolist()
    regime = regime_delta_table(valid, base, {name: candidates[name] for name in top_names if name in candidates})
    plot(summary, deployable_grid, regime)
    write_report(summary, deployable_grid, regime)

    (OUT / "run_metadata.json").write_text(json.dumps({"base": "B1_plus_lead_interactions", "valid_year": 2024}, indent=2), encoding="utf-8")
    return summary, deployable_grid, regime


def plot(summary: pd.DataFrame, deployable_grid: pd.DataFrame, regime: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    show = summary.head(12).sort_values("delta_score_vs_lead_base")
    colors = np.where(show["delta_score_vs_lead_base"] >= 0, "tab:blue", "tab:red")
    axes[0].barh(show["experiment"], show["delta_score_vs_lead_base"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Regime-gated specialist score delta vs lead-interaction base")
    axes[0].set_xlabel("score delta")

    d = deployable_grid.head(15).sort_values("score")
    axes[1].barh(d["experiment"], d["score"], color="tab:green", alpha=0.75)
    axes[1].axvline(float(summary.loc[summary["experiment"].eq("B1_plus_lead_interactions"), "score"].iloc[0]), color="black", linestyle=":", linewidth=0.8)
    axes[1].set_title("Top deployable gates from threshold grid")
    axes[1].set_xlabel("score")

    piv = regime.pivot_table(index="experiment", columns="label_regime", values="delta_abs_error", aggfunc="mean")
    order = ["ficr_boundary_8_12pct", "mid_12_50pct", "high_50_80pct", "very_high_80pct_plus"]
    piv = piv[[c for c in order if c in piv.columns]].head(10)
    im = axes[2].imshow(piv.values, aspect="auto", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    axes[2].set_yticks(range(len(piv.index)))
    axes[2].set_yticklabels(piv.index)
    axes[2].set_xticks(range(len(piv.columns)))
    axes[2].set_xticklabels(piv.columns, rotation=25, ha="right")
    axes[2].set_title("Regime error delta for top gated candidates; blue is better")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[2].text(j, i, f"{piv.iloc[i, j]:.4f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[2], label="delta abs error")
    fig.tight_layout()
    fig.savefig(FIG / "01_regime_gated_specialist_summary.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_report(summary: pd.DataFrame, deployable_grid: pd.DataFrame, regime: pd.DataFrame) -> None:
    lead_base = summary[summary["experiment"].eq("B1_plus_lead_interactions")].iloc[0]
    best = summary.iloc[0]
    best_deploy = deployable_grid.iloc[0]
    oracle = summary[summary["experiment"].str.startswith("oracle_")].sort_values("score", ascending=False).head(8)
    body = f"""# Regime-gated specialist ablation

이 업데이트는 “SCADA-like/proxy feature를 전체 row에 뿌리면 score가 떨어진다”는 이전 결과 이후, 해당 specialist를 특정 regime에서만 켜는 방식이 가능한지 검증합니다.

Base model:

- `B1_plus_lead_interactions`: previous best validation score `{lead_base['score']:.6f}`

Tested gates:

- oracle true very-high gate: 실제 `actual/capacity >= 80%`를 아는 비배포 상한선
- oracle true boundary gate: 실제 FiCR boundary를 아는 비배포 상한선
- deployable high gate: baseline 또는 specialist prediction ratio가 threshold 이상일 때만 specialist 사용
- deployable boundary gate: prediction ratio가 6-14%, 8-12%, 8-16% 구간일 때만 specialist 사용

![Regime gated specialist summary]({RAW}/01_regime_gated_specialist_summary.png)

## Main Result

{md_table(summary, 18, ["experiment", "score", "one_minus_nmae", "avg_nmae", "ficr", "worst_month", "high_generation_score", "delta_score_vs_lead_base", "delta_ficr_vs_lead_base", "delta_nmae_vs_lead_base"])}

해석:

- 전체 최고 후보는 `{best['experiment']}`이고 score delta vs lead-base는 `{best['delta_score_vs_lead_base']:.6f}`입니다.
- 최고 deployable gate는 `{best_deploy['experiment']}`이고 score는 `{best_deploy['score']:.6f}`입니다.
- oracle gate와 deployable gate 차이는 “신호는 있는데 실제 gate가 충분히 정확하지 않은지”를 판단하는 핵심입니다.

## Oracle Upper Bounds

{md_table(oracle, 8, ["experiment", "score", "delta_score_vs_lead_base", "delta_ficr_vs_lead_base", "delta_nmae_vs_lead_base", "high_generation_score"])}

해석:

- oracle very-high gate가 좋아지면 specialist 자체는 올바른 구간에서 가치가 있다는 뜻입니다.
- deployable gate가 못 따라오면 문제는 specialist가 아니라 gate quality입니다.

## Top Deployable Gate Grid

{md_table(deployable_grid, 15, ["experiment", "score", "delta_score_vs_lead_base", "ficr", "avg_nmae", "worst_month", "high_generation_score", "gate", "threshold", "specialist"])}

## Regime Delta For Top Candidates

{md_table(regime.sort_values("impact_delta").head(24), 24, ["experiment", "target", "label_regime", "rows", "delta_abs_error", "impact_delta", "net_pass8"])}

## Conclusion

이 실험의 핵심 질문은 “#32 신호를 전체 row가 아니라 필요한 regime에서만 쓰면 score가 오르는가?”였습니다.

결론은 summary table 기준으로 판단해야 합니다.

- oracle이 좋아지고 deployable이 약하면 gate를 더 잘 학습해야 합니다.
- oracle도 약하면 specialist feature 자체를 폐기하거나 재설계해야 합니다.
- deployable이 lead-base를 넘으면 해당 gate는 다음 제출 후보입니다.
"""
    (BASE / "issue_update.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    s, d, r = run()
    print(s.head(20).to_string(index=False))

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
FUND = ROOT / "experiments" / "fundamental_feature_distillation_20260809" / "results" / "predictions"
DEPLOY = ROOT / "experiments" / "deployable_score_risk_ablation_20260809" / "results" / "predictions"
RAW = "https://raw.githubusercontent.com/shaun0927/BARAM-2026/main/experiments/gate_threshold_robustness_20260809/results/figures"

SPECIALISTS = [
    "B1_plus_scada_wind_proxy",
    "B1_plus_proxy_power_curve",
    "B1_plus_fundamental_all",
]
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 170, "font.size": 9, "axes.titlesize": 12})


def valid_labels() -> pd.DataFrame:
    labels = read_labels(DATA)
    valid = labels[labels["year"].eq(2024)].copy().reset_index(drop=True)
    valid = valid.rename(columns={"kst_dtm": "forecast_kst_dtm"})
    return valid


def read_pred(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")[TARGETS].copy()


def clip(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    for target in TARGETS:
        out[target] = out[target].clip(0.0, CAPACITY[target])
    return out


def score(preds: pd.DataFrame, valid: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    if mask is None:
        y = valid
        p = preds
    else:
        y = valid.loc[mask].reset_index(drop=True)
        p = preds.loc[mask].reset_index(drop=True)
    m = evaluate_predictions(y[TARGETS], clip(p), y["forecast_kst_dtm"])
    return {
        "score": m["score"],
        "one_minus_nmae": m["one_minus_nmae"],
        "avg_nmae": m["avg_nmae"],
        "ficr": m["ficr"],
        "eligible_hours": m["eligible_hours"],
    }


def blend(base: pd.DataFrame, specialist: pd.DataFrame, threshold: float, gate_pred: pd.DataFrame | None = None) -> pd.DataFrame:
    if gate_pred is None:
        gate_pred = specialist
    out = base.copy()
    for target in TARGETS:
        ratio = gate_pred[target] / CAPACITY[target]
        m = ratio >= threshold
        out.loc[m, target] = specialist.loc[m, target]
    return out


def split_masks(valid: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    month = valid["forecast_kst_dtm"].dt.month
    return {
        "cal_h1_eval_h2": (month <= 6, month >= 7),
        "cal_h2_eval_h1": (month >= 7, month <= 6),
        "cal_odd_eval_even": (month % 2 == 1, month % 2 == 0),
        "cal_even_eval_odd": (month % 2 == 0, month % 2 == 1),
        "cal_not_q1_eval_q1": (~month.isin([1, 2, 3]), month.isin([1, 2, 3])),
        "cal_not_q2_eval_q2": (~month.isin([4, 5, 6]), month.isin([4, 5, 6])),
        "cal_not_q3_eval_q3": (~month.isin([7, 8, 9]), month.isin([7, 8, 9])),
        "cal_not_q4_eval_q4": (~month.isin([10, 11, 12]), month.isin([10, 11, 12])),
    }


def candidate_grid(base: pd.DataFrame, specialists: dict[str, pd.DataFrame], valid: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    preds = {}
    base_score = score(base, valid)["score"]
    for spec_name, spec in specialists.items():
        for gate_name, gate_pred in [("specialist_pred", spec), ("lead_pred", base)]:
            for threshold in THRESHOLDS:
                name = f"{gate_name}_ge_{threshold:.2f}::{spec_name}"
                p = blend(base, spec, threshold, gate_pred)
                preds[name] = p
                s = score(p, valid)
                s.update(
                    {
                        "candidate": name,
                        "gate": gate_name,
                        "threshold": threshold,
                        "specialist": spec_name,
                        "delta_score_vs_base": s["score"] - base_score,
                    }
                )
                rows.append(s)
    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid.to_csv(OUT / "full_2024_gate_grid.csv", index=False, encoding="utf-8-sig")
    return grid, preds


def robustness(base: pd.DataFrame, valid: pd.DataFrame, grid_preds: dict[str, pd.DataFrame], full_grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fixed_name = "specialist_pred_ge_0.65::B1_plus_fundamental_all"
    masks = split_masks(valid)
    for split, (cal_mask, eval_mask) in masks.items():
        base_cal = score(base, valid, cal_mask)
        base_eval = score(base, valid, eval_mask)
        scored = []
        for name, p in grid_preds.items():
            s_cal = score(p, valid, cal_mask)
            scored.append({"candidate": name, "cal_score": s_cal["score"], "cal_delta": s_cal["score"] - base_cal["score"]})
        cal_rank = pd.DataFrame(scored).sort_values("cal_score", ascending=False)
        selected = cal_rank.iloc[0]["candidate"]
        for label, name in [("selected_on_calibration", selected), ("fixed_0p65_fundamental", fixed_name)]:
            p = grid_preds[name]
            s_eval = score(p, valid, eval_mask)
            full_meta = full_grid[full_grid["candidate"].eq(name)].iloc[0].to_dict()
            rows.append(
                {
                    "split": split,
                    "policy": label,
                    "candidate": name,
                    "cal_best_candidate": selected,
                    "eval_score": s_eval["score"],
                    "eval_delta_vs_base": s_eval["score"] - base_eval["score"],
                    "eval_ficr": s_eval["ficr"],
                    "eval_avg_nmae": s_eval["avg_nmae"],
                    "base_eval_score": base_eval["score"],
                    "gate": full_meta["gate"],
                    "threshold": full_meta["threshold"],
                    "specialist": full_meta["specialist"],
                    "eval_eligible_hours": s_eval["eligible_hours"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "split_threshold_robustness.csv", index=False, encoding="utf-8-sig")
    return out


def monthly_delta(base: pd.DataFrame, best: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    month = valid["forecast_kst_dtm"].dt.month
    for m in range(1, 13):
        mask = month == m
        b = score(base, valid, mask)
        c = score(best, valid, mask)
        rows.append(
            {
                "month": m,
                "base_score": b["score"],
                "candidate_score": c["score"],
                "delta_score": c["score"] - b["score"],
                "base_ficr": b["ficr"],
                "candidate_ficr": c["ficr"],
                "delta_ficr": c["ficr"] - b["ficr"],
                "base_avg_nmae": b["avg_nmae"],
                "candidate_avg_nmae": c["avg_nmae"],
                "delta_avg_nmae": c["avg_nmae"] - b["avg_nmae"],
                "eligible_hours": c["eligible_hours"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "monthly_delta_fixed_gate.csv", index=False, encoding="utf-8-sig")
    return out


def plot(full_grid: pd.DataFrame, robust: pd.DataFrame, monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    top = full_grid.head(12).sort_values("delta_score_vs_base")
    axes[0].barh(top["candidate"], top["delta_score_vs_base"], color=np.where(top["delta_score_vs_base"] >= 0, "tab:blue", "tab:red"))
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Full-2024 threshold grid: score delta vs lead base")
    axes[0].set_xlabel("score delta")

    piv = robust.pivot(index="split", columns="policy", values="eval_delta_vs_base")
    x = np.arange(len(piv))
    width = 0.38
    axes[1].bar(x - width / 2, piv["selected_on_calibration"], width, label="selected on calibration")
    axes[1].bar(x + width / 2, piv["fixed_0p65_fundamental"], width, label="fixed 0.65 fundamental")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(piv.index, rotation=25, ha="right")
    axes[1].set_ylabel("holdout score delta")
    axes[1].set_title("Threshold robustness: choose gate on calibration, score on held-out months")
    axes[1].legend()

    axes[2].bar(monthly["month"], monthly["delta_score"], color=np.where(monthly["delta_score"] >= 0, "tab:blue", "tab:red"))
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_title("Fixed deployable gate monthly score delta")
    axes[2].set_xlabel("month")
    axes[2].set_ylabel("score delta")
    fig.tight_layout()
    fig.savefig(FIG / "01_gate_threshold_robustness.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, n: int = 12, cols: list[str] | None = None) -> str:
    work = df.copy()
    if cols:
        work = work[cols]
    return work.head(n).to_markdown(index=False)


def write_report(full_grid: pd.DataFrame, robust: pd.DataFrame, monthly: pd.DataFrame) -> None:
    fixed = full_grid[full_grid["candidate"].eq("specialist_pred_ge_0.65::B1_plus_fundamental_all")].iloc[0]
    fixed_robust = robust[robust["policy"].eq("fixed_0p65_fundamental")]
    selected_robust = robust[robust["policy"].eq("selected_on_calibration")]
    body = f"""# Gate threshold robustness check

이 업데이트는 previous best gate가 2024 validation threshold search에 과적합된 것인지 확인합니다.

검증 방식:

- base: `B1_plus_lead_interactions`
- fixed candidate: `specialist_pred >= 0.65 :: B1_plus_fundamental_all`
- calibration/holdout split을 월 단위로 나눔
- calibration에서 threshold/specialist를 고른 뒤 holdout month에서 평가
- fixed 0.65 rule도 같은 holdout에서 별도 평가

![Gate threshold robustness]({RAW}/01_gate_threshold_robustness.png)

## Full 2024 Gate Grid

{md_table(full_grid, 15, ["candidate", "score", "delta_score_vs_base", "ficr", "avg_nmae", "gate", "threshold", "specialist"])}

Fixed gate:

- candidate: `specialist_pred_ge_0.65::B1_plus_fundamental_all`
- full 2024 score: `{fixed['score']:.6f}`
- delta vs base: `{fixed['delta_score_vs_base']:.6f}`

## Split Robustness

{md_table(robust, 20, ["split", "policy", "candidate", "eval_score", "eval_delta_vs_base", "eval_ficr", "eval_avg_nmae", "gate", "threshold", "specialist"])}

Summary:

- fixed 0.65 mean holdout delta: `{fixed_robust['eval_delta_vs_base'].mean():.6f}`
- fixed 0.65 median holdout delta: `{fixed_robust['eval_delta_vs_base'].median():.6f}`
- fixed 0.65 positive splits: `{int((fixed_robust['eval_delta_vs_base'] > 0).sum())}/{len(fixed_robust)}`
- calibration-selected mean holdout delta: `{selected_robust['eval_delta_vs_base'].mean():.6f}`
- calibration-selected positive splits: `{int((selected_robust['eval_delta_vs_base'] > 0).sum())}/{len(selected_robust)}`

## Monthly Delta For Fixed Gate

{md_table(monthly, 12)}

## Interpretation

- If the fixed gate remains positive across most held-out month splits, the gate is robust enough to promote.
- If full-2024 is positive but split holdouts are mixed, the gate is promising but threshold selection is unstable.
- If calibration-selected gates do not transfer, threshold search itself is overfitting and needs a stronger OOF protocol.
"""
    (BASE / "issue_update.md").write_text(body, encoding="utf-8")


def main() -> None:
    setup()
    valid = valid_labels()
    base = read_pred(DEPLOY / "B1_plus_lead_interactions_valid_predictions.csv")
    specialists = {name: read_pred(FUND / f"{name}_valid_predictions.csv") for name in SPECIALISTS}
    full_grid, grid_preds = candidate_grid(base, specialists, valid)
    robust = robustness(base, valid, grid_preds, full_grid)
    best_fixed = grid_preds["specialist_pred_ge_0.65::B1_plus_fundamental_all"]
    monthly = monthly_delta(base, best_fixed, valid)
    plot(full_grid, robust, monthly)
    write_report(full_grid, robust, monthly)
    (OUT / "run_metadata.json").write_text(json.dumps({"base": "B1_plus_lead_interactions", "valid_year": 2024}, indent=2), encoding="utf-8")
    print(full_grid.head(12).to_string(index=False))
    print(robust.to_string(index=False))


if __name__ == "__main__":
    main()

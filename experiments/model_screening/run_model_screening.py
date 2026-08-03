from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

from generate_submission_candidates import build_train_test_features  # noqa: E402
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


BASELINE_ALPHA = 1.0275
BASELINE_LOCAL_SCORE = 0.6207259749
BASELINE_PUBLIC_LB = 0.6220908318
ALPHA_GRID = np.round(np.arange(0.9700, 1.0300001, 0.0025), 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-ensemble-models", type=int, default=6)
    return parser.parse_args()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def clip_pred(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(out[target], 0.0, CAPACITY[target])
    return out


def apply_alpha(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(out[target] * alpha, 0.0, CAPACITY[target])
    return out


def group_score(row: dict) -> float:
    return 0.5 * (1.0 - row["nmae"]) + 0.5 * row["ficr"]


def detailed_metrics(exp_id: str, y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series, alpha: float) -> tuple[dict, list[dict], list[dict]]:
    adjusted = apply_alpha(pred[TARGETS], alpha)
    metrics = evaluate_predictions(y_true, adjusted, timestamps)
    high_score = evaluate_high_generation(y_true, adjusted, timestamps)
    worst_month = min(row["score"] for row in metrics["monthly_rows"])
    row = {
        "model_id": exp_id,
        "alpha": alpha,
        "score": metrics["score"],
        "one_nmae": metrics["one_minus_nmae"],
        "avg_nmae": metrics["avg_nmae"],
        "ficr": metrics["ficr"],
        "worst_month_score": worst_month,
        "high_generation_score": high_score,
    }
    group_rows = []
    for item in metrics["group_rows"]:
        group_rows.append({"model_id": exp_id, **item, "group_score": group_score(item)})
    monthly_rows = [{"model_id": exp_id, **item} for item in metrics["monthly_rows"]]
    return row, group_rows, monthly_rows


def score_only(y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series, alpha: float) -> dict:
    adjusted = apply_alpha(pred[TARGETS], alpha)
    metrics = evaluate_predictions(y_true, adjusted, timestamps, include_monthly=False)
    return metrics


def best_alpha_metrics(y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series) -> tuple[float, dict]:
    best_alpha = 1.0
    best = None
    for alpha in ALPHA_GRID:
        metrics = score_only(y_true, pred, timestamps, float(alpha))
        if best is None or metrics["score"] > best["score"]:
            best = metrics
            best_alpha = float(alpha)
    assert best is not None
    return best_alpha, best


def train_sklearn_model(
    estimator_factory: Callable[[], object],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
) -> tuple[pd.DataFrame, float, float]:
    preds = pd.DataFrame(index=valid.index)
    x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    train_seconds = 0.0
    predict_seconds = 0.0
    for target in TARGETS:
        y = train[target]
        mask = y.notna()
        x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = y.loc[mask]
        estimator = estimator_factory()
        t0 = time.perf_counter()
        estimator.fit(x_train, y_train)
        train_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        preds[target] = estimator.predict(x_valid)
        predict_seconds += time.perf_counter() - t0
    return clip_pred(preds), train_seconds, predict_seconds


def model_specs(random_state: int) -> list[dict]:
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler

    specs = [
        {
            "model_id": "lgbm_l1_baseline",
            "family": "LightGBM",
            "status": "ready",
            "notes": "current B1 baseline parameters",
            "factory": lambda: LGBMRegressor(
                objective="regression_l1",
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=30,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        {
            "model_id": "lgbm_l2",
            "family": "LightGBM",
            "status": "ready",
            "notes": "same params, L2 objective",
            "factory": lambda: LGBMRegressor(
                objective="regression",
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=30,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        {
            "model_id": "lgbm_huber",
            "family": "LightGBM",
            "status": "ready",
            "notes": "same params, huber objective",
            "factory": lambda: LGBMRegressor(
                objective="huber",
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=30,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        },
        {
            "model_id": "hist_gbdt_l2",
            "family": "sklearn HistGradientBoosting",
            "status": "ready",
            "notes": "fast sklearn boosting diversity",
            "factory": lambda: HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=250,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=random_state,
            ),
        },
        {
            "model_id": "extra_trees",
            "family": "sklearn ExtraTrees",
            "status": "ready",
            "notes": "bagging diversity candidate",
            "factory": lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(
                    n_estimators=180,
                    max_features=0.65,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        },
        {
            "model_id": "random_forest",
            "family": "sklearn RandomForest",
            "status": "ready",
            "notes": "bagging diversity candidate",
            "factory": lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(
                    n_estimators=140,
                    max_features=0.65,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        },
        {
            "model_id": "ridge",
            "family": "linear",
            "status": "ready",
            "notes": "linear diversity diagnostic",
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)),
        },
        {
            "model_id": "elastic_net",
            "family": "linear",
            "status": "skipped",
            "notes": "deferred after runtime screen; linear Ridge already covers first-pass linear diversity",
            "factory": lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                ElasticNet(alpha=0.02, l1_ratio=0.05, max_iter=3000, random_state=random_state),
            ),
        },
        {
            "model_id": "huber",
            "family": "linear",
            "status": "skipped",
            "notes": "deferred after runtime screen; robust linear optimization is too slow for first screening",
            "factory": lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler(),
                HuberRegressor(alpha=0.001, max_iter=300),
            ),
        },
    ]
    if module_available("xgboost"):
        from xgboost import XGBRegressor

        specs.append(
            {
                "model_id": "xgboost_l1",
                "family": "XGBoost",
                "status": "ready",
                "notes": "installed wheel; absolute-error objective",
                "factory": lambda: XGBRegressor(
                    objective="reg:absoluteerror",
                    n_estimators=320,
                    learning_rate=0.045,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    min_child_weight=20,
                    reg_lambda=2.0,
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                ),
            }
        )
    if module_available("catboost"):
        from catboost import CatBoostRegressor

        specs.append(
            {
                "model_id": "catboost_mae",
                "family": "CatBoost",
                "status": "ready",
                "notes": "installed wheel; MAE loss",
                "factory": lambda: CatBoostRegressor(
                    loss_function="MAE",
                    iterations=450,
                    learning_rate=0.045,
                    depth=6,
                    l2_leaf_reg=5.0,
                    random_seed=random_state,
                    allow_writing_files=False,
                    verbose=False,
                    thread_count=-1,
                ),
            }
        )
    for missing_id, package, family in [
        ("tabm_feasibility", "tabm", "TabM"),
        ("tabpfn_feasibility", "tabpfn", "TabPFN"),
        ("autogluon_feasibility", "autogluon", "AutoGluon"),
    ]:
        if not module_available(package):
            specs.append(
                {
                    "model_id": missing_id,
                    "family": family,
                    "status": "skipped",
                    "notes": f"{package} not installed; deferred because dependency/runtime is heavy for first screening",
                    "factory": None,
                }
            )
    return specs


def residual_stats(
    model_id: str,
    y_true: pd.DataFrame,
    pred: pd.DataFrame,
    baseline_pred: pd.DataFrame,
) -> list[dict]:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        actual = y_true[target]
        mask = actual.notna() & (actual >= 0.10 * cap)
        residual = actual[mask] - pred.loc[mask, target]
        baseline_residual = actual[mask] - baseline_pred.loc[mask, target]
        err = (pred.loc[mask, target] - actual[mask]).abs() / cap
        base_err = (baseline_pred.loc[mask, target] - actual[mask]).abs() / cap
        improved_boundary = (err <= 0.08) & (base_err > 0.08)
        rows.append(
            {
                "model_id": model_id,
                "target": target,
                "pearson": float(residual.corr(baseline_residual, method="pearson")),
                "spearman": float(residual.corr(baseline_residual, method="spearman")),
                "both_outside_8pct_rate": float(((err > 0.08) & (base_err > 0.08)).mean()),
                "model_better_abs_error_rate": float((err < base_err).mean()),
                "actual_weighted_boundary_gain": float(actual[mask][improved_boundary].sum()),
            }
        )
    return rows


def better_slices(model_id: str, monthly: pd.DataFrame, group: pd.DataFrame, baseline_id: str) -> tuple[int, int]:
    cur_m = monthly[monthly["model_id"] == model_id][["month", "score"]]
    base_m = monthly[monthly["model_id"] == baseline_id][["month", "score"]].rename(columns={"score": "base_score"})
    months_better = int((cur_m.merge(base_m, on="month")["score"] > cur_m.merge(base_m, on="month")["base_score"]).sum())
    cur_g = group[group["model_id"] == model_id][["target", "group_score"]]
    base_g = group[group["model_id"] == baseline_id][["target", "group_score"]].rename(columns={"group_score": "base_score"})
    groups_better = int((cur_g.merge(base_g, on="target")["group_score"] > cur_g.merge(base_g, on="target")["base_score"]).sum())
    return months_better, groups_better


def train_final_prediction_for_spec(spec: dict, data_dir: Path, random_state: int) -> pd.DataFrame:
    labels = read_labels(data_dir)
    train_features, test_features = build_train_test_features(data_dir)
    data = train_features.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_label"))
    feature_cols = [c for c in train_features.columns if c != "forecast_kst_dtm"]
    preds = pd.DataFrame({"forecast_kst_dtm": test_features["forecast_kst_dtm"]})
    x_test = test_features[feature_cols].replace([np.inf, -np.inf], np.nan)
    for target in TARGETS:
        years = [2022, 2023, 2024] if target != "kpx_group_3" else [2023, 2024]
        train = data[data["year"].isin(years)].copy()
        mask = train[target].notna()
        estimator = spec["factory"]()
        estimator.fit(train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan), train.loc[mask, target])
        preds[target] = estimator.predict(x_test)
    return clip_pred(preds)


def make_submission(data_dir: Path, final_pred: pd.DataFrame, alpha: float, output: Path) -> dict:
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    adjusted = pd.concat([final_pred[["forecast_kst_dtm"]], apply_alpha(final_pred[TARGETS], alpha)], axis=1)
    submission = sample[["forecast_id", "forecast_kst_dtm"]].merge(adjusted, on="forecast_kst_dtm", how="left")
    submission.to_csv(output, index=False, encoding="utf-8-sig")
    return {
        "path": str(output),
        "rows": int(len(submission)),
        "missing_predictions": int(submission[TARGETS].isna().sum().sum()),
        "min_prediction": float(submission[TARGETS].min().min()),
        "max_prediction": float(submission[TARGETS].max().max()),
        "columns_match": list(submission.columns) == list(sample.columns),
        "ids_match": bool(submission["forecast_id"].equals(sample["forecast_id"])),
        "dt_match": bool(submission["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"])),
    }


def evaluate_ensemble(
    ens_id: str,
    member_ids: list[str],
    weights: list[float],
    pred_bank: dict[str, pd.DataFrame],
    y_true: pd.DataFrame,
    timestamps: pd.Series,
) -> dict:
    pred = pd.DataFrame(index=next(iter(pred_bank.values())).index)
    for target in TARGETS:
        pred[target] = 0.0
        for model_id, weight in zip(member_ids, weights):
            pred[target] += pred_bank[model_id][target] * weight
    best_alpha, best = best_alpha_metrics(y_true, pred, timestamps)
    row = {
        "ensemble_id": ens_id,
        "model_id": ens_id,
        "members": "+".join(member_ids),
        "weights": json.dumps(dict(zip(member_ids, weights)), ensure_ascii=False),
        "best_alpha": best_alpha,
        "alpha_score": best["score"],
        "alpha_1_nmae": best["one_minus_nmae"],
        "alpha_ficr": best["ficr"],
        "avg_nmae": best["avg_nmae"],
        "worst_month_score": np.nan,
        "high_generation_score": np.nan,
    }
    return row


def ensemble_prediction(member_weights: dict[str, float], pred_bank: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pred = pd.DataFrame(index=next(iter(pred_bank.values())).index)
    for target in TARGETS:
        pred[target] = 0.0
        for model_id, weight in member_weights.items():
            pred[target] += pred_bank[model_id][target] * float(weight)
    return clip_pred(pred)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    pred_dir = out_dir / "predictions"
    results_dir = out_dir / "results"
    submissions_dir = out_dir / "submissions"
    for path in [pred_dir, results_dir, submissions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    existing_model_summary = results_dir / "model_summary.csv"
    existing_ensemble_summary = results_dir / "ensemble_summary.csv"
    if existing_model_summary.exists() and existing_ensemble_summary.exists():
        model_summary = pd.read_csv(existing_model_summary, encoding="utf-8-sig")
        ensemble_df = pd.read_csv(existing_ensemble_summary, encoding="utf-8-sig")
        best_model = model_summary[model_summary["status"] == "ok"].sort_values("alpha_score", ascending=False).iloc[0].to_dict()
        best_ensemble = None if ensemble_df.empty else ensemble_df.sort_values("alpha_score", ascending=False).iloc[0].to_dict()
        candidate_kind = "model"
        candidate = best_model
        if best_ensemble is not None and best_ensemble["alpha_score"] > best_model["alpha_score"]:
            candidate_kind = "ensemble"
            candidate = best_ensemble
        submission_info = None
        best_params = {
            "baseline": {
                "local_score": BASELINE_LOCAL_SCORE,
                "public_lb": BASELINE_PUBLIC_LB,
                "alpha": BASELINE_ALPHA,
            },
            "best_model": best_model,
            "best_ensemble": best_ensemble,
            "selected_candidate_kind": candidate_kind,
            "selected_candidate": candidate,
            "submission_info": submission_info,
            "package_status": {
                "xgboost": module_available("xgboost"),
                "catboost": module_available("catboost"),
                "tabm": module_available("tabm"),
                "tabpfn": module_available("tabpfn"),
                "autogluon": module_available("autogluon"),
            },
            "resume_note": "Loaded existing model/ensemble summaries and regenerated conclusion.",
        }
        (results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        top_models = model_summary.sort_values("alpha_score", ascending=False).head(12)
        top_ensembles = ensemble_df.sort_values("alpha_score", ascending=False).head(12) if not ensemble_df.empty else pd.DataFrame()
        lines = [
            "# Model Screening Results",
            "",
            "## Baseline",
            "",
            f"- local baseline score: {BASELINE_LOCAL_SCORE}",
            f"- Public LB baseline: {BASELINE_PUBLIC_LB}",
            f"- fixed alpha baseline: {BASELINE_ALPHA}",
            "",
            "## Top Models",
            "",
            top_models[[
                "model_id",
                "model_family",
                "raw_score",
                "best_alpha",
                "alpha_score",
                "delta_vs_baseline",
                "alpha_ficr",
                "worst_month_score",
                "high_generation_score",
                "months_better_than_lgbm",
                "groups_better_than_lgbm",
                "status",
            ]].to_markdown(index=False),
            "",
            "## Top Ensembles",
            "",
            top_ensembles[["ensemble_id", "members", "best_alpha", "alpha_score", "delta_vs_baseline", "alpha_ficr", "worst_month_score", "high_generation_score"]].to_markdown(index=False) if not top_ensembles.empty else "No ensembles were evaluated.",
            "",
            "## Selected Candidate",
            "",
            json.dumps({"kind": candidate_kind, "candidate": candidate, "submission": submission_info}, indent=2, ensure_ascii=False, default=str),
            "",
            "## Resume Note",
            "",
            "Existing result CSV files were reused. Delete `experiments/model_screening/results/*.csv` to force a full rerun.",
        ]
        (results_dir / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")
        print((results_dir / "conclusion.md").read_text(encoding="utf-8"))
        return

    labels = read_labels(args.data_dir)
    features = build_feature_frame(args.data_dir, "B1_aggregate")
    data = merge_features_labels(features, labels)
    train = data[data["year"].isin([2022, 2023])].copy().reset_index(drop=True)
    valid = data[data["year"] == 2024].copy().reset_index(drop=True)
    feature_cols = [c for c in features.columns if c != "forecast_kst_dtm" and c in valid.columns]
    y_true = valid[TARGETS]
    timestamps = valid["forecast_kst_dtm"]

    specs = model_specs(args.random_state)
    model_rows = []
    group_rows = []
    monthly_rows = []
    residual_rows = []
    pred_bank: dict[str, pd.DataFrame] = {}

    baseline_raw_pred = None
    for spec in specs:
        model_id = spec["model_id"]
        if spec["status"] != "ready":
            model_rows.append(
                {
                    "model_id": model_id,
                    "model_family": spec["family"],
                    "status": spec["status"],
                    "notes": spec["notes"],
                    "raw_score": np.nan,
                    "alpha_score": np.nan,
                }
            )
            continue
        try:
            pred_path = pred_dir / f"{model_id}_valid_raw.csv"
            if pred_path.exists():
                pred = pd.read_csv(pred_path, encoding="utf-8-sig")
                train_seconds = np.nan
                predict_seconds = np.nan
                total_seconds = 0.0
                cache_note = "loaded cached prediction"
            else:
                t0 = time.perf_counter()
                pred, train_seconds, predict_seconds = train_sklearn_model(
                    spec["factory"], train, valid, feature_cols, args.random_state
                )
                total_seconds = time.perf_counter() - t0
                pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
                cache_note = "trained in this run"
            pred_bank[model_id] = pred
            raw, raw_group, raw_monthly = detailed_metrics(model_id, y_true, pred, timestamps, 1.0)
            best_alpha, alpha_metrics = best_alpha_metrics(y_true, pred, timestamps)
            alpha_detail, alpha_group, alpha_monthly = detailed_metrics(f"{model_id}__alpha", y_true, pred, timestamps, best_alpha)
            apply_alpha(pred, best_alpha).to_csv(pred_dir / f"{model_id}_valid_alpha.csv", index=False, encoding="utf-8-sig")
            group_rows.extend(alpha_group)
            monthly_rows.extend(alpha_monthly)
            if model_id == "lgbm_l1_baseline":
                baseline_raw_pred = pred

            row = {
                "model_id": model_id,
                "model_family": spec["family"],
                "raw_score": raw["score"],
                "raw_1_nmae": raw["one_nmae"],
                "raw_ficr": raw["ficr"],
                "best_alpha": best_alpha,
                "alpha_score": alpha_metrics["score"],
                "alpha_1_nmae": alpha_metrics["one_minus_nmae"],
                "alpha_ficr": alpha_metrics["ficr"],
                "delta_vs_baseline": alpha_metrics["score"] - BASELINE_LOCAL_SCORE,
                "worst_month_score": alpha_detail["worst_month_score"],
                "high_generation_score": alpha_detail["high_generation_score"],
                "group_1_score": group_score(alpha_metrics["group_rows"][0]),
                "group_2_score": group_score(alpha_metrics["group_rows"][1]),
                "group_3_score": group_score(alpha_metrics["group_rows"][2]),
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": total_seconds,
                "status": "ok",
                "notes": f"{spec['notes']}; {cache_note}",
            }
            model_rows.append(row)
        except Exception as exc:
            model_rows.append(
                {
                    "model_id": model_id,
                    "model_family": spec["family"],
                    "status": "failed",
                    "notes": f"{spec['notes']}; failed: {type(exc).__name__}: {str(exc)[:300]}",
                    "raw_score": np.nan,
                    "alpha_score": np.nan,
                }
            )

    if baseline_raw_pred is None:
        raise RuntimeError("lgbm_l1_baseline did not run; cannot compute residual/ensemble baseline")

    for model_id, pred in pred_bank.items():
        residual_rows.extend(residual_stats(model_id, y_true, pred, baseline_raw_pred))

    model_summary = pd.DataFrame(model_rows)
    group_df = pd.DataFrame(group_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    residual_df = pd.DataFrame(residual_rows)

    for idx, row in model_summary[model_summary["status"] == "ok"].iterrows():
        months_better, groups_better = better_slices(f"{row['model_id']}__alpha", monthly_df, group_df, "lgbm_l1_baseline__alpha")
        model_summary.loc[idx, "months_better_than_lgbm"] = months_better
        model_summary.loc[idx, "groups_better_than_lgbm"] = groups_better

    ensemble_rows = []
    ok_models = model_summary[model_summary["status"] == "ok"].sort_values("alpha_score", ascending=False)["model_id"].tolist()
    selected_models = ok_models[: args.max_ensemble_models]

    named_combos = [
        ["lgbm_l1_baseline", "catboost_mae"],
        ["lgbm_l1_baseline", "xgboost_l1"],
        ["lgbm_l1_baseline", "catboost_mae", "xgboost_l1"],
    ]
    for combo in named_combos:
        if all(model_id in pred_bank for model_id in combo):
            weights = [1.0 / len(combo)] * len(combo)
            ensemble_rows.append(evaluate_ensemble("avg_" + "_".join(combo), combo, weights, pred_bank, y_true, timestamps))

    for a, b in itertools.combinations(selected_models, 2):
        for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            ensemble_rows.append(evaluate_ensemble(f"w2_{a}_{b}_{w:.2f}", [a, b], [float(w), float(1.0 - w)], pred_bank, y_true, timestamps))

    for combo in itertools.combinations(selected_models[:5], 3):
        for w1 in np.round(np.arange(0.0, 1.0001, 0.1), 1):
            for w2 in np.round(np.arange(0.0, 1.0 - w1 + 0.0001, 0.1), 1):
                w3 = round(1.0 - float(w1) - float(w2), 1)
                if w3 < -1e-9:
                    continue
                ensemble_rows.append(evaluate_ensemble(f"w3_{'_'.join(combo)}_{w1:.1f}_{w2:.1f}_{w3:.1f}", list(combo), [float(w1), float(w2), float(w3)], pred_bank, y_true, timestamps))

    ensemble_df = pd.DataFrame(ensemble_rows)
    if not ensemble_df.empty:
        ensemble_df["delta_vs_baseline"] = ensemble_df["alpha_score"] - BASELINE_LOCAL_SCORE
        top_ensemble_indices = ensemble_df.sort_values("alpha_score", ascending=False).head(50).index
        for idx in top_ensemble_indices:
            weights = json.loads(ensemble_df.loc[idx, "weights"])
            pred = ensemble_prediction(weights, pred_bank)
            detail, _, _ = detailed_metrics(str(ensemble_df.loc[idx, "ensemble_id"]), y_true, pred, timestamps, float(ensemble_df.loc[idx, "best_alpha"]))
            ensemble_df.loc[idx, "worst_month_score"] = detail["worst_month_score"]
            ensemble_df.loc[idx, "high_generation_score"] = detail["high_generation_score"]

    model_summary.to_csv(results_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(results_dir / "group_scores.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(results_dir / "monthly_scores.csv", index=False, encoding="utf-8-sig")
    residual_df.to_csv(results_dir / "residual_correlation.csv", index=False, encoding="utf-8-sig")
    ensemble_df.to_csv(results_dir / "ensemble_summary.csv", index=False, encoding="utf-8-sig")

    best_model = model_summary[model_summary["status"] == "ok"].sort_values("alpha_score", ascending=False).iloc[0].to_dict()
    best_ensemble = None if ensemble_df.empty else ensemble_df.sort_values("alpha_score", ascending=False).iloc[0].to_dict()
    candidate_kind = "model"
    candidate = best_model
    if best_ensemble is not None and best_ensemble["alpha_score"] > best_model["alpha_score"]:
        candidate_kind = "ensemble"
        candidate = best_ensemble

    submission_info = None
    if candidate["alpha_score"] >= BASELINE_LOCAL_SCORE + 0.0015:
        if candidate_kind == "model":
            spec = next(s for s in specs if s["model_id"] == candidate["model_id"])
            final_pred = train_final_prediction_for_spec(spec, args.data_dir, args.random_state)
            alpha = float(candidate["best_alpha"])
            output = submissions_dir / f"submission_{candidate['model_id']}_alpha.csv"
        else:
            member_weights = json.loads(candidate["weights"])
            final_parts = []
            for member in member_weights:
                spec = next(s for s in specs if s["model_id"] == member)
                final_parts.append((member_weights[member], train_final_prediction_for_spec(spec, args.data_dir, args.random_state)))
            final_pred = pd.DataFrame({"forecast_kst_dtm": final_parts[0][1]["forecast_kst_dtm"]})
            for target in TARGETS:
                final_pred[target] = 0.0
                for weight, part in final_parts:
                    final_pred[target] += float(weight) * part[target]
            alpha = float(candidate["best_alpha"])
            output = submissions_dir / "submission_best_ensemble_alpha.csv"
        submission_info = make_submission(args.data_dir, final_pred, alpha, output)

    best_params = {
        "baseline": {
            "local_score": BASELINE_LOCAL_SCORE,
            "public_lb": BASELINE_PUBLIC_LB,
            "alpha": BASELINE_ALPHA,
        },
        "best_model": best_model,
        "best_ensemble": best_ensemble,
        "selected_candidate_kind": candidate_kind,
        "selected_candidate": candidate,
        "submission_info": submission_info,
        "package_status": {
            "xgboost": module_available("xgboost"),
            "catboost": module_available("catboost"),
            "tabm": module_available("tabm"),
            "tabpfn": module_available("tabpfn"),
            "autogluon": module_available("autogluon"),
        },
    }
    (results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    top_models = model_summary.sort_values("alpha_score", ascending=False).head(12)
    top_ensembles = ensemble_df.sort_values("alpha_score", ascending=False).head(12) if not ensemble_df.empty else pd.DataFrame()
    lines = [
        "# Model Screening Results",
        "",
        "## Baseline",
        "",
        f"- local baseline score: {BASELINE_LOCAL_SCORE}",
        f"- Public LB baseline: {BASELINE_PUBLIC_LB}",
        f"- fixed alpha baseline: {BASELINE_ALPHA}",
        "",
        "## Top Models",
        "",
        top_models[[
            "model_id",
            "model_family",
            "raw_score",
            "best_alpha",
            "alpha_score",
            "delta_vs_baseline",
            "alpha_ficr",
            "worst_month_score",
            "high_generation_score",
            "months_better_than_lgbm",
            "groups_better_than_lgbm",
            "status",
        ]].to_markdown(index=False),
        "",
        "## Top Ensembles",
        "",
        top_ensembles[["ensemble_id", "members", "best_alpha", "alpha_score", "delta_vs_baseline", "alpha_ficr", "worst_month_score", "high_generation_score"]].to_markdown(index=False) if not top_ensembles.empty else "No ensembles were evaluated.",
        "",
        "## Selected Candidate",
        "",
        json.dumps({"kind": candidate_kind, "candidate": candidate, "submission": submission_info}, indent=2, ensure_ascii=False, default=str),
    ]
    (results_dir / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")
    print((results_dir / "conclusion.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

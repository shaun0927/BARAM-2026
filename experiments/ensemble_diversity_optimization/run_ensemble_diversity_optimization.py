from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import subprocess
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
)


BASELINE_ALPHA = 1.0275
BASELINE_LOCAL_SCORE = 0.6207259749
BASELINE_PUBLIC_LB = 0.6220908318
ACCEPT_SCORE = BASELINE_LOCAL_SCORE + 0.0015
CURRENT_ANCHOR_SCORE = 0.6260900872
CURRENT_ANCHOR_FICR = 0.3790961261
CURRENT_ANCHOR_WORST_MONTH = 0.6012469333
CURRENT_ANCHOR_WEIGHTS = {
    "catboost_quantile_055": 0.60,
    "lgbm_l1_baseline": 0.20,
    "extra_trees_wide": 0.20,
}
NEXT_ACCEPT_SCORE = 0.6264
STRONG_ACCEPT_SCORE = 0.6268
HIGH_CONFIDENCE_SCORE = 0.6271
ALPHA_GRID = np.round(np.arange(0.9700, 1.0300001, 0.0025), 4)
W4_WINDOW = Window("W4_2022_2023_to_2024", (2022, 2023), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--phase", choices=["all", "main", "autogluon", "pycaret", "misc", "evaluate"], default="all")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--time-limit-per-target", type=int, default=240)
    parser.add_argument("--sample-rows", type=int, default=7000)
    parser.add_argument("--neural-epochs", type=int, default=35)
    parser.add_argument("--max-ensemble-models", type=int, default=14)
    parser.add_argument("--no-subprocess", action="store_true")
    return parser.parse_args()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def ensure_dirs(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    pred_dir = out_dir / "predictions"
    results_dir = out_dir / "results"
    submissions_dir = out_dir / "submissions"
    artifacts_dir = out_dir / "artifacts"
    for path in [pred_dir, results_dir, submissions_dir, artifacts_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return pred_dir, results_dir, submissions_dir, artifacts_dir


def load_w4_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    labels = read_labels(data_dir)
    features = build_feature_frame(data_dir, "B1_aggregate")
    data = merge_features_labels(features, labels)
    train = data[data["year"].isin([2022, 2023])].copy().reset_index(drop=True)
    valid = data[data["year"] == 2024].copy().reset_index(drop=True)
    feature_cols = [c for c in features.columns if c != "forecast_kst_dtm" and c in valid.columns]
    return labels, train, valid, feature_cols


def clip_pred(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(pd.to_numeric(out[target], errors="coerce").fillna(0.0), 0.0, CAPACITY[target])
    return out


def apply_alpha(pred: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = pred.copy()
    for target in TARGETS:
        out[target] = np.clip(out[target] * alpha, 0.0, CAPACITY[target])
    return out


def group_score(row: dict) -> float:
    return 0.5 * (1.0 - row["nmae"]) + 0.5 * row["ficr"]


def score_only(y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series, alpha: float) -> dict:
    adjusted = apply_alpha(pred[TARGETS], alpha)
    return evaluate_predictions(y_true, adjusted, timestamps, include_monthly=False)


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


def detailed_metrics(exp_id: str, y_true: pd.DataFrame, pred: pd.DataFrame, timestamps: pd.Series, alpha: float) -> tuple[dict, list[dict], list[dict]]:
    adjusted = apply_alpha(pred[TARGETS], alpha)
    metrics = evaluate_predictions(y_true, adjusted, timestamps)
    high_score = evaluate_high_generation(y_true, adjusted, timestamps)
    worst_month = min(row["score"] for row in metrics["monthly_rows"])
    row = {
        "model_id": exp_id,
        "alpha": alpha,
        "score": metrics["score"],
        "one_minus_nmae": metrics["one_minus_nmae"],
        "avg_nmae": metrics["avg_nmae"],
        "ficr": metrics["ficr"],
        "worst_month_score": worst_month,
        "high_generation_score": high_score,
    }
    group_rows = [{"model_id": exp_id, **item, "group_score": group_score(item)} for item in metrics["group_rows"]]
    monthly_rows = [{"model_id": exp_id, **item} for item in metrics["monthly_rows"]]
    return row, group_rows, monthly_rows


def sample_xy(
    x: pd.DataFrame,
    y: pd.Series,
    sample_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_rows <= 0 or len(x) <= sample_rows:
        return x, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(np.arange(len(x)), size=sample_rows, replace=False)
    return x.iloc[idx], y.iloc[idx]


def train_sklearn_model(
    model_id: str,
    estimator_factory: Callable[[], object],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
    sample_rows: int = 0,
) -> tuple[pd.DataFrame, float, float]:
    preds = pd.DataFrame(index=valid.index)
    x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    train_seconds = 0.0
    predict_seconds = 0.0
    for target_idx, target in enumerate(TARGETS):
        y = train[target]
        mask = y.notna()
        x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = y.loc[mask]
        x_fit, y_fit = sample_xy(x_train, y_train, sample_rows, random_state + target_idx)
        estimator = estimator_factory()
        t0 = time.perf_counter()
        estimator.fit(x_fit, y_fit)
        train_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        preds[target] = estimator.predict(x_valid)
        predict_seconds += time.perf_counter() - t0
    return clip_pred(preds), train_seconds, predict_seconds


def main_model_specs(random_state: int, sample_rows: int) -> list[dict]:
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge, SGDRegressor
    from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from sklearn.svm import LinearSVR, SVR

    specs = [
        {
            "model_id": "lgbm_l1_baseline",
            "family": "LightGBM",
            "notes": "current B1/W4 baseline model; included as anchor",
            "sample_rows": 0,
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
            "model_id": "elastic_net",
            "family": "linear",
            "notes": "previously deferred; first-pass linear sparse baseline",
            "sample_rows": min(sample_rows, 6000),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNet(alpha=0.04, l1_ratio=0.08, max_iter=1200, tol=1e-3, random_state=random_state)),
        },
        {
            "model_id": "huber_regressor",
            "family": "linear",
            "notes": "previously deferred; robust linear baseline",
            "sample_rows": min(sample_rows, 3500),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), HuberRegressor(alpha=0.001, max_iter=120, epsilon=1.35)),
        },
        {
            "model_id": "sgd_huber",
            "family": "linear",
            "notes": "fast robust linear SGD surrogate",
            "sample_rows": 0,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SGDRegressor(loss="huber", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=random_state)),
        },
        {
            "model_id": "knn_distance",
            "family": "neighbors",
            "notes": "distance-weighted KNN with robust scaling",
            "sample_rows": min(sample_rows, 4500),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), KNeighborsRegressor(n_neighbors=48, weights="distance", n_jobs=-1)),
        },
        {
            "model_id": "radius_neighbors",
            "family": "neighbors",
            "notes": "radius neighbor feasibility with fallback outlier label",
            "sample_rows": min(sample_rows, 3500),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), RadiusNeighborsRegressor(radius=4.0, weights="distance", n_jobs=-1)),
        },
        {
            "model_id": "linear_svr",
            "family": "svm",
            "notes": "linear SVR sampled first-pass",
            "sample_rows": min(sample_rows, 4500),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearSVR(C=1.0, epsilon=50.0, random_state=random_state, max_iter=3000)),
        },
        {
            "model_id": "nystroem_ridge",
            "family": "kernel",
            "notes": "kernel approximation + ridge; nonlinear low-rank diversity",
            "sample_rows": min(sample_rows, 4500),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Nystroem(kernel="rbf", gamma=0.015, n_components=260, random_state=random_state), Ridge(alpha=30.0)),
        },
        {
            "model_id": "svr_rbf_sampled",
            "family": "svm",
            "notes": "sampled RBF SVR feasibility",
            "sample_rows": min(sample_rows, 2500),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=10.0, epsilon=80.0, gamma="scale", cache_size=800)),
        },
        {
            "model_id": "mlp_regressor",
            "family": "neural_sklearn",
            "notes": "sklearn MLP first-pass tabular neural baseline",
            "sample_rows": min(sample_rows, 7000),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 48), activation="relu", alpha=1e-3, batch_size=512, learning_rate_init=1e-3, early_stopping=True, max_iter=55, random_state=random_state)),
        },
        {
            "model_id": "hist_gbdt_absolute",
            "family": "sklearn HistGradientBoosting",
            "notes": "sklearn boosting absolute-error variant",
            "sample_rows": 0,
            "factory": lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.05, random_state=random_state),
        },
        {
            "model_id": "gradient_boosting_quantile",
            "family": "sklearn GradientBoosting",
            "notes": "quantile boosting q=0.55 for underprediction-bias diversity",
            "sample_rows": sample_rows,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(loss="quantile", alpha=0.55, n_estimators=180, learning_rate=0.045, max_depth=3, random_state=random_state)),
        },
        {
            "model_id": "extra_trees_wide",
            "family": "sklearn ExtraTrees",
            "notes": "bagging diversity; wider than #4 first run",
            "sample_rows": 0,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=180, max_features=0.8, min_samples_leaf=2, random_state=random_state, n_jobs=-1)),
        },
        {
            "model_id": "random_forest_wide",
            "family": "sklearn RandomForest",
            "notes": "bagging diversity; wider than #4 first run",
            "sample_rows": 0,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=140, max_features=0.8, min_samples_leaf=3, random_state=random_state, n_jobs=-1)),
        },
        {
            "model_id": "lgbm_l1_leaf63_seed42",
            "family": "LightGBM",
            "notes": "ensemble-oriented L1 variant: larger leaves",
            "sample_rows": 0,
            "factory": lambda: LGBMRegressor(objective="regression_l1", n_estimators=520, learning_rate=0.035, num_leaves=63, subsample=0.88, colsample_bytree=0.86, min_child_samples=45, random_state=random_state, n_jobs=-1, verbose=-1),
        },
        {
            "model_id": "lgbm_l1_leaf15_regularized",
            "family": "LightGBM",
            "notes": "ensemble-oriented L1 variant: conservative regularized tree",
            "sample_rows": 0,
            "factory": lambda: LGBMRegressor(objective="regression_l1", n_estimators=520, learning_rate=0.035, num_leaves=15, subsample=0.82, colsample_bytree=0.78, min_child_samples=80, reg_lambda=3.0, random_state=random_state + 17, n_jobs=-1, verbose=-1),
        },
        {
            "model_id": "lgbm_l1_seed777",
            "family": "LightGBM",
            "notes": "ensemble-oriented L1 seed variant",
            "sample_rows": 0,
            "factory": lambda: LGBMRegressor(objective="regression_l1", n_estimators=430, learning_rate=0.04, num_leaves=31, subsample=0.86, colsample_bytree=0.86, min_child_samples=35, random_state=777, n_jobs=-1, verbose=-1),
        },
        {
            "model_id": "extra_trees_leaf5_sqrt",
            "family": "sklearn ExtraTrees",
            "notes": "diversity ExtraTrees variant: sqrt features and larger leaves",
            "sample_rows": 0,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=260, max_features="sqrt", min_samples_leaf=5, bootstrap=False, random_state=random_state + 101, n_jobs=-1)),
        },
        {
            "model_id": "extra_trees_bootstrap_leaf3",
            "family": "sklearn ExtraTrees",
            "notes": "diversity ExtraTrees variant: bootstrap",
            "sample_rows": 0,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(n_estimators=260, max_features=0.7, min_samples_leaf=3, bootstrap=True, random_state=random_state + 202, n_jobs=-1)),
        },
        {
            "model_id": "rf_leaf5_sqrt",
            "family": "sklearn RandomForest",
            "notes": "diversity RandomForest variant: sqrt features",
            "sample_rows": 0,
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=220, max_features="sqrt", min_samples_leaf=5, random_state=random_state + 303, n_jobs=-1)),
        },
        {
            "model_id": "hist_gbdt_quantile_055",
            "family": "sklearn HistGradientBoosting",
            "notes": "HistGBDT quantile q=0.55 diversity variant",
            "sample_rows": 0,
            "factory": lambda: HistGradientBoostingRegressor(loss="quantile", quantile=0.55, max_iter=320, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.05, random_state=random_state),
        },
        {
            "model_id": "hist_gbdt_l2_leaf63",
            "family": "sklearn HistGradientBoosting",
            "notes": "HistGBDT squared-error variant with larger leaf budget",
            "sample_rows": 0,
            "factory": lambda: HistGradientBoostingRegressor(loss="squared_error", max_iter=320, learning_rate=0.04, max_leaf_nodes=63, l2_regularization=0.2, random_state=random_state + 404),
        },
        {
            "model_id": "elastic_net_stronger_l1",
            "family": "linear",
            "notes": "linear diversity variant with stronger L1 ratio",
            "sample_rows": min(sample_rows, 9000),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), ElasticNet(alpha=0.015, l1_ratio=0.35, max_iter=2000, tol=1e-3, random_state=random_state)),
        },
        {
            "model_id": "huber_epsilon_18",
            "family": "linear",
            "notes": "robust linear diversity variant with wider Huber epsilon",
            "sample_rows": min(sample_rows, 6000),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), HuberRegressor(alpha=0.0005, max_iter=180, epsilon=1.8)),
        },
        {
            "model_id": "mlp_regressor_medium",
            "family": "neural_sklearn",
            "notes": "medium sklearn MLP diversity variant",
            "sample_rows": min(sample_rows, 12000),
            "factory": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128, 32), activation="relu", alpha=5e-4, batch_size=512, learning_rate_init=7e-4, early_stopping=True, validation_fraction=0.12, max_iter=110, random_state=random_state + 515)),
        },
    ]
    if module_available("xgboost"):
        from xgboost import XGBRegressor

        specs.extend(
            [
                {
                    "model_id": "xgboost_pseudohuber",
                    "family": "XGBoost",
                    "notes": "XGBoost pseudo-Huber objective first-pass",
                    "sample_rows": 0,
                    "factory": lambda: XGBRegressor(objective="reg:pseudohubererror", n_estimators=320, learning_rate=0.045, max_depth=5, subsample=0.88, colsample_bytree=0.88, min_child_weight=15, reg_lambda=3.0, random_state=random_state, n_jobs=-1, tree_method="hist"),
                },
                {
                    "model_id": "xgboost_squarederror",
                    "family": "XGBoost",
                    "notes": "XGBoost squared-error diversity baseline",
                    "sample_rows": 0,
                    "factory": lambda: XGBRegressor(objective="reg:squarederror", n_estimators=320, learning_rate=0.045, max_depth=5, subsample=0.88, colsample_bytree=0.88, min_child_weight=15, reg_lambda=3.0, random_state=random_state, n_jobs=-1, tree_method="hist"),
                },
            ]
        )
    if module_available("catboost"):
        from catboost import CatBoostRegressor

        specs.extend(
            [
                {
                    "model_id": "catboost_rmse",
                    "family": "CatBoost",
                    "notes": "CatBoost RMSE first-pass diversity",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="RMSE", iterations=500, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, random_seed=random_state, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_055",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.55 first-pass",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.55", iterations=500, learning_rate=0.045, depth=6, l2_leaf_reg=6.0, random_seed=random_state, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_050_depth6",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.50 ensemble-oriented variant",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.50", iterations=650, learning_rate=0.04, depth=6, l2_leaf_reg=6.0, random_seed=random_state + 11, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_0525_depth6",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.525 ensemble-oriented variant",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.525", iterations=650, learning_rate=0.04, depth=6, l2_leaf_reg=6.0, random_seed=random_state + 12, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_0575_depth6",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.575 ensemble-oriented variant",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.575", iterations=650, learning_rate=0.04, depth=6, l2_leaf_reg=6.0, random_seed=random_state + 13, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_060_depth6",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.60 ensemble-oriented variant",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.60", iterations=650, learning_rate=0.04, depth=6, l2_leaf_reg=6.0, random_seed=random_state + 14, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_055_depth4_reg",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.55 shallow regularized variant",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.55", iterations=750, learning_rate=0.035, depth=4, l2_leaf_reg=10.0, random_seed=random_state + 15, allow_writing_files=False, verbose=False, thread_count=-1),
                },
                {
                    "model_id": "catboost_quantile_055_depth8",
                    "family": "CatBoost",
                    "notes": "CatBoost Quantile q=0.55 deeper variant",
                    "sample_rows": 0,
                    "factory": lambda: CatBoostRegressor(loss_function="Quantile:alpha=0.55", iterations=550, learning_rate=0.035, depth=8, l2_leaf_reg=8.0, random_seed=random_state + 16, allow_writing_files=False, verbose=False, thread_count=-1),
                },
            ]
        )
    return specs


def run_flaml(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str], pred_dir: Path, random_state: int, time_limit: int) -> list[dict]:
    rows = []
    if (pred_dir / "flaml_automl_valid_raw.csv").exists():
        return [{"model_id": "flaml_automl", "model_family": "FLAML AutoML", "status": "ok", "notes": "loaded cached prediction", "train_seconds": np.nan, "predict_seconds": np.nan}]
    if not module_available("flaml"):
        return [{"model_id": "flaml_automl", "model_family": "FLAML AutoML", "status": "skipped", "notes": "flaml unavailable"}]
    from flaml import AutoML

    preds = pd.DataFrame(index=valid.index)
    x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
    leader = {}
    train_seconds = 0.0
    predict_seconds = 0.0
    for target in TARGETS:
        y = train[target]
        mask = y.notna()
        automl = AutoML()
        settings = {
            "time_budget": time_limit,
            "task": "regression",
            "metric": "mae",
            "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree", "histgb"],
            "seed": random_state,
            "verbose": 0,
            "n_jobs": -1,
        }
        t0 = time.perf_counter()
        automl.fit(train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan), y.loc[mask], **settings)
        train_seconds += time.perf_counter() - t0
        leader[target] = {"best_estimator": str(automl.best_estimator), "best_config": automl.best_config}
        t0 = time.perf_counter()
        preds[target] = automl.predict(x_valid)
        predict_seconds += time.perf_counter() - t0
    preds = clip_pred(preds)
    preds.to_csv(pred_dir / "flaml_automl_valid_raw.csv", index=False, encoding="utf-8-sig")
    (pred_dir.parent / "artifacts" / "flaml_leader.json").write_text(json.dumps(leader, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    rows.append({"model_id": "flaml_automl", "model_family": "FLAML AutoML", "status": "ok", "notes": f"AutoML first-pass; time_limit_per_target={time_limit}; leaders={leader}", "train_seconds": train_seconds, "predict_seconds": predict_seconds})
    return rows


def run_h2o(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str], pred_dir: Path, random_state: int, time_limit: int) -> list[dict]:
    if (pred_dir / "h2o_automl_valid_raw.csv").exists():
        return [{"model_id": "h2o_automl", "model_family": "H2O AutoML", "status": "ok", "notes": "loaded cached prediction", "train_seconds": np.nan, "predict_seconds": np.nan}]
    if not module_available("h2o"):
        return [{"model_id": "h2o_automl", "model_family": "H2O AutoML", "status": "skipped", "notes": "h2o unavailable"}]
    import h2o
    from h2o.automl import H2OAutoML

    rows = []
    preds = pd.DataFrame(index=valid.index)
    train_seconds = 0.0
    predict_seconds = 0.0
    leaders = {}
    try:
        h2o.init(max_mem_size="4G", nthreads=-1, log_level="ERRR")
        x_valid_pd = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid_h2o = h2o.H2OFrame(x_valid_pd)
        for target in TARGETS:
            y = train[target]
            mask = y.notna()
            train_pd = train.loc[mask, feature_cols + [target]].replace([np.inf, -np.inf], np.nan)
            train_h2o = h2o.H2OFrame(train_pd)
            aml = H2OAutoML(max_runtime_secs=time_limit, seed=random_state, sort_metric="MAE", verbosity="warn")
            t0 = time.perf_counter()
            aml.train(x=feature_cols, y=target, training_frame=train_h2o)
            train_seconds += time.perf_counter() - t0
            leaders[target] = aml.leader.model_id if aml.leader is not None else None
            t0 = time.perf_counter()
            pred = aml.leader.predict(valid_h2o).as_data_frame()["predict"]
            predict_seconds += time.perf_counter() - t0
            preds[target] = pred.to_numpy()
        clip_pred(preds).to_csv(pred_dir / "h2o_automl_valid_raw.csv", index=False, encoding="utf-8-sig")
        (pred_dir.parent / "artifacts" / "h2o_leaders.json").write_text(json.dumps(leaders, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append({"model_id": "h2o_automl", "model_family": "H2O AutoML", "status": "ok", "notes": f"H2O AutoML first-pass; time_limit_per_target={time_limit}; leaders={leaders}", "train_seconds": train_seconds, "predict_seconds": predict_seconds})
    finally:
        try:
            h2o.shutdown(prompt=False)
        except Exception:
            pass
    return rows


def run_tabpfn(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str], pred_dir: Path, random_state: int, sample_rows: int) -> list[dict]:
    if (pred_dir / "tabpfn_regressor_valid_raw.csv").exists():
        return [{"model_id": "tabpfn_regressor", "model_family": "TabPFN", "status": "ok", "notes": "loaded cached prediction", "train_seconds": np.nan, "predict_seconds": np.nan}]
    if not module_available("tabpfn"):
        return [{"model_id": "tabpfn_regressor", "model_family": "TabPFN", "status": "skipped", "notes": "tabpfn unavailable"}]
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tabpfn import TabPFNRegressor

    return run_one_custom_sklearn(
        "tabpfn_regressor",
        "TabPFN",
        "TabPFNRegressor first-pass on sampled rows",
        lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), TabPFNRegressor(random_state=random_state, ignore_pretraining_limits=True)),
        train,
        valid,
        feature_cols,
        pred_dir,
        random_state,
        min(sample_rows, 10000),
    )


class TabMTorchRegressor:
    def __init__(self, random_state: int = 42, epochs: int = 35, lr: float = 1e-3, batch_size: int = 512):
        self.random_state = random_state
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.imputer = None
        self.scaler = None
        self.model = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "TabMTorchRegressor":
        import torch
        import tabm
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        x_np = self.scaler.fit_transform(self.imputer.fit_transform(x)).astype("float32")
        y_np = y.to_numpy(dtype="float32")
        self.y_mean = float(np.mean(y_np))
        self.y_std = float(np.std(y_np) if np.std(y_np) > 1e-6 else 1.0)
        y_scaled = ((y_np - self.y_mean) / self.y_std).astype("float32")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tabm.TabM(
            n_num_features=x_np.shape[1],
            d_out=1,
            n_blocks=2,
            d_block=96,
            dropout=0.10,
            k=8,
            arch_type="tabm-mini",
            start_scaling_init="random-signs",
        ).to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = torch.nn.L1Loss()
        x_t = torch.from_numpy(x_np).to(device)
        y_t = torch.from_numpy(y_scaled).to(device)
        n = len(x_np)
        for _ in range(self.epochs):
            order = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                out = self.model(x_t[idx], None).mean(1).squeeze(-1)
                loss = loss_fn(out, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        import torch

        assert self.model is not None and self.imputer is not None and self.scaler is not None
        device = next(self.model.parameters()).device
        x_np = self.scaler.transform(self.imputer.transform(x)).astype("float32")
        outs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(x_np), self.batch_size):
                x_t = torch.from_numpy(x_np[start : start + self.batch_size]).to(device)
                out = self.model(x_t, None).mean(1).squeeze(-1).cpu().numpy()
                outs.append(out)
        return np.concatenate(outs) * self.y_std + self.y_mean


def run_tabm(train: pd.DataFrame, valid: pd.DataFrame, feature_cols: list[str], pred_dir: Path, random_state: int, sample_rows: int, epochs: int) -> list[dict]:
    if (pred_dir / "tabm_torch_valid_raw.csv").exists():
        return [{"model_id": "tabm_torch", "model_family": "TabM", "status": "ok", "notes": "loaded cached prediction", "train_seconds": np.nan, "predict_seconds": np.nan}]
    if not module_available("tabm"):
        return [{"model_id": "tabm_torch", "model_family": "TabM", "status": "skipped", "notes": "tabm unavailable"}]
    return run_one_custom_sklearn(
        "tabm_torch",
        "TabM",
        f"TabM torch wrapper first-pass; epochs={epochs}; sampled rows",
        lambda: TabMTorchRegressor(random_state=random_state, epochs=epochs),
        train,
        valid,
        feature_cols,
        pred_dir,
        random_state,
        min(sample_rows, 9000),
    )


def run_one_custom_sklearn(
    model_id: str,
    family: str,
    notes: str,
    factory: Callable[[], object],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: list[str],
    pred_dir: Path,
    random_state: int,
    sample_rows: int,
) -> list[dict]:
    pred, train_seconds, predict_seconds = train_sklearn_model(model_id, factory, train, valid, feature_cols, random_state, sample_rows=sample_rows)
    pred.to_csv(pred_dir / f"{model_id}_valid_raw.csv", index=False, encoding="utf-8-sig")
    return [{"model_id": model_id, "model_family": family, "status": "ok", "notes": notes, "train_seconds": train_seconds, "predict_seconds": predict_seconds}]


def run_main_phase(args: argparse.Namespace) -> None:
    pred_dir, results_dir, _, _ = ensure_dirs(args.out_dir)
    _, train, valid, feature_cols = load_w4_data(args.data_dir)
    metadata = []
    meta_path = results_dir / "metadata_main.csv"
    for spec in main_model_specs(args.random_state, args.sample_rows):
        model_id = spec["model_id"]
        pred_path = pred_dir / f"{model_id}_valid_raw.csv"
        if pred_path.exists():
            metadata.append({"model_id": model_id, "model_family": spec["family"], "status": "ok", "notes": f"{spec['notes']}; loaded cached prediction", "train_seconds": np.nan, "predict_seconds": np.nan})
            continue
        try:
            pred, train_seconds, predict_seconds = train_sklearn_model(model_id, spec["factory"], train, valid, feature_cols, args.random_state, sample_rows=int(spec.get("sample_rows", 0)))
            pred.to_csv(pred_path, index=False, encoding="utf-8-sig")
            metadata.append({"model_id": model_id, "model_family": spec["family"], "status": "ok", "notes": spec["notes"], "train_seconds": train_seconds, "predict_seconds": predict_seconds})
        except Exception as exc:
            metadata.append({"model_id": model_id, "model_family": spec["family"], "status": "failed", "notes": f"{spec['notes']}; failed: {type(exc).__name__}: {str(exc)[:500]}", "train_seconds": np.nan, "predict_seconds": np.nan})
        pd.DataFrame(metadata).to_csv(meta_path, index=False, encoding="utf-8-sig")
    optional_runners = [
        ("flaml_automl", "FLAML AutoML", lambda: run_flaml(train, valid, feature_cols, pred_dir, args.random_state, args.time_limit_per_target)),
        ("h2o_automl", "H2O AutoML", lambda: run_h2o(train, valid, feature_cols, pred_dir, args.random_state, args.time_limit_per_target)),
        ("tabpfn_regressor", "TabPFN", lambda: run_tabpfn(train, valid, feature_cols, pred_dir, args.random_state, args.sample_rows)),
        ("tabm_torch", "TabM", lambda: run_tabm(train, valid, feature_cols, pred_dir, args.random_state, args.sample_rows, args.neural_epochs)),
    ]
    for fail_id, family, runner in optional_runners:
        try:
            metadata.extend(runner())
        except Exception as exc:
            metadata.append({"model_id": fail_id, "model_family": family, "status": "failed", "notes": f"failed: {type(exc).__name__}: {str(exc)[:500]}", "train_seconds": np.nan, "predict_seconds": np.nan})
        pd.DataFrame(metadata).to_csv(meta_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(metadata).to_csv(meta_path, index=False, encoding="utf-8-sig")


def run_autogluon_phase(args: argparse.Namespace) -> None:
    pred_dir, results_dir, _, artifacts_dir = ensure_dirs(args.out_dir)
    _, train, valid, feature_cols = load_w4_data(args.data_dir)
    from autogluon.tabular import TabularPredictor

    preds = pd.DataFrame(index=valid.index)
    leaders = {}
    train_seconds = 0.0
    predict_seconds = 0.0
    for target in TARGETS:
        model_path = artifacts_dir / "autogluon" / target
        data = train.loc[train[target].notna(), feature_cols + [target]].replace([np.inf, -np.inf], np.nan)
        predictor = TabularPredictor(label=target, problem_type="regression", eval_metric="mean_absolute_error", path=str(model_path), verbosity=0)
        t0 = time.perf_counter()
        predictor.fit(data, presets="medium_quality", time_limit=args.time_limit_per_target, ag_args_fit={"num_gpus": 0})
        train_seconds += time.perf_counter() - t0
        leaders[target] = predictor.model_best
        t0 = time.perf_counter()
        preds[target] = predictor.predict(valid[feature_cols].replace([np.inf, -np.inf], np.nan)).to_numpy()
        predict_seconds += time.perf_counter() - t0
        try:
            predictor.leaderboard(silent=True).to_csv(artifacts_dir / f"autogluon_leaderboard_{target}.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
    clip_pred(preds).to_csv(pred_dir / "autogluon_tabular_valid_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"model_id": "autogluon_tabular", "model_family": "AutoGluon Tabular", "status": "ok", "notes": f"AutoGluon first-pass; time_limit_per_target={args.time_limit_per_target}; leaders={leaders}", "train_seconds": train_seconds, "predict_seconds": predict_seconds}]).to_csv(results_dir / "metadata_autogluon.csv", index=False, encoding="utf-8-sig")


def run_pycaret_phase(args: argparse.Namespace) -> None:
    pred_dir, results_dir, _, artifacts_dir = ensure_dirs(args.out_dir)
    _, train, valid, feature_cols = load_w4_data(args.data_dir)
    from pycaret.regression import compare_models, predict_model, setup

    preds = pd.DataFrame(index=valid.index)
    leaders = {}
    train_seconds = 0.0
    predict_seconds = 0.0
    include = ["lr", "lasso", "ridge", "en", "knn", "dt", "rf", "et", "lightgbm", "gbr", "xgboost", "catboost"]
    for target in TARGETS:
        data = train.loc[train[target].notna(), feature_cols + [target]].replace([np.inf, -np.inf], np.nan)
        t0 = time.perf_counter()
        setup(data=data, target=target, session_id=args.random_state, fold=3, html=False, verbose=False, n_jobs=-1)
        best = compare_models(include=include, sort="MAE", turbo=True, budget_time=max(1, args.time_limit_per_target / 60.0), verbose=False)
        train_seconds += time.perf_counter() - t0
        leaders[target] = str(best)
        t0 = time.perf_counter()
        pred_df = predict_model(best, data=valid[feature_cols].replace([np.inf, -np.inf], np.nan), verbose=False)
        predict_seconds += time.perf_counter() - t0
        label_col = "prediction_label" if "prediction_label" in pred_df.columns else pred_df.columns[-1]
        preds[target] = pred_df[label_col].to_numpy()
    clip_pred(preds).to_csv(pred_dir / "pycaret_compare_valid_raw.csv", index=False, encoding="utf-8-sig")
    (artifacts_dir / "pycaret_leaders.json").write_text(json.dumps(leaders, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{"model_id": "pycaret_compare", "model_family": "PyCaret", "status": "ok", "notes": f"PyCaret compare_models first-pass; budget_time_minutes={args.time_limit_per_target / 60.0}; leaders={leaders}", "train_seconds": train_seconds, "predict_seconds": predict_seconds}]).to_csv(results_dir / "metadata_pycaret.csv", index=False, encoding="utf-8-sig")


def run_misc_phase(args: argparse.Namespace) -> None:
    pred_dir, results_dir, _, _ = ensure_dirs(args.out_dir)
    _, train, valid, feature_cols = load_w4_data(args.data_dir)
    rows = []
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from sklearn.impute import SimpleImputer

        for model_id, n_estimators, learning_rate, sample_cap in [
            ("ngboost_normal", 180, 0.04, 7000),
            ("ngboost_normal_260_lr03", 260, 0.03, 9000),
        ]:
            if (pred_dir / f"{model_id}_valid_raw.csv").exists():
                rows.append({"model_id": model_id, "model_family": "NGBoost", "status": "ok", "notes": "loaded cached prediction", "train_seconds": np.nan, "predict_seconds": np.nan})
                continue
            preds = pd.DataFrame(index=valid.index)
            x_valid_raw = valid[feature_cols].replace([np.inf, -np.inf], np.nan)
            train_seconds = 0.0
            predict_seconds = 0.0
            for target_idx, target in enumerate(TARGETS):
                y = train[target]
                mask = y.notna()
                x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
                y_train = y.loc[mask]
                x_fit, y_fit = sample_xy(x_train, y_train, min(args.sample_rows, sample_cap), args.random_state + target_idx)
                imputer = SimpleImputer(strategy="median")
                x_fit_np = imputer.fit_transform(x_fit)
                x_valid_np = imputer.transform(x_valid_raw)
                estimator = NGBRegressor(Dist=Normal, n_estimators=n_estimators, learning_rate=learning_rate, random_state=args.random_state, verbose=False)
                t0 = time.perf_counter()
                estimator.fit(x_fit_np, y_fit)
                train_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                preds[target] = estimator.predict(x_valid_np)
                predict_seconds += time.perf_counter() - t0
            clip_pred(preds).to_csv(pred_dir / f"{model_id}_valid_raw.csv", index=False, encoding="utf-8-sig")
            rows.append({"model_id": model_id, "model_family": "NGBoost", "status": "ok", "notes": f"NGBoost Normal ensemble-oriented variant n_estimators={n_estimators}, lr={learning_rate}", "train_seconds": train_seconds, "predict_seconds": predict_seconds})
    except Exception as exc:
        rows.append({"model_id": "ngboost_normal", "model_family": "NGBoost", "status": "failed", "notes": f"failed: {type(exc).__name__}: {str(exc)[:500]}", "train_seconds": np.nan, "predict_seconds": np.nan})
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        rows.extend(run_one_custom_sklearn("explainable_boosting", "ExplainableBoostingMachine", "EBM first-pass sampled", lambda: make_pipeline(SimpleImputer(strategy="median"), ExplainableBoostingRegressor(random_state=args.random_state, max_bins=128, max_rounds=1200, interactions=10, n_jobs=-1)), train, valid, feature_cols, pred_dir, args.random_state, min(args.sample_rows, 7000)))
        rows.extend(run_one_custom_sklearn("ebm_bins192_inter12", "ExplainableBoostingMachine", "EBM diversity variant bins=192 interactions=12 single-process", lambda: make_pipeline(SimpleImputer(strategy="median"), ExplainableBoostingRegressor(random_state=args.random_state + 21, max_bins=192, max_rounds=900, interactions=12, learning_rate=0.03, outer_bags=4, n_jobs=1)), train, valid, feature_cols, pred_dir, args.random_state, min(args.sample_rows, 4500)))
        rows.extend(run_one_custom_sklearn("ebm_low_interaction", "ExplainableBoostingMachine", "EBM conservative low-interaction variant single-process", lambda: make_pipeline(SimpleImputer(strategy="median"), ExplainableBoostingRegressor(random_state=args.random_state + 22, max_bins=96, max_rounds=900, interactions=5, learning_rate=0.025, outer_bags=4, n_jobs=1)), train, valid, feature_cols, pred_dir, args.random_state, min(args.sample_rows, 4500)))
    except Exception as exc:
        rows.append({"model_id": "explainable_boosting", "model_family": "ExplainableBoostingMachine", "status": "failed", "notes": f"failed: {type(exc).__name__}: {str(exc)[:500]}", "train_seconds": np.nan, "predict_seconds": np.nan})
    pd.DataFrame(rows).to_csv(results_dir / "metadata_misc.csv", index=False, encoding="utf-8-sig")


def read_metadata(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("metadata_*.csv")):
        try:
            rows.append(pd.read_csv(path, encoding="utf-8-sig"))
        except Exception:
            pass
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["model_id", "model_family", "status", "notes", "train_seconds", "predict_seconds"])


def residual_stats(model_id: str, y_true: pd.DataFrame, pred: pd.DataFrame, baseline_pred: pd.DataFrame, reference_id: str) -> list[dict]:
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
                "reference_id": reference_id,
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


def evaluate_ensemble(ens_id: str, member_ids: list[str], weights: list[float], pred_bank: dict[str, pd.DataFrame], y_true: pd.DataFrame, timestamps: pd.Series) -> dict:
    pred = pd.DataFrame(index=next(iter(pred_bank.values())).index)
    for target in TARGETS:
        pred[target] = 0.0
        for model_id, weight in zip(member_ids, weights):
            pred[target] += pred_bank[model_id][target] * weight
    best_alpha, best = best_alpha_metrics(y_true, pred, timestamps)
    return {
        "ensemble_id": ens_id,
        "model_id": ens_id,
        "members": "+".join(member_ids),
        "weights": json.dumps(dict(zip(member_ids, weights)), ensure_ascii=False),
        "best_alpha": best_alpha,
        "alpha_score": best["score"],
        "alpha_1_nmae": best["one_minus_nmae"],
        "alpha_ficr": best["ficr"],
        "avg_nmae": best["avg_nmae"],
        "delta_vs_baseline": best["score"] - BASELINE_LOCAL_SCORE,
        "worst_month_score": np.nan,
        "high_generation_score": np.nan,
    }


def ensemble_prediction(member_weights: dict[str, float], pred_bank: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pred = pd.DataFrame(index=next(iter(pred_bank.values())).index)
    for target in TARGETS:
        pred[target] = 0.0
        for model_id, weight in member_weights.items():
            pred[target] += float(weight) * pred_bank[model_id][target]
    return clip_pred(pred)


def expanded_anchor_plus_weights(candidate_id: str, candidate_weight: float) -> dict[str, float]:
    anchor_weight = 1.0 - float(candidate_weight)
    weights = {model_id: anchor_weight * weight for model_id, weight in CURRENT_ANCHOR_WEIGHTS.items()}
    weights[candidate_id] = weights.get(candidate_id, 0.0) + float(candidate_weight)
    return {model_id: float(weight) for model_id, weight in weights.items() if abs(float(weight)) > 1e-12}


def evaluate_weighted_ensemble(ens_id: str, weights: dict[str, float], pred_bank: dict[str, pd.DataFrame], y_true: pd.DataFrame, timestamps: pd.Series) -> dict:
    member_ids = list(weights)
    member_weights = [float(weights[m]) for m in member_ids]
    return evaluate_ensemble(ens_id, member_ids, member_weights, pred_bank, y_true, timestamps)


def run_evaluate_phase(args: argparse.Namespace) -> None:
    pred_dir, results_dir, submissions_dir, _ = ensure_dirs(args.out_dir)
    _, _, valid, _ = load_w4_data(args.data_dir)
    y_true = valid[TARGETS]
    timestamps = valid["forecast_kst_dtm"]
    metadata = read_metadata(results_dir)
    meta_by_id = {row["model_id"]: row.to_dict() for _, row in metadata.iterrows()}

    pred_bank: dict[str, pd.DataFrame] = {}
    model_rows = []
    group_rows = []
    monthly_rows = []
    residual_rows = []

    for path in sorted(pred_dir.glob("*_valid_raw.csv")):
        model_id = path.name.removesuffix("_valid_raw.csv")
        pred = clip_pred(pd.read_csv(path, encoding="utf-8-sig"))
        if not all(target in pred.columns for target in TARGETS):
            continue
        pred_bank[model_id] = pred[TARGETS]
        raw, _, _ = detailed_metrics(model_id, y_true, pred, timestamps, 1.0)
        best_alpha, alpha_metrics = best_alpha_metrics(y_true, pred, timestamps)
        alpha_detail, alpha_group, alpha_monthly = detailed_metrics(f"{model_id}__alpha", y_true, pred, timestamps, best_alpha)
        apply_alpha(pred, best_alpha).to_csv(pred_dir / f"{model_id}_valid_alpha.csv", index=False, encoding="utf-8-sig")
        meta = meta_by_id.get(model_id, {})
        row = {
            "model_id": model_id,
            "model_family": meta.get("model_family", "unknown"),
            "raw_score": raw["score"],
            "raw_1_nmae": raw["one_minus_nmae"],
            "raw_ficr": raw["ficr"],
            "best_alpha": best_alpha,
            "alpha_score": alpha_metrics["score"],
            "alpha_1_nmae": alpha_metrics["one_minus_nmae"],
            "alpha_ficr": alpha_metrics["ficr"],
            "avg_nmae": alpha_metrics["avg_nmae"],
            "delta_vs_baseline": alpha_metrics["score"] - BASELINE_LOCAL_SCORE,
            "worst_month_score": alpha_detail["worst_month_score"],
            "high_generation_score": alpha_detail["high_generation_score"],
            "group_1_score": group_score(alpha_metrics["group_rows"][0]),
            "group_2_score": group_score(alpha_metrics["group_rows"][1]),
            "group_3_score": group_score(alpha_metrics["group_rows"][2]),
            "train_seconds": meta.get("train_seconds", np.nan),
            "predict_seconds": meta.get("predict_seconds", np.nan),
            "status": meta.get("status", "ok"),
            "notes": meta.get("notes", ""),
        }
        model_rows.append(row)
        group_rows.extend(alpha_group)
        monthly_rows.extend(alpha_monthly)

    known_model_ids = set(meta_by_id)
    evaluated_ids = set(pred_bank)
    for missing in sorted(known_model_ids - evaluated_ids):
        meta = meta_by_id[missing]
        if meta.get("status") != "ok":
            model_rows.append(
                {
                    "model_id": missing,
                    "model_family": meta.get("model_family", "unknown"),
                    "status": meta.get("status", "failed"),
                    "notes": meta.get("notes", ""),
                    "raw_score": np.nan,
                    "alpha_score": np.nan,
                }
            )

    if "lgbm_l1_baseline" not in pred_bank:
        raise RuntimeError("lgbm_l1_baseline prediction is required for residual and ensemble analysis")
    baseline_pred = pred_bank["lgbm_l1_baseline"]
    missing_anchor_members = [model_id for model_id in CURRENT_ANCHOR_WEIGHTS if model_id not in pred_bank]
    if missing_anchor_members:
        raise RuntimeError(f"current anchor members missing: {missing_anchor_members}")
    anchor_pred = ensemble_prediction(CURRENT_ANCHOR_WEIGHTS, pred_bank)
    pred_bank["current_best_anchor"] = anchor_pred
    anchor_alpha, anchor_metrics = best_alpha_metrics(y_true, anchor_pred, timestamps)
    anchor_detail, anchor_group, anchor_monthly = detailed_metrics("current_best_anchor__alpha", y_true, anchor_pred, timestamps, anchor_alpha)
    group_rows.extend(anchor_group)
    monthly_rows.extend(anchor_monthly)
    for model_id, pred in pred_bank.items():
        residual_rows.extend(residual_stats(model_id, y_true, pred, baseline_pred, "lgbm_l1_baseline"))
        residual_rows.extend(residual_stats(model_id, y_true, pred, anchor_pred, "current_best_anchor"))

    model_summary = pd.DataFrame(model_rows)
    group_df = pd.DataFrame(group_rows)
    monthly_df = pd.DataFrame(monthly_rows)
    residual_df = pd.DataFrame(residual_rows)
    for idx, row in model_summary[model_summary["status"] == "ok"].iterrows():
        months_better, groups_better = better_slices(f"{row['model_id']}__alpha", monthly_df, group_df, "lgbm_l1_baseline__alpha")
        model_summary.loc[idx, "months_better_than_lgbm"] = months_better
        model_summary.loc[idx, "groups_better_than_lgbm"] = groups_better
        months_better_anchor, groups_better_anchor = better_slices(f"{row['model_id']}__alpha", monthly_df, group_df, "current_best_anchor__alpha")
        model_summary.loc[idx, "months_better_than_current_anchor"] = months_better_anchor
        model_summary.loc[idx, "groups_better_than_current_anchor"] = groups_better_anchor
        model_summary.loc[idx, "delta_vs_current_anchor"] = float(row["alpha_score"]) - CURRENT_ANCHOR_SCORE

    ensemble_rows = []
    ensemble_rows.append(evaluate_weighted_ensemble("current_best_anchor_recomputed", CURRENT_ANCHOR_WEIGHTS, pred_bank, y_true, timestamps))
    ok_models = model_summary[model_summary["status"] == "ok"].sort_values("alpha_score", ascending=False)["model_id"].tolist()
    selected = ok_models[: args.max_ensemble_models]
    candidate_models = [model_id for model_id in ok_models if model_id not in CURRENT_ANCHOR_WEIGHTS]
    for other in candidate_models[: max(args.max_ensemble_models, 18)]:
        for w in [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
            weights = expanded_anchor_plus_weights(other, w)
            ensemble_rows.append(evaluate_weighted_ensemble(f"anchor_plus_{other}_{w:.2f}", weights, pred_bank, y_true, timestamps))
    for a, b in itertools.combinations(candidate_models[:10], 2):
        for wa, wb in [(0.05, 0.05), (0.07, 0.03), (0.03, 0.07), (0.10, 0.05)]:
            weights = {model_id: (1.0 - wa - wb) * weight for model_id, weight in CURRENT_ANCHOR_WEIGHTS.items()}
            weights[a] = weights.get(a, 0.0) + wa
            weights[b] = weights.get(b, 0.0) + wb
            ensemble_rows.append(evaluate_weighted_ensemble(f"anchor_plus2_{a}_{b}_{wa:.2f}_{wb:.2f}", weights, pred_bank, y_true, timestamps))
    catboost_variants = [m for m in ok_models if m.startswith("catboost_quantile_")]
    lgbm_variants = [m for m in ok_models if m.startswith("lgbm_l1")]
    tree_variants = [m for m in ok_models if m.startswith("extra_trees") or m.startswith("rf_") or m.startswith("random_forest")]
    for cb in catboost_variants[:6]:
        for lg in lgbm_variants[:4]:
            for tr in tree_variants[:4]:
                for weights in [(0.60, 0.20, 0.20), (0.65, 0.20, 0.15), (0.55, 0.25, 0.20), (0.70, 0.15, 0.15)]:
                    ensemble_rows.append(evaluate_ensemble(f"family3_{cb}_{lg}_{tr}_{'_'.join(f'{w:.2f}' for w in weights)}", [cb, lg, tr], list(weights), pred_bank, y_true, timestamps))
    if "lgbm_l1_baseline" in selected:
        for other in selected:
            if other == "lgbm_l1_baseline":
                continue
            for w in np.round(np.arange(0.50, 1.0001, 0.05), 2):
                ensemble_rows.append(evaluate_ensemble(f"w2_lgbm_{other}_{w:.2f}", ["lgbm_l1_baseline", other], [float(w), float(1 - w)], pred_bank, y_true, timestamps))
    for combo in itertools.combinations(selected[:8], 3):
        if "lgbm_l1_baseline" not in combo:
            continue
        for weights in [(0.7, 0.2, 0.1), (0.7, 0.1, 0.2), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2), (1 / 3, 1 / 3, 1 / 3)]:
            ensemble_rows.append(evaluate_ensemble(f"w3_{'_'.join(combo)}_{'_'.join(f'{w:.2f}' for w in weights)}", list(combo), list(weights), pred_bank, y_true, timestamps))
    # Residual-diversity blends: force in the least-correlated feasible models with a small weight.
    if not residual_df.empty:
        diversity = (
            residual_df[(residual_df["reference_id"] == "current_best_anchor") & (residual_df["model_id"] != "current_best_anchor")]
            .groupby("model_id", as_index=False)["pearson"]
            .mean()
            .sort_values("pearson")
            .head(6)["model_id"]
            .tolist()
        )
        for other in diversity:
            if other in pred_bank:
                ensemble_rows.append(evaluate_weighted_ensemble(f"diverse_anchor_{other}_0.90", expanded_anchor_plus_weights(other, 0.10), pred_bank, y_true, timestamps))

    ensemble_df = pd.DataFrame(ensemble_rows)
    if not ensemble_df.empty:
        ensemble_df["delta_vs_current_anchor"] = ensemble_df["alpha_score"] - CURRENT_ANCHOR_SCORE
        ensemble_df["candidate_class"] = np.select(
            [
                ensemble_df["delta_vs_current_anchor"] >= 0.0010,
                ensemble_df["delta_vs_current_anchor"] >= 0.0007,
                ensemble_df["delta_vs_current_anchor"] >= 0.0003,
                (ensemble_df["delta_vs_current_anchor"] >= 0.0) & (ensemble_df["alpha_ficr"] >= CURRENT_ANCHOR_FICR),
            ],
            ["core", "strong", "keep", "stability"],
            default="reject",
        )
        ensemble_group_rows = []
        ensemble_monthly_rows = []
        for idx in ensemble_df.sort_values("alpha_score", ascending=False).head(60).index:
            weights = json.loads(ensemble_df.loc[idx, "weights"])
            pred = ensemble_prediction(weights, pred_bank)
            detail, ens_group, ens_monthly = detailed_metrics(str(ensemble_df.loc[idx, "ensemble_id"]), y_true, pred, timestamps, float(ensemble_df.loc[idx, "best_alpha"]))
            ensemble_df.loc[idx, "worst_month_score"] = detail["worst_month_score"]
            ensemble_df.loc[idx, "high_generation_score"] = detail["high_generation_score"]
            ensemble_group_rows.extend(ens_group)
            ensemble_monthly_rows.extend(ens_monthly)
        if ensemble_group_rows:
            group_df = pd.concat([group_df, pd.DataFrame(ensemble_group_rows)], ignore_index=True)
        if ensemble_monthly_rows:
            monthly_df = pd.concat([monthly_df, pd.DataFrame(ensemble_monthly_rows)], ignore_index=True)

    model_summary.to_csv(results_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(results_dir / "group_scores.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(results_dir / "monthly_scores.csv", index=False, encoding="utf-8-sig")
    residual_df.to_csv(results_dir / "residual_correlation.csv", index=False, encoding="utf-8-sig")
    ensemble_df.to_csv(results_dir / "ensemble_summary.csv", index=False, encoding="utf-8-sig")
    ensemble_df.sort_values("delta_vs_current_anchor", ascending=False).to_csv(results_dir / "ensemble_contribution.csv", index=False, encoding="utf-8-sig")

    ok = model_summary[model_summary["status"] == "ok"].copy()
    best_model = ok.sort_values("alpha_score", ascending=False).iloc[0].to_dict()
    best_ensemble = None if ensemble_df.empty else ensemble_df.sort_values("alpha_score", ascending=False).iloc[0].to_dict()
    candidate_kind = "model"
    candidate = best_model
    if best_ensemble is not None and best_ensemble["alpha_score"] > best_model["alpha_score"]:
        candidate_kind = "ensemble"
        candidate = best_ensemble

    baseline_row = ok[ok["model_id"] == "lgbm_l1_baseline"].iloc[0].to_dict()
    accepted = bool(
        candidate["alpha_score"] >= NEXT_ACCEPT_SCORE
        and candidate.get("alpha_ficr", 0.0) >= CURRENT_ANCHOR_FICR - 0.0005
        and (math.isnan(float(candidate.get("worst_month_score", np.nan))) or candidate.get("worst_month_score", 0.0) >= CURRENT_ANCHOR_WORST_MONTH - 0.002)
        and candidate.get("alpha_score", 0.0) > CURRENT_ANCHOR_SCORE
    )
    submission_info = None
    if accepted:
        submission_info = create_submission_for_candidate(args, candidate_kind, candidate, submissions_dir)

    package_status = {
        "main": {m: module_available(m) for m in ["lightgbm", "xgboost", "catboost", "sklearn", "optuna", "flaml", "h2o", "tabpfn", "tabm", "torch"]},
        "external_metadata_files": sorted(path.name for path in results_dir.glob("metadata_*.csv")),
    }
    best_params = {
        "baseline": {"local_score": BASELINE_LOCAL_SCORE, "public_lb": BASELINE_PUBLIC_LB, "alpha": BASELINE_ALPHA, "accept_score": ACCEPT_SCORE},
        "current_anchor": {"score": CURRENT_ANCHOR_SCORE, "ficr": CURRENT_ANCHOR_FICR, "worst_month_score": CURRENT_ANCHOR_WORST_MONTH, "weights": CURRENT_ANCHOR_WEIGHTS, "next_accept_score": NEXT_ACCEPT_SCORE, "strong_accept_score": STRONG_ACCEPT_SCORE, "high_confidence_score": HIGH_CONFIDENCE_SCORE, "recomputed_alpha": anchor_alpha, "recomputed_score": anchor_metrics["score"]},
        "best_model": best_model,
        "best_ensemble": best_ensemble,
        "selected_candidate_kind": candidate_kind,
        "selected_candidate": candidate,
        "accepted": accepted,
        "submission_info": submission_info,
        "package_status": package_status,
    }
    (results_dir / "best_params.json").write_text(json.dumps(best_params, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_conclusion(results_dir, model_summary, residual_df, ensemble_df, best_params)


def train_final_prediction_for_model(model_id: str, args: argparse.Namespace) -> pd.DataFrame:
    labels = read_labels(args.data_dir)
    train_features, test_features = build_train_test_features(args.data_dir)
    data = train_features.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_label"))
    feature_cols = [c for c in train_features.columns if c != "forecast_kst_dtm"]
    spec_map = {spec["model_id"]: spec for spec in main_model_specs(args.random_state, args.sample_rows)}
    if model_id not in spec_map:
        raise ValueError(f"final submission reproduction is not implemented for external model: {model_id}")
    spec = spec_map[model_id]
    preds = pd.DataFrame({"forecast_kst_dtm": test_features["forecast_kst_dtm"]})
    x_test = test_features[feature_cols].replace([np.inf, -np.inf], np.nan)
    for target in TARGETS:
        years = [2022, 2023, 2024] if target != "kpx_group_3" else [2023, 2024]
        train = data[data["year"].isin(years)].copy()
        mask = train[target].notna()
        x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = train.loc[mask, target]
        sample_rows = int(spec.get("sample_rows", 0) or 0)
        x_fit, y_fit = sample_xy(x_train, y_train, sample_rows, args.random_state + TARGETS.index(target))
        estimator = spec["factory"]()
        estimator.fit(x_fit, y_fit)
        preds[target] = estimator.predict(x_test)
    return clip_pred(preds)


def write_submission(data_dir: Path, final_pred: pd.DataFrame, alpha: float, output: Path) -> dict:
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    adjusted = pd.concat([final_pred[["forecast_kst_dtm"]], apply_alpha(final_pred[TARGETS], alpha)], axis=1)
    submission = sample[["forecast_id", "forecast_kst_dtm"]].merge(adjusted, on="forecast_kst_dtm", how="left")
    submission.to_csv(output, index=False, encoding="utf-8-sig")
    return {
        "created": True,
        "path": str(output),
        "rows": int(len(submission)),
        "missing_predictions": int(submission[TARGETS].isna().sum().sum()),
        "min_prediction": float(submission[TARGETS].min().min()),
        "max_prediction": float(submission[TARGETS].max().max()),
        "columns_match": list(submission.columns) == list(sample.columns),
        "ids_match": bool(submission["forecast_id"].equals(sample["forecast_id"])),
        "dt_match": bool(submission["forecast_kst_dtm"].equals(sample["forecast_kst_dtm"])),
    }


def create_submission_for_candidate(args: argparse.Namespace, candidate_kind: str, candidate: dict, submissions_dir: Path) -> dict:
    alpha = float(candidate["best_alpha"])
    try:
        if candidate_kind == "model":
            final_pred = train_final_prediction_for_model(str(candidate["model_id"]), args)
            output = submissions_dir / f"submission_{candidate['model_id']}_alpha.csv"
            return write_submission(args.data_dir, final_pred, alpha, output)
        weights = json.loads(candidate["weights"])
        final_parts = []
        for member, weight in weights.items():
            final_parts.append((float(weight), train_final_prediction_for_model(member, args)))
        final_pred = pd.DataFrame({"forecast_kst_dtm": final_parts[0][1]["forecast_kst_dtm"]})
        for target in TARGETS:
            final_pred[target] = 0.0
            for weight, part in final_parts:
                final_pred[target] += weight * part[target]
        safe_id = str(candidate["ensemble_id"])[:120].replace("+", "_")
        output = submissions_dir / f"submission_{safe_id}_alpha.csv"
        return write_submission(args.data_dir, clip_pred(final_pred), alpha, output)
    except Exception as exc:
        return {"created": False, "reason": f"accepted validation candidate, but final submission reproduction failed: {type(exc).__name__}: {str(exc)[:500]}"}


def write_conclusion(results_dir: Path, model_summary: pd.DataFrame, residual_df: pd.DataFrame, ensemble_df: pd.DataFrame, best_params: dict) -> None:
    top_models = model_summary.sort_values("alpha_score", ascending=False, na_position="last").head(20)
    top_ensembles = ensemble_df.sort_values("alpha_score", ascending=False, na_position="last").head(20) if not ensemble_df.empty else pd.DataFrame()
    ok_count = int((model_summary["status"] == "ok").sum())
    failed_count = int((model_summary["status"] != "ok").sum())
    best_model = best_params["best_model"]
    best_ensemble = best_params["best_ensemble"]
    low_corr = pd.DataFrame()
    if not residual_df.empty:
        low_corr = residual_df[(residual_df["reference_id"] == "current_best_anchor") & (residual_df["model_id"] != "current_best_anchor")].groupby("model_id", as_index=False).agg(mean_pearson=("pearson", "mean"), better_abs_error_rate=("model_better_abs_error_rate", "mean"), both_outside_8pct_rate=("both_outside_8pct_rate", "mean")).sort_values("mean_pearson").head(15)
    contribution = ensemble_df.sort_values("delta_vs_current_anchor", ascending=False).head(20) if not ensemble_df.empty and "delta_vs_current_anchor" in ensemble_df.columns else pd.DataFrame()
    improved = contribution[contribution["delta_vs_current_anchor"] > 0] if not contribution.empty else pd.DataFrame()
    lines = [
        "# Ensemble-Oriented Diversity Optimization Results",
        "",
        "## Fixed protocol",
        "",
        "- scope: GitHub issue #4 follow-up ensemble-oriented diversity optimization",
        "- local validation: train 2022-2023, valid 2024",
        "- feature baseline: B1 aggregate only",
        "- metric: official DACON score",
        f"- alpha grid: {ALPHA_GRID[0]:.4f} to {ALPHA_GRID[-1]:.4f}, step 0.0025",
        f"- original baseline score: {BASELINE_LOCAL_SCORE}",
        f"- current best ensemble anchor score: {CURRENT_ANCHOR_SCORE}",
        f"- current best ensemble anchor FiCR: {CURRENT_ANCHOR_FICR}",
        f"- current best ensemble anchor worst month: {CURRENT_ANCHOR_WORST_MONTH}",
        f"- Public LB baseline: {BASELINE_PUBLIC_LB}",
        f"- raised candidate threshold: {NEXT_ACCEPT_SCORE}",
        "",
        "## Execution coverage",
        "",
        f"- successful model predictions: {ok_count}",
        f"- failed/skipped metadata rows: {failed_count}",
        "- broad model search repetition: intentionally not performed.",
        "- full Optuna deep tuning: intentionally not performed.",
        "- feature engineering expansion: intentionally not performed.",
        "",
        "## Top single models",
        "",
        top_models[[c for c in ["model_id", "model_family", "raw_score", "best_alpha", "alpha_score", "delta_vs_baseline", "delta_vs_current_anchor", "alpha_ficr", "worst_month_score", "high_generation_score", "months_better_than_lgbm", "months_better_than_current_anchor", "groups_better_than_lgbm", "groups_better_than_current_anchor", "status", "notes"] if c in top_models.columns]].to_markdown(index=False),
        "",
        "## Top ensembles by score",
        "",
        top_ensembles[[c for c in ["ensemble_id", "members", "best_alpha", "alpha_score", "delta_vs_baseline", "delta_vs_current_anchor", "candidate_class", "alpha_ficr", "worst_month_score", "high_generation_score"] if c in top_ensembles.columns]].to_markdown(index=False) if not top_ensembles.empty else "No ensembles were evaluated.",
        "",
        "## Top ensemble contribution vs current anchor",
        "",
        contribution[[c for c in ["ensemble_id", "members", "best_alpha", "alpha_score", "delta_vs_current_anchor", "candidate_class", "alpha_ficr", "worst_month_score", "high_generation_score"] if c in contribution.columns]].to_markdown(index=False) if not contribution.empty else "No contribution rows available.",
        "",
        "## Lowest residual-correlation candidates vs current anchor",
        "",
        low_corr.to_markdown(index=False) if not low_corr.empty else "No residual correlation rows available.",
        "",
        "## Selected candidate",
        "",
        json.dumps({"kind": best_params["selected_candidate_kind"], "candidate": best_params["selected_candidate"], "accepted": best_params["accepted"], "submission": best_params["submission_info"]}, indent=2, ensure_ascii=False, default=str),
        "",
        "## Answer: did ensemble-oriented diversity optimization improve the anchor?",
        "",
        f"- improved ensembles above current anchor: {len(improved)}",
        f"- best selected score: {best_params['selected_candidate'].get('alpha_score')}",
        f"- accepted submission candidate: {best_params['accepted']}",
        "",
        "This stage evaluates whether shallow/medium tuning of anchor-family and diversity candidates creates marginal ensemble gain beyond the current best ensemble. Final interpretation should prioritize ensemble contribution, FiCR/worst-month guardrails, and residual diversity rather than standalone score alone.",
    ]
    failed = model_summary[model_summary["status"] != "ok"]
    if not failed.empty:
        lines.extend(["", "## Failed or skipped models", "", failed[["model_id", "model_family", "status", "notes"]].to_markdown(index=False)])
    (results_dir / "conclusion.md").write_text("\n".join(lines), encoding="utf-8")
    print((results_dir / "conclusion.md").read_text(encoding="utf-8"))


def run_subprocess_phases(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    commands = [
        [str(ROOT / ".venvs" / "autogluon" / "Scripts" / "python.exe"), str(script), "--phase", "autogluon"],
        [str(ROOT / ".venvs" / "pycaret311" / "Scripts" / "python.exe"), str(script), "--phase", "pycaret"],
        [str(ROOT / ".venvs" / "miscmodels" / "Scripts" / "python.exe"), str(script), "--phase", "misc"],
    ]
    for cmd in commands:
        exe = Path(cmd[0])
        if not exe.exists():
            continue
        full = cmd + [
            "--data-dir",
            str(args.data_dir),
            "--out-dir",
            str(args.out_dir),
            "--random-state",
            str(args.random_state),
            "--time-limit-per-target",
            str(args.time_limit_per_target),
            "--sample-rows",
            str(args.sample_rows),
            "--neural-epochs",
            str(args.neural_epochs),
        ]
        subprocess.run(full, check=False)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    if args.phase == "main":
        run_main_phase(args)
    elif args.phase == "autogluon":
        run_autogluon_phase(args)
    elif args.phase == "pycaret":
        run_pycaret_phase(args)
    elif args.phase == "misc":
        run_misc_phase(args)
    elif args.phase == "evaluate":
        run_evaluate_phase(args)
    elif args.phase == "all":
        run_main_phase(args)
        if not args.no_subprocess:
            run_subprocess_phases(args)
        run_evaluate_phase(args)


if __name__ == "__main__":
    main()

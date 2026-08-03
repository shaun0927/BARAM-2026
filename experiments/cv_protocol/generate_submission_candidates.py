from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from run_cv_protocol import CAPACITY, TARGETS, aggregate_weather, read_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "submissions")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_train_test_features(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ldaps = aggregate_weather(data_dir / "train" / "ldaps_train.csv", "ldaps", include_physics=False)
    train_gfs = aggregate_weather(data_dir / "train" / "gfs_train.csv", "gfs", include_physics=False)
    test_ldaps = aggregate_weather(data_dir / "test" / "ldaps_test.csv", "ldaps", include_physics=False)
    test_gfs = aggregate_weather(data_dir / "test" / "gfs_test.csv", "gfs", include_physics=False)

    train = train_ldaps.merge(train_gfs, on="forecast_kst_dtm", how="inner")
    test = test_ldaps.merge(test_gfs, on="forecast_kst_dtm", how="inner")
    for df in (train, test):
        df["year"] = df["forecast_kst_dtm"].dt.year
        df["month"] = df["forecast_kst_dtm"].dt.month
        df["hour"] = df["forecast_kst_dtm"].dt.hour
        df["dayofyear"] = df["forecast_kst_dtm"].dt.dayofyear
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366.0)
        df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366.0)
    return train, test


def train_predict_policy(
    data: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    policy: str,
    random_state: int,
) -> pd.DataFrame:
    preds = pd.DataFrame({"forecast_kst_dtm": test["forecast_kst_dtm"]})
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)

    for target in TARGETS:
        if policy == "w4_full_history":
            years = [2022, 2023, 2024] if target != "kpx_group_3" else [2023, 2024]
            weights = None
        elif policy == "w5_recency_weighted":
            years = [2022, 2023, 2024] if target != "kpx_group_3" else [2023, 2024]
            weights = {2022: 0.5, 2023: 0.75, 2024: 1.0}
        else:
            raise ValueError(f"unknown policy: {policy}")

        train = data[data["year"].isin(years)].copy()
        mask = train[target].notna()
        x_train = train.loc[mask, feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = train.loc[mask, target]
        sample_weight = None
        if weights:
            sample_weight = train.loc[mask, "year"].map(weights).fillna(1.0).to_numpy()

        model = LGBMRegressor(
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
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        pred = model.predict(x_test)
        preds[target] = np.clip(pred, 0.0, CAPACITY[target])
    return preds


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.data_dir)
    train_features, test_features = build_train_test_features(args.data_dir)
    data = train_features.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", suffixes=("", "_label"))
    feature_cols = [c for c in train_features.columns if c != "forecast_kst_dtm"]

    sample = pd.read_csv(args.data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])

    manifest = []
    for policy in ["w4_full_history", "w5_recency_weighted"]:
        pred = train_predict_policy(data, test_features, feature_cols, policy, args.random_state)
        submission = sample[["forecast_id", "forecast_kst_dtm"]].merge(pred, on="forecast_kst_dtm", how="left")
        output = args.out_dir / f"submission_{policy}.csv"
        submission.to_csv(output, index=False, encoding="utf-8-sig")
        manifest.append(
            {
                "policy": policy,
                "path": str(output),
                "rows": len(submission),
                "missing_predictions": int(submission[TARGETS].isna().sum().sum()),
                "min_prediction": float(submission[TARGETS].min().min()),
                "max_prediction": float(submission[TARGETS].max().max()),
            }
        )

    pd.DataFrame(manifest).to_csv(args.out_dir / "submission_manifest.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(manifest).to_string(index=False))


if __name__ == "__main__":
    main()

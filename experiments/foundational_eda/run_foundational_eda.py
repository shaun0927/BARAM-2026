from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "experiments" / "cv_protocol"
sys.path.insert(0, str(CV_DIR))

from generate_submission_candidates import build_train_test_features  # noqa: E402
from run_cv_protocol import CAPACITY, TARGETS, build_feature_frame, read_labels  # noqa: E402


RATIO_BINS = [-np.inf, 0.01, 0.05, 0.10, 0.30, 0.60, 0.80, 0.90, np.inf]
RATIO_LABELS = ["<1%", "1-5%", "5-10%", "10-30%", "30-60%", "60-80%", "80-90%", "90-100%"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\USER\Desktop\jh0927\open"))
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Path:
    results = out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    return results


def read_csv_header(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", nrows=5)


def count_csv_rows(path: Path) -> int:
    # subtract header; works for normal CSV files and avoids loading large weather files twice.
    with path.open("rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


def timestamp_profile(df: pd.DataFrame, time_col: str) -> dict:
    ts = pd.to_datetime(df[time_col], errors="coerce")
    valid_ts = ts.dropna().sort_values()
    if valid_ts.empty:
        return {
            "time_col": time_col,
            "min_ts": None,
            "max_ts": None,
            "unique_timestamps": 0,
            "duplicate_timestamp_rows": int(ts.duplicated().sum()),
            "missing_hour_gaps": np.nan,
            "non_1h_gap_count": np.nan,
        }
    expected = pd.date_range(valid_ts.min(), valid_ts.max(), freq="h")
    unique_ts = pd.Index(valid_ts.unique()).sort_values()
    gaps = unique_ts.to_series().diff().dropna()
    return {
        "time_col": time_col,
        "min_ts": str(valid_ts.min()),
        "max_ts": str(valid_ts.max()),
        "unique_timestamps": int(len(unique_ts)),
        "expected_hourly_timestamps": int(len(expected)),
        "missing_hour_gaps": int(len(expected.difference(unique_ts))),
        "duplicate_timestamp_rows": int(ts.duplicated().sum()),
        "non_1h_gap_count": int((gaps != pd.Timedelta(hours=1)).sum()),
    }


def dataset_inventory(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = [
        data_dir / "train" / "train_labels.csv",
        data_dir / "train" / "ldaps_train.csv",
        data_dir / "train" / "gfs_train.csv",
        data_dir / "train" / "scada_unison_train.csv",
        data_dir / "train" / "scada_vestas_train.csv",
        data_dir / "test" / "ldaps_test.csv",
        data_dir / "test" / "gfs_test.csv",
        data_dir / "sample_submission.csv",
        data_dir / "data_description.md",
        data_dir / "info.xlsx",
    ]
    inventory_rows = []
    coverage_rows = []
    for path in files:
        row = {
            "file": str(path.relative_to(data_dir)),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else np.nan,
            "rows": np.nan,
            "columns": np.nan,
            "datetime_columns": "",
            "numeric_columns": np.nan,
            "object_columns": np.nan,
        }
        if path.suffix.lower() == ".csv" and path.exists():
            header = read_csv_header(path)
            row["rows"] = count_csv_rows(path)
            row["columns"] = int(header.shape[1])
            row["numeric_columns"] = int(sum(pd.api.types.is_numeric_dtype(header[c]) for c in header.columns))
            row["object_columns"] = int(sum(pd.api.types.is_object_dtype(header[c]) for c in header.columns))
            dt_cols = [c for c in header.columns if "dtm" in c.lower() or "time" in c.lower() or "date" in c.lower()]
            row["datetime_columns"] = ",".join(dt_cols)
            # Load only timestamp columns for coverage.
            for time_col in dt_cols:
                try:
                    ts_df = pd.read_csv(path, encoding="utf-8-sig", usecols=[time_col])
                    prof = timestamp_profile(ts_df, time_col)
                    prof.update({"file": str(path.relative_to(data_dir))})
                    if "forecast_kst_dtm" in ts_df.columns and "grid_id" in header.columns:
                        grid_df = pd.read_csv(path, encoding="utf-8-sig", usecols=["forecast_kst_dtm", "grid_id"])
                        per_ts = grid_df.groupby("forecast_kst_dtm")["grid_id"].nunique()
                        prof["min_grid_count_per_timestamp"] = int(per_ts.min())
                        prof["max_grid_count_per_timestamp"] = int(per_ts.max())
                        prof["mode_grid_count_per_timestamp"] = int(per_ts.mode().iloc[0])
                    coverage_rows.append(prof)
                except Exception as exc:
                    coverage_rows.append({"file": str(path.relative_to(data_dir)), "time_col": time_col, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        elif path.suffix.lower() == ".xlsx" and path.exists():
            try:
                sheets = pd.read_excel(path, sheet_name=None)
                row["rows"] = sum(df.shape[0] for df in sheets.values())
                row["columns"] = sum(df.shape[1] for df in sheets.values())
                row["datetime_columns"] = f"sheets={','.join(sheets.keys())}"
            except Exception as exc:
                row["datetime_columns"] = f"read_error={type(exc).__name__}: {str(exc)[:120]}"
        inventory_rows.append(row)
    return pd.DataFrame(inventory_rows), pd.DataFrame(coverage_rows)


def make_long_labels(labels: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["kst_dtm", "year", "month", "hour"]
    long = labels[base_cols + TARGETS].melt(id_vars=base_cols, value_vars=TARGETS, var_name="target", value_name="actual")
    long["capacity"] = long["target"].map(CAPACITY)
    long["actual_ratio"] = long["actual"] / long["capacity"]
    long["eligible"] = long["actual"] >= 0.10 * long["capacity"]
    long["season"] = pd.cut(
        long["month"],
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "fall", "winter2"],
        include_lowest=True,
    ).astype(str).replace({"winter2": "winter"})
    long["weekday"] = pd.to_datetime(long["kst_dtm"]).dt.weekday
    long["ratio_bin"] = pd.cut(long["actual_ratio"], bins=RATIO_BINS, labels=RATIO_LABELS, include_lowest=True).astype(str)
    return long


def label_coverage(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["target", "year"], dropna=False)
        .agg(
            rows=("actual", "size"),
            non_null=("actual", "count"),
            missing=("actual", lambda s: int(s.isna().sum())),
            eligible_count=("eligible", "sum"),
            eligible_rate=("eligible", "mean"),
            zero_count=("actual", lambda s: int((s == 0).sum())),
            negative_count=("actual", lambda s: int((s < 0).sum())),
            capacity_exceed_count=("actual_ratio", lambda s: int((s > 1.0).sum())),
        )
        .reset_index()
    )


def target_distribution(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys in [["target"], ["target", "year"], ["target", "season"]]:
        g = long.groupby(keys, dropna=False)
        part = g["actual_ratio"].agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            p01=lambda s: s.quantile(0.01),
            p05=lambda s: s.quantile(0.05),
            p10=lambda s: s.quantile(0.10),
            p25=lambda s: s.quantile(0.25),
            p50=lambda s: s.quantile(0.50),
            p75=lambda s: s.quantile(0.75),
            p90=lambda s: s.quantile(0.90),
            p95=lambda s: s.quantile(0.95),
            p99=lambda s: s.quantile(0.99),
            max="max",
        ).reset_index()
        part.insert(0, "slice_type", "+".join(keys))
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def target_ratio_bins(long: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for keys in [["target", "year", "ratio_bin"], ["target", "ratio_bin"], ["year", "ratio_bin"], ["target", "month", "ratio_bin"]]:
        part = (
            long.groupby(keys, dropna=False, observed=False)
            .agg(
                count=("actual", "count"),
                row_count=("actual", "size"),
                actual_sum=("actual", "sum"),
                eligible_count=("eligible", "sum"),
                mean_actual_ratio=("actual_ratio", "mean"),
            )
            .reset_index()
        )
        part.insert(0, "slice_type", "+".join(keys))
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    # Shares inside each slice excluding the bin.
    share_keys = [c for c in ["target", "year", "month"] if c in out.columns]
    out["count_share_hint"] = out["count"]
    return out


def target_by_year_month_hour(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["target", "year", "month", "hour"], dropna=False)
        .agg(
            count=("actual", "count"),
            mean_actual=("actual", "mean"),
            mean_actual_ratio=("actual_ratio", "mean"),
            p50_actual_ratio=("actual_ratio", "median"),
            p90_actual_ratio=("actual_ratio", lambda s: s.quantile(0.90)),
            eligible_rate=("eligible", "mean"),
        )
        .reset_index()
    )


def target_group_correlation(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, df in [("all", labels)] + [(str(y), labels[labels["year"] == y]) for y in sorted(labels["year"].dropna().unique())]:
        corr = df[TARGETS].corr()
        for a in TARGETS:
            for b in TARGETS:
                rows.append({"year": year, "target_a": a, "target_b": b, "correlation": corr.loc[a, b]})
    return pd.DataFrame(rows)


def metric_eligible_distribution(long: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for keys in [["target"], ["target", "year"], ["target", "month"], ["target", "hour"], ["target", "season"], ["target", "ratio_bin"]]:
        part = (
            long.groupby(keys, dropna=False, observed=False)
            .agg(
                rows=("actual", "size"),
                non_null=("actual", "count"),
                eligible_count=("eligible", "sum"),
                eligible_rate=("eligible", "mean"),
                actual_sum=("actual", "sum"),
                eligible_actual_sum=("actual", lambda s: float(s[long.loc[s.index, "eligible"]].sum())),
            )
            .reset_index()
        )
        part.insert(0, "slice_type", "+".join(keys))
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def metric_generation_bin_summary(long: pd.DataFrame) -> pd.DataFrame:
    out = (
        long.groupby(["target", "year", "ratio_bin"], dropna=False, observed=False)
        .agg(
            rows=("actual", "size"),
            non_null=("actual", "count"),
            actual_sum=("actual", "sum"),
            eligible_count=("eligible", "sum"),
            eligible_rate=("eligible", "mean"),
            mean_actual_ratio=("actual_ratio", "mean"),
        )
        .reset_index()
    )
    total = out.groupby(["target", "year"])["actual_sum"].transform("sum")
    out["actual_sum_share"] = out["actual_sum"] / total
    return out


def ficr_boundary_distribution(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, cap in CAPACITY.items():
        target_df = long[long["target"] == target]
        eligible = target_df[target_df["eligible"]]
        rows.append(
            {
                "target": target,
                "capacity": cap,
                "six_pct_boundary_kwh": cap * 0.06,
                "eight_pct_boundary_kwh": cap * 0.08,
                "eligible_count": int(eligible["actual"].count()),
                "eligible_actual_sum": float(eligible["actual"].sum()),
                "eligible_mean_actual": float(eligible["actual"].mean()),
                "eligible_median_actual": float(eligible["actual"].median()),
                "eligible_p10_actual": float(eligible["actual"].quantile(0.10)),
                "eligible_p90_actual": float(eligible["actual"].quantile(0.90)),
                "boundary_8pct_as_share_of_eligible_mean": float((cap * 0.08) / eligible["actual"].mean()) if eligible["actual"].mean() else np.nan,
                "boundary_6pct_as_share_of_eligible_mean": float((cap * 0.06) / eligible["actual"].mean()) if eligible["actual"].mean() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def max_zero_run(series: pd.Series) -> int:
    vals = series.fillna(np.nan).to_numpy()
    best = cur = 0
    for v in vals:
        if pd.notna(v) and v == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def data_quality_inventory(data_dir: Path, labels: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        s = labels[target]
        ratio = s / CAPACITY[target]
        rows.append(
            {
                "scope": "label",
                "name": target,
                "rows": int(len(s)),
                "missing": int(s.isna().sum()),
                "negative": int((s < 0).sum()),
                "zero": int((s == 0).sum()),
                "capacity_exceed": int((ratio > 1).sum()),
                "max_zero_run_hours": max_zero_run(s),
                "min_value": float(s.min(skipna=True)),
                "max_value": float(s.max(skipna=True)),
            }
        )
    for rel in ["train/ldaps_train.csv", "train/gfs_train.csv", "test/ldaps_test.csv", "test/gfs_test.csv"]:
        path = data_dir / rel
        df = pd.read_csv(path, encoding="utf-8-sig")
        numeric = df.select_dtypes(include=[np.number])
        rows.append(
            {
                "scope": "weather",
                "name": rel,
                "rows": int(len(df)),
                "missing": int(df.isna().sum().sum()),
                "negative": np.nan,
                "zero": np.nan,
                "capacity_exceed": np.nan,
                "max_zero_run_hours": np.nan,
                "min_value": float(numeric.min().min()) if not numeric.empty else np.nan,
                "max_value": float(numeric.max().max()) if not numeric.empty else np.nan,
                "duplicate_forecast_grid_rows": int(df.duplicated(subset=["forecast_kst_dtm", "grid_id"]).sum()) if {"forecast_kst_dtm", "grid_id"}.issubset(df.columns) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def label_range_violations(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        s = labels[target]
        mask = s.notna() & ((s < 0) | (s > cap))
        for _, row in labels.loc[mask, ["kst_dtm", "year", "month", "hour", target]].iterrows():
            value = float(row[target])
            rows.append(
                {
                    "kst_dtm": row["kst_dtm"],
                    "target": target,
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "hour": int(row["hour"]),
                    "actual": value,
                    "capacity": cap,
                    "actual_ratio": value / cap,
                    "violation_type": "negative" if value < 0 else "capacity_exceed",
                }
            )
    return pd.DataFrame(rows)


def feature_source(col: str) -> str:
    if col.startswith("ldaps_"):
        return "ldaps"
    if col.startswith("gfs_"):
        return "gfs"
    if col in {"year", "month", "hour", "dayofyear", "month_sin", "month_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos"}:
        return "time"
    return "other"


def feature_stat(col: str) -> str:
    for suffix in ["_mean", "_min", "_max", "_std"]:
        if col.endswith(suffix):
            return suffix[1:]
    if feature_source(col) == "time":
        return "time"
    return "raw"


def ks_stat(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(a, b).statistic)
    except Exception:
        grid = np.unique(np.concatenate([a, b]))
        if len(grid) == 0:
            return np.nan
        return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a) - np.searchsorted(np.sort(b), grid, side="right") / len(b))))


def psi_stat(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    a = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if len(e) == 0 or len(a) == 0:
        return np.nan
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) <= 2:
        return 0.0
    e_counts, _ = np.histogram(e, bins=qs)
    a_counts, _ = np.histogram(a, bins=qs)
    e_pct = np.maximum(e_counts / max(e_counts.sum(), 1), 1e-6)
    a_pct = np.maximum(a_counts / max(a_counts.sum(), 1), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def feature_inventory_and_drift(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features, test_features = build_train_test_features(data_dir)
    train_22_23 = train_features[train_features["year"].isin([2022, 2023])].reset_index(drop=True)
    valid_2024 = train_features[train_features["year"].eq(2024)].reset_index(drop=True)
    test_2025 = test_features[test_features["year"].eq(2025)].reset_index(drop=True)
    feature_cols = [c for c in train_features.columns if c != "forecast_kst_dtm"]

    inv_rows = []
    drift_rows = []
    for col in feature_cols:
        train_s = train_22_23[col]
        valid_s = valid_2024[col]
        test_s = test_2025[col] if col in test_2025.columns else pd.Series(dtype=float)
        all_s = train_features[col]
        inv_rows.append(
            {
                "feature": col,
                "source": feature_source(col),
                "stat": feature_stat(col),
                "train_missing_rate": float(train_s.isna().mean()),
                "valid_missing_rate": float(valid_s.isna().mean()),
                "test_missing_rate": float(test_s.isna().mean()) if len(test_s) else np.nan,
                "train_unique": int(train_s.nunique(dropna=True)),
                "valid_unique": int(valid_s.nunique(dropna=True)),
                "test_unique": int(test_s.nunique(dropna=True)) if len(test_s) else np.nan,
                "near_constant": bool(all_s.nunique(dropna=True) <= 1),
                "train_mean": float(pd.to_numeric(train_s, errors="coerce").mean()),
                "valid_mean": float(pd.to_numeric(valid_s, errors="coerce").mean()),
                "test_mean": float(pd.to_numeric(test_s, errors="coerce").mean()) if len(test_s) else np.nan,
                "train_std": float(pd.to_numeric(train_s, errors="coerce").std()),
                "valid_std": float(pd.to_numeric(valid_s, errors="coerce").std()),
                "test_std": float(pd.to_numeric(test_s, errors="coerce").std()) if len(test_s) else np.nan,
            }
        )
        drift_rows.append(
            {
                "feature": col,
                "source": feature_source(col),
                "stat": feature_stat(col),
                "train_valid_ks": ks_stat(train_s, valid_s),
                "train_test_ks": ks_stat(train_s, test_s) if len(test_s) else np.nan,
                "valid_test_ks": ks_stat(valid_s, test_s) if len(test_s) else np.nan,
                "train_valid_psi": psi_stat(train_s, valid_s),
                "train_test_psi": psi_stat(train_s, test_s) if len(test_s) else np.nan,
                "valid_test_psi": psi_stat(valid_s, test_s) if len(test_s) else np.nan,
                "valid_mean_delta": float(pd.to_numeric(valid_s, errors="coerce").mean() - pd.to_numeric(train_s, errors="coerce").mean()),
                "test_mean_delta_vs_train": float(pd.to_numeric(test_s, errors="coerce").mean() - pd.to_numeric(train_s, errors="coerce").mean()) if len(test_s) else np.nan,
            }
        )
    inv = pd.DataFrame(inv_rows)
    drift = pd.DataFrame(drift_rows).sort_values(["train_valid_ks", "train_test_ks"], ascending=False)
    comp = (
        inv.groupby(["source", "stat"], dropna=False)
        .agg(
            feature_count=("feature", "size"),
            mean_train_missing=("train_missing_rate", "mean"),
            mean_valid_missing=("valid_missing_rate", "mean"),
            mean_test_missing=("test_missing_rate", "mean"),
            near_constant_count=("near_constant", "sum"),
        )
        .reset_index()
    )
    return inv, drift, comp


def scada_meta_inventory(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scada_rows = []
    for rel in ["train/scada_unison_train.csv", "train/scada_vestas_train.csv"]:
        path = data_dir / rel
        df = pd.read_csv(path, encoding="utf-8-sig")
        ts = pd.to_datetime(df["kst_dtm"], errors="coerce").sort_values()
        diffs = ts.drop_duplicates().diff().dropna()
        mode_gap = diffs.mode().iloc[0] if not diffs.empty else pd.NaT
        scada_rows.append(
            {
                "file": rel,
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "min_ts": str(ts.min()),
                "max_ts": str(ts.max()),
                "unique_timestamps": int(ts.nunique()),
                "mode_timestamp_gap": str(mode_gap),
                "missing_values": int(df.isna().sum().sum()),
                "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
            }
        )
    meta_rows = []
    info_path = data_dir / "info.xlsx"
    if info_path.exists():
        raw_info = pd.read_excel(info_path, sheet_name="info", header=None).dropna(how="all")
        header_idx = None
        for idx, row in raw_info.iterrows():
            if row.astype(str).str.contains("KPX그룹", na=False).any():
                header_idx = idx
                break
        if header_idx is not None:
            info = raw_info.loc[header_idx + 1 :].copy()
            info.columns = raw_info.loc[header_idx].fillna("").astype(str).str.strip().tolist()
            info = info.drop(columns=[c for c in info.columns if c == ""], errors="ignore").dropna(how="all")
        else:
            info = raw_info.copy()
        meta_rows.append({"metric": "rows", "value": int(len(info))})
        meta_rows.append({"metric": "columns", "value": int(info.shape[1])})
        if "KPX그룹" in info.columns:
            info["_KPX그룹_ffill"] = info["KPX그룹"].ffill()
        for col in ["KPX그룹", "제작사", "모델명"]:
            if col in info.columns:
                counts = info[col].value_counts(dropna=False)
                for key, value in counts.items():
                    meta_rows.append({"metric": f"{col}_count", "key": str(key), "value": int(value)})
        if "_KPX그룹_ffill" in info.columns:
            counts = info["_KPX그룹_ffill"].value_counts(dropna=False)
            for key, value in counts.items():
                meta_rows.append({"metric": "KPX그룹_ffill_count", "key": str(key), "value": int(value)})
        if "KPX그룹" in info.columns and "설비용량(MW)" in info.columns:
            group_col = "_KPX그룹_ffill" if "_KPX그룹_ffill" in info.columns else "KPX그룹"
            for group, value in info.groupby(group_col)["설비용량(MW)"].sum().items():
                meta_rows.append({"metric": "capacity_mw_by_group_from_info", "key": str(group), "value": float(value)})
        if "KPX그룹" in info.columns and "그룹설비용량(MW)" in info.columns:
            for group, value in info.groupby("KPX그룹")["그룹설비용량(MW)"].max().items():
                meta_rows.append({"metric": "group_capacity_mw_declared", "key": str(group), "value": float(value)})
    return pd.DataFrame(scada_rows), pd.DataFrame(meta_rows)


def weather_availability_inventory(data_dir: Path) -> pd.DataFrame:
    rows = []
    for rel in ["train/ldaps_train.csv", "train/gfs_train.csv", "test/ldaps_test.csv", "test/gfs_test.csv"]:
        path = data_dir / rel
        df = pd.read_csv(path, encoding="utf-8-sig", usecols=["forecast_kst_dtm", "data_available_kst_dtm", "grid_id"])
        forecast = pd.to_datetime(df["forecast_kst_dtm"], errors="coerce")
        available = pd.to_datetime(df["data_available_kst_dtm"], errors="coerce")
        lead_hours = (forecast - available).dt.total_seconds() / 3600.0
        per_forecast_grids = df.groupby("forecast_kst_dtm")["grid_id"].nunique()
        per_forecast_available = df.groupby("forecast_kst_dtm")["data_available_kst_dtm"].nunique()
        expected = pd.date_range(forecast.min(), forecast.max(), freq="h")
        unique_forecast = pd.Index(forecast.dropna().unique()).sort_values()
        rows.append(
            {
                "file": rel,
                "rows": int(len(df)),
                "unique_forecast_timestamps": int(forecast.nunique()),
                "forecast_min": str(forecast.min()),
                "forecast_max": str(forecast.max()),
                "missing_forecast_hours": int(len(expected.difference(unique_forecast))),
                "unique_data_available_timestamps": int(available.nunique()),
                "data_available_min": str(available.min()),
                "data_available_max": str(available.max()),
                "min_lead_hours": float(lead_hours.min()),
                "p50_lead_hours": float(lead_hours.median()),
                "max_lead_hours": float(lead_hours.max()),
                "unique_lead_hours": int(lead_hours.nunique()),
                "min_grid_count_per_forecast": int(per_forecast_grids.min()),
                "max_grid_count_per_forecast": int(per_forecast_grids.max()),
                "mode_grid_count_per_forecast": int(per_forecast_grids.mode().iloc[0]),
                "forecasts_with_multiple_available_times": int((per_forecast_available > 1).sum()),
                "duplicate_forecast_grid_rows": int(df.duplicated(subset=["forecast_kst_dtm", "grid_id"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def weather_schema_inventory(data_dir: Path) -> pd.DataFrame:
    rows = []
    for rel in ["train/ldaps_train.csv", "train/gfs_train.csv", "test/ldaps_test.csv", "test/gfs_test.csv"]:
        path = data_dir / rel
        df = pd.read_csv(path, encoding="utf-8-sig")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in df.columns:
            s = df[col]
            row = {
                "file": rel,
                "column": col,
                "dtype": str(s.dtype),
                "missing_rate": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
                "is_numeric": bool(col in numeric_cols),
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "nonfinite_count": np.nan,
            }
            if col in numeric_cols:
                ns = pd.to_numeric(s, errors="coerce")
                row.update(
                    {
                        "min": float(ns.min()),
                        "max": float(ns.max()),
                        "mean": float(ns.mean()),
                        "std": float(ns.std()),
                        "nonfinite_count": int((~np.isfinite(ns.to_numpy())).sum()),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def weather_schema_consistency(data_dir: Path) -> pd.DataFrame:
    rows = []
    for source in ["ldaps", "gfs"]:
        train_cols = pd.read_csv(data_dir / "train" / f"{source}_train.csv", encoding="utf-8-sig", nrows=1).columns.tolist()
        test_cols = pd.read_csv(data_dir / "test" / f"{source}_test.csv", encoding="utf-8-sig", nrows=1).columns.tolist()
        rows.append(
            {
                "source": source,
                "train_column_count": len(train_cols),
                "test_column_count": len(test_cols),
                "same_ordered_columns": train_cols == test_cols,
                "train_only_columns": ",".join(sorted(set(train_cols) - set(test_cols))),
                "test_only_columns": ",".join(sorted(set(test_cols) - set(train_cols))),
            }
        )
    return pd.DataFrame(rows)


def sample_horizon_alignment(data_dir: Path) -> pd.DataFrame:
    sample = pd.read_csv(data_dir / "sample_submission.csv", encoding="utf-8-sig")
    sample_ts = pd.to_datetime(sample["forecast_kst_dtm"], errors="coerce")
    rows = [
        {
            "source": "sample_submission",
            "rows": int(len(sample)),
            "min_ts": str(sample_ts.min()),
            "max_ts": str(sample_ts.max()),
            "unique_timestamps": int(sample_ts.nunique()),
            "missing_hour_gaps": int(len(pd.date_range(sample_ts.min(), sample_ts.max(), freq="h").difference(pd.Index(sample_ts.dropna().unique()).sort_values()))),
        }
    ]
    for source in ["ldaps", "gfs"]:
        df = pd.read_csv(data_dir / "test" / f"{source}_test.csv", encoding="utf-8-sig", usecols=["forecast_kst_dtm"])
        ts = pd.to_datetime(df["forecast_kst_dtm"], errors="coerce")
        unique_ts = pd.Index(ts.dropna().unique()).sort_values()
        rows.append(
            {
                "source": f"{source}_test",
                "rows": int(len(df)),
                "min_ts": str(ts.min()),
                "max_ts": str(ts.max()),
                "unique_timestamps": int(ts.nunique()),
                "missing_hour_gaps": int(len(pd.date_range(ts.min(), ts.max(), freq="h").difference(unique_ts))),
                "sample_timestamps_missing_from_source": int(len(pd.Index(sample_ts.dropna().unique()).difference(unique_ts))),
                "source_timestamps_not_in_sample": int(len(unique_ts.difference(pd.Index(sample_ts.dropna().unique())))),
            }
        )
    return pd.DataFrame(rows)


def target_temporal_extreme_summary(labels: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        s = labels[["kst_dtm", "year", "month", "hour", target]].copy()
        s["actual_ratio"] = s[target] / CAPACITY[target]
        nonnull = s[s[target].notna()]
        eligible = nonnull[nonnull["actual_ratio"] >= 0.10]
        rows.append(
            {
                "target": target,
                "scope": "all_nonnull",
                "rows": int(len(nonnull)),
                "mean_ratio": float(nonnull["actual_ratio"].mean()),
                "std_ratio": float(nonnull["actual_ratio"].std()),
                "zero_rate": float((nonnull[target] == 0).mean()),
                "near_zero_rate_lt_1pct": float((nonnull["actual_ratio"] < 0.01).mean()),
                "eligible_rate": float((nonnull["actual_ratio"] >= 0.10).mean()),
                "high_generation_rate_ge_80pct": float((nonnull["actual_ratio"] >= 0.80).mean()),
                "capacity_exceed_count": int((nonnull["actual_ratio"] > 1.0).sum()),
                "max_zero_run_hours": int(max_zero_run(s[target])),
                "max_ratio": float(nonnull["actual_ratio"].max()),
            }
        )
        for year, part in nonnull.groupby("year"):
            rows.append(
                {
                    "target": target,
                    "scope": f"year_{int(year)}",
                    "rows": int(len(part)),
                    "mean_ratio": float(part["actual_ratio"].mean()),
                    "std_ratio": float(part["actual_ratio"].std()),
                    "zero_rate": float((part[target] == 0).mean()),
                    "near_zero_rate_lt_1pct": float((part["actual_ratio"] < 0.01).mean()),
                    "eligible_rate": float((part["actual_ratio"] >= 0.10).mean()),
                    "high_generation_rate_ge_80pct": float((part["actual_ratio"] >= 0.80).mean()),
                    "capacity_exceed_count": int((part["actual_ratio"] > 1.0).sum()),
                    "max_zero_run_hours": int(max_zero_run(part[target])),
                    "max_ratio": float(part["actual_ratio"].max()),
                }
            )
        hourly = eligible.groupby("hour")["actual_ratio"].mean()
        monthly = eligible.groupby("month")["actual_ratio"].mean()
        if len(hourly):
            rows.append(
                {
                    "target": target,
                    "scope": "eligible_hourly_range",
                    "rows": int(len(eligible)),
                    "mean_ratio": float(eligible["actual_ratio"].mean()),
                    "std_ratio": float(eligible["actual_ratio"].std()),
                    "eligible_rate": np.nan,
                    "high_generation_rate_ge_80pct": float((eligible["actual_ratio"] >= 0.80).mean()),
                    "capacity_exceed_count": int((eligible["actual_ratio"] > 1.0).sum()),
                    "max_zero_run_hours": np.nan,
                    "max_ratio": float(eligible["actual_ratio"].max()),
                    "best_hour": int(hourly.idxmax()),
                    "worst_hour": int(hourly.idxmin()),
                    "hourly_mean_ratio_range": float(hourly.max() - hourly.min()),
                }
            )
        if len(monthly):
            rows.append(
                {
                    "target": target,
                    "scope": "eligible_monthly_range",
                    "rows": int(len(eligible)),
                    "mean_ratio": float(eligible["actual_ratio"].mean()),
                    "std_ratio": float(eligible["actual_ratio"].std()),
                    "eligible_rate": np.nan,
                    "high_generation_rate_ge_80pct": float((eligible["actual_ratio"] >= 0.80).mean()),
                    "capacity_exceed_count": int((eligible["actual_ratio"] > 1.0).sum()),
                    "max_zero_run_hours": np.nan,
                    "max_ratio": float(eligible["actual_ratio"].max()),
                    "best_month": int(monthly.idxmax()),
                    "worst_month": int(monthly.idxmin()),
                    "monthly_mean_ratio_range": float(monthly.max() - monthly.min()),
                }
            )
    return pd.DataFrame(rows)


def capacity_exceed_context(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        cap = CAPACITY[target]
        df = labels[["kst_dtm", "year", "month", "hour", target]].copy()
        df["actual_ratio"] = df[target] / cap
        exceed = df[df["actual_ratio"] > 1.0].copy()
        for _, row in exceed.iterrows():
            same_month_hour = df[(df["year"] == row["year"]) & (df["month"] == row["month"]) & (df["hour"] == row["hour"]) & df[target].notna()]
            rows.append(
                {
                    "kst_dtm": row["kst_dtm"],
                    "target": target,
                    "year": int(row["year"]),
                    "month": int(row["month"]),
                    "hour": int(row["hour"]),
                    "actual": float(row[target]),
                    "capacity": float(cap),
                    "actual_ratio": float(row["actual_ratio"]),
                    "excess_kwh": float(row[target] - cap),
                    "same_year_month_hour_count": int(len(same_month_hour)),
                    "same_year_month_hour_p95_ratio": float(same_month_hour["actual_ratio"].quantile(0.95)) if len(same_month_hour) else np.nan,
                    "same_year_month_hour_max_ratio": float(same_month_hour["actual_ratio"].max()) if len(same_month_hour) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_reports(results: Path, tables: dict[str, pd.DataFrame]) -> None:
    label_cov = tables["label_coverage"]
    ratio = tables["target_ratio_bin_summary"]
    metric_bins = tables["metric_generation_bin_summary"]
    data_quality = tables["data_quality_inventory"]
    timestamp = tables["timestamp_coverage"]
    feature_inventory = tables.get("feature_inventory", pd.DataFrame())
    feature_drift = tables.get("feature_drift_summary", pd.DataFrame())
    feature_comp = tables.get("feature_composition_summary", pd.DataFrame())
    scada_inventory = tables.get("scada_inventory", pd.DataFrame())
    meta_inventory = tables.get("meta_inventory", pd.DataFrame())
    weather_availability = tables.get("weather_availability_inventory", pd.DataFrame())
    weather_schema_cons = tables.get("weather_schema_consistency", pd.DataFrame())
    target_extreme = tables.get("target_temporal_extreme_summary", pd.DataFrame())
    capacity_context = tables.get("capacity_exceed_context", pd.DataFrame())
    sample_alignment = tables.get("sample_horizon_alignment", pd.DataFrame())

    def md_table(df: pd.DataFrame, n: int = 20) -> str:
        return df.head(n).to_markdown(index=False)

    # High-level facts.
    valid_rows = int(label_cov["non_null"].sum())
    missing_labels = int(label_cov["missing"].sum())
    cap_exceed = int(label_cov["capacity_exceed_count"].sum())
    neg = int(label_cov["negative_count"].sum())
    eligible = int(label_cov["eligible_count"].sum())
    label_non_null = int(label_cov["non_null"].sum())
    eligible_rate = eligible / label_non_null if label_non_null else np.nan

    year_shift = tables["target_distribution_summary"]
    year_rows = year_shift[year_shift["slice_type"].eq("target+year")][["target", "year", "mean", "p50", "p90"]].copy()
    pivot = year_rows.pivot(index="target", columns="year", values="mean")
    shift_lines = []
    for target in pivot.index:
        if 2024 in pivot.columns and 2023 in pivot.columns:
            shift_lines.append(f"- {target}: 2024 mean ratio {pivot.loc[target, 2024]:.4f}, 2023 mean ratio {pivot.loc[target, 2023]:.4f}, delta {pivot.loc[target, 2024] - pivot.loc[target, 2023]:+.4f}")

    eligible_bins = metric_bins[metric_bins["eligible_count"] > 0].copy()
    bin_focus = (
        eligible_bins.groupby("ratio_bin", observed=False)
        .agg(eligible_count=("eligible_count", "sum"), actual_sum=("actual_sum", "sum"))
        .reset_index()
        .sort_values("eligible_count", ascending=False)
    )
    total_eligible_count = bin_focus["eligible_count"].sum()
    total_actual_sum = bin_focus["actual_sum"].sum()
    bin_focus["eligible_share"] = bin_focus["eligible_count"] / total_eligible_count
    bin_focus["actual_sum_share"] = bin_focus["actual_sum"] / total_actual_sum

    drift_fact_lines = []
    if not feature_inventory.empty:
        n_features = int(len(feature_inventory))
        n_sources = feature_inventory["source"].nunique(dropna=True)
        max_train_missing = float(feature_inventory["train_missing_rate"].max())
        max_valid_missing = float(feature_inventory["valid_missing_rate"].max())
        max_test_missing = float(feature_inventory["test_missing_rate"].max())
        near_constant = int(feature_inventory["near_constant"].sum())
        drift_fact_lines.extend(
            [
                f"- B1 aggregate feature count: {n_features:,} across {n_sources} source groups",
                f"- max missing rate train/valid/test: {max_train_missing:.4f} / {max_valid_missing:.4f} / {max_test_missing:.4f}",
                f"- near-constant feature count: {near_constant}",
            ]
        )
    if not feature_drift.empty:
        high_tv_ks = int(feature_drift["train_valid_ks"].gt(0.30).sum())
        high_tt_ks = int(feature_drift["train_test_ks"].gt(0.30).sum())
        high_tv_psi = int(feature_drift["train_valid_psi"].gt(0.25).sum())
        high_tt_psi = int(feature_drift["train_test_psi"].gt(0.25).sum())
        non_time = feature_drift[feature_drift["source"].ne("time")]
        high_tv_ks_non_time = int(non_time["train_valid_ks"].gt(0.30).sum())
        high_tt_ks_non_time = int(non_time["train_test_ks"].gt(0.30).sum())
        drift_fact_lines.extend(
            [
                f"- features with train-valid KS > 0.30: {high_tv_ks} total, {high_tv_ks_non_time} excluding time features",
                f"- features with train-test KS > 0.30: {high_tt_ks} total, {high_tt_ks_non_time} excluding time features",
                f"- features with train-valid PSI > 0.25: {high_tv_psi}",
                f"- features with train-test PSI > 0.25: {high_tt_psi}",
            ]
        )
    if not scada_inventory.empty:
        drift_fact_lines.append(f"- SCADA files inventoried: {len(scada_inventory)}")
    if not meta_inventory.empty:
        info_rows = meta_inventory[meta_inventory["metric"].eq("rows")]
        if not info_rows.empty:
            drift_fact_lines.append(f"- info.xlsx turbine/meta rows: {int(info_rows.iloc[0]['value'])}")

    weather_lines = []
    if not weather_availability.empty:
        min_lead = weather_availability["min_lead_hours"].min()
        max_lead = weather_availability["max_lead_hours"].max()
        multi_avail = int(weather_availability["forecasts_with_multiple_available_times"].sum())
        dup_weather = int(weather_availability["duplicate_forecast_grid_rows"].sum())
        weather_lines.extend(
            [
                f"- forecast lead-hour range across weather files: {min_lead:.1f} to {max_lead:.1f}",
                f"- forecasts with multiple data_available timestamps: {multi_avail}",
                f"- duplicate forecast/grid rows: {dup_weather}",
            ]
        )
    if not weather_schema_cons.empty:
        same_schema = bool(weather_schema_cons["same_ordered_columns"].all())
        weather_lines.append(f"- train/test weather schemas have identical ordered columns: {same_schema}")
    if not sample_alignment.empty:
        missing_from_sources = sample_alignment.filter(like="sample_timestamps_missing_from_source").sum(numeric_only=True).sum()
        weather_lines.append(f"- sample timestamps missing from test weather sources: {int(missing_from_sources)}")

    target_extreme_lines = []
    if not target_extreme.empty:
        all_rows = target_extreme[target_extreme["scope"].eq("all_nonnull")]
        for _, row in all_rows.iterrows():
            target_extreme_lines.append(
                f"- {row['target']}: eligible_rate {row['eligible_rate']:.4f}, zero_rate {row['zero_rate']:.4f}, high>=80% rate {row['high_generation_rate_ge_80pct']:.4f}, max_zero_run_hours {int(row['max_zero_run_hours'])}, max_ratio {row['max_ratio']:.4f}"
            )
    if not capacity_context.empty:
        max_excess = float(capacity_context["excess_kwh"].max())
        max_ratio = float(capacity_context["actual_ratio"].max())
        target_extreme_lines.append(f"- capacity-exceed details: {len(capacity_context)} rows, max excess {max_excess:.3f} kWh, max ratio {max_ratio:.6f}")

    route = "temporal drift and feature inventory audit"
    reason = "Target means and eligible distribution vary materially by year/month/hour; B1 feature coverage is mostly stable, so the next non-modeling uncertainty is target/temporal structure unless data-quality blockers take precedence."
    if cap_exceed or neg:
        route = "data quality / cleansing audit"
        reason = "Capacity-exceed label values exist and must be explained before any cleansing rule, model HPO, or feature expansion."
    elif any("missing_hour_gaps" in timestamp.columns and timestamp["missing_hour_gaps"].fillna(0).gt(0)):
        route = "data quality / timestamp coverage audit"
        reason = "Timestamp gaps are present and should be explained before modeling changes."

    problem = [
        "# Foundational EDA and Problem Framing",
        "",
        "## Scope",
        "",
        "- No model training, no feature engineering, no cleansing, no HPO, no submission.",
        "- This report inventories dataset, target, and metric behavior to reset the ML optimization sequence.",
        "",
        "## Key facts",
        "",
        f"- non-null label row-targets: {label_non_null:,}",
        f"- missing label row-targets: {missing_labels:,}",
        f"- official eligible row-targets (actual >= 10% capacity): {eligible:,}",
        f"- official eligible rate among non-null labels: {eligible_rate:.4f}",
        f"- negative label count: {neg}",
        f"- capacity exceed count: {cap_exceed}",
        "",
        "## Dataset inventory",
        "",
        md_table(tables["dataset_inventory"]),
        "",
        "## Forecast availability and raw weather consistency",
        "",
        "\n".join(weather_lines) if weather_lines else "Weather availability inventory was not available.",
        "",
        "### Weather availability",
        "",
        md_table(weather_availability) if not weather_availability.empty else "No weather availability table.",
        "",
        "### Weather schema consistency",
        "",
        md_table(weather_schema_cons) if not weather_schema_cons.empty else "No weather schema consistency table.",
        "",
        "### Sample/test horizon alignment",
        "",
        md_table(sample_alignment) if not sample_alignment.empty else "No sample horizon alignment table.",
        "",
        "## Label coverage",
        "",
        md_table(label_cov),
        "",
        "## Target temporal/extreme structure",
        "",
        "\n".join(target_extreme_lines) if target_extreme_lines else "No target temporal/extreme table.",
        "",
        "### Target temporal/extreme summary",
        "",
        md_table(target_extreme, 30) if not target_extreme.empty else "No target temporal/extreme table.",
        "",
        "### Capacity exceed context",
        "",
        md_table(capacity_context, 20) if not capacity_context.empty else "No capacity exceed context rows.",
        "",
        "## Eligible generation-bin structure",
        "",
        md_table(bin_focus),
        "",
        "## Year-level target shift signals",
        "",
        "\n".join(shift_lines) if shift_lines else "No year shift rows available.",
        "",
        "## FICR boundaries",
        "",
        md_table(tables["ficr_boundary_distribution"]),
        "",
        "## B1 feature inventory and drift",
        "",
        "\n".join(drift_fact_lines) if drift_fact_lines else "Feature inventory was not available.",
        "",
        "### Feature source/stat composition",
        "",
        md_table(feature_comp) if not feature_comp.empty else "No feature composition table.",
        "",
        "### Top train-valid drift features",
        "",
        md_table(feature_drift.sort_values("train_valid_ks", ascending=False), 15) if not feature_drift.empty else "No feature drift table.",
        "",
        "### Top train-test drift features",
        "",
        md_table(feature_drift.sort_values("train_test_ks", ascending=False), 15) if not feature_drift.empty else "No feature drift table.",
        "",
        "## SCADA/meta inventory",
        "",
        md_table(scada_inventory) if not scada_inventory.empty else "No SCADA inventory table.",
        "",
        "## Initial interpretation",
        "",
        "- The official metric excludes actual/capacity below 10%, so near-zero rows are primarily a training/data-behavior concern, not direct official-score mass.",
        "- The eligible set is dominated by 10~80% ratio bins by count, while high-ratio bins carry high actual mass and stricter underprediction risk.",
        "- Group/year coverage differs structurally because group3 has missing labels in 2022 by competition design.",
        "- Raw weather availability and train/test schema are structurally usable for a fixed B1 feature baseline, but this does not prove feature sufficiency.",
        "- B1 aggregate feature drift appears mild after excluding the deterministic `year` feature; this weakens the case for more model-only search as the next step.",
        "- The correct next action is a focused audit issue, not another broad modeling run. The focused route must be selected from hard data-quality blockers, target/temporal shift, and B1 feature coverage/drift.",
    ]
    (results / "problem_framing.md").write_text("\n".join(problem), encoding="utf-8")

    routing = [
        "# Next Audit Routing",
        "",
        f"- recommended next issue: **{route}**",
        f"- evidence: {reason}",
        "- priority: high",
        "- what not to do yet: do not resume CatBoost HPO, do not apply cleansing, and do not add feature families until the next focused audit explains whether the issue is drift, feature coverage, or data quality.",
        "",
        "## Why not model HPO now",
        "",
        "- The project now has model hints, but the foundational inventory shows the next uncertainty is problem structure: target regime, year/time distribution, metric eligibility, and feature coverage.",
        "- HPO should follow after data/target/metric and feature inventory identify stable intervention targets.",
        "",
        "## Recommended next issue scope",
        "",
        "Open a focused `Data quality / cleansing audit` issue first, because a hard label-range blocker is present:",
        "",
        "- enumerate all capacity-exceed rows and their timestamp/target/month/hour/regime",
        "- verify whether exceedance is true production, capacity metadata mismatch, rounding/noise, aggregation issue, or label anomaly",
        "- test candidate cleansing rules diagnostically only: clip-to-capacity, remove rows, keep-as-is, target-specific rule",
        "- replay historical validation without adding features or HPO to measure whether each rule changes metric-facing behavior",
        "- decide one explicit policy: no cleansing, clipping, exclusion, or metadata correction",
        "- only after this is resolved, move to temporal drift / feature sufficiency audit",
        "",
        "## Supporting tables",
        "",
        "- `dataset_inventory.csv`",
        "- `timestamp_coverage.csv`",
        "- `weather_availability_inventory.csv`",
        "- `weather_schema_inventory.csv`",
        "- `weather_schema_consistency.csv`",
        "- `sample_horizon_alignment.csv`",
        "- `label_coverage.csv`",
        "- `target_temporal_extreme_summary.csv`",
        "- `target_distribution_summary.csv`",
        "- `target_ratio_bin_summary.csv`",
        "- `target_by_year_month_hour.csv`",
        "- `target_group_correlation.csv`",
        "- `metric_eligible_distribution.csv`",
        "- `metric_generation_bin_summary.csv`",
        "- `ficr_boundary_distribution.csv`",
        "- `data_quality_inventory.csv`",
        "- `label_range_violations.csv`",
        "- `capacity_exceed_context.csv`",
        "- `feature_inventory.csv`",
        "- `feature_drift_summary.csv`",
        "- `feature_composition_summary.csv`",
        "- `scada_inventory.csv`",
        "- `meta_inventory.csv`",
    ]
    (results / "next_audit_routing.md").write_text("\n".join(routing), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = ensure_dirs(args.out_dir)

    inv, ts_cov = dataset_inventory(args.data_dir)
    labels = read_labels(args.data_dir)
    long = make_long_labels(labels)
    feature_inventory, feature_drift, feature_comp = feature_inventory_and_drift(args.data_dir)
    scada_inventory, meta_inventory = scada_meta_inventory(args.data_dir)
    weather_availability = weather_availability_inventory(args.data_dir)
    weather_schema = weather_schema_inventory(args.data_dir)
    weather_schema_cons = weather_schema_consistency(args.data_dir)
    sample_alignment = sample_horizon_alignment(args.data_dir)

    tables = {
        "dataset_inventory": inv,
        "timestamp_coverage": ts_cov,
        "weather_availability_inventory": weather_availability,
        "weather_schema_inventory": weather_schema,
        "weather_schema_consistency": weather_schema_cons,
        "sample_horizon_alignment": sample_alignment,
        "label_coverage": label_coverage(long),
        "target_temporal_extreme_summary": target_temporal_extreme_summary(labels, long),
        "target_distribution_summary": target_distribution(long),
        "target_ratio_bin_summary": target_ratio_bins(long),
        "target_by_year_month_hour": target_by_year_month_hour(long),
        "target_group_correlation": target_group_correlation(labels),
        "metric_eligible_distribution": metric_eligible_distribution(long),
        "metric_generation_bin_summary": metric_generation_bin_summary(long),
        "ficr_boundary_distribution": ficr_boundary_distribution(long),
        "data_quality_inventory": data_quality_inventory(args.data_dir, labels, long),
        "label_range_violations": label_range_violations(labels),
        "capacity_exceed_context": capacity_exceed_context(labels),
        "feature_inventory": feature_inventory,
        "feature_drift_summary": feature_drift,
        "feature_composition_summary": feature_comp,
        "scada_inventory": scada_inventory,
        "meta_inventory": meta_inventory,
    }

    for name, df in tables.items():
        df.to_csv(results / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_reports(results, tables)

    sanity = {
        "result_files": sorted(p.name for p in results.glob("*")),
        "label_rows": int(len(labels)),
        "long_label_rows": int(len(long)),
        "expected_long_label_rows": int(len(labels) * len(TARGETS)),
    }
    (results / "sanity_check.json").write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    print((results / "problem_framing.md").read_text(encoding="utf-8"))
    print()
    print((results / "next_audit_routing.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

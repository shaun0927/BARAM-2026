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


def write_reports(results: Path, tables: dict[str, pd.DataFrame]) -> None:
    label_cov = tables["label_coverage"]
    ratio = tables["target_ratio_bin_summary"]
    metric_bins = tables["metric_generation_bin_summary"]
    data_quality = tables["data_quality_inventory"]
    timestamp = tables["timestamp_coverage"]

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

    route = "temporal drift audit plus feature inventory"
    reason = "Target means and eligible distribution vary materially by year/month/hour; before feature engineering or HPO, quantify train-valid/test-facing drift and inspect feature coverage."
    if cap_exceed or neg:
        route = "data quality / cleansing audit"
        reason = "Range violations exist in labels and must be explained before modeling changes."
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
        "## Label coverage",
        "",
        md_table(label_cov),
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
        "## Initial interpretation",
        "",
        "- The official metric excludes actual/capacity below 10%, so near-zero rows are primarily a training/data-behavior concern, not direct official-score mass.",
        "- The eligible set is dominated by 10~80% ratio bins by count, while high-ratio bins carry high actual mass and stricter underprediction risk.",
        "- Group/year coverage differs structurally because group3 has missing labels in 2022 by competition design.",
        "- Before any more model HPO, the next decision should be based on target-year/time structure and feature coverage/drift, not on a single model's residuals alone.",
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
        "If no hard data-quality blocker is present, open a focused `Temporal drift and feature inventory audit` issue:",
        "",
        "- compare train years 2022/2023 vs validation 2024 target distributions by group/month/hour/bin",
        "- inspect B1 feature inventory and train-valid/test feature coverage",
        "- quantify feature drift and missingness drift",
        "- connect drift/feature coverage to the generation regimes identified here",
        "- do not train new models until this audit routes an intervention",
        "",
        "## Supporting tables",
        "",
        "- `dataset_inventory.csv`",
        "- `timestamp_coverage.csv`",
        "- `label_coverage.csv`",
        "- `target_distribution_summary.csv`",
        "- `target_ratio_bin_summary.csv`",
        "- `target_by_year_month_hour.csv`",
        "- `target_group_correlation.csv`",
        "- `metric_eligible_distribution.csv`",
        "- `metric_generation_bin_summary.csv`",
        "- `ficr_boundary_distribution.csv`",
        "- `data_quality_inventory.csv`",
        "- `label_range_violations.csv`",
    ]
    (results / "next_audit_routing.md").write_text("\n".join(routing), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = ensure_dirs(args.out_dir)

    inv, ts_cov = dataset_inventory(args.data_dir)
    labels = read_labels(args.data_dir)
    long = make_long_labels(labels)

    tables = {
        "dataset_inventory": inv,
        "timestamp_coverage": ts_cov,
        "label_coverage": label_coverage(long),
        "target_distribution_summary": target_distribution(long),
        "target_ratio_bin_summary": target_ratio_bins(long),
        "target_by_year_month_hour": target_by_year_month_hour(long),
        "target_group_correlation": target_group_correlation(labels),
        "metric_eligible_distribution": metric_eligible_distribution(long),
        "metric_generation_bin_summary": metric_generation_bin_summary(long),
        "ficr_boundary_distribution": ficr_boundary_distribution(long),
        "data_quality_inventory": data_quality_inventory(args.data_dir, labels, long),
        "label_range_violations": label_range_violations(labels),
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

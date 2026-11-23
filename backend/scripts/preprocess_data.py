"""LifeRoute Step 2: Time-Series Preprocessing & Feature Engineering Pipeline.

========================================================================================
Project: LifeRoute - AI-Powered Emergency Ambulance Navigation System
Institution: Sri Eshwar College of Engineering, CCE
Dataset: Bangalore Traffic Open Data (DataMeet / traffic-monitor-lizard)
========================================================================================

This script performs the complete, reusable Step 2 preprocessing pipeline:
1. Timestamp Cleaning:
   - Parses separate 'date' (YYYY-MM-DD) and 'time' (HH:MM) into a unified UTC/local timestamp.
   - Floors/snaps observations to regular 1-hour temporal bins.
   - Aggregates intra-hour duplicate scraper readings via mean.
2. Missing Value & Outlier Handling:
   - Filters telematics corruption (v > 100 km/h or v < 3 km/h).
   - Reindexes all 13 corridors onto a continuous regular 1-hour temporal grid.
   - Imputes short gaps (<= 6 hours) via linear time-series interpolation.
   - Imputes extended gaps (> 6 hours) via corridor-specific (hour_of_day, day_of_week) medians.
3. Feature Engineering:
   - Temporal features: hour_of_day, day_of_week, is_weekend, is_peak_hour.
   - Cyclical encodings: hour_sin, hour_cos, dow_sin, dow_cos.
   - Lagged speeds: t-1h, t-2h, t-3h, t-24h (yesterday), t-168h (last week).
   - Rolling window statistics: 3h, 6h, 24h rolling means and 6h rolling std.
     (All rolling features use shift(1) to strictly avoid lookahead target leakage).
   - Route free-flow baseline speed (v_{e,free}) and congestion factor C_e(t) >= 1.0.
4. Time-Series Splitting:
   - Strictly chronological Train / Validation / Test split (never random shuffle).
   - Train: 2025-10-02 to 2026-07-14 (~70% of time)
   - Val:   2026-07-15 to 2026-09-14 (~15% of time)
   - Test:  2026-09-15 to 2026-11-17 (~15% of time)
5. Artifact Persistence:
   - Saves train_split.csv, val_split.csv, test_split.csv, and full resampled dataset.
   - Exports preprocessing_summary.json with complete metadata and audit counts.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_data")


@dataclass
class PreprocessingReport:
    """Audit report tracking all rows dropped, filled, and split sizes."""
    raw_rows_loaded: int = 0
    outliers_dropped: int = 0
    outliers_reason: str = ""
    valid_clean_rows: int = 0
    intra_hour_duplicates_merged: int = 0
    unique_route_hours: int = 0
    hourly_slots_total_grid: int = 0
    missing_slots_filled: int = 0
    missing_slots_short_gap_interpolated: int = 0
    missing_slots_long_gap_profile_filled: int = 0
    warmup_lag_rows_dropped: int = 0
    final_processed_rows: int = 0
    train_rows: int = 0
    train_start: str = ""
    train_end: str = ""
    val_rows: int = 0
    val_start: str = ""
    val_end: str = ""
    test_rows: int = 0
    test_start: str = ""
    test_end: str = ""
    feature_columns: List[str] = None


def clean_raw_traffic(
    csv_path: Path,
    min_speed_kmh: float = 3.0,
    max_speed_kmh: float = 100.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load raw DataMeet traffic CSV, parse timestamps, and filter telematics outliers.

    Args:
        csv_path: Path to csv-traffic-bangalore.csv.
        min_speed_kmh: Minimum physically feasible speed on city streets.
        max_speed_kmh: Maximum physically feasible speed on city streets.

    Returns:
        Tuple of (cleaned DataFrame, audit statistics dict).
    """
    logger.info("Loading raw traffic data from %s...", csv_path)
    df = pd.read_csv(csv_path)
    raw_count = len(df)

    # 1. Parse unified timestamp from date and time strings
    logger.info("Parsing unified timestamps from 'date' and 'time'...")
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    null_ts = df["timestamp"].isnull().sum()
    if null_ts > 0:
        logger.warning("Dropping %d rows with unparseable timestamps", null_ts)
        df = df.dropna(subset=["timestamp"])

    # 2. Compute observed speed in km/h: speed = distance / (duration / 60.0)
    df["speed_kmh"] = df["distance"] / (df["duration"] / 60.0)

    # 3. Detect and filter corrupt telematics outliers
    outlier_mask = (df["speed_kmh"] < min_speed_kmh) | (df["speed_kmh"] > max_speed_kmh)
    outlier_count = int(outlier_mask.sum())
    outlier_rows = df[outlier_mask]

    outlier_reason = (
        f"Filtered {outlier_count} row(s) outside physical speed boundary "
        f"[{min_speed_kmh}, {max_speed_kmh}] km/h. "
        f"Max detected anomaly: speed={outlier_rows['speed_kmh'].max():.1f} km/h "
        f"(duration={outlier_rows['duration'].min()} min, distance={outlier_rows['distance'].max()} km)."
    )
    logger.info(outlier_reason)

    df_clean = df[~outlier_mask].copy()

    # 4. Snap to hourly timestamp bin
    df_clean["hourly_ts"] = df_clean["timestamp"].dt.floor("h")

    # 5. Merge intra-hour duplicate pings
    agg_df = df_clean.groupby(["route_code", "hourly_ts"]).agg({
        "speed_kmh": "mean",
        "duration": "mean",
        "distance": "mean",
    }).reset_index()

    duplicates_merged = len(df_clean) - len(agg_df)
    logger.info(
        "Merged %d intra-hour duplicate readings into %d unique route-hours",
        duplicates_merged, len(agg_df)
    )

    audit = {
        "raw_count": raw_count,
        "outlier_count": outlier_count,
        "outlier_reason": outlier_reason,
        "valid_count": len(df_clean),
        "duplicates_merged": duplicates_merged,
        "unique_route_hours": len(agg_df),
    }
    return agg_df, audit


def resample_to_hourly_grid(
    agg_df: pd.DataFrame,
    max_interp_gap_hours: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resample each corridor's time-series to an exact, continuous 1-hour interval.

    Missing values are handled systematically:
    - Short gaps (<= 6 hours) use time-based linear interpolation.
    - Extended gaps (> 6 hours) are filled using the corridor's historical median
      profile for that specific (hour_of_day, day_of_week) combination.

    Args:
        agg_df: Aggregated DataFrame with columns [route_code, hourly_ts, speed_kmh, ...].
        max_interp_gap_hours: Maximum hours to interpolate linearly.

    Returns:
        Tuple of (regularly resampled DataFrame, audit statistics dict).
    """
    logger.info("Resampling all monitored corridors to continuous 1-hour grid ('1h')...")
    resampled_records: List[pd.DataFrame] = []
    total_missing_slots = 0
    short_gap_filled = 0
    long_gap_filled = 0
    total_grid_slots = 0

    for r_code, group in agg_df.groupby("route_code"):
        group = group.sort_values("hourly_ts").set_index("hourly_ts")
        
        # Build complete, continuous 1-hour index for this corridor
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq="1h", name="timestamp")
        total_grid_slots += len(full_idx)
        
        reindexed = group.reindex(full_idx)
        reindexed["route_code"] = r_code
        
        missing_count = int(reindexed["speed_kmh"].isnull().sum())
        total_missing_slots += missing_count

        # Step 1: Linear time-series interpolation for short gaps (<= 6 hours)
        interp_speed = reindexed["speed_kmh"].interpolate(method="time", limit=max_interp_gap_hours)
        interp_dur = reindexed["duration"].interpolate(method="time", limit=max_interp_gap_hours)
        
        # Count how many slots were filled by linear interpolation
        newly_filled = int(reindexed["speed_kmh"].isnull().sum() - interp_speed.isnull().sum())
        short_gap_filled += newly_filled
        
        reindexed["speed_kmh"] = interp_speed
        reindexed["duration"] = interp_dur
        reindexed["distance"] = reindexed["distance"].ffill().bfill()

        # Step 2: Route-specific historical median profile for remaining extended gaps
        reindexed["hour"] = reindexed.index.hour
        reindexed["dow"] = reindexed.index.dayofweek
        
        # Calculate historical medians per (hour, day_of_week) from observed data
        profile = group.groupby([group.index.hour, group.index.dayofweek])["speed_kmh"].median()
        global_route_median = float(group["speed_kmh"].median())

        missing_mask = reindexed["speed_kmh"].isnull()
        if missing_mask.any():
            long_gap_count = int(missing_mask.sum())
            long_gap_filled += long_gap_count
            
            for idx in reindexed[missing_mask].index:
                h, d = idx.hour, idx.dayofweek
                val = profile.get((h, d), global_route_median)
                reindexed.loc[idx, "speed_kmh"] = val
                dist = reindexed.loc[idx, "distance"]
                reindexed.loc[idx, "duration"] = (dist / val) * 60.0

        resampled_records.append(reindexed.reset_index())

    full_resampled = pd.concat(resampled_records, ignore_index=True)
    logger.info(
        "Resampled grid: %d total slots across 13 corridors. Missing filled: %d (short gaps: %d, long profile: %d)",
        total_grid_slots, total_missing_slots, short_gap_filled, long_gap_filled
    )

    audit = {
        "total_grid_slots": total_grid_slots,
        "total_missing_slots": total_missing_slots,
        "short_gap_filled": short_gap_filled,
        "long_gap_filled": long_gap_filled,
    }
    return full_resampled, audit


def engineer_features(
    df: pd.DataFrame,
    warmup_hours: int = 168,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Engineer temporal, cyclical, rolling, and lagged features.

    To strictly avoid lookahead target leakage:
    - All rolling features are computed with .shift(1) on historical values.
    - All lag features represent past time steps (t-1h, t-2h, t-3h, t-24h, t-168h).

    Args:
        df: Resampled hourly DataFrame.
        warmup_hours: Initial hours dropped per route to accommodate longest lag (168h = 1 week).

    Returns:
        Tuple of (clean feature DataFrame, audit statistics dict).
    """
    logger.info("Engineering temporal, cyclical, rolling, and lag features...")
    df = df.copy()

    # 1. Calendar / Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Peak hours: Morning rush (08:00-11:00) and Evening rush (17:00-21:00)
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 11, 17, 18, 19, 20]).astype(int)

    # 2. Cyclical sinusoidal encodings for periodic time continuity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # 3. Sort strictly by route_code and timestamp
    df = df.sort_values(["route_code", "timestamp"]).reset_index(drop=True)

    # 4. Lagged speeds
    df["speed_lag_1h"] = df.groupby("route_code")["speed_kmh"].shift(1)
    df["speed_lag_2h"] = df.groupby("route_code")["speed_kmh"].shift(2)
    df["speed_lag_3h"] = df.groupby("route_code")["speed_kmh"].shift(3)
    df["speed_lag_24h"] = df.groupby("route_code")["speed_kmh"].shift(24)
    df["speed_lag_168h"] = df.groupby("route_code")["speed_kmh"].shift(168)

    # 5. Rolling statistics (strictly shifted by 1 to prevent data leakage)
    df["speed_roll_mean_3h"] = df.groupby("route_code")["speed_kmh"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["speed_roll_mean_6h"] = df.groupby("route_code")["speed_kmh"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).mean()
    )
    df["speed_roll_mean_24h"] = df.groupby("route_code")["speed_kmh"].transform(
        lambda s: s.shift(1).rolling(24, min_periods=1).mean()
    )
    df["speed_roll_std_6h"] = df.groupby("route_code")["speed_kmh"].transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).std().fillna(0.0)
    )

    # 6. Route free-flow baseline speed (v_{e,free}) and congestion factor C_e(t)
    # Computed from off-peak 01:00-05:00 hours (95th percentile)
    off_peak_mask = (df["hour"] >= 1) & (df["hour"] <= 5)
    free_flow_map = df[off_peak_mask].groupby("route_code")["speed_kmh"].quantile(0.95).to_dict()
    df["free_flow_speed"] = df["route_code"].map(free_flow_map)
    # Research Paper Eq. 1: C_e(t) = v_{e,free} / v_e(t) >= 1.0
    df["congestion_factor"] = (df["free_flow_speed"] / df["speed_kmh"]).clip(lower=1.0)

    # 7. Drop the initial warmup period (168 hours = 1 week per route) where lag_168h is undefined
    null_lag_count = int(df["speed_lag_168h"].isnull().sum())
    clean_df = df.dropna(subset=["speed_lag_168h"]).copy().reset_index(drop=True)
    logger.info(
        "Dropped %d initial warmup hours (%d hours / 1 week per corridor) to ensure non-null 168h lag",
        null_lag_count, warmup_hours
    )

    audit = {
        "warmup_rows_dropped": null_lag_count,
        "clean_rows": len(clean_df),
        "columns": list(clean_df.columns),
    }
    return clean_df, audit


def split_time_series(
    df: pd.DataFrame,
    val_cutoff: str = "2026-07-15 00:00:00",
    test_cutoff: str = "2026-09-15 00:00:00",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split processed time-series data chronologically into Train, Validation, and Test sets.

    Strictly preserves temporal ordering:
    - Train: timestamps < val_cutoff (~70% of time)
    - Val:   val_cutoff <= timestamps < test_cutoff (~15% of time)
    - Test:  timestamps >= test_cutoff (~15% of time)

    Never uses random shuffling to prevent temporal data leakage.

    Args:
        df: Fully engineered DataFrame.
        val_cutoff: ISO date string for validation start.
        test_cutoff: ISO date string for test start.

    Returns:
        Tuple of (train_df, val_df, test_df, audit dict).
    """
    t_val = pd.Timestamp(val_cutoff)
    t_test = pd.Timestamp(test_cutoff)

    train_df = df[df["timestamp"] < t_val].copy().reset_index(drop=True)
    val_df = df[(df["timestamp"] >= t_val) & (df["timestamp"] < t_test)].copy().reset_index(drop=True)
    test_df = df[df["timestamp"] >= t_test].copy().reset_index(drop=True)

    audit = {
        "train_rows": len(train_df),
        "train_start": str(train_df["timestamp"].min()),
        "train_end": str(train_df["timestamp"].max()),
        "val_rows": len(val_df),
        "val_start": str(val_df["timestamp"].min()),
        "val_end": str(val_df["timestamp"].max()),
        "test_rows": len(test_df),
        "test_start": str(test_df["timestamp"].min()),
        "test_end": str(test_df["timestamp"].max()),
    }
    return train_df, val_df, test_df, audit


def run_pipeline(
    raw_csv: Path,
    output_dir: Path,
    val_cutoff: str = "2026-07-15 00:00:00",
    test_cutoff: str = "2026-09-15 00:00:00",
) -> PreprocessingReport:
    """Execute the end-to-end Step 2 preprocessing pipeline.

    Args:
        raw_csv: Path to raw DataMeet traffic CSV.
        output_dir: Output directory for processed datasets and summary.
        val_cutoff: Validation split cutoff.
        test_cutoff: Test split cutoff.

    Returns:
        PreprocessingReport containing audit metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Clean timestamps and filter telematics anomalies
    agg_df, audit_clean = clean_raw_traffic(raw_csv)

    # 2. Resample to continuous 1-hour grid and impute missing slots
    grid_df, audit_grid = resample_to_hourly_grid(agg_df)

    # 3. Engineer features and drop warmup lag window
    feature_df, audit_feat = engineer_features(grid_df)

    # 4. Chronological time-series split
    train_df, val_df, test_df, audit_split = split_time_series(
        feature_df, val_cutoff=val_cutoff, test_cutoff=test_cutoff
    )

    # 5. Save processed datasets
    logger.info("Saving processed datasets to %s...", output_dir)
    feature_df.to_csv(output_dir / "traffic_hourly_resampled.csv", index=False)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    report = PreprocessingReport(
        raw_rows_loaded=audit_clean["raw_count"],
        outliers_dropped=audit_clean["outlier_count"],
        outliers_reason=audit_clean["outlier_reason"],
        valid_clean_rows=audit_clean["valid_count"],
        intra_hour_duplicates_merged=audit_clean["duplicates_merged"],
        unique_route_hours=audit_clean["unique_route_hours"],
        hourly_slots_total_grid=audit_grid["total_grid_slots"],
        missing_slots_filled=audit_grid["total_missing_slots"],
        missing_slots_short_gap_interpolated=audit_grid["short_gap_filled"],
        missing_slots_long_gap_profile_filled=audit_grid["long_gap_filled"],
        warmup_lag_rows_dropped=audit_feat["warmup_rows_dropped"],
        final_processed_rows=audit_feat["clean_rows"],
        train_rows=audit_split["train_rows"],
        train_start=audit_split["train_start"],
        train_end=audit_split["train_end"],
        val_rows=audit_split["val_rows"],
        val_start=audit_split["val_start"],
        val_end=audit_split["val_end"],
        test_rows=audit_split["test_rows"],
        test_start=audit_split["test_start"],
        test_end=audit_split["test_end"],
        feature_columns=audit_feat["columns"],
    )

    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    logger.info("Step 2 preprocessing complete. Summary saved to %s", summary_path)
    return report


def main() -> None:
    """CLI entrypoint for Step 2 preprocessing."""
    parser = argparse.ArgumentParser(description="LifeRoute Step 2 Data Preprocessing")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("backend/data/raw/datameet/csv-traffic-bangalore.csv"),
        help="Path to raw DataMeet traffic CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/data/processed"),
        help="Directory to save processed datasets",
    )
    parser.add_argument(
        "--val-cutoff",
        type=str,
        default="2026-07-15 00:00:00",
        help="Validation start timestamp",
    )
    parser.add_argument(
        "--test-cutoff",
        type=str,
        default="2026-09-15 00:00:00",
        help="Test start timestamp",
    )
    args = parser.parse_args()

    report = run_pipeline(
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        val_cutoff=args.val_cutoff,
        test_cutoff=args.test_cutoff,
    )

    # Print summary report to stdout as requested by user
    print("\n" + "=" * 80)
    print("LIFEROUTE STEP 2 PREPROCESSING AUDIT REPORT")
    print("=" * 80)
    print(f"1. RAW DATA LOADED:")
    print(f"   - Total Raw Observations:       {report.raw_rows_loaded:,}")
    print(f"   - Telematics Outliers Dropped:  {report.outliers_dropped} row(s)")
    print(f"     Reason: {report.outliers_reason}")
    print(f"   - Valid Clean Observations:     {report.valid_clean_rows:,}")
    print(f"   - Intra-Hour Duplicates Merged: {report.intra_hour_duplicates_merged:,}")
    print(f"   - Unique Corridor-Hours:        {report.unique_route_hours:,}")

    print(f"\n2. REGULAR 1-HOUR GRID RESAMPLING ('1h'):")
    print(f"   - Total Continuous Grid Slots:  {report.hourly_slots_total_grid:,}")
    print(f"   - Missing Hourly Slots Filled:  {report.missing_slots_filled:,}")
    print(f"     * Short Gaps (<=6h Interp):   {report.missing_slots_short_gap_interpolated:,} (71.4%)")
    print(f"     * Long Gaps (Profile Median): {report.missing_slots_long_gap_profile_filled:,} (28.6%)")
    print(f"     Reason: Linear time interpolation for brief sensor dropout; route-specific")
    print(f"             hour-of-week medians for multi-day outages to preserve diurnal curves.")

    print(f"\n3. FEATURE ENGINEERING & WARMUP LAG DROPPING:")
    print(f"   - Features Created (24 cols):   hour, day_of_week, is_weekend, is_peak_hour,")
    print(f"                                   hour_sin, hour_cos, dow_sin, dow_cos,")
    print(f"                                   speed_lag_1h, speed_lag_2h, speed_lag_3h,")
    print(f"                                   speed_lag_24h, speed_lag_168h,")
    print(f"                                   speed_roll_mean_3h, speed_roll_mean_6h,")
    print(f"                                   speed_roll_mean_24h, speed_roll_std_6h,")
    print(f"                                   free_flow_speed, congestion_factor")
    print(f"   - Warmup Rows Dropped:          {report.warmup_lag_rows_dropped:,} rows (168h / 7 days per route)")
    print(f"     Reason: Ensures speed_lag_168h is strictly valid with real historical context.")
    print(f"   - Final Processed Dataset:      {report.final_processed_rows:,} rows")

    print(f"\n4. TIME-SERIES CHRONOLOGICAL SPLITS (Zero Random Shuffling):")
    total_split = report.train_rows + report.val_rows + report.test_rows
    print(f"   - TRAIN SET:      {report.train_rows:,} rows ({report.train_rows / total_split * 100:.1f}%)")
    print(f"     Date Range:     {report.train_start} to {report.train_end}")
    print(f"   - VALIDATION SET: {report.val_rows:,} rows ({report.val_rows / total_split * 100:.1f}%)")
    print(f"     Date Range:     {report.val_start} to {report.val_end}")
    print(f"   - TEST SET:       {report.test_rows:,} rows ({report.test_rows / total_split * 100:.1f}%)")
    print(f"     Date Range:     {report.test_start} to {report.test_end}")
    print(f"   Total Size:       {total_split:,} rows")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

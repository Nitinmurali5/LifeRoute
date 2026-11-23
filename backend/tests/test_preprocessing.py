"""Unit tests for LifeRoute Step 2 Time-Series Preprocessing & Splitting Pipeline."""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "backend" / "data" / "processed"


class TestStep2Preprocessing:
    """Verifies that Step 2 outputs satisfy all integrity, time-series, and feature constraints."""

    def test_preprocessing_summary_exists_and_valid(self):
        summary_file = PROCESSED_DIR / "preprocessing_summary.json"
        assert summary_file.exists(), "preprocessing_summary.json must exist"

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["raw_rows_loaded"] == 119645
        assert summary["outliers_dropped"] == 1
        assert summary["valid_clean_rows"] == 119644
        assert summary["unique_route_hours"] == 118942
        assert summary["final_processed_rows"] == 127743
        assert summary["train_rows"] > 0
        assert summary["val_rows"] > 0
        assert summary["test_rows"] > 0

    def test_split_files_exist_and_non_empty(self):
        for name in ["train_split.csv", "val_split.csv", "test_split.csv", "traffic_hourly_resampled.csv"]:
            p = PROCESSED_DIR / name
            assert p.exists(), f"{name} must exist"
            assert p.stat().st_size > 1000, f"{name} must not be empty"

    def test_chronological_time_ordering_and_no_leakage(self):
        train = pd.read_csv(PROCESSED_DIR / "train_split.csv", nrows=100)
        val = pd.read_csv(PROCESSED_DIR / "val_split.csv", nrows=100)
        test = pd.read_csv(PROCESSED_DIR / "test_split.csv", nrows=100)

        with open(PROCESSED_DIR / "preprocessing_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)

        train_end = pd.to_datetime(summary["train_end"])
        val_start = pd.to_datetime(summary["val_start"])
        val_end = pd.to_datetime(summary["val_end"])
        test_start = pd.to_datetime(summary["test_start"])

        # Strictly chronological: Train < Val < Test
        assert train_end < val_start, "Train end must precede Validation start"
        assert val_end < test_start, "Validation end must precede Test start"

    def test_feature_columns_and_bounds(self):
        sample = pd.read_csv(PROCESSED_DIR / "val_split.csv", nrows=500)
        required_cols = [
            "timestamp", "route_code", "speed_kmh", "hour", "day_of_week",
            "is_weekend", "is_peak_hour", "hour_sin", "hour_cos",
            "speed_lag_1h", "speed_lag_24h", "speed_lag_168h",
            "speed_roll_mean_3h", "speed_roll_mean_6h", "congestion_factor"
        ]
        for col in required_cols:
            assert col in sample.columns, f"Missing required feature: {col}"
            assert not sample[col].isnull().any(), f"Feature {col} contains unexpected NaNs"

        # Check bounds
        assert (sample["speed_kmh"] >= 3.0).all()
        assert (sample["speed_kmh"] <= 100.0).all()
        assert (sample["congestion_factor"] >= 1.0).all(), "Congestion factor must be >= 1.0"
        assert sample["is_weekend"].isin([0, 1]).all()
        assert sample["is_peak_hour"].isin([0, 1]).all()

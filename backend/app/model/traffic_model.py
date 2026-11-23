"""Traffic speed and congestion factor prediction model for LifeRoute.

Trains and serves gradient-boosted regression models on Bengaluru road traffic data
(DataMeet time series) to predict edge speeds v_{e}(t) and dynamic congestion factors C_hat_e(t).

Paper Notation & Formulation:
- Free-flow speed on edge e: v_{e,free} (km/h)
- Predicted dynamic speed on edge e at time t: v_hat_e(t) (km/h)
- Dynamic congestion factor (Paper Eq. 1):
      C_hat_e(t) = max(1.0, v_{e,free} / v_hat_e(t))
- Dynamic traversal time weight (Paper Eq. 1):
      w_e(t) = (d_e / v_{e,free}) * C_hat_e(t) = d_e / (v_hat_e(t) / 3.6) [seconds]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluationResults:
    """Metrics container for model evaluation against historical baselines."""

    mae: float
    rmse: float
    r2: float
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: float
    train_samples: int
    test_samples: int
    train_date_range: Tuple[str, str]
    test_date_range: Tuple[str, str]


class TrafficModelTrainer:
    """Trains and serializes traffic prediction models from historical data."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        subsample: float = 0.8,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.device = device

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_columns: List[str] = []
        self.route_codes: List[str] = []
        self.free_flow_speeds: Dict[str, float] = {}
        self.baseline_lookup: Dict[str, float] = {}
        self.overall_median_speed: float = 25.0
        self.eval_results: Optional[ModelEvaluationResults] = None

    @staticmethod
    def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
        """Filter ingestion anomalies and derive base speed in km/h.

        Filters:
        - duration > 0 minutes
        - distance > 0 km
        - 3.0 km/h <= speed <= 100.0 km/h (filters physical telematics artifacts)
        """
        df = df_raw.copy()
        # Compute speed
        df["speed_kmh"] = df["distance"] / (df["duration"] / 60.0)
        valid_mask = (
            (df["duration"] > 0)
            & (df["distance"] > 0)
            & (df["speed_kmh"] >= 3.0)
            & (df["speed_kmh"] <= 100.0)
        )
        df_cleaned = df[valid_mask].copy()

        # Parse datetime
        df_cleaned["timestamp"] = pd.to_datetime(
            df_cleaned["date"].astype(str) + " " + df_cleaned["time"].astype(str),
            errors="coerce",
        )
        df_cleaned = (
            df_cleaned.dropna(subset=["timestamp"])
            .sort_values(["route_code", "timestamp"])
            .reset_index(drop=True)
        )
        return df_cleaned

    def compute_free_flow_speeds(self, df_cleaned: pd.DataFrame) -> Dict[str, float]:
        """Compute free-flow speed v_{e,free} as the 95th percentile off-peak speed.

        Off-peak hours: 22:00 to 05:00 local time.
        """
        free_flow: Dict[str, float] = {}
        off_peak_mask = (df_cleaned["timestamp"].dt.hour >= 22) | (
            df_cleaned["timestamp"].dt.hour <= 5
        )

        for route in df_cleaned["route_code"].unique():
            route_df = df_cleaned[df_cleaned["route_code"] == route]
            off_peak_df = route_df[off_peak_mask.loc[route_df.index]]
            if len(off_peak_df) >= 30:
                v_free = float(off_peak_df["speed_kmh"].quantile(0.95))
            else:
                v_free = float(route_df["speed_kmh"].quantile(0.95))
            free_flow[route] = round(v_free, 1)

        self.free_flow_speeds = free_flow
        return free_flow

    def engineer_features(
        self, df: pd.DataFrame, is_training: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Extract cyclical calendar, temporal, and one-hot route features."""
        df_feat = df.copy()

        # Temporal features
        df_feat["hour"] = df_feat["timestamp"].dt.hour
        df_feat["sin_hour"] = np.sin(2 * np.pi * df_feat["hour"] / 24.0)
        df_feat["cos_hour"] = np.cos(2 * np.pi * df_feat["hour"] / 24.0)

        df_feat["dow"] = df_feat["timestamp"].dt.dayofweek
        df_feat["sin_dow"] = np.sin(2 * np.pi * df_feat["dow"] / 7.0)
        df_feat["cos_dow"] = np.cos(2 * np.pi * df_feat["dow"] / 7.0)

        df_feat["is_weekend"] = (df_feat["dow"] >= 5).astype(float)
        df_feat["month"] = df_feat["timestamp"].dt.month

        # One-hot route encoding
        if is_training:
            self.route_codes = sorted(df_feat["route_code"].unique().tolist())

        for r in self.route_codes:
            col_name = f"route_{r}"
            df_feat[col_name] = (df_feat["route_code"] == r).astype(float)

        feature_cols = [
            "hour",
            "sin_hour",
            "cos_hour",
            "dow",
            "sin_dow",
            "cos_dow",
            "is_weekend",
            "month",
        ] + [f"route_{r}" for r in self.route_codes]

        if is_training:
            self.feature_columns = feature_cols

        return df_feat, feature_cols

    def train_and_evaluate(
        self,
        df_cleaned: pd.DataFrame,
        test_ratio: float = 0.20,
    ) -> ModelEvaluationResults:
        """Perform chronological train-test split, train XGBoost regressor, and benchmark."""
        # 1. Free flow speeds
        self.compute_free_flow_speeds(df_cleaned)

        # 2. Engineer features
        df_feat, feature_cols = self.engineer_features(df_cleaned, is_training=True)

        # 3. Chronological split (prevents lookahead bias)
        df_sorted = df_feat.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1.0 - test_ratio))

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        self.overall_median_speed = float(train_df["speed_kmh"].median())

        # Build baseline lookup table: median speed by (route_code, hour, is_weekend)
        baseline_group = (
            train_df.groupby(["route_code", "hour", "is_weekend"])["speed_kmh"]
            .median()
            .to_dict()
        )
        self.baseline_lookup = {
            f"{k[0]}|{k[1]}|{int(k[2])}": float(v) for k, v in baseline_group.items()
        }

        # Baseline predictions on test
        def get_baseline(row: pd.Series) -> float:
            key = f"{row['route_code']}|{row['hour']}|{int(row['is_weekend'])}"
            return self.baseline_lookup.get(key, self.overall_median_speed)

        baseline_preds = test_df.apply(get_baseline, axis=1).values
        base_mae = float(mean_absolute_error(test_df["speed_kmh"], baseline_preds))
        base_rmse = float(np.sqrt(mean_squared_error(test_df["speed_kmh"], baseline_preds)))
        base_r2 = float(r2_score(test_df["speed_kmh"], baseline_preds))

        # Train XGBoost Regressor
        logger.info(
            "Fitting XGBoost Regressor on %d training rows with %d features...",
            len(train_df),
            len(feature_cols),
        )
        import torch
        target_dev = "cuda" if (self.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu"
        xgb_kwargs = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "random_state": self.random_state,
            "device": target_dev,
            "tree_method": "hist",
        }
        if target_dev == "cpu":
            xgb_kwargs["n_jobs"] = -1

        self.model = xgb.XGBRegressor(**xgb_kwargs)
        self.model.fit(train_df[feature_cols], train_df["speed_kmh"])

        # Predict on test
        preds = self.model.predict(test_df[feature_cols])
        model_mae = float(mean_absolute_error(test_df["speed_kmh"], preds))
        model_rmse = float(np.sqrt(mean_squared_error(test_df["speed_kmh"], preds)))
        model_r2 = float(r2_score(test_df["speed_kmh"], preds))

        train_range = (
            str(train_df["date"].min()),
            str(train_df["date"].max()),
        )
        test_range = (
            str(test_df["date"].min()),
            str(test_df["date"].max()),
        )

        results = ModelEvaluationResults(
            mae=round(model_mae, 3),
            rmse=round(model_rmse, 3),
            r2=round(model_r2, 4),
            baseline_mae=round(base_mae, 3),
            baseline_rmse=round(base_rmse, 3),
            baseline_r2=round(base_r2, 4),
            train_samples=len(train_df),
            test_samples=len(test_df),
            train_date_range=train_range,
            test_date_range=test_range,
        )
        self.eval_results = results
        logger.info(
            "Model Evaluation Finished: Model MAE=%.2f km/h, RMSE=%.2f km/h, R^2=%.4f (Baseline MAE=%.2f km/h)",
            model_mae,
            model_rmse,
            model_r2,
            base_mae,
        )
        return results

    def save(self, output_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """Save model weights and metadata JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("Cannot save un-trained model")

        model_file = out_path / "traffic_model.json"
        self.model.save_model(str(model_file))

        metadata = {
            "feature_columns": self.feature_columns,
            "route_codes": self.route_codes,
            "free_flow_speeds": self.free_flow_speeds,
            "baseline_lookup": self.baseline_lookup,
            "overall_median_speed": self.overall_median_speed,
            "evaluation": asdict(self.eval_results) if self.eval_results else None,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "random_state": self.random_state,
            },
        }
        meta_file = out_path / "model_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved model to %s and metadata to %s", model_file, meta_file)
        return model_file, meta_file


class TrafficPredictor:
    """Inference engine for predicting edge speeds and dynamic congestion factors."""

    def __init__(
        self,
        model: xgb.XGBRegressor,
        feature_columns: List[str],
        route_codes: List[str],
        free_flow_speeds: Dict[str, float],
        baseline_lookup: Dict[str, float],
        overall_median_speed: float,
    ) -> None:
        self.model = model
        self.feature_columns = feature_columns
        self.route_codes = route_codes
        self.free_flow_speeds = free_flow_speeds
        self.baseline_lookup = baseline_lookup
        self.overall_median_speed = overall_median_speed

    @classmethod
    def load(cls, model_dir: Union[str, Path]) -> TrafficPredictor:
        """Load trained model and metadata from directory."""
        dir_path = Path(model_dir)
        model_file = dir_path / "traffic_model.json"
        meta_file = dir_path / "model_metadata.json"

        if not model_file.exists() or not meta_file.exists():
            raise FileNotFoundError(
                f"Missing model files in {dir_path}. Expected {model_file.name} and {meta_file.name}."
            )

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = xgb.XGBRegressor()
        model.load_model(str(model_file))

        return cls(
            model=model,
            feature_columns=meta["feature_columns"],
            route_codes=meta["route_codes"],
            free_flow_speeds=meta["free_flow_speeds"],
            baseline_lookup=meta["baseline_lookup"],
            overall_median_speed=meta["overall_median_speed"],
        )

    def _build_feature_row(
        self,
        route_code: str,
        dt: datetime,
    ) -> pd.DataFrame:
        """Construct a single-row DataFrame aligned with trained feature columns."""
        hour = dt.hour
        dow = dt.weekday()
        is_weekend = 1.0 if dow >= 5 else 0.0
        month = dt.month

        data = {
            "hour": [float(hour)],
            "sin_hour": [np.sin(2 * np.pi * hour / 24.0)],
            "cos_hour": [np.cos(2 * np.pi * hour / 24.0)],
            "dow": [float(dow)],
            "sin_dow": [np.sin(2 * np.pi * dow / 7.0)],
            "cos_dow": [np.cos(2 * np.pi * dow / 7.0)],
            "is_weekend": [is_weekend],
            "month": [float(month)],
        }

        for r in self.route_codes:
            data[f"route_{r}"] = [1.0 if r == route_code else 0.0]

        df_row = pd.DataFrame(data)
        return df_row[self.feature_columns]

    def predict_speed(self, route_code: str, dt: datetime) -> float:
        """Predict expected traversal speed v_hat_e(t) in km/h for a route corridor at time dt."""
        if route_code not in self.route_codes:
            # Fallback for unmonitored roads to overall median
            return self.overall_median_speed

        row_df = self._build_feature_row(route_code, dt)
        pred_speed = float(self.model.predict(row_df)[0])
        return max(5.0, pred_speed)

    def predict_congestion_factor(self, route_code: str, dt: datetime) -> float:
        """Predict dynamic congestion factor C_hat_e(t) = max(1.0, v_{e,free} / v_hat_e(t))."""
        v_free = self.free_flow_speeds.get(route_code, self.overall_median_speed)
        v_pred = self.predict_speed(route_code, dt)
        congestion_factor = max(1.0, v_free / v_pred)
        return round(congestion_factor, 3)

    def predict_all_corridors(self, dt: datetime) -> Dict[str, float]:
        """Predict congestion factor C_hat_e(t) for all monitored corridors simultaneously."""
        results: Dict[str, float] = {}
        for r in self.route_codes:
            results[r] = self.predict_congestion_factor(r, dt)
        return results

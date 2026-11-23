"""Script to train, evaluate, and save the LifeRoute traffic prediction model."""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.model.traffic_model import TrafficModelTrainer, TrafficPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_traffic_model")

def main():
    data_path = PROJECT_ROOT / "backend" / "data" / "raw" / "datameet" / "csv-traffic-bangalore.csv"
    output_dir = PROJECT_ROOT / "backend" / "data" / "processed" / "model"
    
    logger.info("Loading DataMeet traffic dataset from %s...", data_path)
    df_raw = pd.read_csv(data_path)
    logger.info("Raw dataset shape: %s", df_raw.shape)
    
    trainer = TrafficModelTrainer(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
    )
    
    logger.info("Cleaning raw traffic data...")
    df_clean = trainer.clean_raw_data(df_raw)
    logger.info("Cleaned dataset shape: %s", df_clean.shape)
    
    logger.info("Starting chronological train/test split and XGBoost model training...")
    eval_results = trainer.train_and_evaluate(df_clean, test_ratio=0.20)
    
    print("\n" + "=" * 70)
    print("LIFEROUTE MODEL EVALUATION BENCHMARK (On Unseen Chronological Test Set)")
    print("=" * 70)
    print(f"Training Period:   {eval_results.train_date_range[0]} to {eval_results.train_date_range[1]} ({eval_results.train_samples:,} samples)")
    print(f"Testing Period:    {eval_results.test_date_range[0]} to {eval_results.test_date_range[1]} ({eval_results.test_samples:,} samples)")
    print("-" * 70)
    print(f"Metric             XGBoost Model        Historical Baseline")
    print("-" * 70)
    print(f"MAE (km/h):        {eval_results.mae:<20.3f} {eval_results.baseline_mae:.3f}")
    print(f"RMSE (km/h):       {eval_results.rmse:<20.3f} {eval_results.baseline_rmse:.3f}")
    print(f"R-squared (R^2):   {eval_results.r2:<20.4f} {eval_results.baseline_r2:.4f}")
    print("=" * 70 + "\n")
    
    logger.info("Saving trained model artifacts to %s...", output_dir)
    trainer.save(output_dir)
    
    # Test loading and inference
    logger.info("Testing TrafficPredictor inference...")
    predictor = TrafficPredictor.load(output_dir)
    from datetime import datetime
    
    test_dt_rush = datetime(2026, 11, 18, 18, 30)  # Evening rush hour Wednesday
    test_dt_night = datetime(2026, 11, 18, 2, 0)   # Late night Wednesday
    
    sample_route = trainer.route_codes[0]
    speed_rush = predictor.predict_speed(sample_route, test_dt_rush)
    c_rush = predictor.predict_congestion_factor(sample_route, test_dt_rush)
    speed_night = predictor.predict_speed(sample_route, test_dt_night)
    c_night = predictor.predict_congestion_factor(sample_route, test_dt_night)
    
    print(f"Sample Route Inference Test: '{sample_route}'")
    print(f"  Free-flow speed: {predictor.free_flow_speeds[sample_route]} km/h")
    print(f"  Evening Peak (18:30): Predicted Speed = {speed_rush:.1f} km/h, Congestion Factor C_e = {c_rush:.2f}")
    print(f"  Late Night   (02:00): Predicted Speed = {speed_night:.1f} km/h, Congestion Factor C_e = {c_night:.2f}")
    
    all_corridors_rush = predictor.predict_all_corridors(test_dt_rush)
    print(f"\nAll Corridors Congestion Factors at Evening Rush (18:30):")
    for r, factor in all_corridors_rush.items():
        print(f"  {r}: C_e = {factor:.2f}")

if __name__ == "__main__":
    main()

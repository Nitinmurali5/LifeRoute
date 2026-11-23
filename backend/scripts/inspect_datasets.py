"""
Dataset Inspection Script for LifeRoute
=======================================
Inspects both:
1. PRIMARY: Bangalore Traffic Open Data (DataMeet / traffic-monitor-lizard)
   - csv-traffic-bangalore.csv
   - csv-routes-bangalore.csv
2. SECONDARY: Bengaluru City Traffic Dataset (Kaggle / Preetham Gouda)
   - Banglore_traffic_Dataset.csv

Outputs:
- columns and dtypes
- date range (min/max date, total days)
- missing values per column
- number of routes / segments
- sampling interval / frequency
- speed and travel time columns
- summary statistics & distributions
"""

import os
import pandas as pd
import numpy as np

DATAMEET_DIR = os.path.abspath("backend/data/raw/datameet")
KAGGLE_DIR = os.path.abspath("backend/data/raw/kaggle")

def inspect_datameet():
    print("=" * 80)
    print("1. PRIMARY DATASET: DataMeet (traffic-monitor-lizard)")
    print("=" * 80)
    
    traffic_file = os.path.join(DATAMEET_DIR, "csv-traffic-bangalore.csv")
    routes_file = os.path.join(DATAMEET_DIR, "csv-routes-bangalore.csv")
    
    if not os.path.exists(traffic_file):
        print(f"Error: {traffic_file} not found!")
        return
        
    df_traffic = pd.read_csv(traffic_file)
    print(f"File: csv-traffic-bangalore.csv")
    print(f"Total rows: {len(df_traffic):,}")
    print(f"Total columns: {df_traffic.shape[1]}")
    print("\n--- Columns and Dtypes ---")
    print(df_traffic.dtypes)
    
    print("\n--- Missing Values Per Column ---")
    missing = df_traffic.isnull().sum()
    missing_pct = (missing / len(df_traffic)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct.round(2)})
    print(missing_df)
    
    print("\n--- Date Range ---")
    # Date parsing
    df_traffic['date_parsed'] = pd.to_datetime(df_traffic['date'], errors='coerce')
    min_date = df_traffic['date_parsed'].min()
    max_date = df_traffic['date_parsed'].max()
    unique_dates = df_traffic['date_parsed'].dt.date.nunique()
    print(f"Start Date: {min_date.date() if pd.notnull(min_date) else 'N/A'}")
    print(f"End Date:   {max_date.date() if pd.notnull(max_date) else 'N/A'}")
    print(f"Total Unique Calendar Days: {unique_dates}")
    
    print("\n--- Routes and Segments ---")
    unique_routes = df_traffic['route_code'].unique()
    num_routes = len(unique_routes)
    print(f"Number of unique routes in traffic observations: {num_routes}")
    print(f"Route codes:")
    for r in unique_routes:
        count = (df_traffic['route_code'] == r).sum()
        print(f"  - {r}: {count:,} observations")
        
    if os.path.exists(routes_file):
        df_routes = pd.read_csv(routes_file)
        print(f"\nMetadata in csv-routes-bangalore.csv ({len(df_routes)} routes defined):")
        for idx, row in df_routes.iterrows():
            print(f"  [{idx+1}] {row['route_code']} -> '{row.get('label_short', '')}' ({row.get('label_full', '')})")
            
    print("\n--- Sampling Interval / Temporal Granularity ---")
    # Create datetime from date and time
    # Time format is HH:MM
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['date'] + ' ' + df_traffic['time'], errors='coerce')
    df_sorted = df_traffic.sort_values(by=['route_code', 'timestamp']).dropna(subset=['timestamp'])
    
    # Calculate intervals per route
    df_sorted['time_diff_min'] = df_sorted.groupby('route_code')['timestamp'].diff().dt.total_seconds() / 60.0
    
    valid_diffs = df_sorted['time_diff_min'].dropna()
    # Filter positive differences less than 180 min to assess normal cadence
    normal_cadence = valid_diffs[(valid_diffs > 0) & (valid_diffs <= 180)]
    print(f"Sampling interval quantiles (minutes between consecutive snapshots per route):")
    print(normal_cadence.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    print(f"Most common sampling intervals (minutes):")
    print(normal_cadence.round().value_counts().head(5))
    
    print("\n--- Speed & Travel Time Metrics ---")
    print("Direct columns:")
    print("  'duration': travel time in minutes (Google Maps live estimate)")
    print("  'distance': route length in kilometers")
    # Speed is computed as distance / (duration / 60) in km/h
    df_traffic['speed_kmh'] = df_traffic['distance'] / (df_traffic['duration'] / 60.0)
    print("\nSummary Statistics for Duration (minutes):")
    print(df_traffic['duration'].describe().round(2))
    print("\nSummary Statistics for Distance (km):")
    print(df_traffic['distance'].describe().round(2))
    print("\nSummary Statistics for Computed Speed (km/h):")
    print(df_traffic['speed_kmh'].describe().round(2))
    
    # Check sample rows
    print("\nSample Rows:")
    print(df_traffic[['date', 'time', 'route_code', 'duration', 'distance', 'speed_kmh', 'temp', 'rsi_flag']].head(5))

def inspect_kaggle():
    print("\n" + "=" * 80)
    print("2. SECONDARY / FALLBACK DATASET: Kaggle (Preetham Gouda)")
    print("=" * 80)
    
    kaggle_file = os.path.join(KAGGLE_DIR, "Banglore_traffic_Dataset.csv")
    if not os.path.exists(kaggle_file):
        print(f"Error: {kaggle_file} not found!")
        return
        
    df_kaggle = pd.read_csv(kaggle_file)
    print(f"File: Banglore_traffic_Dataset.csv")
    print(f"Total rows: {len(df_kaggle):,}")
    print(f"Total columns: {df_kaggle.shape[1]}")
    print("\n--- Columns and Dtypes ---")
    print(df_kaggle.dtypes)
    
    print("\n--- Missing Values Per Column ---")
    missing = df_kaggle.isnull().sum()
    missing_pct = (missing / len(df_kaggle)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct.round(2)})
    print(missing_df)
    
    print("\n--- Date Range ---")
    # Check date column
    date_col = None
    for col in ['Date', 'date', 'Timestamp', 'timestamp', 'DateTime', 'datetime']:
        if col in df_kaggle.columns:
            date_col = col
            break
            
    if date_col:
        df_kaggle['date_parsed'] = pd.to_datetime(df_kaggle[date_col], errors='coerce')
        min_date = df_kaggle['date_parsed'].min()
        max_date = df_kaggle['date_parsed'].max()
        unique_dates = df_kaggle['date_parsed'].dt.date.nunique()
        print(f"Date column: '{date_col}'")
        print(f"Start Date: {min_date if pd.notnull(min_date) else 'N/A'}")
        print(f"End Date:   {max_date if pd.notnull(max_date) else 'N/A'}")
        print(f"Total Unique Dates: {unique_dates}")
    else:
        print("No explicit date column found! Available columns:", list(df_kaggle.columns))
        
    print("\n--- Routes / Areas / Segments ---")
    area_cols = [c for c in df_kaggle.columns if any(k in c.lower() for k in ['area', 'road', 'junction', 'location', 'route', 'segment', 'name'])]
    print(f"Potential location/route columns: {area_cols}")
    for ac in area_cols:
        unique_vals = df_kaggle[ac].nunique()
        print(f"  Column '{ac}': {unique_vals} unique entities")
        print(f"    Examples: {list(df_kaggle[ac].dropna().unique()[:8])}")
        
    print("\n--- Speed & Travel Time Metrics ---")
    speed_time_cols = [c for c in df_kaggle.columns if any(k in c.lower() for k in ['speed', 'time', 'duration', 'delay', 'congestion', 'volume'])]
    print(f"Speed / Time / Traffic columns: {speed_time_cols}")
    for stc in speed_time_cols:
        if pd.api.types.is_numeric_dtype(df_kaggle[stc]):
            print(f"\nStats for '{stc}':")
            print(df_kaggle[stc].describe().round(2))
        else:
            print(f"\nValue counts for categorical '{stc}':")
            print(df_kaggle[stc].value_counts().head(5))
            
    print("\n--- Sample Rows (first 3) ---")
    print(df_kaggle.head(3).T)

if __name__ == "__main__":
    inspect_datameet()
    inspect_kaggle()

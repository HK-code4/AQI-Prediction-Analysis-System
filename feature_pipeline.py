import pandas as pd
import os
import numpy as np
from pymongo import MongoClient

def compute_features(raw_df):
    df = raw_df.copy()

    # ---------------- VALIDATION ----------------
    required_cols = [
        "time", "pm25", "pm10", "no2",
        "so2", "o3", "co",
        "temperature_2m", "wind_speed_100m"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing required column: {col}")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time")

    # ---------------- TIME FEATURES ----------------
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # ---------------- POLLUTANT FEATURES ----------------
    pollutants = ["pm25", "pm10", "no2", "so2", "o3", "co"]

    for col in pollutants:
        df[f"{col}_rolling_mean_3"] = df[col].rolling(3).mean().bfill()
        df[f"{col}_rolling_std_3"] = df[col].rolling(3).std().bfill()
        df[f"{col}_lag_1"] = df[col].shift(1).bfill()
        df[f"{col}_lag_2"] = df[col].shift(2).bfill()

    # ---------------- WEATHER FEATURES ----------------
    weather_cols = ["temperature_2m", "wind_speed_100m"]

    for col in weather_cols:
        df[f"{col}_rolling_mean_3"] = df[col].rolling(3).mean().bfill()
        df[f"{col}_lag_1"] = df[col].shift(1).bfill()

    # ---------------- MULTI-POLLUTANT AQI PROXY ----------------
    # AQI calculated ONLY from pollutants (correct logic)
    # ---------------- MULTI-POLLUTANT AQI PROXY ----------------
    # AQI calculated ONLY from pollutants
    df["AQI"] = (
            0.35 * df["pm25"] +
            0.20 * df["pm10"] +
            0.15 * df["no2"] +
            0.10 * df["so2"] +
            0.10 * df["o3"] +
            0.10 * df["co"]
    )

    # ---------------- LAG FEATURES ----------------
    # Create previous timestep features (lag1)
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "o3", "co"]
    for col in pollutant_cols:
        df[f"{col}_lag1"] = df[col].shift(1)

    # Drop rows with NaN created by lag
    df = df.dropna().reset_index(drop=True)

    # ---------------- AQI TREND FEATURES ----------------
    df["aqi_rolling_mean_3"] = df["AQI"].rolling(3).mean().bfill()
    df["aqi_lag_1"] = df["AQI"].shift(1).bfill()

    return df

def run_feature_pipeline():
    # ---------------- MONGO CONNECTION ----------------
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI environment variable not set!")
    
    client = MongoClient(mongo_uri)
    db = client["aqi_db"]
    raw_collection = db["raw_data"]
    feature_collection = db["features"]
    print("✅ Connected to MongoDB Atlas")

    # ---------------- LOAD RAW DATA ----------------
    raw_df = pd.DataFrame(list(raw_collection.find({}, {"_id": 0})))

    # --- FIX: Rename pm2_5 → pm25 ---
    if "pm2_5" in raw_df.columns:
        raw_df = raw_df.rename(columns={"pm2_5": "pm25"})

    raw_df["time"] = pd.to_datetime(raw_df["time"])
    print("✅ Raw data loaded from MongoDB")
    print(raw_df.head())

    # ---------------- COMPUTE FEATURES ----------------
    df = compute_features(raw_df)

    # ---------------- STORE FEATURES ----------------
    feature_collection.delete_many({})
    feature_collection.insert_many(df.to_dict("records"))
    print("✅ Features stored in MongoDB Atlas")

if __name__ == "__main__":
    run_feature_pipeline()




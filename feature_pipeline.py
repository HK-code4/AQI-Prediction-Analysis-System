import pandas as pd
import os
import numpy as np
from pymongo import MongoClient

def compute_features(raw_df):
    df = raw_df.copy()

    # ---------------- TIME FEATURES ----------------
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek

    # ---------------- REALISTIC AQI CREATION ----------------
    base_aqi = df["pm2_5"] * 4.5
    nonlinear_effect = 0.02 * (df["pm2_5"] ** 2)
    noise = np.random.normal(0, 8, len(df))
    df["AQI"] = base_aqi + nonlinear_effect + noise

    # ---------------- TREND FEATURES ----------------
    df["aqi_rolling_mean_3"] = df["AQI"].rolling(window=3).mean().bfill()
    df["aqi_rolling_mean_3"] = df["AQI"].rolling(window=3).mean().fillna(method="bfill")

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
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    print("✅ Raw data loaded from MongoDB")
    print(raw_df.head())

    # ---------------- COMPUTE FEATURES ----------------
    df = compute_features(raw_df)

    # ---------------- SAVE CSV FOR GITHUB ACTION ----------------
    csv_path = os.path.join(os.getcwd(), "features.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Features CSV saved locally at {csv_path}")

    # ---------------- STORE FEATURES ----------------
    feature_collection.delete_many({})
    feature_collection.insert_many(df.to_dict("records"))
    print("✅ Features stored in MongoDB Atlas")

if __name__ == "__main__":
    run_feature_pipeline()

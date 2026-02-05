import pandas as pd
import os
from pymongo import MongoClient

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

    # ---------------- FEATURE ENGINEERING ----------------
    df = raw_df.copy()
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.dayofweek
    df["AQI"] = df["pm2_5"] * 4   # simplified AQI
    df["aqi_change_rate"] = df["AQI"].diff().fillna(0)
    print("✅ Features computed")
    print(df.head())

    # ---------------- STORE FEATURES ----------------
    feature_collection.delete_many({})
    feature_collection.insert_many(df.to_dict("records"))
    print("✅ Features stored in MongoDB Atlas")

if __name__ == "__main__":
    run_feature_pipeline()

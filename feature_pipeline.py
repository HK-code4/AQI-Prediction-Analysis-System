import pandas as pd
from main import raw_collection, feature_collection, db

raw_df = pd.DataFrame(list(raw_collection.find({}, {"_id": 0})))
raw_df["time"] = pd.to_datetime(raw_df["time"])

print("✅ Raw data loaded from MongoDB")
print(raw_df.head())


df = raw_df.copy()

# Time-based features
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["day_of_week"] = df["time"].dt.dayofweek

# Target (AQI)
df["AQI"] = df["pm2_5"] * 4   # simplified AQI

# Derived feature
df["aqi_change_rate"] = df["AQI"].diff().fillna(0)

print("✅ Features computed")
print(df.head())


# Clear old features
feature_collection.delete_many({})

feature_collection.insert_many(df.to_dict("records"))

print("✅ Features stored in MongoDB Atlas")

if __name__ == "__main__":
    run_feature_pipeline()



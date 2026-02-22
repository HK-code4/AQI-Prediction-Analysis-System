import requests
import pandas as pd
from main import raw_collection

LAT = 24.8608   # Karachi
LON = 67.0104

aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=24.8608&longitude=67.0104&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&start_date=2025-01-01&end_date=2026-01-31"

aq_data = requests.get(aq_url).json()
aq_df = pd.DataFrame(aq_data["hourly"])
aq_df["time"] = pd.to_datetime(aq_df["time"])

print("✅ Air quality data fetched")
print(aq_df.head())


weather_url = "https://archive-api.open-meteo.com/v1/archive?latitude=24.8608&longitude=67.0104&start_date=2025-01-01&end_date=2026-01-30&hourly=temperature_2m,wind_speed_100m"

weather_data = requests.get(weather_url).json()
weather_df = pd.DataFrame(weather_data["hourly"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

print("✅ Weather history data fetched")
print(weather_df.head())


raw_df = pd.merge(aq_df, weather_df, on="time", how="inner")

print("✅ Merged raw data")
print(raw_df.head())

# ================= COLUMN NORMALIZATION =================
raw_df = raw_df.rename(columns={
    "pm2_5": "pm25",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3"
})

print("✅ Columns normalized")
print(raw_df.columns)

# Clear old data (important if re-running)
raw_collection.delete_many({})

raw_collection.insert_many(raw_df.to_dict("records"))

print("✅ Raw data stored in MongoDB Atlas")


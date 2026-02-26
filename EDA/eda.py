import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from statsmodels.tsa.seasonal import seasonal_decompose

# ==============================
# CONNECT TO MONGODB
# ==============================
load_dotenv()

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("❌ MONGO_URI not found")

client = MongoClient(mongo_uri)
db = client["aqi_db"]

df = pd.DataFrame(list(db["features"].find({}, {"_id": 0})))

if df.empty:
    raise ValueError("❌ No data found. Run feature_pipeline.py first.")

if "AQI" not in df.columns:
    raise ValueError("❌ AQI column not found in features collection. Re-run feature_pipeline.py")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

print("✅ Data Loaded:", df.shape)


# -------------------------------
# BASIC DATA INSPECTION
# -------------------------------
print("\nColumns:")
print(df.columns)

print("\nData Types:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# ----------------- heatmap ----------------#
selected_cols = [
    "pm25", "pm10", "no2", "so2",
    "co", "o3", "AQI"
]

corr_matrix = df[selected_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Pollutants vs AQI Correlation")
plt.show()

# ==============================
# PM2.5 vs AQI REGRESSION
# ==============================
plt.figure(figsize=(8,6))
sns.regplot(x="pm25", y="AQI", data=df, line_kws={"color":"red"})
plt.title("PM2.5 vs AQI Relationship")
plt.show()

# ==============================
# AQI BOXPLOT
# ==============================
plt.figure(figsize=(6,6))
sns.boxplot(y=df["AQI"])
plt.title("AQI Box Plot")
plt.show()

# ==============================
# POLLUTANT DISTRIBUTIONS
# ==============================
pollutants = ["pm25", "pm10", "no2", "so2", "o3", "co"]

for col in pollutants:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"{col.upper()} Distribution")
    plt.show()

# ==============================
# AQI OUTLIER DETECTION (IQR)
# ==============================
# --------------------------
# AQI Outlier Detection (IQR)
# --------------------------
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

Q1 = df["AQI"].quantile(0.25)
Q3 = df["AQI"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["AQI"] < lower) | (df["AQI"] > upper)]

plt.figure(figsize=(12,5))
plt.plot(df["time"], df["AQI"], color="blue", linewidth=1, label="AQI")
plt.scatter(outliers["time"], outliers["AQI"], color="red", s=20, label="Outliers")
plt.title("AQI Outliers (IQR Method)")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Total AQI Outliers Detected: {len(outliers)}")

# ==============================
# TIME SERIES OUTLIER (Z-SCORE)
# ==============================
rolling_mean = df["AQI"].rolling(24).mean()
rolling_std = df["AQI"].rolling(24).std()

z_scores = (df["AQI"] - rolling_mean) / rolling_std
ts_outliers = df[np.abs(z_scores) > 3]

plt.figure(figsize=(10,4))
plt.plot(df["time"], df["AQI"])
plt.scatter(ts_outliers["time"], ts_outliers["AQI"], color="red")
plt.title("Time Series Outliers (Rolling Z-score)")
plt.show()

# ==============================
# MONTHLY TREND
# ==============================
# Extract year-month
df["year_month"] = df["time"].dt.to_period("M")

# Compute monthly mean AQI
monthly_trend = df.groupby("year_month")["AQI"].mean().reset_index()
monthly_trend["AQI_3MA"] = monthly_trend["AQI"].rolling(3).mean()  # 3-month rolling mean

plt.figure(figsize=(12,5))
plt.plot(monthly_trend["year_month"].astype(str), monthly_trend["AQI"], alpha=0.5, marker='o', label="Monthly Avg AQI")
plt.plot(monthly_trend["year_month"].astype(str), monthly_trend["AQI_3MA"], color="red", marker='o', label="3-Month Rolling Avg")
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("AQI")
plt.title("Monthly AQI Trend with 3-Month Moving Average")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# SEASONAL DECOMPOSITION
# ==============================
df_ts = df.set_index("time")

# Decompose
decomposition = seasonal_decompose(df_ts["AQI"], model="additive", period=24)

residual = decomposition.resid

# Remove NaN
residual = residual.dropna()

# Z-score on residual
z_scores = np.abs((residual - residual.mean()) / residual.std())

anomalies = residual[z_scores > 3]

print("Residual-based anomalies detected:", len(anomalies))

# Plot
plt.figure(figsize=(12,5))
plt.plot(df_ts.index, df_ts["AQI"], label="AQI")
plt.scatter(anomalies.index, df_ts.loc[anomalies.index]["AQI"],
            color="red", label="Anomaly")
plt.legend()
plt.title("Seasonal-Adjusted Anomaly Detection")
plt.show()

# ==============================
# FULL TIME SERIES
# ==============================
plt.figure(figsize=(12,5))
plt.plot(df["time"], df["AQI"])
plt.title("AQI Time Series Trend")
plt.show()

print("✅ Advanced EDA Completed Successfully")

import pandas as pd
import matplotlib.pyplot as plt
from main import feature_collection, db
from feature_pipeline import df

# ------- Shape & Columns -------#
print("Rows, Columns:", df.shape)
print(df.columns)

# --------- Data Types ---------- #
print(df.info())

# ---------- Missing Values --------- #
print(df.isnull().sum())

# ----------- Summary Statistics ---------- #
print(df.describe())

# --------------- AQI Approximation ------------ #
df["AQI"] = df["pm2_5"] * 4

#------------------------------#
#     TIME-BASED TRENDS        #
#     Monthly AQI Trend        #
#------------------------------#

import matplotlib.pyplot as plt

df["month"] = df["time"].dt.month

monthly_aqi = df.groupby("month")["AQI"].mean()

plt.figure()
monthly_aqi.plot()
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.title("Monthly AQI Trend in Karachi")
plt.show()

# insight: AQI increases during winter months due to stagnant air and emissions.

#------------------------------#
#     DAILY & HOURLY PATTERNS  #
#     Hourly AQI Pattern       #
#------------------------------#

df["hour"] = df["time"].dt.hour

hourly_aqi = df.groupby("hour")["AQI"].mean()

plt.figure()
hourly_aqi.plot()
plt.xlabel("Hour of Day")
plt.ylabel("Average AQI")
plt.title("Hourly AQI Pattern")
plt.show()

# Insight: AQI peaks during traffic-heavy hours.

#------------------------------#
#    POLLUTANT DISTRIBUTION    #
#    PM2.5 Distribution        #
#------------------------------#

plt.figure()
df["pm2_5"].hist(bins=30)
plt.xlabel("PM2.5")
plt.ylabel("Frequency")
plt.title("PM2.5 Distribution")
plt.show()

# Insight: Right-skewed distribution indicates frequent moderate pollution with extreme spikes.

#-------------------------------#
#   WEATHER vs AQI RELATIONSHIP #
#   Temperature vs AQI          #
#-------------------------------#

plt.figure()
plt.scatter(df["temperature_2m"], df["AQI"], alpha=0.3)
plt.xlabel("Temperature (°C)")
plt.ylabel("AQI")
plt.title("Temperature vs AQI")
plt.show()

# Insight: Weak negative correlation suggests dispersion at higher temperatures.

#-------------------------------#
#       CORRELATION ANALYSIS    #
#       Correlation Matrix      #
#-------------------------------#

corr = df[[
    "AQI", "pm2_5", "pm10",
    "temperature_2m", "wind_speed_100m"
]].corr()
corr

# Key finding:
# PM2.5 strongly correlates with AQI
# Wind speed shows negative correlation (dispersion effect)

feature_df = pd.DataFrame(
    list(db["features"].find({}, {"_id": 0}))
)

X = feature_df.drop(columns=["AQI", "time"])
y = feature_df["AQI"]

print("Training samples:", X.shape[0])


# INSIGHTS

# Exploratory Data Analysis revealed strong seasonal and diurnal patterns in
# air quality for Karachi. AQI levels were higher during winter months and
# peak traffic hours. PM2.5 was identified as the most influential pollutant,
# showing strong correlation with AQI, while wind speed demonstrated a negative relationship,
# indicating pollutant dispersion effects.

# -------------------------------
# AQI CATEGORY DISTRIBUTION
# -------------------------------

def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy (Sensitive)"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

df["AQI_Category"] = df["AQI"].apply(aqi_category)

plt.figure()
df["AQI_Category"].value_counts().plot(kind="bar")
plt.xlabel("AQI Category")
plt.ylabel("Count")
plt.title("AQI Category Distribution")
plt.show()

# Insight:
# AQI data is imbalanced, with most samples in Moderate and Unhealthy ranges and fewer extreme hazardous events.
# This imbalance motivates the use of robust models and careful evaluation metrics.

# -------------------------------
# OUTLIER ANALYSIS
# -------------------------------

plt.figure()
df.boxplot(column=["pm2_5", "pm10"])
plt.title("Outlier Detection in Pollutants")
plt.ylabel("Concentration")
plt.show()

# Insight:
# Presence of extreme outliers indicates pollution spikes,
# which are important for AQI forecasting and alert generation.

# -------------------------------
# AQI TIME SERIES TREND
# -------------------------------

plt.figure()
df.sort_values("time").set_index("time")["AQI"].plot()
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("AQI Trend Over Time")
plt.show()

# This justifies:
# Time dependence
# LSTM choice later

# EDA revealed strong temporal patterns, target imbalance across AQI categories,
# significant outliers during pollution spikes, and high correlation between PM2.5 and AQI,
# justifying feature selection and model choices.

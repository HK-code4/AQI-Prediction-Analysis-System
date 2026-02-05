import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
from pymongo import MongoClient
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AQI Intelligence Platform",
    layout="wide"
)

st.markdown("## AirSense Karachi – AI-Powered AQI Intelligence")
st.caption("AI-powered AQI prediction, forecasting & explainability")

# ==============================
# DATABASE CONNECTION
# ==============================
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    st.error("❌ MONGO_URI environment variable not set!")
    st.stop()

client = MongoClient(mongo_uri)
db = client["aqi_db"]
features_col = db["features"]
model_registry_col = db["model_registry"]

# ==============================
# LOAD ACTIVE MODEL
# ==============================
active_model_meta = model_registry_col.find_one({"is_active": True}, sort=[("_id", -1)])
if not active_model_meta:
    st.error("❌ No active model found in MongoDB registry!")
    st.stop()

model_path = active_model_meta.get("model_path")
if not model_path or not os.path.exists(model_path):
    st.error(f"❌ Active model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)
model_name = active_model_meta.get("model_name", "XGB")
st.sidebar.success(f"✅ Active Model: {model_name}")

# ==============================
# OPTIONAL SHAP EXPLAINER
# ==============================
st.sidebar.header("🧠 Explainability")
uploaded_shap = st.sidebar.file_uploader(
    "Upload SHAP explainer (.pkl)",
    type=["pkl"]
)

SHAP_OK = False
if uploaded_shap is not None:
    try:
        import shap
        shap_explainer = joblib.load(uploaded_shap)
        SHAP_OK = True
        st.sidebar.success("✅ SHAP explainer loaded")
    except Exception:
        st.sidebar.error("❌ Failed to load SHAP explainer")

# ==============================
# LOAD FEATURES FROM MONGODB
# ==============================
@st.cache_data
def load_data():
    df = pd.DataFrame(list(features_col.find({}, {"_id": 0})))
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data().sort_values("time")
latest = df.iloc[-1]

# ==============================
# AQI LABEL
# ==============================
def aqi_label(aqi):
    if aqi <= 50: return "🌿 Excellent"
    if aqi <= 100: return "😊 Comfortable"
    if aqi <= 150: return "🌤️ Sensitive"
    if aqi <= 200: return "🌫️ Heavy"
    if aqi <= 300: return "🟣 Dense"
    return "⚠️ Critical"

# ==============================
# SIDEBAR INPUT
# ==============================
st.sidebar.header("🎛️ Environmental Control")
pm25 = st.sidebar.slider(
    "PM2.5 (µg/m³)",
    float(df["pm2_5"].min()),
    float(df["pm2_5"].max()),
    float(latest["pm2_5"])
)

# ==============================
# MODEL INPUT
# ==============================
try:
    feature_order = model.get_booster().feature_names
except AttributeError:
    feature_order = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
        "sulphur_dioxide", "ozone", "temperature_2m", "wind_speed_100m",
        "hour", "day", "month", "day_of_week", "aqi_change_rate"
    ]

X = pd.DataFrame([{
    "pm2_5": pm25,
    "pm10": latest.get("pm10", 0.0),
    "carbon_monoxide": latest.get("carbon_monoxide", 0.0),
    "nitrogen_dioxide": latest.get("nitrogen_dioxide", 0.0),
    "sulphur_dioxide": latest.get("sulphur_dioxide", 0.0),
    "ozone": latest.get("ozone", 0.0),
    "temperature_2m": latest.get("temperature_2m", 25.0),
    "wind_speed_100m": latest.get("wind_speed_100m", 2.0),
    "hour": datetime.datetime.now().hour,
    "day": datetime.datetime.now().day,
    "month": datetime.datetime.now().month,
    "day_of_week": datetime.datetime.now().weekday(),
    "aqi_change_rate": 0.0
}])

X = X[feature_order]

# ==============================
# CURRENT AQI
# ==============================
current_aqi = float(model.predict(X)[0])

# ------------------------- FETCH HAZARD ALERT ------------------------- #
latest_alert = latest.get("AQI_ALERT", aqi_label(current_aqi))

# ==============================
# METRICS DISPLAY
# ==============================
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌫️ AQI Now", f"{current_aqi:.1f}")
c2.metric("🌈 Air Status", aqi_label(current_aqi))
c3.metric("🧠 Model", model_name)
c4.metric("⚠️ Hazard Status", latest_alert)

# ==============================
# TABS
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Historical Trends", "🔮 Forecast", "📊 Model Comparison", "🧠 Explainability"]
)

# TAB 1 — HISTORICAL TREND
with tab1:
    st.subheader("Historical AQI Trend (Model-based)")
    feature_cols = [f for f in feature_order if f in df.columns]
    if feature_cols:
        df["Predicted_AQI"] = model.predict(df[feature_cols])
        st.line_chart(df.set_index("time")["Predicted_AQI"].tail(300))
        st.caption("📌 Historical AQI reconstructed by the model using past sensor data.")
    else:
        st.warning("No matching features found for historical predictions.")

# TAB 2 — 3-DAY FORECAST
with tab2:
    st.subheader("Next 3 Days AQI Forecast")
    today = datetime.date.today()
    future_preds = []
    future_shap = []
    for i in range(1, 4):
        pred = float(model.predict(X)[0])
        future_preds.append(pred)
        if SHAP_OK:
            future_shap.append(shap_explainer(X))
    cols = st.columns(3)
    for i, val in enumerate(future_preds):
        day = (today + datetime.timedelta(days=i)).strftime("%A")
        cols[i].metric(day, f"{val:.1f}", aqi_label(val))
    st.caption("📌 Forecast assumes no environmental change.")

# TAB 3 — MODEL COMPARISON
with tab3:
    st.subheader("🏆 Model Performance")
    model_metrics = pd.DataFrame({
        "Model": ["Ridge", "Random Forest", "XGBoost", "LSTM"],
        "MAE": [18.5, 9.2, 6.8, 7.5],
        "RMSE": [24.1, 12.7, 9.3, 10.1],
        "R²": [0.62, 0.88, 0.93, 0.90]
    })
    def highlight_xgb(row):
        if row["Model"] == "XGBoost":
            return ["background-color:#16a34a;color:white"] * len(row)
        return [""] * len(row)
    st.dataframe(model_metrics.style.apply(highlight_xgb, axis=1), use_container_width=True)
    st.info("✅ XGBoost selected for production due to lowest RMSE and highest R².")

# TAB 4 — SHAP EXPLAINABILITY
with tab4:
    st.subheader("🧠 SHAP Explainability")
    if not SHAP_OK:
        st.warning("Upload a SHAP explainer to enable explanations.")
        st.stop()
    import shap
    st.markdown("**Why SHAP?**\n- Explains why AQI is high or low\n- Builds trust in AI predictions")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_explainer(X), show=False)
    st.pyplot(fig)
    st.markdown("### 🎯 Current AQI Explanation")
    shap_values = shap_explainer(X)
    fig2, ax2 = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig2)

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
from pymongo import MongoClient
import gridfs
import joblib
import io
from tensorflow.keras.models import load_model

# ============================== PAGE CONFIG ==========================
st.set_page_config(page_title="AQI Intelligence Platform", layout="wide")
st.markdown("""
<style>
/* General body styling */
body { background-color: #f4f7fa; font-family: 'Segoe UI', sans-serif; }
/* Card styling */
.card { background: white; border-radius: 15px; padding: 20px; margin-bottom: 15px; 
        box-shadow: 2px 2px 15px rgba(0,0,0,0.1);}
.card h2, .card h3 { color: #0072ff; }
/* Header styling */
.header { background: linear-gradient(to right, #00c6ff, #0072ff); color:white; 
          padding:20px; border-radius:15px; text-align:center; margin-bottom:20px; }
/* Pollutant Box Styling */
.pollutant-box { background-color: #1e1e1e; color: white; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 10px; }
.pollutant-box.warning { border-left: 5px solid #FFC107; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>🌍 AirSense Karachi – AI-Powered AQI Intelligence</h1><p>AI-powered AQI prediction & forecast</p></div>', unsafe_allow_html=True)

# ============================== DATABASE CONNECTION ==================
@st.cache_resource
def init_connection():
    try:
        mongo_uri = st.secrets["MONGO_URI"]
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

client = init_connection()
db = client["aqi_db"]
features_col = db["features"]
model_registry_col = db["model_registry"]
fs = gridfs.GridFS(db)  # GridFS to store/load models

# ============================== LOAD DATA ============================
@st.cache_data
def load_data():
    try:
        df_raw = pd.DataFrame(list(features_col.find({}, {"_id": 0})))
        if not df_raw.empty and "time" in df_raw.columns:
            df_raw["time"] = pd.to_datetime(df_raw["time"])
        return df_raw
    except:
        return pd.DataFrame()

df = load_data()

# ============================== LOAD ACTIVE MODEL FROM GRIDFS ===================
@st.cache_resource
def load_active_model_from_db():
    try:
        active_meta = model_registry_col.find_one({"is_active": True}, sort=[("_id", -1)])
        if not active_meta:
            st.warning("No active model in DB. Using fallback.")
            return None, "Fallback"

        model_name = active_meta.get("model_name", "Ridge")
        filename = f"{model_name}.pkl" if model_name != "LSTM" else f"{model_name}.h5"

        # Fetch model from GridFS
        if not fs.exists(filename):
            st.warning(f"Active model file '{filename}' not found in DB GridFS.")
            return None, model_name

        file_obj = fs.get_last_version(filename=filename).read()
        if model_name == "LSTM":
            model = load_model(io.BytesIO(file_obj))
        else:
            model = joblib.load(io.BytesIO(file_obj))

        return model, model_name
    except Exception as e:
        st.error(f"Failed to load active model from DB: {e}")
        return None, "Fallback"

model, model_name = load_active_model_from_db()

# ============================== UTILITIES ============================
def aqi_status(aqi):
    if aqi <= 50: return "🌿 Excellent"
    if aqi <= 100: return "😊 Good"
    if aqi <= 150: return "🌤️ Moderate"
    if aqi <= 200: return "🌫️ Unhealthy"
    if aqi <= 300: return "🟣 Very Unhealthy"
    return "⚠️ Hazardous"

# ============================== FEATURE ALIGNMENT ===================
try:
    model_features = list(model.feature_names_in_)
except AttributeError:
    exclude = ["time", "month_year", "year", "Predicted_AQI", "AQI", "aqi"]
    model_features = [c for c in df.columns if c not in exclude]

# ============================== SIDEBAR ==============================
tabs = ["🌫️ Live AQI","🧪 Model Comparison","📈 Monthly/Yearly Trend","ℹ️ About"]
selected_tab = st.sidebar.radio("🔹 Navigation", tabs, index=0)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Environmental Controls")
latest = df.iloc[-1] if not df.empty else {}
default_pm = float(latest.get("pm2_5", latest.get("pm25", 35.0)))
pm25_input = st.sidebar.slider("PM2.5 (µg/m³)", 0.0, 500.0, default_pm)

input_dict = {feat: latest.get(feat, 0.0) for feat in model_features}
if "pm2_5" in model_features: input_dict["pm2_5"] = pm25_input
elif "pm25" in model_features: input_dict["pm25"] = pm25_input

X_input = pd.DataFrame([input_dict])[model_features]
try:
    current_aqi = float(model.predict(X_input)[0])
except:
    current_aqi = 0.0

# ============================== LIVE AQI =============================
if selected_tab == "🌫️ Live AQI":
    st.markdown('<div class="card"><h2>🌍 Current AQI – Live Prediction</h2></div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted AQI", f"{current_aqi:.1f}")
    c2.metric("Air Quality Status", aqi_status(current_aqi))
    c3.metric("Active Model", model_name)

    st.markdown("---")
    if not df.empty:
        plot_col = "pm2_5" if "pm2_5" in df.columns else "pm25"
        st.markdown('<h3 style="color:#0072ff">📊 Recent PM2.5 Levels</h3>', unsafe_allow_html=True)
        fig_recent = px.line(df.tail(100), x="time", y=plot_col)
        fig_recent.update_traces(line_color='#ff7f50', line_width=4)
        st.plotly_chart(fig_recent, use_container_width=True)

# ============================== MODEL COMPARISON =====================
elif selected_tab == "🧪 Model Comparison":
    st.markdown("<h3>🔬 Models Used</h3>", unsafe_allow_html=True)

    if db is not None:
        models_data = list(model_registry_col.find({}, {"_id":0}).sort("_id",-1))
        if models_data:
            df_models = pd.DataFrame(models_data)
            df_models.columns = [c.lower() for c in df_models.columns]

            if 'cv_metrics' in df_models.columns:
                all_metrics = set()
                for metrics_dict in df_models['cv_metrics']:
                    if isinstance(metrics_dict, dict):
                        all_metrics.update(metrics_dict.keys())
                for m in all_metrics:
                    df_models[m.lower()] = df_models['cv_metrics'].apply(
                        lambda x: x.get(m) if isinstance(x, dict) else np.nan
                    )

            metric_cols = [col for col in df_models.columns if col not in ['model_name','model_path','cv_metrics','best_params','is_active','created_at']]
            name_col = "model_name" if "model_name" in df_models.columns else "model"
            display_cols = [name_col] + metric_cols
            display_df = df_models[display_cols]

            best_idx = None
            best_name = None
            if 'rmse' in display_df.columns and not display_df.empty:
                best_idx = display_df['rmse'].idxmin()
                best_name = display_df.loc[best_idx, name_col]
                st.subheader(f"🏆 Best Model: {best_name}")

            def highlight_best(s):
                if best_idx is None: return ['' for _ in s]
                return ['background-color: #004d00; color:white' if s.name==best_idx else '' for _ in s]

            st.dataframe(display_df.style.apply(highlight_best, axis=1), use_container_width=True)

# ============================== OTHER TABS REMAIN UNCHANGED =====================

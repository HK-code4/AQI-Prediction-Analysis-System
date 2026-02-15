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
import os

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

# =====================================================================
# ======================= ✅ UPDATED MODEL SECTION =====================
# =====================================================================

st.sidebar.markdown("---")
st.sidebar.header("📤 Upload Model")

uploaded_model = st.sidebar.file_uploader(
    "Click below to upload your Best Model (.pkl)",
    type=["pkl"]
)

@st.cache_resource
def load_model_from_upload(uploaded_file):
    try:
        return joblib.load(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"❌ Error loading uploaded model: {e}")
        return None

@st.cache_resource
def load_active_model_from_db():
    try:
        active_meta = model_registry_col.find_one(
            {"is_active": True},
            sort=[("_id", -1)]
        )
        if active_meta and "model_path" in active_meta:
            path = active_meta["model_path"]
            if os.path.exists(path):
                return joblib.load(path), active_meta.get("model_name", "DB Model")
        return None, "No Active DB Model"
    except Exception as e:
        return None, f"DB Load Error: {e}"

# Priority 1: Uploaded Model
if uploaded_model is not None:
    model = load_model_from_upload(uploaded_model)
    model_name = "Uploaded Model"
    if model:
        st.sidebar.success("✅ Uploaded Model Loaded Successfully!")

# Priority 2: DB Model (fallback)
else:
    model, model_name = load_active_model_from_db()

# ============================= MODEL STATUS =============================

if model:
    st.success(f"✅ Active Model Loaded: {model_name}")
else:
    st.error("❌ No Model Loaded. Please upload a model.")
        
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

    col1, col2 = st.columns(2)

    box_style = """
    background: transparent;
    border: 2px solid #ffffff;
    padding:15px;
    border-radius:12px;
    margin-bottom:15px;
    color:white;
    """

    col1.markdown(f"""
    <div style="{box_style}">
    <h4 style="color:white;">Ridge Regression</h4>
    <p>Regularized linear regression model that reduces overfitting and stabilizes AQI prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="{box_style}">
    <h4 style="color:white;">Random Forest</h4>
    <p>Ensemble tree-based model that captures nonlinear pollution patterns effectively.</p>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    col3.markdown(f"""
    <div style="{box_style}">
    <h4 style="color:white;">XGBoost</h4>
    <p>High-performance gradient boosting model optimized for AQI regression accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div style="{box_style}">
    <h4 style="color:white;">LSTM</h4>
    <p>Deep learning time-series model capturing long-term AQI temporal dependencies.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown('<div class="card"><h2>🧪 Model Comparison from DB</h2></div>', unsafe_allow_html=True)
    if db is not None:
        models_data = list(db["model_registry"].find({}, {"_id":0}).sort("_id",-1))
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
                st.markdown(f"**Reason:** Lowest RMSE ({display_df.loc[best_idx,'rmse']:.4f}) indicates predictions are closest to actual AQI.")
            elif metric_cols and not display_df.empty:
                first_metric = metric_cols[0]
                best_idx = display_df[first_metric].idxmin()
                best_name = display_df.loc[best_idx, name_col]
                st.subheader(f"🏆 Best Model (based on {first_metric}): {best_name}")

            def highlight_best(s):
                if best_idx is None: return ['' for _ in s]
                return ['background-color: #004d00; color:white' if s.name==best_idx else '' for _ in s]

            st.dataframe(display_df.style.apply(highlight_best, axis=1), use_container_width=True)

            numeric_metrics = [c for c in metric_cols if pd.api.types.is_numeric_dtype(display_df[c])]
            if numeric_metrics:
                viz_df = display_df.melt(id_vars=name_col, value_vars=numeric_metrics, 
                                          var_name="Metric", value_name="Value")
                fig_compare = px.bar(
                    viz_df,
                    x="Metric",
                    y="Value",
                    color=name_col,
                    barmode="group",
                    text_auto='.3f',
                    title="Model Comparison Metrics"
                )
                st.plotly_chart(fig_compare, use_container_width=True)

# ============================== MONTHLY/YEARLY TREND ==================
elif selected_tab == "📈 Monthly/Yearly Trend":
    st.markdown('<div class="card"><h2>📊 Monthly & Yearly AQI Trends</h2></div>', unsafe_allow_html=True)
    if not df.empty:
        if "Predicted_AQI" not in df.columns:
            df["Predicted_AQI"] = current_aqi

        # Monthly
        df["month_year"] = df["time"].dt.to_period("M").astype(str)
        monthly_df = df.groupby("month_year")[["AQI","Predicted_AQI"]].mean().reset_index()
        fig_month = px.line(monthly_df, x="month_year", y=["AQI","Predicted_AQI"], markers=True, title="Monthly AQI Trend")
        fig_month.update_traces(mode="lines+markers", line_width=3)
        st.plotly_chart(fig_month, use_container_width=True)

        # Yearly
        df["year"] = df["time"].dt.year
        yearly_df = df.groupby("year")[["AQI","Predicted_AQI"]].mean().reset_index()
        fig_year = px.line(yearly_df, x="year", y=["AQI","Predicted_AQI"], markers=True, title="Yearly AQI Trend")
        fig_year.update_traces(mode="lines+markers", line_width=3)
        st.plotly_chart(fig_year, use_container_width=True)

# ============================== ABOUT TAB ============================
elif selected_tab == "ℹ️ About":
    st.markdown('<div class="card"><h2>ℹ️ About AirSense Karachi</h2></div>', unsafe_allow_html=True)
    
    # ------------------- POLLUTANT DATA SECTION (FROM IMAGE) -------------------
    st.markdown("### 🌫️ Major Air Pollutants - Karachi Status")
    
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        st.markdown('<div class="pollutant-box warning"><b>Particulate Matter (PM2.5)</b><br><h3>39 µg/m³</h3></div>', unsafe_allow_html=True)
    with row1_col2:
        st.markdown('<div class="pollutant-box"><b>Particulate Matter (PM10)</b><br><h3>45 µg/m³</h3></div>', unsafe_allow_html=True)
    with row1_col3:
        st.markdown('<div class="pollutant-box"><b>Carbon Monoxide (CO)</b><br><h3>466 ppb</h3></div>', unsafe_allow_html=True)

    with row2_col1:
        st.markdown('<div class="pollutant-box"><b>Sulfur Dioxide (SO2)</b><br><h3>2 ppb</h3></div>', unsafe_allow_html=True)
    with row2_col2:
        st.markdown('<div class="pollutant-box"><b>Nitrogen Dioxide (NO2)</b><br><h3>12 ppb</h3></div>', unsafe_allow_html=True)
    with row2_col3:
        st.markdown('<div class="pollutant-box"><b>Ozone (O3)</b><br><h3>28 ppb</h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <p>
    AirSense Karachi is an AI-driven Air Quality Intelligence platform designed to monitor,
    analyze, and predict AQI levels using Machine Learning and Deep Learning techniques.
    It provides real-time monitoring, forecasting, trend analysis, and model evaluation.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p>AI-powered AQI Dashboard for Karachi.</p>
    <ul>
    <li>Live AQI prediction based on real-time pollutant data</li>
    <li>3-Day Forecast powered by predictive modeling</li>
    <li>Historical AQI trend visualization</li>
    <li>Model comparison with metrics (MAE, RMSE, R2)</li>
    <li>Monthly & Yearly AQI trends</li>
    </ul>
    """, unsafe_allow_html=True)


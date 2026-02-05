import os
import pandas as pd
import numpy as np
import joblib
import shap
from datetime import datetime
from pymongo import MongoClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def run_training_pipeline():
    # ================= MONGO CONNECTION =================
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI environment variable not set!")
    client = MongoClient(mongo_uri)
    db = client["aqi_db"]
    registry_col = db["model_registry"]
    metrics_col = db["metrics_history"]
    features_col = db["features"]
    print("✅ Connected to MongoDB Atlas")

    # ================= LOAD DATA =================
    if not os.path.exists("features.csv"):
        raise FileNotFoundError("❌ features.csv not found")
    df = pd.read_csv("features.csv").dropna()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # ================= CREATE AQI ALERTS =================
    if "AQI_ALERT" not in df.columns:
        def aqi_alert(aqi):
            if aqi > 300: return "🚨 Severe Hazard"
            elif aqi > 200: return "⚠️ Very Unhealthy"
            elif aqi > 150: return "⚠️ Unhealthy"
            else: return "✅ Safe"

        df["AQI_ALERT"] = df["AQI"].apply(aqi_alert)

    # ================= SAVE ALERTS TO MONGODB =================
    for _, row in df.iterrows():
        features_col.update_one(
            {"time": row["time"]},
            {"$set": {"AQI_ALERT": row["AQI_ALERT"]}},
            upsert=True
        )
    print("✅ AQI alerts saved to MongoDB")

    # ================= FEATURES & TARGET =================
    X = df.drop(columns=["AQI", "time"], errors="ignore")
    y = df["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ================= TRAIN MODELS =================
    models = {}

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    models["Ridge"] = ridge

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # XGBoost
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    # ================= CREATE SHAP EXPLAINER =================
    shap_explainer = shap.Explainer(xgb)
    os.makedirs("saved_models", exist_ok=True)
    shap_path = "saved_models/shap_explainer.pkl"
    joblib.dump(shap_explainer, shap_path)
    print(f"✅ SHAP explainer saved: {shap_path}")

    # ================= LSTM =================
    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(y.values.reshape(-1, 1))

    def create_sequences(data, window=24):
        Xs, ys = [], []
        for i in range(len(data) - window):
            Xs.append(data[i:i+window])
            ys.append(data[i+window])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(scaled_y)
    split = int(0.8 * len(X_seq))
    X_train_lstm, X_test_lstm = X_seq[:split], X_seq[split:]
    y_train_lstm, y_test_lstm = y_seq[:split], y_seq[split:]

    lstm = Sequential([
        LSTM(50, activation="relu", input_shape=(24, 1)),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)
    models["LSTM"] = lstm

    # ================= METRICS EVALUATION =================
    metrics = {}
    for name, model in models.items():
        if name == "LSTM":
            preds_scaled = model.predict(X_test_lstm).flatten()
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            y_true = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
        else:
            preds = model.predict(X_test)
            y_true = y_test
        metrics[name] = {
            "MAE": mean_absolute_error(y_true, preds),
            "RMSE": np.sqrt(mean_squared_error(y_true, preds)),
            "R2": r2_score(y_true, preds)
        }

    # ================= SELECT BEST MODEL =================
    best_model_name = min(metrics, key=lambda m: metrics[m]["RMSE"])
    best_model = models[best_model_name]
    print(f"🏆 Best Model: {best_model_name}")

    # ================= MODEL VERSIONING =================
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = f"saved_models/{best_model_name}_v{version}.pkl"
    joblib.dump(best_model, model_path)

    # ================= DEACTIVATE OLD MODELS =================
    registry_col.update_many({"is_active": True}, {"$set": {"is_active": False}})

    # ================= STORE REGISTRY ENTRY =================
    registry_col.insert_one({
        "model_name": best_model_name,
        "version": version,
        "metrics": metrics[best_model_name],
        "model_path": model_path,
        "shap_path": shap_path,
        "trained_at": datetime.utcnow(),
        "is_active": True
    })

    # ================= METRICS HISTORY =================
    metrics_col.insert_one({
        "version": version,
        "all_metrics": metrics,
        "best_model": best_model_name,
        "created_at": datetime.utcnow()
    })

    print("✅ Model registry, SHAP, metrics history updated")
    print(f"✅ Model saved at {model_path}")

if __name__ == "__main__":
    run_training_pipeline()

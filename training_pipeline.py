from pymongo import MongoClient
import os
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def run_training_pipeline():
    # ----------------- MONGO CONNECTION -----------------
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI environment variable not set!")
    client = MongoClient(mongo_uri)
    db = client["aqi_db"]
    print("✅ Connected to MongoDB Atlas")

    # ----------------- LOAD DATA -----------------
    if not os.path.exists("features.csv"):
        raise FileNotFoundError("❌ features.csv not found in repo. Please add it.")
    df = pd.read_csv("features.csv")
    df = df.dropna()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # ----------------- FEATURES & TARGET -----------------
    X = df.drop(columns=["AQI", "time"], errors="ignore")
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # ----------------- RIDGE -----------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    # ----------------- RANDOM FOREST -----------------
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # ----------------- XGBOOST -----------------
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train, y_train)

    # ----------------- LSTM -----------------
    scaler = MinMaxScaler()
    scaled_aqi = scaler.fit_transform(y.values.reshape(-1, 1))
    def create_sequences(data, window=24):
        X_seq, y_seq = [], []
        for i in range(len(data) - window):
            X_seq.append(data[i:i+window])
            y_seq.append(data[i+window])
        return np.array(X_seq), np.array(y_seq)
    X_seq, y_seq = create_sequences(scaled_aqi)
    split = int(0.8 * len(X_seq))
    X_train_lstm, X_test_lstm = X_seq[:split], X_seq[split:]
    y_train_lstm, y_test_lstm = y_seq[:split], y_seq[split:]
    model = Sequential([LSTM(50, activation="relu", input_shape=(24,1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32)

    # ----------------- SAVE BEST MODEL -----------------
    trained_models = {"Ridge": ridge, "RF": rf, "XGB": xgb, "LSTM": model}
    best_model_name = "XGB"  # Use your logic if needed
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(trained_models[best_model_name], f"saved_models/best_model_{best_model_name}.pkl")
    print(f"✅ Best model saved: {best_model_name}")

    # ----------------- MONGODB MODEL REGISTRY -----------------
    MAX_BSON_SIZE = 15 * 1024 * 1024
    for name, m in trained_models.items():
        try:
            data = {"model_name": name}
            binary = pickle.dumps(m)
            if len(binary) <= MAX_BSON_SIZE:
                data["model_binary"] = binary
            else:
                data["model_binary"] = f"Model too large ({len(binary)/(1024*1024):.2f} MB)"
            db["model_registry"].insert_one(data)
            print(f"✅ {name} stored in MongoDB")
        except Exception as e:
            print(f"🚨 Failed to store {name}: {e}")

if __name__ == "__main__":
    run_training_pipeline()

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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

   # ---------------- Load Data ----------------
    data = list(feature_collection.find({}, {"_id": 0}))
    if len(data) == 0:
        raise ValueError("❌ No data found in feature collection")

    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records from MongoDB")

    TARGET_COLUMN = "AQI"
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"❌ Target column '{TARGET_COLUMN}' not found in data")

    # ---------------- CLEAN FEATURES ----------------
    drop_cols = ["time", "AQI_ALERT"]
    X = df.drop(columns=[TARGET_COLUMN] + [c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[TARGET_COLUMN]

    # ---------------- Train/Test Split ----------------
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    tscv = TimeSeriesSplit(n_splits=5)

    # ---------------- Models ----------------
    model_grids = {
        "Ridge": {
            "model": Ridge(),
            "params": {"alpha": [0.01, 0.1, 1.0]}
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {"n_estimators": [100], "max_depth": [5, 10]}
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, verbosity=0),
            "params": {"n_estimators": [200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]}
        }
    }

    cv_results = {}
    best_models = {}

    # ---------------- Traditional ML ----------------
    for name, config in model_grids.items():
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_models[name] = best_model

        maes, rmses, r2s = [], [], []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            best_model.fit(X_tr, y_tr)
            preds = best_model.predict(X_val)

            maes.append(mean_absolute_error(y_val, preds))
            rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            r2s.append(r2_score(y_val, preds))

        mae_mean = np.mean(maes)
        r2_mean = np.mean(r2s)

        print(f"\n📊 {name} Performance:")
        print(f"   MAE  : {mae_mean:.4f}")
        print(f"   RMSE : {rmse_mean:.4f}")
        print(f"   R2   : {r2_mean:.4f}")
        print(f"   Robust Score : {(rmse_mean + rmse_std):.4f}")
        print(f"   Best Params  : {grid.best_params_}")
    # ---------------- LSTM ----------------
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
        LSTM(50, activation="relu", input_shape=(X_train_lstm.shape[1], 1)),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=0)

    preds_scaled = lstm.predict(X_test_lstm).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

    rmse_mean = np.sqrt(mean_squared_error(y_true, preds))

    mae_lstm = mean_absolute_error(y_true, preds)
    r2_lstm = r2_score(y_true, preds)

    print(f"\n📊 LSTM Performance:")
    print(f"   MAE  : {mae_lstm:.4f}")
    print(f"   RMSE : {rmse_mean:.4f}")
    print(f"   R2   : {r2_lstm:.4f}")
    print(f"   Robust Score : {rmse_mean:.4f}")

    best_models["LSTM"] = lstm

    # ---------------- Select Best ----------------
    best_model_name = min(cv_results, key=lambda m: cv_results[m]["Robust_Score"])
    print("🏆 Best Model:", best_model_name)

    # ---------------- Save All Models ----------------
    registry_collection.update_many({}, {"$set": {"is_active": False}})
    os.makedirs("saved_models", exist_ok=True)

    for model_name, model_obj in best_models.items():

        if model_name == "LSTM":
            model_path = f"saved_models/{model_name}.h5"
            model_obj.save(model_path)
        else:
            model_path = f"saved_models/{model_name}.pkl"
            joblib.dump(model_obj, model_path)

        registry_entry = {
            "model_name": model_name,
            "model_path": model_path,
            "cv_metrics": cv_results[model_name],
            "is_active": model_name == best_model_name,
            "created_at": datetime.utcnow()
        }

        registry_collection.insert_one(registry_entry)
        print(f"✅ {model_name} saved (Active={model_name == best_model_name})")

    print(" Training Completed Successfully!")

if __name__ == "__main__":
    run_training_pipeline()

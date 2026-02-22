import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from prophet import Prophet  # <-- new import for Prophet

from dotenv import load_dotenv
load_dotenv()


def run_training_pipeline():

    # ================= MONGO CONNECTION =================
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI environment variable not set!")

    client = MongoClient(mongo_uri)
    db = client["aqi_db"]
    registry_col = db["model_registry"]
    features_col = db["features"]

    print("✅ Connected to MongoDB Atlas")

    # ================= LOAD DATA =================
    data = list(features_col.find({}, {"_id": 0}))
    if len(data) == 0:
        raise ValueError("❌ No data found in feature collection")

    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records")

    # ================= SORT BY TIME =================
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # ================= CREATE AQI =================
    print("ℹ Creating AQI from pollutants...")

    df["AQI"] = (
        0.35 * df["pm25"] +
        0.20 * df["pm10"] +
        0.15 * df["no2"] +
        0.10 * df["so2"] +
        0.10 * df["o3"] +
        0.10 * df["co"]
    )

    # ================= CREATE LAG FEATURES =================
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "o3", "co"]

    for col in pollutant_cols:
        df[f"{col}_lag1"] = df[col].shift(1)

    df["AQI_lag1"] = df["AQI"].shift(1)

    # ================= CREATE FUTURE TARGET =================
    df["AQI_target"] = df["AQI"].shift(-1)

    # Remove rows from shifting
    df = df.dropna().reset_index(drop=True)

    TARGET_COLUMN = "AQI_target"

    # ================= REMOVE LEAKAGE =================
    leakage_cols = [
        "AQI", "AQI_target",
        "pm25", "pm10", "no2", "so2", "o3", "co"
    ]

    X = df.drop(columns=leakage_cols + ["time"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.loc[:, X.nunique() > 1]

    y = df[TARGET_COLUMN]

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    if X.shape[1] == 0:
        raise ValueError("❌ No features left after leakage removal!")

    # ================= TRAIN TEST SPLIT =================
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    tscv = TimeSeriesSplit(n_splits=5)

    # ================= MODEL DEFINITIONS =================
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
            "params": {
                "n_estimators": [200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6]
            }
        }
    }

    cv_results = {}
    best_models = {}

    # ================= TRADITIONAL MODELS =================
    for name, config in model_grids.items():
        print(f"\n🔍 Training {name}...")
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

        preds = best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        cv_results[name] = {
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": rmse,
            "R2": r2_score(y_test, preds),
            "Robust_Score": rmse
        }
        print(f"   RMSE: {rmse:.4f}")

    # ================= PROPHET =================
    print("\n🔍 Training Prophet...")

    prophet_df = df[['time', 'AQI_target']].rename(columns={'time':'ds', 'AQI_target':'y'})

    train_df = prophet_df.iloc[:split_index]
    test_df = prophet_df.iloc[split_index:]

    prophet_model = Prophet()
    prophet_model.fit(train_df)

    future = test_df[['ds']]
    forecast = prophet_model.predict(future)

    rmse_prophet = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
    print(f"   RMSE: {rmse_prophet:.4f}")

    cv_results["Prophet"] = {
        "MAE": mean_absolute_error(test_df['y'], forecast['yhat']),
        "RMSE": rmse_prophet,
        "R2": r2_score(test_df['y'], forecast['yhat']),
        "Robust_Score": rmse_prophet
    }
    best_models["Prophet"] = prophet_model

       # ================= SAVE MODELS =================
    run_id = os.getenv("GITHUB_RUN_NUMBER")

    if run_id:
        version_name = f"run-{run_id}"
    else:
        version_name = f"run-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    versioned_model_dir = os.path.join("models", version_name)
    os.makedirs(versioned_model_dir, exist_ok=True)

    registry_col.delete_many({})

    for model_name, metrics in cv_results.items():
        is_best = model_name == best_model_name

        model_path = os.path.join(versioned_model_dir, f"{model_name}.pkl")
        joblib.dump(best_models[model_name], model_path)

        registry_col.insert_one({
            "model_name": model_name,
            "model_path": model_path,
            "cv_metrics": metrics,
            "is_active": is_best,
            "created_at": datetime.utcnow()
        })

    print(f"✅ Models saved to: {versioned_model_dir}")
    print("🎉 Training Completed Successfully!")


if __name__ == "__main__":
    run_training_pipeline()

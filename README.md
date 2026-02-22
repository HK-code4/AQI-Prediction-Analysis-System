# 🌍 AirSense Karachi – AI-Powered AQI Intelligence

🔗 **Live UI (Streamlit Dashboard):**
👉 *[http://localhost:8501](https://aqi-prediction-analysis-system.streamlit.app/)*


## 1. Abstract

AirSense Karachi is an intelligent system designed to predict, forecast, and monitor the Air Quality Index (AQI) of Karachi using machine learning and deep learning models. The platform provides real-time hazard alerts, historical AQI trends, 3-day forecasting. All components are integrated into an interactive Streamlit dashboard backed by MongoDB Atlas for scalable data storage and model versioning.

## 2. Motivation

Air pollution is a serious environmental and health issue in metropolitan cities like Karachi. Accurate AQI prediction and monitoring helps:

* Raise public awareness about hazardous air conditions
* Enable citizens to take preventive health measures
* Support data-driven environmental planning and policymaking

## 3. Objectives

* Predict real-time AQI using sensor and environmental data
* Forecast AQI for the next 3 days
* Automatically generate AQI hazard alerts
* Store models, metrics, and alerts in MongoDB
* Develop an interactive Streamlit-based dashboard

## 4. Technology Stack

**Data Source**

* Open-Meteo API (Air Quality & Weather)

**Programming & Tools**

* Python
* VS Code / PyCharm / Google Colab

**Machine Learning**

* Scikit-learn
* XGBoost
* TensorFlow / Keras (LSTM)

**Database**

* MongoDB Atlas (Cloud NoSQL)

**Backend & Pipelines**

* Python Feature Pipeline
* Training Pipeline
* GitHub Actions (CI/CD)

**Frontend**

* Streamlit

## 5. Dataset

The dataset contains engineered AQI features including:

* **Pollutants:** PM2.5, PM10, CO, NO₂, SO₂, Ozone
* **Environmental:** Temperature, Wind Speed
* **Temporal:** Hour, Day, Month, Day of Week
* **Target:** AQI
* **Derived Feature:** AQI change rate

All data is stored and updated in MongoDB Atlas for real-time and historical analysis.

## 6. Features

* **Real-time AQI Prediction**
  Predicts current AQI using machine learning and deep learning models including XGBoost, Random Forest, Ridge Regression, and LSTM.
* **3-Day AQI Forecast**
  Forecasts AQI values for the next three days based on the latest environmental data.
* **Hazard Alerts**
  Automatically categorizes AQI levels as ✅ Safe, ⚠️ Unhealthy, or 🚨 Severe Hazard. Alerts are stored and retrieved from MongoDB in real time.
* **Model Comparison**
  Compares multiple models (Ridge, Random Forest, XGBoost, LSTM) using MAE, RMSE, and R² metrics with both tabular and visual representation.
* **Historical AQI Trends**
  Visualizes past AQI trends reconstructed using the trained model.
* **MongoDB Integration**
  Stores features, AQI alerts, trained models, SHAP explainers, and evaluation metrics in MongoDB Atlas.
* **Interactive Streamlit UI**
  User-friendly dashboard with sidebar controls for environmental inputs and real-time visualization.

## 7. System Architecture

The AQI Intelligence Platform follows a **three-layer architecture** with an end-to-end automated data flow.

### 🔷 Overall Architecture Diagram (Horizontal)

```
┌────────────────────┐
│   Open-Meteo API   │
│  (Air Quality Data)│
└─────────┬──────────┘
          │
          ▼
┌────────────────────────┐
│  Local Environment     │
│ (VS Code / PyCharm)    │
│ Feature Pipeline       │
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────────────────┐
│        MongoDB Atlas (Cloud)       │
│ ┌──────────────┐ ┌───────────────┐ │
│ │ features     │ │ model_registry│ │
│ │ AQI + ALERT  │ │     model     │ │
│ └──────────────┘ └───────────────┘ │
│ ┌────────────────────────────────┐ │
│ │ metrics_history                │ │
│ │ Model evaluation records       │ │
│ └────────────────────────────────┘ │
└─────────┬──────────────────────────┘
          │
          ▼
┌────────────────────────┐
│  Model Training Layer  │
│ Ridge | RF | XGB |LSTM │                  
└─────────┬──────────────┘
          │
          ▼
┌────────────────────────┐
│   Streamlit Dashboard  │
│ AQI | Forecast         │
└────────────────────────┘
```
### 7.1 Data Layer

* Open-Meteo API collects AQI data
* Data stored in MongoDB:

  * `features` → sensor data + AQI + AQI_ALERT
  * `model_registry` → active model 
  * `metrics_history` → model performance

### 7.2 Model Layer

* Models: Ridge, Random Forest, XGBoost, LSTM
* Evaluation: MAE, RMSE, R²
* Best model selected automatically

**Hazard Alert Rules**

* > 300 → 🚨 Severe Hazard
* 201–300 → ⚠️ Very Unhealthy
* 151–200 → ⚠️ Unhealthy
* ≤150 → ✅ Safe

### 7.3 Application / UI Layer

* Streamlit dashboard loads:

  * Active model
  * Latest AQI data
* UI includes:

  * Metric cards
  * Historical trends
  * 3-day forecast
  * Model comparison table

## 8. Methodology

### 8.1 Data Preprocessing

* Missing values removed
* Datetime conversion
* Feature scaling for LSTM
* Feature order alignment for XGBoost

### 8.2 Model Training

1. Train multiple models
2. Evaluate MAE, RMSE, R²
3. Select best model
5. Update MongoDB registry

### 8.3 Hazard Alert System

* AQI thresholds applied
* Alerts stored in MongoDB
* Displayed live in UI

## 9. Results

| Model         | MAE     | RMSE    | R²       |
| ------------- | ------- | ------- | -------- |
| **Ridge**     | 6.68    | 9.312   | 0.97     |
| Random Forest | 14      | 22.57   | 0.86     |
| XGBoost       | 14      | 21.59   | 0.86     |
| LSTM          | 22      | 31      | 0.74     |

**Best Model:** **Ridge**

## 10. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run feature pipeline
python feature_pipeline.py

# Run training pipeline
python training_pipeline.py

# Launch Streamlit UI
streamlit run app.py
```
## 11. Usage

* Adjust environmental parameters such as **PM2.5** using the sidebar controls to observe AQI prediction changes.
* Upload a **SHAP explainer file** (optional) to enable explainability.
* Monitor **current AQI, air status, and hazard alerts** in real time.
* Analyze **historical AQI trends** through interactive charts.
* View **model comparison table and metric visualizations**.
* Forecast **AQI for the next three days** using the trained model.

## 12. Conclusion

AirSense Karachi delivers a complete AI-driven AQI monitoring solution with:

* Accurate predictions
* Real-time hazard alerts
* Explainable AI
* Scalable cloud architecture
* User-friendly dashboard

The system is production-ready and extendable to other cities and real-time IoT deployments.

# AirSense Karachi – AI-Powered AQI Intelligence
## 1. Abstract
AirSense Karachi is an intelligent system designed to predict, forecast, and monitor Air Quality Index (AQI) in Karachi using machine learning and deep learning models. The platform provides real-time hazard alerts, historical trends, 3-day forecasting, and model explainability through SHAP, all integrated into an interactive Streamlit dashboard with MongoDB backend storage.

## 2. Motivation
Air pollution is a major concern in urban areas like Karachi. Real-time AQI monitoring helps:

          Raise awareness about hazardous air conditions.

          Allow residents to make informed health decisions.

          Support environmental monitoring and policy-making.

## 3. Objectives

Predict current AQI using sensor data and machine learning models.

Forecast next 3-day AQI trends.

Automatically generate hazard alerts based on AQI levels.

Provide model explainability with SHAP.

Store all data, metrics, and models in MongoDB for reproducibility and version control.

Develop an interactive Streamlit UI for visualization and real-time interaction.

4. Dataset

The system uses a features dataset containing:

Air pollutants: pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone.

Environmental features: temperature_2m, wind_speed_100m.

Time-related features: hour, day, month, day_of_week.

Target: AQI (Air Quality Index)

Derived feature: aqi_change_rate

Data is stored and maintained in MongoDB, enabling both historical and real-time analysis.

"""
config.py — Central configuration for the AQI Forecasting System.

All secrets and environment-specific values are loaded from environment
variables (via python-dotenv).  Every other module must import configuration
from this file — never call ``os.getenv`` directly elsewhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (ignored in CI; env vars are injected via GitHub Secrets)
# ---------------------------------------------------------------------------
load_dotenv()

# ===========================================================================
# 1. Environment Variables (single source of truth)
# ===========================================================================
HOPSWORKS_API_KEY: str = os.getenv("HOPSWORKS_API_KEY", "aqi_pro")
HOPSWORKS_PROJECT: str = os.getenv("HOPSWORKS_PROJECT", "aqi_project")
HOPSWORKS_HOST: str = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
WINDOWS_TMP_DIR: str = os.getenv("WINDOWS_TMP_DIR", "D:/tmp")
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
CITY: str = os.getenv("CITY", "")
LAT: float = float(os.getenv("LAT", "0.0"))
LON: float = float(os.getenv("LON", "0.0"))

# ===========================================================================
# 2. Project Paths
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 3. Hopsworks Feature Store
# ===========================================================================
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

# ===========================================================================
# 4. OpenWeatherMap API URLs
# ===========================================================================
OPENWEATHER_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_HISTORY_POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
OPENWEATHER_TIMEMACHINE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# ===========================================================================
# 5. Feature Engineering
# ===========================================================================
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "pm2_5",
    "pm10",
    "no2",
    "o3",
    "co",
]
DERIVED_COLUMNS = [
    "hour",
    "day",
    "month",
    "aqi_change_rate",
    "rolling_mean_3",
]
TARGET_COLUMN = "aqi"
DATETIME_COLUMN = "timestamp"
CITY_COLUMN = "city"

LAG_FEATURES = [1, 2, 3, 6, 12, 24]  # hours
ROLLING_WINDOWS = [6, 12, 24]          # hours

# ===========================================================================
# 6. Model Training
# ===========================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Scikit-learn (Random Forest baseline)
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 16,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Scikit-learn (Ridge Regression)
RIDGE_PARAMS = {
    "alpha": 1.0,
    "solver": "auto",
}

# TensorFlow / Keras (Deep Neural Network)
DNN_PARAMS = {
    "hidden_units": [256, 128, 64, 32],   # neurons per hidden layer
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 150,
    "patience": 15,                         # early stopping patience
    "activation": "relu",
    "weight_decay": 1e-4,                   # L2 kernel regularisation
}

MODEL_REGISTRY_NAME = "aqi_forecast_model"

# ===========================================================================
# 7. Prediction / Inference
# ===========================================================================
FORECAST_HORIZON = 72  # hours ahead
AQI_CATEGORIES = {
    (0, 50): "Good",
    (51, 100): "Moderate",
    (101, 300): "Unhealthy",
    (301, 500): "Hazardous",
}

# ===========================================================================
# 8. Application / Serving
# ===========================================================================
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

STREAMLIT_PORT = 8501

# ===========================================================================
# 9. Logging
# ===========================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}"

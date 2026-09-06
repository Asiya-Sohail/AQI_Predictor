# AQI Forecasting System

A **production-ready, 100% serverless** Air Quality Index (AQI) forecasting system built with Python, Scikit-learn, TensorFlow, Hopsworks Feature Store, GitHub Actions, Streamlit, Flask, and SHAP.

---

## Project Structure

```
aqi_forecasting_system/
│
├── data/                              # Raw & processed data artefacts
├── notebooks/                         # Jupyter exploration notebooks
│
├── src/
│   ├── config.py                      # Central configuration & secrets
│   ├── feature_pipeline.py            # Data ingestion → feature engineering → Hopsworks
│   ├── training_pipeline.py           # Model training (RF + Ridge + DNN), evaluation, registry
│   ├── predict.py                     # Recursive 72-hour PM2.5 forecasting
│   ├── eda.py                         # Exploratory Data Analysis
│   ├── shap_explainer.py              # SHAP-based model explainability
│   ├── utils.py                       # Shared helpers (logging, metrics, I/O)
│
├── app/
│   ├── streamlit_app.py               # Interactive Streamlit dashboard
│   ├── api.py                         # Flask REST API for predictions
│
├── .github/workflows/
│   ├── feature_pipeline.yml           # Hourly feature ingestion (GitHub Actions)
│   ├── training_pipeline.yml          # Daily model retraining (GitHub Actions)
│
├── requirements.txt                   # Pinned Python dependencies
├── README.md                          # This file
└── main.py                            # CLI entry point
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-org>/aqi_forecasting_system.git
cd aqi_forecasting_system
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements-pipeline.txt
```

For Streamlit Cloud deployment, `requirements.txt` is intentionally lean and
cloud-compatible. Local training and GitHub Actions use
`requirements-pipeline.txt`.

### 2. Configure Secrets

Create a `.env` file in the project root (never commit this file):

```env
HOPSWORKS_API_KEY=<your-hopsworks-api-key>
HOPSWORKS_PROJECT=<your-hopsworks-project>
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
WINDOWS_TMP_DIR=D:/tmp
OPENWEATHER_API_KEY=<your-openweather-api-key>
CITY=<your-city>
LAT=<latitude>
LON=<longitude>
```

If your project is in EU cloud, `HOPSWORKS_HOST` must be `eu-west.cloud.hopsworks.ai`.

### 3. Run Pipelines

```bash
# Feature engineering (fetch data → clean → engineer → push to Hopsworks)
python main.py --pipeline features

# Historical backfill (populate training data for a date range)
python main.py --pipeline backfill --backfill-start 2024-01-01 --backfill-end 2024-12-31

# Exploratory Data Analysis
python main.py --pipeline eda

# Model training (Random Forest + Ridge + DNN)
python main.py --pipeline training

# 72-hour recursive PM2.5 forecast
python main.py --pipeline predict

# SHAP explanations
python main.py --pipeline shap

# Run everything end-to-end (features → training → predict → shap)
python main.py --pipeline all

# Use local data files instead of Hopsworks
python main.py --pipeline training --local
```

### 4. Serve

```bash
# Flask REST API (port 5000)
python main.py --serve api

# Streamlit Dashboard (port 8501)
python main.py --serve streamlit
```

---

## API Reference

| Method | Endpoint          | Description                              |
|--------|-------------------|------------------------------------------|
| GET    | `/health`         | Service readiness probe                  |
| GET    | `/predict`        | 3-day (72 h) recursive PM2.5 forecast   |
| GET    | `/model/info`     | Training report & metrics                |

### Example — 72-Hour Forecast

```bash
# Default 72-hour forecast
curl http://localhost:5000/predict

# Custom horizon (48 hours) with local feature source
curl "http://localhost:5000/predict?horizon=48&source=local"
```

**Response:**

```json
{
  "city": "Karachi",
  "model_type": "rf",
  "horizon_hours": 72,
  "count": 72,
  "elapsed_seconds": 1.234,
  "generated_at": "2026-02-22T12:00:00+00:00",
  "forecast": [
    {
      "timestamp": "2026-02-22T13:00:00+00:00",
      "predicted_pm2_5": 62.14,
      "predicted_aqi_category": "Moderate"
    }
  ]
}
```

---

## Models

| Model           | Framework     | Purpose                          |
|-----------------|---------------|----------------------------------|
| Random Forest   | Scikit-learn  | Fast, interpretable baseline     |
| Ridge Regression| Scikit-learn  | Regularised linear baseline      |
| Deep Neural Net | TensorFlow    | Fully-connected DNN with L2, BatchNorm, Dropout |

The training pipeline evaluates all three models (RMSE / MAE / R²) and automatically registers the best (lowest RMSE) in the Hopsworks Model Registry.

---

## Automation (GitHub Actions)

| Workflow              | Schedule        | Purpose                           |
|-----------------------|-----------------|-----------------------------------|
| `feature_pipeline.yml`| Every hour      | Fetch, clean, and store features  |
| `training_pipeline.yml`| Daily 02:00 UTC | Retrain models & generate SHAP   |

### Required GitHub Secrets

| Secret                 | Description                    |
|------------------------|--------------------------------|
| `HOPSWORKS_API_KEY`    | Hopsworks API key              |
| `HOPSWORKS_PROJECT`    | Hopsworks project name         |
| `HOPSWORKS_HOST`       | Hopsworks host/region endpoint |
| `OPENWEATHER_API_KEY`  | OpenWeatherMap API key         |
| `CITY`                 | Target city name               |
| `LAT`                  | City latitude                  |
| `LON`                  | City longitude                 |

---

## Hopsworks Troubleshooting (Windows)

If you see `API key invalid` or `Project does not exist`:

```powershell
& .\.venv\Scripts\Activate.ps1
Get-Command python | Format-List Source
python -c "import hopsworks; print(hopsworks.__version__)"
python check_hopsworks.py
```

Known-good startup sequence:

```powershell
& .\.venv\Scripts\Activate.ps1
$env:HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
python check_hopsworks.py
python main.py --serve api
```

Notes:
- Project names are case-sensitive.
- API key must belong to a user with access to that project.
- Rotate/regenerate API keys after any accidental exposure.

---

## Explainability

SHAP (SHapley Additive exPlanations) support includes:
- **Beeswarm summary plot** — global feature importance
- **Bar plot** — mean |SHAP| feature ranking
- **Feature ranking CSV** — full ranked feature table
- **JSON report** — top 5 features and sample metadata

Supports all three model types:
- RandomForest → `shap.TreeExplainer`
- Ridge → `shap.LinearExplainer`
- DNN → `shap.GradientExplainer`

Generated artefacts are saved to `data/shap_outputs/`.

---

## Dashboard (Streamlit)

- **Current AQI** — PM2.5 metric card with colour-coded category badge
- **3-day forecast chart** — interactive Plotly line chart with AQI bands
- **Daily AQI badges** — Day 1 / 2 / 3 summary with avg/min/max
- **Hazardous alerts** — warning banner when PM2.5 exceeds 150 µg/m³
- **SHAP explanations** — interactive feature importance bar chart
- **Flask API integration** — connects to the REST API with local fallback

---

## Production Deployment (Recommended)

### 1) Deploy API separately (Render)

- This repo includes `render.yaml` for a Flask API service.
- Render builds using `requirements-api.txt` and starts:

```bash
gunicorn --workers=2 --threads=4 --timeout=120 --bind=0.0.0.0:$PORT app.api:app
```

- Set these Render environment variables:
  - `OPENWEATHER_API_KEY`
  - `CITY`, `LAT`, `LON`
  - `HOPSWORKS_API_KEY` (optional)
  - `HOPSWORKS_PROJECT` (optional)
  - `HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai`

### 2) Connect Streamlit Cloud to deployed API

- Main file path: `app/streamlit_app.py`
- In Streamlit app secrets, set:

```toml
AQI_API_BASE_URL = "https://<your-render-service>.onrender.com"
OPENWEATHER_API_KEY = "<your-openweather-key>"
CITY = "Karachi"
LAT = "24.8607"
LON = "67.0011"
```

Use `.streamlit/secrets.toml.example` as reference.

### 3) Keep pipelines running

- GitHub Actions workflows are configured for scheduled feature and training
  jobs and install from `requirements-pipeline.txt`.

---

## Tech Stack

| Component        | Technology                        |
|------------------|-----------------------------------|
| Language         | Python 3.11+                      |
| ML (Classical)   | Scikit-learn                      |
| ML (Deep)        | TensorFlow / Keras                |
| Feature Store    | Hopsworks                         |
| Explainability   | SHAP                              |
| Dashboard        | Streamlit                         |
| REST API         | Flask                             |
| CI/CD            | GitHub Actions                    |
| Logging          | Loguru                            |
| Visualisation    | Plotly, Matplotlib, Seaborn       |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

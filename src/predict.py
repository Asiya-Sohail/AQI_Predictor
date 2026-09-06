"""
predict.py — Inference module for PM2.5 forecasting (72-hour horizon).

Pipeline
--------
1.  Download the best model from the Hopsworks Model Registry.
2.  Fetch the latest feature row from the Hopsworks Feature Store (or local).
3.  Recursively predict one-hour steps for the next 72 hours, feeding each
    predicted PM2.5 value back into the feature vector for the subsequent
    step.
4.  Return a DataFrame with ``timestamp``, ``predicted_pm2_5``, and
    ``predicted_aqi_category``.

AQI classification (PM2.5 µg/m³):
    Good       :   0 –  50
    Moderate   :  51 – 100
    Unhealthy  : 101 – 300
    Hazardous  : 301 – 500
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.config import (
    AQI_CATEGORIES,
    CITY,
    DATA_DIR,
    DATETIME_COLUMN,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    FORECAST_HORIZON,
    MODEL_DIR,
    MODEL_REGISTRY_NAME,
)
from src.hopsworks_client import login_to_hopsworks
from src.utils import (
    load_json,
    load_keras_model,
    load_sklearn_model,
    log,
    save_dataframe,
)

# ---------------------------------------------------------------------------
# Model key → local artefact path (relative to MODEL_DIR)
# ---------------------------------------------------------------------------
_MODEL_ARTEFACTS: Dict[str, str] = {
    "rf": "random_forest_pm25.joblib",
    "ridge": "ridge_pm25.joblib",
    "dnn": "dnn_pm25_model",
}


# ---------------------------------------------------------------------------
# Diurnal profile helpers — simulate realistic hourly variation for features
# that would otherwise stay frozen during recursive forecasting.
# ---------------------------------------------------------------------------
def _diurnal_factor(hour: int, peak_hour: int = 14, amplitude: float = 1.0) -> float:
    """Return a sinusoidal multiplier centred on *peak_hour*."""
    return amplitude * np.cos(2 * np.pi * (hour - peak_hour) / 24)


def _update_weather_features(
    current: pd.Series,
    hour: int,
    seed_row: pd.Series,
) -> None:
    """
    Apply realistic diurnal variation to weather & co-pollutant features.

    Instead of keeping them frozen at the seed value, we modulate each
    feature around its seed value using physically-motivated hourly profiles:

    - **temperature**: peaks ~14:00, range ±5 °C
    - **humidity**: inverse of temperature, peaks ~05:00, range ±12 %
    - **wind_speed**: peaks ~14:00, range ±40 % of seed
    - **pressure**: very small diurnal tide (±1 hPa), peaks ~10:00 and ~22:00
    - **co**: traffic-driven, peaks morning (08) and evening (20)
    - **no2**: same traffic profile as CO
    - **o3**: photochemical, peaks ~15:00
    - **pm10**: coarse dust, peaks morning rush + afternoon, range ±30 %
    """
    # --- Meteorological features ---
    if "temperature" in current.index:
        base = float(seed_row["temperature"])
        current["temperature"] = base + _diurnal_factor(hour, peak_hour=14, amplitude=5.0)

    if "humidity" in current.index:
        base = float(seed_row["humidity"])
        # Humidity is inversely related to temperature
        delta = _diurnal_factor(hour, peak_hour=5, amplitude=12.0)
        current["humidity"] = np.clip(base + delta, 10, 100)

    if "wind_speed" in current.index:
        base = float(seed_row["wind_speed"])
        factor = 1.0 + 0.4 * _diurnal_factor(hour, peak_hour=14, amplitude=1.0)
        current["wind_speed"] = max(base * factor, 0.1)

    if "pressure" in current.index:
        base = float(seed_row["pressure"])
        # Semi-diurnal tide (~12 h period)
        current["pressure"] = base + 1.0 * np.cos(2 * np.pi * (hour - 10) / 12)

    # --- Co-pollutant features (traffic + photochemistry) ---
    if "co" in current.index:
        base = float(seed_row["co"])
        # Bimodal traffic peaks at 08:00 and 20:00
        morning = np.exp(-0.5 * ((hour - 8) / 2.5) ** 2)
        evening = np.exp(-0.5 * ((hour - 20) / 2.5) ** 2)
        factor = 1.0 + 0.5 * (morning + evening) - 0.3
        current["co"] = max(base * factor, 50.0)

    if "no2" in current.index:
        base = float(seed_row["no2"])
        morning = np.exp(-0.5 * ((hour - 8) / 3.0) ** 2)
        evening = np.exp(-0.5 * ((hour - 20) / 3.0) ** 2)
        factor = 1.0 + 0.6 * (morning + evening) - 0.3
        current["no2"] = max(base * factor, 0.01)

    if "o3" in current.index:
        base = float(seed_row["o3"])
        # Ozone peaks in afternoon (photochemical)
        factor = 1.0 + 0.5 * _diurnal_factor(hour, peak_hour=15, amplitude=1.0)
        current["o3"] = max(base * factor, 1.0)

    if "pm10" in current.index:
        base = float(seed_row["pm10"])
        morning = np.exp(-0.5 * ((hour - 9) / 3.0) ** 2)
        afternoon = np.exp(-0.5 * ((hour - 17) / 3.0) ** 2)
        factor = 1.0 + 0.3 * (morning + afternoon) - 0.2
        current["pm10"] = max(base * factor, 1.0)


# ===========================================================================
# 1.  AQI Classification
# ===========================================================================
def classify_aqi(pm25_value: float) -> str:
    """
    Map a PM2.5 concentration (µg/m³) to a human-readable AQI tier.

    Uses the AQI_CATEGORIES dict from config.py.
    """
    for (low, high), label in AQI_CATEGORIES.items():
        if low <= pm25_value <= high:
            return label
    if pm25_value > 500:
        return "Hazardous"
    return "Good"


# ===========================================================================
# 2.  Model & Artefact Loading
# ===========================================================================
def download_model_from_registry() -> Tuple[str, Path]:
    """
    Download the latest model version from the Hopsworks Model Registry.

    Returns
    -------
    model_name : str
        The registry model name.
    local_dir : Path
        Directory where artefacts were downloaded.
    """
    project = login_to_hopsworks()
    mr = project.get_model_registry()
    hw_model = mr.get_best_model(
        name=MODEL_REGISTRY_NAME,
        metric="rmse",
        direction="min",
    )
    local_dir = Path(hw_model.download())
    log.info(
        f"Downloaded model '{MODEL_REGISTRY_NAME}' v{hw_model.version} "
        f"→ {local_dir}"
    )
    return MODEL_REGISTRY_NAME, local_dir


def _load_model(model_type: str) -> Any:
    """Load the model artefact matching *model_type*."""
    artefact = _MODEL_ARTEFACTS.get(model_type)
    if artefact is None:
        raise ValueError(f"Unknown model type: {model_type}")
    if model_type == "dnn":
        return load_keras_model(artefact)
    return load_sklearn_model(artefact)


class PM25Predictor:
    """
    Stateful predictor that caches model, scaler, and feature metadata.

    Load order
    ----------
    1.  Try Hopsworks Model Registry (downloads artefacts).
    2.  Fall back to local ``models/`` directory.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_registry: bool = True,
    ) -> None:
        self.model_dir = model_dir or MODEL_DIR
        self._use_registry = use_registry
        self._model: Any = None
        self._scaler: Any = None
        self._model_type: Optional[str] = None
        self._feature_columns: Optional[List[str]] = None

    # ----- lazy loaders -----------------------------------------------------
    def _load_artefacts(self) -> None:
        """Load model, scaler, and feature-column metadata."""
        # 1. Try downloading from Hopsworks
        if self._use_registry:
            try:
                _, dl_dir = download_model_from_registry()
                # Artefacts land inside dl_dir; copy them to MODEL_DIR for
                # consistency, then load from MODEL_DIR as usual.
                import shutil

                for item in dl_dir.iterdir():
                    dest = self.model_dir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                log.info("Registry artefacts synced to local MODEL_DIR")
            except Exception as exc:
                log.warning(
                    f"Model registry download failed ({exc}); "
                    "falling back to local artefacts"
                )

        # 2. Load training report to decide which model
        report = load_json(self.model_dir / "training_report.json")
        self._model_type = report["best_model"]

        artefact = _MODEL_ARTEFACTS.get(self._model_type)
        if artefact is None:
            raise ValueError(
                f"Unknown model type in training_report.json: {self._model_type}"
            )

        if self._model_type == "dnn":
            self._model = load_keras_model(artefact)
        else:
            self._model = load_sklearn_model(artefact)

        self._scaler = joblib.load(self.model_dir / "scaler.joblib")

        meta = load_json(self.model_dir / "feature_columns.json")
        self._feature_columns = meta["feature_columns"]

        log.info(
            f"Predictor ready — model={self._model_type}, "
            f"features={len(self._feature_columns)}"
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            self._load_artefacts()
        return self._model

    @property
    def scaler(self) -> Any:
        if self._scaler is None:
            self._load_artefacts()
        return self._scaler

    @property
    def model_type(self) -> str:
        if self._model_type is None:
            self._load_artefacts()
        return self._model_type

    @property
    def feature_columns(self) -> List[str]:
        if self._feature_columns is None:
            self._load_artefacts()
        return self._feature_columns

    # ----- single-step prediction -------------------------------------------
    def _predict_one(self, feature_row: np.ndarray) -> float:
        """
        Predict PM2.5 for a single 1-D feature vector.

        Parameters
        ----------
        feature_row : np.ndarray, shape (n_features,)

        Returns
        -------
        float  — predicted PM2.5 (clipped ≥ 0).
        """
        X = feature_row.reshape(1, -1).astype(np.float32)
        X_scaled = self.scaler.transform(X)

        if self.model_type == "dnn":
            pred = float(self.model.predict(X_scaled, verbose=0).flatten()[0])
        else:
            pred = float(self.model.predict(X_scaled)[0])

        return max(pred, 0.0)

    # ----- recursive multi-step forecast ------------------------------------
    def forecast(
        self,
        seed_row: pd.Series,
        horizon: int = FORECAST_HORIZON,
    ) -> pd.DataFrame:
        """
        Recursively forecast PM2.5 for the next *horizon* hours.

        At each step the predicted PM2.5 value is inserted back into the
        feature vector (and time features are advanced by one hour) before
        predicting the next step.

        Parameters
        ----------
        seed_row : pd.Series
            The most-recent feature row.  Must contain all columns listed in
            ``self.feature_columns`` plus ``timestamp`` (or equivalent).
        horizon : int
            Number of hourly steps to forecast (default: 72).

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``predicted_pm2_5``,
            ``predicted_aqi_category``.
        """
        feature_cols = self.feature_columns
        current = seed_row.copy()
        seed_snapshot = seed_row.copy()       # keep original for diurnal base

        # Determine the starting timestamp
        if DATETIME_COLUMN in current.index:
            ts = pd.Timestamp(current[DATETIME_COLUMN])
        else:
            ts = pd.Timestamp.now(tz="UTC")

        predictions: List[Dict[str, Any]] = []

        for step in range(1, horizon + 1):
            # --- advance timestamp first so features reflect the target hour ---
            ts = ts + pd.Timedelta(hours=1)

            # --- update time-based features ---
            if "hour" in current.index:
                current["hour"] = ts.hour
            if "day" in current.index:
                current["day"] = ts.day
            if "month" in current.index:
                current["month"] = ts.month

            # --- apply diurnal variation to weather / co-pollutant features ---
            _update_weather_features(current, ts.hour, seed_snapshot)

            # --- build the feature vector in the correct column order ---
            feat_vec = np.array(
                [float(current[c]) for c in feature_cols], dtype=np.float32
            )

            # --- predict this step ---
            pm25_pred = self._predict_one(feat_vec)

            # --- classify ---
            category = classify_aqi(pm25_pred)

            predictions.append(
                {
                    "timestamp": ts,
                    "predicted_pm2_5": round(pm25_pred, 2),
                    "predicted_aqi_category": category,
                }
            )

            # --- feed prediction back into the feature vector for next step ---
            if "pm2_5" in current.index:
                current["pm2_5"] = pm25_pred

            # Update rolling / derived features (simple approximations)
            if "rolling_mean_3" in current.index:
                # Exponential moving average as lightweight proxy
                alpha = 2 / (3 + 1)
                current["rolling_mean_3"] = (
                    alpha * pm25_pred
                    + (1 - alpha) * float(current["rolling_mean_3"])
                )
            if "aqi_change_rate" in current.index:
                prev_pm25 = feat_vec[feature_cols.index("pm2_5")] if "pm2_5" in feature_cols else pm25_pred
                if prev_pm25 > 0:
                    current["aqi_change_rate"] = (
                        (pm25_pred - prev_pm25) / prev_pm25
                    ) * 100
                else:
                    current["aqi_change_rate"] = 0.0

        forecast_df = pd.DataFrame(predictions)
        log.info(
            f"Forecast complete — {len(forecast_df)} hourly steps "
            f"({forecast_df['predicted_aqi_category'].value_counts().to_dict()})"
        )
        return forecast_df


# ===========================================================================
# 3.  Latest Features from Hopsworks
# ===========================================================================
def fetch_latest_features_from_hopsworks() -> pd.DataFrame:
    """
    Pull the latest rows from the Hopsworks Feature Store and return
    the most recent feature row as a single-row DataFrame.
    """
    project = login_to_hopsworks()
    fs = project.get_feature_store()

    try:
        fv = fs.get_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
        )
    except Exception:
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )
        fv = fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=fg.select_all(),
        )

    df = fv.get_batch_data()

    if df.empty:
        raise RuntimeError("Feature view returned no data")

    # Sort by timestamp descending → take the latest row
    if DATETIME_COLUMN in df.columns:
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], utc=True)
        df = df.sort_values(DATETIME_COLUMN, ascending=False)

    log.info(f"Latest features fetched — {df.shape[0]} total rows")
    return df


def fetch_latest_features_local(
    path: str = "",
) -> pd.DataFrame:
    """Fallback: load from a local file and return sorted by timestamp."""
    from src.utils import load_dataframe

    if not path:
        path = str(DATA_DIR / "features_latest.parquet")
    df = load_dataframe(path)
    if DATETIME_COLUMN in df.columns:
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], utc=True)
        df = df.sort_values(DATETIME_COLUMN, ascending=False)
    return df


# ===========================================================================
# 4.  End-to-End Prediction Pipeline
# ===========================================================================
def run_prediction_pipeline(
    use_hopsworks: bool = True,
    horizon: int = FORECAST_HORIZON,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Full prediction pipeline: load model → fetch features → forecast.

    Parameters
    ----------
    use_hopsworks : bool
        If True, attempt to load model from the Hopsworks Model Registry
        and features from the Feature Store.
    horizon : int
        Number of hours to forecast (default: 72).
    save_output : bool
        If True, persist the forecast DataFrame to ``data/``.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``predicted_pm2_5``,
        ``predicted_aqi_category``.
    """
    log.info("=" * 60)
    log.info("PREDICTION PIPELINE — START")
    log.info(f"City: {CITY}  |  Horizon: {horizon}h")
    log.info("=" * 60)

    # --- 1. Initialise predictor (downloads model if needed) ---------------
    predictor = PM25Predictor(use_registry=use_hopsworks)

    # --- 2. Fetch latest features ------------------------------------------
    if use_hopsworks:
        try:
            features_df = fetch_latest_features_from_hopsworks()
        except Exception as exc:
            log.warning(
                f"Hopsworks feature fetch failed ({exc}); "
                "falling back to local data"
            )
            features_df = fetch_latest_features_local()
    else:
        features_df = fetch_latest_features_local()

    if features_df.empty:
        raise RuntimeError("No feature data available for prediction")

    # Take the latest single row as the seed
    seed_row: pd.Series = features_df.iloc[0]
    log.info(
        f"Seed timestamp: {seed_row.get(DATETIME_COLUMN, 'N/A')}  |  "
        f"Seed PM2.5: {seed_row.get('pm2_5', 'N/A')}"
    )

    # --- 3. Recursive forecast ---------------------------------------------
    forecast_df = predictor.forecast(seed_row, horizon=horizon)

    # --- 4. Persist --------------------------------------------------------
    if save_output:
        out_path = DATA_DIR / "forecast_latest.csv"
        save_dataframe(forecast_df, out_path, index=False)

    log.info("=" * 60)
    log.info("PREDICTION PIPELINE — END")
    log.info("=" * 60)
    return forecast_df


# ===========================================================================
# 5.  Convenience Helpers
# ===========================================================================
_predictor: Optional[PM25Predictor] = None


def get_predictor(use_registry: bool = True) -> PM25Predictor:
    """Return a module-level singleton predictor."""
    global _predictor
    if _predictor is None:
        _predictor = PM25Predictor(use_registry=use_registry)
    return _predictor


def predict_pm25(
    features: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Quick-call: predict PM2.5 for an arbitrary feature matrix.

    Returns a 1-D array of predicted PM2.5 values.
    """
    p = get_predictor(use_registry=False)
    if isinstance(features, pd.DataFrame):
        features = features[p.feature_columns].values.astype(np.float32)
    X_scaled = p.scaler.transform(features)
    if p.model_type == "dnn":
        return np.clip(p.model.predict(X_scaled, verbose=0).flatten(), 0, None)
    return np.clip(p.model.predict(X_scaled), 0, None)


if __name__ == "__main__":
    forecast = run_prediction_pipeline(use_hopsworks=True, horizon=72)
    print(forecast.to_string(index=False))

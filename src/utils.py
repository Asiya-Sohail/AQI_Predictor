"""
utils.py — Shared utility helpers for the AQI Forecasting System.

Provides logging setup, data I/O, AQI category mapping, and metric helpers.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    AQI_CATEGORIES,
    LOG_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    MODEL_DIR,
)


# ===========================================================================
# 1. Logging
# ===========================================================================
def setup_logger(name: str = "aqi_system"):  # type: ignore[no-untyped-def]
    """Configure and return a Loguru logger instance."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        colorize=True,
    )
    logger.add(
        LOG_DIR / f"{name}.log",
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )
    return logger


log = setup_logger()


# ===========================================================================
# 2. AQI Category Mapping
# ===========================================================================
def get_aqi_category(value: float) -> str:
    """Return the descriptive AQI category for a numeric value."""
    for (low, high), label in AQI_CATEGORIES.items():
        if low <= value <= high:
            return label
    return "Beyond Index"


def add_aqi_category_column(df: pd.DataFrame, col: str = "aqi") -> pd.DataFrame:
    """Append an ``aqi_category`` column to *df* based on *col*."""
    df = df.copy()
    df["aqi_category"] = df[col].apply(get_aqi_category)
    return df


# ===========================================================================
# 3. Evaluation Metrics
# ===========================================================================
def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Return a dict of standard regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(
            np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
        ),
    }


def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print metrics to the configured logger."""
    tag = f"[{prefix}] " if prefix else ""
    for name, value in metrics.items():
        log.info(f"{tag}{name.upper()}: {value:.4f}")


# ===========================================================================
# 4. Model Persistence
# ===========================================================================
def save_sklearn_model(model: Any, filename: str) -> Path:
    """Persist a scikit-learn model to the models directory."""
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    log.info(f"Scikit-learn model saved → {path}")
    return path


def load_sklearn_model(filename: str) -> Any:
    """Load a scikit-learn model from the models directory."""
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    log.info(f"Scikit-learn model loaded ← {path}")
    return model


def save_keras_model(model: Any, name: str) -> Path:
    """Save a Keras / TF model in native Keras format."""
    if not name.endswith(".keras"):
        name = name + ".keras"
    path = MODEL_DIR / name
    model.save(path)
    log.info(f"Keras model saved \u2192 {path}")
    return path


def load_keras_model(name: str) -> Any:
    """Load a Keras / TF model from .keras file."""
    import tensorflow as tf

    if not name.endswith(".keras"):
        name = name + ".keras"
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Keras model not found: {path}")
    model = tf.keras.models.load_model(path)
    log.info(f"Keras model loaded ← {path}")
    return model


# ===========================================================================
# 5. Data I/O Helpers
# ===========================================================================
def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
) -> None:
    """Save a DataFrame to CSV or Parquet based on file extension."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.suffix == ".parquet":
        df.to_parquet(filepath, index=index)
    else:
        df.to_csv(filepath, index=index)
    log.info(f"DataFrame ({df.shape}) saved → {filepath}")


def load_dataframe(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a DataFrame from CSV or Parquet."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    log.info(f"DataFrame ({df.shape}) loaded ← {filepath}")
    return df


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Serialize a dict to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    log.info(f"JSON saved → {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Deserialize JSON file into a dict."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# 6. Data Validation
# ===========================================================================
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list,
    name: str = "DataFrame",
) -> bool:
    """Raise if required columns are missing from *df*."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    log.info(f"{name} validated — {len(df)} rows, {len(df.columns)} cols")
    return True


def clip_aqi_values(
    df: pd.DataFrame,
    col: str = "aqi",
    lower: float = 0.0,
    upper: float = 500.0,
) -> pd.DataFrame:
    """Clip AQI values to the valid [0, 500] range."""
    df = df.copy()
    df[col] = df[col].clip(lower=lower, upper=upper)
    return df

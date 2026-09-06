"""
training_pipeline.py — Model training, evaluation, and registry pipeline.

Trains three models on historical AQI feature data:
  1. Random Forest Regressor  (scikit-learn)  — tree-based baseline
  2. Ridge Regression         (scikit-learn)  — regularised linear baseline
  3. Deep Neural Network      (TensorFlow)    — fully-connected DNN

Target variable: ``pm2_5``

All models are evaluated with RMSE, MAE, and R² score.  The best model
(lowest RMSE) is automatically selected and registered in the Hopsworks
Model Registry.  A ``training_report.json`` file is saved with full metrics
for every model.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.callbacks import (  # type: ignore[import-untyped]
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.layers import BatchNormalization, Dense, Dropout, Input  # type: ignore[import-untyped]
from keras.models import Model  # type: ignore[import-untyped]
from keras.optimizers import Adam  # type: ignore[import-untyped]
from keras.regularizers import l2  # type: ignore[import-untyped]

from src.config import (
    CITY_COLUMN,
    DATA_DIR,
    DATETIME_COLUMN,
    DNN_PARAMS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODEL_DIR,
    MODEL_REGISTRY_NAME,
    RANDOM_STATE,
    RF_PARAMS,
    RIDGE_PARAMS,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.hopsworks_client import login_to_hopsworks
from src.utils import (
    compute_regression_metrics,
    load_dataframe,
    load_json,
    log,
    log_metrics,
    save_json,
    save_keras_model,
    save_sklearn_model,
)

# Directory for training output plots
_TRAINING_OUTPUT_DIR = DATA_DIR / "training_outputs"
_TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Training-specific constants
# ---------------------------------------------------------------------------
TRAINING_TARGET = "pm2_5"

# Columns to exclude from features (non-numeric / target / identifiers)
_NON_FEATURE_COLS = {DATETIME_COLUMN, CITY_COLUMN, TRAINING_TARGET, "aqi"}


# ===========================================================================
# 1. Data Loading from Hopsworks
# ===========================================================================
def load_training_data_from_hopsworks() -> pd.DataFrame:
    """Pull the latest training set from the Hopsworks Feature Store."""
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
    log.info(f"Training data loaded from Hopsworks: {df.shape}")
    return df


def load_training_data_local(
    path: str = "",
) -> pd.DataFrame:
    """Fallback: load features from a local Parquet/CSV file."""
    if path:
        return load_dataframe(path)

    candidate_paths = [
        DATA_DIR / "features_history.parquet",
        DATA_DIR / "features_latest.parquet",
        *sorted(DATA_DIR.glob("backfill_*.parquet")),
    ]

    frames: List[pd.DataFrame] = []
    used_paths: List[str] = []
    for p in candidate_paths:
        if p.exists():
            try:
                df_part = load_dataframe(p)
                if not df_part.empty:
                    frames.append(df_part)
                    used_paths.append(str(p))
            except Exception as exc:
                log.warning(f"Skipping unreadable local training file '{p}': {exc}")

    if not frames:
        return load_dataframe(DATA_DIR / "features_latest.parquet")

    df = pd.concat(frames, ignore_index=True)
    if {DATETIME_COLUMN, CITY_COLUMN}.issubset(df.columns):
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], utc=True)
        df = df.sort_values([CITY_COLUMN, DATETIME_COLUMN])
        df = df.drop_duplicates(
            subset=[DATETIME_COLUMN, CITY_COLUMN],
            keep="last",
        )

    df = df.reset_index(drop=True)
    log.info(
        f"Local training data assembled from {len(used_paths)} file(s) → {df.shape}"
    )
    return df


# ===========================================================================
# 2. Data Preparation
# ===========================================================================
def prepare_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Clean, split, and scale the data.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Scaled feature arrays and raw target arrays.
    scaler : StandardScaler
        Fitted scaler (for inference-time re-use).
    feature_cols : list[str]
        Ordered list of feature column names.
    """
    # Drop rows where the target is missing
    df = df.dropna(subset=[TRAINING_TARGET]).copy()

    # Determine feature columns (exclude identifiers, target, and aqi)
    feature_cols = sorted(
        c for c in df.columns if c not in _NON_FEATURE_COLS
    )

    # Drop any remaining rows with NaN in feature columns
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values.astype(np.float32)
    y = df[TRAINING_TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log.info(
        f"Data prepared — Train: {X_train.shape}, Test: {X_test.shape}, "
        f"Target: {TRAINING_TARGET}"
    )
    return X_train, X_test, y_train, y_test, scaler, feature_cols


# ===========================================================================
# 3. Random Forest Regressor
# ===========================================================================
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train a Random Forest regressor and return model + metrics."""
    log.info("Training Random Forest …")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    log_metrics(metrics, prefix="RF")

    save_sklearn_model(rf, "random_forest_pm25.joblib")
    return rf, metrics


# ===========================================================================
# 4. Ridge Regression
# ===========================================================================
def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Ridge, Dict[str, float]]:
    """Train a Ridge regressor and return model + metrics."""
    log.info("Training Ridge Regression …")
    ridge = Ridge(**RIDGE_PARAMS)
    ridge.fit(X_train, y_train)

    y_pred = ridge.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    log_metrics(metrics, prefix="Ridge")

    save_sklearn_model(ridge, "ridge_pm25.joblib")
    return ridge, metrics


# ===========================================================================
# 5. TensorFlow Deep Neural Network
# ===========================================================================
def build_dnn_model(n_features: int) -> Model:
    """
    Build an optimised fully-connected Deep Neural Network.

    Architecture
    ------------
    Input → [Dense(L2) → BatchNorm → ReLU → Dropout] × N → Dense(1)

    Optimisations over a vanilla DNN:
    - **L2 kernel regularisation** on every hidden layer to reduce
      over-fitting (weight decay = 1e-4).
    - **Dropout** after each hidden block (rate from ``DNN_PARAMS``).
    - **Batch Normalisation** for faster, more stable convergence.
    - **ReduceLROnPlateau + EarlyStopping + ModelCheckpoint** callbacks
      (applied in ``train_dnn``).
    """
    hidden_units: List[int] = DNN_PARAMS["hidden_units"]
    dropout_rate: float = DNN_PARAMS["dropout_rate"]
    activation: str = DNN_PARAMS["activation"]
    weight_decay: float = DNN_PARAMS.get("weight_decay", 1e-4)

    inputs = Input(shape=(n_features,), name="features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = Dense(
            units,
            activation=None,
            kernel_regularizer=l2(weight_decay),
            name=f"dense_{i}",
        )(x)
        x = BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.activations.get(activation)(x)
        x = Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = Dense(1, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="DNN_PM25")
    model.compile(
        optimizer=Adam(learning_rate=DNN_PARAMS["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    model.summary(print_fn=log.info)
    return model


def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Optional[Model], Dict[str, float], Optional[tf.keras.callbacks.History]]:
    """
    Train an optimised TensorFlow DNN and return model + metrics + history.

    Callback stack
    --------------
    1. **EarlyStopping** — halts training when ``val_loss`` stops improving
       for ``DNN_PARAMS["patience"]`` epochs and restores the best weights.
    2. **ReduceLROnPlateau** — halves the learning rate when ``val_loss``
       stalls for ``patience // 3`` epochs (floor 1e-6).
    3. **ModelCheckpoint** — persists the best weights to disk so they
       survive OOM or interruption.

    Returns ``(None, {}, None)`` if training fails.
    """
    log.info("Training TensorFlow DNN …")
    try:
        model = build_dnn_model(n_features=X_train.shape[1])

        checkpoint_path = str(MODEL_DIR / "dnn_best_weights.weights.h5")

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=DNN_PARAMS["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(1, DNN_PARAMS["patience"] // 3),
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SIZE,
            epochs=DNN_PARAMS["epochs"],
            batch_size=DNN_PARAMS["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        # Restore best checkpoint weights (belt-and-suspenders)
        if Path(checkpoint_path).exists():
            model.load_weights(checkpoint_path)
            log.info("Best checkpoint weights restored")

        y_pred = model.predict(X_test, verbose=0).flatten()
        metrics = compute_regression_metrics(y_test, y_pred)
        log_metrics(metrics, prefix="DNN")

        save_keras_model(model, "dnn_pm25_model")
        return model, metrics, history

    except Exception as exc:
        log.error(f"DNN training failed: {exc}")
        return None, {}, None


# ===========================================================================
# 6. Model Selection
# ===========================================================================
def select_best_model(
    all_metrics: Dict[str, Dict[str, float]],
) -> str:
    """
    Return the model key with the lowest RMSE.

    Parameters
    ----------
    all_metrics : dict
        Mapping of model name → metrics dict (must contain ``"rmse"``).

    Returns
    -------
    str
        Key of the best model (e.g. ``"rf"``, ``"ridge"``, ``"dnn"``).
    """
    # Filter out models that returned empty metrics (training failures)
    valid = {k: v for k, v in all_metrics.items() if v}
    if not valid:
        raise RuntimeError("No models produced valid metrics — cannot select best.")

    best_key = min(valid, key=lambda k: valid[k]["rmse"])
    log.info(
        f"Best model → {best_key} "
        f"(RMSE={valid[best_key]['rmse']:.4f}, "
        f"MAE={valid[best_key]['mae']:.4f}, "
        f"R²={valid[best_key]['r2']:.4f})"
    )
    return best_key


# ===========================================================================
# 7. Hopsworks Model Registry
# ===========================================================================
def register_model_hopsworks(
    model_name: str,
    metrics: Dict[str, float],
    model_dir: str,
) -> None:
    """Register the best model in the Hopsworks Model Registry."""
    project = login_to_hopsworks()
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(
        name=MODEL_REGISTRY_NAME,
        metrics={
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        },
        description=f"PM2.5 forecast model ({model_name})",
    )
    hw_model.save(model_dir)
    log.info(f"Model '{model_name}' registered in Hopsworks Model Registry")


# ===========================================================================
# 8. Metrics Persistence
# ===========================================================================
def save_training_report(
    best_model: str,
    all_metrics: Dict[str, Dict[str, float]],
) -> Path:
    """
    Persist a JSON training report with per-model metrics and the winner.

    Returns the path to the saved report.
    """
    report = {
        "target_variable": TRAINING_TARGET,
        "best_model": best_model,
    }
    for name, met in all_metrics.items():
        report[f"{name}_metrics"] = met

    report_path = MODEL_DIR / "training_report.json"
    save_json(report, report_path)
    log.info(f"Training report saved → {report_path}")
    return report_path


# ===========================================================================
# 9. Pipeline Runner
# ===========================================================================
# Maps model key → saved artefact path (relative to MODEL_DIR)
_MODEL_ARTEFACTS = {
    "rf": "random_forest_pm25.joblib",
    "ridge": "ridge_pm25.joblib",
    "dnn": "dnn_pm25_model",
}


def _can_reuse_existing_training_outputs() -> bool:
    """Return True when required local artefacts exist for inference reuse."""
    report_path = MODEL_DIR / "training_report.json"
    scaler_path = MODEL_DIR / "scaler.joblib"
    feature_cols_path = MODEL_DIR / "feature_columns.json"
    return report_path.exists() and scaler_path.exists() and feature_cols_path.exists()


def _load_existing_training_summary() -> Dict[str, Any]:
    """Load summary from an existing training report for fallback mode."""
    report = load_json(MODEL_DIR / "training_report.json")
    summary: Dict[str, Any] = {
        "best_model": report.get("best_model"),
        "reused_existing_artifacts": True,
    }
    for key in ("rf_metrics", "ridge_metrics", "dnn_metrics"):
        if key in report:
            summary[key] = report[key]
    return summary


# ===========================================================================
# 9a. Model Comparison Visualization
# ===========================================================================
def plot_model_comparison(
    all_metrics: Dict[str, Dict[str, float]],
    dnn_history: Optional[tf.keras.callbacks.History] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Generate and save model comparison charts.

    Produces
    --------
    1. ``model_comparison.html`` — interactive Plotly grouped bar chart
       comparing RMSE, MAE, and R² for every trained model.
    2. ``dnn_training_curves.png`` — training vs. validation loss over
       epochs (only when *dnn_history* is provided).
    3. ``model_comparison.png`` — static fallback of the bar chart.
    """
    out = save_dir or _TRAINING_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    valid = {k: v for k, v in all_metrics.items() if v}
    if not valid:
        log.warning("No valid metrics to plot.")
        return

    model_names = list(valid.keys())
    display_names = {"rf": "Random Forest", "ridge": "Ridge", "dnn": "DNN"}
    labels = [display_names.get(m, m) for m in model_names]

    rmse_vals = [valid[m]["rmse"] for m in model_names]
    mae_vals = [valid[m]["mae"] for m in model_names]
    r2_vals = [valid[m]["r2"] for m in model_names]

    # ── 1. Interactive Plotly grouped bar chart ────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RMSE", x=labels, y=rmse_vals,
        marker_color="#ef553b",
        text=[f"{v:.3f}" for v in rmse_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="MAE", x=labels, y=mae_vals,
        marker_color="#636efa",
        text=[f"{v:.3f}" for v in mae_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="R²", x=labels, y=r2_vals,
        marker_color="#00cc96",
        text=[f"{v:.4f}" for v in r2_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Model Comparison — RMSE / MAE / R²",
        barmode="group",
        template="plotly_dark",
        yaxis_title="Score",
        height=450,
        margin=dict(t=60, b=40),
    )
    fig.write_html(str(out / "model_comparison.html"))
    log.info(f"Interactive comparison chart → {out / 'model_comparison.html'}")

    # ── 2. Static matplotlib bar chart ────────────────────────────────────
    x_pos = np.arange(len(labels))
    width = 0.25

    fig_m, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x_pos - width, rmse_vals, width, label="RMSE", color="#ef553b")
    bars2 = ax.bar(x_pos, mae_vals, width, label="MAE", color="#636efa")
    bars3 = ax.bar(x_pos + width, r2_vals, width, label="R²", color="#00cc96")

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — RMSE / MAE / R²", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig_m.tight_layout()
    fig_m.savefig(out / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    log.info(f"Static comparison chart → {out / 'model_comparison.png'}")

    # ── 3. DNN training curves ────────────────────────────────────────────
    if dnn_history is not None:
        hist = dnn_history.history
        epochs = range(1, len(hist["loss"]) + 1)

        fig_h, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(epochs, hist["loss"], label="Train Loss", color="#636efa")
        axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#ef553b")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_title("DNN Training & Validation Loss", fontweight="bold")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # MAE
        axes[1].plot(epochs, hist["mae"], label="Train MAE", color="#636efa")
        axes[1].plot(epochs, hist["val_mae"], label="Val MAE", color="#ef553b")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("DNN Training & Validation MAE", fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig_h.tight_layout()
        fig_h.savefig(out / "dnn_training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig_h)
        log.info(f"DNN training curves → {out / 'dnn_training_curves.png'}")


# ===========================================================================
# 10. Pipeline Runner
# ===========================================================================
def run_training_pipeline(use_hopsworks: bool = True) -> Dict[str, Any]:
    """
    End-to-end training pipeline.

    1.  Load features from Hopsworks (or local fallback).
    2.  Prepare data (clean, split, scale).
    3.  Train Random Forest, Ridge, and DNN models.
    4.  Evaluate each with RMSE / MAE / R².
    5.  Auto-select the best model (lowest RMSE).
    6.  Generate model comparison visualisation.
    7.  Save scaler, feature columns, and training report.
    8.  Register the best model in Hopsworks Model Registry.

    Parameters
    ----------
    use_hopsworks : bool
        If True, attempt to load data from Hopsworks first.

    Returns
    -------
    dict
        Summary containing best model name and all metrics.
    """
    log.info("=== Training Pipeline START ===")

    # ── 1. Load data ──────────────────────────────────────────────────────
    if use_hopsworks:
        try:
            df = load_training_data_from_hopsworks()
        except Exception as exc:
            log.warning(
                f"Hopsworks load failed ({exc}), falling back to local data"
            )
            df = load_training_data_local()
    else:
        df = load_training_data_local()

    if len(df) < 5:
        msg = (
            f"Insufficient training rows ({len(df)}). "
            "Need at least 5 rows to retrain reliably."
        )
        if _can_reuse_existing_training_outputs():
            log.warning(f"{msg} Reusing existing model artefacts in models/.")
            return _load_existing_training_summary()
        raise ValueError(
            f"{msg} No existing model artefacts found in {MODEL_DIR} to reuse."
        )

    # ── 2. Prepare ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_data(df)

    # Save scaler for inference
    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info(f"Scaler saved → {scaler_path}")

    # Save ordered feature column names
    save_json(
        {"feature_columns": feature_cols, "target": TRAINING_TARGET},
        MODEL_DIR / "feature_columns.json",
    )

    # ── 3. Train all models ───────────────────────────────────────────────
    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test,
    )
    ridge_model, ridge_metrics = train_ridge(
        X_train, y_train, X_test, y_test,
    )
    dnn_model, dnn_metrics, dnn_history = train_dnn(
        X_train, y_train, X_test, y_test,
    )

    all_metrics: Dict[str, Dict[str, float]] = {
        "rf": rf_metrics,
        "ridge": ridge_metrics,
        "dnn": dnn_metrics,
    }

    # ── 4. Select best model ─────────────────────────────────────────────
    best = select_best_model(all_metrics)

    # ── 5. Model comparison visualization ─────────────────────────────────
    try:
        plot_model_comparison(all_metrics, dnn_history=dnn_history)
    except Exception as exc:
        log.warning(f"Comparison visualization failed: {exc}")

    # ── 6. Save training report ───────────────────────────────────────────
    save_training_report(best, all_metrics)

    # ── 7. Register best model in Hopsworks ───────────────────────────────
    best_metrics = all_metrics[best]
    best_dir = str(MODEL_DIR / _MODEL_ARTEFACTS[best])

    try:
        register_model_hopsworks(best, best_metrics, best_dir)
    except Exception as exc:
        log.error(f"Model registration failed: {exc}")

    log.info("=== Training Pipeline END ===")
    return {"best_model": best, **all_metrics}


if __name__ == "__main__":
    run_training_pipeline(use_hopsworks=True)

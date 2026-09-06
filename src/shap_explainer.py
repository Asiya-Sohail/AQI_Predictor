"""
shap_explainer.py — SHAP-based model explainability for PM2.5 predictions.

Supports
--------
- **RandomForest / Ridge** → ``shap.TreeExplainer`` (RF) or
  ``shap.LinearExplainer`` (Ridge)
- **TensorFlow DNN**       → ``shap.GradientExplainer``

Outputs
-------
- ``shap_summary.png``           Beeswarm summary plot
- ``shap_feature_importance.png``  Mean |SHAP| bar chart
- ``shap_feature_ranking.csv``   Full ranked feature table
- ``shap_report.json``           Top-5 features + sample count

All plots are saved to ``data/shap_outputs/`` by default.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import DATA_DIR, MODEL_DIR
from src.utils import (
    load_json,
    load_keras_model,
    load_sklearn_model,
    log,
    save_json,
)

import joblib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_ARTEFACTS = {
    "rf": "random_forest_pm25.joblib",
    "ridge": "ridge_pm25.joblib",
    "dnn": "dnn_pm25_model",
}
_DPI = 150
_MAX_SAMPLES = 500          # cap for SHAP computation
_TOP_N = 5                  # number of top features to return
_BACKGROUND_SAMPLES = 100   # background set size for Gradient/Kernel explainer


# ===========================================================================
# 1.  Model + Metadata Loader
# ===========================================================================
def _load_model_and_type(
    model_dir: Path,
) -> Tuple[Any, str, List[str]]:
    """
    Load the best model, its type key, and feature column names.

    Returns
    -------
    model : fitted estimator or Keras Model
    model_type : str  — ``"rf"``, ``"ridge"``, or ``"dnn"``
    feature_names : list[str]
    """
    report = load_json(model_dir / "training_report.json")
    model_type: str = report["best_model"]

    # DNN's GradientExplainer is incompatible with modern Keras;
    # fall back to Ridge (LinearExplainer) which is reliable.
    if model_type == "dnn":
        log.warning(
            "Best model is DNN but SHAP GradientExplainer has Keras "
            "compatibility issues — falling back to Ridge for SHAP."
        )
        model_type = "ridge"

    artefact = _MODEL_ARTEFACTS.get(model_type)
    if artefact is None:
        raise ValueError(f"Unknown model type in training report: {model_type}")

    if model_type == "dnn":
        model = load_keras_model(artefact)
    else:
        model = load_sklearn_model(artefact)

    meta = load_json(model_dir / "feature_columns.json")
    feature_names: List[str] = meta["feature_columns"]

    log.info(f"Loaded model type='{model_type}' with {len(feature_names)} features")
    return model, model_type, feature_names


# ===========================================================================
# 2.  Explainer Factory
# ===========================================================================
def _build_explainer(
    model: Any,
    model_type: str,
    background: np.ndarray,
) -> Any:
    """
    Return the appropriate SHAP explainer for the given model type.

    - RF          → ``shap.TreeExplainer``
    - Ridge       → ``shap.LinearExplainer``
    - DNN (Keras) → ``shap.GradientExplainer``
    """
    if model_type == "rf":
        explainer = shap.TreeExplainer(model)
        log.info("Built SHAP TreeExplainer (RandomForest)")
    elif model_type == "ridge":
        explainer = shap.LinearExplainer(model, background)
        log.info("Built SHAP LinearExplainer (Ridge)")
    elif model_type == "dnn":
        explainer = shap.GradientExplainer(model, background)
        log.info("Built SHAP GradientExplainer (TensorFlow DNN)")
    else:
        raise ValueError(f"No SHAP explainer strategy for model type: {model_type}")
    return explainer


# ===========================================================================
# 3.  SHAP Value Computation
# ===========================================================================
def compute_shap_values(
    explainer: Any,
    model_type: str,
    X: np.ndarray,
    feature_names: List[str],
) -> shap.Explanation:
    """
    Compute SHAP values and return a ``shap.Explanation`` object.

    Handles differences between Tree / Linear / Gradient explainer
    return types so callers always get a uniform ``Explanation``.
    """
    raw = explainer.shap_values(X)

    # GradientExplainer may return a list (one array per output);
    # take the first (single-output regression).
    if isinstance(raw, list):
        raw = raw[0]

    explanation = shap.Explanation(
        values=raw,
        base_values=np.full(raw.shape[0], _base_value(explainer, model_type)),
        data=X,
        feature_names=feature_names,
    )
    log.info(f"SHAP values computed — shape {raw.shape}")
    return explanation


def _base_value(explainer: Any, model_type: str) -> float:
    """Extract a scalar expected / base value from the explainer."""
    ev = getattr(explainer, "expected_value", 0.0)
    if isinstance(ev, np.ndarray):
        return float(ev.mean())
    if isinstance(ev, (list, tuple)):
        return float(ev[0])
    return float(ev)


# ===========================================================================
# 4.  Plot Generators
# ===========================================================================
def plot_summary(
    explanation: shap.Explanation,
    save_path: Path,
) -> None:
    """Beeswarm summary plot."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        show=False,
    )
    plt.title("SHAP Summary — PM2.5 Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Summary plot saved → {save_path}")


def plot_feature_importance_bar(
    explanation: shap.Explanation,
    save_path: Path,
) -> None:
    """Horizontal bar chart of mean |SHAP| per feature."""
    mean_abs = np.abs(explanation.values).mean(axis=0)
    names = explanation.feature_names
    order = np.argsort(mean_abs)

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    ax.barh(
        [names[i] for i in order],
        mean_abs[order],
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        "Feature Importance — Mean |SHAP|",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Feature importance bar chart saved → {save_path}")


# ===========================================================================
# 5.  Feature Ranking
# ===========================================================================
def rank_features(
    explanation: shap.Explanation,
) -> pd.DataFrame:
    """
    Return a DataFrame of features ranked by mean |SHAP value|.

    Columns: ``rank``, ``feature``, ``mean_abs_shap``.
    """
    mean_abs = np.abs(explanation.values).mean(axis=0)
    ranking = (
        pd.DataFrame(
            {"feature": explanation.feature_names, "mean_abs_shap": mean_abs}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking.index += 1
    ranking.index.name = "rank"
    return ranking


def top_n_features(
    explanation: shap.Explanation,
    n: int = _TOP_N,
) -> List[str]:
    """Return the top *n* feature names by mean |SHAP|."""
    ranking = rank_features(explanation)
    return ranking["feature"].head(n).tolist()


# ===========================================================================
# 6.  Main Explainer Class
# ===========================================================================
class ShapExplainer:
    """
    High-level SHAP explainability wrapper.

    Supports RandomForest, Ridge, and TensorFlow DNN models.
    Loads the best model from ``MODEL_DIR`` automatically, or accepts a
    pre-loaded model via constructor arguments.

    Usage
    -----
    >>> explainer = ShapExplainer()
    >>> result = explainer.run(X_scaled)
    >>> print(result["top_features"])
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_type: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.model_dir = model_dir or MODEL_DIR

        if model is not None and model_type is not None and feature_names is not None:
            self._model = model
            self._model_type = model_type
            self._feature_names = feature_names
        else:
            self._model, self._model_type, self._feature_names = (
                _load_model_and_type(self.model_dir)
            )

        self._explainer: Optional[Any] = None
        self._scaler = joblib.load(self.model_dir / "scaler.joblib")
        log.info(
            f"ShapExplainer ready — model={self._model_type}, "
            f"features={len(self._feature_names)}"
        )

    # ------------------------------------------------------------------
    def _get_explainer(self, background: np.ndarray) -> Any:
        """Build (or return cached) SHAP explainer."""
        if self._explainer is None:
            self._explainer = _build_explainer(
                self._model, self._model_type, background
            )
        return self._explainer

    # ------------------------------------------------------------------
    def run(
        self,
        X: np.ndarray,
        save_dir: Optional[Path] = None,
        max_samples: int = _MAX_SAMPLES,
    ) -> Dict[str, Any]:
        """
        End-to-end SHAP explanation pipeline.

        Steps
        -----
        1. Sub-sample *X* if necessary.
        2. Build background set and SHAP explainer.
        3. Compute SHAP values.
        4. Generate summary plot + feature importance bar chart.
        5. Save ranking CSV and report JSON.
        6. Return dict with ``top_features``, ``output_dir``, ``n_samples``.

        Parameters
        ----------
        X : np.ndarray
            Scaled feature matrix (same scaling used during training).
        save_dir : Path, optional
            Override output directory (default ``data/shap_outputs``).
        max_samples : int
            Maximum rows to process (default 500).

        Returns
        -------
        dict  with keys ``top_features``, ``output_dir``, ``n_samples``.
        """
        out = save_dir or DATA_DIR / "shap_outputs"
        out.mkdir(parents=True, exist_ok=True)

        log.info("=" * 60)
        log.info("SHAP EXPLAINER — START")
        log.info(f"Model: {self._model_type}  |  Input shape: {X.shape}")
        log.info("=" * 60)

        # --- sub-sample ---
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X = X[idx]

        # --- background set ---
        bg_size = min(_BACKGROUND_SAMPLES, len(X))
        background = X[:bg_size]

        # --- build explainer & compute ---
        explainer = self._get_explainer(background)
        explanation = compute_shap_values(
            explainer, self._model_type, X, self._feature_names
        )

        # --- plots ---
        plot_summary(explanation, save_path=out / "shap_summary.png")
        plot_feature_importance_bar(explanation, save_path=out / "shap_feature_importance.png")

        # --- ranking ---
        ranking = rank_features(explanation)
        ranking.to_csv(out / "shap_feature_ranking.csv")
        log.info(f"Feature ranking saved → {out / 'shap_feature_ranking.csv'}")

        top5 = top_n_features(explanation, n=_TOP_N)

        save_json(
            {
                "model_type": self._model_type,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "top_features": top5,
            },
            out / "shap_report.json",
        )

        log.info(f"Top {_TOP_N} features: {top5}")
        log.info("=" * 60)
        log.info("SHAP EXPLAINER — END")
        log.info("=" * 60)

        return {
            "top_features": top5,
            "output_dir": str(out),
            "n_samples": int(X.shape[0]),
        }


# ===========================================================================
# 7.  Convenience — run from CLI
# ===========================================================================
def run_shap_explanation(
    data_path: str = "",
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Stand-alone entry point: load data, scale, and run SHAP pipeline.

    Returns the result dict from ``ShapExplainer.run()``.
    """
    from src.utils import load_dataframe

    if data_path:
        df = load_dataframe(data_path)
    else:
        candidate_paths = [
            DATA_DIR / "features_history.parquet",
            DATA_DIR / "features_latest.parquet",
            *sorted(DATA_DIR.glob("backfill_*.parquet")),
        ]
        best_df: Optional[pd.DataFrame] = None
        best_path: Optional[Path] = None
        best_rows = -1

        for p in candidate_paths:
            if not p.exists():
                continue
            try:
                part = load_dataframe(p)
            except Exception as exc:
                log.warning(f"Skipping unreadable SHAP input file '{p}': {exc}")
                continue
            if len(part) > best_rows:
                best_rows = len(part)
                best_df = part
                best_path = p

        if best_df is None:
            raise FileNotFoundError("No local data file found for SHAP explanation")

        log.info(f"Using SHAP input file: {best_path} ({best_rows} rows)")
        df = best_df

    meta = load_json(MODEL_DIR / "feature_columns.json")
    feature_cols = meta["feature_columns"]

    X = df[feature_cols].values.astype(np.float32)

    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    X_scaled = scaler.transform(X)

    explainer = ShapExplainer()
    return explainer.run(X_scaled, save_dir=save_dir)


if __name__ == "__main__":
    result = run_shap_explanation()
    print(f"Top {_TOP_N} features: {result['top_features']}")

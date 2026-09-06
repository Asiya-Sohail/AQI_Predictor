"""
api.py — Flask REST API for PM2.5 forecast serving.

Endpoints
---------
GET  /health     → Service readiness probe
GET  /predict    → 3-day (72-hour) recursive PM2.5 forecast as JSON
GET  /model/info → Training report and model metadata
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src.*`` imports resolve.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CITY,
    FLASK_DEBUG,
    FLASK_HOST,
    FLASK_PORT,
    FORECAST_HORIZON,
    MODEL_DIR,
)
from src.predict import (
    PM25Predictor,
    fetch_latest_features_from_hopsworks,
    fetch_latest_features_local,
)
from src.feature_pipeline import refresh_latest_features
from src.utils import load_json, log


# ===========================================================================
# Application factory
# ===========================================================================
def create_app() -> Flask:
    """Create and configure the Flask application."""
    application = Flask(__name__)
    CORS(application)

    # --- state -----------------------------------------------------------------
    _predictor: PM25Predictor | None = None

    def _get_predictor() -> PM25Predictor:
        nonlocal _predictor
        if _predictor is None:
            _predictor = PM25Predictor(use_registry=True)
        return _predictor

    # ===========================================================================
    # GET /health
    # ===========================================================================
    @application.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        """
        Service readiness probe.

        Returns 200 with service metadata when the API is reachable.
        Useful for load-balancer health checks and CI smoke tests.
        """
        model_ready = (MODEL_DIR / "training_report.json").exists()
        return jsonify({
            "status": "healthy",
            "service": "pm25-forecast-api",
            "city": CITY.strip().title(),
            "forecast_horizon_hours": FORECAST_HORIZON,
            "model_ready": model_ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200

    # ===========================================================================
    # GET /predict
    # ===========================================================================
    @application.route("/predict", methods=["GET"])
    def predict() -> tuple[Response, int]:
        """
        Return a 3-day (72-hour) recursive PM2.5 forecast.

        Query parameters
        ----------------
        horizon : int, optional
            Override the default forecast horizon (default: 72).
        source  : str, optional
            ``"hopsworks"`` (default) or ``"local"`` — where to fetch the
            latest feature row from.

        Response JSON
        -------------
        {
            "city":          "Karachi",
            "model_type":    "rf",
            "horizon_hours": 72,
            "count":         72,
            "generated_at":  "2026-02-22T12:00:00+00:00",
            "forecast": [
                {
                    "timestamp":              "2026-02-22T13:00:00+00:00",
                    "predicted_pm2_5":        62.14,
                    "predicted_aqi_category": "Moderate"
                },
                ...
            ]
        }
        """
        horizon = request.args.get("horizon", FORECAST_HORIZON, type=int)
        source = request.args.get("source", "hopsworks", type=str).lower()
        refresh = request.args.get("refresh", "true", type=str).lower() in {
            "1",
            "true",
            "yes",
            "y",
        }

        if horizon < 1 or horizon > 168:
            return jsonify({
                "error": "horizon must be between 1 and 168 hours"
            }), 400

        start = time.perf_counter()

        try:
            # --- 1. Load model -------------------------------------------------
            pred = _get_predictor()

            # --- 1a. Refresh from OpenWeather ---------------------------------
            if refresh:
                try:
                    refresh_latest_features(push_to_hopsworks=(source == "hopsworks"))
                except Exception as exc:
                    log.warning(f"Live OpenWeather refresh failed ({exc})")

            # --- 2. Fetch latest features --------------------------------------
            if source == "local":
                features_df = fetch_latest_features_local()
            else:
                try:
                    features_df = fetch_latest_features_from_hopsworks()
                except Exception as exc:
                    log.warning(
                        f"Hopsworks feature fetch failed ({exc}); "
                        "falling back to local data"
                    )
                    features_df = fetch_latest_features_local()

            if features_df.empty:
                return jsonify({
                    "error": "No feature data available. Run the feature pipeline first."
                }), 503

            seed_row = features_df.iloc[0]

            # --- 3. Recursive forecast -----------------------------------------
            forecast_df = pred.forecast(seed_row, horizon=horizon)

            # Convert timestamps to ISO-8601 strings for JSON
            forecast_df["timestamp"] = forecast_df["timestamp"].apply(
                lambda t: t.isoformat() if hasattr(t, "isoformat") else str(t)
            )

            elapsed = round(time.perf_counter() - start, 3)

            return jsonify({
                "city": CITY.strip().title(),
                "model_type": pred.model_type,
                "horizon_hours": horizon,
                "count": len(forecast_df),
                "elapsed_seconds": elapsed,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "forecast": forecast_df.to_dict(orient="records"),
            }), 200

        except FileNotFoundError as exc:
            log.error(f"/predict — artefact not found: {exc}")
            return jsonify({
                "error": "Model artefacts not found. Run the training pipeline first."
            }), 503

        except Exception as exc:
            log.error(f"/predict — unexpected error: {exc}", exc_info=True)
            return jsonify({"error": str(exc)}), 500

    # ===========================================================================
    # GET /model/info
    # ===========================================================================
    @application.route("/model/info", methods=["GET"])
    def model_info() -> tuple[Response, int]:
        """Return model metadata and most recent training metrics."""
        report_path = MODEL_DIR / "training_report.json"
        if not report_path.exists():
            return jsonify({"error": "No training report found"}), 404

        report = load_json(report_path)
        return jsonify(report), 200

    # ===========================================================================
    # Error handlers
    # ===========================================================================
    @application.errorhandler(404)
    def not_found(error: Exception) -> tuple[Response, int]:
        return jsonify({"error": "Endpoint not found"}), 404

    @application.errorhandler(405)
    def method_not_allowed(error: Exception) -> tuple[Response, int]:
        return jsonify({"error": "Method not allowed"}), 405

    @application.errorhandler(500)
    def internal_error(error: Exception) -> tuple[Response, int]:
        return jsonify({"error": "Internal server error"}), 500

    return application


# ===========================================================================
# Module-level app instance (used by ``flask run`` and main.py)
# ===========================================================================
app = create_app()


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    log.info(f"Starting Flask API on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

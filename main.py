"""
main.py — CLI entry point for the AQI Forecasting System.

Usage:
    python main.py --pipeline features      # Run feature engineering pipeline
    python main.py --pipeline training      # Train models
    python main.py --pipeline predict       # Run batch prediction
    python main.py --pipeline eda           # Generate EDA report
    python main.py --pipeline shap          # Generate SHAP explanations
    python main.py --pipeline backfill      # Historical backfill (requires --backfill-start / --backfill-end)
    python main.py --pipeline all           # Run features → training → predict → shap
    python main.py --serve api              # Start Flask REST API
    python main.py --serve streamlit        # Start Streamlit dashboard
"""

import sys
from pathlib import Path

import click

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


@click.command()
@click.option(
    "--pipeline",
    type=click.Choice(
        ["features", "training", "predict", "eda", "shap", "backfill", "all"],
        case_sensitive=False,
    ),
    default=None,
    help="Pipeline stage to execute.",
)
@click.option(
    "--serve",
    type=click.Choice(["api", "streamlit"], case_sensitive=False),
    default=None,
    help="Launch a serving application.",
)
@click.option(
    "--local",
    is_flag=True,
    default=False,
    help="Use local data files instead of Hopsworks.",
)
@click.option(
    "--backfill-start",
    type=str,
    default=None,
    help="Backfill start date (YYYY-MM-DD). Required for --pipeline backfill.",
)
@click.option(
    "--backfill-end",
    type=str,
    default=None,
    help="Backfill end date (YYYY-MM-DD). Required for --pipeline backfill.",
)
def main(pipeline: str, serve: str, local: bool, backfill_start: str, backfill_end: str) -> None:
    """AQI Forecasting System — CLI entry point."""
    from src.utils import log

    if pipeline is None and serve is None:
        click.echo("Specify --pipeline or --serve. Use --help for details.")
        sys.exit(1)

    # ----- Pipelines -----
    if pipeline:
        if pipeline == "backfill":
            if not backfill_start or not backfill_end:
                click.echo(
                    "Error: --backfill-start and --backfill-end are required "
                    "for --pipeline backfill."
                )
                sys.exit(1)
            log.info(">>> Running Historical Backfill")
            from src.feature_pipeline import run_backfill

            run_backfill(
                start_date=backfill_start,
                end_date=backfill_end,
                push_to_hopsworks=not local,
            )

        if pipeline in ("features", "all"):
            log.info(">>> Running Feature Pipeline")
            from src.feature_pipeline import run_feature_pipeline

            run_feature_pipeline()

        if pipeline in ("eda",):
            log.info(">>> Running EDA Pipeline")
            from src.config import DATA_DIR
            from src.eda import run_full_eda
            from src.utils import load_dataframe

            df = load_dataframe(str(DATA_DIR / "features_latest.parquet"))
            run_full_eda(df)

        if pipeline in ("training", "all"):
            log.info(">>> Running Training Pipeline")
            from src.training_pipeline import run_training_pipeline

            run_training_pipeline(use_hopsworks=not local)

        if pipeline in ("predict", "all"):
            log.info(">>> Running Prediction Pipeline (72h forecast)")
            from src.predict import run_prediction_pipeline

            forecast_df = run_prediction_pipeline(
                use_hopsworks=not local,
                horizon=72,
                save_output=True,
            )
            log.info(
                f"Forecast saved — {len(forecast_df)} hourly steps"
            )

        if pipeline in ("shap", "all"):
            log.info(">>> Running SHAP Explanations")
            from src.shap_explainer import run_shap_explanation

            result = run_shap_explanation()
            log.info(f"Top 5 features: {result['top_features']}")

    # ----- Serving -----
    if serve:
        if serve == "api":
            log.info(">>> Starting Flask API")
            from app.api import app
            from src.config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT

            app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

        elif serve == "streamlit":
            import subprocess

            log.info(">>> Starting Streamlit Dashboard")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app/streamlit_app.py",
                    "--server.port=8501",
                    "--server.headless=true",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()

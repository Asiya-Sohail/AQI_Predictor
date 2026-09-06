"""
eda.py — Exploratory Data Analysis module for the AQI Forecasting System.

Analyses
--------
1.  Correlation matrix heatmap
2.  Time-series trend plots (PM2.5, temperature, humidity, …)
3.  Seasonal analysis (hourly, daily, monthly boxplots)
4.  Distribution plots (histograms + KDE for every numeric feature)

All plots are saved as PNG files to ``DATA_DIR/eda_outputs/``.
Every public function accepts an optional ``save_dir`` so callers can
override the output location.  The module is designed to be imported and
called piecemeal or via the convenience runner ``run_full_eda(df)``.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                       # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    CITY,
    CITY_COLUMN,
    DATA_DIR,
    DATETIME_COLUMN,
    DERIVED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.utils import log

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.05)
_PALETTE = "mako"
_DPI = 150

# All numeric columns available for analysis
_ALL_NUMERIC = FEATURE_COLUMNS + [TARGET_COLUMN] + DERIVED_COLUMNS


# ===========================================================================
# Internal helpers
# ===========================================================================
def _ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_save_dir(save_dir: Optional[Path]) -> Path:
    """Return the effective output directory, defaulting to DATA_DIR/eda_outputs."""
    return _ensure_dir(save_dir or DATA_DIR / "eda_outputs")


def _savefig(fig: plt.Figure, path: Path) -> None:
    """Save figure and close it."""
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved → {path}")


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with timestamp parsed and sorted."""
    df = df.copy()
    if DATETIME_COLUMN in df.columns:
        df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], utc=True)
        df = df.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    return df


def _numeric_cols(df: pd.DataFrame, columns: Optional[List[str]] = None) -> List[str]:
    """Return the subset of *columns* that exist in *df* and are numeric."""
    candidates = columns or _ALL_NUMERIC
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


# ===========================================================================
# 1. Summary Statistics
# ===========================================================================
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for every numeric column.

    Returns a transposed ``describe()`` DataFrame with an extra
    ``missing_pct`` column.
    """
    stats = df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    log.info(f"Summary statistics computed for {len(stats)} columns")
    return stats


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value report, sorted descending by percentage."""
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": total, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    log.info(f"Missing-value report: {len(report)} columns with nulls")
    return report


# ===========================================================================
# 2. Correlation Matrix
# ===========================================================================
def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Plot a lower-triangle Pearson correlation heatmap and return the
    correlation DataFrame.

    Saved as ``correlation_matrix.png``.
    """
    cols = _numeric_cols(df, columns)
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.9), max(8, len(cols) * 0.75)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out = _resolve_save_dir(save_dir)
    _savefig(fig, out / "correlation_matrix.png")
    corr.to_csv(out / "correlation_matrix.csv")
    log.info(f"Correlation CSV saved → {out / 'correlation_matrix.csv'}")
    return corr


# ===========================================================================
# 3. Time-Series Trends
# ===========================================================================
def plot_timeseries_trends(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Plot line charts for each numeric feature over time.

    If the ``city`` column exists the data is filtered to the configured
    ``CITY``.  One subplot per feature, saved as ``timeseries_trends.png``.
    """
    df = _prepare_df(df)
    if DATETIME_COLUMN not in df.columns:
        log.warning(f"'{DATETIME_COLUMN}' column missing — skipping time-series trends")
        return

    if CITY_COLUMN in df.columns:
        df = df[df[CITY_COLUMN] == CITY.strip().lower()]

    cols = _numeric_cols(df, columns) or _numeric_cols(df, FEATURE_COLUMNS + [TARGET_COLUMN])
    if not cols:
        log.warning("No numeric columns available for time-series trends")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3.5 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)
    flat_axes = axes.flatten()

    colours = sns.color_palette(_PALETTE, len(cols))

    for idx, col in enumerate(cols):
        ax = flat_axes[idx]
        ax.plot(
            df[DATETIME_COLUMN],
            df[col],
            linewidth=0.8,
            color=colours[idx],
            alpha=0.85,
        )
        ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)

    for idx in range(len(cols), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(
        f"Time-Series Trends — {CITY.strip().title()}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    out = _resolve_save_dir(save_dir)
    _savefig(fig, out / "timeseries_trends.png")


def plot_pm25_trend(
    df: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Dedicated PM2.5 time-series with a 24-hour rolling average overlay.

    Saved as ``pm25_trend.png``.
    """
    df = _prepare_df(df)
    if DATETIME_COLUMN not in df.columns or "pm2_5" not in df.columns:
        log.warning("Required columns missing — skipping PM2.5 trend plot")
        return

    if CITY_COLUMN in df.columns:
        df = df[df[CITY_COLUMN] == CITY.strip().lower()]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        df[DATETIME_COLUMN],
        df["pm2_5"],
        linewidth=0.6,
        alpha=0.5,
        color="grey",
        label="Hourly PM2.5",
    )
    rolling = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    ax.plot(
        df[DATETIME_COLUMN],
        rolling,
        linewidth=1.8,
        color="crimson",
        label="24h Rolling Mean",
    )
    ax.set_title(
        f"PM2.5 Concentration — {CITY.strip().title()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.tight_layout()

    out = _resolve_save_dir(save_dir)
    _savefig(fig, out / "pm25_trend.png")


# ===========================================================================
# 4. Seasonal Analysis
# ===========================================================================
def plot_seasonal_analysis(
    df: pd.DataFrame,
    target: str = "pm2_5",
    save_dir: Optional[Path] = None,
) -> None:
    """
    Boxplots of *target* grouped by hour-of-day, day-of-month, and month.

    Requires time features (``hour``, ``day``, ``month``) — if missing they
    are derived from ``timestamp``.  Saved as ``seasonal_analysis.png``.
    """
    df = _prepare_df(df)
    if target not in df.columns:
        log.warning(f"Target '{target}' not in DataFrame — skipping seasonal analysis")
        return

    # Ensure time features exist
    if DATETIME_COLUMN in df.columns:
        dt = pd.to_datetime(df[DATETIME_COLUMN])
        if "hour" not in df.columns:
            df["hour"] = dt.dt.hour
        if "day" not in df.columns:
            df["day"] = dt.dt.day
        if "month" not in df.columns:
            df["month"] = dt.dt.month

    group_cols = [("hour", "Hour of Day"), ("day", "Day of Month"), ("month", "Month")]
    available = [(c, label) for c, label in group_cols if c in df.columns]

    if not available:
        log.warning("No time-group columns available for seasonal analysis")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    palette = sns.color_palette("viridis", n_colors=31)

    for ax, (col, label) in zip(axes, available):
        sns.boxplot(
            data=df,
            x=col,
            y=target,
            ax=ax,
            palette=palette,
            fliersize=2,
            linewidth=0.7,
        )
        ax.set_title(f"{target.replace('_', ' ').upper()} by {label}", fontsize=12, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel(target.replace("_", " ").title() + " (µg/m³)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Seasonal / Temporal Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = _resolve_save_dir(save_dir)
    _savefig(fig, out / "seasonal_analysis.png")


def plot_hourly_heatmap(
    df: pd.DataFrame,
    target: str = "pm2_5",
    save_dir: Optional[Path] = None,
) -> None:
    """
    Heatmap of mean *target* by day-of-week × hour-of-day.

    Saved as ``hourly_heatmap.png``.
    """
    df = _prepare_df(df)
    if target not in df.columns or DATETIME_COLUMN not in df.columns:
        log.warning("Required columns missing — skipping hourly heatmap")
        return

    dt = pd.to_datetime(df[DATETIME_COLUMN])
    df["_dow"] = dt.dt.day_name()
    df["_hour"] = dt.dt.hour

    pivot = df.pivot_table(values=target, index="_dow", columns="_hour", aggfunc="mean")

    # Reorder days Mon→Sun
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        linewidths=0.3,
        cbar_kws={"label": f"Mean {target.replace('_', ' ').upper()} (µg/m³)"},
        ax=ax,
    )
    ax.set_title(
        f"Mean {target.replace('_', ' ').upper()} — Day of Week × Hour",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("")
    plt.tight_layout()

    out = _resolve_save_dir(save_dir)
    _savefig(fig, out / "hourly_heatmap.png")

    # Clean up temp columns
    df.drop(columns=["_dow", "_hour"], inplace=True, errors="ignore")


# ===========================================================================
# 5. Distribution Plots
# ===========================================================================
def plot_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Histogram + KDE for every numeric feature.

    Saved as ``distributions.png``.
    """
    cols = _numeric_cols(df, columns)
    if not cols:
        log.warning("No numeric columns for distribution plots")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    colours = sns.color_palette(_PALETTE, len(cols))

    for idx, col in enumerate(cols):
        sns.histplot(
            df[col].dropna(),
            kde=True,
            ax=axes[idx],
            color=colours[idx],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[idx].set_title(
            col.replace("_", " ").title(),
            fontsize=11,
            fontweight="bold",
        )
        axes[idx].set_xlabel("")
        axes[idx].grid(axis="y", alpha=0.2)

    for idx in range(len(cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = _resolve_save_dir(save_dir)
    _savefig(fig, out / "distributions.png")


def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Seaborn pair-plot for a subset of features (max 6 to keep readable).

    Saved as ``pairplot.png``.
    """
    cols = _numeric_cols(df, columns) or _numeric_cols(
        df, ["pm2_5", "pm10", "temperature", "humidity", "wind_speed", TARGET_COLUMN]
    )
    cols = cols[:6]  # cap for readability
    if len(cols) < 2:
        log.warning("Need ≥ 2 columns for pair-plot — skipping")
        return

    sample = df[cols].dropna()
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)

    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12})
    g.figure.suptitle("Feature Pair-Plot", fontsize=15, fontweight="bold", y=1.01)

    out = _resolve_save_dir(save_dir)
    g.savefig(out / "pairplot.png", dpi=_DPI, bbox_inches="tight")
    plt.close(g.figure)
    log.info(f"Plot saved → {out / 'pairplot.png'}")


# ===========================================================================
# 6. Runner — Full EDA Report
# ===========================================================================
def run_full_eda(
    df: pd.DataFrame,
    save_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Execute the complete EDA pipeline and persist all artefacts.

    Steps
    -----
    1.  Summary statistics  → ``summary_statistics.csv``
    2.  Missing-value report → ``missing_value_report.csv``
    3.  Correlation matrix   → ``correlation_matrix.png`` / ``.csv``
    4.  Time-series trends   → ``timeseries_trends.png``
    5.  PM2.5 trend          → ``pm25_trend.png``
    6.  Seasonal analysis    → ``seasonal_analysis.png``
    7.  Hourly heatmap       → ``hourly_heatmap.png``
    8.  Distributions        → ``distributions.png``
    9.  Pair-plot            → ``pairplot.png``

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (output of the feature pipeline).
    save_dir : Path, optional
        Override output directory (default: ``data/eda_outputs``).

    Returns
    -------
    dict
        Mapping of artefact name → file path for downstream use.
    """
    out = _resolve_save_dir(save_dir)
    log.info("=" * 60)
    log.info("EDA PIPELINE — START")
    log.info(f"Rows: {len(df)}  |  Columns: {len(df.columns)}  |  Output: {out}")
    log.info("=" * 60)

    artefacts: Dict[str, str] = {"output_dir": str(out)}

    # --- 1. Summary statistics ---
    stats = summary_statistics(df)
    stats_path = out / "summary_statistics.csv"
    stats.to_csv(stats_path)
    artefacts["summary_statistics"] = str(stats_path)

    # --- 2. Missing-value report ---
    missing = missing_value_report(df)
    missing_path = out / "missing_value_report.csv"
    missing.to_csv(missing_path)
    artefacts["missing_value_report"] = str(missing_path)

    # --- 3. Correlation matrix ---
    plot_correlation_matrix(df, save_dir=out)
    artefacts["correlation_matrix"] = str(out / "correlation_matrix.png")

    # --- 4. Time-series trends ---
    plot_timeseries_trends(df, save_dir=out)
    artefacts["timeseries_trends"] = str(out / "timeseries_trends.png")

    # --- 5. PM2.5 trend ---
    plot_pm25_trend(df, save_dir=out)
    artefacts["pm25_trend"] = str(out / "pm25_trend.png")

    # --- 6. Seasonal analysis ---
    plot_seasonal_analysis(df, save_dir=out)
    artefacts["seasonal_analysis"] = str(out / "seasonal_analysis.png")

    # --- 7. Hourly heatmap ---
    plot_hourly_heatmap(df, save_dir=out)
    artefacts["hourly_heatmap"] = str(out / "hourly_heatmap.png")

    # --- 8. Distributions ---
    plot_distributions(df, save_dir=out)
    artefacts["distributions"] = str(out / "distributions.png")

    # --- 9. Pair-plot ---
    plot_pairplot(df, save_dir=out)
    artefacts["pairplot"] = str(out / "pairplot.png")

    log.info("=" * 60)
    log.info(f"EDA PIPELINE — END  ({len(artefacts) - 1} artefacts)")
    log.info("=" * 60)
    return artefacts


if __name__ == "__main__":
    from src.utils import load_dataframe

    df = load_dataframe(str(DATA_DIR / "features_latest.parquet"))
    run_full_eda(df)

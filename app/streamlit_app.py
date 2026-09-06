"""
streamlit_app.py — Interactive Streamlit dashboard for PM2.5 AQI forecasting.

Features
--------
- Current AQI metric card with colour-coded category badge
- 3-day (72 h) interactive Plotly forecast line chart with AQI bands
- AQI category badges per hour in the forecast table
- SHAP feature-importance interactive bar chart (Plotly re-render)
- Real-time ⚠️ alerts when any forecasted hour exceeds PM2.5 > 150
- Connects to the Flask REST API (``GET /predict``, ``GET /health``)
  with automatic local-fallback when the API is unreachable

Run
---
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``src.*`` imports resolve.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AQI_CATEGORIES,
    CITY,
    DATA_DIR,
    FLASK_HOST,
    FLASK_PORT,
    FORECAST_HORIZON,
)
from src.feature_pipeline import refresh_latest_features
from src.predict import PM25Predictor, classify_aqi
from src.utils import load_dataframe, load_json, log

# ═══════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="10Pearls AQI Forecast",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# Design constants
# ═══════════════════════════════════════════════════════════════════════════
CATEGORY_COLORS: Dict[str, str] = {
    "Good": "#00e400",
    "Moderate": "#ffbf00",
    "Unhealthy": "#ff4d4d",
    "Hazardous": "#7e0023",
}

CATEGORY_TEXT_COLORS: Dict[str, str] = {
    "Good": "#003300",
    "Moderate": "#3d2e00",
    "Unhealthy": "#ffffff",
    "Hazardous": "#ffffff",
}

_ALERT_THRESHOLD_PM25 = 150  # µg/m³ — show warning if exceeded

_API_BASE = f"http://{'127.0.0.1' if FLASK_HOST == '0.0.0.0' else FLASK_HOST}:{FLASK_PORT}"


# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS — injected once at app start
# ═══════════════════════════════════════════════════════════════════════════
def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ---- metric cards ---- */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
            border: 1px solid #3a3a5c;
            border-radius: 12px;
            padding: 18px 22px;
            box-shadow: 0 4px 14px rgba(0,0,0,.25);
        }
        div[data-testid="stMetric"] label {
            color: #a0a0c0;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }

        /* ---- sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

        /* ---- alert box ---- */
        .aqi-alert {
            background: linear-gradient(90deg, #ff4d4d22, #7e002322);
            border-left: 5px solid #ff4d4d;
            border-radius: 8px;
            padding: 14px 20px;
            margin: 12px 0;
            font-size: 0.95rem;
        }

        /* ---- category badge ---- */
        .cat-badge {
            display: inline-block;
            padding: 8px 22px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.05rem;
            text-align: center;
            letter-spacing: 0.3px;
        }

        /* ---- forecast table ---- */
        .forecast-table td, .forecast-table th {
            text-align: center !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data fetching — Flask API first, local fallback second
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner="Fetching forecast from API …")
def fetch_forecast_from_api(horizon: int) -> Optional[Dict[str, Any]]:
    """
    Call ``GET /predict`` on the Flask API using live OpenWeather refresh.

    Returns the full JSON response dict or *None* on failure.
    """
    try:
        resp = requests.get(
            f"{_API_BASE}/predict",
            params={"horizon": horizon, "source": "local", "refresh": "true"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"API returned {resp.status_code}: {resp.text[:200]}")
    except requests.ConnectionError:
        log.warning("Flask API unreachable — will fall back to local inference")
    except Exception as exc:
        log.warning(f"API request failed: {exc}")
    return None


@st.cache_data(ttl=300, show_spinner="Checking API health …")
def fetch_api_health() -> Optional[Dict[str, Any]]:
    """Call ``GET /health`` on the Flask API."""
    try:
        resp = requests.get(f"{_API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner="Loading model …")
def _load_predictor() -> PM25Predictor:
    return PM25Predictor(use_registry=False)


@st.cache_data(ttl=600, show_spinner="Loading local feature data …")
def _load_latest_features() -> pd.DataFrame:
    path = DATA_DIR / "features_latest.parquet"
    if path.exists():
        return load_dataframe(path)
    return pd.DataFrame()


def _local_forecast(horizon: int) -> Optional[Dict[str, Any]]:
    """
    Run inference directly (no API) and return a dict matching the API shape.
    """
    try:
        try:
            refresh_latest_features(push_to_hopsworks=False)
        except Exception as exc:
            log.warning(f"OpenWeather refresh failed in local mode: {exc}")

        _load_latest_features.clear()
        pred = _load_predictor()
        df = _load_latest_features()
        if df.empty:
            return None
        seed_row = df.iloc[0]
        forecast_df = pred.forecast(seed_row, horizon=horizon)
        forecast_df["timestamp"] = forecast_df["timestamp"].apply(
            lambda t: t.isoformat() if hasattr(t, "isoformat") else str(t)
        )
        return {
            "city": CITY.strip().title(),
            "model_type": pred.model_type,
            "horizon_hours": horizon,
            "count": len(forecast_df),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "forecast": forecast_df.to_dict(orient="records"),
        }
    except Exception as exc:
        log.error(f"Local forecast failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SHAP data loader
# ═══════════════════════════════════════════════════════════════════════════
def _load_shap_data() -> Optional[Dict[str, Any]]:
    """
    Return a dict with ``ranking_df``, ``report``, ``summary_path``,
    ``bar_path`` if SHAP outputs exist.
    """
    shap_dir = DATA_DIR / "shap_outputs"
    ranking_path = shap_dir / "shap_feature_ranking.csv"
    report_path = shap_dir / "shap_report.json"
    summary_path = shap_dir / "shap_summary.png"
    bar_path = shap_dir / "shap_feature_importance.png"

    if not ranking_path.exists():
        return None

    ranking_df = pd.read_csv(ranking_path)
    report = load_json(report_path) if report_path.exists() else {}

    return {
        "ranking_df": ranking_df,
        "report": report,
        "summary_path": summary_path if summary_path.exists() else None,
        "bar_path": bar_path if bar_path.exists() else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# UI components
# ═══════════════════════════════════════════════════════════════════════════

# ---------- Sidebar -------------------------------------------------------
def _render_sidebar() -> Dict[str, Any]:
    """Sidebar controls — returns user selections."""
    st.sidebar.markdown("## ⚙️ Controls")

    horizon = st.sidebar.slider(
        "Forecast horizon (hours)",
        min_value=6,
        max_value=168,
        value=FORECAST_HORIZON,
        step=6,
    )

    show_shap = st.sidebar.checkbox("Show SHAP explanations", value=True)
    show_table = st.sidebar.checkbox("Show forecast table", value=False)

    st.sidebar.markdown("---")

    # API health indicator
    health = fetch_api_health()
    if health:
        st.sidebar.success(f"API: **{health.get('status', 'OK').upper()}**")
    else:
        st.sidebar.warning("API offline — using local inference")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"City: **{CITY.strip().title()}**  \n"
        f"Default horizon: **{FORECAST_HORIZON} h**  \n"
        f"Alert threshold: **{_ALERT_THRESHOLD_PM25} µg/m³**"
    )

    return {"horizon": horizon, "show_shap": show_shap, "show_table": show_table}


# ---------- Header ---------------------------------------------------------
def _render_header() -> None:
    st.markdown(
        "<h1 style='margin-bottom:0; color:#ffffff;'>🏭 10Pearls AQI Forecast Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"📍 Real-time 72-hour recursive PM2.5 forecasting for "
        f"**{CITY.strip().title()}** — powered by RF · Ridge · DNN"
    )
    st.markdown("")  # spacer


# ---------- Current AQI card -----------------------------------------------
def _render_current_aqi(data: Dict[str, Any]) -> None:
    """Top metric row: current PM2.5, AQI category badge, model info."""
    forecast_list: List[Dict] = data.get("forecast", [])
    if not forecast_list:
        st.info("No forecast data available.")
        return

    first = forecast_list[0]
    pm25_now = first.get("predicted_pm2_5", 0.0)
    cat_now = first.get("predicted_aqi_category", classify_aqi(pm25_now))
    bg = CATEGORY_COLORS.get(cat_now, "#888")
    fg = CATEGORY_TEXT_COLORS.get(cat_now, "#fff")

    col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

    with col1:
        st.metric("Current PM2.5", f"{pm25_now:.1f} µg/m³")

    with col2:
        st.markdown(
            f"<div class='cat-badge' style='background:{bg};color:{fg};'>"
            f"{cat_now}</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.metric("Model", data.get("model_type", "—").upper())

    with col4:
        st.metric("Horizon", f"{data.get('horizon_hours', '—')} h")


# ---------- Alerts ----------------------------------------------------------
def _render_alerts(forecast_list: List[Dict]) -> None:
    """Show a prominent warning banner if any hour exceeds the threshold."""
    bad_hours = [
        h for h in forecast_list
        if h.get("predicted_pm2_5", 0) > _ALERT_THRESHOLD_PM25
    ]
    if not bad_hours:
        return

    worst = max(bad_hours, key=lambda h: h["predicted_pm2_5"])
    worst_pm = worst["predicted_pm2_5"]
    worst_cat = worst.get("predicted_aqi_category", classify_aqi(worst_pm))
    n_hours = len(bad_hours)

    st.markdown(
        f"""
        <div class='aqi-alert'>
            ⚠️ <strong>Air Quality Alert</strong> — <strong>{n_hours}</strong>
            hour{"s" if n_hours > 1 else ""} forecast above
            <strong>{_ALERT_THRESHOLD_PM25} µg/m³</strong>.
            Peak: <strong>{worst_pm:.1f} µg/m³</strong>
            (<strong>{worst_cat}</strong>) at {worst.get("timestamp", "—")}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hazardous-level alert (PM2.5 > 200)
    hazardous_hours = [
        h for h in forecast_list
        if h.get("predicted_pm2_5", 0) > 200
    ]
    if hazardous_hours:
        st.error("🚨 Hazardous Air Quality Alert!")


# ---------- Forecast line chart --------------------------------------------
def _render_forecast_chart(data: Dict[str, Any]) -> None:
    """Interactive Plotly line chart with AQI threshold bands."""
    forecast_list = data.get("forecast", [])
    if not forecast_list:
        return

    df = pd.DataFrame(forecast_list)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    city = data.get("city", CITY.strip().title())
    horizon = data.get("horizon_hours", len(df))

    fig = go.Figure()

    # ---- colour each segment by AQI category ----
    for cat, colour in CATEGORY_COLORS.items():
        mask = df["predicted_aqi_category"] == cat
        if mask.any():
            seg = df[mask]
            fig.add_trace(go.Scatter(
                x=seg["timestamp"],
                y=seg["predicted_pm2_5"],
                mode="markers",
                marker=dict(color=colour, size=6, line=dict(width=0.5, color="#222")),
                name=cat,
                legendgroup=cat,
                showlegend=True,
                hovertemplate=(
                    "<b>%{x|%b %d, %H:%M}</b><br>"
                    "PM2.5: %{y:.1f} µg/m³<br>"
                    f"Category: {cat}<extra></extra>"
                ),
            ))

    # continuous line over all points
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["predicted_pm2_5"],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.45)", width=1.5),
        showlegend=False,
        hoverinfo="skip",
    ))

    # ---- AQI threshold bands ----
    band_alpha = 0.06
    for (low, high), label in AQI_CATEGORIES.items():
        fig.add_hrect(
            y0=low, y1=min(high, df["predicted_pm2_5"].max() * 1.3),
            fillcolor=CATEGORY_COLORS.get(label, "#ccc"),
            opacity=band_alpha,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color="#888",
        )

    # ---- alert threshold line ----
    fig.add_hline(
        y=_ALERT_THRESHOLD_PM25,
        line_dash="dash",
        line_color="#ff4d4d",
        line_width=1,
        annotation_text=f"Alert ({_ALERT_THRESHOLD_PM25})",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#ff4d4d",
    )

    fig.update_layout(
        title=dict(
            text=f"PM2.5 Forecast — {city} (next {horizon} h)",
            font=dict(size=18),
        ),
        xaxis_title="Time",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=480,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font_size=12,
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(
        tickformat="%b %d\n%H:%M",
        dtick=6 * 3600 * 1000,   # tick every 6 h
    )

    st.plotly_chart(fig, use_container_width=True, key="forecast_chart")


# ---------- Category badges timeline ----------------------------------------
def _render_category_badges(forecast_list: List[Dict]) -> None:
    """Render a compact row of colour-coded AQI badges for Day 1, 2, 3."""
    if not forecast_list:
        return

    st.subheader("Daily AQI Summary")

    # Group into 24-hour blocks
    n = len(forecast_list)
    day_labels = ["Day 1 (0–24 h)", "Day 2 (24–48 h)", "Day 3 (48–72 h)"]
    cols = st.columns(min(3, (n // 24) + (1 if n % 24 else 0)))

    for i, col in enumerate(cols):
        start = i * 24
        end = min(start + 24, n)
        if start >= n:
            break

        chunk = forecast_list[start:end]
        avg_pm = np.mean([h["predicted_pm2_5"] for h in chunk])
        max_pm = max(h["predicted_pm2_5"] for h in chunk)
        min_pm = min(h["predicted_pm2_5"] for h in chunk)
        cat = classify_aqi(avg_pm)
        bg = CATEGORY_COLORS.get(cat, "#888")
        fg = CATEGORY_TEXT_COLORS.get(cat, "#fff")

        with col:
            label = day_labels[i] if i < len(day_labels) else f"Day {i+1}"
            st.markdown(
                f"<div style='text-align:center;margin-bottom:6px;"
                f"font-weight:600;font-size:0.9rem;'>{label}</div>"
                f"<div class='cat-badge' style='background:{bg};color:{fg};"
                f"width:100%;box-sizing:border-box;'>{cat}</div>"
                f"<div style='text-align:center;margin-top:6px;font-size:0.8rem;"
                f"color:#aaa;'>"
                f"Avg {avg_pm:.1f} · Min {min_pm:.1f} · Max {max_pm:.1f} µg/m³"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------- Forecast table --------------------------------------------------
def _render_forecast_table(forecast_list: List[Dict]) -> None:
    """Expandable table with all hourly predictions."""
    if not forecast_list:
        return

    df = pd.DataFrame(forecast_list)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

    df = df.rename(columns={
        "timestamp": "Time",
        "predicted_pm2_5": "PM2.5 (µg/m³)",
        "predicted_aqi_category": "AQI Category",
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "PM2.5 (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


# ---------- SHAP section ---------------------------------------------------
def _render_shap_section() -> None:
    """Interactive SHAP feature-importance chart + pre-generated plots."""
    shap_data = _load_shap_data()

    if shap_data is None:
        st.info(
            "SHAP explanations not available yet.  \n"
            "Run `python main.py --pipeline shap` to generate them."
        )
        return

    ranking_df: pd.DataFrame = shap_data["ranking_df"]
    report: Dict = shap_data.get("report", {})

    # ---- Interactive Plotly bar chart (top features) ----
    top_n = ranking_df.head(15).copy()
    top_n = top_n.sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(go.Bar(
        x=top_n["mean_abs_shap"],
        y=top_n["feature"],
        orientation="h",
        marker_color=px.colors.sequential.Tealgrn_r[:len(top_n)],
        text=top_n["mean_abs_shap"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Feature Importance — Mean |SHAP Value|", font_size=17),
        xaxis_title="Mean |SHAP value|",
        yaxis_title="",
        template="plotly_dark",
        height=max(350, len(top_n) * 28 + 80),
        margin=dict(l=10, r=80, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key="shap_bar")

    # ---- Top features callout ----
    top_features = report.get("top_features", ranking_df["feature"].head(5).tolist())
    st.markdown(
        "**Top 5 features:** "
        + " · ".join(f"`{f}`" for f in top_features[:5])
    )

    # ---- Pre-generated SHAP plots (Beeswarm + Bar) ----
    img_col1, img_col2 = st.columns(2)
    if shap_data.get("summary_path"):
        img_col1.image(
            str(shap_data["summary_path"]),
            caption="SHAP Beeswarm Summary",
            use_column_width=True,
        )
    if shap_data.get("bar_path"):
        img_col2.image(
            str(shap_data["bar_path"]),
            caption="Mean |SHAP| Bar Chart",
            use_column_width=True,
        )

    # ---- Full ranking table ----
    with st.expander("Full feature ranking table"):
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Application entry point
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Top-level Streamlit app runner."""
    _inject_css()
    _render_header()

    selections = _render_sidebar()
    horizon = selections["horizon"]

    # ── Fetch data: API → local fallback ──────────────────────────────────
    data = fetch_forecast_from_api(horizon)
    source_label = "Flask API"

    if data is None:
        data = _local_forecast(horizon)
        source_label = "Local inference"

    if data is None:
        st.error(
            "**Unable to generate forecast.**  \n\n"
            "Ensure the feature pipeline has been run at least once:\n\n"
            "```bash\npython main.py --pipeline features\n```\n\n"
            "Then train a model:\n\n"
            "```bash\npython main.py --pipeline training\n```"
        )
        return

    st.caption(f"Source: **{source_label}** · Generated: {data.get('generated_at', '—')}")

    # ── Current AQI metrics ───────────────────────────────────────────────
    _render_current_aqi(data)

    # ── Alerts ────────────────────────────────────────────────────────────
    _render_alerts(data.get("forecast", []))

    st.markdown("---")

    # ── 3-day forecast chart ──────────────────────────────────────────────
    _render_forecast_chart(data)

    # ── Daily AQI category badges ─────────────────────────────────────────
    _render_category_badges(data.get("forecast", []))

    # ── Forecast table (optional) ─────────────────────────────────────────
    if selections["show_table"]:
        st.markdown("---")
        st.subheader("Hourly Forecast Table")
        _render_forecast_table(data.get("forecast", []))

    # ── SHAP section ──────────────────────────────────────────────────────
    if selections["show_shap"]:
        st.markdown("---")
        st.subheader("� Model Explainability (SHAP)")
        _render_shap_section()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "🏢 10Pearls AQI Forecast Dashboard · Built with Streamlit & Plotly · "
        f"© {datetime.now().year}"
    )


if __name__ == "__main__":
    main()

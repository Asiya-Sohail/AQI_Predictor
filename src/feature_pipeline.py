"""
feature_pipeline.py — Feature engineering & Hopsworks Feature Store pipeline.

Fetches weather + air-pollution data from the OpenWeatherMap API, engineers
time-based and derived features, and stores the result in the Hopsworks
Feature Store via the hsfs library.

Extracted raw features : temperature, humidity, pressure, wind_speed,
                         pm2_5, pm10, no2, o3, co
Time-based features    : hour, day, month
Derived features       : aqi_change_rate, rolling_mean_3
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import hsfs
except Exception:  # pragma: no cover - optional for local-only serving
    hsfs = None
import pandas as pd
import requests

from src.config import (
    CITY,
    CITY_COLUMN,
    DATA_DIR,
    DATETIME_COLUMN,
    FEATURE_COLUMNS,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    LAT,
    LON,
    OPENWEATHER_API_KEY,
    OPENWEATHER_HISTORY_POLLUTION_URL,
    OPENWEATHER_POLLUTION_URL,
    OPENWEATHER_TIMEMACHINE_URL,
    OPENWEATHER_WEATHER_URL,
    TARGET_COLUMN,
)
from src.hopsworks_client import login_to_hopsworks
from src.utils import log, save_dataframe


# ============================================================================
# 1. Data Ingestion — OpenWeatherMap API
# ============================================================================
def _get_coords() -> Tuple[float, float]:
    """Return (lat, lon) from environment configuration."""
    return (LAT, LON)


def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    """
    GET /data/2.5/weather — current weather data.

    Returns a flat dict with: temperature, humidity, pressure, wind_speed.
    Raises ``requests.RequestException`` on network / HTTP errors.
    """
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    resp = requests.get(OPENWEATHER_WEATHER_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    main = data.get("main", {})
    wind = data.get("wind", {})

    return {
        "temperature": main.get("temp"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),
    }


def fetch_pollution(lat: float, lon: float) -> Dict[str, Any]:
    """
    GET /data/2.5/air_pollution — current air-quality data.

    Returns a flat dict with: pm2_5, pm10, no2, o3, co, aqi.
    Raises ``requests.RequestException`` on network / HTTP errors.
    """
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    resp = requests.get(OPENWEATHER_POLLUTION_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    entry = data.get("list", [{}])[0]
    components = entry.get("components", {})
    main_aqi = entry.get("main", {}).get("aqi")  # 1-5 scale

    return {
        "pm2_5": components.get("pm2_5"),
        "pm10": components.get("pm10"),
        "no2": components.get("no2"),
        "o3": components.get("o3"),
        "co": components.get("co"),
        TARGET_COLUMN: main_aqi,
    }


def fetch_city_data() -> Optional[Dict[str, Any]]:
    """Fetch combined weather + pollution record for the configured city."""
    lat, lon = _get_coords()
    try:
        weather = fetch_weather(lat, lon)
        pollution = fetch_pollution(lat, lon)
    except requests.RequestException as exc:
        log.error(f"OpenWeather API error for '{CITY}' (lat={lat}, lon={lon}): {exc}")
        return None

    record: Dict[str, Any] = {
        DATETIME_COLUMN: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        CITY_COLUMN: CITY.strip().lower(),
    }
    record.update(weather)
    record.update(pollution)
    return record


def fetch_data() -> pd.DataFrame:
    """Fetch data for the configured city and return a DataFrame."""
    record = fetch_city_data()
    if record is None:
        log.warning("No data fetched from OpenWeather — returning empty DataFrame")
        return pd.DataFrame()

    df = pd.DataFrame([record])
    df[DATETIME_COLUMN] = pd.to_datetime(df[DATETIME_COLUMN], utc=True)

    # Coerce numeric columns
    numeric_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    log.info(f"Fetched data for '{CITY}' → shape {df.shape}")
    return df


# ============================================================================
# 2. Cleaning & Imputation
# ============================================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce types, impute missing values."""
    df = df.copy()
    df = df.drop_duplicates(subset=[DATETIME_COLUMN, CITY_COLUMN], keep="last")

    numeric_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Per-city forward-fill then backward-fill residual gaps
    df = df.sort_values([CITY_COLUMN, DATETIME_COLUMN])
    df[numeric_cols] = df.groupby(CITY_COLUMN)[numeric_cols].transform(
        lambda s: s.ffill().bfill()
    )

    # Final fallback — fill remaining NaNs with column median
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.warning(f"Filled {col} NaNs with median ({median_val:.2f})")

    # Drop rows where target is still missing
    before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with missing '{TARGET_COLUMN}'")

    log.info(f"Cleaned data → {df.shape}")
    return df.reset_index(drop=True)


# ============================================================================
# 3. Feature Engineering
# ============================================================================
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from the timestamp column.

    Creates: hour, day, month
    """
    df = df.copy()
    if DATETIME_COLUMN not in df.columns:
        log.warning(f"'{DATETIME_COLUMN}' column missing — skipping time features")
        return df

    dt = pd.to_datetime(df[DATETIME_COLUMN])
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month

    log.info("Created time features: hour, day, month")
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived analytical features.

    Creates:
        aqi_change_rate  — percentage change in AQI from previous row (per city)
        rolling_mean_3   — 3-period rolling mean of AQI (per city)
    """
    df = df.sort_values([CITY_COLUMN, DATETIME_COLUMN]).copy()

    # --- aqi_change_rate (% change, per city) ---
    df["aqi_change_rate"] = df.groupby(CITY_COLUMN)[TARGET_COLUMN].pct_change() * 100
    df["aqi_change_rate"] = df["aqi_change_rate"].fillna(0.0)

    # --- rolling_mean_3 (window=3, per city) ---
    df["rolling_mean_3"] = df.groupby(CITY_COLUMN)[TARGET_COLUMN].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )

    log.info("Created derived features: aqi_change_rate, rolling_mean_3")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pipeline."""
    df = create_time_features(df)
    df = create_derived_features(df)
    df = df.dropna().reset_index(drop=True)
    log.info(f"Feature engineering complete → {df.shape}")
    return df


# ============================================================================
# 4. Hopsworks Feature Store (hsfs) Integration
# ============================================================================
def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """
    Authenticate with Hopsworks using the API key from the .env file
    and return the feature-store handle.
    """
    if hsfs is None:
        raise RuntimeError("hsfs is not installed. Install hsfs to use Hopsworks feature store.")

    project = login_to_hopsworks()
    fs: hsfs.feature_store.FeatureStore = project.get_feature_store()
    log.info("Connected to Hopsworks feature store")
    return fs


def upsert_to_feature_store(df: pd.DataFrame) -> None:
    """
    Create (or get) the ``aqi_features`` feature group and insert *df*.

    Primary key : timestamp
    """
    fs = get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI, weather, and derived features from OpenWeatherMap",
        primary_key=[DATETIME_COLUMN],
        event_time=DATETIME_COLUMN,
    )

    fg.insert(df, write_options={"wait_for_job": True})
    log.info(
        f"Inserted {len(df)} rows into feature group "
        f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION}"
    )


# ============================================================================
# 5. Historical Backfill
# ============================================================================
def _to_unix(dt: datetime) -> int:
    """Convert a datetime to a UTC Unix timestamp (int)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def fetch_historical_pollution(
    lat: float,
    lon: float,
    start_ts: int,
    end_ts: int,
) -> List[Dict[str, Any]]:
    """
    Fetch historical air-pollution data between two Unix timestamps.

    Uses the OpenWeatherMap ``/air_pollution/history`` endpoint which
    returns hourly data points for the requested range.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_ts,
        "end": end_ts,
        "appid": OPENWEATHER_API_KEY,
    }
    resp = requests.get(OPENWEATHER_HISTORY_POLLUTION_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("list", [])


def fetch_historical_weather(
    lat: float,
    lon: float,
    dt_unix: int,
) -> Dict[str, Any]:
    """
    Fetch historical weather for a single timestamp.

    Uses the OpenWeatherMap ``/onecall/timemachine`` endpoint.
    Returns a flat dict with temperature, humidity, pressure, wind_speed.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "dt": dt_unix,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    resp = requests.get(OPENWEATHER_TIMEMACHINE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # The /timemachine response nests current-hour data under "data[0]"
    entry = data.get("data", [{}])[0]
    return {
        "temperature": entry.get("temp"),
        "humidity": entry.get("humidity"),
        "pressure": entry.get("pressure"),
        "wind_speed": entry.get("wind_speed"),
    }


def _build_backfill_records(
    start_date: datetime,
    end_date: datetime,
) -> List[Dict[str, Any]]:
    """
    Build hourly records for the configured city between *start_date* and
    *end_date*.

    1. Bulk-fetch pollution history for the full range (single API call).
    2. For each hourly pollution entry, fetch corresponding weather.
    3. Merge into one record per hour.
    """
    lat, lon = _get_coords()
    start_ts = _to_unix(start_date)
    end_ts = _to_unix(end_date)

    # --- Pollution (bulk) ---
    try:
        pollution_entries = fetch_historical_pollution(lat, lon, start_ts, end_ts)
    except requests.RequestException as exc:
        log.error(f"Historical pollution fetch failed for '{CITY}': {exc}")
        return []

    records: List[Dict[str, Any]] = []
    for entry in pollution_entries:
        entry_ts = entry.get("dt", 0)
        components = entry.get("components", {})
        main_aqi = entry.get("main", {}).get("aqi")

        record: Dict[str, Any] = {
            DATETIME_COLUMN: datetime.fromtimestamp(entry_ts, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            CITY_COLUMN: CITY.strip().lower(),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "co": components.get("co"),
            TARGET_COLUMN: main_aqi,
        }

        # --- Weather (per timestamp) ---
        try:
            weather = fetch_historical_weather(lat, lon, entry_ts)
            record.update(weather)
        except requests.RequestException:
            # Fill weather fields with None; will be imputed during cleaning
            record.update(
                {"temperature": None, "humidity": None, "pressure": None, "wind_speed": None}
            )

        records.append(record)

        # Respect API rate limits (≤60 calls/min on free tier)
        time.sleep(1.1)

    log.info(f"Built {len(records)} historical records for '{CITY}'")
    return records


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate primary keys (timestamp) keeping the latest row.

    Ensures uniqueness required by the Hopsworks feature group.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[DATETIME_COLUMN], keep="last")
    dropped = before - len(df)
    if dropped:
        log.info(f"Deduplicated {dropped} rows on primary key '{DATETIME_COLUMN}'")
    return df.reset_index(drop=True)


def run_backfill(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    batch_size: int = 500,
    push_to_hopsworks: bool = True,
) -> pd.DataFrame:
    """
    Historical backfill: loop through dates hourly and generate a dataset.

    Parameters
    ----------
    start_date : str or datetime
        Start of the backfill range (inclusive).  Strings are parsed as
        ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``.
    end_date : str or datetime
        End of the backfill range (inclusive).
    batch_size : int
        Number of rows per Hopsworks insert batch (default 500).
    push_to_hopsworks : bool
        If True, upsert the result to the feature store.

    Returns
    -------
    pd.DataFrame
        The fully-engineered, deduplicated backfill DataFrame.
    """
    # --- Parse dates ---
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if end_date <= start_date:
        raise ValueError(f"end_date ({end_date}) must be after start_date ({start_date})")

    total_hours = int((end_date - start_date).total_seconds() / 3600)

    log.info("=" * 60)
    log.info("BACKFILL PIPELINE — START")
    log.info(
        f"City  : {CITY} (lat={LAT}, lon={LON})"
    )
    log.info(
        f"Range : {start_date:%Y-%m-%d %H:%M} → {end_date:%Y-%m-%d %H:%M}  "
        f"({total_hours} hours)"
    )
    log.info("=" * 60)

    # --- Collect raw records ---
    log.info(f"Backfilling '{CITY}' …")
    all_records = _build_backfill_records(start_date, end_date)

    if not all_records:
        log.error("No historical data collected — aborting backfill")
        return pd.DataFrame()

    raw_df = pd.DataFrame(all_records)
    raw_df[DATETIME_COLUMN] = pd.to_datetime(raw_df[DATETIME_COLUMN], utc=True)

    # --- Clean ---
    clean_df = clean_data(raw_df)

    # --- Deduplicate on primary key (timestamp) ---
    clean_df = _deduplicate(clean_df)

    # --- Engineer features ---
    feat_df = engineer_features(clean_df)

    # --- Persist locally ---
    backfill_path = str(DATA_DIR / f"backfill_{start_date:%Y%m%d}_{end_date:%Y%m%d}.parquet")
    save_dataframe(feat_df, backfill_path)

    # --- Push to Hopsworks in batches ---
    if push_to_hopsworks:
        try:
            total_rows = len(feat_df)
            for batch_start in range(0, total_rows, batch_size):
                batch_end = min(batch_start + batch_size, total_rows)
                batch_df = feat_df.iloc[batch_start:batch_end]
                upsert_to_feature_store(batch_df)
                log.info(
                    f"Upserted batch [{batch_start}:{batch_end}] "
                    f"({len(batch_df)} rows)"
                )
        except Exception as exc:
            log.error(f"Hopsworks backfill upsert failed: {exc}")
            log.info(f"Full backfill saved locally → {backfill_path}")

    log.info("=" * 60)
    log.info(f"BACKFILL PIPELINE — END  ({len(feat_df)} rows)")
    log.info("=" * 60)
    return feat_df


# ============================================================================
# 6. Pipeline Runner
# ============================================================================
def run_feature_pipeline() -> pd.DataFrame:
    """
    End-to-end pipeline:
        fetch (OpenWeather) → clean → engineer features → store (Hopsworks)
    """
    log.info("=" * 60)
    log.info("FEATURE PIPELINE — START")
    log.info("=" * 60)

    # --- Fetch ---
    raw_df = fetch_data()
    if raw_df.empty:
        log.error("No data fetched — aborting pipeline")
        return raw_df

    # --- Clean ---
    clean_df = clean_data(raw_df)

    # --- Engineer ---
    feat_df = engineer_features(clean_df)

    # --- Persist locally ---
    save_dataframe(feat_df, str(DATA_DIR / "features_latest.parquet"))

    # --- Push to Hopsworks ---
    try:
        upsert_to_feature_store(feat_df)
    except Exception as exc:
        log.error(f"Hopsworks upsert failed: {exc}")
        log.info("Features saved locally — retry Hopsworks upsert later")

    log.info("=" * 60)
    log.info("FEATURE PIPELINE — END")
    log.info("=" * 60)
    return feat_df


def refresh_latest_features(push_to_hopsworks: bool = False) -> pd.DataFrame:
    """
    Refresh one latest feature row from OpenWeather and persist locally.

    This is a lightweight path for serving-time inference to ensure predictions
    are driven by fresh external data instead of stale cached rows.
    """
    raw_df = fetch_data()
    if raw_df.empty:
        raise RuntimeError("No data fetched from OpenWeather")

    clean_df = clean_data(raw_df)
    feat_df = engineer_features(clean_df)

    if feat_df.empty:
        raise RuntimeError("Feature engineering produced no rows")

    save_dataframe(feat_df, str(DATA_DIR / "features_latest.parquet"))

    if push_to_hopsworks:
        try:
            upsert_to_feature_store(feat_df)
        except Exception as exc:
            log.warning(f"Hopsworks upsert during refresh failed: {exc}")

    return feat_df


if __name__ == "__main__":
    run_feature_pipeline()

"""
download_openweather_csv.py
===========================
Downloads historical air-pollution data from the OpenWeatherMap free API
and saves it as  data/air_quality_historical.csv  — exactly the format
that augment_data.py expects.

Free-tier coverage
------------------
  /data/2.5/air_pollution/history  →  hourly data for the last 30 days

Weather columns (temperature, humidity, pressure, wind_speed) are
synthesised by augment_data.py, so we don't need historical weather here.

Usage
-----
    python download_openweather_csv.py              # last 30 days (free limit)
    python download_openweather_csv.py --days 7     # last 7 days

After it finishes, run:
    python augment_data.py
    python main.py --pipeline training --local
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
LAT     = float(os.getenv("LAT", "24.8607"))
LON     = float(os.getenv("LON", "67.0011"))
CITY    = os.getenv("CITY", "Karachi")

DATA_DIR  = Path(__file__).parent / "data"
OUT_CSV   = DATA_DIR / "air_quality_historical.csv"

POLLUTION_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"

# ---------------------------------------------------------------------------
# US AQI helper (EPA standard breakpoints for PM2.5 ug/m3)
# ---------------------------------------------------------------------------
_PM25_BREAKPOINTS = [
    # (C_low, C_high, I_low, I_high)
    (0.0,   12.0,   0,   50),
    (12.1,  35.4,  51,  100),
    (35.5,  55.4, 101,  150),
    (55.5, 150.4, 151,  200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def pm25_to_us_aqi(pm25: float) -> int:
    """Convert a PM2.5 concentration (ug/m3) to the US AQI integer (0-500)."""
    if pm25 < 0:
        pm25 = 0.0
    for c_lo, c_hi, i_lo, i_hi in _PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            aqi = (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
            return round(aqi)
    return 500  # beyond hazardous


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_pollution_history(start_ts: int, end_ts: int) -> list:
    """
    Call /air_pollution/history for the configured city.
    Returns the raw 'list' array from the API response.
    """
    params = {
        "lat":   LAT,
        "lon":   LON,
        "start": start_ts,
        "end":   end_ts,
        "appid": API_KEY,
    }
    resp = requests.get(POLLUTION_HISTORY_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    entries = data.get("list", [])
    return entries


# ---------------------------------------------------------------------------
# Build daily CSV rows from hourly API entries
# ---------------------------------------------------------------------------

def hourly_to_daily(entries: list) -> pd.DataFrame:
    """
    Convert hourly OpenWeatherMap pollution entries to daily-average rows
    in the column format expected by augment_data.py:

        date, pm2_5, pm10, carbon_monoxide, nitrogen_dioxide,
        ozone, us_aqi, sulphur_dioxide
    """
    rows = []
    for entry in entries:
        dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc)
        comp = entry.get("components", {})
        rows.append({
            "datetime":         dt,
            "pm2_5":            comp.get("pm2_5"),
            "pm10":             comp.get("pm10"),
            "carbon_monoxide":  comp.get("co"),
            "nitrogen_dioxide": comp.get("no2"),
            "ozone":            comp.get("o3"),
            "sulphur_dioxide":  comp.get("so2"),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = df["datetime"].dt.date

    # Aggregate to daily means
    daily = (
        df.groupby("date", as_index=False)
        .agg({
            "pm2_5":            "mean",
            "pm10":             "mean",
            "carbon_monoxide":  "mean",
            "nitrogen_dioxide": "mean",
            "ozone":            "mean",
            "sulphur_dioxide":  "mean",
        })
    )

    # Compute US AQI from daily-average PM2.5
    daily["us_aqi"] = daily["pm2_5"].apply(pm25_to_us_aqi)

    # Round sensibly
    float_cols = ["pm2_5", "pm10", "carbon_monoxide",
                  "nitrogen_dioxide", "ozone", "sulphur_dioxide"]
    daily[float_cols] = daily[float_cols].round(4)

    # Final column order matching augment_data.py rename_map
    daily = daily[[
        "date", "pm2_5", "pm10",
        "carbon_monoxide", "nitrogen_dioxide", "ozone",
        "us_aqi", "sulphur_dioxide",
    ]]

    return daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download OpenWeatherMap historical AQI data to CSV."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of past days to download (max 30 on free plan). Default: 30",
    )
    args = parser.parse_args()

    # ---- Validate ----------------------------------------------------------
    if not API_KEY:
        print(
            "\n[ERROR] OPENWEATHER_API_KEY is not set.\n"
            "  Open your .env file and add:\n"
            "  OPENWEATHER_API_KEY=your_actual_key_here\n"
        )
        sys.exit(1)

    if args.days > 30:
        print(
            f"[WARNING] Free-tier /air_pollution/history only covers the last "
            f"30 days. Clamping {args.days} -> 30."
        )
        args.days = 30

    # ---- Time range --------------------------------------------------------
    now_utc   = datetime.now(tz=timezone.utc)
    end_dt    = now_utc
    start_dt  = now_utc - timedelta(days=args.days)

    start_ts  = int(start_dt.timestamp())
    end_ts    = int(end_dt.timestamp())

    print(f"\n{'='*60}")
    print(f"  City   : {CITY}  (lat={LAT}, lon={LON})")
    print(f"  Range  : {start_dt.date()}  ->  {end_dt.date()}  ({args.days} days)")
    print(f"  Output : {OUT_CSV}")
    print(f"{'='*60}\n")

    # ---- Fetch -------------------------------------------------------------
    print("Fetching hourly pollution history from OpenWeatherMap ...")
    try:
        entries = fetch_pollution_history(start_ts, end_ts)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 401:
            print(
                "\n[ERROR 401] Invalid API key.\n"
                "  Double-check OPENWEATHER_API_KEY in your .env file.\n"
                "  New keys can take up to 2 hours to activate.\n"
            )
        elif status == 429:
            print(
                "\n[ERROR 429] Rate limit exceeded.\n"
                "  Wait a moment and try again.\n"
            )
        else:
            print(f"\n[ERROR {status}] {exc}\n")
        sys.exit(1)
    except requests.RequestException as exc:
        print(f"\n[ERROR] Network problem: {exc}\n")
        sys.exit(1)

    print(f"  Received {len(entries)} hourly readings")

    if not entries:
        print("\n[WARNING] No data returned. Check your API key and coordinates.")
        sys.exit(1)

    # ---- Process -----------------------------------------------------------
    print("Aggregating hourly -> daily ...")
    df = hourly_to_daily(entries)
    print(f"  {len(df)} daily rows  |  columns: {list(df.columns)}")

    # ---- Show preview ------------------------------------------------------
    print("\nPreview (last 5 days):")
    print(df.tail().to_string(index=False))

    # ---- Save --------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists():
        # Merge with any existing rows and deduplicate by date
        existing = pd.read_csv(OUT_CSV, parse_dates=["date"])
        existing["date"] = existing["date"].dt.date
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        combined.to_csv(OUT_CSV, index=False)
        print(f"\nMerged with existing data -> {len(combined)} total rows saved to:\n  {OUT_CSV}")
    else:
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {len(df)} rows to:\n  {OUT_CSV}")

    print(
        "\nNext steps:\n"
        "  1. python augment_data.py                       <- process CSV -> parquet\n"
        "  2. python main.py --pipeline training --local   <- retrain models\n"
        "  3. python main.py --serve streamlit             <- launch dashboard\n"
    )


if __name__ == "__main__":
    main()

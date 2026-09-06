"""
Data Augmentation Script
========================
Takes the 1,298-row air_quality_historical.csv, maps columns to our
feature schema, synthesises missing weather variables with realistic
Karachi seasonal patterns, derives time & rolling features, and
augments to ~10 k rows using Gaussian noise.

Output: data/features_history.parquet  (ready for training_pipeline)
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
SRC_CSV = DATA_DIR / "air_quality_historical.csv"
OUT_PARQUET = DATA_DIR / "features_history.parquet"

# ---------------------------------------------------------------------------
# 1. Load & rename
# ---------------------------------------------------------------------------
raw = pd.read_csv(SRC_CSV, parse_dates=["date"])
raw = raw.dropna(subset=["pm2_5"])  # drop the 3 fully-NaN rows

rename_map = {
    "date": "timestamp",
    "carbon_monoxide": "co",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "ozone": "o3",
    "us_aqi": "aqi",
}
raw = raw.rename(columns=rename_map)

# Keep only columns we need + extras we'll drop later
KEEP = ["timestamp", "pm2_5", "pm10", "co", "no2", "o3", "aqi",
        "so2", "aerosol_optical_depth", "dust", "uv_index", "european_aqi"]
raw = raw[[c for c in KEEP if c in raw.columns]]

print(f"Raw rows after NaN drop: {len(raw)}")

# ---------------------------------------------------------------------------
# 2. Synthesise realistic Karachi weather columns
# ---------------------------------------------------------------------------
n = len(raw)
day_of_year = raw["timestamp"].dt.dayofyear.values

# Temperature (°C): Karachi ranges ~18°C (Jan) to ~35°C (Jun)
temp_base = 26.5 + 8.5 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
raw["temperature"] = temp_base + np.random.normal(0, 1.5, n)

# Humidity (%): higher in monsoon (Jul-Sep), lower in winter
hum_base = 55 + 15 * np.sin(2 * np.pi * (day_of_year - 200) / 365)
raw["humidity"] = np.clip(hum_base + np.random.normal(0, 8, n), 20, 95)

# Pressure (hPa): relatively stable, slight seasonal variation
pres_base = 1012 + 4 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
raw["pressure"] = pres_base + np.random.normal(0, 2, n)

# Wind speed (m/s): monsoon winds stronger
ws_base = 4.0 + 1.5 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
raw["wind_speed"] = np.clip(ws_base + np.random.normal(0, 1.2, n), 0.3, 15)

# ---------------------------------------------------------------------------
# 3. Add city column & derive time features
# ---------------------------------------------------------------------------
raw["city"] = "karachi"

# Since the CSV is daily data, assign a fixed hour (e.g. 12:00 noon)
raw["hour"] = 12
raw["day"] = raw["timestamp"].dt.day
raw["month"] = raw["timestamp"].dt.month

# Sort by time for rolling features
raw = raw.sort_values("timestamp").reset_index(drop=True)

# Rolling mean of pm2_5 over last 3 observations
raw["rolling_mean_3"] = raw["pm2_5"].rolling(window=3, min_periods=1).mean()

# AQI change rate (difference from previous row, 0 for first row)
raw["aqi_change_rate"] = raw["pm2_5"].diff().fillna(0)

# ---------------------------------------------------------------------------
# 4. Select final columns in the schema order
# ---------------------------------------------------------------------------
FINAL_COLS = [
    "timestamp", "city",
    "temperature", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "o3", "co",
    "aqi",
    "hour", "day", "month",
    "aqi_change_rate", "rolling_mean_3",
]
base = raw[FINAL_COLS].copy()

# Fill any remaining NaN with column medians
for c in base.columns:
    if base[c].dtype in ("float64", "float32"):
        base[c] = base[c].fillna(base[c].median())

print(f"Base dataset: {base.shape}")
print(f"Columns: {list(base.columns)}")

# ---------------------------------------------------------------------------
# 5. Augment with Gaussian noise  →  ~10 k rows
# ---------------------------------------------------------------------------
NOISE_COPIES = 7          # 1295 * (1 + 7) ≈ 10,360 rows
NOISE_FRAC = 0.05         # 5 % standard-deviation relative to column std

augmented_frames = [base]  # start with original

# Columns to add noise to (skip identifiers / categorical)
noise_cols = [
    "temperature", "humidity", "pressure", "wind_speed",
    "pm2_5", "pm10", "no2", "o3", "co", "aqi",
    "aqi_change_rate", "rolling_mean_3",
]

col_stds = base[noise_cols].std()

for i in range(NOISE_COPIES):
    noisy = base.copy()
    for c in noise_cols:
        sigma = col_stds[c] * NOISE_FRAC
        noise = np.random.normal(0, sigma, len(noisy))
        noisy[c] = noisy[c] + noise
        # Keep non-negative for physical quantities
        if c not in ("aqi_change_rate",):
            noisy[c] = noisy[c].clip(lower=0)
    # Slightly jitter hour ±2 for variety
    noisy["hour"] = np.clip(noisy["hour"] + np.random.randint(-2, 3, len(noisy)), 0, 23)
    augmented_frames.append(noisy)

df_aug = pd.concat(augmented_frames, ignore_index=True)

# Shuffle so train/test split is well-mixed
df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nAugmented dataset: {df_aug.shape}")
print(f"PM2.5 stats:\n{df_aug['pm2_5'].describe()}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
df_aug.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved to {OUT_PARQUET}")
print("Done! Run  python main.py --pipeline training --local  to retrain.")

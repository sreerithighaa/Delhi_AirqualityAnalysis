""" 
You do not run this file directly. eda.py and visualisation.py import it.
Result: two tables ready for analysis
    df    = all cleaned rows (2023-2026)
    data  = only 2024 and 2025 (used for all analysis)
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent          # the folder this file is in
RAW_DIR = BASE / "data" / "raw"
TABLE_DIR = BASE / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)    # create the output folder if missing

YEARS = [2024, 2025]                            # the two years we compare
FOCUS_MONTHS = [10, 11, 12]                     # October, November, December
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Provisional groups (NOT verified land use)
STATION_GROUPS = {
    "Bawana": "Industrial", "Jahangirpuri": "Industrial",
    "Anand Vihar": "Traffic hub", "ITO": "Traffic hub",
    "Dwarka-Sector 8": "Residential/commercial",
    "Punjabi Bagh": "Residential/commercial",
    "R K Puram": "Residential/commercial",
}
STATION_ORDER = ["Bawana", "Jahangirpuri", "Anand Vihar", "ITO",
                 "Dwarka-Sector 8", "Punjabi Bagh", "R K Puram"]
GROUP_ORDER = ["Industrial", "Traffic hub", "Residential/commercial"]

THRESHOLD = 60                  # exploratory hourly PM2.5 level (ug/m3), NOT an official limit
SENSITIVITY = [30, 60, 90, 120]  # other levels used to test the result
PM_CEILING = 990                # PM readings at or above this look like a sensor maximum (998/1000)
WIND_SPEED_MAX = 15             # wind speeds above this (m/s) are treated as sensor faults
MIN_N = 30                      # minimum readings needed to draw/report a category
WIND_STATION = "Anand Vihar"    # station used for the wind-direction chart
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def wind_sector(degrees):
    """Turn a wind direction (0-360 degrees) into one of 8 compass sectors.
    Each sector is 45 degrees wide and centred on N=0, NE=45, E=90 ... NW=315.
    Adding 22.5 shifts the boundaries so that N covers 337.5-360 AND 0-22.5.
    The % 8 makes 360 degrees wrap round to N, so 0 and 360 both give N."""
    if pd.isna(degrees):
        return np.nan
    number = int(((degrees % 360) + 22.5) // 45) % 8
    return SECTORS[number]

#Load and clean
print("Cleaning: loading and cleaning data ...")
readings = pd.read_csv(RAW_DIR / "caaqms_hourly.csv", low_memory=False)
stations = pd.read_csv(RAW_DIR / "stations.csv")

# Timestamps look like 2024-01-01T00:00:00+05:30. All are Indian time (+05:30),
# so we keep the first 19 characters and turn them into a date-time.
readings["ts"] = pd.to_datetime(readings["timestamp"].str[:19])

# Add the station name to every row (the readings file only has station_id)
df = readings.merge(stations[["station_id", "station_name"]], on="station_id", how="left")
rows_in = len(df)

# Remove duplicate station + time records (the supplied file has none)
df = df.drop_duplicates(subset=["station_id", "ts"])
duplicates_removed = rows_in - len(df)

# Sort by station, then by time (keeps the rows in a tidy, predictable order)
df = df.sort_values(["station_id", "ts"]).reset_index(drop=True)

# Keep the ORIGINAL columns untouched. Create new "_clean" columns where
# invalid readings become empty (NaN). Nothing is filled in or guessed.
df["pm25_clean"] = df["pm25"].where((df["pm25"] >= 0) & (df["pm25"] < PM_CEILING))
df["pm10_clean"] = df["pm10"].where((df["pm10"] >= 0) & (df["pm10"] < PM_CEILING))
df["wind_speed_clean"] = df["wind_speed"].where((df["wind_speed"] >= 0) & (df["wind_speed"] <= WIND_SPEED_MAX))
df["temperature_clean"] = df["temperature"].where(df["temperature"].between(-5, 50))
df["humidity_clean"] = df["humidity"].where(df["humidity"].between(0, 100))
df["wind_direction_clean"] = df["wind_direction"].where(df["wind_direction"].between(0, 360))

# PM2.5 is part of PM10, so PM2.5 > PM10 is physically inconsistent: flag those hours
df["pm25_above_pm10"] = df["pm25_clean"] > df["pm10_clean"]

# Cleaning log: how many readings were affected by each rule
cleaning_log = pd.DataFrame([
    {"step": "Input rows", "n_affected": rows_in},
    {"step": "Duplicate station-timestamp rows removed", "n_affected": duplicates_removed},
    {"step": "PM2.5 negative or >= 990 set to missing in pm25_clean", "n_affected": int((df["pm25"].notna() & df["pm25_clean"].isna()).sum())},
    {"step": "PM10 negative or >= 990 set to missing in pm10_clean", "n_affected": int((df["pm10"].notna() & df["pm10_clean"].isna()).sum())},
    {"step": "Wind speed outside 0-15 m/s set to missing", "n_affected": int((df["wind_speed"].notna() & df["wind_speed_clean"].isna()).sum())},
    {"step": "Temperature outside -5 to 50 set to missing", "n_affected": int((df["temperature"].notna() & df["temperature_clean"].isna()).sum())},
    {"step": "Humidity outside 0-100 set to missing", "n_affected": int((df["humidity"].notna() & df["humidity_clean"].isna()).sum())},
    {"step": "Wind direction outside 0-360 set to missing", "n_affected": int((df["wind_direction"].notna() & df["wind_direction_clean"].isna()).sum())},
    {"step": "PM2.5 <= 1.0 (possible sensor floor) - kept, only counted", "n_affected": int((df["pm25"] <= 1.0).sum())},
    {"step": "PM2.5 > PM10 hours - excluded from PM2.5 vs PM10 chart only", "n_affected": int(df["pm25_above_pm10"].sum())},
    {"step": "Missing values filled in (imputed)", "n_affected": 0},
])

# Add simple helper columns
df["year"] = df["ts"].dt.year
df["month"] = df["ts"].dt.month
df["hour"] = df["ts"].dt.hour
df["station_group"] = df["station_name"].map(STATION_GROUPS)
df["wind_sector"] = df["wind_direction_clean"].apply(wind_sector)

# 'data' = only 2024 and 2025, which is what all the analysis uses
data = df[df["year"].isin(YEARS)].copy()
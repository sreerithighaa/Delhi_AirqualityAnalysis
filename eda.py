import numpy as np
import pandas as pd

from cleaning import (data, cleaning_log, TABLE_DIR, YEARS, FOCUS_MONTHS, STATION_ORDER,
                      GROUP_ORDER, THRESHOLD, SENSITIVITY, MIN_N, WIND_STATION, SECTORS)


def save_table(table, filename):
    """Save a table as CSV in outputs/tables."""
    table.to_csv(TABLE_DIR / filename, index=False)
    print("  table saved:", filename)


def spearman_rho(a, b):
    """Spearman correlation = ordinary (Pearson) correlation of the RANKS.
    Written by hand so SciPy is not needed."""
    return a.rank().corr(b.rank())


def describe(values):
    """Descriptive statistics for one column of numbers (missing values ignored)."""
    v = values.dropna()
    return {"n_valid": len(v), "mean": v.mean(), "median": v.median(), "std": v.std(),
            "min": v.min(), "q1": v.quantile(0.25), "q3": v.quantile(0.75),
            "iqr": v.quantile(0.75) - v.quantile(0.25),
            "p90": v.quantile(0.90), "p95": v.quantile(0.95), "max": v.max()}


print("EDA: building tables ...")
save_table(cleaning_log, "cleaning_log.csv")

# Table: coverage (how much valid PM2.5 data each station has per year)
hours_in_year = {2024: 8784, 2025: 8760}        # 2024 is a leap year
rows = []
for station in STATION_ORDER:
    for year in YEARS:
        part = data[(data["station_name"] == station) & (data["year"] == year)]
        valid = part.dropna(subset=["pm25_clean"]).sort_values("ts")
        gaps = valid["ts"].diff().dt.total_seconds() / 3600 - 1   # missing hours between valid readings
        rows.append({
            "station": station, "year": year, "rows": len(part),
            "valid_pm25": len(valid), "valid_pm10": int(part["pm10_clean"].notna().sum()),
            "valid_wind_speed": int(part["wind_speed_clean"].notna().sum()),
            "valid_temperature": int(part["temperature_clean"].notna().sum()),
            "pm25_completeness_pct": round(100 * len(valid) / hours_in_year[year], 1),
            "longest_pm25_gap_hours": int(gaps.max()),
            "weather_available": part["wind_speed_clean"].notna().any(),
        })
coverage = pd.DataFrame(rows)
save_table(coverage, "coverage_station_year.csv")

# Table: % of readings missing in the ORIGINAL columns, per station and year
variables = ["pm25", "pm10", "temperature", "humidity", "wind_speed", "wind_direction"]
missing = data[variables].isna().groupby([data["station_name"], data["year"]]).mean() * 100
missing = missing.round(1).reset_index()
save_table(missing, "missing_percent_by_station_year.csv")

# Table: PM2.5 descriptive statistics per station and year
rows = []
for station in STATION_ORDER:
    for year in YEARS:
        part = data[(data["station_name"] == station) & (data["year"] == year)]
        row = {"station": station, "year": year}
        row.update(describe(part["pm25_clean"]))
        rows.append(row)
save_table(pd.DataFrame(rows), "pm25_summary_station_year.csv")

# Table (Chart 1): median PM2.5 per station, 2024 vs 2025
medians = data.groupby(["station_name", "year"])["pm25_clean"].median().unstack()
counts = data.groupby(["station_name", "year"])["pm25_clean"].count().unstack()
yoy = pd.DataFrame({
    "station": STATION_ORDER,
    "median_2024": medians.loc[STATION_ORDER, 2024].values,
    "median_2025": medians.loc[STATION_ORDER, 2025].values,
    "n_2024": counts.loc[STATION_ORDER, 2024].values,
    "n_2025": counts.loc[STATION_ORDER, 2025].values,
})
yoy["change_ug_m3"] = yoy["median_2025"] - yoy["median_2024"]
yoy["change_pct"] = 100 * yoy["change_ug_m3"] / yoy["median_2024"]
save_table(yoy, "chart1_station_yoy_median.csv")

# Table: fair 2024 vs 2025 check using ONLY calendar hours valid in BOTH years
valid_data = data.dropna(subset=["pm25_clean"]).copy()
valid_data = valid_data[~((valid_data["ts"].dt.month == 2) & (valid_data["ts"].dt.day == 29))]  # drop 29 Feb
valid_data["slot"] = valid_data["ts"].dt.strftime("%m-%d %H")      # e.g. "11-05 08" = 5 Nov, 08:00
rows = []
for station in STATION_ORDER:
    part = valid_data[valid_data["station_name"] == station]
    table = part.pivot_table(index="slot", columns="year", values="pm25_clean").dropna()
    rows.append({"station": station, "matched_hours": len(table),
                 "median_2024": table[2024].median(), "median_2025": table[2025].median()})
matched = pd.DataFrame(rows)
matched["change_pct"] = 100 * (matched["median_2025"] - matched["median_2024"]) / matched["median_2024"]
save_table(matched, "yoy_matched_hours_comparison.csv")

# Table (Chart 2): PM2.5 vs PM10 using the SAME station and the SAME hour
both = data.dropna(subset=["pm25_clean", "pm10_clean"])             # both pollutants valid
pairs = both[~both["pm25_above_pm10"]]                              # remove PM2.5 > PM10 hours
a = pairs["pm25_clean"]
b = pairs["pm10_clean"]
slope, intercept = np.polyfit(b, a, 1)                              # trend line: a = slope*b + intercept
ratio = a / b
pm_stats = {
    "pairs_with_both_valid": len(both), "pairs_excluded_pm25_above_pm10": len(both) - len(pairs),
    "n_pairs": len(pairs), "pm25_median": a.median(), "pm10_median": b.median(),
    "median_pm25_to_pm10_ratio": ratio.median(),
    "ratio_q1": ratio.quantile(0.25), "ratio_q3": ratio.quantile(0.75),
    "pearson_r": a.corr(b), "spearman_rho": spearman_rho(a, b),
    "trend_slope": slope, "trend_intercept": intercept,
}
save_table(pd.DataFrame({"measure": list(pm_stats.keys()), "value": list(pm_stats.values())}),
           "chart2_pm25_pm10_summary.csv")

# Table (Chart 3): median PM2.5 for every hour of the day, by station group
hourly = data.groupby(["year", "station_group", "hour"])["pm25_clean"].median().reset_index()
hourly = hourly.rename(columns={"pm25_clean": "median"})
save_table(hourly, "chart3_hourly_median_by_group.csv")

rows = []                                       # peak hour and lowest hour for each group-year
for year in YEARS:
    for group in GROUP_ORDER:
        part = hourly[(hourly["year"] == year) & (hourly["station_group"] == group)]
        peak = part.loc[part["median"].idxmax()]
        low = part.loc[part["median"].idxmin()]
        rows.append({"year": year, "station_group": group,
                     "peak_hour": int(peak["hour"]), "peak_median": peak["median"],
                     "lowest_hour": int(low["hour"]), "lowest_median": low["median"]})
save_table(pd.DataFrame(rows), "chart3_peak_hours.csv")

# Table (Chart 4): Anand Vihar PM2.5 by wind sector
anand = data[(data["station_name"] == WIND_STATION) & data["wind_direction_clean"].notna()]
rows = []
for year in YEARS:
    for sector in SECTORS:
        part = anand[(anand["year"] == year) & (anand["wind_sector"] == sector)]
        pm = part["pm25_clean"].dropna()
        rows.append({"year": year, "sector": sector, "n_direction_obs": len(part),
                     "n_pm25_valid": len(pm), "pm25_median": pm.median(),
                     "wind_speed_median": part["wind_speed_clean"].median(),
                     "pct_above_threshold": 100 * (pm > THRESHOLD).mean()})
wind_table = pd.DataFrame(rows)
wind_table["share_of_direction_obs_pct"] = (100 * wind_table["n_direction_obs"] /
                                            wind_table.groupby("year")["n_direction_obs"].transform("sum")).round(1)
save_table(wind_table, "chart4_anand_vihar_wind_sectors.csv")

# Table (Chart 5): median PM2.5 per month, all stations pooled
monthly = data.groupby(["year", "month"])["pm25_clean"].median().reset_index()
monthly = monthly.rename(columns={"pm25_clean": "median"})
monthly["n"] = data.groupby(["year", "month"])["pm25_clean"].count().values
save_table(monthly, "chart5_monthly_pooled.csv")

# Table (Chart 6): October-December by station, year and month
rows = []
for station in STATION_ORDER:
    for year in YEARS:
        for month in FOCUS_MONTHS:
            part = data[(data["station_name"] == station) & (data["year"] == year) & (data["month"] == month)]
            valid = part["pm25_clean"].dropna()
            rows.append({"station_name": station, "year": year, "month": month,
                         "median": valid.median(), "n_valid": len(valid),
                         "pct_above": round(100 * (valid > THRESHOLD).mean(), 1)})
focus = pd.DataFrame(rows)
save_table(focus, "chart6_oct_dec_station_month.csv")

# Table (Chart 7): % of valid hours above 60 ug/m3, per station and year
rows = []
for station in STATION_ORDER:
    for year in YEARS:
        valid = data[(data["station_name"] == station) & (data["year"] == year)]["pm25_clean"].dropna()
        rows.append({"station_name": station, "year": year, "valid_hours": len(valid),
                     "hours_above": int((valid > THRESHOLD).sum()),
                     "pct_above": round(100 * (valid > THRESHOLD).mean(), 1)})
freq = pd.DataFrame(rows)
save_table(freq, "chart7_elevated_frequency_station_year.csv")

# Table: does the ranking change if we use a different threshold?
rows = []
for station in STATION_ORDER:
    for year in YEARS:
        valid = data[(data["station_name"] == station) & (data["year"] == year)]["pm25_clean"].dropna()
        row = {"station_name": station, "year": year}
        for level in SENSITIVITY:
            row["above_" + str(level)] = round(100 * (valid > level).mean(), 1)
        rows.append(row)
save_table(pd.DataFrame(rows), "elevated_threshold_sensitivity.csv")

# Table: Spearman correlation of PM2.5 with weather (ITO has no weather data)
rows = []
for station in STATION_ORDER:
    part = data[data["station_name"] == station]
    for weather in ["temperature_clean", "humidity_clean", "wind_speed_clean"]:
        aligned = part[["pm25_clean", weather]].dropna()        # only hours where BOTH are valid
        rho = np.nan
        if len(aligned) >= MIN_N:
            rho = spearman_rho(aligned["pm25_clean"], aligned[weather])
        rows.append({"station": station, "variable": weather.replace("_clean", ""),
                     "n_aligned": len(aligned), "spearman_rho": rho})
save_table(pd.DataFrame(rows), "pm25_weather_spearman_by_station.csv")

# Table: PM2.5 in low wind / cool hours compared with other hours
weather_data = data.dropna(subset=["pm25_clean", "wind_speed_clean", "temperature_clean"])
rows = []
for period in ["Jan-Dec", "Oct-Dec"]:
    if period == "Jan-Dec":
        subset = weather_data
    else:
        subset = weather_data[weather_data["month"].isin(FOCUS_MONTHS)]
    for factor, column, limits in [("wind speed", "wind_speed_clean", [0.5, 1.0, 1.5]),
                                   ("temperature", "temperature_clean", [10, 15, 20])]:
        for limit in limits:
            group = subset[subset[column] < limit]["pm25_clean"]
            reference = subset[subset[column] >= limit]["pm25_clean"]
            rows.append({"factor": factor, "period": period, "threshold": "< " + str(limit),
                         "n_group": len(group), "n_reference": len(reference),
                         "pct_hours_in_group": round(100 * len(group) / len(subset), 1),
                         "median_group": group.median(), "median_reference": reference.median(),
                         "ratio": group.median() / reference.median()})
save_table(pd.DataFrame(rows), "weather_threshold_sensitivity.csv")
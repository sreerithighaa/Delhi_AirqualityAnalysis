"""
RUN THIS FILE:   python visualisation.py
It imports eda.py (which imports cleaning.py), so the whole project runs from here:
    cleaning.py -> eda.py -> visualisation.py
Charts are saved in outputs/charts.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save charts to files instead of opening windows
import matplotlib.pyplot as plt

from cleaning import (BASE, YEARS, MONTH_NAMES, FOCUS_MONTHS, STATION_ORDER, GROUP_ORDER,
                      THRESHOLD, MIN_N, WIND_STATION, SECTORS)
from eda import yoy, pairs, pm_stats, slope, intercept, hourly, wind_table, monthly, focus, freq

CHART_DIR = BASE / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLOURS = {"Industrial": "#7b5ea7", "Traffic hub": "#d1495b",
                 "Residential/commercial": "#2e86ab"}
YEAR_COLOURS = {2024: "#2e86ab", 2025: "#e07a1f"}
MONTH_COLOURS = {10: "#f4a261", 11: "#e76f51", 12: "#9d0208"}

SOURCE = "Source: CAAQMS hourly data, valid hours only. PM2.5 in µg/m³ (units not verified)."
THRESHOLD_NOTE = ("Exploratory threshold: hourly PM2.5 > 60 µg/m³ "
                  "(not an official AQI category, legal limit or health standard).")

# Chart style (same for all charts)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})


def wrap_name(name):
    """Put a long station name on two lines for chart labels."""
    return name.replace(" ", "\n", 1)


def grouped_bars(ax, list_of_values, colours, labels, offsets, width, number_format):
    """Draw side-by-side bars on 'ax' with a value label above each bar.
    list_of_values: one array of heights per series (e.g. one per year)."""
    x = np.arange(len(list_of_values[0]))
    for values, colour, label, offset in zip(list_of_values, colours, labels, offsets):
        ax.bar(x + offset * width, values, width, color=colour, label=label)
        for xi, v in zip(x + offset * width, values):
            if np.isfinite(v):
                ax.text(xi, v, number_format.format(v), ha="center", va="bottom", fontsize=8)
    return x


def save_chart(fig, filename, note=SOURCE):
    """Add the source note at the bottom, tidy the layout and save as PNG."""
    fig.text(0.01, 0.01, note, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(CHART_DIR / filename, dpi=150)
    plt.close(fig)
    print("  chart saved:", filename)


print("Visualisation: drawing charts ...")

# ---------------- Chart 1: median PM2.5 by station, 2024 vs 2025 ----------------
fig, ax = plt.subplots(figsize=(10, 5.5))
x = grouped_bars(ax, [yoy["median_2024"].values, yoy["median_2025"].values],
                 [YEAR_COLOURS[2024], YEAR_COLOURS[2025]], ["2024", "2025"], [-0.5, 0.5], 0.38, "{:.0f}")
ax.set_xticks(x)
ax.set_xticklabels([wrap_name(s) + "\n" + f"{int(n1):,} / {int(n2):,}"
                    for s, n1, n2 in zip(yoy["station"], yoy["n_2024"], yoy["n_2025"])], fontsize=8.5)
ax.set_xlabel("Station   (valid hourly readings n: 2024 / 2025)")
ax.set_ylabel("Median PM2.5 (µg/m³)")
ax.set_ylim(0, yoy[["median_2024", "median_2025"]].max().max() * 1.15)
ax.set_title("Chart 1: Median PM2.5 by station, 2024 vs 2025", fontweight="bold")
ax.legend(title="Year")
save_chart(fig, "chart1_annual_station_median_pm25.png")

# ---------------- Chart 2: PM2.5 vs PM10 (same station, same hour) ----------------
fig, ax = plt.subplots(figsize=(8, 6.5))
xmax = pairs["pm10_clean"].quantile(0.999)      # cut the axes at the 99.9th percentile
ymax = pairs["pm25_clean"].quantile(0.999)
ax.scatter(pairs["pm10_clean"], pairs["pm25_clean"], s=3, alpha=0.1, color="#2e86ab", rasterized=True)
line_x = np.array([0, xmax])
ax.plot(line_x, slope * line_x + intercept, color="black", lw=1.5, label="Trend line (descriptive only)")
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.set_xlabel("PM10 (µg/m³)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Chart 2: PM2.5 vs PM10 (paired hourly readings)", fontweight="bold")
summary = (f"Paired hours: {int(pm_stats['n_pairs']):,}\n"
           f"Median PM2.5: {pm_stats['pm25_median']:.0f}   Median PM10: {pm_stats['pm10_median']:.0f}\n"
           f"Median PM2.5/PM10 ratio: {pm_stats['median_pm25_to_pm10_ratio']:.2f}\n"
           f"Pearson r = {pm_stats['pearson_r']:.2f}   Spearman rho = {pm_stats['spearman_rho']:.2f}")
ax.text(0.03, 0.97, summary, transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="#aaaaaa"))
ax.legend(loc="lower right")
save_chart(fig, "chart2_pm25_vs_pm10.png", SOURCE + " Association, not causation.")

# ---------------- Chart 3: hourly median PM2.5 by station group ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
for ax, year in zip(axes, YEARS):
    for group in GROUP_ORDER:
        part = hourly[(hourly["year"] == year) & (hourly["station_group"] == group)]
        ax.plot(part["hour"], part["median"], marker="o", ms=3, lw=2,
                color=GROUP_COLOURS[group], label=group)
    ax.set_title(str(year))
    ax.set_xlabel("Hour of day (0 = 00:00-00:59)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(0, hourly["median"].max() * 1.1)
axes[0].set_ylabel("Median PM2.5 (µg/m³)")
axes[0].legend()
fig.suptitle("Chart 3: Hourly median PM2.5 by station group", fontweight="bold")
save_chart(fig, "chart3_hourly_profiles.png", SOURCE + " Station groups are provisional.")

# ---------------- Chart 4: PM2.5 by wind direction at Anand Vihar ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
for ax, year in zip(axes, YEARS):
    part = wind_table[wind_table["year"] == year]
    heights = part["pm25_median"].where(part["n_pm25_valid"] >= MIN_N).values   # hide sectors with n < 30
    ax.bar(range(8), np.nan_to_num(heights), color=YEAR_COLOURS[year])
    for i, h in enumerate(heights):
        if np.isfinite(h):
            ax.text(i, h, f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{s}\nn={int(n):,}\n{w:.1f} m/s" for s, n, w in
                        zip(SECTORS, part["n_pm25_valid"], part["wind_speed_median"])], fontsize=8)
    ax.set_xlabel("Wind sector (direction wind comes from)\nn = valid PM2.5 hours; m/s = median wind speed")
    ax.set_title(str(year))
axes[0].set_ylabel("Median PM2.5 (µg/m³)")
fig.suptitle("Chart 4: PM2.5 by wind direction at " + WIND_STATION, fontweight="bold")
save_chart(fig, "chart4_anand_vihar_wind_sectors.png",
           SOURCE + " Sectors with n<30 are not drawn. Wind direction alone cannot identify sources.")

# ---------------- Chart 5: monthly median PM2.5 ----------------
fig, ax = plt.subplots(figsize=(10, 5.5))
for year in YEARS:
    part = monthly[monthly["year"] == year]
    ax.plot(part["month"], part["median"], marker="o", lw=2, color=YEAR_COLOURS[year], label=str(year))
ax.axhline(THRESHOLD, color="grey", ls="--", lw=1)
ax.text(1, THRESHOLD + 4, f"exploratory threshold ({THRESHOLD})", color="grey", fontsize=8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_NAMES)
ax.set_ylabel("Median PM2.5 (µg/m³), all stations pooled")
ax.set_title("Chart 5: Monthly median PM2.5, 2024 vs 2025", fontweight="bold")
ax.legend(title="Year")
save_chart(fig, "chart5_monthly_seasonality.png", SOURCE + " Descriptive pattern only; causes are not tested.")

# ---------------- Chart 6: October-December by station ----------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
# top row = median PM2.5, bottom row = % of hours above 60; left = 2024, right = 2025
for row, (column, title) in enumerate([("median", "Median PM2.5 (µg/m³)"),
                                       ("pct_above", f"% of hours above {THRESHOLD} µg/m³")]):
    for col, year in enumerate(YEARS):
        ax = axes[row, col]
        series = []
        for month in FOCUS_MONTHS:                 # one set of bars per month
            part = focus[(focus["year"] == year) & (focus["month"] == month)].set_index("station_name")
            series.append(part.loc[STATION_ORDER, column].values)
        x = grouped_bars(ax, series, [MONTH_COLOURS[m] for m in FOCUS_MONTHS],
                         [MONTH_NAMES[m - 1] for m in FOCUS_MONTHS], [-1, 0, 1], 0.27, "{:.0f}")
        ax.set_xticks(x)
        ax.set_xticklabels([wrap_name(s) for s in STATION_ORDER], fontsize=8)
        ax.set_title(f"{title} - {year}", fontsize=10)
        if column == "pct_above":
            ax.set_ylim(0, 115)
        else:
            ax.set_ylim(0, focus["median"].max() * 1.15)
        if row == 0 and col == 0:
            ax.legend(title="Month")
fig.suptitle("Chart 6: October-December comparison by station", fontweight="bold")
save_chart(fig, "chart6_oct_dec_comparison.png", SOURCE + " " + THRESHOLD_NOTE)

# ---------------- Chart 7: share of hours above the threshold ----------------
fig, ax = plt.subplots(figsize=(10, 5.5))
f2024 = freq[freq["year"] == 2024].set_index("station_name").loc[STATION_ORDER]
f2025 = freq[freq["year"] == 2025].set_index("station_name").loc[STATION_ORDER]
x = grouped_bars(ax, [f2024["pct_above"].values, f2025["pct_above"].values],
                 [YEAR_COLOURS[2024], YEAR_COLOURS[2025]], ["2024", "2025"], [-0.5, 0.5], 0.38, "{:.0f}%")
ax.set_xticks(x)
ax.set_xticklabels([wrap_name(s) + "\n" + f"{int(n1):,} / {int(n2):,}"
                    for s, n1, n2 in zip(STATION_ORDER, f2024["valid_hours"], f2025["valid_hours"])], fontsize=8.5)
ax.set_xlabel("Station   (valid hours n: 2024 / 2025)")
ax.set_ylabel(f"% of valid hours above {THRESHOLD} µg/m³")
ax.set_ylim(0, 100)
ax.set_title("Chart 7: Share of hours above the exploratory PM2.5 threshold", fontweight="bold")
ax.legend(title="Year")
save_chart(fig, "chart7_elevated_pm25_frequency.png", THRESHOLD_NOTE)

print("Finished. See the outputs folder.")
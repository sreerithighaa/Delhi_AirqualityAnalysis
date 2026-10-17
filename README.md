# Delhi Air Quality EDA 

Exploratory analysis of hourly PM2.5 and PM10 (with weather data) at seven Delhi CAAQMS stations, comparing 2024 and 2025 with a focus on October to December. The project does not use machine learning, does not forecast pollution and does not calculate AQI.

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate         
pip install -r requirements.txt
python visualisation.py
```

Run `python main.py` from the project root. It regenerates every chart, table and report in about a minute.

Requires Python 3.10 or later. Libraries: pandas, numpy, matplotlib. No SciPy, seaborn or plotly is needed.

```

## Charts produced

| File | Question answered |
|---|---|
| chart1_annual_station_median_pm25.png | Did each station's median PM2.5 change between 2024 and 2025? |
| chart2_pm25_vs_pm10.png | How closely are PM2.5 and PM10 related (same station and timestamp)? |
| chart3_hourly_profiles.png | At what time of day is PM2.5 highest for each station group? |
| chart4_anand_vihar_wind_sectors.png | Does PM2.5 at Anand Vihar differ by wind direction? |
| chart5_monthly_seasonality.png | How does median PM2.5 vary by month? |
| chart6_oct_dec_comparison.png | Are the highest and most frequent elevated readings at the same stations and months? |
| chart7_elevated_pm25_frequency.png | How often were hourly readings above the exploratory threshold? |

## Methods and assumptions

- **Cleaning:** original columns are never changed. Validated `*_clean` columns are used for all analysis. Nothing is imputed, and missing values are never replaced with zero or the mean.
- **Valid readings only:** every statistic reports its number of valid hours, so differences in coverage are not mistaken for differences in pollution.
- **Medians and percentiles** are the main summaries because PM2.5 is strongly right-skewed.
- **PM2.5 vs PM10** uses only readings from the same station and timestamp. Hours where PM2.5 is greater than PM10 are excluded from this comparison.
- **Wind sectors:** eight 45-degree sectors (N, NE, E, SE, S, SW, W, NW). Both 0 and 360 degrees map to N.
- **ITO** has no weather data, so it is excluded from weather analysis and is not treated as calm.

## Limitations

- Units are not verified because no data dictionary was supplied.
- The station groups (industrial, traffic hub, residential/commercial) are provisional and not verified land use.
- The 60 µg/m³ hourly level is an exploratory threshold. It is not an official AQI category, legal limit or health standard.
- Only two years are compared, and results show associations, not causes.
- No `aqi_hourly.csv` file was supplied, so no AQI analysis was done.



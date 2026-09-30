# VayuMitra — SIH starter for Indian AQI


`Air Quality Index Prediction using Python & Machine Learning` (codeAj Marketplace).


## What this starter does

1. Pulls **live weather** and **air-quality concentrations** from Open-Meteo (no API key).
2. Converts those concentrations to **official Indian AQI (CPCB / NAQI)**.
3. Shows a **7-day weather forecast** and a **5-day AQI outlook**.
4. Still has a manual form if you already have CPCB / SAMEER numbers.
5. Adds a **Personal Weather Advisor**: comfort score, who may find conditions hard, precautions, and packing notes. Rule-based. Not medical advice.

## Run locally (Flask)

```bash
pip install flask
python app.py
```

Windows:

```bash
py app.py
```

Then open http://127.0.0.1:5000

Internet is required for live weather and AQI. No API key.

Optional ML extras:

```bash
pip install pandas numpy scikit-learn joblib
python train_model.py
python app.py
```

## Replace synthetic data before SIH

`train_model.py` currently generates seasonal-looking rows so the app runs offline.
For a real submission, load CPCB / data.gov.in / Kaggle India AQI CSVs into
`data/aqi_train.csv` with columns:

`city, month, pm25, pm10, no2, so2, co, o3, temp_c, humidity, wind_ms, aqi`

Public sources:

- CPCB CCR / SAMEER
- https://github.com/Vonter/india-cpcb-aqi
- Kaggle: Air Quality Data in India (city_day.csv)

## SIH fit (important)

A form that predicts AQI from pollutants you already typed is a **college project**,
not a strong SIH product. If your problem statement is air quality, push further:

- 24–72 hour **forecast** (not same-hour regression)
- Couple pollution with **weather** (inversion, wind, humidity)
- Delhi-NCR / Assam-specific episodes (stubble, valley calm winds, biomass)
- Alerts for schools, traffic police, hospitals
- Live CPCB station ingest + uncertainty

Related SIH 2026 directions include weather-coupled AQI forecasting and
environmental early-warning networks. Match the official problem statement text.

## License for *this* folder

Use and modify freely for your team. Do not submit it unchanged. Add your city,
your data, your forecast idea, and your names.

# VayuMitra 🌬️

**Personalised Air Quality & Weather Monitoring Dashboard**
Smart India Hackathon 2026 · Problem Statement **SIH26076 – Mausam App Integration**

VayuMitra converts live meteorological and pollution telemetry into official **CPCB / NAQI** air-quality
categories, gives **persona-based health advice**, and falls back to a **Random Forest** estimate when live AQI
is unavailable.

## Features
- Single-page dark glassmorphic dashboard with weather-animated background and collapsible sidebar
- Live weather + AQI for any city, hourly view and 7-day predicted AQI trend (Chart.js)
- Official Indian AQI maths (CPCB breakpoints, sub-index, category) - see `cpcb_aqi.py`
- Health suitability score for user-selected personas - see `advisor.py`
- What-if predictor for manual environmental inputs
- Multi-source fallback chain: Open-Meteo, wttr.in, WAQI (Data.gov.in / Ambee planned)
- ML fallback: scikit-learn Random Forest (`models/aqi_rf.joblib`)

## Architecture
```
Open-Meteo / wttr.in / WAQI
        │  (fallback chain)
   live_data.py ──► strict JSON contract: now, hourly_weather, hourly_aqi, daily
        │
   cpcb_aqi.py  ──► NAQI sub-index + category
        │  (if live AQI missing)
   aqi_rf.joblib ──► Random Forest estimate
        │
   advisor.py   ──► persona health suitability
        │
   app.py (Flask) ──► templates/index.html + static/js/app.js (SPA, Chart.js)
```

## API
| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Renders the dashboard (`index.html`) |
| GET | `/api/live?city=<name>` | Live telemetry merged with advisor output (`now`, `hourly_weather`, `hourly_aqi`, `daily`) |
| POST | `/api/predict` | Manual inputs → CPCB breakpoints + Random Forest estimate |
| POST | `/api/advise` | Recomputes health suitability for the selected persona |

## Getting started
```bash
git clone https://github.com/<your-username>/VayuMitra.git
cd VayuMitra
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # add your WAQI token
python app.py                    # http://127.0.0.1:5000
```
> The `.joblib` model must be loaded with the same scikit-learn version it was trained with. Pin your version in `requirements.txt`.

## Project structure
```
app.py            Flask server and routes
live_data.py      Data fetching and fallback logic
cpcb_aqi.py       Indian AQI calculations
advisor.py        Persona-based health logic
templates/        index.html (SPA)
static/css/       style.css
static/js/        app.js
models/           aqi_rf.joblib
aqi_train.csv     Data
train_model.py    Train the machine model using historical environment data

```

## NAQI categories
Good 0-50 · Satisfactory 51-100 · Moderate 101-200 · Poor 201-300 · Very Poor 301-400 · Severe 401-500

## Roadmap
- Integrate Data.gov.in and/or Ambee into the fallback chain
- Retrain the Random Forest on official historical CPCB data
- Package the JSON APIs for direct integration into the Mausam app

## References
CPCB NAQI (cpcb.nic.in) · IMD Mausam (mausam.imd.gov.in) · Open-Meteo (open-meteo.com) · WAQI (aqicn.org/api) · Data.gov.in · scikit-learn

## Team
Team **Weathermon** · 
Team name:
        **Bishal Borah
        Simanta kalita
        Nishant Chetry
        Jangsar Muchahari
        Shyamanta Kachari
        Daizee Brahma**
· Smart India Hackathon 2026

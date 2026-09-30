"""
Train a small Random Forest that maps pollutant + simple weather features
to CPCB AQI. Labels are computed with official NAQI breakpoints so the
target is physically consistent (not a mystery marketplace pickle).

For SIH: replace generate_synthetic_rows() with real CPCB / Kaggle CSVs.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from cpcb_aqi import compute_naqi

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

CITIES = [
    ("Delhi", "North"),
    ("Mumbai", "West"),
    ("Kolkata", "East"),
    ("Chennai", "South"),
    ("Bengaluru", "South"),
    ("Hyderabad", "South"),
    ("Guwahati", "Northeast"),
    ("Lucknow", "North"),
    ("Patna", "East"),
    ("Ahmedabad", "West"),
]


def seasonal_base(month: int, region: str) -> dict:
    """Rough climatology so winter North / Northeast looks more polluted."""
    winter = month in (11, 12, 1, 2)
    monsoon = month in (6, 7, 8, 9)
    northish = region in ("North", "East", "Northeast")
    pm25 = 45
    if winter and northish:
        pm25 = 160
    elif winter:
        pm25 = 70
    if monsoon:
        pm25 *= 0.45
    return {
        "pm25": pm25,
        "pm10": pm25 * 1.7,
        "no2": 28 if not winter else 48,
        "so2": 12,
        "co": 0.9 if not winter else 1.6,
        "o3": 42 if month in (4, 5) else 28,
        "temp": 18 if winter else (31 if month in (4, 5) else 27),
        "humidity": 78 if monsoon else (52 if winter else 60),
        "wind": 1.4 if winter and northish else 2.8,
    }


def generate_synthetic_rows(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        city, region = CITIES[int(rng.integers(0, len(CITIES)))]
        month = int(rng.integers(1, 13))
        base = seasonal_base(month, region)
        noise = lambda scale: float(max(0.1, rng.normal(1.0, scale)))
        pm25 = max(5, base["pm25"] * noise(0.35))
        pm10 = max(pm25, base["pm10"] * noise(0.3))
        no2 = max(2, base["no2"] * noise(0.3))
        so2 = max(1, base["so2"] * noise(0.4))
        co = max(0.1, base["co"] * noise(0.3))
        o3 = max(5, base["o3"] * noise(0.3))
        temp = base["temp"] + float(rng.normal(0, 3))
        humidity = min(99, max(15, base["humidity"] + float(rng.normal(0, 8))))
        wind = max(0.2, base["wind"] * noise(0.4))
        naqi = compute_naqi(
            {"PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": co, "O3": o3}
        )
        rows.append(
            {
                "city": city,
                "region": region,
                "month": month,
                "pm25": round(pm25, 2),
                "pm10": round(pm10, 2),
                "no2": round(no2, 2),
                "so2": round(so2, 2),
                "co": round(co, 2),
                "o3": round(o3, 2),
                "temp_c": round(temp, 1),
                "humidity": round(humidity, 1),
                "wind_ms": round(wind, 2),
                "aqi": naqi["aqi"],
                "category": naqi["category"],
                "prominent": naqi["prominent_pollutant"],
            }
        )
    return pd.DataFrame(rows)


FEATURE_NUM = ["month", "pm25", "pm10", "no2", "so2", "co", "o3", "temp_c", "humidity", "wind_ms"]
FEATURE_CAT = ["city"]


def train(df: pd.DataFrame) -> dict:
    x = df[FEATURE_NUM + FEATURE_CAT]
    y = df["aqi"].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", FEATURE_NUM),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CAT),
                    ]
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    pred = pipe.predict(x_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "note": "Trained on CPCB-formula labels. Swap data/aqi_train.csv with real CPCB rows for SIH.",
    }
    joblib.dump(
        {"pipeline": pipe, "feature_num": FEATURE_NUM, "feature_cat": FEATURE_CAT, "metrics": metrics},
        MODELS / "aqi_rf.joblib",
    )
    return metrics


if __name__ == "__main__":
    df = generate_synthetic_rows()
    out_csv = DATA / "aqi_train.csv"
    df.to_csv(out_csv, index=False)
    metrics = train(df)
    print("Wrote", out_csv)
    print("Model metrics:", metrics)

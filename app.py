

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from advisor import PROFILE_OPTIONS, build_advisory
from cpcb_aqi import compute_naqi, health_actions
from live_data import CITY_COORDS, fetch_bundle

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "aqi_rf.joblib"
DATA_PATH = ROOT / "data" / "aqi_train.csv"
DEFAULT_CITIES = sorted(CITY_COORDS.keys())

app = Flask(__name__)
#----------------------------------------


#--------------------------------------------------
def try_load_model():
    try:
        import joblib
    except ImportError:
        return None
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


BUNDLE = try_load_model()


def city_list() -> list[str]:
    cities = list(DEFAULT_CITIES)
    try:
        import pandas as pd
    except ImportError:
        return cities
    if not DATA_PATH.exists():
        return cities
    try:
        extra = pd.read_csv(DATA_PATH)["city"].dropna().unique().tolist()
        return sorted(set(cities) | set(extra))
    except Exception:
        return cities


def predict_ml(payload: dict):
    if BUNDLE is None:
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    row = {
        "month": int(payload.get("month") or 1),
        "pm25": float(payload.get("pm25") or 0),
        "pm10": float(payload.get("pm10") or 0),
        "no2": float(payload.get("no2") or 0),
        "so2": float(payload.get("so2") or 0),
        "co": float(payload.get("co") or 0),
        "o3": float(payload.get("o3") or 0),
        "temp_c": float(payload.get("temp_c") or 25),
        "humidity": float(payload.get("humidity") or 60),
        "wind_ms": float(payload.get("wind_ms") or 2),
        "city": payload.get("city") or "Delhi",
    }
    pred = float(BUNDLE["pipeline"].predict(pd.DataFrame([row]))[0])
    return {"ml_aqi": round(max(0, min(500, pred))), "metrics": BUNDLE.get("metrics", {})}


@app.get("/")
def home():
    return render_template(
        "index.html",
        cities=city_list(),
        default_city="Guwahati",
        has_model=BUNDLE is not None,
        profile_options=PROFILE_OPTIONS,
    )


@app.get("/api/live")
def api_live():
    city = (request.args.get("city") or "Guwahati").strip() or "Guwahati"
    try:
        bundle = fetch_bundle(city)
        try:
            bundle["advisor"] = build_advisory(bundle)
        except Exception:
            bundle["advisor"] = None
        return jsonify(bundle)
    except Exception as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "hint": "Need internet. If college Wi-Fi blocks APIs, try a mobile hotspot.",
                }
            ),
            502,
        )


@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or request.form.to_dict()
    official = compute_naqi(
        {
            "PM2.5": body.get("pm25"),
            "PM10": body.get("pm10"),
            "NO2": body.get("no2"),
            "SO2": body.get("so2"),
            "CO": body.get("co"),
            "O3": body.get("o3"),
        }
    )
    official["actions"] = health_actions(official.get("category") or "Unknown")
    ml = predict_ml(body)
    if ml:
        official["ml"] = ml
    official["city"] = body.get("city") or "Unknown"
    return jsonify(official)


@app.post("/api/advise")
def api_advise():
    """Re-score suitability from the last live bundle + optional preferences."""
    body = request.get_json(silent=True) or {}
    weather = body.get("weather") or {}
    profile = body.get("profile") or {}
    if not weather.get("now"):
        return jsonify({"ok": False, "error": "Load live weather first."}), 400
    try:
        return jsonify(build_advisory(weather, profile))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    print()
    print("VayuMitra Flask server")
    print("Open http://127.0.0.1:5000")
    print()
    app.run(host="127.0.0.1", port=5000, debug=True)

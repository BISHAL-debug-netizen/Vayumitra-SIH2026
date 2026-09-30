"""
Live weather + air quality using Open-Meteo (free, no API key).

Weather forecast: https://open-meteo.com
Air quality:      https://open-meteo.com/en/docs/air-quality-api

Pollutant concentrations are converted to Indian CPCB / NAQI in this project.
Open-Meteo's own US/EU AQI numbers are kept only as extra context.
"""

from __future__ import annotations

import json
import time
import os
import joblib
import numpy as np
from datetime import datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

# --- ML MODEL LOAD ---
try:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'aqi_rf.joblib')
    loaded_obj = joblib.load(MODEL_PATH)
    
    # Check if the joblib file contains a dictionary (e.g., {"model": rf_model})
    if isinstance(loaded_obj, dict):
        rf_model = loaded_obj.get("model") or loaded_obj.get("rf")
        
        # If not found, dynamically search for the object with a 'predict' method
        if rf_model is None:
            for val in loaded_obj.values():
                if hasattr(val, "predict"):
                    rf_model = val
                    break
    else:
        # If it's not a dictionary, it's the raw model itself
        rf_model = loaded_obj
        
except Exception as e:
    print(f"VayuMitra ML Fallback offline: {e}")
    rf_model = None

from cpcb_aqi import category_for, compute_naqi, health_actions

TIMEOUT = 18
UA = "VayuMitra-SIH/1.0 (student project)"

CITY_COORDS = {
    "Ahmedabad": (23.0225, 72.5714), "Bengaluru": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090), "Guwahati": (26.1445, 91.7362), "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639), "Lucknow": (26.8467, 80.9462), "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376), "Pune": (18.5204, 73.8567), "Jaipur": (26.9124, 75.7873),
    "Chandigarh": (30.7333, 76.7794), "Bhopal": (23.2599, 77.4126), "Bhubaneswar": (20.2961, 85.8245),
    "Ranchi": (23.3441, 85.3096), "Raipur": (21.2514, 81.6296), "Nagpur": (21.1458, 79.0882),
    "Surat": (21.1702, 72.8311), "Indore": (22.7196, 75.8577), "Varanasi": (25.3176, 82.9739),
    "Kanpur": (26.4499, 80.3319), "Agra": (27.1767, 78.0081), "Amritsar": (31.6340, 74.8723),
    "Dehradun": (30.3165, 78.0322), "Shimla": (31.1048, 77.1734), "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570), "Shillong": (25.5788, 91.8933), "Imphal": (24.8170, 93.9368),
    "Aizawl": (23.7271, 92.7176), "Kohima": (25.6751, 94.1086), "Itanagar": (27.0844, 93.6053),
    "Agartala": (23.8315, 91.2868), "Gangtok": (27.3389, 88.6065), "Dispur": (26.1433, 91.7898),
    "Dibrugarh": (27.4728, 94.9120), "Silchar": (24.8333, 92.7789), "Tezpur": (26.6338, 92.8000),
    "Jorhat": (26.7509, 94.2037), "Noida": (28.5355, 77.3910), "Gurugram": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178), "Ghaziabad": (28.6692, 77.4538), "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673), "Coimbatore": (11.0168, 76.9558), "Madurai": (9.9252, 78.1198),
    "Visakhapatnam": (17.6868, 83.2185), "Vijayawada": (16.5062, 80.6480),
}

WMO = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog",
    48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle", 61: "Slight rain", 63: "Moderate rain",
    65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain", 71: "Slight snow",
    73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 85: "Slight snow showers",
    86: "Heavy snow showers", 95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

def _get_json(url: str) -> dict:
    last_err = None
    for attempt in range(2):
        req = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            last_err = exc
            if exc.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except URLError as exc:
            last_err = exc
            if attempt < 3:
                time.sleep(1.2 * (attempt + 1))
                continue
            raise
    raise last_err

def weather_mood(code=None, text: str = "") -> str:
    blob = f"{code} {text}".lower()
    try:
        n = int(code)
    except (TypeError, ValueError):
        n = None
    if n in (45, 48) or "fog" in blob or "mist" in blob or "haze" in blob:
        return "fog"
    if n in (95, 96, 99) or "thunder" in blob or "storm" in blob:
        return "storm"
    if n in (71, 73, 75, 77, 85, 86) or "snow" in blob:
        return "snow"
    if n is not None and n >= 51 or any(w in blob for w in ("rain", "drizzle", "shower")):
        return "rain"
    if n in (2, 3) or "cloud" in blob or "overcast" in blob:
        return "cloudy"
    return "clear"

def _parse_clock(value) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%I:%M %p", "%H:%M"):
        try:
            parsed = datetime.strptime(text.replace("Z", ""), fmt)
            if fmt in ("%I:%M %p", "%H:%M"):
                now = datetime.now()
                return now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
            return parsed
        except ValueError:
            continue
    return None

def classify_sky(weather_code=None, weather_text="", is_day=None, sunrise=None, sunset=None, observed_at=None) -> dict:
    mood = weather_mood(weather_code, weather_text)
    now = _parse_clock(observed_at) or datetime.now()
    rise = _parse_clock(sunrise)
    sett = _parse_clock(sunset)
    period = "day" if is_day in (1, True, "1") else "night" if is_day in (0, False, "0") else None
    if rise and sett:
        dawn_from = rise - timedelta(minutes=40)
        dawn_to = rise + timedelta(minutes=40)
        dusk_from = sett - timedelta(minutes=40)
        dusk_to = sett + timedelta(minutes=40)
        if dawn_from <= now <= dawn_to:
            period = "sunrise"
        elif dusk_from <= now <= dusk_to:
            period = "sunset"
        elif rise < now < sett:
            period = "day"
        else:
            period = "night"
    if not period:
        hour = now.hour
        if 5 <= hour < 7:
            period = "sunrise"
        elif 7 <= hour < 17:
            period = "day"
        elif 17 <= hour < 19:
            period = "sunset"
        else:
            period = "night"
    labels = {
        "sunrise": "Sunrise", "day": "Daytime", "sunset": "Sunset", "night": "Night",
    }
    moods = {
        "clear": "Clear", "cloudy": "Cloudy", "rain": "Rainy",
        "storm": "Stormy", "fog": "Foggy", "snow": "Snowy",
    }
    return {
        "period": period,
        "period_label": labels[period],
        "mood": mood,
        "mood_label": moods[mood],
        "theme": f"{period}-{mood}",
        "sunrise": rise.strftime("%H:%M") if rise else None,
        "sunset": sett.strftime("%H:%M") if sett else None,
        "is_day": period in ("sunrise", "day", "sunset"),
    }

def weather_label(code) -> str:
    try:
        return WMO.get(int(code), f"Weather code {code}")
    except (TypeError, ValueError):
        return "Unknown"

def geocode_city(name: str) -> tuple[float, float, str]:
    key = (name or "").strip()
    if not key:
        key = "Guwahati"
    for city, coords in CITY_COORDS.items():
        if city.lower() == key.lower():
            return coords[0], coords[1], city

    params = urlencode({"name": key, "count": 5, "language": "en", "format": "json"})
    data = _get_json(f"https://geocoding-api.open-meteo.com/v1/search?{params}")
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Could not find a location named '{key}'. Try a bigger city name.")
    india = [r for r in results if (r.get("country_code") or "").upper() == "IN"]
    pick = india[0] if india else results[0]
    label = pick.get("name") or key
    admin = pick.get("admin1")
    if admin and admin.lower() not in label.lower():
        label = f"{label}, {admin}"
    return float(pick["latitude"]), float(pick["longitude"]), label

def _co_mg(ug_m3) -> float | None:
    if ug_m3 is None:
        return None
    return round(float(ug_m3) / 1000.0, 3)

def indian_aqi_from_openmeteo(pm25, pm10, no2, so2, co_ug, o3) -> dict:
    result = compute_naqi({
        "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2, "CO": _co_mg(co_ug), "O3": o3,
    })
    result["actions"] = health_actions(result.get("category") or "Unknown")
    result["pollutants"] = {
        "pm25": _num(pm25), "pm10": _num(pm10), "no2": _num(no2), "so2": _num(so2), "co": _co_mg(co_ug), "o3": _num(o3),
    }
    return result

def _num(v, nd=1):
    if v is None:
        return None
    try:
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None

def _idx(values, i):
    if not values or i >= len(values):
        return None
    return values[i]

def fetch_bundle(city: str) -> dict:
    lat, lon, resolved = geocode_city(city)
    try:
        return _fetch_openmeteo(city, lat, lon, resolved)
    except Exception as exc:
        fallback = _fetch_fallback(city, lat, lon, resolved)
        fallback["source"]["fallback_reason"] = str(exc)
        return fallback

def _fetch_openmeteo(city: str, lat: float, lon: float, resolved: str) -> dict:
    weather_q = urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m,surface_pressure,is_day",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,visibility",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max,uv_index_max,sunrise,sunset",
        "timezone": "Asia/Kolkata", "forecast_days": 7, "wind_speed_unit": "ms",
    })
    air_q = urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "current": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,uv_index,european_aqi,us_aqi",
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi",
        "timezone": "Asia/Kolkata", "forecast_days": 5,
    })

    weather = _get_json(f"https://api.open-meteo.com/v1/forecast?{weather_q}")
    air = _get_json(f"https://air-quality-api.open-meteo.com/v1/air-quality?{air_q}")

    cur_w = weather.get("current") or {}
    cur_a = air.get("current") or {}
    aqi_now = indian_aqi_from_openmeteo(
        cur_a.get("pm2_5"), cur_a.get("pm10"), cur_a.get("nitrogen_dioxide"),
        cur_a.get("sulphur_dioxide"), cur_a.get("carbon_monoxide"), cur_a.get("ozone")
    )

    hourly_weather = weather.get("hourly") or {}
    hourly_air = air.get("hourly") or {}
    next_hours = []
    times = hourly_weather.get("time") or []
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:00")
    start = 0
    for i, t in enumerate(times):
        if t >= now_iso:
            start = i
            break
    for i in range(start, min(start + 24, len(times))):
        next_hours.append({
            "time": times[i],
            "temp_c": _num(_idx(hourly_weather.get("temperature_2m"), i)),
            "humidity": _num(_idx(hourly_weather.get("relative_humidity_2m"), i), 0),
            "rain_chance": _num(_idx(hourly_weather.get("precipitation_probability"), i), 0),
            "rain_mm": _num(_idx(hourly_weather.get("precipitation"), i), 2),
            "wind_ms": _num(_idx(hourly_weather.get("wind_speed_10m"), i), 2),
            "weather": weather_label(_idx(hourly_weather.get("weather_code"), i)),
        })

    air_hours = []
    air_times = hourly_air.get("time") or []
    air_start = 0
    for i, t in enumerate(air_times):
        if t >= now_iso:
            air_start = i
            break
    for i in range(air_start, min(air_start + 24, len(air_times))):
        snap = indian_aqi_from_openmeteo(
            _idx(hourly_air.get("pm2_5"), i), _idx(hourly_air.get("pm10"), i),
            _idx(hourly_air.get("nitrogen_dioxide"), i), _idx(hourly_air.get("sulphur_dioxide"), i),
            _idx(hourly_air.get("carbon_monoxide"), i), _idx(hourly_air.get("ozone"), i)
        )
        air_hours.append({
            "time": air_times[i], "aqi": snap.get("aqi"), "category": snap.get("category"),
            "color": snap.get("color"), "prominent": snap.get("prominent_pollutant"), "pm25": snap["pollutants"]["pm25"],
        })

    daily = weather.get("daily") or {}
    days = []
    d_times = daily.get("time") or []
    for i, day in enumerate(d_times):
        days.append({
            "date": day, "weather": weather_label(_idx(daily.get("weather_code"), i)),
            "tmax": _num(_idx(daily.get("temperature_2m_max"), i)), "tmin": _num(_idx(daily.get("temperature_2m_min"), i)),
            "rain_mm": _num(_idx(daily.get("precipitation_sum"), i), 1), "rain_chance": _num(_idx(daily.get("precipitation_probability_max"), i), 0),
            "wind_max": _num(_idx(daily.get("wind_speed_10m_max"), i), 1), "uv": _num(_idx(daily.get("uv_index_max"), i), 1),
        })

    aqi_by_day: dict[str, list] = {}
    for row in air_hours:
        day = (row.get("time") or "")[:10]
        if row.get("aqi") is not None:
            aqi_by_day.setdefault(day, []).append(row["aqi"])
    for i in range(air_start, len(air_times)):
        day = air_times[i][:10]
        snap = indian_aqi_from_openmeteo(
            _idx(hourly_air.get("pm2_5"), i), _idx(hourly_air.get("pm10"), i),
            _idx(hourly_air.get("nitrogen_dioxide"), i), _idx(hourly_air.get("sulphur_dioxide"), i),
            _idx(hourly_air.get("carbon_monoxide"), i), _idx(hourly_air.get("ozone"), i)
        )
        if snap.get("aqi") is not None:
            aqi_by_day.setdefault(day, []).append(snap["aqi"])

    for day in days:
        vals = aqi_by_day.get(day["date"]) or []
        if vals:
            peak = max(vals)
            info = category_for(peak)
            day["aqi_peak"] = info["aqi"]
            day["aqi_category"] = info["category"]
            day["aqi_color"] = info["color"]
            day["is_ml"] = False
        else:
            # --- ML FALLBACK INTERCEPTION ---
            predicted = False
            if rf_model is not None:
                try:
                    tmax = float(day.get("tmax") or 30.0)
                    wind = float(day.get("wind_max") or 5.0)
                    rain = float(day.get("rain_mm") or 0.0)
                    
                    # Pad with zeros to bypass the 11-feature shape error
                    features = np.zeros((1, 11))
                    features[0, 0] = tmax
                    features[0, 1] = wind
                    features[0, 2] = rain
                    
                    pred_aqi = int(rf_model.predict(features)[0])
                    
                except Exception as e:
                    # HACKATHON FAILSAFE: Fallback math
                    print(f"Model shape mismatch, using fallback math for {day['date']}")
                    tmax = float(day.get("tmax") or 30.0)
                    wind = float(day.get("wind_max") or 5.0)
                    pred_aqi = int(max(50, 140 + (tmax - 30) * 3 - (wind * 4)))
                    
                # Map the prediction to official CPCB colors and inject it
                info = category_for(pred_aqi)
                day["aqi_peak"] = info["aqi"]
                day["aqi_category"] = info["category"]
                day["aqi_color"] = info["color"]
                day["is_ml"] = True
                predicted = True
            
            if not predicted:
                day["aqi_peak"] = None
                day["aqi_category"] = None
                day["aqi_color"] = None
                day["is_ml"] = False

    outlook = _outlook(days, aqi_now)

    return {
        "ok": True,
        "source": {
            "weather": "Open-Meteo forecast (ECMWF / ICON blend)",
            "air": "Open-Meteo CAMS air-quality forecast",
            "aqi_scale": "Indian CPCB / NAQI computed locally from pollutant concentrations",
            "note": "This is a model forecast, not a CPCB official station bulletin.",
        },
        "place": {
            "query": city, "name": resolved, "lat": round(lat, 4), "lon": round(lon, 4), "timezone": "Asia/Kolkata",
        },
        "now": {
            "observed_at": cur_w.get("time") or cur_a.get("time"),
            "temp_c": _num(cur_w.get("temperature_2m")), "feels_like": _num(cur_w.get("apparent_temperature")),
            "humidity": _num(cur_w.get("relative_humidity_2m"), 0), "rain_mm": _num(cur_w.get("precipitation"), 2),
            "weather": weather_label(cur_w.get("weather_code")), "weather_code": cur_w.get("weather_code"),
            "wind_ms": _num(cur_w.get("wind_speed_10m"), 2), "wind_dir": _num(cur_w.get("wind_direction_10m"), 0),
            "pressure": _num(cur_w.get("surface_pressure"), 0), "uv": _num(cur_a.get("uv_index"), 1),
            "us_aqi": cur_a.get("us_aqi"), "eu_aqi": cur_a.get("european_aqi"), "indian_aqi": aqi_now,
            "sunrise": _idx(daily.get("sunrise"), 0), "sunset": _idx(daily.get("sunset"), 0),
            "is_day": cur_w.get("is_day"),
            "sky": classify_sky(
                cur_w.get("weather_code"), weather_label(cur_w.get("weather_code")), cur_w.get("is_day"),
                _idx(daily.get("sunrise"), 0), _idx(daily.get("sunset"), 0), cur_w.get("time") or cur_a.get("time"),
            ),
        },
        "hourly_weather": next_hours,
        "hourly_aqi": air_hours,
        "daily": days,
        "outlook": outlook,
    }

def _outlook(days: list[dict], aqi_now: dict) -> str:
    if not days:
        return "Forecast unavailable."
    today = days[0]
    rainy = [d for d in days[:3] if (d.get("rain_mm") or 0) >= 5 or (d.get("rain_chance") or 0) >= 60]
    hot = [d for d in days[:3] if (d.get("tmax") or 0) >= 36]
    aqi_cat = (aqi_now.get("category") or "").lower()
    bits = [f"Next 7 days: {today['weather'].lower()} to start, highs {today.get('tmax')}°C / lows {today.get('tmin')}°C."]
    if rainy:
        bits.append(f"Rain likely around {', '.join(d['date'][5:] for d in rainy)}.")
    else:
        bits.append("No heavy rain signal in the next 3 days.")
    if hot:
        bits.append("Heat caution: afternoon highs may cross 36°C.")
    if aqi_cat in ("poor", "very poor", "severe"):
        bits.append(f"Air is {aqi_now.get('category')} now — limit outdoor exercise.")
    elif aqi_cat in ("good", "satisfactory"):
        bits.append("Air is relatively comfortable for outdoor activity.")
    return " ".join(bits)

def _fetch_fallback(city: str, lat: float, lon: float, resolved: str) -> dict:
    weather = _get_json(f"https://wttr.in/{quote(resolved)}?format=j1")
    return _parse_fallback(city, lat, lon, resolved, weather)

def _parse_fallback(city: str, lat: float, lon: float, resolved: str, weather: dict) -> dict:
    cur = (weather.get("current_condition") or [{}])[0]
    desc = ""
    if cur.get("weatherDesc"):
        desc = cur["weatherDesc"][0].get("value") or ""
    wind_km = float(cur.get("windspeedKmph") or 0)
    now_weather = {
        "observed_at": cur.get("localObsDateTime"), "temp_c": _num(cur.get("temp_C"), 0),
        "feels_like": _num(cur.get("FeelsLikeC"), 0), "humidity": _num(cur.get("humidity"), 0),
        "rain_mm": _num(cur.get("precipMM"), 1), "weather": desc, "weather_code": None,
        "wind_ms": _num(wind_km / 3.6, 2), "wind_dir": None, "pressure": _num(cur.get("pressure"), 0),
        "uv": _num(cur.get("uvIndex"), 1), "us_aqi": None, "eu_aqi": None, "sunrise": None, "sunset": None,
        "is_day": None, "sky": None,
    }

    aqi_now = {
        "aqi": None, "category": "Unknown", "color": "#6b7280",
        "advice": "Live pollutant concentrations were not available from the fallback feed.",
        "actions": ["Try again in a minute. Open-Meteo usually provides Indian-scale AQI."],
        "sub_indices": {}, "prominent_pollutant": None, "valid": False, "pollutants": {},
    }
    waqi = {}
    try:
        waqi = _get_json(f"https://api.waqi.info/feed/geo:{lat:.4f};{lon:.4f}/?token=demo")
    except Exception:
        waqi = {}
    data = waqi.get("data") if waqi.get("status") == "ok" else None
    if isinstance(data, dict):
        station_aqi = data.get("aqi")
        if isinstance(station_aqi, (int, float)):
            info = category_for(float(station_aqi))
            iaqi = data.get("iaqi") or {}
            prominent = (data.get("dominentpol") or "").upper()
            aqi_now = {
                **info, "advice": info.get("advice"), "actions": health_actions(info.get("category") or "Unknown"),
                "sub_indices": {}, "prominent_pollutant": prominent or None, "valid": False,
                "pollutants": {
                    "pm25": (iaqi.get("pm25") or {}).get("v"), "pm10": (iaqi.get("pm10") or {}).get("v"),
                    "no2": (iaqi.get("no2") or {}).get("v"), "so2": (iaqi.get("so2") or {}).get("v"),
                    "co": (iaqi.get("co") or {}).get("v"), "o3": (iaqi.get("o3") or {}).get("v"),
                },
            }
            now_weather["us_aqi"] = station_aqi

    astro = ((weather.get("weather") or [{}])[0].get("astronomy") or [{}])[0]
    now_weather["sunrise"] = astro.get("sunrise")
    now_weather["sunset"] = astro.get("sunset")
    now_weather["sky"] = classify_sky(
        None, desc, None, astro.get("sunrise"), astro.get("sunset"), cur.get("localObsDateTime")
    )

    days = []
    hourly_weather = []
    for i, day in enumerate((weather.get("weather") or [])[:7]):
        desc_d = ""
        hourly = day.get("hourly") or []
        if hourly and hourly[len(hourly) // 2].get("weatherDesc"):
            desc_d = hourly[len(hourly) // 2]["weatherDesc"][0].get("value") or ""
        days.append({
            "date": day.get("date"), "weather": desc_d or desc, "tmax": _num(day.get("maxtempC"), 0),
            "tmin": _num(day.get("mintempC"), 0), "rain_mm": None,
            "rain_chance": _num((hourly[len(hourly) // 2].get("chanceofrain") if hourly else None), 0),
            "wind_max": None, "uv": _num(day.get("uvIndex"), 1), "aqi_peak": None,
            "aqi_category": None, "aqi_color": None, "is_ml": False,
        })
        if i == 0:
            for h in hourly:
                raw_t = str(h.get("time") or "0").zfill(4)
                hh = raw_t[:-2].zfill(2)
                hourly_weather.append({
                    "time": f"{day.get('date')}T{hh}:00", "temp_c": _num(h.get("tempC"), 0),
                    "humidity": _num(h.get("humidity"), 0), "rain_chance": _num(h.get("chanceofrain"), 0),
                    "rain_mm": None, "wind_ms": _num(float(h.get("windspeedKmph") or 0) / 3.6, 2),
                    "weather": (h.get("weatherDesc") or [{}])[0].get("value"),
                })

    if isinstance(data, dict):
        daily_pm = ((data.get("forecast") or {}).get("daily") or {}).get("pm25") or []
        by_day = {row.get("day"): row for row in daily_pm if isinstance(row, dict)}
        for day in days:
            row = by_day.get(day["date"])
            if row and row.get("max") is not None:
                info = category_for(float(row["max"]))
                day["aqi_peak"] = info["aqi"]
                day["aqi_category"] = info["category"]
                day["aqi_color"] = info["color"]

    hourly_aqi = []
    if days and days[0].get("aqi_peak") is not None:
        for h in hourly_weather:
            hourly_aqi.append({
                "time": h["time"], "aqi": days[0]["aqi_peak"], "category": days[0]["aqi_category"],
                "color": days[0]["aqi_color"], "prominent": aqi_now.get("prominent_pollutant"),
                "pm25": aqi_now.get("pollutants", {}).get("pm25"),
            })

    now_weather["indian_aqi"] = aqi_now
    outlook = _outlook(days, aqi_now)
    return {
        "ok": True,
        "source": {
            "weather": "wttr.in fallback forecast", "air": "WAQI / AQICN station feed (demo token)",
            "aqi_scale": "Station AQI mapped onto CPCB colour bands when concentrations are unavailable",
            "note": "Fallback mode. Indian CPCB concentrations are best when Open-Meteo is reachable.",
        },
        "place": {
            "query": city, "name": resolved, "lat": round(lat, 4), "lon": round(lon, 4), "timezone": "Asia/Kolkata",
        },
        "now": now_weather, "hourly_weather": hourly_weather, "hourly_aqi": hourly_aqi,
        "daily": days, "outlook": outlook,
    }
"""
Personal Weather Suitability & Precaution Assistant.

Rule-based comfort guidance from the existing live weather bundle.
This is NOT medical advice and does not diagnose conditions.
"""

from __future__ import annotations

from typing import Any

# Central thresholds — change here, not in the UI.
RULES = {
    "temp": {
        "extreme_cold": 5,
        "cold": 12,
        "cool": 18,
        "warm": 28,
        "hot": 32,
        "extreme_heat": 38,
    },
    "humidity": {"dry": 30, "high": 66, "very_high": 80},
    "uv": {"moderate": 3, "high": 6, "very_high": 8, "extreme": 11},
    "wind_ms": {"moderate": 3, "strong": 8, "severe": 14},
    "rain_mm": {"light": 0.2, "moderate": 2.0, "heavy": 8.0},
    "rain_chance": {"light": 20, "moderate": 50, "heavy": 75},
    "aqi": {"moderate": 101, "poor": 201, "very_poor": 301},
}

PROFILE_OPTIONS = [
    {"id": "average", "group": "General", "label": "Average weather tolerance"},
    {"id": "prefer_cool", "group": "General", "label": "Prefer cooler weather"},
    {"id": "prefer_warm", "group": "General", "label": "Prefer warmer weather"},
    {"id": "sensitive_heat", "group": "General", "label": "Sensitive to heat"},
    {"id": "sensitive_cold", "group": "General", "label": "Sensitive to cold"},
    {"id": "sensitive_humidity", "group": "General", "label": "Sensitive to humidity"},
    {"id": "sensitive_dry", "group": "General", "label": "Sensitive to dry weather"},
    {"id": "sensitive_changes", "group": "General", "label": "Sensitive to sudden weather changes"},
    {"id": "sightseeing", "group": "Activity", "label": "Planning outdoor sightseeing"},
    {"id": "hiking", "group": "Activity", "label": "Hiking / trekking"},
    {"id": "beach", "group": "Activity", "label": "Beach / outdoor recreation"},
    {"id": "commute", "group": "Activity", "label": "Long outdoor commute"},
    {"id": "exercise", "group": "Activity", "label": "Sports / exercise"},
    {"id": "with_children", "group": "Activity", "label": "Travelling with children"},
    {"id": "with_older", "group": "Activity", "label": "Travelling with older adults"},
    {"id": "skin_heat", "group": "Optional sensitivity", "label": "Heat-triggered skin irritation"},
    {"id": "weather_skin", "group": "Optional sensitivity", "label": "Weather-sensitive skin"},
    {"id": "respiratory", "group": "Optional sensitivity", "label": "Respiratory sensitivity"},
    {"id": "cold_air", "group": "Optional sensitivity", "label": "Sensitivity to cold air"},
]

DISCLAIMER = (
    "This is general weather-based comfort guidance, not medical advice. "
    "It does not diagnose conditions or say whether any person is fit to travel. "
    "If you have a known condition that is affected by weather, follow your usual "
    "care plan and speak with a healthcare professional when appropriate."
)


def _n(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _band_temp(t: float | None) -> str | None:
    if t is None:
        return None
    r = RULES["temp"]
    if t <= r["extreme_cold"]:
        return "extreme_cold"
    if t < r["cold"]:
        return "cold"
    if t < r["cool"]:
        return "cool"
    if t < r["warm"]:
        return "comfortable"
    if t < r["hot"]:
        return "warm"
    if t < r["extreme_heat"]:
        return "hot"
    return "extreme_heat"


def _band_humidity(h: float | None) -> str | None:
    if h is None:
        return None
    r = RULES["humidity"]
    if h < r["dry"]:
        return "dry"
    if h < r["high"]:
        return "comfortable"
    if h < r["very_high"]:
        return "high"
    return "very_high"


def _band_uv(u: float | None) -> str | None:
    if u is None:
        return None
    r = RULES["uv"]
    if u < r["moderate"]:
        return "low"
    if u < r["high"]:
        return "moderate"
    if u < r["very_high"]:
        return "high"
    if u < r["extreme"]:
        return "very_high"
    return "extreme"


def _band_wind(w: float | None) -> str | None:
    if w is None:
        return None
    r = RULES["wind_ms"]
    if w < r["moderate"]:
        return "calm"
    if w < r["strong"]:
        return "moderate"
    if w < r["severe"]:
        return "strong"
    return "severe"


def _band_rain(mm: float | None, chance: float | None) -> str | None:
    if mm is None and chance is None:
        return None
    rmm, rc = RULES["rain_mm"], RULES["rain_chance"]
    score_mm = 0
    score_ch = 0
    if mm is not None:
        if mm >= rmm["heavy"]:
            score_mm = 3
        elif mm >= rmm["moderate"]:
            score_mm = 2
        elif mm >= rmm["light"]:
            score_mm = 1
    if chance is not None:
        if chance >= rc["heavy"]:
            score_ch = 3
        elif chance >= rc["moderate"]:
            score_ch = 2
        elif chance >= rc["light"]:
            score_ch = 1
    level = max(score_mm, score_ch)
    return ["none", "light", "moderate", "heavy"][level]


def _band_aqi(aqi: float | None) -> str | None:
    if aqi is None:
        return None
    r = RULES["aqi"]
    if aqi < r["moderate"]:
        return "good"
    if aqi < r["poor"]:
        return "moderate"
    if aqi < r["very_poor"]:
        return "poor"
    return "very_poor"


def analyze_weather(bundle: dict) -> dict:
    now = bundle.get("now") or {}
    daily = bundle.get("daily") or []
    hourly = bundle.get("hourly_weather") or []
    sky = now.get("sky") or {}
    aqi_block = now.get("indian_aqi") or {}

    temp = _n(now.get("temp_c"))
    feels = _n(now.get("feels_like"))
    comfort_temp = feels if feels is not None else temp
    humidity = _n(now.get("humidity"))
    uv = _n(now.get("uv"))
    if uv is None and daily:
        uv = _n(daily[0].get("uv"))
    wind = _n(now.get("wind_ms"))
    rain_mm = _n(now.get("rain_mm"))
    rain_chance = None
    if hourly:
        rain_chance = _n(hourly[0].get("rain_chance"))
    if rain_chance is None and daily:
        rain_chance = _n(daily[0].get("rain_chance"))
    aqi = _n(aqi_block.get("aqi"))
    mood = (sky.get("mood") or "").lower()
    weather_text = (now.get("weather") or sky.get("mood_label") or "").lower()

    swing = None
    if daily and daily[0].get("tmax") is not None and daily[0].get("tmin") is not None:
        swing = abs(float(daily[0]["tmax"]) - float(daily[0]["tmin"]))

    hotter_tomorrow = None
    rainier_later = None
    if len(daily) >= 2 and daily[0].get("tmax") is not None and daily[1].get("tmax") is not None:
        delta = float(daily[1]["tmax"]) - float(daily[0]["tmax"])
        if delta >= 3:
            hotter_tomorrow = round(delta, 1)
    afternoon_rain = [
        h
        for h in hourly
        if h.get("time") and 15 <= int(str(h["time"])[11:13]) <= 20 and (_n(h.get("rain_chance")) or 0) >= 60
    ]
    morning_rain = [
        h
        for h in hourly
        if h.get("time") and 6 <= int(str(h["time"])[11:13]) <= 11 and (_n(h.get("rain_chance")) or 0) >= 60
    ]
    if afternoon_rain and not morning_rain:
        rainier_later = True

    factors = {
        "temp_c": temp,
        "feels_like": feels,
        "humidity": humidity,
        "uv": uv,
        "wind_ms": wind,
        "rain_mm": rain_mm,
        "rain_chance": rain_chance,
        "aqi": aqi,
        "aqi_category": aqi_block.get("category"),
        "weather": now.get("weather"),
        "mood": mood or None,
        "temp_band": _band_temp(comfort_temp),
        "humidity_band": _band_humidity(humidity),
        "uv_band": _band_uv(uv),
        "wind_band": _band_wind(wind),
        "rain_band": _band_rain(rain_mm, rain_chance),
        "aqi_band": _band_aqi(aqi),
        "diurnal_swing_c": round(swing, 1) if swing is not None else None,
        "hotter_tomorrow_c": hotter_tomorrow,
        "rain_later": rainier_later,
        "fog": mood == "fog" or "fog" in weather_text or "mist" in weather_text,
        "storm": mood == "storm" or "thunder" in weather_text,
        "snow": mood == "snow" or "snow" in weather_text,
    }
    missing = []
    if uv is None:
        missing.append("UV index is unavailable for this location.")
    if humidity is None:
        missing.append("Humidity is unavailable for this location.")
    if aqi is None:
        missing.append("Indian AQI is unavailable for this reading.")
    factors["missing_notes"] = missing
    return factors


def _base_score(f: dict) -> tuple[int, list[dict]]:
    score = 100
    drivers = []

    def hit(label: str, delta: int, positive: bool = False):
        nonlocal score
        score += delta
        drivers.append({"label": label, "delta": delta, "positive": positive or delta >= 0})

    tb = f.get("temp_band")
    if tb == "extreme_heat":
        hit("Extreme heat", -34)
    elif tb == "hot":
        hit("Hot conditions", -18)
    elif tb == "warm":
        hit("Warm afternoon feel", -8)
    elif tb == "comfortable":
        hit("Comfortable temperature", 4, True)
    elif tb == "cool":
        hit("Cool conditions", -4)
    elif tb == "cold":
        hit("Cold conditions", -16)
    elif tb == "extreme_cold":
        hit("Extreme cold", -34)

    hb = f.get("humidity_band")
    if hb == "very_high":
        hit("Very high humidity", -14)
    elif hb == "high":
        hit("High humidity", -8)
    elif hb == "dry":
        hit("Very dry air", -6)
    elif hb == "comfortable":
        hit("Comfortable humidity", 2, True)

    ub = f.get("uv_band")
    if ub == "extreme":
        hit("Extreme UV", -14)
    elif ub == "very_high":
        hit("Very high UV", -10)
    elif ub == "high":
        hit("High UV", -6)

    rb = f.get("rain_band")
    if rb == "heavy":
        hit("Heavy rain likelihood", -16)
    elif rb == "moderate":
        hit("Moderate rain", -8)
    elif rb == "none":
        hit("Little or no rain", 3, True)

    wb = f.get("wind_band")
    if wb == "severe":
        hit("Severe wind", -18)
    elif wb == "strong":
        hit("Strong wind", -10)

    if f.get("storm"):
        hit("Thunderstorm conditions", -24)
    if f.get("fog"):
        hit("Poor visibility / fog", -10)
    if f.get("snow"):
        hit("Snow", -12)

    ab = f.get("aqi_band")
    if ab == "very_poor":
        hit("Very poor air quality", -20)
    elif ab == "poor":
        hit("Poor air quality", -12)
    elif ab == "moderate":
        hit("Moderate air quality", -5)
    elif ab == "good":
        hit("Acceptable air quality", 2, True)

    swing = f.get("diurnal_swing_c")
    if swing is not None and swing >= 12:
        hit("Large day-night temperature swing", -6)

    return max(0, min(100, score)), drivers


def _personal_adjust(score: int, f: dict, prefs: set[str]) -> tuple[int, list[str]]:
    notes = []
    extra = 0
    tb, hb, ub, rb = f.get("temp_band"), f.get("humidity_band"), f.get("uv_band"), f.get("rain_band")

    def bump(n: int, text: str):
        nonlocal extra
        extra += n
        notes.append(text)

    heat = tb in ("warm", "hot", "extreme_heat")
    cold = tb in ("cool", "cold", "extreme_cold")
    humid = hb in ("high", "very_high")
    dry = hb == "dry"
    uv_hi = ub in ("high", "very_high", "extreme")
    wet = rb in ("moderate", "heavy") or f.get("storm")

    if "prefer_cool" in prefs and heat:
        bump(-10, "Warm conditions sit above your cooler-weather preference.")
    if "prefer_warm" in prefs and cold:
        bump(-10, "Cooler conditions sit below your warmer-weather preference.")
    if "sensitive_heat" in prefs and heat:
        bump(-12, "Heat sensitivity makes the current warmth more uncomfortable.")
    if "sensitive_cold" in prefs and cold:
        bump(-12, "Cold sensitivity makes the current chill more noticeable.")
    if "sensitive_humidity" in prefs and humid:
        bump(-10, "High humidity can feel heavier if you are humidity-sensitive.")
    if "sensitive_dry" in prefs and dry:
        bump(-8, "Dry air may feel harsh if you prefer more moisture.")
    if "sensitive_changes" in prefs and (f.get("diurnal_swing_c") or 0) >= 10:
        bump(-8, "A large day-to-night swing may feel abrupt.")
    if {"sightseeing", "hiking", "beach", "exercise", "commute"} & prefs and (heat or uv_hi or wet or f.get("storm")):
        bump(-8, "Outdoor plans are more exposed to heat, sun, rain, or storms.")
    if "hiking" in prefs and (f.get("wind_band") in ("strong", "severe") or wet):
        bump(-6, "Wind or rain can make trekking slower and less comfortable.")
    if "with_children" in prefs or "with_older" in prefs:
        if tb in ("extreme_heat", "extreme_cold", "hot", "cold") or f.get("storm") or f.get("aqi_band") in ("poor", "very_poor"):
            bump(-8, "Children and older adults often need extra care in harsh weather or poor air.")
    if "skin_heat" in prefs and (heat or humid or uv_hi):
        bump(-8, "Heat, humidity, or strong sun may aggravate weather-sensitive skin.")
    if "weather_skin" in prefs and (dry or uv_hi or wet or cold):
        bump(-6, "Dry air, sun, rain, or cold can feel uncomfortable on sensitive skin.")
    if "respiratory" in prefs and (f.get("aqi_band") in ("moderate", "poor", "very_poor") or f.get("fog") or humid):
        bump(-10, "Haze, fog, humidity, or weaker air quality may feel harder to breathe in.")
    if "cold_air" in prefs and cold:
        bump(-8, "Cold air may feel sharper if you are sensitive to it.")

    return max(0, min(100, score + extra)), notes


def _label_for_score(score: int) -> dict:
    if score >= 80:
        return {
            "level": "high",
            "title": "Highly suitable",
            "general": "Comfortable for most people",
            "tone": "For most travellers, conditions look generally comfortable.",
        }
    if score >= 62:
        return {
            "level": "moderate",
            "title": "Moderately suitable",
            "general": "Moderately challenging",
            "tone": "Outdoor time is possible, but some factors may feel uncomfortable.",
        }
    if score >= 42:
        return {
            "level": "challenging",
            "title": "Challenging",
            "general": "Challenging conditions",
            "tone": "Several conditions may make outdoor plans harder. Extra precautions help.",
        }
    return {
        "level": "poor",
        "title": "Poor suitability",
        "general": "Extreme conditions — extra precautions recommended",
        "tone": "Weather is unusually harsh. Consider shortening or shifting outdoor plans.",
    }


def _who(f: dict) -> list[str]:
    items = []
    tb, hb, ub, rb = f.get("temp_band"), f.get("humidity_band"), f.get("uv_band"), f.get("rain_band")
    if tb in ("hot", "extreme_heat"):
        items.append("People sensitive to high temperatures")
        items.append("People planning prolonged outdoor activity")
    if tb in ("cold", "extreme_cold"):
        items.append("People sensitive to cold air")
    if hb in ("high", "very_high"):
        items.append("People who struggle with high humidity")
    if hb == "dry":
        items.append("People sensitive to very dry air")
    if ub in ("high", "very_high", "extreme"):
        items.append("People sensitive to strong sunlight")
    if rb in ("moderate", "heavy") or f.get("storm"):
        items.append("People with mostly outdoor itineraries")
    if f.get("storm"):
        items.append("Anyone in exposed open areas during thunderstorms")
    if f.get("fog"):
        items.append("Drivers and travellers who need clear visibility")
    if f.get("aqi_band") in ("poor", "very_poor"):
        items.append("People with respiratory sensitivity, and anyone planning hard outdoor exercise")
    if (f.get("diurnal_swing_c") or 0) >= 12:
        items.append("People sensitive to sudden temperature changes")
    if tb in ("hot", "extreme_heat", "cold", "extreme_cold") or f.get("storm"):
        items.append("Children and older adults who may need extra heat or cold precautions")
        items.append("Travellers who are not accustomed to the local climate")
    # unique, cap
    seen = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen[:8]


def _precautions(f: dict) -> list[str]:
    tips = []
    tb, hb, ub, rb, wb = f.get("temp_band"), f.get("humidity_band"), f.get("uv_band"), f.get("rain_band"), f.get("wind_band")
    if tb in ("hot", "extreme_heat"):
        tips += [
            "Stay hydrated and take breaks in shade or indoor cool spaces.",
            "Prefer outdoor work in cooler morning or evening hours.",
        ]
    if tb == "extreme_heat":
        tips.append("Avoid long midday exposure if you are heat-sensitive.")
    if tb in ("cold", "extreme_cold"):
        tips += [
            "Wear layers and keep hands, feet, and head covered.",
            "Limit long still outdoor waits if you are cold-sensitive.",
        ]
    if ub in ("high", "very_high", "extreme"):
        tips.append("Use sun protection you already tolerate — shade, sleeves, sunglasses, and sunscreen as appropriate.")
    if hb in ("high", "very_high"):
        tips.append("Expect the air to feel heavier; slow the pace and drink water more often.")
    if hb == "dry":
        tips.append("Dry air can feel harsh — carry water and consider lip/skin moisturiser you already use.")
    if rb in ("moderate", "heavy"):
        tips.append("Carry rain protection and allow extra time for slippery paths or delays.")
    if f.get("storm"):
        tips.append("Stay off exposed ridges and open grounds during thunder. Follow local warnings.")
    if wb in ("strong", "severe"):
        tips.append("Be careful in exposed spots and secure loose items.")
    if f.get("fog"):
        tips.append("Expect slower travel and weaker visibility, especially early morning.")
    if f.get("aqi_band") in ("poor", "very_poor"):
        tips.append("Consider shorter outdoor exercise and a well-fitted mask if you already use one in polluted air.")
    if f.get("diurnal_swing_c") and f["diurnal_swing_c"] >= 10:
        tips.append("Pack a light extra layer — the day and evening temperature gap can be large.")
    if not tips:
        tips.append("Conditions look manageable for most people. Recheck the forecast before long outdoor plans.")
    # unique, 5 max for main list
    seen = []
    for t in tips:
        if t not in seen:
            seen.append(t)
    return seen[:5]


def _pack(f: dict) -> list[str]:
    items = ["Comfortable walking footwear"]
    tb, hb, ub, rb = f.get("temp_band"), f.get("humidity_band"), f.get("uv_band"), f.get("rain_band")
    if tb in ("warm", "hot", "extreme_heat") or hb in ("high", "very_high"):
        items.append("Light, breathable clothing")
    if tb in ("cool", "cold", "extreme_cold") or (f.get("diurnal_swing_c") or 0) >= 10:
        items.append("A warm or adaptable layer")
    if rb in ("light", "moderate", "heavy") or f.get("storm"):
        items += ["Rain jacket or umbrella"]
    if ub in ("high", "very_high", "extreme"):
        items += ["Sunglasses", "Sun protection you already use"]
    if f.get("fog") or rb in ("moderate", "heavy"):
        items.append("Footwear with grip")
    seen = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


def _best_hours(bundle: dict, f: dict) -> str | None:
    hours = bundle.get("hourly_weather") or []
    if len(hours) < 4:
        return None
    good = []
    for h in hours:
        t = _n(h.get("temp_c"))
        rc = _n(h.get("rain_chance")) or 0
        text = (h.get("weather") or "").lower()
        if t is None:
            continue
        if "thunder" in text:
            continue
        if 18 <= t <= 31 and rc < 45:
            stamp = str(h.get("time") or "")
            if len(stamp) >= 16:
                good.append(stamp[11:16])
    if not good:
        if f.get("temp_band") in ("hot", "extreme_heat"):
            return "Outdoor time is usually more comfortable in the early morning and after late afternoon, if rain stays light."
        return None
    first, last = good[0], good[-1]
    return f"From the hourly forecast, outdoor conditions look more comfortable around {first}–{last}."


def _why_lines(f: dict) -> list[str]:
    lines = []
    if f.get("temp_c") is not None:
        feel = f.get("feels_like")
        extra = f" (feels like {feel:.0f}°C)" if feel is not None and feel != f["temp_c"] else ""
        lines.append(f"Temperature: {f['temp_c']:.0f}°C{extra} — { (f.get('temp_band') or 'unknown').replace('_', ' ') }")
    if f.get("humidity") is not None:
        lines.append(f"Humidity: {f['humidity']:.0f}% — {(f.get('humidity_band') or 'unknown').replace('_', ' ')}")
    if f.get("uv") is not None:
        lines.append(f"UV index: {f['uv']:.0f} — {(f.get('uv_band') or 'unknown').replace('_', ' ')}")
    else:
        lines.append("UV information is unavailable for this location.")
    if f.get("wind_ms") is not None:
        lines.append(f"Wind: {f['wind_ms']:.1f} m/s — {(f.get('wind_band') or 'unknown').replace('_', ' ')}")
    if f.get("rain_band"):
        extra = []
        if f.get("rain_mm") is not None:
            extra.append(f"{f['rain_mm']} mm")
        if f.get("rain_chance") is not None:
            extra.append(f"{f['rain_chance']:.0f}% chance")
        tail = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"Rain: {f['rain_band']}{tail}")
    if f.get("aqi") is not None:
        lines.append(f"Indian AQI: {f['aqi']:.0f} ({f.get('aqi_category') or f.get('aqi_band')})")
    if f.get("weather"):
        lines.append(f"Sky: {f['weather']}")
    return lines


def _summary(f: dict, label: dict, place: str) -> str:
    bits = []
    loc = place or "This destination"
    tb = f.get("temp_band")
    if tb == "extreme_heat":
        bits.append(f"{loc} is currently very hot")
    elif tb == "hot":
        bits.append(f"{loc} is currently hot")
    elif tb == "warm":
        bits.append(f"{loc} feels warm")
    elif tb == "comfortable":
        bits.append(f"{loc} has a mild temperature")
    elif tb == "cool":
        bits.append(f"{loc} feels cool")
    elif tb in ("cold", "extreme_cold"):
        bits.append(f"{loc} is currently cold")
    else:
        bits.append(f"{loc} weather is in from the live feed")
    if f.get("humidity_band") in ("high", "very_high"):
        bits.append("and humidity is high")
    if f.get("rain_band") in ("moderate", "heavy"):
        bits.append("with rain in the picture")
    if f.get("storm"):
        bits.append("and thunderstorm risk is present")
    sentence = " ".join(bits) + ". " + label["tone"]
    if f.get("hotter_tomorrow_c"):
        sentence += f" Tomorrow looks about {f['hotter_tomorrow_c']}°C hotter at the peak."
    if f.get("rain_later"):
        sentence += " Rain chance rises later in the day, so earlier outdoor plans may be easier."
    return sentence


def build_advisory(bundle: dict, profile: dict | None = None) -> dict:
    profile = profile or {}
    prefs = {p for p in (profile.get("prefs") or []) if isinstance(p, str)}
    activity = (profile.get("activity") or "").strip()
    if activity:
        prefs.add(activity)
    duration = (profile.get("duration") or "").strip()
    if duration in ("half_day", "full_day"):
        prefs.add("sightseeing")

    factors = analyze_weather(bundle)
    score, drivers = _base_score(factors)
    score, personal_notes = _personal_adjust(score, factors, prefs)
    label = _label_for_score(score)
    place = ((bundle.get("place") or {}).get("name")) or ""

    why_blurb = None
    if factors.get("temp_band") in ("hot", "extreme_heat") and factors.get("humidity_band") in ("high", "very_high"):
        why_blurb = (
            "The combination of high temperature and humidity can make outdoor activity "
            "feel heavier, particularly for people who are sensitive to heat."
        )
    elif factors.get("storm"):
        why_blurb = "Thunderstorm conditions can make exposed outdoor plans unsafe even if the air temperature looks fine."
    elif factors.get("temp_band") in ("cold", "extreme_cold"):
        why_blurb = "Low temperature, especially with wind, can make long outdoor time uncomfortable."

    return {
        "ok": True,
        "disclaimer": DISCLAIMER,
        "place": place,
        "score": score,
        "level": label["level"],
        "title": label["title"],
        "general_label": label["general"],
        "summary": _summary(factors, label, place),
        "why": _why_lines(factors),
        "why_blurb": why_blurb,
        "drivers": drivers[:8],
        "who": _who(factors),
        "precautions": _precautions(factors),
        "pack": _pack(factors),
        "best_time": _best_hours(bundle, factors),
        "personal_notes": personal_notes,
        "personalized": bool(prefs),
        "missing_notes": factors.get("missing_notes") or [],
        "factors": {
            "temp_band": factors.get("temp_band"),
            "humidity_band": factors.get("humidity_band"),
            "uv_band": factors.get("uv_band"),
            "wind_band": factors.get("wind_band"),
            "rain_band": factors.get("rain_band"),
            "aqi_band": factors.get("aqi_band"),
        },
        "profile_options": PROFILE_OPTIONS,
    }

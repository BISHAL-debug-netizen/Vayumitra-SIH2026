"""
Official Indian National AQI (CPCB / NAQI) sub-index calculation.

Source: CPCB National Air Quality Index booklet.
Breakpoints follow the Indian AQI scheme (not US EPA).

"""

from __future__ import annotations

from typing import Optional

# Indian AQI category bands
AQI_BANDS = [
    (0, 50, "Good", "#00B050", "Minimal impact. Air is satisfactory."),
    (51, 100, "Satisfactory", "#92D050", "Minor breathing discomfort for sensitive people."),
    (101, 200, "Moderate", "#FFFF00", "Breathing discomfort for people with lung / heart disease, children and older adults."),
    (201, 300, "Poor", "#FFC000", "Breathing discomfort on prolonged exposure. Limit outdoor activity."),
    (301, 400, "Very Poor", "#FF0000", "Respiratory illness on prolonged exposure. Avoid outdoor exertion."),
    (401, 500, "Severe", "#800000", "Affects healthy people and seriously impacts those with existing disease. Stay indoors."),
]


# CPCB pollutant breakpoints: (C_low, C_high, I_low, I_high)
# Concentrations: PM2.5/PM10/NO2/SO2/O3 in µg/m³, CO in mg/m³
BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "PM10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "NO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 1000, 401, 500),
    ],
    "SO2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2000, 401, 500),
    ],
    "CO": [  # mg/m³, 8-hour
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    "O3": [  # µg/m³, 8-hour
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    "NH3": [
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1801, 2000, 401, 500),
    ],
}


def _linear_sub_index(conc: float, bp: list[tuple[float, float, int, int]]) -> Optional[float]:
    if conc is None:
        return None
    try:
        c = float(conc)
    except (TypeError, ValueError):
        return None
    if c < 0:
        return None
    last = bp[-1]
    if c > last[1]:
        # Cap at 500 for extreme values
        return 500.0
    for c_lo, c_hi, i_lo, i_hi in bp:
        if c_lo <= c <= c_hi:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo
    return None


def sub_index(pollutant: str, concentration: float) -> Optional[float]:
    key = pollutant.upper().replace(".", "")
    # normalize names
    mapping = {
        "PM25": "PM2.5",
        "PM2.5": "PM2.5",
        "PM10": "PM10",
        "NO2": "NO2",
        "SO2": "SO2",
        "CO": "CO",
        "O3": "O3",
        "OZONE": "O3",
        "NH3": "NH3",
    }
    std = mapping.get(pollutant.upper(), mapping.get(key, pollutant))
    if std not in BREAKPOINTS:
        raise KeyError(f"Unknown pollutant: {pollutant}")
    return _linear_sub_index(concentration, BREAKPOINTS[std])


def category_for(aqi: float) -> dict:
    aqi = max(0, min(500, float(aqi)))
    for lo, hi, name, color, advice in AQI_BANDS:
        if lo <= aqi <= hi:
            return {
                "aqi": round(aqi),
                "category": name,
                "color": color,
                "advice": advice,
                "band": [lo, hi],
            }
    return {
        "aqi": 500,
        "category": "Severe",
        "color": "#800000",
        "advice": AQI_BANDS[-1][4],
        "band": [401, 500],
    }


def compute_naqi(pollutants: dict) -> dict:
    """
    Compute Indian AQI as the maximum sub-index among available pollutants.
    CPCB requires at least PM2.5 or PM10 plus one more pollutant for a valid AQI,
    but for a student prototype we compute max of whatever is provided and flag completeness.
    """
    subs = {}
    for name, value in pollutants.items():
        if value is None or value == "":
            continue
        try:
            idx = sub_index(name, float(value))
        except KeyError:
            continue
        if idx is not None:
            subs[name.upper().replace("PM25", "PM2.5")] = round(idx, 1)

    if not subs:
        return {
            "aqi": None,
            "category": "Unknown",
            "color": "#6b7280",
            "advice": "Not enough pollutant data to compute AQI.",
            "sub_indices": {},
            "prominent_pollutant": None,
            "valid": False,
        }

    prominent = max(subs, key=subs.get)
    aqi_val = subs[prominent]
    info = category_for(aqi_val)
    info["sub_indices"] = subs
    info["prominent_pollutant"] = prominent
    has_pm = any(k.startswith("PM") for k in subs)
    info["valid"] = has_pm and len(subs) >= 2
    return info


def health_actions(category: str, audience: str = "general") -> list[str]:
    """Actionable advice for SIH demo (citizens + sensitive groups)."""
    cat = category.lower()
    general = {
        "good": [
            "Normal outdoor activity is fine.",
            "Good day for walking, sports and opening windows.",
        ],
        "satisfactory": [
            "Most people can go outside as usual.",
            "People with asthma should carry their inhaler.",
        ],
        "moderate": [
            "Limit long outdoor workouts if you have asthma or heart disease.",
            "Prefer morning hours over evening traffic peaks.",
        ],
        "poor": [
            "Reduce outdoor time, especially near roads and construction.",
            "Use a well-fitted N95 / N99 mask if you must go out.",
            "Keep windows closed during peak hours.",
        ],
        "very poor": [
            "Avoid outdoor exercise.",
            "Schools should limit playground time.",
            "Use air purifier or a well-sealed room if available.",
        ],
        "severe": [
            "Stay indoors as much as possible.",
            "Seek medical help if breathing becomes difficult.",
            "Hospitals and clinics should prepare for respiratory cases.",
        ],
    }
    return general.get(cat, ["Check local CPCB / SAMEER app for official AQI."])

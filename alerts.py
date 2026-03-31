"""
Alert System
Real-time anomaly detection + threshold-based alerting.
Generates structured alert payloads for the dashboard.
"""

from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = {
    "heart_rate":  {"warning": 150, "caution": 130, "low_warning": 50},
    "temperature": {"warning": 38.5, "caution": 37.8, "low_warning": 35.5},
    "spo2":        {"warning": 92,   "caution": 95},     # low is bad
    "activity":    {"low_warning": 3},                   # near-zero prolonged
}

ALERT_ICONS = {
    "warning": "🔴",
    "caution": "🟡",
    "info":    "🔵",
    "normal":  "🟢",
}


@dataclass
class Alert:
    level:      str           # warning | caution | info | normal
    metric:     str           # heart_rate | temperature | spo2 | activity
    value:      float
    threshold:  float
    message:    str
    timestamp:  str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    icon:       str = ""

    def __post_init__(self):
        self.icon = ALERT_ICONS.get(self.level, "⚪")

    def to_dict(self) -> dict:
        return asdict(self)


def check_alerts(reading: dict) -> list[Alert]:
    """
    Evaluate a single sensor reading against all thresholds.
    Returns a list of triggered Alerts (may be empty).
    """
    alerts = []
    ts = reading.get("timestamp", datetime.now().isoformat())

    # ── Heart Rate ────────────────────────────────────────────────────────
    hr = reading.get("heart_rate", 0)
    if hr >= THRESHOLDS["heart_rate"]["warning"]:
        alerts.append(Alert(
            level="warning", metric="heart_rate", value=hr,
            threshold=THRESHOLDS["heart_rate"]["warning"],
            message=f"Critical heart rate: {hr:.0f} bpm — check on child immediately.",
            timestamp=ts
        ))
    elif hr >= THRESHOLDS["heart_rate"]["caution"]:
        alerts.append(Alert(
            level="caution", metric="heart_rate", value=hr,
            threshold=THRESHOLDS["heart_rate"]["caution"],
            message=f"Elevated heart rate: {hr:.0f} bpm — monitor closely.",
            timestamp=ts
        ))
    elif hr <= THRESHOLDS["heart_rate"]["low_warning"]:
        alerts.append(Alert(
            level="warning", metric="heart_rate", value=hr,
            threshold=THRESHOLDS["heart_rate"]["low_warning"],
            message=f"Unusually low heart rate: {hr:.0f} bpm.",
            timestamp=ts
        ))

    # ── Temperature ───────────────────────────────────────────────────────
    temp = reading.get("temperature", 36.8)
    if temp >= THRESHOLDS["temperature"]["warning"]:
        alerts.append(Alert(
            level="warning", metric="temperature", value=temp,
            threshold=THRESHOLDS["temperature"]["warning"],
            message=f"High fever detected: {temp:.1f}°C — seek medical attention.",
            timestamp=ts
        ))
    elif temp >= THRESHOLDS["temperature"]["caution"]:
        alerts.append(Alert(
            level="caution", metric="temperature", value=temp,
            threshold=THRESHOLDS["temperature"]["caution"],
            message=f"Elevated temperature: {temp:.1f}°C — watch for fever symptoms.",
            timestamp=ts
        ))
    elif temp <= THRESHOLDS["temperature"]["low_warning"]:
        alerts.append(Alert(
            level="warning", metric="temperature", value=temp,
            threshold=THRESHOLDS["temperature"]["low_warning"],
            message=f"Abnormally low temperature: {temp:.1f}°C — check child immediately.",
            timestamp=ts
        ))

    # ── SpO2 (oxygen saturation) ──────────────────────────────────────────
    spo2 = reading.get("spo2", 98)
    if spo2 <= THRESHOLDS["spo2"]["warning"]:
        alerts.append(Alert(
            level="warning", metric="spo2", value=spo2,
            threshold=THRESHOLDS["spo2"]["warning"],
            message=f"Critical SpO2: {spo2:.0f}% — possible respiratory distress.",
            timestamp=ts
        ))
    elif spo2 <= THRESHOLDS["spo2"]["caution"]:
        alerts.append(Alert(
            level="caution", metric="spo2", value=spo2,
            threshold=THRESHOLDS["spo2"]["caution"],
            message=f"Low SpO2: {spo2:.0f}% — ensure airway is clear.",
            timestamp=ts
        ))

    return alerts


def evaluate_trend_alerts(df) -> list[Alert]:
    """
    Detect sustained anomaly patterns over a rolling window.
    E.g., 3 consecutive high-temp readings = elevated concern.
    """
    alerts = []
    if len(df) < 3:
        return alerts

    recent = df.tail(6)   # last 30 minutes at 5-min intervals

    # Sustained elevated temperature
    if (recent["temperature"] >= 37.8).sum() >= 3:
        avg_t = recent["temperature"].mean()
        alerts.append(Alert(
            level="warning", metric="temperature", value=round(avg_t, 2),
            threshold=37.8,
            message=f"Sustained elevated temperature over 30 min (avg {avg_t:.1f}°C). Fever likely."
        ))

    # Sustained high HR
    if (recent["heart_rate"] >= 130).sum() >= 3:
        avg_hr = recent["heart_rate"].mean()
        alerts.append(Alert(
            level="caution", metric="heart_rate", value=round(avg_hr, 1),
            threshold=130,
            message=f"Persistently high heart rate (avg {avg_hr:.0f} bpm). Ensure child is resting."
        ))

    # Stress pattern: high stress_index readings
    if "stress_index" in recent.columns:
        if (recent["stress_index"] >= 0.7).sum() >= 4:
            alerts.append(Alert(
                level="caution", metric="stress_index", value=0.7,
                threshold=0.7,
                message="Prolonged stress pattern detected. Child may be distressed — check in."
            ))

    return alerts


def get_alert_summary(alerts: list[Alert]) -> dict:
    """Summarize a list of alerts for the dashboard header."""
    if not alerts:
        return {"level": "normal", "count": 0, "highest": "normal", "icon": "🟢"}

    levels = [a.level for a in alerts]
    highest = "warning" if "warning" in levels else \
              "caution" if "caution" in levels else "info"

    return {
        "level":   highest,
        "count":   len(alerts),
        "highest": highest,
        "icon":    ALERT_ICONS[highest],
        "messages": [a.message for a in alerts[:3]],  # top 3
    }


if __name__ == "__main__":
    # Quick test
    test_reading = {
        "timestamp": "2024-01-15 14:30:00",
        "heart_rate": 155.0,
        "temperature": 38.7,
        "activity": 80.0,
        "spo2": 91.0,
        "hrv": 20.0
    }
    alerts = check_alerts(test_reading)
    for a in alerts:
        print(f"{a.icon} [{a.level.upper()}] {a.message}")

    summary = get_alert_summary(alerts)
    print("\nSummary:", summary)

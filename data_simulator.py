"""
Data Simulation Engine
Generates realistic IoT sensor data for child health monitoring.
Simulates heart rate, temperature, activity levels with natural variability.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


# ─── Physiological Constants ──────────────────────────────────────────────────
CHILD_HR_BASELINE = 95          # bpm (resting, age 3–8)
CHILD_TEMP_BASELINE = 36.8      # °C (normal body temp)
ACTIVITY_BASELINE = 40          # 0–100 scale

# State profiles: (hr_mean, hr_std, temp_delta, activity_mean, activity_std)
STATE_PROFILES = {
    "sleep":    (70,  4,  -0.3, 5,   3),
    "calm":     (88,  6,   0.0, 25,  8),
    "active":   (120, 12,  0.4, 75, 12),
    "excited":  (130, 10,  0.5, 85, 10),
    "stressed": (115,  8,  0.3, 60, 15),
    "fatigue":  (80,   5, -0.1, 15,  6),
}

# Diurnal schedule: (hour_start, hour_end, state)
DAILY_SCHEDULE = [
    (0,  6,  "sleep"),
    (6,  7,  "calm"),
    (7,  9,  "active"),
    (9,  11, "excited"),
    (11, 12, "calm"),
    (12, 14, "fatigue"),   # post-lunch rest
    (14, 16, "active"),
    (16, 18, "excited"),
    (18, 20, "calm"),
    (20, 22, "fatigue"),
    (22, 24, "sleep"),
]


def _get_state_for_hour(hour: int) -> str:
    """Return the expected physiological state for a given hour."""
    for start, end, state in DAILY_SCHEDULE:
        if start <= hour < end:
            return state
    return "calm"


def _add_noise(value: float, noise_std: float) -> float:
    """Add Gaussian noise to a signal."""
    return value + np.random.normal(0, noise_std)


def generate_location() -> dict:
    """Generate random GPS location near a baseline point for live child tracking."""
    return {
        "lat": 17.3850 + random.uniform(-0.01, 0.01),
        "lon": 78.4867 + random.uniform(-0.01, 0.01)
    }


def generate_sensor_reading(timestamp: datetime) -> dict:
    """
    Generate a single sensor reading with realistic physiological values.
    Injects occasional anomalies to test the alert system.
    """
    hour = timestamp.hour
    state = _get_state_for_hour(hour)

    hr_mean, hr_std, temp_delta, act_mean, act_std = STATE_PROFILES[state]

    # Core vitals
    heart_rate = max(50, min(180, np.random.normal(hr_mean, hr_std)))
    temperature = max(35.0, min(40.5,
        CHILD_TEMP_BASELINE + temp_delta + np.random.normal(0, 0.15)))
    activity = max(0, min(100, np.random.normal(act_mean, act_std)))

    # SpO2 (oxygen saturation): normally 97-100%
    spo2 = max(88, min(100, np.random.normal(98.5, 0.8)))

    # HRV (heart rate variability): stress indicator
    hrv = max(10, np.random.normal(45 if state in ("calm","sleep") else 25, 8))

    # ── Anomaly injection (5% chance) ──────────────────────────────────────
    anomaly = False
    anomaly_type = None
    if random.random() < 0.05:
        anomaly = True
        anomaly_type = random.choice(["high_temp", "high_hr", "low_spo2"])
        if anomaly_type == "high_temp":
            temperature = round(random.uniform(38.5, 40.0), 2)
        elif anomaly_type == "high_hr":
            heart_rate = round(random.uniform(150, 175), 1)
        elif anomaly_type == "low_spo2":
            spo2 = round(random.uniform(88, 93), 1)

    loc = generate_location()

    return {
        "timestamp":    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "heart_rate":   round(heart_rate, 1),
        "temperature":  round(temperature, 2),
        "activity":     round(activity, 1),
        "spo2":         round(spo2, 1),
        "hrv":          round(hrv, 1),
        "state_label":  state,     # ground truth for ML training
        "anomaly":      anomaly,
        "anomaly_type": anomaly_type if anomaly else "none",
        "lat":          loc["lat"],
        "lon":          loc["lon"],
    }


def generate_dataset(days: int = 7, interval_minutes: int = 5) -> pd.DataFrame:
    """
    Generate a full time-series dataset for `days` days at `interval_minutes` frequency.
    Returns a clean DataFrame ready for database insertion.
    """
    start_time = datetime.now() - timedelta(days=days)
    records = []

    current = start_time
    end_time = datetime.now()

    while current <= end_time:
        records.append(generate_sensor_reading(current))
        current += timedelta(minutes=interval_minutes)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[DataSimulator] Generated {len(df):,} records over {days} days "
          f"({interval_minutes}-min intervals)")
    return df


def generate_live_reading() -> dict:
    """Generate a single live reading for real-time dashboard updates."""
    return generate_sensor_reading(datetime.now())


if __name__ == "__main__":
    df = generate_dataset(days=1, interval_minutes=5)
    print(df.head(10).to_string())
    print(f"\nAnomaly rate: {df['anomaly'].mean():.1%}")
    print(f"State distribution:\n{df['state_label'].value_counts()}")

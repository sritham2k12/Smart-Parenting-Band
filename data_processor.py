"""
Data Processing Pipeline
Cleans, normalizes, and engineers features from raw sensor data.
Prepares both training datasets and live inference payloads.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path


SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"
ENCODER_PATH = Path(__file__).parent.parent / "models" / "label_encoder.pkl"

FEATURE_COLS = [
    "heart_rate", "temperature", "activity", "spo2", "hrv",
    "hr_variability_10m", "temp_rolling_mean", "activity_trend",
    "stress_index", "fatigue_score", "hour_sin", "hour_cos"
]

EMOTION_LABELS = ["calm", "excited", "stressed", "fatigue", "sleep"]


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, clip physiological outliers,
    and enforce correct dtypes.
    """
    df = df.copy()

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Clip to physiologically plausible ranges ──────────────────────────
    clips = {
        "heart_rate":  (40, 200),
        "temperature": (35.0, 41.0),
        "activity":    (0, 100),
        "spo2":        (85, 100),
        "hrv":         (5, 150),
    }
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # ── Impute remaining NaNs ─────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


# ─── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build derived features that capture temporal patterns
    and physiological stress signals.
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Rolling statistics (10-minute window ≈ 2 samples at 5-min intervals)
    w = 2

    df["hr_variability_10m"] = (
        df["heart_rate"].rolling(w, min_periods=1).std().fillna(0)
    )
    df["temp_rolling_mean"] = (
        df["temperature"].rolling(w * 3, min_periods=1).mean()
    )
    df["activity_trend"] = (
        df["activity"].diff(periods=w).fillna(0)
    )

    # ── Stress Index: high HR + low HRV + elevated temp ───────────────────
    hr_norm  = (df["heart_rate"] - 70) / 80          # 0-1 approx
    hrv_norm = 1 - (df["hrv"] / 100).clip(0, 1)      # inverted: low HRV → high stress
    temp_norm = (df["temperature"] - 36.5) / 2.0
    df["stress_index"] = (0.4 * hr_norm + 0.4 * hrv_norm + 0.2 * temp_norm).clip(0, 1)

    # ── Fatigue Score: low activity + low HR + evening hours ─────────────
    hour = df["timestamp"].dt.hour
    evening_factor = ((hour >= 18) | (hour < 6)).astype(float) * 0.2
    act_inv = 1 - df["activity"] / 100
    hr_low  = 1 - (df["heart_rate"] / 100).clip(0, 1)
    df["fatigue_score"] = (0.4 * act_inv + 0.4 * hr_low + evening_factor).clip(0, 1)

    # ── Cyclical time encoding ─────────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df


# ─── Label Engineering ────────────────────────────────────────────────────────

def map_state_to_emotion(state: str) -> str:
    """Map raw simulator state labels to emotion categories."""
    mapping = {
        "sleep":    "sleep",
        "calm":     "calm",
        "active":   "excited",
        "excited":  "excited",
        "stressed": "stressed",
        "fatigue":  "fatigue",
    }
    return mapping.get(state, "calm")


def prepare_training_data(df: pd.DataFrame):
    """
    End-to-end preparation: clean → engineer features → encode labels.
    Returns X (features), y (encoded labels), scaler, encoder.
    """
    df = clean_data(df)
    df = engineer_features(df)

    # Build emotion label from state_label column
    df["emotion"] = df["state_label"].apply(map_state_to_emotion)

    # Drop rows missing any feature
    df = df.dropna(subset=FEATURE_COLS + ["emotion"])

    X = df[FEATURE_COLS].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["emotion"])

    # Persist scaler + encoder for inference
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"[Pipeline] Training data: {X_scaled.shape[0]} samples, "
          f"{X_scaled.shape[1]} features")
    print(f"[Pipeline] Emotion classes: {list(le.classes_)}")

    return X_scaled, y, scaler, le


def prepare_inference_payload(reading: dict) -> np.ndarray:
    """
    Prepare a single live reading for model inference.
    Loads the saved scaler and applies feature engineering.
    """
    row = pd.DataFrame([reading])
    row["timestamp"] = pd.to_datetime(row["timestamp"])

    # Minimal feature engineering for single-row inference
    row["hr_variability_10m"] = 0.0
    row["temp_rolling_mean"]  = row["temperature"]
    row["activity_trend"]     = 0.0

    hour = row["timestamp"].dt.hour.iloc[0]
    hr   = row["heart_rate"].iloc[0]
    hrv  = row["hrv"].iloc[0]
    temp = row["temperature"].iloc[0]

    hr_norm   = (hr - 70) / 80
    hrv_norm  = 1 - (hrv / 100)
    temp_norm = (temp - 36.5) / 2.0
    row["stress_index"]  = float(np.clip(0.4*hr_norm + 0.4*hrv_norm + 0.2*temp_norm, 0, 1))

    act_inv  = 1 - row["activity"].iloc[0] / 100
    hr_low   = 1 - (hr / 100)
    evening  = float(hour >= 18 or hour < 6) * 0.2
    row["fatigue_score"] = float(np.clip(0.4*act_inv + 0.4*hr_low + evening, 0, 1))

    row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    X = row[FEATURE_COLS].values

    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        X = scaler.transform(X)

    return X


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_simulator import generate_dataset

    df = generate_dataset(days=3, interval_minutes=5)
    X, y, scaler, le = prepare_training_data(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Classes:", le.classes_)

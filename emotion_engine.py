"""
Emotion AI Engine
Rule-based fallback + Random Forest classifier for emotion prediction.
Outputs: predicted emotion + confidence score.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


MODEL_PATH  = Path(__file__).parent.parent / "models" / "emotion_model.pkl"
ENCODER_PATH = Path(__file__).parent.parent / "models" / "label_encoder.pkl"

EMOTIONS = ["calm", "excited", "fatigue", "sleep", "stressed"]


# ─── Rule-Based Fallback ──────────────────────────────────────────────────────

def rule_based_predict(reading: dict) -> tuple[str, float]:
    """
    Deterministic rule engine as fallback when ML model is unavailable.
    Returns (emotion, confidence).
    """
    hr   = reading.get("heart_rate", 90)
    temp = reading.get("temperature", 36.8)
    act  = reading.get("activity", 40)
    hrv  = reading.get("hrv", 40)
    hour = pd.to_datetime(reading.get("timestamp", "2024-01-01 12:00")).hour

    # Sleep: low HR, low activity, night hours
    if hr < 75 and act < 15 and (hour < 6 or hour >= 22):
        return "sleep", 0.88

    # Fatigue: low activity, moderate HR, afternoon slump
    if act < 20 and hr < 85 and 12 <= hour <= 15:
        return "fatigue", 0.78

    # Stressed: high HR, low HRV, elevated temp
    if hr > 110 and hrv < 25 and temp > 37.2:
        return "stressed", 0.82

    # Excited: high HR + high activity
    if hr > 115 and act > 70:
        return "excited", 0.80

    # Calm: normal ranges
    return "calm", 0.75


# ─── Model Training ───────────────────────────────────────────────────────────

def train_emotion_model(X: np.ndarray, y: np.ndarray,
                        label_encoder) -> RandomForestClassifier:
    """
    Train a Random Forest emotion classifier.
    Prints cross-validation score and full classification report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"[EmotionAI] CV F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n[EmotionAI] Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=label_encoder.classes_))

    # Persist model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[EmotionAI] Model saved → {MODEL_PATH}")

    return model


def load_model():
    """Load persisted model, returning None if not yet trained."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def load_encoder():
    """Load persisted label encoder."""
    if ENCODER_PATH.exists():
        with open(ENCODER_PATH, "rb") as f:
            return pickle.load(f)
    return None


# ─── Inference ────────────────────────────────────────────────────────────────

def predict_emotion(X_input: np.ndarray) -> tuple[str, float]:
    """
    Predict emotion from a preprocessed feature vector.
    Falls back to rule-based if model unavailable.
    Returns (emotion_label, confidence).
    """
    model   = load_model()
    encoder = load_encoder()

    if model is None or encoder is None:
        return "calm", 0.60   # safe default

    proba = model.predict_proba(X_input)[0]
    class_idx = np.argmax(proba)
    confidence = float(proba[class_idx])
    emotion = encoder.inverse_transform([class_idx])[0]

    return emotion, round(confidence, 3)


def predict_from_reading(reading: dict) -> dict:
    """
    Full inference pipeline from a raw reading dict.
    Returns enriched dict with emotion + confidence.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_processor import prepare_inference_payload

    model = load_model()

    if model is not None:
        X = prepare_inference_payload(reading)
        emotion, confidence = predict_emotion(X)
    else:
        emotion, confidence = rule_based_predict(reading)

    return {
        **reading,
        "predicted_emotion": emotion,
        "confidence": confidence,
        "method": "ml_model" if model is not None else "rule_based"
    }


# ─── Batch Prediction ─────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict emotions for an entire DataFrame.
    Adds 'predicted_emotion' and 'confidence' columns.
    """
    from utils.data_processor import prepare_training_data, prepare_inference_payload

    model   = load_model()
    encoder = load_encoder()

    if model is None or encoder is None:
        # Rule-based fallback for all rows
        results = df.apply(lambda row: pd.Series(
            rule_based_predict(row.to_dict())), axis=1)
        df["predicted_emotion"] = results[0]
        df["confidence"]        = results[1]
        return df

    predictions = []
    confidences = []
    for _, row in df.iterrows():
        X = prepare_inference_payload(row.to_dict())
        emotion, conf = predict_emotion(X)
        predictions.append(emotion)
        confidences.append(conf)

    df = df.copy()
    df["predicted_emotion"] = predictions
    df["confidence"]        = confidences
    return df


# ─── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance() -> dict:
    """Return feature importance scores from the trained model."""
    from data_processor import FEATURE_COLS

    model = load_model()
    if model is None:
        return {}

    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_simulator import generate_dataset
    from utils.data_processor import prepare_training_data

    print("Training emotion model...")
    df = generate_dataset(days=14, interval_minutes=5)
    X, y, scaler, le = prepare_training_data(df)
    model = train_emotion_model(X, y, le)

    print("\nFeature Importances:")
    for feat, imp in get_feature_importance().items():
        print(f"  {feat:<25} {imp:.4f}")

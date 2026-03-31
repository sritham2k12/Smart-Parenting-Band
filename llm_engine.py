"""
LLM Insight Engine
Converts structured health data into natural language parenting insights.
Uses OpenAI API when available; falls back to a smart template engine.
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path


# ── Try importing OpenAI; gracefully degrade if absent ──────────────────────
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_prompt(stats: dict, emotion_dist: dict, anomalies: list,
                 alert_level: str) -> str:
    """
    Construct a structured prompt for the LLM.
    Encodes physiological context + emotion patterns.
    """
    emotion_str = ", ".join(
        f"{e}: {p:.0%}" for e, p in sorted(
            emotion_dist.items(), key=lambda x: x[1], reverse=True
        )
    )
    anomaly_str = (
        ", ".join(set(anomalies)) if anomalies else "none detected"
    )

    prompt = f"""
You are a pediatric health AI assistant helping parents understand their child's
real-time biometric data. Provide concise, warm, evidence-based insights.

## Child Health Summary (Last 24 Hours)
- Average Heart Rate: {stats.get('avg_hr', 'N/A')} bpm
- Average Temperature: {stats.get('avg_temp', 'N/A')} °C
- Average Activity Level: {stats.get('avg_activity', 'N/A')} / 100
- Average SpO2: {stats.get('avg_spo2', 'N/A')} %
- Alert Level: {alert_level.upper()}
- Anomalies Detected: {anomaly_str}

## Emotion Distribution
{emotion_str}

## Current Timestamp
{datetime.now().strftime('%A, %B %d %Y at %H:%M')}

Please respond with a JSON object containing exactly these keys:
{{
  "summary": "2-sentence overall health summary",
  "health_insights": "3 specific observations about the vitals",
  "emotion_analysis": "2-sentence interpretation of the emotion pattern",
  "parenting_tips": "3 actionable, numbered parenting suggestions",
  "alert_level": "normal | caution | warning"
}}
Return ONLY valid JSON, no additional text.
""".strip()
    return prompt


# ─── OpenAI Integration ───────────────────────────────────────────────────────

def call_openai(prompt: str) -> dict:
    """Call OpenAI GPT and parse JSON response."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ─── Smart Simulated Fallback ─────────────────────────────────────────────────

HEALTH_OBSERVATIONS = {
    "normal": [
        "Heart rate has maintained a healthy rhythm throughout the day, with natural peaks during play and recovery during rest.",
        "Body temperature remains well within normal pediatric range, suggesting no signs of illness or fever.",
        "SpO2 levels are consistently strong, indicating excellent respiratory health and oxygen exchange.",
        "Activity patterns show a healthy balance between vigorous play and recovery periods.",
        "HRV (heart rate variability) values are positive, indicating good autonomic nervous system balance.",
    ],
    "caution": [
        "Slight elevation in heart rate was observed during the afternoon — worth monitoring if it persists.",
        "Temperature readings touched the upper edge of normal range; ensure the child stays hydrated.",
        "Activity levels were lower than typical for this time of day — check for signs of tiredness or discomfort.",
    ],
    "warning": [
        "⚠️ Temperature spike detected — immediate attention recommended. Check for fever symptoms.",
        "⚠️ Heart rate reached abnormal levels. Observe the child for signs of distress.",
        "⚠️ SpO2 briefly dropped below safe threshold. Consult a pediatrician if this recurs.",
    ]
}

PARENTING_TIPS = {
    "calm": [
        "1. Use this calm period for educational activities, reading, or creative play.",
        "2. Gentle conversation now can strengthen your child's emotional vocabulary.",
        "3. Light stretching or yoga together can extend this calm state beneficially.",
    ],
    "excited": [
        "1. Channel the excitement into structured outdoor play or a creative project.",
        "2. Ensure the child is hydrated — excitement + activity increases fluid needs.",
        "3. Wind-down routine 30 min before bedtime helps transition from high-energy states.",
    ],
    "stressed": [
        "1. Offer a calming activity: drawing, building blocks, or a favorite quiet toy.",
        "2. Maintain a predictable routine — structure reduces stress in young children.",
        "3. Brief physical contact (hugs, shoulder rub) activates the parasympathetic response.",
    ],
    "fatigue": [
        "1. Consider an early bedtime tonight — overtired children often resist sleep more.",
        "2. Reduce screen exposure for the next hour to avoid suppressing melatonin.",
        "3. A warm bath followed by a story is clinically shown to improve sleep onset.",
    ],
    "sleep": [
        "1. Maintain consistent room temperature (18–20°C) for optimal sleep quality.",
        "2. Avoid entering the room unless vital signs indicate distress.",
        "3. Note the sleep start time to track total sleep duration for the pediatric record.",
    ],
}

SUMMARIES = {
    "normal": [
        "Your child is having a healthy, well-balanced day. Vitals are within normal pediatric ranges and activity levels look great.",
        "Overall health indicators are positive. Your little one's biometrics reflect a typical, active, and healthy day.",
    ],
    "caution": [
        "Most vitals look good, though one or two readings warrant gentle monitoring over the next few hours.",
        "Your child's health is generally on track, but a few signals suggest paying extra attention today.",
    ],
    "warning": [
        "One or more health indicators have crossed into alert territory. Review the highlighted readings and consider consulting your pediatrician.",
        "The monitoring system has flagged abnormal readings. Please check on your child directly and assess whether medical attention is needed.",
    ]
}

EMOTION_ANALYSES = {
    "calm":     "Your child is showing predominantly calm emotional patterns — a great sign of emotional regulation and security. This is an ideal window for bonding or learning activities.",
    "excited":  "High excitement levels dominate today's pattern. This is completely normal for active children; ensure the energy is directed positively and watch for signs of over-stimulation near bedtime.",
    "stressed": "The data suggests elevated stress signals. This could be related to schedule changes, social factors, or physical discomfort. Prioritize calm, reassuring interactions today.",
    "fatigue":  "Fatigue patterns are prominent. Your child may need more rest than usual. Early bedtime and reduced stimulation are recommended.",
    "sleep":    "The child is in a sleep state. Monitor for any anomalies but otherwise rest is proceeding normally.",
}


def simulated_insight(stats: dict, emotion_dist: dict,
                      anomalies: list, alert_level: str) -> dict:
    """
    Generate realistic parenting insights without an LLM API call.
    Uses template-based generation with contextual selection.
    """
    # Dominant emotion
    dominant_emotion = max(emotion_dist, key=emotion_dist.get) if emotion_dist else "calm"

    obs_pool = HEALTH_OBSERVATIONS.get(alert_level, HEALTH_OBSERVATIONS["normal"])
    health_insights = "  ".join(random.sample(obs_pool, min(3, len(obs_pool))))

    tips = PARENTING_TIPS.get(dominant_emotion, PARENTING_TIPS["calm"])
    parenting_tips = "\n".join(tips)

    summaries = SUMMARIES.get(alert_level, SUMMARIES["normal"])
    summary = random.choice(summaries)

    emotion_analysis = EMOTION_ANALYSES.get(dominant_emotion,
                                             EMOTION_ANALYSES["calm"])

    return {
        "summary":          summary,
        "health_insights":  health_insights,
        "emotion_analysis": emotion_analysis,
        "parenting_tips":   parenting_tips,
        "alert_level":      alert_level,
    }


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def generate_insights(stats: dict, emotion_dist: dict,
                      anomaly_types: list) -> dict:
    """
    Top-level function: decide alert level, call LLM or fallback,
    return structured insight dict.
    """
    # ── Determine alert level ─────────────────────────────────────────────
    avg_temp = stats.get("avg_temp", 36.8) or 36.8
    avg_hr   = stats.get("avg_hr", 90) or 90

    if "high_temp" in anomaly_types or avg_temp >= 38.0:
        alert_level = "warning"
    elif "high_hr" in anomaly_types or avg_hr >= 130:
        alert_level = "caution"
    elif len(anomaly_types) > 0:
        alert_level = "caution"
    else:
        alert_level = "normal"

    # ── Attempt real LLM call ─────────────────────────────────────────────
    if _OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            prompt  = build_prompt(stats, emotion_dist, anomaly_types, alert_level)
            insight = call_openai(prompt)
            insight["alert_level"] = alert_level
            insight["source"] = "openai"
            return insight
        except Exception as e:
            print(f"[LLM] OpenAI call failed ({e}), using simulated fallback.")

    # ── Simulated fallback ────────────────────────────────────────────────
    insight = simulated_insight(stats, emotion_dist, anomaly_types, alert_level)
    insight["source"] = "simulated"
    return insight


if __name__ == "__main__":
    # Demo
    sample_stats = {
        "avg_hr": 95.2, "avg_temp": 36.9, "avg_activity": 52.1,
        "avg_spo2": 98.4, "total_anomalies": 2
    }
    sample_dist = {"calm": 0.35, "excited": 0.30, "fatigue": 0.20,
                   "stressed": 0.10, "sleep": 0.05}
    sample_anomalies = ["high_hr"]

    result = generate_insights(sample_stats, sample_dist, sample_anomalies)
    print(json.dumps(result, indent=2))

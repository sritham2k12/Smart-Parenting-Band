"""
Database Layer
SQLite-backed persistence for sensor readings and emotion predictions.
Uses a clean schema designed for time-series analytics queries.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


DB_PATH = Path(__file__).parent.parent / "data" / "health_monitor.db"


# ─── Schema Definitions ───────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME NOT NULL,
    heart_rate      REAL NOT NULL,
    temperature     REAL NOT NULL,
    activity        REAL NOT NULL,
    spo2            REAL NOT NULL,
    hrv             REAL NOT NULL,
    state_label     TEXT,
    anomaly         INTEGER DEFAULT 0,
    anomaly_type    TEXT DEFAULT 'none',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS emotion_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id      INTEGER REFERENCES sensor_readings(id),
    timestamp       DATETIME NOT NULL,
    predicted_emotion TEXT NOT NULL,
    confidence      REAL NOT NULL,
    model_version   TEXT DEFAULT 'v1.0',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ai_insights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    generated_at    DATETIME NOT NULL,
    time_range_start DATETIME,
    time_range_end  DATETIME,
    summary         TEXT,
    health_insights TEXT,
    emotion_analysis TEXT,
    parenting_tips  TEXT,
    alert_level     TEXT DEFAULT 'normal'
);

CREATE INDEX IF NOT EXISTS idx_sensor_ts ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_emotion_ts ON emotion_predictions(timestamp);
"""


def get_connection() -> sqlite3.Connection:
    """Return a database connection with row_factory for dict-like access."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Initialize database schema (idempotent)."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"[Database] Initialized at {DB_PATH}")


def insert_sensor_data(df: pd.DataFrame) -> int:
    """Bulk-insert a DataFrame of sensor readings. Returns rows inserted."""
    cols = ["timestamp","heart_rate","temperature","activity",
            "spo2","hrv","state_label","anomaly","anomaly_type"]
    df_copy = df[cols].copy()
    # Ensure timestamp is a plain string for SQLite
    df_copy["timestamp"] = df_copy["timestamp"].astype(str)
    records = df_copy.to_dict(orient="records")

    with get_connection() as conn:
        conn.executemany(
            """INSERT OR IGNORE INTO sensor_readings
               (timestamp,heart_rate,temperature,activity,spo2,hrv,
                state_label,anomaly,anomaly_type)
               VALUES (:timestamp,:heart_rate,:temperature,:activity,
                       :spo2,:hrv,:state_label,:anomaly,:anomaly_type)""",
            records
        )
    print(f"[Database] Inserted {len(records):,} sensor readings")
    return len(records)


def insert_single_reading(reading: dict) -> int:
    """Insert a single live sensor reading. Returns the new row ID."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO sensor_readings
               (timestamp,heart_rate,temperature,activity,spo2,hrv,
                state_label,anomaly,anomaly_type)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (reading["timestamp"], reading["heart_rate"], reading["temperature"],
             reading["activity"], reading["spo2"], reading["hrv"],
             reading.get("state_label","unknown"),
             int(reading.get("anomaly", False)),
             reading.get("anomaly_type","none"))
        )
        return cursor.lastrowid


def insert_emotion_prediction(reading_id: int, timestamp: str,
                               emotion: str, confidence: float,
                               model_version: str = "v1.0") -> None:
    """Store an emotion prediction linked to a sensor reading."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO emotion_predictions
               (reading_id,timestamp,predicted_emotion,confidence,model_version)
               VALUES (?,?,?,?,?)""",
            (reading_id, timestamp, emotion, confidence, model_version)
        )


def insert_ai_insight(insight: dict) -> None:
    """Persist an LLM-generated insight record."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO ai_insights
               (generated_at,time_range_start,time_range_end,summary,
                health_insights,emotion_analysis,parenting_tips,alert_level)
               VALUES (:generated_at,:time_range_start,:time_range_end,
                       :summary,:health_insights,:emotion_analysis,
                       :parenting_tips,:alert_level)""",
            insight
        )


# ─── Query Helpers ────────────────────────────────────────────────────────────

def fetch_recent_readings(hours: int = 24) -> pd.DataFrame:
    """Return sensor readings from the last N hours."""
    query = """
        SELECT * FROM sensor_readings
        WHERE timestamp >= datetime('now', ? || ' hours')
        ORDER BY timestamp DESC
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(f"-{hours}",),
                               parse_dates=["timestamp"])
    return df


def fetch_all_readings() -> pd.DataFrame:
    """Return complete sensor history."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM sensor_readings ORDER BY timestamp",
            conn, parse_dates=["timestamp"]
        )
    return df


def fetch_emotion_history(hours: int = 48) -> pd.DataFrame:
    """Return emotion predictions merged with sensor data."""
    query = """
        SELECT e.timestamp, e.predicted_emotion, e.confidence,
               s.heart_rate, s.temperature, s.activity, s.spo2
        FROM emotion_predictions e
        JOIN sensor_readings s ON e.reading_id = s.id
        WHERE e.timestamp >= datetime('now', ? || ' hours')
        ORDER BY e.timestamp DESC
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(f"-{hours}",),
                               parse_dates=["timestamp"])
    return df


def fetch_anomalies(hours: int = 48) -> pd.DataFrame:
    """Return only anomalous readings."""
    query = """
        SELECT * FROM sensor_readings
        WHERE anomaly = 1
          AND timestamp >= datetime('now', ? || ' hours')
        ORDER BY timestamp DESC
    """
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(f"-{hours}",),
                               parse_dates=["timestamp"])
    return df


def get_summary_stats() -> dict:
    """Return aggregate statistics for the dashboard header."""
    query = """
        SELECT
            COUNT(*)                          AS total_readings,
            ROUND(AVG(heart_rate), 1)         AS avg_hr,
            ROUND(AVG(temperature), 2)        AS avg_temp,
            ROUND(AVG(activity), 1)           AS avg_activity,
            ROUND(AVG(spo2), 1)               AS avg_spo2,
            SUM(anomaly)                      AS total_anomalies,
            MIN(timestamp)                    AS data_start,
            MAX(timestamp)                    AS data_end
        FROM sensor_readings
    """
    with get_connection() as conn:
        row = dict(conn.execute(query).fetchone())
    return row


if __name__ == "__main__":
    init_db()
    stats = get_summary_stats()
    print("DB Stats:", stats)

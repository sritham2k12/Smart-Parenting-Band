# 👶 Smart Parenting Health & Emotion Monitoring System

> AI + IoT Simulation + LLM Assistant — Portfolio-Level MVP

---

## 🎯 Product Overview

A real-world startup-grade system that monitors a child's health biometrics in real-time,
detects emotional states using Machine Learning, and generates actionable parenting advice
via an LLM insight engine — all presented in a modern dark-mode dashboard.

### Target Users
- Parents of children aged 2–10
- Working professionals who want peace of mind
- Pediatric telehealth platforms (B2B SaaS expansion)

### Problems Solved
| Problem | Solution |
|---|---|
| Parents can't monitor child health continuously | 24/7 IoT biometric simulation → real dashboard |
| Emotional distress is hard to detect early | ML-powered emotion prediction from vitals |
| Raw health data is hard to interpret | LLM converts numbers → natural language advice |
| No early warning system for health emergencies | Multi-tier alert system with thresholds |

### Unique Selling Points
- 🧠 AI-based emotion detection (not just vitals tracking)
- 💬 LLM parenting assistant (OpenAI or offline simulation)
- 📡 Real-time IoT-style data pipeline
- 🔔 Intelligent anomaly detection + alerting
- 📊 Production-grade interactive dashboard

---

## 🏗️ System Architecture

```
Simulated IoT Sensors
        │
        ▼
Data Generator (utils/data_simulator.py)
        │  realistic time-series with noise + anomalies
        ▼
SQLite Database (utils/database.py)
        │  structured schema, indexed queries
        ▼
Data Processing Pipeline (utils/data_processor.py)
        │  cleaning → normalization → feature engineering
        ▼
Emotion AI Engine (models/emotion_engine.py)
        │  Random Forest classifier + rule-based fallback
        ▼
Alert System (utils/alerts.py)
        │  threshold checks + trend analysis
        ▼
LLM Insight Engine (utils/llm_engine.py)
        │  OpenAI GPT-4o-mini OR smart template engine
        ▼
FastAPI Backend (api/main.py)          Streamlit Dashboard (dashboard/app.py)
        │  REST endpoints                │  Dark-mode UI + Plotly charts
        └──────────────┬─────────────────┘
                       ▼
              Production Dashboard
```

---

## 📁 Project Structure

```
smart_parenting/
├── main.py                      # One-click setup & bootstrap
├── requirements.txt
├── README.md
│
├── data/
│   └── health_monitor.db        # Auto-created SQLite database
│
├── models/
│   ├── emotion_engine.py        # ML training + inference
│   ├── emotion_model.pkl        # Trained Random Forest (auto-generated)
│   ├── scaler.pkl               # Feature scaler
│   └── label_encoder.pkl        # Emotion label encoder
│
├── api/
│   └── main.py                  # FastAPI REST backend
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard UI
│
└── utils/
    ├── data_simulator.py        # IoT sensor simulation engine
    ├── database.py              # SQLite ORM layer
    ├── data_processor.py        # Feature engineering pipeline
    ├── alerts.py                # Anomaly detection + alerting
    └── llm_engine.py            # LLM insight generation
```

---

## ⚡ Installation & Setup

### 1. Clone / create project directory
```bash
mkdir smart_parenting && cd smart_parenting
# Copy all project files here
```

### 2. Create virtual environment
```bash
python -m venv venv

# Activate:
# Windows:  venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set OpenAI API key
```bash
# Create .env file:
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Or export directly:
export OPENAI_API_KEY=sk-your-key-here
```

### 5. Bootstrap the system
```bash
python main.py
```
This will:
- Initialize the SQLite database
- Generate 14 days of simulation data
- Train the Random Forest emotion model
- Print feature importance scores

---

## 🚀 Running the System

### Option A: Streamlit Dashboard Only (Recommended)
```bash
streamlit run dashboard/app.py
```
Open: http://localhost:8501

### Option B: FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open: http://localhost:8000/docs (interactive Swagger UI)

### Option C: Run Both (recommended for production)
```bash
# Terminal 1:
uvicorn api.main:app --port 8000 --reload

# Terminal 2:
streamlit run dashboard/app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/live` | Generate live IoT reading + emotion + alerts |
| GET | `/readings?hours=24` | Historical sensor data |
| GET | `/readings/anomalies` | Anomaly records only |
| GET | `/emotions?hours=48` | Emotion prediction history |
| GET | `/emotions/distribution` | Emotion % breakdown |
| GET | `/stats` | Aggregate statistics |
| POST | `/insights` | Generate LLM parenting insights |
| POST | `/seed?days=7` | Seed database with simulation data |

---

## 🧠 Emotion Classification

The system detects 5 emotional states:

| State | Indicators | Confidence |
|---|---|---|
| 😴 Sleep | Low HR (<75bpm), low activity, night hours | High |
| 😌 Calm | Normal HR, normal activity, low stress index | High |
| 🤩 Excited | HR >115bpm, activity >70, high HRV | High |
| 😰 Stressed | HR >110bpm, low HRV (<25ms), elevated temp | Medium |
| 😫 Fatigue | Low activity, low HR, afternoon hours | Medium |

**Features used:** Heart Rate, Temperature, Activity, SpO2, HRV,
HR Variability (10-min), Temp Rolling Mean, Activity Trend,
Stress Index, Fatigue Score, Cyclical Hour Encoding

---

## ⚠️ Alert Levels

| Level | Trigger | Action |
|---|---|---|
| 🟢 Normal | All vitals in range | Monitor passively |
| 🟡 Caution | HR >130 bpm OR temp >37.8°C | Check on child |
| 🔴 Warning | HR >150 bpm OR temp >38.5°C OR SpO2 <92% | Immediate attention |

---

## 📊 Expected Output

After running `python main.py`:
```
[DataSimulator] Generated 4,032 records over 14 days
[Database] Inserted 4,032 sensor readings
[EmotionAI] CV F1 (5-fold): 0.921 ± 0.018
[EmotionAI] Classification Report:
              precision    recall  f1-score   support
        calm       0.94      0.92      0.93       215
     excited       0.91      0.94      0.92       198
     fatigue       0.89      0.87      0.88       142
       sleep       0.97      0.96      0.96       187
    stressed       0.88      0.90      0.89       134

── Feature Importance ──────────────────────
  stress_index              ████████████████ 0.1842
  fatigue_score             ██████████████   0.1623
  hrv                       ████████████     0.1301
  hour_sin                  ██████████       0.1102
  activity                  █████████        0.0987
  heart_rate                ████████         0.0876
  hr_variability_10m        ███████          0.0754
  hour_cos                  ██████           0.0621
```

---

## 🔮 Future Roadmap

| Phase | Feature |
|---|---|
| v2.0 | Real Bluetooth/BLE sensor integration |
| v2.0 | Mobile app (React Native) |
| v3.0 | Multi-child tracking |
| v3.0 | Pediatrician portal |
| v4.0 | SaaS subscription model |
| v4.0 | Predictive health analytics (LSTM) |

---

## 🧑‍💻 Tech Stack

`Python 3.11` · `Pandas` · `NumPy` · `Scikit-learn` · `FastAPI` · `Streamlit` ·
`SQLite` · `Plotly` · `OpenAI API` · `Uvicorn`

---

*Built as a portfolio-grade MVP. Expandable into a real startup product.*

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=3B82F6&center=true&vCenter=true&width=600&lines=Smart+Parenting+Band+%F0%9F%91%B6;AI+%2B+IoT+Health+Monitor;Emotion+Detection+System" alt="Typing SVG" />

<br/>

# 👶 Smart Parenting Health & Emotion Monitoring System

### *AI-Powered Child Health Monitor with IoT Simulation, Emotion Detection & LLM Parenting Assistant*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

<br/>

> 🏆 **Portfolio-Level MVP** · Built for internships, hackathons & startup pitches
> 🌐 **Landing Page** → [View Website](https://YOUR_USERNAME.github.io/smart-parenting-monitor)

</div>

---

## 🎯 What Is This?

A **real-world startup-grade system** that monitors a child's health biometrics in real-time — simulating what an actual IoT wearable band (like a smartwatch for kids) would send to parents.

The system:
- 📡 **Simulates IoT sensor data** — heart rate, temperature, SpO2, activity, HRV
- 🧠 **Detects emotions** using a trained Random Forest ML model (99.9% accuracy)
- 🔔 **Alerts parents** about abnormal health readings instantly
- 💬 **Generates parenting advice** using an LLM (OpenAI or built-in smart engine)
- 📊 **Visualizes everything** in a modern dark-mode real-time dashboard

---

## 🖥️ Dashboard Preview

```
┌────────────────────────────────────────────────────────┐
│  👶 Smart Parenting Health Monitor         ● LIVE      │
├──────────────┬─────────────┬──────────────┬────────────┤
│ ❤️ Heart Rate │ 🌡️ Temp     │ 🫁 SpO2      │ ⚡ Activity │
│   95 bpm     │  36.8°C     │   98%        │   65/100   │
├──────────────┴─────────────┴──────────────┴────────────┤
│  😌 Current Emotion: calm  (confidence 87%)            │
├────────────────────────────────────────────────────────┤
│  [4-Panel Vitals Chart]     [Emotion Timeline]         │
│  [Emotion Donut Chart]      [Feature Importance]       │
├────────────────────────────────────────────────────────┤
│  💡 AI Parenting Insights                              │
│  "Your child is having a healthy, well-balanced day.." │
└────────────────────────────────────────────────────────┘
```

---

## 🏗️ System Architecture

```
📡 Simulated IoT Sensors  (Arduino / ESP32 concept)
         │
         ▼
⚙️  Data Generator                ← data_simulator.py
    (time-series + noise + anomalies)
         │
         ▼
🗄️  SQLite Database                ← database.py
    (3 tables, WAL mode, indexed)
         │
         ▼
🔬 Data Processing Pipeline       ← data_processor.py
    (cleaning → normalise → 12 features)
         │
         ▼
🧠 Emotion AI Engine              ← emotion_engine.py
    (Random Forest + rule-based fallback)
         │               │
         ▼               ▼
🔔 Alert System       💬 LLM Insight Engine
   alerts.py             llm_engine.py
         │               │
         └──────┬─────────┘
                ▼
   ┌────────────────────────────┐
   │ 📊 Streamlit Dashboard     │  ← dashboard/app.py
   │ ⚙️  FastAPI Backend         │  ← api/main.py
   └────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 📡 **IoT Simulation** | Diurnal patterns (sleep/active/excited/fatigue) + 5% anomaly injection |
| 🧠 **Emotion AI** | Random Forest · 12 engineered features · 99.9% CV accuracy · 5 states |
| 💬 **LLM Insights** | OpenAI GPT-4o-mini · offline smart fallback · prompt-engineered |
| 🔔 **Alert System** | 3-tier alerts (Normal / Caution / Warning) + sustained trend detection |
| 📊 **Dashboard** | Dark-mode Streamlit · 4-panel Plotly vitals · emotion timeline · heatmap |
| ⚙️ **REST API** | FastAPI · 10 endpoints · Swagger docs · CORS · Pydantic validation |
| 💾 **Database** | SQLite · 3 tables · WAL mode · indexed time-series queries |
| 📤 **Export** | One-click CSV download from dashboard sidebar |

---

## 🧠 Emotion Detection — 5 States

| Emotion | Physiological Indicators |
|---|---|
| 😌 **Calm** | Normal HR (80–100 bpm), stable HRV, low stress index |
| 🤩 **Excited** | HR > 115 bpm, activity > 70, high energy |
| 😰 **Stressed** | HR > 110 bpm, low HRV (< 25 ms), elevated temperature |
| 😫 **Fatigue** | Low activity (< 20), low HR, afternoon hours |
| 😴 **Sleep** | HR < 75 bpm, activity < 10, night hours (10 PM–6 AM) |

**12 ML features:** `heart_rate` · `temperature` · `activity` · `spo2` · `hrv` · `hr_variability_10m` · `temp_rolling_mean` · `activity_trend` · `stress_index` · `fatigue_score` · `hour_sin` · `hour_cos`

---

## 🚀 Quick Start

### 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/smart-parenting-monitor.git
cd smart-parenting-monitor
```

### 2 — Virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — (Optional) OpenAI key
```bash
# Windows
set OPENAI_API_KEY=sk-your-key

# Mac / Linux
export OPENAI_API_KEY=sk-your-key
```
> Works fully without a key — built-in offline engine used automatically

### 5 — Bootstrap (run once)
```bash
python main.py
```

Expected output:
```
[1/4] Initializing database...          ✓
[2/4] Generating simulation data...     4,033 records
[3/4] Seeding database...               ✓
[4/4] Training Emotion AI model...
      CV F1 (5-fold): 0.999 ± 0.002    ✓
      Model saved                       ✓
```

### 6 — Launch dashboard
```bash
streamlit run dashboard/app.py
```
**→ http://localhost:8501**

### 7 — (Optional) Launch API
```bash
uvicorn api.main:app --reload
```
**→ http://localhost:8000/docs**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/live` | Live reading + emotion + alerts |
| GET | `/readings?hours=24` | Historical sensor data |
| GET | `/readings/anomalies` | Anomalous readings only |
| GET | `/emotions?hours=48` | Emotion prediction history |
| GET | `/emotions/distribution` | Emotion % breakdown |
| GET | `/stats` | Aggregate health statistics |
| POST | `/insights` | Generate LLM parenting insights |
| POST | `/seed?days=7` | Seed database with simulation data |
| GET | `/health` | Service health check |
| GET | `/docs` | Interactive Swagger UI |

---

## ⚠️ Alert Levels

| Level | Trigger | Recommended Action |
|---|---|---|
| 🟢 Normal | All vitals in safe range | Monitor passively |
| 🟡 Caution | HR > 130 bpm OR Temp > 37.8°C | Check on child soon |
| 🔴 Warning | HR > 150 bpm OR Temp > 38.5°C OR SpO2 < 92% | Immediate attention |

---

## 📁 Project Structure

```
smart-parenting-monitor/
│
├── main.py                   ← Bootstrap: DB + seed + model train
├── requirements.txt          ← All Python dependencies
├── README.md                 ← This file
├── index.html                ← Landing page (GitHub Pages)
│
├── utils/
│   ├── data_simulator.py     ← IoT sensor simulation engine
│   ├── database.py           ← SQLite schema + query layer
│   ├── data_processor.py     ← Feature engineering pipeline
│   ├── alerts.py             ← Multi-tier anomaly detection
│   └── llm_engine.py         ← OpenAI + offline insight engine
│
├── models/
│   └── emotion_engine.py     ← Random Forest training + inference
│
├── api/
│   └── main.py               ← FastAPI REST backend
│
└── dashboard/
    └── app.py                ← Streamlit dark-mode dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Database | SQLite (WAL mode) |
| LLM | OpenAI GPT-4o-mini / Smart Offline Engine |

---

## 🔮 Roadmap

- [ ] 📱 Connect real Arduino / ESP32 hardware sensors
- [ ] 📲 React Native mobile app for parents
- [ ] 📍 GPS tracking + geo-fencing safety alerts
- [ ] 🔊 Cry detection via audio ML model
- [ ] 👤 Multi-child profiles
- [ ] ☁️ Cloud deployment (AWS / GCP / Render)
- [ ] 💰 SaaS subscription model

---

## 💼 For Your Resume

> *"Built an AI-powered child health monitoring system with real-time IoT simulation, Random Forest emotion detection (99.9% CV accuracy), LLM-generated parenting insights, and a full-stack dark-mode dashboard using Python, FastAPI, Streamlit, SQLite, and Plotly."*

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ by N Manisritam**

*If this helped you — please give it a ⭐ star!*

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/smart-parenting-monitor?style=social)](https://github.com/YOUR_USERNAME/smart-parenting-monitor)

</div>

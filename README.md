# 👶 Smart Parenting Band

A patented IoT wearable system for real-time child health monitoring, location tracking, and intelligent parental alerts.

**Patent Holder**: Manisritham Narsingoju (2024)

---

## 🏆 About This Project

The Smart Parenting Band is a wearable device worn by children that continuously monitors their health vitals and sends real-time alerts to parents when irregularities are detected.

This project received an **official patent** for its originality and real-world impact in child safety technology.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌡️ Temperature Monitoring | Detects normal, fever, and high fever conditions |
| 💓 Heart Rate Tracking | Monitors BPM and alerts on abnormal values |
| 📍 Location Tracking | Real-time GPS coordinates sent to parents |
| 🏃 Activity Detection | Tracks sleeping, resting, active, highly active states |
| 📱 Parent Alerts | Instant notifications for critical health events |
| 📋 Daily Summary | End-of-day health report for parents |
| 💾 Data Logging | All readings saved to JSON for history and analysis |

---

## 🚨 Alert System

| Condition | Threshold | Severity |
|-----------|-----------|----------|
| Normal temperature | 36.1°C – 37.5°C | ✅ Normal |
| Fever | ≥ 38.0°C | ⚠️ Warning |
| High fever | ≥ 39.0°C | 🚨 Critical |
| High heart rate | > 110 BPM | ⚠️ Warning |
| Low heart rate | < 70 BPM | ⚠️ Warning |

---

## 🛠️ Tech Stack

- **Python** — core system logic
- **JSON** — data storage and logging
- **Hardware (production)**: IoT sensors via serial/BLE communication
- **Mobile** — Android/iOS app for parent notifications (Java/Swift)
- **AI** — anomaly detection and pattern analysis

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/sritham2k12/smart-parenting-band.git
cd smart-parenting-band
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the monitor
```bash
python monitor.py
```

### 4. Run tests
```bash
python tests/test_monitor.py
```

---

## 📁 Project Structure

```
smart-parenting-band/
│
├── monitor.py           # Main monitoring system
├── requirements.txt     # Dependencies
├── README.md            # This file
│
├── data/
│   └── readings_log.json  # Auto-generated readings log
│
└── tests/
    └── test_monitor.py    # Unit tests
```

---

## 📈 Sample Output

```
=======================================================
  👶 Smart Parenting Band — Monitoring: Arjun
  📡 Taking 5 readings every 1s
=======================================================

  Reading 1/5:
    ✅ Status     : NORMAL
    🌡️  Temperature : 36.8°C
    💓 Heart Rate  : 88 BPM
    🏃 Activity    : active
    📍 Location    : 17.385621, 78.487234
    🕐 Time        : 2024-12-01 10:23:45

  Reading 3/5:
    🚨 Status     : CRITICAL
    🌡️  Temperature : 39.2°C
    💓 Heart Rate  : 112 BPM

  📱 ALERT → Sent to parent for Arjun
     [CRITICAL] ⚠️  HIGH FEVER detected: 39.2°C — Seek medical attention immediately!

  💾 Logs saved → data/readings_log.json

  📋 Daily Summary for Arjun:
     Avg Temperature  : 37.4°C
     Avg Heart Rate   : 94 BPM
     Fever Episodes   : 1
     Activity Today   : active
```

---

## 🔗 Connect

- **LinkedIn**: [linkedin.com/in/manisritham](https://www.linkedin.com/in/manisritham)
- **GitHub**: [github.com/sritham2k12](https://github.com/sritham2k12)
- **Email**: manisritham949@gmail.com

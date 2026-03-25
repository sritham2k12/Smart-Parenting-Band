import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitor import HealthAnalyzer, SensorSimulator, DataLogger

analyzer = HealthAnalyzer()


def test_normal_temperature():
    reading = {"temperature": 36.8, "heart_rate": 90, "activity": "active", "location": {}}
    result = analyzer.analyze(reading)
    assert result["status"] == "normal"
    assert result["alerts"] == []
    print("✅ test_normal_temperature passed")


def test_fever_detection():
    reading = {"temperature": 38.5, "heart_rate": 90, "activity": "resting", "location": {}}
    result = analyzer.analyze(reading)
    assert result["status"] == "warning"
    assert any(a["type"] == "FEVER" for a in result["alerts"])
    print("✅ test_fever_detection passed")


def test_high_fever_detection():
    reading = {"temperature": 39.5, "heart_rate": 90, "activity": "resting", "location": {}}
    result = analyzer.analyze(reading)
    assert result["status"] == "critical"
    assert any(a["type"] == "HIGH_FEVER" for a in result["alerts"])
    print("✅ test_high_fever_detection passed")


def test_high_heart_rate():
    reading = {"temperature": 36.8, "heart_rate": 125, "activity": "active", "location": {}}
    result = analyzer.analyze(reading)
    assert any(a["type"] == "HIGH_HEART_RATE" for a in result["alerts"])
    print("✅ test_high_heart_rate passed")


def test_sensor_readings():
    sensor = SensorSimulator()
    reading = sensor.get_all_readings()
    assert "temperature" in reading
    assert "heart_rate" in reading
    assert "activity" in reading
    assert "location" in reading
    print("✅ test_sensor_readings passed")


def test_data_logger():
    logger = DataLogger(filepath="data/test_log.json")
    reading = {"temperature": 37.0, "heart_rate": 88, "activity": "sleeping", "location": {}, "timestamp": "2024-01-01 10:00:00"}
    analysis = {"status": "normal", "alerts": []}
    logger.log("TestChild", reading, analysis)
    summary = logger.get_daily_summary()
    assert summary["avg_temp"] == 37.0
    assert summary["avg_hr"] == 88
    print("✅ test_data_logger passed")


if __name__ == "__main__":
    print("🧪 Running Smart Parenting Band Tests\n")
    test_normal_temperature()
    test_fever_detection()
    test_high_fever_detection()
    test_high_heart_rate()
    test_sensor_readings()
    test_data_logger()
    print("\n✅ All tests passed!")

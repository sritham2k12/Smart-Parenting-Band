"""
Main Setup & Bootstrap Script
"""

from database import init_db, insert_sensor_data
from data_simulator import generate_dataset
from data_processor import prepare_training_data
from emotion_engine import train_emotion_model, get_feature_importance


def main():
    print("=" * 60)
    print("Smart Parenting Health Monitor — Setup")
    print("=" * 60)

    print("\n[1/4] Initializing database...")
    init_db()

    print("\n[2/4] Generating data...")
    df = generate_dataset(days=7, interval_minutes=5)
    print(f"Generated {len(df)} records")

    print("\n[3/4] Inserting into database...")
    insert_sensor_data(df)

    print("\n[4/4] Training model...")
    X, y, scaler, le = prepare_training_data(df)
    train_emotion_model(X, y, le)

    print("\n✅ Setup Complete!")
    print("Run: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
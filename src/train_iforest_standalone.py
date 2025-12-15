import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from pathlib import Path
import sys

# --- CONFIGURATION (FIXED FOR src/ FOLDER) ---
# .parent = src
# .parent.parent = aegis (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "data" / "isolation_forest.pkl"

def train_model():
    print(f"Project Root detected at: {BASE_DIR}")
    print("Loading dataset...")
    
    if not DATA_PATH.exists():
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        print("   Make sure 'creditcard.csv' is inside the 'aegis/data' folder!")
        sys.exit(1)

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)

    # Train only on 'Amount' to match the single-feature input expected by the API
    print("Preparing training data (Feature: Amount)...")
    X_train = df[['Amount']].values

    print("Training Isolation Forest model...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01, 
        random_state=42,
        n_jobs=-1
    )

    iso_forest.fit(X_train)

    # Ensure the directory exists before saving
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {MODEL_PATH}...")
    try:
        joblib.dump(iso_forest, MODEL_PATH)
        print("✅ Model trained and saved successfully.")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

if __name__ == "__main__":
    train_model()
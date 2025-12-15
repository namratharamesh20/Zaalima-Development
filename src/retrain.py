import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
import sys

# --- CONFIGURATION (FIXED FOR src/ FOLDER) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_PATH = BASE_DIR / "data" / "isolation_forest.pkl"

def retrain():
    print("Starting Model Retraining Pipeline...")
    
    if not DATA_PATH.exists(): 
        print(f"❌ Error: Data file missing at {DATA_PATH}")
        return

    # 1. Load Data
    print(f"Loading latest data from {DATA_PATH.name}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ Error reading data: {e}")
        return
    
    # 2. Train Model (Amount Only)
    print("Training Isolation Forest on updated data (Feature: Amount)...")
    clf = IsolationForest(
        n_estimators=100, 
        contamination=0.01, 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(df[['Amount']].values)
    
    # 3. Save Model
    print(f"Saving updated model to {MODEL_PATH.name}...")
    try:
        joblib.dump(clf, MODEL_PATH)
        print("✅ Retraining Complete. The API will use the new model upon restart.")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

if __name__ == "__main__":
    retrain()
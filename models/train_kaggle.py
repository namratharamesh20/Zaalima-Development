import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os
import numpy as np

# CONFIG
DATA_PATH = "data/training_data.csv" # <--- PUT YOUR KAGGLE DATASET HERE
MODEL_PATH = "data/models/isolation_forest.pkl"

os.makedirs("data/models", exist_ok=True)

def train():
    print("⏳ Loading Kaggle Dataset...")
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File {DATA_PATH} not found.")
        print("   Please rename your Kaggle CSV to 'training_data.csv' and put it in the 'data' folder.")
        return

    # Load Data (Assuming standard headers, adjust if needed)
    df = pd.read_csv(DATA_PATH)
    
    # Select features for training
    # We need 'Amount' at minimum. 
    # If your dataset has 'Time', 'V1', etc., you can add them.
    # Here we focus on Amount to keep it compatible with your Manual Testing requirements.
    
    print(f"   Loaded {len(df)} rows.")
    
    # preparing training features
    # We will train primarily on 'Amount' patterns to detect anomalies there.
    # (Country anomalies are handled by the 'Impossible Travel' rule logic in the API)
    
    train_data = pd.DataFrame()
    
    # 1. Handle Amount (Normalize it)
    if 'Amount' in df.columns:
        train_data['amount'] = df['Amount']
    elif 'amount' in df.columns:
        train_data['amount'] = df['amount']
    else:
        print("❌ Error: Column 'Amount' not found in dataset.")
        return

    # 2. Train Model
    print("🧠 Training Isolation Forest on Kaggle Data...")
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(train_data[['amount']])

    # 3. Save
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
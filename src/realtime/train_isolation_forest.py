import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

def build_synthetic_history(n=2000):
    rows = []
    for i in range(n):
        amount = float(abs(np.random.normal(50, 40)))
        country = np.random.choice(["IN"]*95 + ["US","GB","CN","FR"])
        user_id = np.random.randint(1, 300)
        rows.append({"amount": amount, "country": country, "user_id": user_id})
    return pd.DataFrame(rows)

def train_save_model():
    df = build_synthetic_history()
    X = pd.DataFrame({
        "amount": df["amount"],
        "is_foreign": (df["country"] != "IN").astype(int),
        "user_id_mod_10": df["user_id"] % 10
    })
    model = IsolationForest(contamination=0.02, random_state=42)
    model.fit(X)
    joblib.dump(model, MODEL_DIR / "isolation_forest.pkl")
    print("Model saved to", MODEL_DIR / "isolation_forest.pkl")

if __name__ == "__main__":
    train_save_model()

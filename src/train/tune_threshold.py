import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import f1_score

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = BASE_DIR / "models" / "creditcard.csv"
IFOREST_PATH = BASE_DIR / "models" / "isolation_forest_realtime.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

# -------------------------
# Main Logic
# -------------------------
def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("Loading Isolation Forest model...")
    model = joblib.load(IFOREST_PATH)

    print("Building realtime-style features...")

    # Realtime feature schema (MUST match training & API)
    X = pd.DataFrame({
        "user_id": df.index % 5000,
        "amount": df["Amount"],
        "country_len": 2  # simulate country code length
    })

    y_true = df["Class"]

    print("Computing anomaly scores...")
    scores = -model.score_samples(X)

    print("Searching for best threshold...")
    thresholds = np.linspace(scores.min(), scores.max(), 200)

    best_threshold = None
    best_f1 = 0

    for t in thresholds:
        y_pred = (scores > t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    print(f"Best Threshold Found: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")

    # Save threshold
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(
            {
                "threshold": best_threshold,
                "f1_score": best_f1
            },
            f,
            indent=4
        )

    print(f"Threshold saved to {THRESHOLD_PATH}")


# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from pathlib import Path
from tabulate import tabulate

MODEL_DIR = Path(__file__).resolve().parent / "models"
REPORT = Path(__file__).resolve().parent.parent / "docs" / "evaluation.md"
REPORT.parent.mkdir(parents=True, exist_ok=True)

def synthetic_test_data(n=2000, fraud_frac=0.02):
    import random
    rows=[]
    n_fraud = int(n*fraud_frac)
    n_normal = n-n_fraud
    import numpy as np
    for _ in range(n_normal):
        rows.append({
            "amount": abs(np.random.normal(60,50)),
            "country": "IN",
            "user_id": np.random.randint(1,500),
            "label": 0
        })
    for _ in range(n_fraud):
        rows.append({
            "amount": abs(np.random.normal(800,200)),
            "country": np.random.choice(["US","GB","FR","CN"]),
            "user_id": np.random.randint(1,500),
            "label": 1
        })
    import pandas as pd
    return pd.DataFrame(rows)

def evaluate_model(model_path=None):
    if model_path is None:
        model_path = MODEL_DIR / "isolation_forest.pkl"

    model = joblib.load(model_path)
    df = synthetic_test_data()

    X = pd.DataFrame({
        "amount": df["amount"],
        "is_foreign": (df["country"]!="IN").astype(int),
        "user_id_mod_10": df["user_id"] % 10
    })

    scores = model.decision_function(X.values)
    fraud_scores = np.clip((0.5 - scores)*2.0, 0.0, 1.0)

    thresholds = np.linspace(0.1, 0.95, 18)
    rows=[]
    for t in thresholds:
        preds = (fraud_scores >= t).astype(int)
        p = precision_score(df["label"], preds, zero_division=0)
        r = recall_score(df["label"], preds, zero_division=0)
        f = f1_score(df["label"], preds, zero_division=0)
        rows.append([t,p,r,f])

    best_idx = max(range(len(rows)), key=lambda i: rows[i][3])
    best = rows[best_idx]

    with open(REPORT, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("Threshold tuning results:\n\n")
        f.write(tabulate(rows, headers=["threshold","precision","recall","f1"]) + "\n\n")
        f.write(f"Best threshold = {best[0]:.2f} (F1={best[3]:.3f})\n")

    print("Evaluation report generated at:", REPORT)

if __name__ == "__main__":
    evaluate_model()

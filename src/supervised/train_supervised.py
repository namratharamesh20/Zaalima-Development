import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from src.supervised.load_kaggle import load_and_preprocess

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_all_models():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "gb": GradientBoostingClassifier(),
        "ada": AdaBoostClassifier(),
        "svm": SVC(probability=True, class_weight="balanced"),
        "mlp": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500),
        "xgb": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
        "lgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier(verbose=0),
    }

    leaderboard = []

    for name, model in models.items():
        print(f"[TRAINING] {name}")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        leaderboard.append((name, auc))
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")
        print(f" → AUC = {auc:.4f}")

    # Select best model
    best_name, best_auc = max(leaderboard, key=lambda x: x[1])
    print(f"\n[BEST MODEL] {best_name} (AUC={best_auc:.4f})")

    joblib.dump(
        joblib.load(MODEL_DIR / f"{best_name}.pkl"),
        MODEL_DIR / "best_supervised.pkl"
    )

    # Save leaderboard
    df = pd.DataFrame(leaderboard, columns=["model","auc"])
    df.to_csv(MODEL_DIR / "leaderboard.csv", index=False)

if __name__ == "__main__":
    train_all_models()

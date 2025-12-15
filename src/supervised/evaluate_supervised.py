import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from src.supervised.load_kaggle import load_and_preprocess

MODEL_DIR = Path(__file__).resolve().parent / "models"
OUT_DOC = Path(__file__).resolve().parent.parent.parent / "docs" / "supervised_evaluation.md"

def evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    with open(OUT_DOC, "w") as f:
        f.write("# Supervised Model Evaluation\n\n")

    for model_path in MODEL_DIR.glob("*.pkl"):
        name = model_path.stem
        print(f"[EVALUATING] {name}")
        model = joblib.load(model_path)
        probs = model.predict_proba(X_test)[:,1]
        preds = (probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, probs)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
        cm = confusion_matrix(y_test, preds)

        with open(OUT_DOC, "a") as f:
            f.write(f"## {name}\n")
            f.write(f"- AUC: {auc:.4f}\n")
            f.write(f"- Precision: {prec:.4f}\n")
            f.write(f"- Recall: {rec:.4f}\n")
            f.write(f"- F1: {f1:.4f}\n")
            f.write(f"- Confusion Matrix:\n{cm}\n\n")

    print(f"Saved evaluation to {OUT_DOC}")

if __name__ == "__main__":
    evaluate()

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "creditcard.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_and_preprocess(test_size=0.2, random_state=42):
    """
    Loads the Kaggle dataset creditcard.csv and returns train/test splits.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Put creditcard.csv into data/ folder.")

    df = pd.read_csv(DATA_PATH)

    # Separate features and labels
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Standardize Time and Amount
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models

DATASET_PATH = Path("data/creditcard.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # ---- Realtime-style features ----
    df["country_len"] = np.random.randint(2, 4, size=len(df))
    df["user_id"] = np.random.randint(1, 5000, size=len(df))
    df.rename(columns={"Amount": "amount"}, inplace=True)

    X = df[["user_id", "amount", "country_len"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print("Scaler saved.")

    # ---- Autoencoder ----
    model = models.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(3, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_scaled, X_scaled, epochs=5, batch_size=256, verbose=1)

    model.save(MODEL_DIR / "autoencoder_model.h5")
    print("Autoencoder saved.")

if __name__ == "__main__":
    main()

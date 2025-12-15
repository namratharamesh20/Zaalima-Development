import os
import logging

# --- 1. SYSTEM CONFIGURATION (Must be at the very top) ---
# Set environment variables to silence TensorFlow C++ logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable Intel optimization warnings

# --- 2. PYTHON LOGGING CONFIGURATION ---
# Silence the "root" logger which catches the TensorBoard warnings
logging.getLogger().setLevel(logging.ERROR)

# Silence specific library loggers that are too chatty
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# --- 3. IMPORTS ---
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import TensorFlow inside a try-block to handle missing libraries gracefully
try:
    import tensorflow as tf
    # Prevent TF from printing additional logs during initialization
    tf.get_logger().setLevel('ERROR')
    
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("⚠️  TensorFlow not found. Please install it using 'pip install tensorflow'.")
    sys.exit(1)

# --- 4. PROJECT CONFIGURATION ---
# .parent = src, .parent.parent = aegis (Project Root)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_FILE = BASE_DIR / "data" / "autoencoder.h5"

def build_autoencoder(input_dim):
    """
    Constructs a basic Autoencoder neural network.
    Structure: Input -> Encoder (Compression) -> Decoder (Reconstruction) -> Output
    """
    input_layer = Input(shape=(input_dim,))
    
    # Encoder: Compresses input to 8 features
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)
    
    # Decoder: Reconstructs back to original dimensions
    decoded = Dense(16, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="sigmoid")(decoded)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train_autoencoder():
    print(f"Project Root: {BASE_DIR}")
    
    if not DATA_PATH.exists():
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    print("Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)

    # Preprocessing: Use only 'Normal' transactions (Class = 0)
    # We want the model to learn what "Normal" looks like so it fails on "Fraud".
    normal = df[df["Class"] == 0]
    
    # Drop target and time columns
    normal = normal.drop(columns=["Class", "Time"]) 

    print(f"Training on {len(normal)} normal transactions...")

    # Scaling: Deep Learning requires features to be between 0 and 1
    scaler = MinMaxScaler()
    normal_scaled = scaler.fit_transform(normal)

    # Build Model
    input_dim = normal_scaled.shape[1]
    model = build_autoencoder(input_dim)

    # Train
    print("Training Deep Learning model (Autoencoder)...")
    model.fit(
        normal_scaled, normal_scaled,
        epochs=10,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
        verbose=0  # verbose=0 keeps the terminal clean
    )

    # Save
    # Ensure directory exists
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {MODEL_FILE.name}...")
    model.save(MODEL_FILE)
    print("✅ Autoencoder trained and saved successfully.")

if __name__ == "__main__":
    train_autoencoder()
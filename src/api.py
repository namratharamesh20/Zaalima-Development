import os
import warnings
import logging

# --- 1. LOGGING AND WARNING CONFIGURATION ---

# Suppress TensorFlow C++ logs (Must be set before importing TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress Python-level warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Configure standard logging to suppress root and library-specific logs
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# --- 2. LIBRARY IMPORTS ---
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

# --- 3. TENSORFLOW SAFE IMPORT ---
try:
    import tensorflow as tf
    # Apply additional silencing within the TensorFlow context
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = FastAPI()

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "results.csv"
ISO_FOREST_FILE = BASE_DIR / "data" / "isolation_forest.pkl"
AUTOENCODER_FILE = BASE_DIR / "data" / "autoencoder.h5"

# --- MODEL LOADING ---

# Attempt to load the Isolation Forest model
iso_model = None
if ISO_FOREST_FILE.exists():
    try:
        iso_model = joblib.load(ISO_FOREST_FILE)
        print("System: Isolation Forest Model Loaded Successfully")
    except Exception as e:
        print(f"System Error: Isolation Forest Load Failed - {e}")
else:
    print(f"System Warning: Isolation Forest file not found at {ISO_FOREST_FILE}")

# Attempt to load the Autoencoder model
ae_model = None
if TF_AVAILABLE and AUTOENCODER_FILE.exists():
    try:
        ae_model = load_model(AUTOENCODER_FILE)
        print("System: Autoencoder Model Loaded Successfully")
    except Exception as e:
        pass
elif not AUTOENCODER_FILE.exists():
    print(f"System Warning: Autoencoder file not found at {AUTOENCODER_FILE}")

class Transaction(BaseModel):
    txn_id: str
    user_id: str
    amount: float
    country: str
    timestamp: str

# --- MEMORY MANAGEMENT ---
user_history = {}

def load_memory():
    """
    Restores user location history from the existing dataset.
    This ensures the system remembers the last known valid location of a user
    to accurately detect impossible travel even after a system restart.
    """
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            if "Timestamp" in df.columns:
                df = df.sort_values("Timestamp")
            for _, row in df.iterrows():
                # Only update history for legitimate transactions
                if row.get("Decision") == "ALLOWED":
                    user_history[str(row["User ID"])] = {
                        "country": row["Country"], 
                        "timestamp": row["Timestamp"]
                    }
            # CHANGED: Generic success message without user count
            print("System: User location memory restored successfully.")
        except Exception as e:
            print(f"System Error: Failed to load memory - {e}")

# Initialize memory on startup
load_memory()

@app.post("/check")
async def check_transaction(txn: Transaction):
    """
    Main endpoint to process a transaction.
    Runs the transaction through a multi-layer fraud detection engine.
    """
    decision = "ALLOWED"
    reasons = []
    email_sent = "NO"
    
    # --- LAYER 1: DETERMINISTIC RULES ---
    
    # Rule 1: Impossible Travel Check
    prev_loc = user_history.get(txn.user_id)
    if prev_loc:
        last_country = prev_loc["country"]
        if last_country != txn.country:
            try:
                last_time = datetime.fromisoformat(prev_loc["timestamp"])
                curr_time = datetime.fromisoformat(txn.timestamp)
                time_diff = (curr_time - last_time).total_seconds() / 3600.0
                
                # If travel happens faster than physically possible (e.g., cross-country in < 2 hours)
                if time_diff < 2.0 and time_diff > -2.0:
                    decision = "BLOCKED"
                    reasons.append(f"IMPOSSIBLE TRAVEL: {last_country} to {txn.country}")
                    email_sent = "YES"
            except: pass

    # Rule 2: High Value Transaction Limit
    if txn.amount > 50000:
        decision = "BLOCKED"
        reasons.append("HIGH VALUE LIMIT EXCEEDED")
        email_sent = "YES"

    # --- LAYER 2: STATISTICAL ANOMALY DETECTION (Isolation Forest) ---
    if decision == "ALLOWED" and iso_model is not None:
        try:
            # Reshape the amount into a 2D array for the model
            features = np.array([[txn.amount]])
            pred = iso_model.predict(features) # Returns -1 for anomaly, 1 for normal
            
            if pred[0] == -1:
                decision = "BLOCKED"
                reasons.append("STATISTICAL ANOMALY (Isolation Forest)")
                email_sent = "YES"
                print(f"System Alert: Isolation Forest flagged amount ${txn.amount}")
        except: pass

    # --- LAYER 3: DEEP LEARNING (Autoencoder) ---
    if decision == "ALLOWED" and ae_model is not None:
        try:
            # Construct a feature vector matching the training input shape (30 features)
            input_vector = np.zeros((1, 30))
            
            # Place the transaction amount in the last feature column
            input_vector[0, 29] = txn.amount
            
            # Generate reconstruction and calculate error (MSE)
            reconstruction = ae_model.predict(input_vector, verbose=0)
            mse = np.mean(np.power(input_vector - reconstruction, 2), axis=1)
            
            # If the reconstruction error is high, the pattern is suspicious
            if mse[0] > 5000: 
                decision = "BLOCKED"
                reasons.append(f"SUSPICIOUS PATTERN (AI Score: {int(mse[0])})")
                email_sent = "YES"
                print(f"System Alert: Autoencoder flagged transaction. Error: {mse[0]:.2f}")
        except Exception as e:
            pass

    # --- MEMORY UPDATE AND PERSISTENCE ---
    
    # Only update user location if the transaction is valid
    if decision == "ALLOWED":
        user_history[txn.user_id] = {"country": txn.country, "timestamp": txn.timestamp}

    # Save result to CSV for the dashboard
    new_data = pd.DataFrame([{
        "Transaction ID": txn.txn_id,
        "User ID": txn.user_id,
        "Amount": txn.amount,
        "Decision": decision,
        "Email Sent": email_sent,
        "Reasons": "; ".join(reasons),
        "Timestamp": txn.timestamp,
        "Country": txn.country 
    }])
    
    new_data.to_csv(DATA_FILE, mode='a', header=not DATA_FILE.exists(), index=False)
    
    return {"status": "processed", "decision": decision}
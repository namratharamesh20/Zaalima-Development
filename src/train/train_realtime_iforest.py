import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Generate synthetic realtime-style data
np.random.seed(42)

n_samples = 50000
user_id_mod = np.random.randint(0, 5000, n_samples)
amount = np.random.exponential(scale=200, size=n_samples)
country_len = np.random.randint(2, 3, n_samples)

X = np.column_stack([user_id_mod, amount, country_len])

from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42
)

model.fit(X)

joblib.dump(model, MODEL_DIR / "isolation_forest_realtime.pkl")

print("✅ Realtime Isolation Forest trained & saved")

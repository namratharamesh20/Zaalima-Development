import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load Kaggle dataset
df = pd.read_csv("data/creditcard.csv")

# Remove label column
features = df.drop(columns=["Class"])

# Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved!")

# Autoencoder structure
input_dim = scaled.shape[1]
encoding_dim = 14

inp = Input(shape=(input_dim,))
enc = Dense(encoding_dim, activation="relu")(inp)
dec = Dense(input_dim, activation="linear")(enc)

autoencoder = Model(inp, dec)
autoencoder.compile(optimizer="adam", loss="mse")

# Train autoencoder
autoencoder.fit(
    scaled, scaled,
    epochs=5,
    batch_size=256,
    shuffle=True
)

# Save autoencoder
autoencoder.save("models/autoencoder_model.h5")
print("Autoencoder saved!")

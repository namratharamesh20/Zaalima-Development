import nbformat as nbf

# ============================================================
# Create a new Jupyter notebook file
# ============================================================
nb = nbf.v4.new_notebook()
cells = []

# ============================================================
# 1. Introduction
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""# Kaggle Credit Card Fraud — EDA & Model Training

This notebook performs:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Training multiple supervised models
- Autoencoder anomaly detection
- Model comparison
- Fraud detection insights

Dataset: **Kaggle - Credit Card Fraud Detection**
"""))

# ============================================================
# 2. Imports & Load Dataset
# ============================================================
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

# Autoencoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load dataset
df = pd.read_csv('data/creditcard.csv')
df.head()"""))

# ============================================================
# 3. EDA Section
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## Exploratory Data Analysis (EDA)"))

cells.append(nbf.v4.new_code_cell("df.describe()"))

cells.append(nbf.v4.new_code_cell("""# Class distribution
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Non-Fraud Count')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Amount distribution
plt.hist(df['Amount'], bins=50)
plt.title('Transaction Amount Distribution')
plt.show()"""))

# ============================================================
# 4. Preprocessing
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## Data Preprocessing"))

cells.append(nbf.v4.new_code_cell("""X = df.drop(['Class'], axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_test.shape"""))

# ============================================================
# 5. Train Supervised Models
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## Train Supervised Classification Models"))

cells.append(nbf.v4.new_code_cell("""results = {}

def evaluate(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, preds)
    results[name] = auc
    print(f"\\n{name} — AUC: {auc}")
    print(classification_report(y_test, preds))

# Logistic Regression
evaluate(LogisticRegression(max_iter=2000), "Logistic Regression")

# Random Forest
evaluate(RandomForestClassifier(n_estimators=200), "Random Forest")

# Gradient Boosting
evaluate(GradientBoostingClassifier(), "Gradient Boosting")

# SVM (linear)
evaluate(SVC(kernel='linear'), "SVM Linear")

# MLP Neural Network
evaluate(MLPClassifier(hidden_layer_sizes=(32,16), max_iter=500), "MLP Classifier")

# AdaBoost
evaluate(AdaBoostClassifier(), "AdaBoost")
"""))

# ============================================================
# 6. Autoencoder
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## Autoencoder Anomaly Detection"))

cells.append(nbf.v4.new_code_cell("""# Autoencoder architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Train only on non-fraud
X_train_nonfraud = X_train[y_train == 0]

history = autoencoder.fit(
    X_train_nonfraud, X_train_nonfraud,
    epochs=10, batch_size=256, validation_split=0.1, verbose=1
)

# Compute reconstruction errors
recon = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - recon, 2), axis=1)

df_mse = pd.DataFrame({'mse': mse, 'true': y_test})
df_mse.head()
"""))

cells.append(nbf.v4.new_code_cell("""sns.histplot(df_mse[df_mse['true']==0]['mse'], label='Normal', color='blue', bins=50)
sns.histplot(df_mse[df_mse['true']==1]['mse'], label='Fraud', color='red', bins=50)
plt.legend()
plt.title("Autoencoder Reconstruction Error")
plt.show()"""))

# ============================================================
# 7. Model Comparison
# ============================================================
cells.append(nbf.v4.new_markdown_cell("## Compare Models"))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10,5))
plt.bar(results.keys(), results.values())
plt.ylabel("AUC Score")
plt.xticks(rotation=45)
plt.title("Model Comparison — AUC")
plt.show()

results"""))

# ============================================================
# 8. Summary
# ============================================================
cells.append(nbf.v4.new_markdown_cell("""# Summary

- Performed EDA on Kaggle credit card fraud dataset  
- Trained multiple supervised models  
- Implemented autoencoder anomaly detection  
- Compared model performance  
- Identified strong fraud detection candidates  

This notebook is part of the **Aegis Fraud Detection System**.
"""))

# ============================================================
# Finalize notebook
# ============================================================
nb["cells"] = cells

with open("notebooks/kaggle_eda_and_training.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook created successfully!")

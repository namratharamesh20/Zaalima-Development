# 🛡️ Aegis: Real-Time Transaction Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Scikit--Learn-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![Kafka](https://img.shields.io/badge/Streaming-Apache%20Kafka-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)

> **"Stopping fraud before it happens."**

## 📖 Overview

**Aegis** is a robust, real-time fraud detection framework designed to secure financial transactions against evolving cyber threats. Unlike traditional systems that detect fraud hours or days after the fact, Aegis processes transactions in **milliseconds**, blocking fraudulent attempts instantly.

This project was developed during my internship at **Zaalima Development** to demonstrate the power of combining **Data Engineering**, **MLOps**, and **Deep Learning** into a cohesive security solution.

---

## 🏗️ System Architecture

Aegis uses a **"Defense in Depth"** strategy, passing every transaction through a hybrid ensemble model.

1.  **Ingestion Layer (Apache Kafka):** Acts as the central nervous system, buffering high-velocity transaction streams.
2.  **Inference Engine (FastAPI):**
    * **Layer 1 (The Gatekeeper):** Instant statistical rules (e.g., velocity checks, impossible travel).
    * **Layer 2 (The Scout):** An **Isolation Forest** model to detect statistical outliers.
    * **Layer 3 (The Specialist):** A **Deep Learning Autoencoder** to identify complex, non-linear fraud patterns via reconstruction error.
3.  **Visualization Layer (Streamlit):** A live command center for monitoring threats and system health.

---

## 🚀 Key Features

* **⚡ Sub-Second Latency:** End-to-end processing time of <50ms per transaction.
* **🧠 Hybrid AI Models:** Combines the speed of machine learning with the depth of neural networks.
* **🔄 Real-Time Streaming:** Fully integrated with Apache Kafka for scalable data ingestion.
* **📊 Interactive Dashboard:** Live visualization of blocked transactions, risk distribution, and server status.
* **⚖️ Imbalance Handling:** Trained using Semi-Supervised Learning (One-Class Classification) to robustly detect unknown fraud types.

---

## 🛠️ Technology Stack

* **Language:** Python 3.x
* **Streaming:** Apache Kafka, Zookeeper
* **Machine Learning:** Scikit-Learn (Isolation Forest), TensorFlow/Keras (Autoencoder), Joblib
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy

---

## 📂 Project Structure

```text
aegis/
├── data/                   # Dataset and serialized models (.pkl, .h5)
│   ├── creditcard.csv      # Source dataset (Kaggle)
│   ├── isolation_forest.pkl
│   ├── autoencoder.h5
│   └── scaler.pkl
├── kafka/                  # Apache Kafka infrastructure
├── notebooks/              # Experimentation lab (EDA & Training)
│   └── kaggle_eda_and_training.ipynb
├── src/                    # Source code
│   ├── api.py              # Fraud detection inference engine
│   ├── producer.py         # Transaction stream simulator
│   └── dashboard/          # Streamlit frontend app
├── venv/                   # Python Virtual Environment
└── README.md               # Project Documentation
Here is a professional, detailed, and human-friendly README.md file. You can copy and paste this directly into your GitHub repository.Markdown# 🛡️ Aegis: Real-Time Transaction Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-TensorFlow%20%7C%20Scikit--Learn-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![Kafka](https://img.shields.io/badge/Streaming-Apache%20Kafka-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)

> **"Stopping fraud before it happens."**

## 📖 Overview

**Aegis** is a robust, real-time fraud detection framework designed to secure financial transactions against evolving cyber threats. Unlike traditional systems that detect fraud hours or days after the fact, Aegis processes transactions in **milliseconds**, blocking fraudulent attempts instantly.

This project was developed during my internship at **Zaalima Development** to demonstrate the power of combining **Data Engineering**, **MLOps**, and **Deep Learning** into a cohesive security solution.

---

## 🏗️ System Architecture

Aegis uses a **"Defense in Depth"** strategy, passing every transaction through a hybrid ensemble model.

1.  **Ingestion Layer (Apache Kafka):** Acts as the central nervous system, buffering high-velocity transaction streams.
2.  **Inference Engine (FastAPI):**
    * **Layer 1 (The Gatekeeper):** Instant statistical rules (e.g., velocity checks, impossible travel).
    * **Layer 2 (The Scout):** An **Isolation Forest** model to detect statistical outliers.
    * **Layer 3 (The Specialist):** A **Deep Learning Autoencoder** to identify complex, non-linear fraud patterns via reconstruction error.
3.  **Visualization Layer (Streamlit):** A live command center for monitoring threats and system health.

---

## 🚀 Key Features

* **⚡ Sub-Second Latency:** End-to-end processing time of <50ms per transaction.
* **🧠 Hybrid AI Models:** Combines the speed of machine learning with the depth of neural networks.
* **🔄 Real-Time Streaming:** Fully integrated with Apache Kafka for scalable data ingestion.
* **📊 Interactive Dashboard:** Live visualization of blocked transactions, risk distribution, and server status.
* **⚖️ Imbalance Handling:** Trained using Semi-Supervised Learning (One-Class Classification) to robustly detect unknown fraud types.

---

## 🛠️ Technology Stack

* **Language:** Python 3.x
* **Streaming:** Apache Kafka, Zookeeper
* **Machine Learning:** Scikit-Learn (Isolation Forest), TensorFlow/Keras (Autoencoder), Joblib
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy

---

## 📂 Project Structure

```text
aegis/
├── data/                   # Dataset and serialized models (.pkl, .h5)
│   ├── creditcard.csv      # Source dataset (Kaggle)
│   ├── isolation_forest.pkl
│   ├── autoencoder.h5
│   └── scaler.pkl
├── kafka/                  # Apache Kafka infrastructure
├── notebooks/              # Experimentation lab (EDA & Training)
│   └── kaggle_eda_and_training.ipynb
├── src/                    # Source code
│   ├── api.py              # Fraud detection inference engine
│   ├── producer.py         # Transaction stream simulator
│   └── dashboard/          # Streamlit frontend app
├── venv/                   # Python Virtual Environment
└── README.md               # Project Documentation
💻 Getting Started (The Runbook)Follow these steps to deploy the full Aegis system on your local machine.PrerequisitesPython 3.8+ installed.Java Runtime Environment (JRE) (Required for Kafka).1. InstallationClone the repo and install dependencies:Bashgit clone [https://github.com/namratharamesh20/Zaalima-Development.git](https://github.com/namratharamesh20/Zaalima-Development.git)
cd Aegis
pip install -r requirements.txt
2. Running the SystemTo simulate a real distributed environment, you need to run 5 separate terminals.Phase 1: Infrastructure (Start these first)Terminal 1: ZookeeperBashcd C:\kafka\kafka_2.13-3.7.2
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
Terminal 2: Kafka BrokerBashcd C:\kafka\kafka_2.13-3.7.2
.\bin\windows\kafka-server-start.bat .\config\server.properties
Phase 2: The BrainTerminal 3: FastAPI BackendBashuvicorn src.api:app --reload
You should see: "Application startup complete."Phase 3: The DashboardTerminal 4: Streamlit UIBashstreamlit run src/dashboard/app.py
This will automatically open your browser to http://localhost:8501Phase 4: SimulationTerminal 5: Transaction ProducerBashpython src/producer.py
You will see: "Sent transaction to Kafka..."📊 Model PerformanceWe trained our models on the Kaggle Credit Card Fraud Detection dataset (284k transactions).ModelRolePerformance MetricIsolation ForestStatistical Anomaly DetectionROC-AUC: 0.95AutoencoderDeep Pattern RecognitionROC-AUC: 0.96Note: Models were trained on a strict train/test split to prevent data leakage, utilizing reconstruction error (MSE) as the primary threshold for the Autoencoder.🔮 Future ImprovementsActive Learning: Implement a feedback loop to retrain models on flagged transactions that were verified as true fraud.Explainability: Integrate SHAP values into the dashboard to tell analysts why a transaction was blocked.Containerization: Dockerize the entire pipeline for easier deployment.👨‍💻 AuthorNamratha RameshRole: Intern, Zaalima DevelopmentProject Duration: 4 Weeks (Dec 2025)This project is for educational and demonstration purposes as part of the Zaalima Development internship program.

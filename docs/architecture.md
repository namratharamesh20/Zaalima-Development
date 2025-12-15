# Architecture — Aegis Real-time Fraud Detection

Pipeline:
Mock Stream → Consumer → API → Model → Actions (block/alert) → Results → Dashboard

Modules:
- Streaming ingestion: src/mock_stream.py
- Feature engineering: src/utils.py
- Model training: src/train_isolation_forest.py
- Online scoring API: src/api.py
- Blocking + alerting: src/actions.py
- Consumer pipeline: src/consumer.py
- Retraining: src/retrain.py
- Evaluation: src/evaluate.py
- Dashboard: dashboard/app.py

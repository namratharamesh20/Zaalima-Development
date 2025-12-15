import csv
from pathlib import Path
from datetime import datetime
import json

BLOCKED_FILE = Path(__file__).resolve().parent / "blocked.csv"
ALERTS_FILE = Path(__file__).resolve().parent / "alerts.csv"

def _ensure_csv(path, headers):
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

_ensure_csv(BLOCKED_FILE, ["txn_id","timestamp","reason","raw_txn"])
_ensure_csv(ALERTS_FILE, ["txn_id","timestamp","channel","recipient","message","raw_txn"])

def block_transaction(txn, reason="high_fraud_score"):
    row = {
        "txn_id": txn.get("txn_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "reason": reason,
        "raw_txn": json.dumps(txn)
    }
    with open(BLOCKED_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
    return True

def send_alert(txn, channel="sms", recipient="+91-9999999999"):
    message = f"ALERT: transaction {txn.get('txn_id')} scored high fraud ({txn.get('fraud_score')})."
    row = {
        "txn_id": txn.get("txn_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "channel": channel,
        "recipient": recipient,
        "message": message,
        "raw_txn": json.dumps(txn)
    }
    with open(ALERTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)
    return True

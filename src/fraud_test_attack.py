from kafka import KafkaProducer
import json
import time
from datetime import datetime

# CONFIG
KAFKA_TOPIC = "transactions"
KAFKA_SERVER = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=[KAFKA_SERVER],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send(txn):
    producer.send(KAFKA_TOPIC, txn)
    print(f"😈 SENT: {txn['user_id']} | ${txn['amount']} | {txn['country']}")
    time.sleep(0.5) # Fast fire

print("🔥 STARTING FRAUD ATTACK SIMULATION 🔥")

# User 666 (The Victim)
USER_ID = "666"

# 1. ESTABLISH BASELINE (Normal behavior)
print("\n--- 🟢 Phase 1: Establishing Normal Behavior (US, ~$20) ---")
for i in range(3):
    send({
        "txn_id": f"setup_{i}",
        "user_id": USER_ID,
        "amount": 20.00,
        "country": "US",
        "timestamp": datetime.utcnow().isoformat()
    })

# 2. VELOCITY ATTACK (Spamming transactions)
print("\n--- 🚀 Phase 2: Velocity Attack (Spamming) ---")
for i in range(6):
    send({
        "txn_id": f"spam_{i}",
        "user_id": USER_ID,
        "amount": 25.00,
        "country": "US",
        "timestamp": datetime.utcnow().isoformat()
    })

# 3. IMPOSSIBLE TRAVEL (Teleport US -> Japan)
print("\n--- ✈️ Phase 3: Impossible Travel (US -> JP in 1 sec) ---")
send({
    "txn_id": "teleport_1",
    "user_id": USER_ID,
    "amount": 50.00,
    "country": "JP",
    "timestamp": datetime.utcnow().isoformat()
})

# 4. AMOUNT SHOCK (Buying a Yacht)
print("\n--- 💰 Phase 4: Amount Shock ($50 vs $50,000) ---")
send({
    "txn_id": "whale_1",
    "user_id": USER_ID,
    "amount": 50000.00,
    "country": "JP",
    "timestamp": datetime.utcnow().isoformat()
})

print("\n✅ ATTACK COMPLETE. CHECK YOUR CONSUMER TERMINAL!")
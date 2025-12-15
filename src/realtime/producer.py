from kafka import KafkaProducer
import json
import time
import random
import uuid
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# CONFIG
KAFKA_TOPIC = "transactions"
KAFKA_SERVER = "localhost:9092"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "creditcard.csv"

try:
    producer = KafkaProducer(bootstrap_servers=[KAFKA_SERVER], value_serializer=lambda v: json.dumps(v).encode("utf-8"))
    print(f"Connected to Kafka")
except: producer = None

ALL_USERS = [str(i) for i in range(100, 600)] 
LOCATIONS = {"HOME": "IN", "RISK": ["US", "RU", "CN", "KP", "NG", "JP"]}

# CLOCK: Start 2 hours ago (Keeps simulation close to 'Now')
simulated_time = datetime.utcnow() - timedelta(hours=2)

def load_data():
    if not DATA_FILE.exists(): exit(1)
    df = pd.read_csv(DATA_FILE)
    return df.sample(frac=1).reset_index(drop=True)

data_stream = load_data()
count = 0

# PENDING ATTACKS QUEUE
pending_attacks = []

print(f"Simulation Started. Slow Clock Mode (Seconds).")

for index, row in data_stream.iterrows():
    count += 1
    
    # --- FIX: SLOWER TIME JUMPS (Seconds instead of Minutes) ---
    # This prevents the simulation from racing into the future
    time_jump = timedelta(seconds=random.randint(5, 45)) 
    simulated_time += time_jump
    
    # Default: Genuine Transaction
    user_id = random.choice(ALL_USERS)
    country = LOCATIONS["HOME"]
    amt = float(row['Amount'])
    is_fraud = False
    
    # --- DYNAMIC FRAUD LOGIC (Target: ~5 blocked per 100 txns) ---
    
    # A. Check if a TRAP needs to be sprung
    pending_attacks.sort(key=lambda x: x['trigger_at'])
    
    if pending_attacks and count >= pending_attacks[0]["trigger_at"]:
        attack = pending_attacks.pop(0)
        user_id = attack["user"]
        country = attack["country"]
        amt = attack["amount"]
        is_fraud = True
        
        # Impossible travel needs a tiny time gap (2 mins) to trigger alert
        simulated_time += timedelta(minutes=random.randint(2, 5))
        print(f"🚨 SPRINGING TRAP on User {user_id} in {country}")

    # B. Start new attack sequence (7% chance)
    elif random.random() < 0.07:
        attack_type = random.choice(["IMPOSSIBLE", "HIGH_VALUE"])
        
        if attack_type == "HIGH_VALUE":
            amt = random.uniform(60000, 95000) # > $50k Limit
            is_fraud = True
        
        elif attack_type == "IMPOSSIBLE":
            # Step 1: Send Safe Transaction NOW (India)
            victim = user_id 
            country = "IN"
            amt = random.uniform(50, 500)
            
            # Step 2: Schedule Fraud 5-10 txns later
            trap_country = random.choice(LOCATIONS["RISK"])
            trigger_turn = count + random.randint(5, 10)
            
            pending_attacks.append({
                "trigger_at": trigger_turn,
                "user": victim,
                "country": trap_country,
                "amount": random.uniform(2500, 9000)
            })
            print(f"👀 Setting Trap for User {victim}...")

    # Send Transaction
    txn = {
        "txn_id": str(uuid.uuid4()),
        "user_id": user_id,
        "amount": round(amt, 2),
        "country": country,
        "timestamp": simulated_time.isoformat()
    }
    
    if producer: producer.send(KAFKA_TOPIC, txn)
    
    icon = "😈" if is_fraud else "💳"
    print(f"{icon} #{count}: {user_id} | {country} | ${amt:.2f}")
    
    time.sleep(1.0)
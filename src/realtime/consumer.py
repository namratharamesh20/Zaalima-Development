import json
from kafka import KafkaConsumer
import requests
import colorama
import sys

# Initialize Colors
colorama.init(autoreset=True)

# CONFIG
TOPIC = "transactions"
API_URL = "http://127.0.0.1:8000/check"
KAFKA_SERVER = "localhost:9092"

print(f"{colorama.Fore.CYAN}🎧 Consumer Listening on '{TOPIC}'...")

try:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest" # Only read new data
    )
except Exception as e:
    print(f"{colorama.Fore.RED}❌ Kafka Connection Error: {e}")
    print("Ensure Zookeeper and Kafka are running.")
    sys.exit(1)

for msg in consumer:
    txn = msg.value
    
    try:
        # Send to API (The "Brain")
        response = requests.post(API_URL, json=txn)
        
        if response.status_code == 200:
            result = response.json()
            
           
            decision = result.get("decision", "UNKNOWN").upper()
            
            # Print Output
            user_id = txn.get("user_id", "Unknown")
            country = txn.get("country", "??")
            amt = txn.get("amount", 0)

            if decision == "BLOCKED":
                print(f"{colorama.Fore.RED}🚫 BLOCKED: User {user_id} | {country} | ${amt}")
            else:
                print(f"{colorama.Fore.GREEN}✅ ALLOWED: User {user_id} | {country} | ${amt}")
        else:
            print(f"{colorama.Fore.YELLOW}⚠️ API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"{colorama.Fore.RED}❌ System Error: {e}")
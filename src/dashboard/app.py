import streamlit as st
import pandas as pd
import time
import requests
import altair as alt
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "results.csv"
API_URL = "http://127.0.0.1:8000/check"

st.set_page_config(page_title="Aegis Fraud Monitor", layout="wide")

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
    <style>
        /* Global Font Settings */
        html, body, [class*="css"] { 
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
        }
        
        /* Metric Card Styling */
        div[data-testid="stMetricValue"] { 
            font-size: 26px; 
            font-weight: 700; 
            color: #0f172a; 
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #64748b;
            font-weight: 500;
        }
        
        /* Header Styling with Gradients */
        h1, h2, h3 { 
            background: linear-gradient(90deg, #1e3a8a, #3b82f6); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            font-weight: 700;
        }
        
        /* Data Table Styling */
        div[data-testid="stDataFrame"] { 
            border: 1px solid #e2e8f0; 
            border-radius: 6px; 
        }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Admin Controls")
    
    # Search Filter
    st.subheader("Filter Data")
    filter_user = st.text_input("Search by User ID", placeholder="Enter ID...")
    
    st.markdown("---")
    
    # Manual Test Interface
    st.subheader("Manual Injection")
    uid = st.text_input("User ID", "Test_User_1")
    amt = st.number_input("Amount ($)", value=100.0)
    
    loc_options = ["IN (India)", "US (USA)", "JP (Japan)", "RU (Russia)", "CN (China)", "NG (Nigeria)"]
    loc_selection = st.selectbox("Location", loc_options, label_visibility="collapsed")
    loc_code = loc_selection.split(" ")[0]
    
    col_d, col_t = st.columns(2)
    d = col_d.date_input("Date", datetime.now())
    t = col_t.time_input("Time", datetime.now().time())
    
    if st.button("Inject Transaction", use_container_width=True):
        ts = datetime.combine(d, t).isoformat()
        payload = {
            "txn_id": f"man_{int(time.time())}",
            "user_id": uid, "amount": amt, "country": loc_code, "timestamp": ts
        }
        try:
            requests.post(API_URL, json=payload)
            st.success("Transaction submitted successfully.")
            time.sleep(1.0)
            st.rerun()
        except Exception as e:
            st.error(f"Connection Error: {e}")

    # Reporting Feature
    if DATA_FILE.exists():
        st.markdown("---")
        st.subheader("Reporting")
        try:
            df_download = pd.read_csv(DATA_FILE)
            csv_data = df_download.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Full Report (CSV)",
                data=csv_data,
                file_name=f"fraud_report_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception:
            st.warning("Unable to generate report.")

    st.markdown("---")
    st.subheader("System")
    pause = st.toggle("Pause Live Feed", value=False)
    
    if st.button("Reset System Data", type="primary", use_container_width=True):
        try:
            cols = ["Transaction ID", "User ID", "Amount", "Decision", "Email Sent", "Reasons", "Timestamp", "Country"]
            with open(DATA_FILE, "w") as f: f.write(",".join(cols) + "\n")
            st.success("System reset complete.")
            time.sleep(0.5)
            st.rerun()
        except: pass

# --- MAIN DASHBOARD ---
st.title("Aegis Credit Card Fraud Monitor")

# Top Level Metrics
m1, m2, m3, m4 = st.columns(4)
metric_total = m1.empty()
metric_blocked = m2.empty()
metric_emails = m3.empty()
metric_status = m4.empty()

st.divider()

# Live Data Grid
c1, c2 = st.columns([1, 1.2])

with c1: 
    st.subheader("Live Transaction Feed")
    table_live = st.empty()

with c2: 
    st.subheader("Detected Threats")
    table_log = st.empty()

st.divider()

# Analytics Section
st.subheader("Analytics Overview")
chart_container = st.empty()

# --- HELPER: CATEGORIZE RISKS ---
def categorize_risk(reason):
    """
    Simplifies complex API error messages into clean, readable categories.
    NOW FIXED: Catches Isolation Forest anomalies correctly.
    """
    r = str(reason).upper()
    
    if "HIGH VALUE" in r:
        return "High Value Limit"
    
    elif "IMPOSSIBLE" in r:
        return "Impossible Travel"
    
    elif "SUSPICIOUS" in r or "AUTOENCODER" in r:
        return "AI Pattern Anomaly"
        
    elif "ISOLATION" in r:  # <--- THIS WAS MISSING BEFORE
        return "Statistical Anomaly"
        
    return "Other Risk"

# --- MAIN DATA LOOP ---
while True:
    if not pause:
        if DATA_FILE.exists():
            try:
                df = pd.read_csv(DATA_FILE)
                if not df.empty and "Timestamp" in df.columns:
                    # 1. Sort Data
                    df = df.sort_values(by="Timestamp", ascending=False)
                    
                    # 2. Format Timestamps
                    try:
                        df["Time_Obj"] = pd.to_datetime(df["Timestamp"])
                        df["Timestamp_Display"] = df["Time_Obj"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        df["Timestamp_Display"] = df["Timestamp"]

                    # 3. Apply Search Filter
                    if filter_user:
                        df_display = df[df["User ID"].astype(str) == filter_user]
                    else:
                        df_display = df

                    # 4. Update Metrics
                    total = len(df)
                    blocked_df = df[df["Decision"].str.upper() == "BLOCKED"].copy()
                    blocked_count = len(blocked_df)
                    emails_count = len(df[df["Email Sent"].str.upper() == "YES"]) if "Email Sent" in df.columns else 0
                    
                    metric_total.metric("Total Processed", total)
                    metric_blocked.metric("Blocked Threats", blocked_count)
                    metric_emails.metric("Alerts Sent", emails_count)
                    metric_status.metric("System Status", "ONLINE")

                    # 5. Live Feed Table
                    if "Country" not in df_display.columns: df_display["Country"] = "..."
                    
                    cols_live = ["User ID", "Country", "Amount", "Decision", "Email Sent", "Timestamp_Display"]
                    valid_live = [c for c in cols_live if c in df_display.columns]
                    
                    def style_feed(row):
                        status = str(row.get("Decision", "")).upper()
                        if status == "BLOCKED": 
                            return ['background-color: #fee2e2; color: #991b1b; font-weight: 500'] * len(row)
                        return ['background-color: #f0fdf4; color: #166534'] * len(row)

                    display_table = df_display[valid_live].head(15).rename(columns={"Timestamp_Display": "Time"})
                    
                    table_live.dataframe(
                        display_table.style.apply(style_feed, axis=1), 
                        use_container_width=True, 
                        hide_index=True
                    )

                    # 6. Threat Log Table (With Categorized Risks)
                    if not blocked_df.empty:
                        # Apply the cleaner function to the 'Reasons' column
                        blocked_df["Risk Category"] = blocked_df["Reasons"].apply(categorize_risk)
                        blocked_df["Time"] = pd.to_datetime(blocked_df["Timestamp"]).dt.strftime('%H:%M:%S')
                        
                        cols_log = ["Time", "User ID", "Country", "Amount", "Risk Category"]
                        log_display = blocked_df[cols_log].head(50)
                        
                        table_log.dataframe(
                            log_display, 
                            column_config={
                                "Amount": st.column_config.NumberColumn(format="$%.2f"),
                                "Risk Category": st.column_config.TextColumn("Risk Type"), 
                                "User ID": st.column_config.TextColumn("User"),
                            },
                            use_container_width=True, 
                            hide_index=True
                        )
                    else:
                        table_log.info("No active threats detected in current session.")

                    # 7. Advanced Analytics (Pie Chart Only)
                    if not blocked_df.empty:
                        # Group by the SAME 'Risk Category' used in the table above
                        risk_counts = blocked_df["Risk Category"].value_counts().reset_index()
                        risk_counts.columns = ["Risk Type", "Count"]
                        
                        # Create Pie Chart
                        base = alt.Chart(risk_counts).encode(
                            theta=alt.Theta("Count", stack=True)
                        )
                        
                        pie = base.mark_arc(outerRadius=120, innerRadius=0).encode(
                            color=alt.Color("Risk Type", scale=alt.Scale(scheme='set1')),
                            order=alt.Order("Count", sort="descending"),
                            tooltip=["Risk Type", "Count"]
                        )
                        
                        text = base.mark_text(radius=140).encode(
                            text=alt.Text("Count"),
                            order=alt.Order("Count", sort="descending"),
                            color=alt.value("black")  
                        )
                        
                        chart_container.altair_chart(
                            (pie + text).properties(title="Threat Distribution by Reason"), 
                            use_container_width=True
                        )
                    else:
                        chart_container.info("Waiting for data to generate analytics...")

            except Exception:
                pass
    time.sleep(1)
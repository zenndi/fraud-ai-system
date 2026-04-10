"""
Fraud Guard AI — Dashboard v5.0
Gelişmiş görsel, gerçek zamanlı WebSocket ve kapsamlı analitik
Kullanım: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import redis
import json
import queue
import random

# ==================== CONFIG ====================
API_BASE = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/alerts"

# Redis Bağlantısı 
r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)

# Global thread-safe kuyruk (session_state dışı)
if "shared_queue" not in globals():
    shared_queue = queue.Queue()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Fraud Guard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== GLOBAL CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;500;700;900&display=swap');

:root {
    --bg-primary: #060b14;
    --bg-card: #0d1626;
    --bg-card-hover: #131f35;
    --border: #1e3a5f;
    --border-bright: #2563eb;
    --text-primary: #e2e8f0;
    --text-secondary: #64748b;
    --text-muted: #334155;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-green: #22c55e;
    --accent-yellow: #f59e0b;
    --accent-red: #ef4444;
    --accent-orange: #f97316;
    --font-mono: 'JetBrains Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}


html, body, [class*="css"], .stMarkdown, p, div, label { 
    font-family: var(--font-body) !important; 
}
code, .mono { font-family: var(--font-mono) !important; }

/* Streamlit root background */
.stApp { background: var(--bg-primary) !important; }
section[data-testid="stSidebar"] { background: #080f1c !important; border-right: 1px solid var(--border); }
.block-container { padding-top: 5.5rem !important; max-width: 1600px; }

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--border-bright); }
[data-testid="metric-container"] label { color: var(--text-secondary) !important; font-size: 0.75rem !important; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.9rem !important; font-weight: 900 !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ---- Tabs ---- */
.stTabs [role="tablist"] { background: var(--bg-card); border-radius: 10px; padding: 4px; border: 1px solid var(--border); gap: 4px; }
.stTabs [role="tab"] { color: var(--text-secondary) !important; border-radius: 7px !important; font-weight: 600 !important; font-size: 0.85rem !important; }
.stTabs [role="tab"][aria-selected="true"] { background: var(--border-bright) !important; color: white !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ---- DataFrames ---- */
.stDataFrame { border-radius: 10px; border: 1px solid var(--border); overflow: hidden; }

/* ---- Inputs / Sliders ---- */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] { padding-top: 4px; }

/* ---- Info / Warning / Error ---- */
.stInfo { background: rgba(59,130,246,0.1) !important; border: 1px solid rgba(59,130,246,0.3) !important; border-radius: 10px !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 10px !important; }
.stSuccess { background: rgba(34,197,94,0.1) !important; border: 1px solid rgba(34,197,94,0.3) !important; border-radius: 10px !important; }
.stError { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 10px !important; }


/* ---- Custom components ---- */
.header-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.header-subtitle { color: var(--text-secondary); font-size: 0.9rem; margin-top: 4px; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em;
}
.badge-online { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
.badge-offline { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }
.badge-sim { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

.fraud-alert-banner {
    background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(220,38,38,0.1));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    animation: alertPulse 2s ease-in-out infinite;
}
@keyframes alertPulse { 0%,100% { border-color: rgba(239,68,68,0.5); } 50% { border-color: #ef4444; box-shadow: 0 0 20px rgba(239,68,68,0.2); } }

.alert-title { color: #ef4444; font-weight: 900; font-size: 0.8rem; letter-spacing: 0.12em; margin-bottom: 4px; }
.alert-body { color: var(--text-primary); font-size: 0.9rem; }
.alert-meta { color: var(--text-secondary); font-size: 0.75rem; font-family: var(--font-mono); margin-top: 4px; }

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.stat-label { color: var(--text-secondary); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.stat-value { color: var(--text-primary); font-size: 1.5rem; font-weight: 900; }
.stat-sub { color: var(--text-secondary); font-size: 0.75rem; margin-top: 2px; }

.tx-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; margin-bottom: 6px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; transition: border-color 0.2s;
}
.tx-row:hover { border-color: var(--border-bright); }
.tx-id { color: var(--text-secondary); font-family: var(--font-mono); font-size: 0.78rem; }
.tx-amount { color: var(--text-primary); font-weight: 700; }
.risk-pill {
    padding: 3px 10px; border-radius: 12px; font-size: 0.72rem; font-weight: 700;
}
.risk-critical { background: rgba(239,68,68,0.2); color: #ef4444; }
.risk-high { background: rgba(249,115,22,0.2); color: #f97316; }
.risk-medium { background: rgba(245,158,11,0.2); color: #f59e0b; }
.risk-low { background: rgba(34,197,94,0.2); color: #22c55e; }
.risk-none { background: rgba(100,116,139,0.2); color: #64748b; }

.sidebar-section { margin-bottom: 24px; }
.sidebar-label { color: var(--text-secondary); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }

/* Sidebar'daki ikon hatasını gizler */
[data-testid="stSidebarCollapseIcon"], 
[data-testid="collapsedControl"],
button[kind="headerNoContext"] {
    display: none !important;
}

/* ---- Footer ---- */
.footer-container {
    margin-top: 5rem;
    padding: 2rem 0;
    border-top: 1px solid var(--border);
    text-align: center;
}
.footer-text {
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-family: var(--font-body);
    letter-spacing: 0.02em;
}
.footer-credits {
    font-family: var(--font-mono);
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


</style>
""", unsafe_allow_html=True)


#==================== DATA FETCHING FUNCTIONS ====================
def get_redis_stats():
    try:
        return {
            "total_processed": int(r.get("stats:total_processed") or 0),
            "total_fraud": int(r.get("stats:total_fraud") or 0),
            "total_amount": float(r.get("stats:total_amount") or 0.0),
        }
    except:
        return {"total_processed": 0, "total_fraud": 0, "total_amount": 0.0}

def fetch_live_txns(limit=100):
    try:
        raw_data = r.lrange("fraud_stream", 0, limit - 1)
        return [json.loads(x) for x in raw_data]
    except:
        return []



# ==================== SESSION STATE INIT ====================
# Redis'ten güncel rakamları alıp başlangıç değerlerini onlarla dolduruyoruz
current_stats = get_redis_stats()

defaults = {
    "alerts": [],
    "all_transactions": [],
    "ws_thread_started": False,
    "api_online": False,
    "total_processed": current_stats["total_processed"], # Başlangıcı Redis'ten al,
    "total_fraud": current_stats["total_fraud"],  # Başlangıcı Redis'ten al
    "sim_running": False,
    "auto_refresh": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== WEBSOCKET THREAD ====================
def start_websocket_thread():
    def run():
        while True:
            try:
                import websocket
                def on_message(ws, message):
                    try:
                        data = json.loads(message)
                        shared_queue.put(data)
                    except Exception:
                        pass
                ws = websocket.WebSocketApp(WS_URL, on_message=on_message)
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception:
                pass
            time.sleep(5)

    t = threading.Thread(target=run, daemon=True)
    t.start()

if not st.session_state.ws_thread_started:
    try:
        import websocket as _ws_test
        start_websocket_thread()
        st.session_state.ws_thread_started = True
    except ImportError:
        pass

# ==================== QUEUE PROCESSOR ====================
def process_queue():
    has_new = False
    while not shared_queue.empty():
        try:
            data = shared_queue.get_nowait()
            event = data.get("event", "")
            if event == "fraud_alert":
                st.session_state.alerts.insert(0, data)
                st.session_state.all_transactions.insert(0, data)
                has_new = True
            elif event == "transaction":
                st.session_state.all_transactions.insert(0, data)
                has_new = True
            elif event == "stats_update":
                st.session_state.total_processed = data.get("total_processed", st.session_state.total_processed)
                st.session_state.total_fraud = data.get("total_fraud", st.session_state.total_fraud)
        except queue.Empty:
            break
    # Trim lists
    st.session_state.alerts = st.session_state.alerts[:100]
    st.session_state.all_transactions = st.session_state.all_transactions[:500]
    return has_new

process_queue()

# ==================== API HELPERS ====================
@st.cache_data(ttl=3)
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/api/v1/stats", timeout=2)
        if r.status_code == 200:
            return r.json(), True
    except Exception:
        pass
    return {}, False

@st.cache_data(ttl=1)
def fetch_live_txns(limit=100):
    try:
        #redisteki listeyi çekelim
        #0'dan limit- 1'e kadar olan elemanları alalım
        raw_data = r.lrange("fraud_stream", 0, limit-1)

        #redisten gelen her bir json stringini python sözlüğüne çevririrz
        transactions = [json.loads(x) for x in raw_data]
        return transactions
    except Exception as e:
        # Eğer Redis'te bir sorun olursa boş liste dön ki dashboard hata vermesin
        return []


   
   
    #try:
        #r = requests.get(f"{API_BASE}/api/v1/live?limit={limit}", timeout=2)
        #if r.status_code == 200:
            #return r.json().get("transactions", [])
    #except Exception:
        #pass
    #return []

@st.cache_data(ttl=5)
def fetch_model_info():
    try:
        r = requests.get(f"{API_BASE}/model/info", timeout=2)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

@st.cache_data(ttl=5)
def fetch_hourly():
    try:
        r = requests.get(f"{API_BASE}/api/v1/analytics/hourly", timeout=2)
        if r.status_code == 200:
            return r.json().get("hourly", [])
    except Exception:
        pass
    return []

def send_predict(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
        if r.status_code == 200:
            return r.json(), None
        return None, r.text
    except Exception as e:
        return None, str(e)



# ==================== CHART HELPERS ====================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(30,58,95,0.4)", linecolor="rgba(30,58,95,0.6)"),
    yaxis=dict(gridcolor="rgba(30,58,95,0.4)", linecolor="rgba(30,58,95,0.6)"),
    #legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(30,58,95,0.3)", borderwidth=1),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(**kwargs)
    return fig

def risk_color(score: float) -> str:
    if score >= 0.9:   return "#ef4444"
    elif score >= 0.7: return "#f97316"
    elif score >= 0.5: return "#f59e0b"
    elif score >= 0.3: return "#22c55e"
    return "#64748b"

def risk_label(score: float) -> str:
    if score >= 0.9:   return "CRITICAL"
    elif score >= 0.7: return "HIGH"
    elif score >= 0.5: return "MEDIUM"
    elif score >= 0.3: return "LOW"
    return "NONE"

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown('<div class="header-title" style="font-size:1.6rem;">🛡️ Fraud Guard</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">AI Detection System</div>', unsafe_allow_html=True)
    st.markdown("---")

    # API Status
    stats, api_ok = fetch_stats()
    st.session_state.api_online = api_ok

    if api_ok:
        st.markdown('<span class="status-badge badge-online">● API ONLINE</span>', unsafe_allow_html=True)
        model_loaded = stats.get("model_loaded", False)
        if model_loaded:
            st.markdown('<span class="status-badge badge-online" style="margin-top:6px;">🧠 MODEL LOADED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-sim" style="margin-top:6px;">⚙️ SIMULATION MODE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-offline">● API OFFLINE</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick stats sidebar
    st.markdown('<div class="sidebar-label">Session Stats</div>', unsafe_allow_html=True)
    total = stats.get("total_processed", 0)
    fraud = stats.get("total_fraud", 0)
    rate = stats.get("fraud_rate", 0)
    saved = stats.get("amount_saved", 0)

    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Processed</div>
        <div class="stat-value">{total:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Fraud Detected</div>
        <div class="stat-value" style="color:#ef4444;">{fraud:,}</div>
        <div class="stat-sub">Fraud Rate: {rate:.2f}%</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Amount Blocked</div>
        <div class="stat-value" style="color:#22c55e;">${saved:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Threshold control
    st.markdown('<div class="sidebar-label">Decision Threshold</div>', unsafe_allow_html=True)
    current_thresh = stats.get("threshold", 0.5)
    new_thresh = st.slider("Risk Threshold", 0.1, 0.9, float(current_thresh), 0.05,
                           help="Minimum risk score for flagging as fraud")
    if new_thresh != current_thresh and api_ok:
        if st.button("Apply Threshold", use_container_width=True):
            try:
                requests.post(f"{API_BASE}/model/threshold?new_threshold={new_thresh}", timeout=2)
                st.success(f"Threshold → {new_thresh}")
                fetch_stats.clear()
            except Exception:
                st.error("Update failed")

    st.markdown("---")

    # Auto refresh
    st.markdown('<div class="sidebar-label">Display Options</div>', unsafe_allow_html=True)
    st.session_state.auto_refresh = st.toggle("Auto Refresh (3s)", value=st.session_state.auto_refresh)
    show_normals = st.toggle("Show Normal Transactions", value=True)
    alert_sound = st.toggle("Alert Notifications", value=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="color:#334155;font-size:0.7rem;font-family:JetBrains Mono;text-align:center;">Uptime: {stats.get("uptime","—")}<br>WS: {stats.get("ws_connections",0)} conn</div>', unsafe_allow_html=True)

# ==================== MAIN HEADER ====================
col_title, col_time = st.columns([4, 1])
with col_title:
    st.markdown('<div class="header-title">🛡️ Fraud Guard AI Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Real-time AI-Powered Transaction Monitoring & Fraud Detection</div>', unsafe_allow_html=True)
with col_time:
    st.markdown(f'<div style="text-align:right;color:#64748b;font-family:JetBrains Mono;font-size:0.8rem;padding-top:8px;">{datetime.now().strftime("%d %b %Y")}<br><span style="font-size:1.2rem;color:#e2e8f0;font-weight:700;">{datetime.now().strftime("%H:%M:%S")}</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== LIVE FRAUD ALERT BANNER ====================
if st.session_state.alerts:
    latest = st.session_state.alerts[0]
    risk = float(latest.get("risk_score", 0))
    amount = float(latest.get("amount", 0))
    tx_id = str(latest.get("transaction_id", ""))[:12]
    status = latest.get("status", "").upper()
    ts = latest.get("timestamp", "")[:19].replace("T", " ")
    rules = latest.get("rules", [])
    st.markdown(f"""
    <div class="fraud-alert-banner">
        <div class="alert-title">🚨 LIVE FRAUD ALERT — {len(st.session_state.alerts)} ACTIVE</div>
        <div class="alert-body">
            Risk Score: <strong style="color:#ef4444;">{risk:.1%}</strong> &nbsp;|&nbsp;
            Amount: <strong>${amount:,.2f}</strong> &nbsp;|&nbsp;
            Status: <strong>{status}</strong>
            {(' &nbsp;|&nbsp; Rules: <strong>' + ', '.join(rules) + '</strong>') if rules else ''}
        </div>
        <div class="alert-meta">TX: {tx_id}... &nbsp;•&nbsp; {ts}</div>
    </div>
    """, unsafe_allow_html=True)

# ==================== KPI METRICS ====================
k1, k2, k3, k4, k5 = st.columns(5)

fraud_rate = stats.get("fraud_rate", 0)
blocked = stats.get("total_blocked", 0)
ws_conn = stats.get("ws_connections", 0)

k1.metric("Total Transactions", f"{total:,}", f"+{max(0,total-st.session_state.get('_prev_total',total))} new")
k2.metric("Fraud Detected", f"{fraud:,}", f"{fraud_rate:.2f}% rate", delta_color="inverse")
k3.metric("Blocked", f"{blocked:,}", f"${saved:,.0f} saved")
k4.metric("Risk Threshold", f"{current_thresh:.2f}", "Active model config")
k5.metric("WS Connections", ws_conn, f"{'🟢' if ws_conn > 0 else '⚫'} live")

# Store previous total for delta
st.session_state["_prev_total"] = total

st.markdown("<br>", unsafe_allow_html=True)

# ==================== MAIN TABS ====================
tab_live, tab_analytics, tab_predict, tab_batch, tab_model = st.tabs([
    "📡 Live Monitor",
    "📊 Analytics",
    "🎯 Manual Predict",
    "⚡ Batch Test",
    "🤖 Model Info"
])

# ──────────────────────────────────────────────────────────────
# TAB 1 — LIVE MONITOR
# ──────────────────────────────────────────────────────────────
with tab_live:
    col_feed, col_risk_gauge = st.columns([3, 1])

    with col_feed:
        st.markdown("#### 🔴 Real-time Transaction Feed")
        st.caption("Real-time transaction stream updated via WebSocket")

        # Fetch live from API
        live_txns = fetch_live_txns(100)

        # Merge with WS alerts
        all_shown = list(st.session_state.all_transactions) + live_txns
        seen_ids = set()
        unique_txns = []
        for tx in all_shown:
            tid = tx.get("transaction_id", "")
            if tid not in seen_ids:
                seen_ids.add(tid)
                unique_txns.append(tx)

        if not unique_txns and not st.session_state.api_online:
            st.info("API'ye bağlanılamıyor. `uvicorn api:app --port 8000` komutunu çalıştırın.")
        elif not unique_txns:
            st.info("📭 No requests yet. Send a request to the `/predict` endpoint.")
        else:
            # Build display table
            rows = []
            for tx in unique_txns[:50]:
                risk = float(tx.get("risk_score", 0))
                is_fraud = tx.get("is_fraud", risk >= 0.5)
                if not show_normals and not is_fraud:
                    continue
                rows.append({
                    "🔑 TX ID": str(tx.get("transaction_id", ""))[:14] + "...",
                    "💰 Amount": f"${float(tx.get('amount', 0)):,.2f}",
                    "⚠️ Risk": f"{risk:.1%}",
                    "🏷️ Level": risk_label(risk),
                    "📋 Status": str(tx.get("status", "")).upper(),
                    "🕐 Time": str(tx.get("timestamp", ""))[:19].replace("T", " "),
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True, height=420)
            else:
                st.info("No transactions to display (filter: fraud only)")

        col_clear, col_refresh = st.columns([1, 1])
        with col_clear:
            if st.button("🗑️ Clear Feed", use_container_width=True):
                st.session_state.alerts = []
                st.session_state.all_transactions = []
                fetch_live_txns.clear()
                st.rerun()
        with col_refresh:
            if st.button("🔄 Refresh Now", use_container_width=True):
                fetch_live_txns.clear()
                fetch_stats.clear()
                st.rerun()

    with col_risk_gauge:
        st.markdown("#### Risk Snapshot")

        # Real-time risk gauge
        last_risk = 0.0
        if unique_txns:
            last_risk = float(unique_txns[0].get("risk_score", 0))

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=last_risk * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0", "family": "JetBrains Mono"}},
            delta={"reference": 50, "font": {"size": 14}},
            title={"text": "Latest Risk", "font": {"size": 14, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar": {"color": risk_color(last_risk), "thickness": 0.3},
                "bgcolor": "#0d1626",
                "borderwidth": 1,
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                    {"range": [30, 50], "color": "rgba(245,158,11,0.1)"},
                    {"range": [50, 70], "color": "rgba(249,115,22,0.1)"},
                    {"range": [70, 100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line": {"color": "#ef4444", "width": 2}, "thickness": 0.8, "value": current_thresh * 100},
            }
        ))
        fig_gauge = apply_layout(
            fig_gauge,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            height=240,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Fraud alerts list
        st.markdown("#### 🚨 Recent Alerts")
        if st.session_state.alerts:
            for alert in st.session_state.alerts[:8]:
                risk_a = float(alert.get("risk_score", 0))
                amt_a = float(alert.get("amount", 0))
                label = risk_label(risk_a)
                css_class = f"risk-{label.lower()}"
                st.markdown(f"""
                <div class="tx-row">
                    <div>
                        <div class="tx-id">{str(alert.get('transaction_id',''))[:10]}...</div>
                        <div class="tx-amount">${amt_a:,.2f}</div>
                    </div>
                    <span class="risk-pill {css_class}">{label}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#334155;font-size:0.85rem;text-align:center;padding:20px;">No active alerts</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 — ANALYTICS
# ──────────────────────────────────────────────────────────────
with tab_analytics:
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("#### Hourly Transaction Volume")
        hourly = fetch_hourly()
        if hourly:
            df_hourly = pd.DataFrame(hourly)
            fig_hourly = go.Figure()
            fig_hourly.add_trace(go.Bar(
                x=df_hourly["hour"], y=df_hourly["total"],
                name="Total", marker_color="rgba(59,130,246,0.7)",
                marker_line_color="#3b82f6", marker_line_width=1
            ))
            fig_hourly.add_trace(go.Bar(
                x=df_hourly["hour"], y=df_hourly["fraud"],
                name="Fraud", marker_color="rgba(239,68,68,0.85)",
                marker_line_color="#ef4444", marker_line_width=1
            ))
            fig_hourly = apply_layout(fig_hourly, title="", barmode="overlay",
                                     height=280, showlegend=True,
                                     legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig_hourly, use_container_width=True)
        else:
            # Demo chart
            hours = [f"{h:02d}:00" for h in range(24)]
            totals = [random.randint(5, 80) for _ in hours]
            frauds = [max(0, int(t * random.uniform(0.02, 0.15))) for t in totals]
            df_demo = pd.DataFrame({"hour": hours, "total": totals, "fraud": frauds})
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(x=df_demo["hour"], y=df_demo["total"], name="Total", marker_color="rgba(59,130,246,0.6)"))
            fig_d.add_trace(go.Bar(x=df_demo["hour"], y=df_demo["fraud"], name="Fraud", marker_color="rgba(239,68,68,0.8)"))
            fig_d = apply_layout(fig_d, barmode="overlay", height=280, showlegend=True,
                                legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig_d, use_container_width=True)
            st.caption("⚠️ Demo data is displayed — use the API for real trading data.")

    with col_a2:
        st.markdown("#### Risk Score Distribution")
        live_data = fetch_live_txns(200)
        if live_data:
            scores = [float(t.get("risk_score", 0)) for t in live_data]
            fraud_scores = [s for t, s in zip(live_data, scores) if t.get("is_fraud")]
            normal_scores = [s for t, s in zip(live_data, scores) if not t.get("is_fraud")]

            fig_dist = go.Figure()
            if normal_scores:
                fig_dist.add_trace(go.Histogram(
                    x=normal_scores, name="Normal", nbinsx=20,
                    marker_color="rgba(34,197,94,0.6)",
                    marker_line_color="#22c55e", marker_line_width=1
                ))
            if fraud_scores:
                fig_dist.add_trace(go.Histogram(
                    x=fraud_scores, name="Fraud", nbinsx=20,
                    marker_color="rgba(239,68,68,0.7)",
                    marker_line_color="#ef4444", marker_line_width=1
                ))
            fig_dist.add_vline(x=current_thresh, line_color="#f59e0b",
                               line_width=2, line_dash="dash",
                               annotation_text=f"Threshold ({current_thresh})",
                               annotation_font_color="#f59e0b")
            fig_dist = apply_layout( fig_dist, barmode="overlay", height=280, showlegend=True, legend=dict(orientation="h", y=1.1, x=0))
           
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            # Demo distribution
            normal_d = np.random.beta(2, 6, 200) * 0.6
            fraud_d = np.random.beta(6, 2, 30) * 0.5 + 0.5
            fig_d2 = go.Figure()
            fig_d2.add_trace(go.Histogram(x=normal_d, name="Normal", nbinsx=20,
                                          marker_color="rgba(34,197,94,0.6)"))
            fig_d2.add_trace(go.Histogram(x=fraud_d, name="Fraud", nbinsx=10,
                                          marker_color="rgba(239,68,68,0.7)"))
            fig_d2.add_vline(x=current_thresh, line_color="#f59e0b", line_width=2, line_dash="dash")
            fig_d2 = apply_layout(fig_d2, barmode="overlay", height=280,
                                 showlegend=True, legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig_d2, use_container_width=True)
            st.caption("⚠️ Demo data — submit a transaction for the actual distribution.")

    # Risk Category Pie
    col_pie, col_trend = st.columns(2)

    with col_pie:
        st.markdown("#### Risk Level Breakdown")
        if live_data:
            counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
            for tx in live_data:
                lbl = risk_label(float(tx.get("risk_score", 0)))
                counts[lbl] = counts.get(lbl, 0) + 1
        else:
            counts = {"CRITICAL": 5, "HIGH": 12, "MEDIUM": 28, "LOW": 85, "NONE": 170}

        labels = list(counts.keys())
        values = list(counts.values())
        colors = ["#ef4444", "#f97316", "#f59e0b", "#22c55e", "#475569"]

        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.65,
            marker=dict(colors=colors, line=dict(color="#060b14", width=3)),
            textfont=dict(size=12, color="white"),
            textinfo="percent+label",
        ))
        fig_pie.add_annotation(
            text=f"<b>{sum(values)}</b><br><span style='font-size:10px'>Total</span>",
            x=0.5, y=0.5, font_size=18, showarrow=False, font_color="#e2e8f0"
        )
        fig_pie = apply_layout(
            fig_pie,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(orientation="v", bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94a3b8", size=11))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_trend:
        st.markdown("#### Risk Score Trend (Recent 50)")
        if live_data:
            recent = live_data[-50:]
            times_series = [t.get("timestamp", "")[-8:-3] for t in recent]
            risks_series = [float(t.get("risk_score", 0)) for t in recent]
            is_fraud_series = [bool(t.get("is_fraud", False)) for t in recent]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=list(range(len(risks_series))), y=risks_series,
                mode="lines",
                line=dict(color="#3b82f6", width=1.5),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
                name="Risk Score"
            ))
            # Mark fraud points
            fraud_idx = [i for i, f in enumerate(is_fraud_series) if f]
            fraud_vals = [risks_series[i] for i in fraud_idx]
            if fraud_idx:
                fig_trend.add_trace(go.Scatter(
                    x=fraud_idx, y=fraud_vals, mode="markers",
                    marker=dict(color="#ef4444", size=8, symbol="circle",
                                line=dict(color="white", width=1)),
                    name="Fraud"
                ))
            fig_trend.add_hline(y=current_thresh, line_color="#f59e0b",
                                line_width=1.5, line_dash="dot")
            fig_trend = apply_layout(fig_trend, height=300, showlegend=True,
                                    legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            # Demo trend
            n = 50
            xs = list(range(n))
            ys = np.cumsum(np.random.randn(n) * 0.05) * 0.3 + 0.25
            ys = np.clip(ys, 0, 1)
            fig_t = go.Figure(go.Scatter(x=xs, y=ys, mode="lines",
                                         line=dict(color="#3b82f6", width=2),
                                         fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
            fig_t.add_hline(y=current_thresh, line_color="#f59e0b", line_width=1.5, line_dash="dot")
            fig_t = apply_layout(fig_t, height=300)
            st.plotly_chart(fig_t, use_container_width=True)
            st.caption("⚠️ Demo verisi.")

# ──────────────────────────────────────────────────────────────
# TAB 3 — MANUAL PREDICT
# ──────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("#### 🎯 Single Transaction Analysis")
    st.caption("Analyze a transaction manually and view the AI prediction.")

    col_form, col_result = st.columns([2, 1])

    with col_form:
        with st.container():
            st.markdown("**Transaction Details**")
            p1, p2 = st.columns(2)
            with p1:
                amount = st.number_input("💰 Amount (USD)", min_value=0.01, value=250.00, step=10.0, format="%.2f")
                tx_time = st.slider("🕐 Transaction Time (seconds)", 0, 86400, 43200,
                                    help="0 = midnight, 43200 = noon, 86400 = end of day")
                time_label = str(timedelta(seconds=tx_time))[:-3]
                st.caption(f"Local time approx: {time_label}")
            with p2:
                currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "TRY", "JPY"])
                payment = st.selectbox("Payment Method", ["credit_card", "debit_card", "digital_wallet", "bank_transfer"])

            st.markdown("**Optional Metadata**")
            p3, p4, p5 = st.columns(3)
            with p3:
                user_id = st.text_input("User ID", placeholder="user_123")
                merchant_id = st.text_input("Merchant ID", placeholder="merch_456")
            with p4:
                country = st.text_input("Country", placeholder="TR")
                city = st.text_input("City", placeholder="Istanbul")
            with p5:
                device_type = st.selectbox("Device Type", ["mobile", "desktop", "tablet", "pos"])
                is_emulator = st.toggle("Is Emulator?", value=False)

            st.markdown("**Feature Vector (V1–V28)**")
            v_input = st.text_area(
                "28 PCA features (comma-separated or leave blank for random)",
                placeholder="-1.35,1.19,0.26,0.16,0.17,0.45,-0.53,0.79,0.27,-0.34,...",
                height=80
            )

            col_rand, col_sub = st.columns([1, 2])
            with col_rand:
                randomize = st.button("🎲 Random Features", use_container_width=True)
            with col_sub:
                submitted = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    # Generate features
    v_features = None
    if v_input.strip():
        try:
            parsed = [float(x.strip()) for x in v_input.split(",") if x.strip()]
            if len(parsed) == 28:
                v_features = parsed
            else:
                st.warning(f"28 features are required; {len(parsed)} was provided. They will be used randomly.")
        except ValueError:
            st.warning("Invalid format. Will be used randomly.")

    if v_features is None:
        np.random.seed(int(amount) % 100)
        v_features = list(np.random.randn(28).round(4))

    if randomize:
        v_features = list(np.random.randn(28).round(4))

    payload = {
        "amount": amount,
        "time": tx_time,
        "currency": currency,
        "v_features": v_features,
        "payment_method": payment,
        "user_id": user_id if user_id else None,
        "merchant_id": merchant_id if merchant_id else None,
        "location": {"country": country, "city": city} if country else None,
        "device": {"device_type": device_type, "is_emulator": is_emulator},
    }

    with col_result:
        st.markdown("**Analysis Result**")
        if submitted:
            if not st.session_state.api_online:
                st.error("API is offline. Please start the server.")
            else:
                with st.spinner("Analyzing..."):
                    result, err = send_predict(payload)

                if result:
                    risk = result["risk_score"]
                    is_fraud = result["is_fraud"]
                    status = result["status"]
                    lbl = risk_label(risk)
                    color = risk_color(risk)

                    # Result card
                    verdict_bg = "rgba(239,68,68,0.15)" if is_fraud else "rgba(34,197,94,0.1)"
                    verdict_border = "#ef4444" if is_fraud else "#22c55e"
                    verdict_text = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"
                    verdict_color = "#ef4444" if is_fraud else "#22c55e"

                    st.markdown(f"""
                    <div style="background:{verdict_bg};border:1px solid {verdict_border};border-radius:12px;padding:16px;margin-bottom:12px;">
                        <div style="font-size:1.2rem;font-weight:900;color:{verdict_color};">{verdict_text}</div>
                        <div style="font-size:2.5rem;font-weight:900;color:{color};font-family:JetBrains Mono;margin:8px 0;">{risk:.1%}</div>
                        <div style="color:#94a3b8;font-size:0.8rem;">Risk Score · {lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Status:** `{status.upper()}`")
                    st.markdown(f"**Confidence:** `{result['confidence']:.1%}`")
                    st.markdown(f"**Processing:** `{result['processing_time_ms']:.1f}ms`")
                    st.markdown(f"**TX ID:** `{result['transaction_id'][:20]}...`")

                    if result.get("rules_triggered"):
                        st.markdown("**Rules Triggered:**")
                        for rule in result["rules_triggered"]:
                            st.markdown(f"- 🔴 {rule}")

                    st.markdown("**Recommendations:**")
                    for rec in result.get("recommendations", []):
                        st.markdown(f"- 💡 {rec}")
                else:
                    st.error(f"API Error: {err}")
        else:
            st.markdown("""
            <div style="background:#0d1626;border:1px dashed #1e3a5f;border-radius:12px;padding:30px;text-align:center;color:#334155;">
                <div style="font-size:2rem;">🎯</div>
                <div style="margin-top:8px;font-size:0.85rem;">Submit a transaction to see the AI analysis result here</div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 4 — BATCH TEST
# ──────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("#### ⚡ Batch Transaction Simulator")
    st.caption("Submit multiple tasks at the same time and view the aggregated analysis results.")

    col_b1, col_b2 = st.columns([1, 2])

    with col_b1:
        n_batch = st.slider("Number of Transactions", 5, 100, 20)
        fraud_pct = st.slider("Simulated Fraud %", 0, 100, 15)
        amount_min = st.number_input("Min Amount", value=10.0)
        amount_max = st.number_input("Max Amount", value=5000.0)
        include_edge = st.toggle("Include Edge Cases (Amount 12345, Emulator)", value=True)

        if st.button("🚀 Run Batch Simulation", use_container_width=True, type="primary"):
            if not st.session_state.api_online:
                st.error("API offline")
            else:
                transactions = []
                for i in range(n_batch):
                    is_fraud_sim = random.random() < (fraud_pct / 100)
                    if is_fraud_sim:
                        amt = random.uniform(5000, 20000)
                        tx_t = random.choice([1000, 2000, 75000, 80000])  # Unusual hours
                    else:
                        amt = random.uniform(amount_min, amount_max)
                        tx_t = random.uniform(28800, 64800)  # Normal hours
                    if include_edge and i == 0:
                        amt = 12345  # Known fraud pattern

                    transactions.append({
                        "amount": round(amt, 2),
                        "time": round(tx_t, 0),
                        "v_features": list(np.random.randn(28).round(4)),
                        "device": {"is_emulator": include_edge and i == 1},
                    })

                batch_payload = {"transactions": transactions}
                with st.spinner(f"Processing {n_batch} transactions..."):
                    try:
                        r = requests.post(f"{API_BASE}/predict/batch", json=batch_payload, timeout=30)
                        if r.status_code == 200:
                            st.session_state["batch_result"] = r.json()
                        else:
                            st.error(f"Error: {r.text}")
                    except Exception as e:
                        st.error(str(e))

    with col_b2:
        if "batch_result" in st.session_state:
            br = st.session_state["batch_result"]
            results = br.get("results", [])

            # Summary metrics
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total", br.get("total", 0))
            bc2.metric("Fraud Found", br.get("fraud_count", 0), delta_color="inverse")
            bc3.metric("Avg Risk", f"{br.get('avg_risk_score', 0):.1%}")
            bc4.metric("Time", f"{br.get('processing_time_ms', 0):.0f}ms")

            # Results table
            rows_b = []
            for r_item in results:
                risk_b = r_item["risk_score"]
                rows_b.append({
                    "Amount": f"${float(r_item.get('amount', 0)) if 'amount' in r_item else '—'}",
                    "Risk": f"{risk_b:.1%}",
                    "Level": risk_label(risk_b),
                    "Fraud": "🔴 Yes" if r_item["is_fraud"] else "🟢 No",
                    "Status": r_item["status"].upper(),
                })
            df_batch = pd.DataFrame(rows_b)
            st.dataframe(df_batch, use_container_width=True, hide_index=True, height=380)

            # Mini chart
            risk_vals = [r_item["risk_score"] for r_item in results]
            colors_b = [risk_color(v) for v in risk_vals]
            fig_bar_b = go.Figure(go.Bar(
                x=list(range(len(risk_vals))), y=risk_vals,
                marker_color=colors_b,
                marker_line_width=0,
            ))
            fig_bar_b.add_hline(y=current_thresh, line_color="#f59e0b", line_dash="dot", line_width=1.5)
            fig_bar_b = apply_layout(fig_bar_b, height=180, title="Risk Scores per Transaction",
                                    yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig_bar_b, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#0d1626;border:1px dashed #1e3a5f;border-radius:12px;padding:40px;text-align:center;color:#334155;">
                <div style="font-size:2.5rem;">⚡</div>
                <div style="margin-top:8px;font-size:0.85rem;">Batch simulation results will appear here</div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 5 — MODEL INFO
# ──────────────────────────────────────────────────────────────
with tab_model:
    st.markdown("#### 🤖 Model & System Information")
    model_data = fetch_model_info()

    if model_data:
        mc1, mc2 = st.columns(2)
        with mc1:
            status_txt = "✅ Neural Network Loaded" if model_data.get("model_loaded") else "⚙️ Simulation Mode"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Model Status</div>
                <div class="stat-value" style="font-size:1.1rem;">{status_txt}</div>
                <div class="stat-sub">Version: {model_data.get('model_version','—')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Decision Threshold</div>
                <div class="stat-value">{model_data.get('threshold', 0.5):.2f}</div>
                <div class="stat-sub">Transactions above this are flagged as fraud</div>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Active Rules</div>
                <div class="stat-value">{model_data.get('active_rules', 0)}</div>
                <div class="stat-sub">Business rules applied pre/post model</div>
            </div>
            """, unsafe_allow_html=True)

        rules = model_data.get("rules", [])
        if rules:
            st.markdown("**Active Fraud Detection Rules:**")
            severity_colors = {"critical": "#ef4444", "high": "#f97316", "medium": "#f59e0b", "low": "#22c55e"}
            for rule in rules:
                sev = rule.get("severity", "low")
                col = severity_colors.get(sev, "#64748b")
                st.markdown(f"""
                <div class="tx-row">
                    <div>
                        <span style="color:#64748b;font-family:JetBrains Mono;font-size:0.75rem;">{rule.get('id','')}</span>
                        <span style="color:#e2e8f0;font-weight:600;margin-left:12px;">{rule.get('name','')}</span>
                    </div>
                    <span class="risk-pill risk-{sev}">{sev.upper()}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Model information could not be loaded. The API may be offline.")

    # Threshold visualizer
    st.markdown("#### Threshold Impact Visualizer")
    st.caption("See how the fraud detection rate changes as the threshold value changes.")

    thresh_range = np.linspace(0.1, 0.9, 80)
    # Simulated fraud rates at different thresholds
    tpr = 1 / (1 + np.exp(10 * (thresh_range - 0.5)))  # True positive rate sim
    fpr = 1 / (1 + np.exp(15 * (thresh_range - 0.3)))  # False positive rate sim

    fig_thresh = go.Figure()
    fig_thresh.add_trace(go.Scatter(
        x=thresh_range, y=tpr, name="Detection Rate (TPR)",
        line=dict(color="#22c55e", width=2), mode="lines"
    ))
    fig_thresh.add_trace(go.Scatter(
        x=thresh_range, y=fpr, name="False Alarm Rate (FPR)",
        line=dict(color="#ef4444", width=2), mode="lines"
    ))
    fig_thresh.add_vline(x=current_thresh, line_color="#f59e0b",
                         line_width=2, line_dash="dash",
                         annotation_text=f"Current: {current_thresh}",
                         annotation_font_color="#f59e0b")
    fig_thresh = apply_layout(fig_thresh, height=280,
                             xaxis_title="Threshold", yaxis_title="Rate",
                             showlegend=True, legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig_thresh, use_container_width=True)
    st.caption("Note: This image was generated using simulation data. Actual curves may vary depending on model performance.")


# ==================== Footer ==================== 
st.markdown(
    """
    <div class="footer-container">
        <p class="footer-text">
            Developed by <span class="footer-credits">Cagla Eren & Zendi</span> © 2026
            <br>
            <span style="font-size: 0.75rem; color: var(--text-muted);">
                Fraud AI Detection System - Advanced Neural Network Architecture
            </span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================== AUTO REFRESH ====================
if st.session_state.auto_refresh and st.session_state.api_online:
    fetch_stats.clear()
    fetch_live_txns.clear()
    time.sleep(3)
    st.rerun()



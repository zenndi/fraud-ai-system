"""
Fraud Detection API v5.0 - Enterprise Edition
Kullanım: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import yaml
import redis
import json
import os
import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import time
import random

# Redis Bağlantısı
r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
print("--- REDIS TESTI YAPILIYOR ---")
r.set("deneme", "calisiyor")
print(f"REDIS'TEN GELEN CEVAP: {r.get('deneme')}")


# Logging — format düzeltmesi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== APP ====================
app = FastAPI(
    title="🛡️ Fraud Guard AI",
    description="Enterprise-grade Real-time Fraud Detection System",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENUMS ====================
class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TransactionStatus(str, Enum):
    APPROVED = "approved"
    FLAGGED = "flagged"
    REVIEW = "manual_review"
    BLOCKED = "blocked"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ==================== PYDANTIC MODELS ====================
class GeoLocation(BaseModel):
    country: Optional[str] = None
    city: Optional[str] = None
    ip_address: Optional[str] = None

class DeviceInfo(BaseModel):
    device_type: Optional[str] = None
    os: Optional[str] = None
    is_emulator: bool = False

class Transaction(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    amount: float = Field(..., gt=0, example=125.50)
    currency: str = "USD"
    time: float = Field(..., ge=0, le=86400, example=43200)
    v_features: List[float] = Field(..., min_length=28, max_length=28)
    user_id: Optional[str] = None
    merchant_id: Optional[str] = None
    location: Optional[GeoLocation] = None
    device: Optional[DeviceInfo] = None
    payment_method: Optional[str] = None

    @validator('v_features')
    def check_features(cls, v):
        if len(v) != 28:
            raise ValueError('v_features must have exactly 28 values')
        return v

class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: RiskLevel
    is_fraud: bool
    confidence: float
    threshold: float
    model_version: str
    timestamp: str
    status: TransactionStatus
    processing_time_ms: float
    rules_triggered: List[str]
    recommendations: List[str]

class BatchTransaction(BaseModel):
    transactions: List[Transaction]
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @validator('transactions')
    def check_batch_size(cls, v):
        if len(v) > 500:
            raise ValueError('Maximum 500 transactions per batch')
        return v

class BatchPredictionResponse(BaseModel):
    batch_id: str
    results: List[PredictionResponse]
    total: int
    fraud_count: int
    processing_time_ms: float
    avg_risk_score: float

class AnalyticsSummary(BaseModel):
    period: str
    total_transactions: int
    fraud_detected: int
    fraud_rate: float
    avg_risk_score: float
    total_amount: float
    blocked_amount: float

class SystemHealth(BaseModel):
    status: str
    version: str
    uptime: str
    model_loaded: bool
    ws_connections: int
    total_processed: int

# ==================== GLOBAL STORES ====================
live_transactions = deque(maxlen=10000)
fraud_alerts = deque(maxlen=1000)
transaction_stats = {
    "total_processed": 0,
    "total_fraud": 0,
    "total_blocked": 0,
    "amount_saved": 0.0,
    "total_amount": 0.0,
    "hourly_counts": defaultdict(int),
    "hourly_fraud": defaultdict(int),
}
model = None
threshold = 0.2
model_version = "v5.0.0"
start_time = datetime.now()
active_rules = []

# ==================== WEBSOCKET MANAGER ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, client_type: str = "dashboard"):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "client_type": client_type,
            "id": str(uuid.uuid4())[:8]
        }
        logger.info(f"🔌 WebSocket connected ({client_type}) — total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_info.pop(websocket, None)
            logger.info(f"🔌 WebSocket disconnected — total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        if not self.active_connections:
            return
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# ==================== RULES ENGINE ====================
def load_rules():
    global active_rules
    active_rules = [
        {"id": "R001", "name": "High Amount", "condition": lambda t: t.amount > 10000, "severity": AlertSeverity.HIGH},
        {"id": "R002", "name": "Test Fraud Amount", "condition": lambda t: t.amount == 12345, "severity": AlertSeverity.CRITICAL},
        {"id": "R003", "name": "Unusual Hour", "condition": lambda t: t.time < 3600 or t.time > 72000, "severity": AlertSeverity.MEDIUM},
        {"id": "R004", "name": "Emulator Device", "condition": lambda t: t.device and t.device.is_emulator, "severity": AlertSeverity.HIGH},
        {"id": "R005", "name": "Very Large Amount", "condition": lambda t: t.amount > 50000, "severity": AlertSeverity.CRITICAL},
    ]
    logger.info(f"Loaded {len(active_rules)} rules")

def check_rules(transaction: Transaction) -> List[Dict]:
    triggered = []
    for rule in active_rules:
        try:
            if rule["condition"](transaction):
                triggered.append(rule)
        except Exception:
            pass
    return triggered

# ==================== ML MODEL ====================
def load_model():
    global model, threshold
    try:
        import tensorflow as tf
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            threshold = config.get('evaluation', {}).get('threshold', 0.5)
        model_dir = "models/"
        if os.path.exists(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
            if files:
                latest = sorted(files)[-1]
                model = tf.keras.models.load_model(os.path.join(model_dir, latest))
                logger.info(f"✅ Model loaded from {latest}")
                return True
        logger.warning("⚠️ No model found, using simulation mode")
        return False
    except Exception as e:
        logger.error(f"Model load error: {e}")
        return False

load_model()
load_rules()

def simulate_risk_score(transaction: Transaction) -> float:
    score = 0.1
    if transaction.amount > 1000:
        score += 0.2
    if transaction.amount > 5000:
        score += 0.3
    if transaction.amount == 12345:
        score += 0.8
    if transaction.time < 3600 or transaction.time > 72000:
        score += 0.1
    score += random.uniform(0, 0.15)
    return min(score, 1.0)

def predict_risk(transaction: Transaction) -> float:
    global model
    if model is not None:
        try:
           # Eğitimdeki StandardScaler'ın taklidini yapıyoruz
            scaled_amount = (transaction.amount - 88.0) / 250.0  # Ortalama Amount 88 civarıdır
            scaled_time = (transaction.time - 94813.0) / 47488.0 # Ortalama Time civarı
            
            # 2. SIRALAMA: Time, V1-V28, Amount (Kaggle Standardı)
            ordered_features = [scaled_time] + transaction.v_features + [scaled_amount]
            
            features = np.array(ordered_features).reshape(1, -1)
            
            # 3. TAHMİN
            prediction = model.predict(features, verbose=0)
            risk_score = float(prediction[0][0])
            
            logger.info(f"📊 MODEL TAHMİNİ (SCALED): {risk_score}")
            return risk_score

        except Exception as err:
            # hatayı yakalamak için
            logger.error(f"❌ MODEL ÇALIŞMA HATASI: {err}")
            return simulate_risk_score(transaction)
            
    else:
        #logger.warning("⚠️ Model yüklü değil, simülasyon puanı dönüyor.")
        return simulate_risk_score(transaction)

# ==================== HELPER ====================
def get_risk_level(risk: float) -> RiskLevel:
    if risk >= 0.9:
        return RiskLevel.CRITICAL
    elif risk >= 0.7:
        return RiskLevel.HIGH
    elif risk >= 0.5:
        return RiskLevel.MEDIUM
    elif risk >= 0.3:
        return RiskLevel.LOW
    return RiskLevel.NONE

def get_status(risk: float, threshold_val: float) -> TransactionStatus:
    if risk >= 0.9:
        return TransactionStatus.BLOCKED
    elif risk >= 0.7:
        return TransactionStatus.REVIEW
    elif risk >= threshold_val:
        return TransactionStatus.FLAGGED
    return TransactionStatus.APPROVED

# ==================== API ENDPOINTS ====================
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="background:#0f172a;color:white;text-align:center;padding:50px;font-family:system-ui;">
    <h1 style="background:linear-gradient(90deg,#3b82f6,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:3rem;">🛡️ Fraud Guard AI</h1>
    <p style="color:#94a3b8;">Enterprise Real-time Fraud Detection System v5.0</p>
    <div style="margin-top:30px;">
      <a href="/docs" style="color:#3b82f6;margin:10px;text-decoration:none;padding:10px 20px;border:1px solid #3b82f6;border-radius:8px;">📚 API Docs</a>
      <a href="/health" style="color:#22c55e;margin:10px;text-decoration:none;padding:10px 20px;border:1px solid #22c55e;border-radius:8px;">💚 Health</a>
    </div>
    </body></html>
    """

@app.get("/health", response_model=SystemHealth)
async def health_check():
    uptime = datetime.now() - start_time
    return SystemHealth(
        status="healthy",
        version=model_version,
        uptime=str(uptime).split('.')[0],
        model_loaded=model is not None,
        ws_connections=len(manager.active_connections),
        total_processed=transaction_stats["total_processed"]
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    start = time.time()
    try:
        # Rules check
        triggered = check_rules(transaction)
        rule_names = [rule["name"] for rule in triggered]

        # ml prediction
        risk = predict_risk(transaction)

        # Boost by rules
        for rule in triggered:
            if rule["severity"] == AlertSeverity.CRITICAL:
                risk = min(risk + 0.3, 1.0)
            elif rule["severity"] == AlertSeverity.HIGH:
                risk = min(risk + 0.15, 1.0)
            elif rule["severity"] == AlertSeverity.MEDIUM:
                risk = min(risk + 0.07, 1.0)

        is_fraud = risk >= threshold
        confidence = min(abs(risk - 0.5) * 2 + 0.3, 1.0)
        risk_level = get_risk_level(risk)
        status = get_status(risk, threshold)

        # Recommendations
        recs = []
        if status == TransactionStatus.BLOCKED:
            recs.append("Block transaction immediately")
            recs.append("Notify security team")
            recs.append("Flag user account for review")
        elif status == TransactionStatus.REVIEW:
            recs.append("Send to manual review queue")
            recs.append("Request additional authentication")
        elif status == TransactionStatus.FLAGGED:
            recs.append("Apply enhanced monitoring")
        else:
            recs.append("Process normally")
        if rule_names:
            recs.append(f"Rules triggered: {', '.join(rule_names)}")

        timestamp = datetime.now().isoformat()
        processing_ms = (time.time() - start) * 1000

        # Store transaction
        hour_key = datetime.now().strftime("%H:00")
        record = {
            "transaction_id": transaction.transaction_id,
            "amount": transaction.amount,
            "risk_score": round(risk, 4),
            "risk_level": risk_level.value,
            "is_fraud": is_fraud,
            "status": status.value,
            "timestamp": timestamp,
            "rules": rule_names,
            "processing_ms": round(processing_ms, 2),
        }

        # REDIS eklendi
        try:
            pipe = r.pipeline()
            #1. kaydı json olarak listenin başına ekler
            pipe.lpush("fraud_stream", json.dumps(record))
            pipe.ltrim("fraud_stream", 0, 99)
            pipe.incr("stats:total_processed")
            pipe.incrbyfloat("stats:total_amount", transaction.amount)
            if is_fraud:
                pipe.incr("stats:total_fraud")
            pipe.execute()
        except Exception as e:
            logger.error(f"Redis hatası: {e}")

        live_transactions.append(record)
        transaction_stats["total_processed"] += 1
        transaction_stats["total_amount"] += transaction.amount
        transaction_stats["hourly_counts"][hour_key] += 1
        if is_fraud:
            transaction_stats["total_fraud"] += 1
            transaction_stats["hourly_fraud"][hour_key] += 1
            if status == TransactionStatus.BLOCKED:
                transaction_stats["total_blocked"] += 1
                transaction_stats["amount_saved"] += transaction.amount

        # WebSocket broadcast for fraud events
        if is_fraud:
            alert = {
                "event": "fraud_alert",
                "transaction_id": transaction.transaction_id,
                "risk_score": round(risk, 4),
                "risk_level": risk_level.value,
                "amount": transaction.amount,
                "status": status.value,
                "rules": rule_names,
                "timestamp": timestamp
            }
            asyncio.create_task(manager.broadcast(alert))
        else:
            # Also broadcast normal transactions for live feed
            live_event = {
                "event": "transaction",
                "transaction_id": transaction.transaction_id,
                "risk_score": round(risk, 4),
                "risk_level": risk_level.value,
                "amount": transaction.amount,
                "status": status.value,
                "timestamp": timestamp
            }
            asyncio.create_task(manager.broadcast(live_event))

        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            risk_score=round(risk, 4),
            risk_level=risk_level,
            is_fraud=is_fraud,
            confidence=round(confidence, 4),
            threshold=threshold,
            model_version=model_version,
            timestamp=timestamp,
            status=status,
            processing_time_ms=round(processing_ms, 2),
            rules_triggered=rule_names,
            recommendations=recs
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchTransaction):
    start = time.time()
    results = []
    fraud_count = 0
    total_risk = 0.0
    for txn in batch.transactions:
        risk = predict_risk(txn)
        is_fraud = risk >= threshold
        if is_fraud:
            fraud_count += 1
        total_risk += risk
        results.append(PredictionResponse(
            transaction_id=txn.transaction_id,
            risk_score=round(risk, 4),
            risk_level=get_risk_level(risk),
            is_fraud=is_fraud,
            confidence=min(abs(risk - 0.5) * 2 + 0.3, 1.0),
            threshold=threshold,
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            status=get_status(risk, threshold),
            processing_time_ms=0,
            rules_triggered=[],
            recommendations=[]
        ))
    processing_ms = (time.time() - start) * 1000
    transaction_stats["total_processed"] += len(results)
    transaction_stats["total_fraud"] += fraud_count
    return BatchPredictionResponse(
        batch_id=batch.batch_id,
        results=results,
        total=len(results),
        fraud_count=fraud_count,
        processing_time_ms=round(processing_ms, 2),
        avg_risk_score=round(total_risk / len(results), 4)
    )

@app.get("/api/v1/live")
async def get_live_transactions(limit: int = 100, fraud_only: bool = False):
    txns = list(live_transactions)
    if fraud_only:
        txns = [t for t in txns if t.get("is_fraud")]
    return {"transactions": txns[-limit:], "total": len(txns)}

@app.get("/api/v1/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics(period: str = "24h"):
    total = transaction_stats["total_processed"]
    fraud = transaction_stats["total_fraud"]
    return AnalyticsSummary(
        period=period,
        total_transactions=total,
        fraud_detected=fraud,
        fraud_rate=fraud / max(1, total),
        avg_risk_score=0.35,
        total_amount=transaction_stats["total_amount"],
        blocked_amount=transaction_stats["amount_saved"]
    )

@app.get("/api/v1/analytics/hourly")
async def get_hourly_analytics():
    """Saatlik işlem ve fraud verisi döndürür"""
    hourly = []
    for hour, count in sorted(transaction_stats["hourly_counts"].items()):
        hourly.append({
            "hour": hour,
            "total": count,
            "fraud": transaction_stats["hourly_fraud"].get(hour, 0)
        })
    return {"hourly": hourly}

@app.get("/api/v1/stats")
async def get_stats():
    """Dashboard için özet istatistikler"""
    total = transaction_stats["total_processed"]
    fraud = transaction_stats["total_fraud"]
    return {
        "total_processed": total,
        "total_fraud": fraud,
        "total_blocked": transaction_stats["total_blocked"],
        "amount_saved": round(transaction_stats["amount_saved"], 2),
        "total_amount": round(transaction_stats["total_amount"], 2),
        "fraud_rate": round(fraud / max(1, total) * 100, 2),
        "model_loaded": model is not None,
        "threshold": threshold,
        "uptime": str(datetime.now() - start_time).split('.')[0],
        "ws_connections": len(manager.active_connections),
    }

@app.get("/model/info")
async def model_info():
    rules_info = [
        {"id": r["id"], "name": r["name"], "severity": r["severity"].value}
        for r in active_rules
    ]
    return {
        "model_version": model_version,
        "model_loaded": model is not None,
        "threshold": threshold,
        "active_rules": len(active_rules),
        "rules": rules_info
    }

@app.post("/model/threshold")
async def update_threshold(new_threshold: float = Query(..., ge=0.01, le=0.99)):
    global threshold
    threshold = new_threshold
    try:
        with open("config.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault('evaluation', {})['threshold'] = threshold
        with open("config.yaml", 'w') as f:
            yaml.dump(cfg, f)
    except Exception:
        pass
    logger.info(f"Threshold updated to {threshold}")
    return {"message": f"Threshold updated to {threshold}", "threshold": threshold}

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket, "dashboard")
    # İlk bağlantıda mevcut istatistikleri gönder
    try:
        await websocket.send_json({
            "event": "connected",
            "stats": {
                "total_processed": transaction_stats["total_processed"],
                "total_fraud": transaction_stats["total_fraud"],
            }
        })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            elif data == "stats":
                stats = {
                    "event": "stats_update",
                    "total_processed": transaction_stats["total_processed"],
                    "total_fraud": transaction_stats["total_fraud"],
                    "fraud_rate": transaction_stats["total_fraud"] / max(1, transaction_stats["total_processed"]),
                }
                await websocket.send_json(stats)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup():
    logger.info("🚀 Fraud Guard AI API v5.0 started")
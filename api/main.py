# api/main.py
# Your production FastAPI application
# Run with: uvicorn api.main:app --reload --port 8000

import os, sys, time, pickle
sys.path.append(os.getcwd())

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.db import (
    SessionLocal, Prediction, ModelVersion,
    get_db, get_active_model_version
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Pydantic schemas ─────────────────────────────────────────
# These define the shape of requests and responses
# FastAPI validates every incoming request against these automatically
# If a request doesn't match — 422 error returned before your code runs

class PredictRequest(BaseModel):
    text: str = Field(
        ...,                          # ... means required
        min_length=10,                # reject empty or near-empty inputs
        max_length=10000,             # reject absurdly long inputs
        description="News article text to classify"
    )
    source: Optional[str] = Field(
        default="api",
        description="Where this request came from"
    )

class PredictResponse(BaseModel):
    prediction_id  : int
    predicted_label: str
    confidence     : float
    all_probs      : dict
    model_version  : str
    latency_ms     : float
    created_at     : datetime

class HealthResponse(BaseModel):
    status         : str
    model_name     : str
    model_version  : str
    test_f1_macro  : float
    uptime_seconds : float

# ─── App initialisation ───────────────────────────────────────
app = FastAPI(
    title       = "News Classifier API",
    description = "Classifies news articles into World / Sports / Business / Sci-Tech",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Global state ─────────────────────────────────────────────
# Model loaded ONCE at startup — not on every request
# This is the standard pattern for ML APIs
APP_STATE = {
    "pipeline"      : None,
    "model_version" : None,
    "label_map"     : {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    "start_time"    : None,
}
# ─── Custom exception handlers ────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    FastAPI's default validation error response is verbose and confusing.
    This returns a clean, actionable error message instead.
    """
    errors = exc.errors()
    messages = []
    for error in errors:
        field   = " → ".join(str(e) for e in error["loc"])
        message = error["msg"]
        messages.append(f"{field}: {message}")

    return JSONResponse(
        status_code=422,
        content={
            "error"  : "Validation failed",
            "details": messages,
            "hint"   : "Check your request body matches the expected schema at /docs"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
):
    """
    Catches any unhandled exception and returns a clean 500
    instead of exposing internal stack traces to the caller.
    Stack trace still logged server-side for debugging.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error"  : "Internal server error",
            "message": "Something went wrong. Check server logs.",
        }
    )
# ─── Startup event ────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    logger.info("Starting up — loading model...")
    APP_STATE["start_time"] = time.time()

    db = SessionLocal()
    try:
        active_model  = get_active_model_version(db)
        artifact_path = active_model.artifact_path

        logger.info(f"Loading model: {active_model.name} v{active_model.version}")

        with open(artifact_path, "rb") as f:
            APP_STATE["pipeline"] = pickle.load(f)

        APP_STATE["model_version"] = active_model

        # ── Warmup prediction ──────────────────────────────────
        # Runs one dummy prediction to pre-load everything into memory
        # Eliminates cold start latency on the first real request
        # Standard practice in every production ML API
        warmup_text = "Apple reported strong quarterly earnings beating estimates"
        APP_STATE["pipeline"].predict([warmup_text])
        APP_STATE["pipeline"].predict_proba([warmup_text])

        logger.info("Model warmed up — first request will not be slow.")
        logger.info(f"Test F1: {active_model.metrics.get('test_f1_macro')}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
    finally:
        db.close()

# ─── Health endpoint ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Returns API status and loaded model info.
    Load balancers ping this every 30 seconds.
    Monitoring systems alert if this returns non-200.
    """
    if APP_STATE["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = APP_STATE["model_version"]
    uptime = time.time() - APP_STATE["start_time"]

    return HealthResponse(
        status        = "healthy",
        model_name    = model.name,
        model_version = model.version,
        test_f1_macro = model.metrics.get("test_f1_macro", 0.0),
        uptime_seconds= round(uptime, 2)
    )
    
# ─── Predict endpoint ─────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts raw text, runs inference, stores result in PostgreSQL.
    Returns predicted label, confidence, and all class probabilities.

    Depends(get_db) — FastAPI automatically:
      1. Creates a DB session before this function runs
      2. Passes it as the db parameter
      3. Closes it after the response is sent
    You never manually open or close sessions in endpoints.
    """
    if APP_STATE["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health endpoint."
        )

    pipeline  = APP_STATE["pipeline"]
    label_map = APP_STATE["label_map"]
    model_ver = APP_STATE["model_version"]

    # ── Inference ──────────────────────────────────────────────
    start      = time.time()
    pred_idx   = pipeline.predict([request.text])[0]
    proba      = pipeline.predict_proba([request.text])[0]
    latency_ms = (time.time() - start) * 1000

    predicted_label = label_map[pred_idx]
    confidence      = float(proba.max())
    all_probs       = {
        label_map[i]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    # ── Store in PostgreSQL ────────────────────────────────────
    prediction = Prediction(
        input_text      = request.text,
        predicted_label = predicted_label,
        confidence      = confidence,
        all_probs       = all_probs,
        model_version   = model_ver.version,
        latency_ms      = round(latency_ms, 3),
        metadata        = {
            "source"    : request.source,
            "char_count": len(request.text),
            "word_count": len(request.text.split())
        },
        created_at      = datetime.utcnow()
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    logger.info(
        f"Predicted [{predicted_label}] "
        f"conf={confidence:.4f} "
        f"latency={latency_ms:.2f}ms "
        f"id={prediction.id}"
    )

    return PredictResponse(
        prediction_id   = prediction.id,
        predicted_label = predicted_label,
        confidence      = confidence,
        all_probs       = all_probs,
        model_version   = model_ver.version,
        latency_ms      = round(latency_ms, 3),
        created_at      = prediction.created_at
    )
    
# ─── Predictions history endpoint ─────────────────────────────
@app.get("/predictions")
async def get_predictions(
    label      : Optional[str]   = Query(default=None,  description="Filter by predicted label"),
    min_conf   : Optional[float] = Query(default=None,  description="Minimum confidence score"),
    max_conf   : Optional[float] = Query(default=None,  description="Maximum confidence score"),
    limit      : int             = Query(default=10,    ge=1, le=100),
    offset     : int             = Query(default=0,     ge=0),
    db         : Session         = Depends(get_db)
):
    """
    Returns prediction history with optional filtering.
    Useful for monitoring model behaviour over time.

    Examples:
      GET /predictions                          → last 10 predictions
      GET /predictions?label=Business           → Business predictions only
      GET /predictions?min_conf=0.9             → high confidence only
      GET /predictions?max_conf=0.7             → low confidence — review queue
      GET /predictions?label=Sci/Tech&min_conf=0.8  → combined filters
    """
    query = db.query(Prediction)

    # Apply filters — only if the parameter was provided
    if label:
        # Case-insensitive label filter
        query = query.filter(
            Prediction.predicted_label.ilike(f"%{label}%")
        )
    if min_conf is not None:
        query = query.filter(Prediction.confidence >= min_conf)
    if max_conf is not None:
        query = query.filter(Prediction.confidence <= max_conf)

    # Total count before pagination — useful for frontend pagination
    total = query.count()

    # Apply pagination
    predictions = query.order_by(
        Prediction.created_at.desc()
    ).offset(offset).limit(limit).all()

    return {
        "total"  : total,
        "offset" : offset,
        "limit"  : limit,
        "results": [
            {
                "id"             : p.id,
                "predicted_label": p.predicted_label,
                "confidence"     : round(p.confidence, 4),
                "all_probs"      : p.all_probs,
                "latency_ms"     : p.latency_ms,
                "word_count"     : p.metadata.get("word_count") if p.metadata else None,
                "source"         : p.metadata.get("source") if p.metadata else None,
                "created_at"     : p.created_at,
            }
            for p in predictions
        ]
    }


# ─── Stats endpoint ───────────────────────────────────────────
@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """
    Aggregate statistics across all predictions.
    This is your model monitoring dashboard endpoint.
    In production this powers a Grafana dashboard.
    """
    from sqlalchemy import func

    total = db.query(func.count(Prediction.id)).scalar()

    if total == 0:
        return {"message": "No predictions yet"}

    # Per-label breakdown
    label_stats = db.query(
        Prediction.predicted_label,
        func.count(Prediction.id).label("count"),
        func.avg(Prediction.confidence).label("avg_confidence"),
        func.min(Prediction.confidence).label("min_confidence"),
        func.max(Prediction.confidence).label("max_confidence"),
        func.avg(Prediction.latency_ms).label("avg_latency_ms")
    ).group_by(Prediction.predicted_label).all()

    # Low confidence predictions — human review queue
    low_conf_count = db.query(func.count(Prediction.id)).filter(
        Prediction.confidence < 0.70
    ).scalar()

    # High confidence predictions
    high_conf_count = db.query(func.count(Prediction.id)).filter(
        Prediction.confidence >= 0.90
    ).scalar()

    return {
        "total_predictions" : total,
        "low_conf_count"    : low_conf_count,
        "high_conf_count"   : high_conf_count,
        "low_conf_pct"      : round(low_conf_count / total * 100, 2),
        "per_label"         : [
            {
                "label"          : row.predicted_label,
                "count"          : row.count,
                "avg_confidence" : round(float(row.avg_confidence), 4),
                "min_confidence" : round(float(row.min_confidence), 4),
                "max_confidence" : round(float(row.max_confidence), 4),
                "avg_latency_ms" : round(float(row.avg_latency_ms), 3),
            }
            for row in label_stats
        ]
    }
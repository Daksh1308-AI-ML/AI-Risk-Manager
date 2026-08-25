"""FastAPI application for AI Risk Manager.

Exposes endpoints for risk scoring, fraud classification, drift detection,
and model evaluation.  Models are loaded on startup via the lifespan handler;
if artefacts are missing the API runs in degraded mode with heuristic
fallbacks.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    DriftStatusResponse,
    EvaluateRequest,
    EvaluateResponse,
    ExplanationItem,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger("ai_risk_manager")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models on startup, release on shutdown."""
    logger.info("Loading RiskManager pipeline ...")
    try:
        from models.pipeline import RiskManager

        app.state.risk_manager = RiskManager()
        logger.info("RiskManager loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            "Model artefacts not found: %s — API running in degraded mode.", exc
        )
        app.state.risk_manager = None

    logger.info("Initialising DriftDetector ...")
    from models.drift_detector import DriftDetector

    app.state.drift_detector = DriftDetector(warning_level=0.1, drift_level=0.2)
    app.state.last_retrained = datetime.now(timezone.utc)
    logger.info("DriftDetector ready.")

    yield

    logger.info("Shutting down — releasing resources.")
    app.state.risk_manager = None
    app.state.drift_detector = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Risk Manager",
    description="Chargeback Evidence Auto-Responder for Indian BFSI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request logging & response timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every incoming request and track response time."""
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()

    logger.info("[%s] %s %s — started", request_id, request.method, request.url.path)

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

    logger.info(
        "[%s] %s %s — %d (%.1fms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_risk_manager():
    rm = getattr(app.state, "risk_manager", None)
    if rm is None:
        raise HTTPException(
            status_code=503,
            detail="Model artefacts not loaded. Train models and restart the server.",
        )
    return rm


def _get_drift_detector():
    dd = getattr(app.state, "drift_detector", None)
    if dd is None:
        raise HTTPException(status_code=503, detail="DriftDetector not initialised.")
    return dd


# ---------------------------------------------------------------------------
# Heuristic fallbacks (degraded mode — no trained model)
# ---------------------------------------------------------------------------

def _heuristic_score(req: ScoreRequest) -> float:
    risk = 0.10
    if req.amount > 50000:
        risk += 0.25
    elif req.amount > 20000:
        risk += 0.15
    if req.is_new_device:
        risk += 0.20
    if req.is_new_address:
        risk += 0.15
    risk += min(req.past_disputes * 0.09, 0.30)
    if req.account_age_days < 30:
        risk += 0.10
    if req.merchant_category == "electronics":
        risk += 0.05
    return min(risk, 1.0)


def _heuristic_classify(req: ClassifyRequest) -> str:
    txn = req.transaction
    if txn.past_disputes >= 2 and req.login_after_purchase:
        return "friendly_fraud"
    if txn.is_new_device and txn.is_new_address:
        return "account_takeover"
    return "genuine"


_EVIDENCE_CHECKLISTS: dict[str, list[str]] = {
    "genuine": [
        "Delivery confirmation with signature",
        "IP address logs",
        "Device fingerprint match",
    ],
    "friendly_fraud": [
        "Proof of delivery (photo/POD)",
        "Customer login after delivery",
        "Account activity log",
        "Terms of service acceptance",
    ],
    "account_takeover": [
        "Password reset log",
        "Device change history",
        "Location mismatch proof",
        "Session logs",
    ],
    "technical_failure": [
        "System error logs",
        "Duplicate transaction proof",
        "Gateway confirmation",
    ],
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict:
    """Liveness probe."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/v1/score", response_model=ScoreResponse)
async def score_transaction(req: ScoreRequest) -> ScoreResponse:
    """Score a single transaction for fraud risk."""
    try:
        rm = _get_risk_manager()
    except HTTPException:
        # Degraded mode: heuristic fallback
        risk = _heuristic_score(req)
        action = "BLOCK" if risk >= 0.65 else ("REVIEW" if risk >= 0.35 else "ALLOW")
        return ScoreResponse(
            risk_score=round(risk, 4),
            recommended_action=action,
            estimated_cost_if_fraud=round(req.amount * 0.95, 2),
            threshold_used=0.35,
            explanation=[
                ExplanationItem(
                    feature="is_new_device",
                    contribution=0.23 if req.is_new_device else 0.0,
                ),
                ExplanationItem(
                    feature="past_disputes",
                    contribution=min(req.past_disputes * 0.09, 0.45),
                ),
            ],
        )

    txn = {
        "transaction_id": f"api-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": req.amount,
        "payment_method": req.payment_method,
        "merchant_category": req.merchant_category,
        "customer_id": "api-customer",
        "device_fingerprint": "api-fp",
        "ip_address": "0.0.0.0",
        "is_new_device": req.is_new_device,
        "is_new_address": req.is_new_address,
        "account_age_days": req.account_age_days,
        "past_disputes": req.past_disputes,
        "chargeback_label": False,
        "fraud_type": "genuine",
        "chargeback_reason": "not_received",
    }

    try:
        result = rm.score(txn)
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    return ScoreResponse(
        risk_score=result["risk_score"],
        recommended_action=result["recommended_action"],
        estimated_cost_if_fraud=result["estimated_cost_if_fraud"],
        threshold_used=result["threshold_used"],
        explanation=result["explanation"],
    )


@app.post("/api/v1/classify", response_model=ClassifyResponse)
async def classify_transaction(req: ClassifyRequest) -> ClassifyResponse:
    """Classify the fraud type for a flagged transaction."""
    try:
        rm = _get_risk_manager()
    except HTTPException:
        fraud_type = _heuristic_classify(req)
        return ClassifyResponse(
            fraud_type=fraud_type,
            confidence=0.85,
            evidence_checklist=_EVIDENCE_CHECKLISTS.get(fraud_type, []),
        )

    txn = req.transaction.model_dump()
    txn.update({
        "transaction_id": f"api-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": "api-customer",
        "device_fingerprint": "api-fp",
        "ip_address": "0.0.0.0",
        "chargeback_label": False,
        "fraud_type": "genuine",
        "chargeback_reason": "not_received",
    })

    try:
        result = rm.score(txn)
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {exc}") from exc

    if result["fraud_type"] is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Transaction scored below review threshold — "
                "no fraud classification available."
            ),
        )

    return ClassifyResponse(
        fraud_type=result["fraud_type"],
        confidence=result["confidence"],
        evidence_checklist=result["evidence_checklist"],
    )


@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
async def evaluate_model(req: EvaluateRequest) -> EvaluateResponse:
    """Evaluate the model on a test dataset."""
    csv_path = Path(req.test_data_path)
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Test data not found: {req.test_data_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    if "chargeback_label" not in df.columns:
        raise HTTPException(status_code=422, detail="Dataset missing 'chargeback_label' column")

    rm = _get_risk_manager()  # raises 503 if unavailable

    try:
        from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

        from models.cost_matrix import calculate_cost
        from models.feature_engine import FEATURES

        features_df = rm.feature_engine.transform(df)
        X = features_df[FEATURES].values
        y_true = df["chargeback_label"].astype(int).values
        amounts = df["amount"].values.astype(np.float64)

        y_proba = rm.stage1.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= rm.threshold).astype(int)

        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            auc = 0.0

        cost_result = calculate_cost(y_true, y_pred, amounts)
        no_model_pred = np.zeros(len(y_true), dtype=int)
        total_cost_none = calculate_cost(y_true, no_model_pred, amounts)
        savings = total_cost_none["total_cost"] - cost_result["total_cost"]
        savings_pct = (
            (savings / total_cost_none["total_cost"] * 100)
            if total_cost_none["total_cost"] > 0
            else 0.0
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return EvaluateResponse(
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_roc=auc,
        total_cost_savings=round(savings, 2),
        cost_savings_percentage=round(savings_pct, 2),
    )


@app.get("/api/v1/drift/status", response_model=DriftStatusResponse)
async def drift_status() -> DriftStatusResponse:
    """Get the current drift detection status."""
    try:
        dd = _get_drift_detector()
    except HTTPException:
        return DriftStatusResponse(
            adwin_status="stable",
            psi_value=0.0,
            page_hinkley_value=0.0,
            drift_detected=False,
            last_retrained=datetime.now(timezone.utc),
        )

    status = dd.get_status()
    adwin_status = (
        "drift"
        if status["adwin_drift"]
        else ("warning" if status["adwin_warning"] else "stable")
    )

    return DriftStatusResponse(
        adwin_status=adwin_status,
        psi_value=0.0,
        page_hinkley_value=round(status["page_hinkley_value"], 4),
        drift_detected=status["overall_drift_detected"],
        last_retrained=getattr(app.state, "last_retrained", datetime.now(timezone.utc)),
    )


@app.post("/api/v1/drift/simulate", response_model=DriftStatusResponse)
async def drift_simulate(
    mean_shift: float = 3.0,
    std_scale: float = 1.5,
    n_observations: int = 200,
) -> DriftStatusResponse:
    """Simulate concept drift by feeding shifted observations into the detector."""
    try:
        dd = _get_drift_detector()
    except HTTPException:
        return DriftStatusResponse(
            adwin_status="stable",
            psi_value=0.0,
            page_hinkley_value=0.0,
            drift_detected=False,
            last_retrained=datetime.now(timezone.utc),
        )

    dd.reset()
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, 100)
    for v in baseline:
        dd.update(float(v))

    drifted = rng.normal(mean_shift, std_scale, n_observations)
    for v in drifted:
        dd.update(float(v))

    psi_result = dd.detect_psi(baseline, drifted)
    status = dd.get_status()
    adwin_status = (
        "drift"
        if status["adwin_drift"]
        else ("warning" if status["adwin_warning"] else "stable")
    )

    return DriftStatusResponse(
        adwin_status=adwin_status,
        psi_value=round(psi_result["psi_value"], 4),
        page_hinkley_value=round(status["page_hinkley_value"], 4),
        drift_detected=status["overall_drift_detected"],
        last_retrained=getattr(app.state, "last_retrained", datetime.now(timezone.utc)),
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(400)
async def bad_request_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=400, content={"detail": exc.detail})


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"detail": exc.detail})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

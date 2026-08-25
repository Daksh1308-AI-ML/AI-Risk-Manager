"""API integration tests for the Trishul FastAPI application.

Uses httpx.AsyncClient with ASGITransport to test every endpoint
against the real FastAPI app (with model fallbacks when no trained
artifacts are present).

Run:  pytest tests/test_api.py -v
"""

import sys
sys.path.insert(0, ".")

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app

transport = ASGITransport(app=app)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_score_request():
    """Valid score request payload."""
    return {
        "amount": 15999.0,
        "payment_method": "credit_card",
        "merchant_category": "electronics",
        "is_new_device": True,
        "is_new_address": False,
        "account_age_days": 12,
        "past_disputes": 2,
    }


@pytest.fixture
def sample_classify_request():
    """Valid classify request payload."""
    return {
        "transaction": {
            "amount": 15999.0,
            "payment_method": "credit_card",
            "merchant_category": "electronics",
            "is_new_device": True,
            "is_new_address": False,
            "account_age_days": 12,
            "past_disputes": 2,
        },
        "login_after_purchase": True,
        "support_contacted": False,
        "return_requested": False,
        "account_activity_level": "medium",
    }


# ── Health ───────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_status(self):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"


# ── Score ────────────────────────────────────────────────────────────────


class TestScoreEndpoint:
    """Tests for POST /api/v1/score."""

    @pytest.mark.asyncio
    async def test_score_returns_200(self, sample_score_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=sample_score_request)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_score_returns_risk_score(self, sample_score_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=sample_score_request)
        body = resp.json()
        assert "risk_score" in body
        assert 0.0 <= body["risk_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_returns_action(self, sample_score_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=sample_score_request)
        body = resp.json()
        assert body["recommended_action"] in ("ALLOW", "REVIEW", "BLOCK")

    @pytest.mark.asyncio
    async def test_score_returns_explanation(self, sample_score_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=sample_score_request)
        body = resp.json()
        assert isinstance(body["explanation"], list)
        assert len(body["explanation"]) > 0
        for item in body["explanation"]:
            assert "feature" in item
            assert "contribution" in item

    @pytest.mark.asyncio
    async def test_score_returns_threshold(self, sample_score_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=sample_score_request)
        body = resp.json()
        assert "threshold_used" in body
        assert 0.0 <= body["threshold_used"] <= 1.0

    @pytest.mark.asyncio
    async def test_score_returns_cost_estimate(self, sample_score_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=sample_score_request)
        body = resp.json()
        assert body["estimated_cost_if_fraud"] > 0

    @pytest.mark.asyncio
    async def test_score_invalid_amount(self):
        payload = {
            "amount": 50.0,  # below minimum of 100
            "payment_method": "credit_card",
            "merchant_category": "electronics",
            "is_new_device": True,
            "is_new_address": False,
            "account_age_days": 12,
            "past_disputes": 2,
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_invalid_payment_method(self):
        payload = {
            "amount": 15999.0,
            "payment_method": "bitcoin",
            "merchant_category": "electronics",
            "is_new_device": True,
            "is_new_address": False,
            "account_age_days": 12,
            "past_disputes": 2,
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_missing_required_field(self):
        payload = {"amount": 15999.0}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_empty_body(self):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_score_high_value_triggers_block(self):
        """A high-risk request should yield BLOCK or REVIEW."""
        payload = {
            "amount": 499999.0,
            "payment_method": "credit_card",
            "merchant_category": "electronics",
            "is_new_device": True,
            "is_new_address": True,
            "account_age_days": 1,
            "past_disputes": 10,
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=payload)
        body = resp.json()
        assert body["recommended_action"] in ("BLOCK", "REVIEW")

    @pytest.mark.asyncio
    async def test_score_low_value_allows(self):
        """A low-risk request should yield ALLOW or REVIEW."""
        payload = {
            "amount": 500.0,
            "payment_method": "upi",
            "merchant_category": "grocery",
            "is_new_device": False,
            "is_new_address": False,
            "account_age_days": 365,
            "past_disputes": 0,
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/score", json=payload)
        body = resp.json()
        assert body["recommended_action"] in ("ALLOW", "REVIEW")


# ── Classify ─────────────────────────────────────────────────────────────


class TestClassifyEndpoint:
    """Tests for POST /api/v1/classify."""

    @pytest.mark.asyncio
    async def test_classify_returns_200(self, sample_classify_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=sample_classify_request)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_classify_returns_fraud_type(self, sample_classify_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=sample_classify_request)
        body = resp.json()
        assert body["fraud_type"] in (
            "genuine", "friendly_fraud", "account_takeover", "technical_failure",
        )

    @pytest.mark.asyncio
    async def test_classify_returns_confidence(self, sample_classify_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=sample_classify_request)
        body = resp.json()
        assert 0.0 <= body["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_classify_returns_evidence(self, sample_classify_request):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=sample_classify_request)
        body = resp.json()
        assert isinstance(body["evidence_checklist"], list)

    @pytest.mark.asyncio
    async def test_classify_missing_transaction(self):
        payload = {
            "login_after_purchase": True,
            "support_contacted": False,
            "return_requested": False,
            "account_activity_level": "medium",
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_classify_new_device_new_address_detects_takeover(self):
        """is_new_device + is_new_address should classify as account_takeover."""
        payload = {
            "transaction": {
                "amount": 99999.0,
                "payment_method": "credit_card",
                "merchant_category": "electronics",
                "is_new_device": True,
                "is_new_address": True,
                "account_age_days": 5,
                "past_disputes": 0,
            },
            "login_after_purchase": False,
            "support_contacted": False,
            "return_requested": False,
            "account_activity_level": "low",
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=payload)
        body = resp.json()
        assert body["fraud_type"] == "account_takeover"

    @pytest.mark.asyncio
    async def test_classify_disputes_with_login_detects_friendly_fraud(self):
        """High disputes + login_after_purchase → friendly_fraud."""
        payload = {
            "transaction": {
                "amount": 25000.0,
                "payment_method": "credit_card",
                "merchant_category": "fashion",
                "is_new_device": False,
                "is_new_address": False,
                "account_age_days": 90,
                "past_disputes": 3,
            },
            "login_after_purchase": True,
            "support_contacted": True,
            "return_requested": True,
            "account_activity_level": "high",
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/classify", json=payload)
        body = resp.json()
        assert body["fraud_type"] == "friendly_fraud"


# ── Drift Status ─────────────────────────────────────────────────────────


class TestDriftEndpoint:
    """Tests for GET /api/v1/drift/status."""

    @pytest.mark.asyncio
    async def test_drift_status_returns_200(self):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/drift/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_drift_status_returns_fields(self):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/drift/status")
        body = resp.json()
        assert "adwin_status" in body
        assert body["adwin_status"] in ("stable", "warning", "drift")
        assert "psi_value" in body
        assert body["psi_value"] >= 0.0
        assert "page_hinkley_value" in body
        assert body["page_hinkley_value"] >= 0.0
        assert "drift_detected" in body
        assert isinstance(body["drift_detected"], bool)
        assert "last_retrained" in body


# ── Evaluate ─────────────────────────────────────────────────────────────


class TestEvaluateEndpoint:
    """Tests for POST /api/v1/evaluate."""

    @pytest.mark.asyncio
    async def test_evaluate_missing_file_returns_404(self):
        payload = {
            "test_data_path": "data/nonexistent.csv",
            "threshold_mode": "cost_optimized",
        }
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/evaluate", json=payload)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_evaluate_missing_path_field(self):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/evaluate", json={})
        assert resp.status_code == 422

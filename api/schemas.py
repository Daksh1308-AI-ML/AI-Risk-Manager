from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExplanationItem(BaseModel):
    """Feature contribution to risk score."""

    feature: str = Field(..., description="Name of the contributing feature")
    contribution: float = Field(
        ..., ge=0.0, le=1.0, description="Feature's contribution to the risk score"
    )


class ScoreRequest(BaseModel):
    """Transaction input for risk scoring."""

    amount: float = Field(
        ..., ge=100, le=500000, description="Transaction amount in INR"
    )
    payment_method: Literal["upi", "credit_card", "debit_card", "netbanking", "wallet"] = (
        Field(..., description="Payment method used")
    )
    merchant_category: Literal["electronics", "grocery", "fashion", "travel", "digital_services"] = (
        Field(..., description="Merchant category")
    )
    is_new_device: bool = Field(..., description="Whether the device is new")
    is_new_address: bool = Field(..., description="Whether the address is new")
    account_age_days: int = Field(
        ..., ge=0, description="Age of the account in days"
    )
    past_disputes: int = Field(
        ..., ge=0, description="Number of past disputes"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 15999.0,
                    "payment_method": "credit_card",
                    "merchant_category": "electronics",
                    "is_new_device": True,
                    "is_new_address": False,
                    "account_age_days": 12,
                    "past_disputes": 2,
                }
            ]
        }
    }


class ScoreResponse(BaseModel):
    """Risk score result with recommended action and explanation."""

    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Computed risk score"
    )
    recommended_action: Literal["ALLOW", "REVIEW", "BLOCK"] = Field(
        ..., description="Recommended action based on risk score"
    )
    estimated_cost_if_fraud: float = Field(
        ..., ge=0, description="Estimated loss if this transaction is fraud"
    )
    threshold_used: float = Field(
        ..., ge=0.0, le=1.0, description="Threshold applied for decision"
    )
    explanation: list[ExplanationItem] = Field(
        ..., description="Top contributing features to the risk score"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "risk_score": 0.82,
                    "recommended_action": "BLOCK",
                    "estimated_cost_if_fraud": 17450.0,
                    "threshold_used": 0.35,
                    "explanation": [
                        {"feature": "is_new_device", "contribution": 0.23},
                        {"feature": "past_disputes", "contribution": 0.18},
                    ],
                }
            ]
        }
    }


class TransactionInput(BaseModel):
    """Nested transaction data for classification."""

    amount: float = Field(..., ge=100, le=500000)
    payment_method: Literal["upi", "credit_card", "debit_card", "netbanking", "wallet"]
    merchant_category: Literal["electronics", "grocery", "fashion", "travel", "digital_services"]
    is_new_device: bool
    is_new_address: bool
    account_age_days: int = Field(..., ge=0)
    past_disputes: int = Field(..., ge=0)


class ClassifyRequest(BaseModel):
    """Transaction plus behavioral signals for fraud classification."""

    transaction: TransactionInput = Field(..., description="Transaction details")
    login_after_purchase: bool = Field(
        ..., description="Whether user logged in after the purchase"
    )
    support_contacted: bool = Field(
        ..., description="Whether customer contacted support"
    )
    return_requested: bool = Field(
        ..., description="Whether a return was requested"
    )
    account_activity_level: Literal["low", "medium", "high"] = Field(
        ..., description="Historical account activity level"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class ClassifyResponse(BaseModel):
    """Fraud classification result with evidence."""

    fraud_type: Literal["genuine", "friendly_fraud", "account_takeover", "technical_failure"] = Field(
        ..., description="Classified fraud type"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence"
    )
    evidence_checklist: list[str] = Field(
        ..., description="Evidence items supporting the classification"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fraud_type": "friendly_fraud",
                    "confidence": 0.87,
                    "evidence_checklist": [
                        "Delivery confirmation",
                        "Customer login history",
                        "Communication logs",
                    ],
                }
            ]
        }
    }


class DriftStatusResponse(BaseModel):
    """Current status of drift detection monitors."""

    adwin_status: str = Field(..., description="ADWIN detector status")
    psi_value: float = Field(..., ge=0.0, description="Population Stability Index value")
    page_hinkley_value: float = Field(
        ..., ge=0.0, description="Page-Hinkley test value"
    )
    drift_detected: bool = Field(..., description="Whether drift was detected")
    last_retrained: datetime = Field(..., description="ISO timestamp of last retraining")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "adwin_status": "stable",
                    "psi_value": 0.12,
                    "page_hinkley_value": 2.3,
                    "drift_detected": False,
                    "last_retrained": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }


class EvaluateRequest(BaseModel):
    """Parameters for model evaluation."""

    test_data_path: str = Field(..., description="Path to the test dataset CSV")
    threshold_mode: str = Field(
        default="cost_optimized",
        description="Threshold optimization mode",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "test_data_path": "data/test.csv",
                    "threshold_mode": "cost_optimized",
                }
            ]
        }
    }


class EvaluateResponse(BaseModel):
    """Model evaluation metrics."""

    precision: float = Field(
        ..., ge=0.0, le=1.0, description="Precision score"
    )
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    auc_roc: float = Field(
        ..., ge=0.0, le=1.0, description="Area under ROC curve"
    )
    total_cost_savings: float = Field(
        ..., ge=0, description="Total cost savings in INR"
    )
    cost_savings_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Cost savings as percentage"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "precision": 0.87,
                    "recall": 0.83,
                    "f1_score": 0.85,
                    "auc_roc": 0.91,
                    "total_cost_savings": 1250000.0,
                    "cost_savings_percentage": 78.5,
                }
            ]
        }
    }

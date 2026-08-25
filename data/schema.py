from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FraudLabel(str, Enum):
    """Fraud classification label assigned to a transaction."""

    GENUINE = "genuine"
    FRIENDLY_FRAUD = "friendly_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    TECHNICAL_FAILURE = "technical_failure"


class ChargebackReason(str, Enum):
    """Reason cited for initiating a chargeback dispute."""

    NOT_RECEIVED = "not_received"
    DEFECTIVE = "defective"
    UNAUTHORIZED = "unauthorized"
    DUPLICATE = "duplicate"


class PaymentMethod(str, Enum):
    """Payment channel used for the transaction."""

    UPI = "upi"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class MerchantCategory(str, Enum):
    """Merchant industry vertical."""

    ELECTRONICS = "electronics"
    GROCERY = "grocery"
    FASHION = "fashion"
    TRAVEL = "travel"
    DIGITAL_SERVICES = "digital_services"


class TransactionSchema(BaseModel):
    """Schema for a single payment transaction record.

    Used for data validation, feature engineering, and model training
    within the AI Risk Manager chargeback prediction pipeline.
    """

    transaction_id: str = Field(..., description="Unique UUID identifying the transaction")
    timestamp: datetime = Field(..., description="Transaction timestamp in ISO 8601 format")
    amount: float = Field(..., ge=100, le=500000, description="Transaction amount in INR, range 100-500000")
    payment_method: str = Field(..., description="Payment channel enum value")
    merchant_category: str = Field(..., description="Merchant vertical enum value")
    customer_id: str = Field(..., description="Anonymized customer identifier")
    device_fingerprint: str = Field(..., description="Hash of device attributes")
    ip_address: str = Field(..., description="IPv4 address string")
    is_new_device: bool = Field(..., description="Whether the device is new to this customer")
    is_new_address: bool = Field(..., description="Whether the shipping/billing address is new")
    account_age_days: int = Field(..., ge=0, description="Age of customer account in days")
    past_disputes: int = Field(..., ge=0, description="Historical count of disputes by customer")
    chargeback_label: bool = Field(..., description="Target variable: True if chargeback occurred")
    fraud_type: str = Field(..., description="FraudLabel enum value")
    chargeback_reason: str = Field(..., description="ChargebackReason enum value")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 100 or v > 500000:
            raise ValueError(f"Amount must be between 100 and 500000 INR, got {v}")
        return v

    @field_validator("account_age_days")
    @classmethod
    def validate_account_age(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Account age cannot be negative, got {v}")
        return v

    @field_validator("past_disputes")
    @classmethod
    def validate_past_disputes(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Past disputes cannot be negative, got {v}")
        return v

    @field_validator("fraud_type")
    @classmethod
    def validate_fraud_type(cls, v: str) -> str:
        valid_values = [e.value for e in FraudLabel]
        if v not in valid_values:
            raise ValueError(f"Invalid fraud_type: {v}. Must be one of {valid_values}")
        return v

    @field_validator("chargeback_reason")
    @classmethod
    def validate_chargeback_reason(cls, v: str) -> str:
        valid_values = [e.value for e in ChargebackReason]
        if v not in valid_values:
            raise ValueError(f"Invalid chargeback_reason: {v}. Must be one of {valid_values}")
        return v

    @field_validator("payment_method")
    @classmethod
    def validate_payment_method(cls, v: str) -> str:
        valid_values = [e.value for e in PaymentMethod]
        if v not in valid_values:
            raise ValueError(f"Invalid payment_method: {v}. Must be one of {valid_values}")
        return v

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant_category(cls, v: str) -> str:
        valid_values = [e.value for e in MerchantCategory]
        if v not in valid_values:
            raise ValueError(f"Invalid merchant_category: {v}. Must be one of {valid_values}")
        return v

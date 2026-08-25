"""Synthetic transaction data generator for chargeback prediction models.

Generates labeled Indian e-commerce payment transactions with realistic
fraud patterns for model training and evaluation.
"""

from __future__ import annotations

import hashlib
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from data.schema import (
    ChargebackReason,
    FraudLabel,
    MerchantCategory,
    PaymentMethod,
    TransactionSchema,
)

fake = Faker("en_IN")

# --- Configuration ---

RANDOM_SEED = 42

# Among all 10K transactions: 70% legitimate, 30% fraud
LEGITIMATE_RATIO = 0.70

# Among fraud cases only (sums to 1.0)
FRAUD_DISTRIBUTION: dict[str, float] = {
    FraudLabel.FRIENDLY_FRAUD.value: 0.60,
    FraudLabel.GENUINE.value: 0.25,
    FraudLabel.ACCOUNT_TAKEOVER.value: 0.10,
    FraudLabel.TECHNICAL_FAILURE.value: 0.05,
}

# Chargeback reason mapping per fraud type (string keys)
CHARGEBACK_REASON_MAP: dict[str, list[str]] = {
    FraudLabel.FRIENDLY_FRAUD.value: [
        ChargebackReason.NOT_RECEIVED.value,
        ChargebackReason.DEFECTIVE.value,
    ],
    FraudLabel.GENUINE.value: [ChargebackReason.UNAUTHORIZED.value],
    FraudLabel.ACCOUNT_TAKEOVER.value: [ChargebackReason.UNAUTHORIZED.value],
    FraudLabel.TECHNICAL_FAILURE.value: [ChargebackReason.DUPLICATE.value],
}

# Payment method weights (Indian market) — string keys
PAYMENT_METHODS: list[str] = [
    PaymentMethod.UPI.value,
    PaymentMethod.CREDIT_CARD.value,
    PaymentMethod.DEBIT_CARD.value,
    PaymentMethod.WALLET.value,
    PaymentMethod.NETBANKING.value,
]
PAYMENT_METHOD_WEIGHTS: list[float] = [0.40, 0.20, 0.15, 0.15, 0.10]

# Merchant category weights — string keys
MERCHANT_CATEGORIES: list[str] = [
    MerchantCategory.ELECTRONICS.value,
    MerchantCategory.GROCERY.value,
    MerchantCategory.FASHION.value,
    MerchantCategory.DIGITAL_SERVICES.value,
    MerchantCategory.TRAVEL.value,
]
MERCHANT_CATEGORY_WEIGHTS: list[float] = [0.25, 0.20, 0.20, 0.20, 0.15]

# Amount ranges per payment method (INR) — string keys
AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    PaymentMethod.UPI.value: (100, 100000),
    PaymentMethod.CREDIT_CARD.value: (1000, 500000),
    PaymentMethod.DEBIT_CARD.value: (100, 200000),
    PaymentMethod.WALLET.value: (100, 50000),
    PaymentMethod.NETBANKING.value: (500, 300000),
}


def _generate_device_fingerprint() -> str:
    """Generate a realistic device fingerprint hash."""
    raw = f"{fake.user_agent()}-{fake.mac_address()}-{random.randint(1000, 9999)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _generate_timestamp(rng: np.random.Generator) -> datetime:
    """Generate a timestamp within 2024, weighted toward peak hours."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31, 23, 59, 59)
    span_seconds = int((end - start).total_seconds())
    ts = start + timedelta(seconds=int(rng.integers(0, span_seconds)))

    # Bias toward peak hours: 10am-2pm, 7pm-11pm
    hour_weights = np.zeros(24)
    for h in range(24):
        if 10 <= h <= 14:
            hour_weights[h] = 3.0
        elif 19 <= h <= 23:
            hour_weights[h] = 4.0
        elif 0 <= h <= 6:
            hour_weights[h] = 0.3
        else:
            hour_weights[h] = 1.0
    hour_weights /= hour_weights.sum()
    chosen_hour = int(rng.choice(24, p=hour_weights))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return ts.replace(hour=chosen_hour, minute=minute, second=second)


def _generate_amount(
    rng: np.random.Generator,
    payment_method: str,
    fraud_type: str,
) -> float:
    """Generate transaction amount with realistic distribution per payment method."""
    low, high = AMOUNT_RANGES[payment_method]

    # Lognormal for realistic right-skewed distribution
    mean_log = np.log((low + high) / 3)
    std_log = 0.8
    amount = rng.lognormal(mean_log, std_log)
    amount = float(np.clip(amount, low, high))

    # Account takeover tends toward higher amounts
    if fraud_type == FraudLabel.ACCOUNT_TAKEOVER.value:
        amount = float(np.clip(amount * rng.uniform(1.5, 3.0), low, high))

    # Round to realistic values
    if amount < 1000:
        amount = round(amount, 0)
    elif amount < 10000:
        amount = round(amount / 10) * 10
    else:
        amount = round(amount / 100) * 100

    return amount


def _generate_fraud_type_labels(
    num_transactions: int,
    rng: np.random.Generator,
) -> list[tuple[bool, str, str]]:
    """Return (chargeback_label, fraud_type, chargeback_reason) for each row."""
    num_fraud = int(num_transactions * (1 - LEGITIMATE_RATIO))
    num_legit = num_transactions - num_fraud

    results: list[tuple[bool, str, str]] = []

    # Legitimate transactions
    all_reasons = list(CHARGEBACK_REASON_MAP.values())
    flat_reasons = [r for group in all_reasons for r in group]
    for _ in range(num_legit):
        reason = str(rng.choice(flat_reasons))
        results.append((False, FraudLabel.GENUINE.value, reason))

    # Fraudulent transactions with the specified distribution
    fraud_type_vals = list(FRAUD_DISTRIBUTION.keys())
    fraud_probs = np.array(list(FRAUD_DISTRIBUTION.values()))
    fraud_probs /= fraud_probs.sum()

    fraud_labels = rng.choice(fraud_type_vals, size=num_fraud, p=fraud_probs)
    for ft_val in fraud_labels:
        ft_str = str(ft_val)
        reason = str(rng.choice(CHARGEBACK_REASON_MAP[ft_str]))
        results.append((True, ft_str, reason))

    rng.shuffle(results)
    return results


def _generate_customer_profile(
    rng: np.random.Generator,
    fraud_type: str,
) -> tuple[bool, bool, int, int]:
    """Generate (is_new_device, is_new_address, account_age_days, past_disputes).

    Patterns differ by fraud type to create learnable signals.
    """
    if fraud_type == FraudLabel.ACCOUNT_TAKEOVER.value:
        # New device + new address + old account (compromised)
        return (True, True, int(rng.integers(180, 2000)), int(rng.integers(0, 3)))

    if fraud_type == FraudLabel.GENUINE.value:
        # Real fraud: new device, high amount, no history
        return (
            bool(rng.choice([True, True, True, False])),  # 75% new device
            bool(rng.choice([False, True, False, False])),  # 25% new address
            int(rng.integers(0, 500)),
            0,
        )

    if fraud_type == FraudLabel.FRIENDLY_FRAUD.value:
        # Established user, some history, occasional dispute
        return (
            False,
            False,
            int(rng.integers(90, 1800)),
            int(rng.poisson(1)),
        )

    # Technical failure: normal profile
    return (
        bool(rng.choice([False, False, True])),  # 33% new device
        False,
        int(rng.integers(30, 1500)),
        int(rng.poisson(0.5)),
    )


def generate_transactions(
    num_transactions: int = 10000,
    output_path: str = "data/transactions.csv",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic labeled transactions for chargeback prediction.

    Args:
        num_transactions: Number of transactions to generate.
        output_path: File path for the CSV export.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with validated transaction records.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    labels = _generate_fraud_type_labels(num_transactions, rng)

    records: list[dict] = []
    for i in range(num_transactions):
        chargeback_label, fraud_type, chargeback_reason = labels[i]

        payment_method = str(rng.choice(PAYMENT_METHODS, p=PAYMENT_METHOD_WEIGHTS))
        merchant_category = str(rng.choice(MERCHANT_CATEGORIES, p=MERCHANT_CATEGORY_WEIGHTS))

        amount = _generate_amount(rng, payment_method, fraud_type)
        timestamp = _generate_timestamp(rng)
        is_new_device, is_new_address, account_age, disputes = (
            _generate_customer_profile(rng, fraud_type)
        )

        records.append({
            "transaction_id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "amount": amount,
            "payment_method": payment_method,
            "merchant_category": merchant_category,
            "customer_id": f"CUST-{uuid.uuid4().hex[:8].upper()}",
            "device_fingerprint": _generate_device_fingerprint(),
            "ip_address": fake.ipv4(),
            "is_new_device": is_new_device,
            "is_new_address": is_new_address,
            "account_age_days": account_age,
            "past_disputes": disputes,
            "chargeback_label": chargeback_label,
            "fraud_type": fraud_type,
            "chargeback_reason": chargeback_reason,
        })

    df = pd.DataFrame(records)

    # Validate every row against the Pydantic schema
    print(f"Validating {len(df)} transactions against TransactionSchema...")
    for idx, row in df.iterrows():
        try:
            TransactionSchema(**row.to_dict())
        except Exception as e:
            raise ValueError(f"Validation failed at row {idx}: {e}") from e
    print("All rows passed validation.")

    # Export
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} transactions to {output_path}")

    # --- Summary statistics ---
    print("\n=== Summary Statistics ===")
    print(f"Total transactions: {len(df)}")
    print(f"Chargeback rate: {df['chargeback_label'].mean():.1%}")

    print("\nFraud type distribution:")
    for ft, count in df["fraud_type"].value_counts().items():
        print(f"  {ft}: {count} ({count / len(df):.1%})")

    print("\nAmount stats (INR):")
    print(f"  Mean:   {df['amount'].mean():,.0f}")
    print(f"  Median: {df['amount'].median():,.0f}")
    print(f"  Min:    {df['amount'].min():,.0f}")
    print(f"  Max:    {df['amount'].max():,.0f}")

    print("\nPayment method distribution:")
    for pm, count in df["payment_method"].value_counts().items():
        print(f"  {pm}: {count} ({count / len(df):.1%})")

    return df


if __name__ == "__main__":
    generate_transactions()

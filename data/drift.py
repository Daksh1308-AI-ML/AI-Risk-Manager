"""Concept drift simulation data generator for chargeback prediction models.

Generates 12 months of transaction data with realistic drift patterns:
- Months 1-3: Baseline (2% fraud, normal distribution)
- Months 4-6: Seasonal shift / Diwali (8% fraud, higher amounts, more electronics)
- Months 7-9: Adversarial shift (5% fraud, new patterns, device bypass)
- Months 10-12: Partial recovery (3% fraud, mixed patterns)
"""

from __future__ import annotations

import hashlib
import random
import uuid
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Drift scenario definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftScenario:
    """Configuration for a single drift scenario."""

    name: str
    fraud_rate: float
    amount_mean: float
    amount_std: float
    merchant_weights: dict[str, float] = field(default_factory=dict)
    payment_weights: dict[str, float] = field(default_factory=dict)
    fraud_distribution: dict[str, float] = field(default_factory=dict)
    account_takeover_ratio: float = 0.10
    adversarial_bypass: bool = False
    months: tuple[int, ...] = ()


SCENARIOS: dict[str, DriftScenario] = {
    "baseline": DriftScenario(
        name="baseline",
        fraud_rate=0.02,
        amount_mean=5000.0,
        amount_std=2000.0,
        merchant_weights={
            MerchantCategory.ELECTRONICS.value: 0.25,
            MerchantCategory.GROCERY.value: 0.20,
            MerchantCategory.FASHION.value: 0.20,
            MerchantCategory.DIGITAL_SERVICES.value: 0.20,
            MerchantCategory.TRAVEL.value: 0.15,
        },
        payment_weights={
            PaymentMethod.UPI.value: 0.40,
            PaymentMethod.CREDIT_CARD.value: 0.20,
            PaymentMethod.DEBIT_CARD.value: 0.15,
            PaymentMethod.WALLET.value: 0.15,
            PaymentMethod.NETBANKING.value: 0.10,
        },
        fraud_distribution={
            FraudLabel.FRIENDLY_FRAUD.value: 0.60,
            FraudLabel.GENUINE.value: 0.25,
            FraudLabel.ACCOUNT_TAKEOVER.value: 0.10,
            FraudLabel.TECHNICAL_FAILURE.value: 0.05,
        },
        months=(1, 2, 3),
    ),
    "seasonal": DriftScenario(
        name="seasonal",
        fraud_rate=0.08,
        amount_mean=15000.0,
        amount_std=5000.0,
        merchant_weights={
            MerchantCategory.ELECTRONICS.value: 0.50,
            MerchantCategory.GROCERY.value: 0.10,
            MerchantCategory.FASHION.value: 0.15,
            MerchantCategory.DIGITAL_SERVICES.value: 0.15,
            MerchantCategory.TRAVEL.value: 0.10,
        },
        payment_weights={
            PaymentMethod.UPI.value: 0.30,
            PaymentMethod.CREDIT_CARD.value: 0.35,
            PaymentMethod.DEBIT_CARD.value: 0.15,
            PaymentMethod.WALLET.value: 0.10,
            PaymentMethod.NETBANKING.value: 0.10,
        },
        fraud_distribution={
            FraudLabel.FRIENDLY_FRAUD.value: 0.60,
            FraudLabel.GENUINE.value: 0.25,
            FraudLabel.ACCOUNT_TAKEOVER.value: 0.10,
            FraudLabel.TECHNICAL_FAILURE.value: 0.05,
        },
        months=(4, 5, 6),
    ),
    "adversarial": DriftScenario(
        name="adversarial",
        fraud_rate=0.05,
        amount_mean=8000.0,
        amount_std=3500.0,
        merchant_weights={
            MerchantCategory.ELECTRONICS.value: 0.30,
            MerchantCategory.GROCERY.value: 0.20,
            MerchantCategory.FASHION.value: 0.20,
            MerchantCategory.DIGITAL_SERVICES.value: 0.15,
            MerchantCategory.TRAVEL.value: 0.15,
        },
        payment_weights={
            PaymentMethod.UPI.value: 0.35,
            PaymentMethod.CREDIT_CARD.value: 0.25,
            PaymentMethod.DEBIT_CARD.value: 0.15,
            PaymentMethod.WALLET.value: 0.15,
            PaymentMethod.NETBANKING.value: 0.10,
        },
        fraud_distribution={
            FraudLabel.FRIENDLY_FRAUD.value: 0.50,
            FraudLabel.GENUINE.value: 0.15,
            FraudLabel.ACCOUNT_TAKEOVER.value: 0.20,
            FraudLabel.TECHNICAL_FAILURE.value: 0.15,
        },
        account_takeover_ratio=0.20,
        adversarial_bypass=True,
        months=(7, 8, 9),
    ),
    "recovery": DriftScenario(
        name="recovery",
        fraud_rate=0.03,
        amount_mean=7000.0,
        amount_std=3000.0,
        merchant_weights={
            MerchantCategory.ELECTRONICS.value: 0.28,
            MerchantCategory.GROCERY.value: 0.20,
            MerchantCategory.FASHION.value: 0.20,
            MerchantCategory.DIGITAL_SERVICES.value: 0.18,
            MerchantCategory.TRAVEL.value: 0.14,
        },
        payment_weights={
            PaymentMethod.UPI.value: 0.38,
            PaymentMethod.CREDIT_CARD.value: 0.22,
            PaymentMethod.DEBIT_CARD.value: 0.15,
            PaymentMethod.WALLET.value: 0.13,
            PaymentMethod.NETBANKING.value: 0.12,
        },
        fraud_distribution={
            FraudLabel.FRIENDLY_FRAUD.value: 0.55,
            FraudLabel.GENUINE.value: 0.20,
            FraudLabel.ACCOUNT_TAKEOVER.value: 0.15,
            FraudLabel.TECHNICAL_FAILURE.value: 0.10,
        },
        months=(10, 11, 12),
    ),
}

CHARGEBACK_REASON_MAP: dict[str, list[str]] = {
    FraudLabel.FRIENDLY_FRAUD.value: [
        ChargebackReason.NOT_RECEIVED.value,
        ChargebackReason.DEFECTIVE.value,
    ],
    FraudLabel.GENUINE.value: [ChargebackReason.UNAUTHORIZED.value],
    FraudLabel.ACCOUNT_TAKEOVER.value: [ChargebackReason.UNAUTHORIZED.value],
    FraudLabel.TECHNICAL_FAILURE.value: [ChargebackReason.DUPLICATE.value],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _month_to_scenario(month: int) -> DriftScenario:
    """Map a month number (1-12) to its DriftScenario."""
    for scenario in SCENARIOS.values():
        if month in scenario.months:
            return scenario
    raise ValueError(f"Invalid month: {month}. Must be 1-12.")


def _generate_device_fingerprint(rng: np.random.Generator) -> str:
    """Generate a realistic device fingerprint hash."""
    raw = f"{fake.user_agent()}-{fake.mac_address()}-{rng.integers(1000, 9999)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _generate_adversarial_fingerprint(rng: np.random.Generator) -> str:
    """Generate a device fingerprint that mimics trusted device patterns.

    During adversarial shifts, attackers craft fingerprints that look like
    established devices (similar prefix patterns) to bypass trust scoring.
    """
    base = _generate_device_fingerprint(rng)
    prefix = rng.choice(["a1b2", "c3d4", "e5f6", "dead", "beef"])
    return f"{prefix}{base[4:]}"


def _generate_timestamp(
    rng: np.random.Generator,
    year: int,
    month: int,
) -> datetime:
    """Generate a timestamp within a given month, weighted toward peak hours."""
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(seconds=1)

    span_seconds = int((end - start).total_seconds())
    ts = start + timedelta(seconds=int(rng.integers(0, span_seconds)))

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
    scenario: DriftScenario,
    fraud_type: str,
) -> float:
    """Generate transaction amount using the scenario's distribution.

    Uses a lognormal base shaped by the scenario's mean/std, then applies
    fraud-type multipliers (account takeover skews higher).
    """
    amount = float(rng.normal(scenario.amount_mean, scenario.amount_std))

    if fraud_type == FraudLabel.ACCOUNT_TAKEOVER.value:
        amount *= rng.uniform(1.5, 3.0)
    elif fraud_type == FraudLabel.TECHNICAL_FAILURE.value:
        amount *= rng.uniform(0.5, 1.2)

    amount = float(np.clip(amount, 100, 500000))

    if amount < 1000:
        amount = round(amount, 0)
    elif amount < 10000:
        amount = round(amount / 10) * 10
    else:
        amount = round(amount / 100) * 100

    return amount


def _generate_fraud_labels(
    num_transactions: int,
    scenario: DriftScenario,
    rng: np.random.Generator,
) -> list[tuple[bool, str, str]]:
    """Return (chargeback_label, fraud_type, chargeback_reason) for each row."""
    num_fraud = int(num_transactions * scenario.fraud_rate)
    num_legit = num_transactions - num_fraud

    results: list[tuple[bool, str, str]] = []

    flat_reasons = [r for group in CHARGEBACK_REASON_MAP.values() for r in group]
    for _ in range(num_legit):
        reason = str(rng.choice(flat_reasons))
        results.append((False, FraudLabel.GENUINE.value, reason))

    fraud_type_vals = list(scenario.fraud_distribution.keys())
    fraud_probs = np.array(list(scenario.fraud_distribution.values()))
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
    scenario: DriftScenario,
) -> tuple[bool, bool, int, int]:
    """Generate (is_new_device, is_new_address, account_age_days, past_disputes).

    During adversarial shifts, account takeover profiles look more legitimate
    (lower new-device signal) to mimic trusted users.
    """
    if fraud_type == FraudLabel.ACCOUNT_TAKEOVER.value:
        if scenario.adversarial_bypass:
            return (
                bool(rng.choice([True, False])),  # 50% new device (lower signal)
                bool(rng.choice([True, False])),
                int(rng.integers(180, 2000)),
                int(rng.integers(0, 2)),
            )
        return (
            True,
            True,
            int(rng.integers(180, 2000)),
            int(rng.integers(0, 3)),
        )

    if fraud_type == FraudLabel.GENUINE.value:
        return (
            bool(rng.choice([True, True, True, False])),
            bool(rng.choice([False, True, False, False])),
            int(rng.integers(0, 500)),
            0,
        )

    if fraud_type == FraudLabel.FRIENDLY_FRAUD.value:
        return (
            False,
            False,
            int(rng.integers(90, 1800)),
            int(rng.poisson(1)),
        )

    return (
        bool(rng.choice([False, False, True])),
        False,
        int(rng.integers(30, 1500)),
        int(rng.poisson(0.5)),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DriftSimulator:
    """Generate time-series transaction data with concept drift."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize the simulator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_monthly_data(
        self,
        month: int,
        num_transactions: int = 1000,
    ) -> pd.DataFrame:
        """Generate transactions for a specific month.

        Args:
            month: Month number (1-12).
            num_transactions: Number of transactions to generate.

        Returns:
            DataFrame with transactions + month and drift_scenario columns.
        """
        if not 1 <= month <= 12:
            raise ValueError(f"month must be 1-12, got {month}")

        scenario = _month_to_scenario(month)
        labels = _generate_fraud_labels(num_transactions, scenario, self.rng)

        records: list[dict] = []
        for i in range(num_transactions):
            chargeback_label, fraud_type, chargeback_reason = labels[i]

            payment_method = str(
                self.rng.choice(
                    list(scenario.payment_weights.keys()),
                    p=list(scenario.payment_weights.values()),
                )
            )
            merchant_category = str(
                self.rng.choice(
                    list(scenario.merchant_weights.keys()),
                    p=list(scenario.merchant_weights.values()),
                )
            )

            amount = _generate_amount(self.rng, scenario, fraud_type)
            timestamp = _generate_timestamp(self.rng, 2024, month)

            is_new_device, is_new_address, account_age, disputes = (
                _generate_customer_profile(self.rng, fraud_type, scenario)
            )

            if scenario.adversarial_bypass and fraud_type in (
                FraudLabel.ACCOUNT_TAKEOVER.value,
                FraudLabel.TECHNICAL_FAILURE.value,
            ):
                device_fp = _generate_adversarial_fingerprint(self.rng)
            else:
                device_fp = _generate_device_fingerprint(self.rng)

            records.append({
                "transaction_id": str(uuid.uuid4()),
                "timestamp": timestamp.isoformat(),
                "amount": amount,
                "payment_method": payment_method,
                "merchant_category": merchant_category,
                "customer_id": f"CUST-{uuid.uuid4().hex[:8].upper()}",
                "device_fingerprint": device_fp,
                "ip_address": fake.ipv4(),
                "is_new_device": is_new_device,
                "is_new_address": is_new_address,
                "account_age_days": account_age,
                "past_disputes": disputes,
                "chargeback_label": chargeback_label,
                "fraud_type": fraud_type,
                "chargeback_reason": chargeback_reason,
                "month": month,
                "drift_scenario": scenario.name,
            })

        df = pd.DataFrame(records)

        for idx, row in df.iterrows():
            try:
                TransactionSchema(**{
                    k: v for k, v in row.to_dict().items()
                    if k not in ("month", "drift_scenario")
                })
            except Exception as e:
                raise ValueError(f"Validation failed at row {idx}: {e}") from e

        return df

    def generate_full_timeline(
        self,
        transactions_per_month: int = 1000,
    ) -> pd.DataFrame:
        """Generate 12 months of data with concept drift.

        Args:
            transactions_per_month: Transactions to generate for each month.

        Returns:
            DataFrame with all transactions + month and drift_scenario columns.
        """
        frames: list[pd.DataFrame] = []
        for month in range(1, 13):
            df = self.generate_monthly_data(month, transactions_per_month)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def get_drift_summary(df: pd.DataFrame) -> dict:
        """Return drift metrics per month.

        Args:
            df: DataFrame produced by generate_full_timeline.

        Returns:
            dict keyed by month number with fraud_rate, avg_amount,
            and category_distribution for each.
        """
        summary: dict = {}
        for month, group in df.groupby("month"):
            fraud_rate = float(group["chargeback_label"].mean())
            avg_amount = float(group["amount"].mean())
            cat_dist = group["merchant_category"].value_counts(normalize=True).to_dict()
            summary[int(month)] = {
                "fraud_rate": round(fraud_rate, 4),
                "avg_amount": round(avg_amount, 2),
                "category_distribution": cat_dist,
            }
        return summary


if __name__ == "__main__":
    sim = DriftSimulator(seed=42)

    print("Generating 12-month drift timeline (1000 txns/month)...")
    timeline = sim.generate_full_timeline(transactions_per_month=1000)

    output_path = "data/drift_transactions.csv"
    timeline.to_csv(output_path, index=False)
    print(f"\nSaved {len(timeline)} transactions to {output_path}")

    summary = sim.get_drift_summary(timeline)

    print("\n=== Drift Summary ===")
    print(f"{'Month':<8} {'Scenario':<14} {'Fraud %':<10} {'Avg Amount':<12}")
    print("-" * 44)
    for m in range(1, 13):
        info = summary[m]
        scenario_name = timeline[timeline["month"] == m]["drift_scenario"].iloc[0]
        print(
            f"{m:<8} {scenario_name:<14} "
            f"{info['fraud_rate']:<10.2%} "
            f"INR {info['avg_amount']:>10,.0f}"
        )

    print("\n=== Category Distribution per Scenario ===")
    for scenario_name in ["baseline", "seasonal", "adversarial", "recovery"]:
        scenario_df = timeline[timeline["drift_scenario"] == scenario_name]
        cats = scenario_df["merchant_category"].value_counts(normalize=True)
        print(f"\n  [{scenario_name}]")
        for cat, pct in cats.items():
            print(f"    {cat:<20} {pct:.1%}")

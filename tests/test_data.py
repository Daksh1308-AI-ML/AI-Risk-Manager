import pytest
import pandas as pd
import sys
sys.path.insert(0, ".")

from data.schema import (
    TransactionSchema, FraudLabel, ChargebackReason,
    PaymentMethod, MerchantCategory
)
from data.generate import generate_transactions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """Generate a small DataFrame for reuse across generator tests."""
    return generate_transactions(
        num_transactions=200,
        output_path="tests/test_transactions.csv",
        seed=99,
    )


@pytest.fixture(scope="module")
def small_df():
    """Minimal DataFrame for fast validation tests."""
    return generate_transactions(
        num_transactions=50,
        output_path="tests/test_small.csv",
        seed=123,
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchema:
    """Tests for TransactionSchema validation."""

    def _valid_row(self, **overrides):
        base = {
            "transaction_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "timestamp": "2024-06-15T10:30:00",
            "amount": 5000.0,
            "payment_method": "upi",
            "merchant_category": "electronics",
            "customer_id": "CUST-ABCDEF12",
            "device_fingerprint": "a" * 16,
            "ip_address": "192.168.1.1",
            "is_new_device": False,
            "is_new_address": False,
            "account_age_days": 365,
            "past_disputes": 0,
            "chargeback_label": False,
            "fraud_type": "genuine",
            "chargeback_reason": "not_received",
        }
        base.update(overrides)
        return base

    def test_valid_transaction(self):
        """A valid transaction should pass validation."""
        row = self._valid_row()
        txn = TransactionSchema(**row)
        assert txn.amount == 5000.0

    def test_invalid_amount_low(self):
        """Amount below 100 should fail."""
        with pytest.raises(Exception):
            TransactionSchema(**self._valid_row(amount=50.0))

    def test_invalid_amount_high(self):
        """Amount above 500000 should fail."""
        with pytest.raises(Exception):
            TransactionSchema(**self._valid_row(amount=600000.0))

    def test_invalid_fraud_type(self):
        """Invalid fraud_type should fail."""
        with pytest.raises(Exception):
            TransactionSchema(**self._valid_row(fraud_type="not_a_fraud"))

    def test_invalid_payment_method(self):
        """Invalid payment_method should fail."""
        with pytest.raises(Exception):
            TransactionSchema(**self._valid_row(payment_method="bitcoin"))

    def test_negative_account_age(self):
        """Negative account_age_days should fail."""
        with pytest.raises(Exception):
            TransactionSchema(**self._valid_row(account_age_days=-1))

    def test_negative_past_disputes(self):
        """Negative past_disputes should fail."""
        with pytest.raises(Exception):
            TransactionSchema(**self._valid_row(past_disputes=-5))


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------

class TestEnums:
    """Tests for enum values."""

    def test_fraud_label_values(self):
        """FraudLabel should have correct values."""
        expected = {"genuine", "friendly_fraud", "account_takeover", "technical_failure"}
        assert {e.value for e in FraudLabel} == expected

    def test_payment_method_values(self):
        """PaymentMethod should have correct values."""
        expected = {"upi", "credit_card", "debit_card", "netbanking", "wallet"}
        assert {e.value for e in PaymentMethod} == expected

    def test_merchant_category_values(self):
        """MerchantCategory should have correct values."""
        expected = {"electronics", "grocery", "fashion", "travel", "digital_services"}
        assert {e.value for e in MerchantCategory} == expected

    def test_chargeback_reason_values(self):
        """ChargebackReason should have correct values."""
        expected = {"not_received", "defective", "unauthorized", "duplicate"}
        assert {e.value for e in ChargebackReason} == expected


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    """Tests for generate_transactions()."""

    def test_generates_correct_row_count(self):
        """Should generate requested number of rows."""
        df = generate_transactions(
            num_transactions=150,
            output_path="tests/test_count.csv",
            seed=42,
        )
        assert len(df) == 150

    def test_has_all_columns(self, small_df):
        """Output should have all required columns."""
        expected = {
            "transaction_id", "timestamp", "amount", "payment_method",
            "merchant_category", "customer_id", "device_fingerprint",
            "ip_address", "is_new_device", "is_new_address",
            "account_age_days", "past_disputes", "chargeback_label",
            "fraud_type", "chargeback_reason",
        }
        assert set(small_df.columns) == expected

    def test_no_nulls(self, small_df):
        """Output should have no null values."""
        assert small_df.isnull().sum().sum() == 0

    def test_fraud_distribution(self, sample_df):
        """Fraud types should match expected distribution."""
        legit = (sample_df["chargeback_label"] == False).sum()
        fraud = (sample_df["chargeback_label"] == True).sum()
        total = len(sample_df)
        fraud_ratio = fraud / total
        assert 0.25 <= fraud_ratio <= 0.40, f"Fraud ratio {fraud_ratio:.2%} outside 25-40%"

    def test_amount_range(self, sample_df):
        """Amounts should be within valid range."""
        assert (sample_df["amount"] >= 100).all(), "Some amounts below 100"
        assert (sample_df["amount"] <= 500000).all(), "Some amounts above 500000"

    def test_payment_methods_valid(self, sample_df):
        """All payment methods should be valid enum values."""
        valid = {e.value for e in PaymentMethod}
        assert set(sample_df["payment_method"].unique()).issubset(valid)

    def test_merchant_categories_valid(self, sample_df):
        """All merchant categories should be valid enum values."""
        valid = {e.value for e in MerchantCategory}
        assert set(sample_df["merchant_category"].unique()).issubset(valid)

    def test_chargeback_labels_consistent(self, sample_df):
        """Non-genuine fraud types should always have chargeback_label=True."""
        non_genuine = sample_df[sample_df["fraud_type"] != FraudLabel.GENUINE.value]
        assert non_genuine["chargeback_label"].all()

    def test_fraud_rows_have_chargeback(self, sample_df):
        """Rows with chargeback_label=True should have a valid fraud_type."""
        chargeback_rows = sample_df[sample_df["chargeback_label"] == True]
        valid_fraud_types = {e.value for e in FraudLabel}
        assert set(chargeback_rows["fraud_type"].unique()).issubset(valid_fraud_types)

    def test_reproducibility(self):
        """Same seed should produce same results (excluding UUIDs)."""
        df1 = generate_transactions(
            num_transactions=100,
            output_path="tests/test_repro1.csv",
            seed=555,
        )
        df2 = generate_transactions(
            num_transactions=100,
            output_path="tests/test_repro2.csv",
            seed=555,
        )
        cols_to_check = [
            "amount", "payment_method", "merchant_category",
            "chargeback_label", "fraud_type", "chargeback_reason",
            "is_new_device", "is_new_address",
            "account_age_days", "past_disputes",
        ]
        pd.testing.assert_frame_equal(df1[cols_to_check], df2[cols_to_check])

    def test_schema_validation_all_rows(self, sample_df):
        """Every row should pass TransactionSchema validation."""
        for idx, row in sample_df.iterrows():
            try:
                TransactionSchema(**row.to_dict())
            except Exception as e:
                pytest.fail(f"Row {idx} failed validation: {e}")

"""Unified inference pipeline for AI Risk Manager.

Combines Stage 1 (XGBoost risk scoring) and Stage 2 (Random Forest fraud
type classification) into a single ``RiskManager`` class that accepts raw
transaction dicts and returns actionable risk assessments with explanations.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.cost_matrix import COST_MATRIX, calculate_cost
from models.feature_engine import FEATURES, FeatureEngine
from models.stage1_risk_scorer import Stage1RiskScorer
from models.stage2_fraud_classifier import Stage2FraudClassifier

# Action thresholds (risk_score → action)
_ALLOW_MAX = 0.3
_BLOCK_MIN = 0.7


class RiskManager:
    """Unified inference pipeline combining Stage 1 and Stage 2."""

    def __init__(
        self,
        stage1_path: str = "models/artifacts/stage1_model.pkl",
        stage2_path: str = "models/artifacts/stage2_model.pkl",
    ) -> None:
        """Load both models from artifacts.

        Parameters
        ----------
        stage1_path : str
            Path to the pickled Stage 1 artifact.
        stage2_path : str
            Path to the pickled Stage 2 artifact.

        Raises
        ------
        FileNotFoundError
            If either artifact file does not exist.
        """
        p1 = Path(stage1_path)
        p2 = Path(stage2_path)

        if not p1.exists():
            raise FileNotFoundError(
                f"Stage 1 artifact not found at {p1}. "
                "Run models/stage1_risk_scorer.py first."
            )
        if not p2.exists():
            raise FileNotFoundError(
                f"Stage 2 artifact not found at {p2}. "
                "Run models/stage2_fraud_classifier.py first."
            )

        self.stage1 = Stage1RiskScorer.load(p1)
        self.stage2 = Stage2FraudClassifier.load(p2)

        # The FeatureEngine is embedded in the Stage 1 artifact.
        # Fall back to Stage 2's copy if Stage 1's is missing.
        self.feature_engine: FeatureEngine = (
            self.stage1.feature_engine or self.stage2.feature_engine
        )

        self.threshold: float = self.stage1.threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, transaction: dict) -> dict:
        """Score a single transaction through the full pipeline.

        Stage 1: Risk scoring → probability + action
        Stage 2: If flagged → fraud type classification + evidence

        Parameters
        ----------
        transaction : dict
            Raw transaction dict matching ``TransactionSchema`` fields.

        Returns
        -------
        dict
            Unified result with risk_score, recommended_action,
            estimated_cost_if_fraud, threshold_used, explanation,
            and (if flagged) fraud_type, confidence, evidence_checklist.
        """
        # 1. Convert dict → single-row DataFrame
        txn_df = pd.DataFrame([transaction])

        # 2. Feature engineering
        features_df = self.feature_engine.transform(txn_df)

        # 3. Stage 1: risk probability
        risk_score = float(self.stage1.predict_proba(features_df)[0])
        recommended_action = self.get_action(risk_score, self.threshold)

        # 4. Estimated cost if this transaction turns out to be fraud
        amount = float(transaction.get("amount", 0))
        estimated_cost_if_fraud = self._estimate_cost_if_fraud(amount)

        # 5. Stage 2: fraud type classification (only if flagged)
        fraud_type: str | None = None
        confidence: float | None = None
        evidence_checklist: list[str] | None = None

        if recommended_action != "ALLOW":
            stage2_result = self.stage2.predict(features_df)
            fraud_type = stage2_result["fraud_types"][0]
            confidence = stage2_result["confidences"][0]
            evidence_checklist = stage2_result["evidence_checklists"][0]

        # 6. Feature contribution explanation
        explanation = self.explain(features_df, top_k=5)

        return {
            "risk_score": round(risk_score, 4),
            "recommended_action": recommended_action,
            "estimated_cost_if_fraud": round(estimated_cost_if_fraud, 2),
            "threshold_used": self.threshold,
            "explanation": explanation,
            "fraud_type": fraud_type,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "evidence_checklist": evidence_checklist,
        }

    def score_batch(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of transactions.

        Parameters
        ----------
        transactions : pd.DataFrame
            DataFrame with transaction columns matching ``TransactionSchema``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame augmented with scoring result columns:
            risk_score, recommended_action, estimated_cost_if_fraud,
            fraud_type, confidence.
        """
        result = transactions.copy()

        # Batch feature engineering
        features_df = self.feature_engine.transform(transactions)

        # Stage 1 batch predictions
        risk_scores = self.stage1.predict_proba(features_df)
        result["risk_score"] = np.round(risk_scores, 4)
        result["recommended_action"] = [
            self.get_action(s, self.threshold) for s in risk_scores
        ]

        # Estimated cost if fraud
        amounts = transactions["amount"].values.astype(np.float64)
        result["estimated_cost_if_fraud"] = np.round(
            amounts * (1.0 + 0.05 + 0.02 + 0.05 * 2.0 + 0.02 * 5.0), 2
        )

        # Stage 2 for flagged transactions
        flagged = result["recommended_action"] != "ALLOW"
        fraud_types = ["genuine"] * len(result)
        confidences = [0.0] * len(result)

        if flagged.any():
            fraud_mask = flagged.values
            stage2_result = self.stage2.predict(features_df, fraud_mask=fraud_mask)

            for i in range(len(result)):
                if fraud_mask[i]:
                    fraud_types[i] = stage2_result["fraud_types"][i]
                    confidences[i] = stage2_result["confidences"][i]

        result["fraud_type"] = fraud_types
        result["confidence"] = confidences

        return result

    def get_action(self, risk_score: float, threshold: float) -> str:
        """Map risk score to recommended action.

        Rules:
        - risk_score < 0.3 → "ALLOW"
        - 0.3 <= risk_score < 0.7 → "REVIEW"
        - risk_score >= 0.7 → "BLOCK"

        Parameters
        ----------
        risk_score : float
            Fraud probability from Stage 1 (0.0 – 1.0).
        threshold : float
            Model's cost-optimised threshold (unused here, kept for
            signature compatibility).

        Returns
        -------
        str
            "ALLOW", "REVIEW", or "BLOCK".
        """
        if risk_score < _ALLOW_MAX:
            return "ALLOW"
        elif risk_score >= _BLOCK_MIN:
            return "BLOCK"
        return "REVIEW"

    def explain(
        self, features_df: pd.DataFrame, top_k: int = 5
    ) -> list[dict]:
        """Generate feature contribution explanations.

        Uses XGBoost's built-in feature importance multiplied by the
        normalised feature values to estimate per-transaction contributions.

        Parameters
        ----------
        features_df : pd.DataFrame
            Single-row DataFrame with the 20 engineered feature columns.
        top_k : int
            Number of top features to return.

        Returns
        -------
        list[dict]
            Each dict has 'feature' (str) and 'contribution' (float) keys,
            sorted by descending absolute contribution.
        """
        if self.stage1.model is None:
            return []

        importances = self.stage1.model.feature_importances_
        feature_values = features_df[FEATURES].values[0]

        contributions = importances * np.abs(feature_values)

        # Sort by absolute contribution descending
        top_indices = np.argsort(contributions)[::-1][:top_k]

        return [
            {
                "feature": FEATURES[idx],
                "contribution": round(float(contributions[idx]), 4),
            }
            for idx in top_indices
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_cost_if_fraud(amount: float) -> float:
        """Estimate the total financial impact if a transaction is fraudulent.

        Uses the ``COST_MATRIX`` false-negative coefficients to compute
        the expected loss: chargeback + processing fee + operational cost +
        expected churn LTV loss + expected RBI penalty.

        Parameters
        ----------
        amount : float
            Transaction amount in INR.

        Returns
        -------
        float
            Estimated cost in INR.
        """
        fn = COST_MATRIX["false_negative"]
        cost = (
            amount * fn["chargeback_amount_multiplier"]
            + fn["processing_fee"]
            + fn["operational_cost"]
            + amount * fn["churn_probability"] * fn["churn_ltv_cost"]
            + amount * fn["rbi_penalty_probability"] * fn["rbi_penalty_amount"]
        )
        return float(cost)


if __name__ == "__main__":
    from data.schema import (
        ChargebackReason,
        FraudLabel,
        MerchantCategory,
        PaymentMethod,
    )

    print("=" * 60)
    print("AI Risk Manager — Unified Inference Pipeline Demo")
    print("=" * 60)

    # 1. Load pipeline
    print("\n[Step 1] Loading RiskManager pipeline ...")
    try:
        rm = RiskManager()
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        raise SystemExit(1)

    print(f"  Stage 1 threshold: {rm.threshold:.4f}")

    # 2. Single transaction scoring
    print("\n[Step 2] Scoring a single transaction ...")
    sample_txn = {
        "transaction_id": "txn-demo-001",
        "timestamp": "2025-07-15T14:30:00",
        "amount": 45000.0,
        "payment_method": PaymentMethod.UPI.value,
        "merchant_category": MerchantCategory.ELECTRONICS.value,
        "customer_id": "cust-9001",
        "device_fingerprint": "fp-abc-123",
        "ip_address": "103.21.58.44",
        "is_new_device": True,
        "is_new_address": True,
        "account_age_days": 12,
        "past_disputes": 3,
        "chargeback_label": False,
        "fraud_type": FraudLabel.GENUINE.value,
        "chargeback_reason": ChargebackReason.NOT_RECEIVED.value,
    }

    result = rm.score(sample_txn)
    print(f"  risk_score:              {result['risk_score']}")
    print(f"  recommended_action:      {result['recommended_action']}")
    print(f"  estimated_cost_if_fraud: INR {result['estimated_cost_if_fraud']:,.2f}")
    print(f"  threshold_used:          {result['threshold_used']}")
    print(f"  fraud_type:              {result['fraud_type']}")
    print(f"  confidence:              {result['confidence']}")
    print(f"  evidence_checklist:      {result['evidence_checklist']}")
    print(f"  explanation:")
    for exp in result["explanation"]:
        print(f"    - {exp['feature']:>25s}  contribution: {exp['contribution']}")

    # 3. Batch scoring
    print("\n[Step 3] Batch scoring from CSV ...")
    csv_path = Path("data/transactions.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        batch = df.head(10).copy()
        scored = rm.score_batch(batch)
        cols = [
            "transaction_id",
            "amount",
            "risk_score",
            "recommended_action",
            "fraud_type",
            "confidence",
        ]
        print(scored[cols].to_string(index=False))
    else:
        print(f"  [SKIP] {csv_path} not found — run data/generate.py first.")

    print("\n" + "=" * 60)
    print("Pipeline demo complete.")
    print("=" * 60)

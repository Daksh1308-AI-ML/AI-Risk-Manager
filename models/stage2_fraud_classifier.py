from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data.schema import FraudLabel
from models.feature_engine import FEATURES, FeatureEngine

RANDOM_FOREST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
}

EVIDENCE_CHECKLISTS: dict[str, list[str]] = {
    FraudLabel.GENUINE.value: [
        "Delivery confirmation with signature",
        "IP address logs",
        "Device fingerprint match",
        "Customer communication history",
        "Transaction velocity proof",
    ],
    FraudLabel.FRIENDLY_FRAUD.value: [
        "Proof of delivery (photo/POD)",
        "Customer login after delivery",
        "Account activity log",
        "Terms of service acceptance",
        "Return policy compliance",
    ],
    FraudLabel.ACCOUNT_TAKEOVER.value: [
        "Password reset log",
        "Device change history",
        "Location mismatch proof",
        "Session logs",
        "Recovery email verification",
    ],
    FraudLabel.TECHNICAL_FAILURE.value: [
        "System error logs",
        "Duplicate transaction proof",
        "Gateway confirmation",
        "Merchant notification",
        "Refund processing record",
    ],
}


class Stage2FraudClassifier:
    """Stage 2: Multi-class fraud type classifier using Random Forest."""

    def __init__(self) -> None:
        """Initialize the classifier."""
        self.model: RandomForestClassifier | None = None
        self.feature_engine: FeatureEngine | None = None
        self.class_names: list[str] = []
        self.feature_names: list[str] = []
        self.metrics: dict[str, Any] = {}

    def train(self, df: pd.DataFrame, feature_engine: FeatureEngine) -> dict:
        """Train Random Forest classifier on fraud type labels.

        Only trains on rows where chargeback_label=True (fraud cases).
        For legitimate transactions, returns 'genuine' with low confidence.

        Args:
            df: Raw transaction DataFrame
            feature_engine: Fitted FeatureEngine instance

        Returns:
            Training metrics dict (classification report, per-class metrics)
        """
        print("[Stage2] Transforming data with FeatureEngine...")
        features_df = feature_engine.transform(df)

        fraud_mask = df["chargeback_label"].values
        fraud_features = features_df.loc[fraud_mask, FEATURES]
        fraud_types = df.loc[fraud_mask, "fraud_type"].values

        self.class_names = sorted(df["fraud_type"].unique().tolist())
        self.feature_names = FEATURES

        print(f"[Stage2] Training on {len(fraud_features)} fraud cases "
              f"({len(self.class_names)} classes: {self.class_names})")

        self.model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        self.model.fit(fraud_features, fraud_types)

        train_predictions = self.model.predict(fraud_features)
        report = classification_report(
            fraud_types, train_predictions, output_dict=True
        )
        self.metrics = {
            "classification_report": report,
            "n_fraud_cases": int(fraud_mask.sum()),
            "n_total_cases": int(len(df)),
            "class_names": self.class_names,
        }

        print("[Stage2] Training complete.")
        print(classification_report(fraud_types, train_predictions))

        self.feature_engine = feature_engine
        return self.metrics

    def predict(
        self, features_df: pd.DataFrame, fraud_mask: np.ndarray | None = None
    ) -> dict:
        """Predict fraud type for flagged transactions.

        Args:
            features_df: DataFrame with 20 feature columns
            fraud_mask: Boolean mask indicating which rows are fraud (from Stage 1)

        Returns:
            dict with keys: fraud_types (list), confidences (list),
                           evidence_checklists (list of lists)
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        n_rows = len(features_df)
        fraud_types: list[str] = []
        confidences: list[float] = []
        evidence_checklists: list[list[str]] = []

        if fraud_mask is None:
            fraud_mask = np.ones(n_rows, dtype=bool)

        for i in range(n_rows):
            if not fraud_mask[i]:
                fraud_types.append(FraudLabel.GENUINE.value)
                confidences.append(0.0)
                evidence_checklists.append([])
            else:
                row = features_df.iloc[[i]][FEATURES]
                prediction = self.model.predict(row)[0]
                probability = float(np.max(self.model.predict_proba(row)))
                checklist = self.get_evidence_checklist(prediction)

                fraud_types.append(prediction)
                confidences.append(probability)
                evidence_checklists.append(checklist)

        return {
            "fraud_types": fraud_types,
            "confidences": confidences,
            "evidence_checklists": evidence_checklists,
        }

    def get_evidence_checklist(self, fraud_type: str) -> list[str]:
        """Return the evidence checklist for a given fraud type."""
        return EVIDENCE_CHECKLISTS.get(fraud_type, [])

    def save(self, path: str = "models/artifacts/stage2_model.pkl") -> None:
        """Save trained model to disk."""
        artifacts = {
            "model": self.model,
            "feature_engine": self.feature_engine,
            "class_names": self.class_names,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"[Stage2] Model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/artifacts/stage2_model.pkl") -> Stage2FraudClassifier:
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        instance = cls()
        instance.model = artifacts["model"]
        instance.feature_engine = artifacts["feature_engine"]
        instance.class_names = artifacts["class_names"]
        instance.metrics = artifacts["metrics"]
        instance.feature_names = artifacts["feature_names"]
        print(f"[Stage2] Model loaded from {path}")
        return instance


if __name__ == "__main__":
    from models.feature_engine import FeatureEngine

    print("=== Stage 2 Fraud Type Classifier ===\n")

    csv_path = Path("data/transactions.csv")
    if not csv_path.exists():
        print(f"Data not found at {csv_path}. Generating...")
        from data.generate import generate_transactions
        generate_transactions(n_transactions=5000)
        print("Data generated.\n")

    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} transactions\n")

    print("Fitting FeatureEngine...")
    feature_engine = FeatureEngine()
    feature_engine.fit(df)
    print("  FeatureEngine fitted\n")

    print("Training Stage2FraudClassifier...")
    classifier = Stage2FraudClassifier()
    metrics = classifier.train(df, feature_engine)

    print("\n--- Per-Class Metrics ---")
    for cls_name in classifier.class_names:
        cls_metrics = metrics["classification_report"].get(cls_name, {})
        print(f"  {cls_name}:")
        print(f"    precision: {cls_metrics.get('precision', 0):.3f}")
        print(f"    recall:    {cls_metrics.get('recall', 0):.3f}")
        print(f"    f1-score:  {cls_metrics.get('f1-score', 0):.3f}")

    print("\n--- Evidence Checklist Test ---")
    for fraud_type in classifier.class_names:
        checklist = classifier.get_evidence_checklist(fraud_type)
        print(f"\n  {fraud_type}:")
        for item in checklist:
            print(f"    - {item}")

    classifier.save()
    print("\nDone.")

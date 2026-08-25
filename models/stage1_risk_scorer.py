"""Stage 1: Binary fraud risk scorer using XGBoost.

Trains an XGBoost classifier on the 20 engineered features to produce
a fraud probability score for each transaction. Thresholds are optimised
using the project's cost matrix for minimal total financial impact.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from xgboost import XGBClassifier

from models.feature_engine import FEATURES, FeatureEngine
from models.cost_matrix import optimize_threshold

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "eval_metric": "aucpr",
    "random_state": 42,
}

THRESHOLDS = {
    "default": 0.5,
    "cost_optimized": 0.35,
    "f1_optimized": 0.65,
}

ARTIFACT_DIR = Path("models/artifacts")
DEFAULT_ARTIFACT_PATH = ARTIFACT_DIR / "stage1_model.pkl"


class Stage1RiskScorer:
    """Stage 1: Binary fraud risk scorer using XGBoost."""

    def __init__(self, threshold_mode: str = "cost_optimized") -> None:
        """Initialize with threshold mode.

        Parameters
        ----------
        threshold_mode : str
            One of "default", "cost_optimized", or "f1_optimized".
        """
        if threshold_mode not in THRESHOLDS:
            raise ValueError(
                f"Unknown threshold_mode '{threshold_mode}'. "
                f"Choose from {list(THRESHOLDS.keys())}"
            )
        self.threshold_mode = threshold_mode
        self.threshold: float = THRESHOLDS[threshold_mode]
        self.model: XGBClassifier | None = None
        self.feature_engine: FeatureEngine | None = None
        self.metrics: dict = {}
        self.feature_names: list[str] = list(FEATURES)

    def train(self, df: pd.DataFrame, feature_engine: FeatureEngine) -> dict:
        """Train XGBoost classifier on engineered features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame (from data/transactions.csv).
        feature_engine : FeatureEngine
            Fitted FeatureEngine instance.

        Returns
        -------
        dict
            Training metrics: precision, recall, f1, auc_roc, auc_pr.
        """
        print("[Stage1] Engineering features ...")
        self.feature_engine = feature_engine
        features_df = feature_engine.transform(df)

        X = features_df[FEATURES].values
        y = features_df["chargeback_label"].astype(int).values

        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        print(f"[Stage1] Class distribution — positive: {n_pos}, negative: {n_neg}")

        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"[Stage1] scale_pos_weight = {scale_pos_weight:.2f}")

        params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
        self.model = XGBClassifier(**params)

        print("[Stage1] Training XGBoost ...")
        self.model.fit(X, y)
        print("[Stage1] Training complete.")

        # --- Evaluate ---
        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        self.metrics = {
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, y_proba)),
            "auc_pr": float(average_precision_score(y, y_proba)),
        }

        # --- Optimise threshold ---
        amounts = df["amount"].values.astype(np.float64)
        self.threshold = optimize_threshold(
            y, y_proba, amounts, self.threshold_mode
        )
        self.metrics["threshold"] = self.threshold
        self.metrics["threshold_mode"] = self.threshold_mode

        print(f"[Stage1] Optimised threshold ({self.threshold_mode}): {self.threshold:.2f}")
        print(f"[Stage1] Metrics: {self.metrics}")

        return self.metrics

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict fraud probability.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with the 20 feature columns.

        Returns
        -------
        np.ndarray
            Array of probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        X = features_df[FEATURES].values
        return self.model.predict_proba(X)[:, 1]

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict binary labels using the configured threshold.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with the 20 feature columns.

        Returns
        -------
        np.ndarray
            Binary predictions (1 = flagged fraud).
        """
        proba = self.predict_proba(features_df)
        return (proba >= self.threshold).astype(int)

    def predict_with_cost(
        self, features_df: pd.DataFrame, amounts: np.ndarray
    ) -> dict:
        """Predict with full cost breakdown.

        Parameters
        ----------
        features_df : pd.DataFrame
            DataFrame with the 20 feature columns (must also contain
            ``chargeback_label`` for ground-truth comparison).
        amounts : np.ndarray
            Transaction amounts in INR.

        Returns
        -------
        dict
            Keys: predictions, probabilities, threshold, cost_breakdown.
        """
        from models.cost_matrix import calculate_cost

        proba = self.predict_proba(features_df)
        preds = (proba >= self.threshold).astype(int)

        if "chargeback_label" in features_df.columns:
            y_true = features_df["chargeback_label"].astype(int).values
            cost_breakdown = calculate_cost(y_true, preds, amounts)
        else:
            cost_breakdown = {}

        return {
            "predictions": preds,
            "probabilities": proba,
            "threshold": self.threshold,
            "cost_breakdown": cost_breakdown,
        }

    def save(self, path: str | Path = DEFAULT_ARTIFACT_PATH) -> None:
        """Save trained model and metadata to disk.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "feature_engine": self.feature_engine,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "metrics": self.metrics,
            "feature_names": self.feature_names,
        }

        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"[Stage1] Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> Stage1RiskScorer:
        """Load a trained model from disk.

        Parameters
        ----------
        path : str | Path
            Path to a pickled artifact file.

        Returns
        -------
        Stage1RiskScorer
            A scorer instance ready for prediction.
        """
        path = Path(path)
        with open(path, "rb") as f:
            artifact = pickle.load(f)

        scorer = cls(threshold_mode=artifact["threshold_mode"])
        scorer.model = artifact["model"]
        scorer.feature_engine = artifact["feature_engine"]
        scorer.threshold = artifact["threshold"]
        scorer.metrics = artifact["metrics"]
        scorer.feature_names = artifact["feature_names"]
        print(f"[Stage1] Model loaded from {path}")
        return scorer

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance rankings.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance — sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 — XGBoost Risk Scorer Training")
    print("=" * 60)

    # 1. Load data
    data_path = Path("data/transactions.csv")
    if not data_path.exists():
        print(f"[ERROR] {data_path} not found. Run data/generate.py first.")
        raise SystemExit(1)

    print(f"\n[Step 1] Loading data from {data_path} ...")
    raw_df = pd.read_csv(data_path)
    print(f"  Rows: {len(raw_df)}  Columns: {len(raw_df.columns)}")

    # 2. Fit FeatureEngine
    print("\n[Step 2] Fitting FeatureEngine ...")
    fe = FeatureEngine()
    fe.fit(raw_df)
    print(f"  Feature count: {len(fe.feature_names)}")

    # 3. Train scorer
    print("\n[Step 3] Training Stage1RiskScorer ...")
    scorer = Stage1RiskScorer(threshold_mode="cost_optimized")
    metrics = scorer.train(raw_df, fe)

    # 4. Print metrics
    print("\n[Step 4] Training Metrics:")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:>12s}: {val:.4f}")
        else:
            print(f"  {key:>12s}: {val}")

    # 5. Feature importances (top 10)
    print("\n[Step 5] Top 10 Feature Importances:")
    importance_df = scorer.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # 6. Save artifacts
    print("\n[Step 6] Saving model artifacts ...")
    scorer.save()

    print("\n" + "=" * 60)
    print("Stage 1 training complete.")
    print("=" * 60)

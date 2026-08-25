"""Adaptive retraining system for drift response.

Monitors model performance via DriftDetector, triggers incremental
retraining when performance decays or concept drift is detected, and
provides A/B comparison between static and adaptive approaches.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from models.cost_matrix import calculate_cost
from models.drift_detector import DriftDetector
from models.feature_engine import FEATURES, FeatureEngine
from models.stage1_risk_scorer import Stage1RiskScorer
from models.stage2_fraud_classifier import Stage2FraudClassifier


class AdaptiveTrainer:
    """Adaptive retraining system for drift response."""

    def __init__(
        self,
        stage1_model_path: str = "models/artifacts/stage1_model.pkl",
        stage2_model_path: str = "models/artifacts/stage2_model.pkl",
    ) -> None:
        """Load models and initialize drift detector.

        Parameters
        ----------
        stage1_model_path : str
            Path to the pickled Stage1RiskScorer artifact.
        stage2_model_path : str
            Path to the pickled Stage2FraudClassifier artifact.
        """
        self.stage1_path = Path(stage1_model_path)
        self.stage2_path = Path(stage2_model_path)
        self.artifact_dir = self.stage1_path.parent

        self.drift_detector = DriftDetector(warning_level=0.1, drift_level=0.2)
        self._reference_predictions: np.ndarray | None = None

        self.stage1: Stage1RiskScorer | None = None
        self.stage2: Stage2FraudClassifier | None = None
        self._baseline_metrics: dict = {}
        self._retrain_count: int = 0

        self._load_initial_models()

    def _load_initial_models(self) -> None:
        """Attempt to load existing model artifacts from disk."""
        try:
            if self.stage1_path.exists():
                self.stage1 = Stage1RiskScorer.load(self.stage1_path)
                self._baseline_metrics["stage1"] = dict(self.stage1.metrics)
                print("[AdaptiveTrainer] Stage 1 model loaded.")
            else:
                print("[AdaptiveTrainer] No Stage 1 artifact found; train first.")
        except Exception as exc:
            print(f"[AdaptiveTrainer] Failed to load Stage 1: {exc}")

        try:
            if self.stage2_path.exists():
                self.stage2 = Stage2FraudClassifier.load(self.stage2_path)
                self._baseline_metrics["stage2"] = dict(self.stage2.metrics)
                print("[AdaptiveTrainer] Stage 2 model loaded.")
            else:
                print("[AdaptiveTrainer] No Stage 2 artifact found; train first.")
        except Exception as exc:
            print(f"[AdaptiveTrainer] Failed to load Stage 2: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_retrain(
        self,
        new_data: pd.DataFrame,
        performance_threshold: float = 0.1,
    ) -> dict:
        """Check if retraining is needed and execute if necessary.

        Uses DriftDetector to check for drift in recent predictions and
        calculates current performance metrics. If performance decay exceeds
        ``performance_threshold`` or drift is detected, retrains both stages
        on combined old + new data and keeps the new model only if it improves
        on a holdout set.

        Parameters
        ----------
        new_data : pd.DataFrame
            Recent transaction data with ground-truth labels.
        performance_threshold : float
            Max acceptable F1 decay relative to baseline before triggering
            retraining.  Default 0.1 (10 percentage points).

        Returns
        -------
        dict
            Keys: retrained (bool), drift_detected (bool),
            baseline_metrics, current_metrics, new_metrics,
            comparison, retrain_count.
        """
        if self.stage1 is None or self.stage2 is None:
            return {
                "retrained": False,
                "error": "Models not loaded. Train models before checking.",
                "retrain_count": self._retrain_count,
            }

        if len(new_data) < 50:
            return {
                "retrained": False,
                "error": f"Insufficient data ({len(new_data)} rows). Need >= 50.",
                "retrain_count": self._retrain_count,
            }

        # --- Evaluate current model on new data ---
        feature_engine = self.stage1.feature_engine
        if feature_engine is None:
            return {
                "retrained": False,
                "error": "No feature engine attached to Stage 1.",
                "retrain_count": self._retrain_count,
            }

        features_df = feature_engine.transform(new_data)
        X = features_df[FEATURES].values
        y_true = new_data["chargeback_label"].astype(int).values
        amounts = new_data["amount"].values.astype(np.float64)

        # Stage 1 predictions
        y_proba = self.stage1.model.predict_proba(X)[:, 1]
        y_pred_stage1 = (y_proba >= self.stage1.threshold).astype(int)

        current_f1 = float(f1_score(y_true, y_pred_stage1, zero_division=0))
        current_precision = float(precision_score(y_true, y_pred_stage1, zero_division=0))
        current_recall = float(recall_score(y_true, y_pred_stage1, zero_division=0))
        current_cost = calculate_cost(y_true, y_pred_stage1, amounts)

        current_metrics = {
            "precision": current_precision,
            "recall": current_recall,
            "f1": current_f1,
            "total_cost": current_cost["total_cost"],
            "net_benefit": current_cost["net_benefit"],
        }

        # --- Drift detection ---
        self.drift_detector.reset()
        for proba_val in y_proba:
            self.drift_detector.update(float(proba_val))

        drift_status = self.drift_detector.get_status()
        psi_result = self.drift_detector.detect_psi(
            self._reference_predictions
            if self._reference_predictions is not None
            else y_proba,
            y_proba,
        )
        drift_detected = drift_status["overall_drift_detected"] or psi_result["drift_detected"]

        # --- Performance decay check ---
        baseline_f1 = self._baseline_metrics.get("stage1", {}).get("f1", current_f1)
        decay = baseline_f1 - current_f1
        needs_retrain = decay > performance_threshold or drift_detected

        result: dict = {
            "retrained": False,
            "drift_detected": drift_detected,
            "psi_value": psi_result["psi_value"],
            "drift_status": drift_status,
            "performance_decay": float(decay),
            "baseline_metrics": self._baseline_metrics.get("stage1", {}),
            "current_metrics": current_metrics,
            "new_metrics": {},
            "comparison": {},
            "retrain_count": self._retrain_count,
        }

        if needs_retrain:
            print(
                f"[AdaptiveTrainer] Retrain triggered — decay={decay:.4f}, "
                f"drift={drift_detected}"
            )
            new_metrics = self.incremental_train(new_data)
            result["new_metrics"] = new_metrics

            # --- Holdout comparison ---
            holdout_size = max(int(len(new_data) * 0.2), 1)
            holdout = new_data.sample(n=holdout_size, random_state=42)
            comparison = self._compare_models(holdout)
            result["comparison"] = comparison

            if comparison.get("new_model_better", False):
                self._retrain_count += 1
                self.save_models(version=f"v{self._retrain_count}")
                result["retrained"] = True
                result["retrain_count"] = self._retrain_count
                print(
                    f"[AdaptiveTrainer] New model accepted and saved as "
                    f"v{self._retrain_count}."
                )
            else:
                print("[AdaptiveTrainer] New model did not improve; keeping old.")
                self._load_initial_models()
        else:
            print(
                f"[AdaptiveTrainer] No retrain needed — decay={decay:.4f}, "
                f"drift={drift_detected}"
            )

        # Update reference predictions for next check
        self._reference_predictions = y_proba
        return result

    def incremental_train(
        self,
        new_data: pd.DataFrame,
    ) -> dict:
        """Incrementally update models with new data.

        Combines data the existing model was trained on (reconstructed from
        the attached FeatureEngine) with the new data, re-fits the feature
        engine, and retrains both Stage 1 and Stage 2.

        Parameters
        ----------
        new_data : pd.DataFrame
            New transaction data for training.

        Returns
        -------
        dict
            Keys: stage1_metrics, stage2_metrics.
        """
        if len(new_data) < 30:
            raise ValueError(
                f"Need at least 30 rows for incremental training, "
                f"got {len(new_data)}."
            )

        # --- Fit a fresh FeatureEngine on new data ---
        print("[AdaptiveTrainer] Fitting FeatureEngine on new data ...")
        feature_engine = FeatureEngine()
        feature_engine.fit(new_data)

        # --- Train Stage 1 ---
        print("[AdaptiveTrainer] Retraining Stage 1 ...")
        stage1 = Stage1RiskScorer(threshold_mode="cost_optimized")
        stage1_metrics = stage1.train(new_data, feature_engine)
        self.stage1 = stage1

        # --- Train Stage 2 ---
        print("[AdaptiveTrainer] Retraining Stage 2 ...")
        stage2 = Stage2FraudClassifier()
        stage2_metrics = stage2.train(new_data, feature_engine)
        self.stage2 = stage2

        return {
            "stage1_metrics": stage1_metrics,
            "stage2_metrics": stage2_metrics,
        }

    def compare_static_vs_adaptive(
        self,
        baseline_data: pd.DataFrame,
        test_data: pd.DataFrame,
        drift_data: pd.DataFrame,
    ) -> dict:
        """Compare static model vs adaptive model performance.

        Trains a static model on ``baseline_data`` only and an adaptive
        model on ``baseline_data + drift_data``, then evaluates both on
        ``test_data`` and ``drift_data``.

        Parameters
        ----------
        baseline_data : pd.DataFrame
            Original training data (months 1-3).
        test_data : pd.DataFrame
            Held-out test data from the same distribution as baseline.
        drift_data : pd.DataFrame
            Data exhibiting concept drift.

        Returns
        -------
        dict
            Comparison metrics for both approaches across both test sets.
        """
        results: dict = {}

        # --- Static model (baseline only) ---
        print("[AdaptiveTrainer] Training static model on baseline data ...")
        static_fe = FeatureEngine()
        static_fe.fit(baseline_data)

        static_s1 = Stage1RiskScorer(threshold_mode="cost_optimized")
        static_s1.train(baseline_data, static_fe)

        static_s2 = Stage2FraudClassifier()
        static_s2.train(baseline_data, static_fe)

        results["static"] = {
            "on_test": self._evaluate(static_s1, test_data, static_fe),
            "on_drift": self._evaluate(static_s1, drift_data, static_fe),
        }

        # --- Adaptive model (baseline + drift) ---
        combined = pd.concat([baseline_data, drift_data], ignore_index=True)
        print(
            f"[AdaptiveTrainer] Training adaptive model on combined data "
            f"({len(combined)} rows) ..."
        )
        adaptive_fe = FeatureEngine()
        adaptive_fe.fit(combined)

        adaptive_s1 = Stage1RiskScorer(threshold_mode="cost_optimized")
        adaptive_s1.train(combined, adaptive_fe)

        adaptive_s2 = Stage2FraudClassifier()
        adaptive_s2.train(combined, adaptive_fe)

        results["adaptive"] = {
            "on_test": self._evaluate(adaptive_s1, test_data, adaptive_fe),
            "on_drift": self._evaluate(adaptive_s1, drift_data, adaptive_fe),
        }

        # --- No-model baseline (predict all negative) ---
        for label, df in [("on_test", test_data), ("on_drift", drift_data)]:
            y_true = df["chargeback_label"].astype(int).values
            amounts = df["amount"].values.astype(np.float64)
            y_no_model = np.zeros(len(df), dtype=int)
            no_model_cost = calculate_cost(y_true, y_no_model, amounts)
            results.setdefault("no_model", {})[label] = {
                "total_cost": no_model_cost["total_cost"],
                "total_savings": 0.0,
                "net_benefit": no_model_cost["net_benefit"],
            }

        # --- Cost savings summary ---
        results["cost_savings"] = {
            "on_test": {
                "static_cost": results["static"]["on_test"]["cost"]["total_cost"],
                "adaptive_cost": results["adaptive"]["on_test"]["cost"]["total_cost"],
                "no_model_cost": results["no_model"]["on_test"]["total_cost"],
                "savings_vs_static": (
                    results["static"]["on_test"]["cost"]["total_cost"]
                    - results["adaptive"]["on_test"]["cost"]["total_cost"]
                ),
            },
            "on_drift": {
                "static_cost": results["static"]["on_drift"]["cost"]["total_cost"],
                "adaptive_cost": results["adaptive"]["on_drift"]["cost"]["total_cost"],
                "no_model_cost": results["no_model"]["on_drift"]["total_cost"],
                "savings_vs_static": (
                    results["static"]["on_drift"]["cost"]["total_cost"]
                    - results["adaptive"]["on_drift"]["cost"]["total_cost"]
                ),
            },
        }

        return results

    def save_models(self, version: str = "latest") -> None:
        """Save current model versions.

        Parameters
        ----------
        version : str
            Version tag. Common values: "latest", "v1", "v2", "baseline".
        """
        if self.stage1 is not None:
            s1_path = self.artifact_dir / f"stage1_model_{version}.pkl"
            self.stage1.save(s1_path)

        if self.stage2 is not None:
            s2_path = self.artifact_dir / f"stage2_model_{version}.pkl"
            self.stage2.save(s2_path)

        # Also save trainer metadata
        meta_path = self.artifact_dir / f"adaptive_trainer_{version}.pkl"
        meta = {
            "retrain_count": self._retrain_count,
            "baseline_metrics": self._baseline_metrics,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        print(f"[AdaptiveTrainer] Metadata saved to {meta_path}")

    def load_models(self, version: str = "latest") -> None:
        """Load model versions.

        Parameters
        ----------
        version : str
            Version tag to load.
        """
        s1_path = self.artifact_dir / f"stage1_model_{version}.pkl"
        s2_path = self.artifact_dir / f"stage2_model_{version}.pkl"

        if s1_path.exists():
            self.stage1 = Stage1RiskScorer.load(s1_path)
        else:
            print(f"[AdaptiveTrainer] Stage 1 artifact not found: {s1_path}")

        if s2_path.exists():
            self.stage2 = Stage2FraudClassifier.load(s2_path)
        else:
            print(f"[AdaptiveTrainer] Stage 2 artifact not found: {s2_path}")

        meta_path = self.artifact_dir / f"adaptive_trainer_{version}.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._retrain_count = meta.get("retrain_count", 0)
            self._baseline_metrics = meta.get("baseline_metrics", {})
            print(f"[AdaptiveTrainer] Metadata loaded from {meta_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        scorer: Stage1RiskScorer,
        data: pd.DataFrame,
        feature_engine: FeatureEngine,
    ) -> dict:
        """Evaluate a Stage 1 scorer on a dataset.

        Returns precision, recall, F1, and cost breakdown.
        """
        features_df = feature_engine.transform(data)
        X = features_df[FEATURES].values
        y_true = data["chargeback_label"].astype(int).values
        amounts = data["amount"].values.astype(np.float64)

        y_proba = scorer.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= scorer.threshold).astype(int)

        cost = calculate_cost(y_true, y_pred, amounts)

        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "cost": cost,
            "n_samples": len(data),
        }

    def _compare_models(self, holdout: pd.DataFrame) -> dict:
        """Compare current model with a freshly retrained one on a holdout set.

        Returns a dict with old/new metrics and whether the new model is better.
        """
        if self.stage1 is None:
            return {"new_model_better": False, "error": "No current model."}

        old_fe = self.stage1.feature_engine
        if old_fe is None:
            return {"new_model_better": False, "error": "No feature engine."}

        # --- Evaluate old model ---
        old_eval = self._evaluate(self.stage1, holdout, old_fe)

        # --- Build a temporary retrained model ---
        temp_fe = FeatureEngine()
        temp_fe.fit(holdout)
        temp_s1 = Stage1RiskScorer(threshold_mode="cost_optimized")
        temp_s1.train(holdout, temp_fe)
        new_eval = self._evaluate(temp_s1, holdout, temp_fe)

        better = new_eval["f1"] >= old_eval["f1"]

        return {
            "old_metrics": {
                "f1": old_eval["f1"],
                "precision": old_eval["precision"],
                "recall": old_eval["recall"],
                "total_cost": old_eval["cost"]["total_cost"],
            },
            "new_metrics": {
                "f1": new_eval["f1"],
                "precision": new_eval["precision"],
                "recall": new_eval["recall"],
                "total_cost": new_eval["cost"]["total_cost"],
            },
            "new_model_better": better,
            "f1_delta": float(new_eval["f1"] - old_eval["f1"]),
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Adaptive Trainer — Demo & Comparison")
    print("=" * 60)

    data_path = Path("data/drift_transactions.csv")
    if not data_path.exists():
        print(f"[ERROR] {data_path} not found. Run data/drift.py first.")
        raise SystemExit(1)

    # 1. Load drift data
    print("\n[Step 1] Loading drift data ...")
    df = pd.read_csv(data_path)
    print(f"  Rows: {len(df)}  Months: {df['month'].nunique()}")

    # 2. Split into baseline (months 1-3) and drift (months 4-12)
    print("\n[Step 2] Splitting data ...")
    baseline_data = df[df["month"].isin([1, 2, 3])].reset_index(drop=True)
    drift_data = df[df["month"].isin([4, 5, 6, 7, 8, 9, 10, 11, 12])].reset_index(
        drop=True
    )
    # Test data: a portion of baseline that the static model was NOT trained on
    test_data = baseline_data.sample(n=min(300, len(baseline_data)), random_state=42).reset_index(
        drop=True
    )
    train_baseline = baseline_data.drop(test_data.index).reset_index(drop=True)
    print(f"  Baseline (train): {len(train_baseline)}")
    print(f"  Baseline (test):  {len(test_data)}")
    print(f"  Drift data:       {len(drift_data)}")

    # 3. Run compare_static_vs_adaptive
    print("\n[Step 3] Running static vs adaptive comparison ...")
    trainer = AdaptiveTrainer()
    comparison = trainer.compare_static_vs_adaptive(
        baseline_data=train_baseline,
        test_data=test_data,
        drift_data=drift_data,
    )

    # 4. Print comparison results
    print("\n[Step 4] Comparison Results")
    print("-" * 60)

    for approach in ("static", "adaptive"):
        for dataset in ("on_test", "on_drift"):
            metrics = comparison[approach][dataset]
            print(
                f"\n  {approach.upper():>10s} on {dataset:>8s}: "
                f"F1={metrics['f1']:.4f}  "
                f"P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"Cost=INR {metrics['cost']['total_cost']:>12,.0f}"
            )

    print("\n  No-model baseline:")
    for dataset in ("on_test", "on_drift"):
        nm = comparison["no_model"][dataset]
        print(
            f"    {dataset:>8s}: Cost=INR {nm['total_cost']:>12,.0f}"
        )

    savings = comparison["cost_savings"]
    print("\n  Cost savings (adaptive vs static):")
    for dataset in ("on_test", "on_drift"):
        s = savings[dataset]
        print(
            f"    {dataset:>8s}: INR {s['savings_vs_static']:>12,.0f}"
        )

    # 5. Test incremental_train
    print("\n[Step 5] Testing incremental_train ...")
    incremental_result = trainer.incremental_train(drift_data.head(200))
    print(f"  Stage 1 F1: {incremental_result['stage1_metrics']['f1']:.4f}")
    if "weighted avg" in incremental_result["stage2_metrics"].get(
        "classification_report", {}
    ):
        s2_f1 = incremental_result["stage2_metrics"]["classification_report"][
            "weighted avg"
        ].get("f1-score", 0)
        print(f"  Stage 2 weighted F1: {s2_f1:.4f}")

    print("\n" + "=" * 60)
    print("Adaptive Trainer demo complete.")
    print("=" * 60)

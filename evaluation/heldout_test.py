"""Held-out test evaluation module for AI Risk Manager.

Runs a full 80/20 train/test split evaluation pipeline, trains both
Stage 1 and Stage 2 models on the training set, evaluates on the held-out
test set, and generates comprehensive reports and visualisation plots.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from sklearn.model_selection import train_test_split

from data.schema import FraudLabel
from evaluation.cost_analysis import EvaluationCostAnalyzer
from evaluation.metrics import MetricsCalculator
from models.feature_engine import FEATURES, FeatureEngine
from models.stage1_risk_scorer import Stage1RiskScorer
from models.stage2_fraud_classifier import Stage2FraudClassifier

FRAUD_LABELS: list[str] = [e.value for e in FraudLabel]


def _format_inr(value: float) -> str:
    """Format a number in Indian Rupee style (lakhs/crores grouping).

    Example: 1234567 -> "12,34,567"
    """
    if value < 0:
        return f"-{_format_inr(-value)}"
    n = int(round(value))
    if n < 1000:
        return str(n)
    # Last 3 digits
    last_three = str(n % 1000).zfill(3)
    remaining = n // 1000
    if remaining == 0:
        return last_three.lstrip("0") or "0"
    # Group remaining digits in pairs from the right
    groups: list[str] = []
    while remaining > 0:
        groups.append(str(remaining % 100).zfill(2) if remaining >= 100 else str(remaining % 100))
        remaining //= 100
    groups.reverse()
    # The first group should not be zero-padded
    groups[0] = str(int(groups[0]))
    return ",".join(groups) + "," + last_three


class HeldOutEvaluator:
    """Full evaluation pipeline with 80/20 train/test split."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """Initialize evaluator.

        Parameters
        ----------
        test_size : float
            Fraction of data to reserve for testing.
        random_state : int
            Random seed for reproducible splits.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.feature_engine: FeatureEngine | None = None
        self.stage1_scorer: Stage1RiskScorer | None = None
        self.stage2_classifier: Stage2FraudClassifier | None = None
        self.metrics_calculator = MetricsCalculator()
        self.cost_analyzer = EvaluationCostAnalyzer()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_full_evaluation(self, data_path: str = "data/transactions.csv") -> dict:
        """Run complete evaluation pipeline.

        1. Load data
        2. Split 80/20
        3. Train Stage 1 on train set
        4. Train Stage 2 on train set
        5. Evaluate on test set
        6. Generate report

        Parameters
        ----------
        data_path : str
            Path to the transactions CSV.

        Returns
        -------
        dict
            All evaluation results.
        """
        print("=" * 60)
        print("AI Risk Manager — Held-Out Test Evaluation")
        print("=" * 60)

        # --- 1. Load data ---
        path = Path(data_path)
        if not path.exists():
            print(f"[ERROR] Data file not found: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")

        print(f"\n[Step 1/6] Loading data from {path} ...")
        df = pd.read_csv(path)
        df["chargeback_label"] = df["chargeback_label"].astype(int)
        print(f"  Loaded {len(df):,} transactions")

        # --- 2. Split data ---
        print(f"\n[Step 2/6] Splitting data ({1 - self.test_size:.0%}/{self.test_size:.0%}) ...")
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["chargeback_label"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        fraud_rate_train = train_df["chargeback_label"].mean()
        fraud_rate_test = test_df["chargeback_label"].mean()
        print(f"  Train set: {len(train_df):,} transactions ({fraud_rate_train:.1%} fraud)")
        print(f"  Test set:  {len(test_df):,} transactions ({fraud_rate_test:.1%} fraud)")

        # --- 3. Fit FeatureEngine on train ---
        print("\n[Step 3/6] Fitting FeatureEngine on train set ...")
        self.feature_engine = FeatureEngine()
        self.feature_engine.fit(train_df)
        print(f"  Feature count: {len(self.feature_engine.feature_names)}")

        # --- 4. Train Stage 1 ---
        print("\n[Step 4/6] Training Stage 1 (Risk Scorer) ...")
        self.stage1_scorer = Stage1RiskScorer(threshold_mode="cost_optimized")
        train_metrics_s1 = self.stage1_scorer.train(train_df, self.feature_engine)

        # --- 5. Train Stage 2 ---
        print("\n[Step 5/6] Training Stage 2 (Fraud Classifier) ...")
        self.stage2_classifier = Stage2FraudClassifier()
        train_metrics_s2 = self.stage2_classifier.train(train_df, self.feature_engine)

        # --- 6. Evaluate on test set ---
        print("\n[Step 6/6] Evaluating on held-out test set ...")
        test_features = self.feature_engine.transform(test_df)
        amounts_test = test_df["amount"].values.astype(np.float64)
        y_true_binary = test_features["chargeback_label"].astype(int).values

        # Stage 1 predictions
        y_proba = self.stage1_scorer.predict_proba(test_features)
        threshold = self.stage1_scorer.threshold
        y_pred_binary = (y_proba >= threshold).astype(int)

        stage1_metrics = self.metrics_calculator.calculate_all(
            y_true_binary, y_pred_binary, y_proba
        )
        cost_analysis = self.cost_analyzer.analyze(
            y_true_binary, y_pred_binary, amounts_test
        )
        savings_report = self.cost_analyzer.savings_report(
            y_true_binary, y_pred_binary, amounts_test
        )
        threshold_strategies = self.cost_analyzer.compare_strategies(
            y_true_binary, y_proba, amounts_test
        )

        # Stage 2 predictions on flagged transactions only
        stage2_report: str = ""
        stage2_classification: dict = {}
        if y_pred_binary.sum() > 0:
            flagged_mask = y_pred_binary.astype(bool)
            stage2_results = self.stage2_classifier.predict(
                test_features, fraud_mask=flagged_mask
            )
            # Build ground-truth fraud types for flagged rows
            true_fraud_types = test_df.loc[flagged_mask, "fraud_type"].values
            pred_fraud_types = np.array(stage2_results["fraud_types"])
            stage2_report = self.metrics_calculator.classification_report(
                true_fraud_types,
                pred_fraud_types,
                labels=FRAUD_LABELS,
            )
            stage2_classification = {
                "precision": float(
                    np.mean(
                        [
                            true_fraud_types[i] == pred_fraud_types[i]
                            for i in range(len(true_fraud_types))
                        ]
                    )
                ) if len(true_fraud_types) > 0 else 0.0,
                "n_flagged": int(flagged_mask.sum()),
            }

        # Feature importance
        feature_importance = self.stage1_scorer.get_feature_importance()

        # Compile results
        results = {
            "data_summary": {
                "total_transactions": len(df),
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_fraud_rate": float(fraud_rate_train),
                "test_fraud_rate": float(fraud_rate_test),
            },
            "stage1_metrics": stage1_metrics,
            "stage1_cost_analysis": cost_analysis,
            "stage1_savings_report": savings_report,
            "stage1_threshold_strategies": threshold_strategies,
            "stage1_feature_importance": feature_importance.to_dict(orient="records"),
            "stage2_report": stage2_report,
            "stage2_classification": stage2_classification,
            "test_amounts": amounts_test,
            "y_true": y_true_binary,
            "y_pred": y_pred_binary,
            "y_proba": y_proba,
            "train_metrics_s1": train_metrics_s1,
            "train_metrics_s2": train_metrics_s2,
        }

        print("\nEvaluation complete.")
        return results

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, results: dict, output_dir: str = "evaluation/reports") -> str:
        """Generate comprehensive evaluation report.

        Parameters
        ----------
        results : dict
            Output from ``run_full_evaluation()``.
        output_dir : str
            Directory to save the report.

        Returns
        -------
        str
            Path to saved report file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ds = results["data_summary"]
        s1 = results["stage1_metrics"]
        cm = s1["confusion_matrix"]
        cost = results["stage1_cost_analysis"]
        savings = results["stage1_savings_report"]
        strategies = results["stage1_threshold_strategies"]

        lines: list[str] = []
        lines.append("=== AI Risk Manager - Evaluation Report ===")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Data summary
        lines.append("--- Data Summary ---")
        lines.append(f"Total transactions: {ds['total_transactions']:,}")
        lines.append(f"Train set: {ds['train_size']:,} ({ds['train_size'] / ds['total_transactions']:.0%})")
        lines.append(f"Test set: {ds['test_size']:,} ({ds['test_size'] / ds['total_transactions']:.0%})")
        lines.append(f"Fraud rate (train): {ds['train_fraud_rate']:.1%}")
        lines.append(f"Fraud rate (test): {ds['test_fraud_rate']:.1%}")
        lines.append("")

        # Stage 1 metrics
        lines.append("--- Stage 1: Risk Scorer Metrics ---")
        lines.append(f"Precision: {s1['precision']:.2f}")
        lines.append(f"Recall: {s1['recall']:.2f}")
        lines.append(f"F1-Score: {s1['f1']:.2f}")
        lines.append(f"AUC-ROC: {s1['auc_roc']:.2f}" if s1["auc_roc"] is not None else "AUC-ROC: N/A")
        lines.append(f"AUC-PR: {s1['auc_pr']:.2f}" if s1["auc_pr"] is not None else "AUC-PR: N/A")
        lines.append("")

        # Confusion matrix
        lines.append("--- Confusion Matrix ---")
        lines.append(f"TP: {cm['tp']}  FP: {cm['fp']}")
        lines.append(f"FN: {cm['fn']}  TN: {cm['tn']}")
        lines.append("")

        # Cost analysis
        lines.append("--- Cost Analysis ---")
        lines.append(f"Total cost (model): INR {_format_inr(cost['total_cost'])}")
        lines.append(f"Total savings: INR {_format_inr(cost['total_savings'])}")
        lines.append(f"Net benefit: INR {_format_inr(cost['net_benefit'])}")
        lines.append(f"Savings rate: {savings['savings_pct']:.1f}%")
        lines.append("")

        # Threshold analysis
        lines.append("--- Threshold Analysis ---")
        for name, data in strategies.items():
            label = name.replace("_", " ").title()
            lines.append(
                f"{label} ({data['threshold']:.2f}): "
                f"F1={data['f1_score']:.2f}, "
                f"Cost=INR {_format_inr(data['metrics']['total_cost'])}"
            )
        lines.append("")

        # Stage 2
        lines.append("--- Stage 2: Fraud Type Classification ---")
        if results["stage2_report"]:
            lines.append(results["stage2_report"])
        else:
            lines.append("No flagged transactions to classify.")
        lines.append("")

        # Feature importance
        lines.append("--- Top 10 Feature Importance ---")
        for i, feat in enumerate(results["stage1_feature_importance"][:10]):
            lines.append(f"  {i + 1:>2d}. {feat['feature']:<25s} {feat['importance']:.4f}")
        lines.append("")

        report_text = "\n".join(lines)
        report_path = out / "evaluation_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        # Also save metrics as JSON (excluding non-serialisable fields)
        json_results = {
            "data_summary": results["data_summary"],
            "stage1_metrics": {
                k: v for k, v in results["stage1_metrics"].items() if k != "confusion_matrix"
            },
            "stage1_confusion_matrix": results["stage1_metrics"]["confusion_matrix"],
            "stage1_cost_analysis": results["stage1_cost_analysis"],
            "stage1_savings_report": results["stage1_savings_report"],
            "stage2_classification": results["stage2_classification"],
        }
        json_path = out / "metrics_summary.json"
        json_path.write_text(json.dumps(json_results, indent=2, default=str), encoding="utf-8")

        print(f"  Report saved to {report_path}")
        print(f"  Metrics JSON saved to {json_path}")
        return str(report_path)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self, results: dict, output_dir: str = "evaluation/reports") -> None:
        """Generate all visualisation plots.

        Parameters
        ----------
        results : dict
            Output from ``run_full_evaluation()``.
        output_dir : str
            Directory to save the plots.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        y_true = results["y_true"]
        y_pred = results["y_pred"]
        y_proba = results["y_proba"]
        amounts = results["test_amounts"]

        self._plot_confusion_matrix(y_true, y_pred, out / "confusion_matrix.png")
        self._plot_roc_curve(y_true, y_proba, out / "roc_curve.png")
        self._plot_pr_curve(y_true, y_proba, out / "pr_curve.png")
        self._plot_cost_curve(y_true, y_proba, amounts, out / "cost_curve.png")
        self._plot_feature_importance(
            results["stage1_feature_importance"], out / "feature_importance.png"
        )
        print(f"  All plots saved to {out}")

    # ------------------------------------------------------------------
    # Private plot helpers
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Path
    ) -> None:
        """Plot confusion matrix heatmap."""
        labels = [
            "genuine",
            "friendly_fraud",
            "account_takeover",
            "technical_failure",
        ]
        # For binary evaluation, map to genuine vs fraud
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["genuine", "fraud"],
            yticklabels=["genuine", "fraud"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Stage 1 (Fraud Detection)")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_roc_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Path
    ) -> None:
        """Plot ROC curve with AUC score."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Stage 1")
        ax.legend(loc="lower right")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_pr_curve(
        self, y_true: np.ndarray, y_proba: np.ndarray, save_path: Path
    ) -> None:
        """Plot Precision-Recall curve with AUC-PR score."""
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall_vals, precision_vals, linewidth=2, label=f"PR (AUC = {auc_pr:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve - Stage 1")
        ax.legend(loc="lower left")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_cost_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
        save_path: Path,
    ) -> None:
        """Plot cost curve across thresholds."""
        self.cost_analyzer.plot_cost_curve(
            y_true, y_scores, amounts, save_path=str(save_path)
        )

    def _plot_feature_importance(
        self, importance_records: list[dict], save_path: Path
    ) -> None:
        """Plot top 10 feature importances as a horizontal bar chart."""
        top10 = importance_records[:10]
        features = [r["feature"] for r in top10][::-1]
        importances = [r["importance"] for r in top10][::-1]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_style("whitegrid")
        ax.barh(features, importances, color="#3498db", edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importance — Stage 1")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    evaluator = HeldOutEvaluator(test_size=0.2, random_state=42)
    results = evaluator.run_full_evaluation()
    evaluator.generate_report(results)
    evaluator.plot_results(results)
    print("\nDone.")

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix as sk_confusion_matrix,
)


class MetricsCalculator:
    """Calculate standard ML metrics for fraud detection."""

    def __init__(self) -> None:
        """Initialize the calculator."""
        pass

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict:
        """Calculate all standard metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)

        Returns:
            dict with precision, recall, f1, auc_roc, auc_pr, confusion_matrix
        """
        metrics: dict = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": self.confusion_matrix(y_true, y_pred),
        }

        if y_proba is not None and len(np.unique(y_true)) > 1:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_proba))
        else:
            metrics["auc_roc"] = None
            metrics["auc_pr"] = None

        return metrics

    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Generate confusion matrix with labels.

        Returns:
            dict with tp, fp, fn, tn counts and rates
        """
        cm = sk_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = tp + fp + fn + tn

        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "tp_rate": float(tp / total) if total > 0 else 0.0,
            "fp_rate": float(fp / total) if total > 0 else 0.0,
            "fn_rate": float(fn / total) if total > 0 else 0.0,
            "tn_rate": float(tn / total) if total > 0 else 0.0,
            "total": int(total),
        }

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str] | None = None,
    ) -> str:
        """Generate formatted classification report.

        Returns:
            String formatted report like sklearn's
        """
        unique_labels = sorted(set(y_true) | set(y_pred))
        if labels is not None:
            unique_labels = [i for i in range(len(labels)) if i in unique_labels]

        label_names = labels if labels is not None else [str(l) for l in unique_labels]

        header = f"{'':>20}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
        lines: list[str] = [header, ""]

        support_total = 0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0

        for idx, label in enumerate(unique_labels):
            tp = int(((y_true == label) & (y_pred == label)).sum())
            fp = int(((y_true != label) & (y_pred == label)).sum())
            fn = int(((y_true == label) & (y_pred != label)).sum())

            support = int((y_true == label).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            name = label_names[idx] if idx < len(label_names) else str(label)
            lines.append(f"{name:>20}{precision:>10.2f}{recall:>10.2f}{f1:>10.2f}{support:>10d}")

            support_total += support
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1

        n_classes = len(unique_labels)
        accuracy = float((y_true == y_pred).sum()) / len(y_true) if len(y_true) > 0 else 0.0
        lines.append("")
        lines.append(f"{'accuracy':>20}{'':>10}{'':>10}{accuracy:>10.2f}{support_total:>10d}")

        macro_precision = precision_sum / n_classes if n_classes > 0 else 0.0
        macro_recall = recall_sum / n_classes if n_classes > 0 else 0.0
        macro_f1 = f1_sum / n_classes if n_classes > 0 else 0.0
        lines.append(
            f"{'macro avg':>20}{macro_precision:>10.2f}{macro_recall:>10.2f}{macro_f1:>10.2f}{support_total:>10d}"
        )

        weighted_precision = precision_sum * support_total / (n_classes * support_total) if support_total > 0 else 0.0
        weighted_recall = recall_sum * support_total / (n_classes * support_total) if support_total > 0 else 0.0
        weighted_f1 = f1_sum * support_total / (n_classes * support_total) if support_total > 0 else 0.0
        lines.append(
            f"{'weighted avg':>20}{weighted_precision:>10.2f}{weighted_recall:>10.2f}{weighted_f1:>10.2f}{support_total:>10d}"
        )

        return "\n".join(lines)

    def cost_weighted_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
    ) -> float:
        """Calculate F1 weighted by transaction amount.

        Returns:
            float: Cost-weighted F1 score
        """
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)

        weighted_tp = float(amounts[tp_mask].sum())
        weighted_fp = float(amounts[fp_mask].sum())
        weighted_fn = float(amounts[fn_mask].sum())

        weighted_precision = (
            weighted_tp / (weighted_tp + weighted_fp)
            if (weighted_tp + weighted_fp) > 0
            else 0.0
        )
        weighted_recall = (
            weighted_tp / (weighted_tp + weighted_fn)
            if (weighted_tp + weighted_fn) > 0
            else 0.0
        )

        if (weighted_precision + weighted_recall) > 0:
            return 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        return 0.0


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    n = 5250

    y_true = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    noise = rng.random(n)
    y_pred = np.where(
        (y_true == 1) & (noise < 0.8), 1,
        np.where((y_true == 0) & (noise < 0.85), 0,
                 np.where((y_true == 1), 0, 1))
    )

    y_proba = np.clip(
        y_pred.astype(float) * 0.8 + rng.random(n) * 0.2,
        0.0, 1.0,
    )

    calc = MetricsCalculator()

    print("=" * 60)
    print("ALL METRICS")
    print("=" * 60)
    all_metrics = calc.calculate_all(y_true, y_pred, y_proba)
    for key, value in all_metrics.items():
        if key == "confusion_matrix":
            continue
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = all_metrics["confusion_matrix"]
    for key, value in cm.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    labels = ["genuine", "friendly_fraud", "account_takeover", "technical_failure"]
    report = calc.classification_report(y_true, y_pred, labels=labels)
    print(report)

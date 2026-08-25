from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


COST_MATRIX = {
    "false_negative": {
        "chargeback_amount_multiplier": 1.0,
        "processing_fee": 500,
        "operational_cost": 200,
        "churn_probability": 0.05,
        "churn_ltv_cost": 2000,
        "rbi_penalty_probability": 0.02,
        "rbi_penalty_amount": 5000,
    },
    "false_positive": {
        "lost_sale_probability": 0.70,
        "manual_review_cost": 150,
        "churn_probability": 0.03,
        "churn_ltv_cost": 2000,
        "investigation_time_minutes": 30,
        "hourly_rate": 500,
    },
    "true_positive": {
        "verification_cost": 100,
        "prevention_benefit": 1.0,
    },
    "true_negative": {
        "cost": 0,
    },
}

RBI_ZERO_LIABILITY_THRESHOLD = 50000
RBI_MAX_COMPENSATION = 25000
RBI_COMPENSATION_RATE = 0.85


def calculate_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_matrix: dict = COST_MATRIX,
) -> dict:
    """Calculate fraud detection costs across all four confusion matrix quadrants.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (1 = fraud, 0 = legitimate).
    y_pred : np.ndarray
        Predicted labels (1 = flagged, 0 = not flagged).
    amounts : np.ndarray
        Transaction amounts in INR.
    cost_matrix : dict
        Cost configuration dictionary.

    Returns
    -------
    dict
        Keys: total_fn_cost, total_fp_cost, total_tp_cost, total_tn_cost,
        total_cost, total_savings, net_benefit.
    """
    fn = cost_matrix["false_negative"]
    fp = cost_matrix["false_positive"]
    tp = cost_matrix["true_positive"]

    mask_fn = (y_true == 1) & (y_pred == 0)
    mask_fp = (y_true == 0) & (y_pred == 1)
    mask_tp = (y_true == 1) & (y_pred == 1)

    amounts_fn = amounts[mask_fn]
    amounts_fp = amounts[mask_fp]
    amounts_tp = amounts[mask_tp]

    # False negatives: missed fraud
    chargeback = amounts_fn * fn["chargeback_amount_multiplier"]
    churn_cost_fn = amounts_fn * fn["churn_probability"] * fn["churn_ltv_cost"]
    rbi_penalty = amounts_fn * fn["rbi_penalty_probability"] * fn["rbi_penalty_amount"]
    total_fn_cost = float(
        chargeback.sum()
        + fn["processing_fee"] * len(amounts_fn)
        + fn["operational_cost"] * len(amounts_fn)
        + churn_cost_fn.sum()
        + rbi_penalty.sum()
    )

    # False positives: legitimate flagged as fraud
    lost_sale = amounts_fp * fp["lost_sale_probability"]
    churn_cost_fp = amounts_fp * fp["churn_probability"] * fp["churn_ltv_cost"]
    investigation_cost = (
        fp["investigation_time_minutes"] / 60.0 * fp["hourly_rate"] * len(amounts_fp)
    )
    total_fp_cost = float(
        lost_sale.sum()
        + fp["manual_review_cost"] * len(amounts_fp)
        + churn_cost_fp.sum()
        + investigation_cost
    )

    # True positives: correctly detected fraud
    total_tp_cost = float(tp["verification_cost"] * len(amounts_tp))

    # True negatives: no cost
    total_tn_cost = 0.0

    total_cost = total_fn_cost + total_fp_cost + total_tp_cost
    total_savings = float((amounts_tp * tp["prevention_benefit"]).sum())
    net_benefit = total_savings - total_cost

    return {
        "total_fn_cost": total_fn_cost,
        "total_fp_cost": total_fp_cost,
        "total_tp_cost": total_tp_cost,
        "total_tn_cost": total_tn_cost,
        "total_cost": total_cost,
        "total_savings": total_savings,
        "net_benefit": net_benefit,
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    amounts: np.ndarray,
    mode: str = "cost_optimized",
    cost_matrix: dict = COST_MATRIX,
) -> float:
    """Find the optimal classification threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_scores : np.ndarray
        Predicted fraud probabilities.
    amounts : np.ndarray
        Transaction amounts in INR.
    mode : str
        "default" returns 0.5, "cost_optimized" minimises total cost,
        "f1_optimized" maximises F1 score.
    cost_matrix : dict
        Cost configuration dictionary.

    Returns
    -------
    float
        Optimal threshold value.
    """
    if mode == "default":
        return 0.5

    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5

    if mode == "cost_optimized":
        best_cost = np.inf
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            result = calculate_cost(y_true, preds, amounts, cost_matrix)
            if result["total_cost"] < best_cost:
                best_cost = result["total_cost"]
                best_threshold = float(t)

    elif mode == "f1_optimized":
        best_f1 = -1.0
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

    return best_threshold


class CostAnalyzer:
    """Analyse fraud detection costs and compare threshold strategies."""

    def __init__(self, cost_matrix: dict = COST_MATRIX) -> None:
        self.cost_matrix = cost_matrix

    def analyze(self, y_true: np.ndarray, y_pred: np.ndarray, amounts: np.ndarray) -> dict:
        """Return full cost breakdown for given predictions."""
        return calculate_cost(y_true, y_pred, amounts, self.cost_matrix)

    def cost_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Generate cost curve data across a threshold range.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth labels.
        y_scores : np.ndarray
            Predicted fraud probabilities.
        amounts : np.ndarray
            Transaction amounts in INR.
        thresholds : np.ndarray | None
            Threshold values to evaluate. Defaults to 0.05 – 0.95 in 0.01 steps.

        Returns
        -------
        pd.DataFrame
            Columns: threshold, total_cost, total_savings, net_benefit.
        """
        if thresholds is None:
            thresholds = np.arange(0.05, 0.96, 0.01)

        rows = []
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            result = calculate_cost(y_true, preds, amounts, self.cost_matrix)
            rows.append(
                {
                    "threshold": float(t),
                    "total_cost": result["total_cost"],
                    "total_savings": result["total_savings"],
                    "net_benefit": result["net_benefit"],
                }
            )

        return pd.DataFrame(rows)

    def compare_strategies(
        self, y_true: np.ndarray, y_scores: np.ndarray, amounts: np.ndarray
    ) -> dict:
        """Compare default, cost-optimized, and F1-optimized thresholds.

        Returns
        -------
        dict
            Keys: default, cost_optimized, f1_optimized, each mapping to
            {'threshold': float, 'cost_breakdown': dict}.
        """
        strategies = {}
        for mode, label in [
            ("default", "default"),
            ("cost_optimized", "cost_optimized"),
            ("f1_optimized", "f1_optimized"),
        ]:
            t = optimize_threshold(y_true, y_scores, amounts, mode, self.cost_matrix)
            preds = (y_scores >= t).astype(int)
            breakdown = calculate_cost(y_true, preds, amounts, self.cost_matrix)
            strategies[label] = {"threshold": t, "cost_breakdown": breakdown}

        return strategies

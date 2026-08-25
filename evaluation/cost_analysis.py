from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score

from models.cost_matrix import COST_MATRIX, calculate_cost, optimize_threshold


class EvaluationCostAnalyzer:
    """Cost analysis for model evaluation with visualizations."""

    def __init__(self, cost_matrix: dict | None = None) -> None:
        self.cost_matrix = cost_matrix or COST_MATRIX

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
    ) -> dict:
        """Full cost analysis with breakdown.

        Returns
        -------
        dict
            total_cost, total_savings, net_benefit, cost_per_txn,
            savings_rate, cost_breakdown (FN, FP, TP, TN).
        """
        result = calculate_cost(y_true, y_pred, amounts, self.cost_matrix)

        n = len(y_true)
        cost_per_txn = result["total_cost"] / n if n > 0 else 0.0
        baseline_cost = float(amounts[y_true == 1].sum())
        savings_rate = (
            result["total_savings"] / baseline_cost if baseline_cost > 0 else 0.0
        )

        return {
            "total_cost": result["total_cost"],
            "total_savings": result["total_savings"],
            "net_benefit": result["net_benefit"],
            "cost_per_txn": cost_per_txn,
            "savings_rate": savings_rate,
            "cost_breakdown": {
                "false_negative": result["total_fn_cost"],
                "false_positive": result["total_fp_cost"],
                "true_positive": result["total_tp_cost"],
                "true_negative": result["total_tn_cost"],
            },
        }

    def cost_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
    ) -> pd.DataFrame:
        """Generate cost curve data across threshold range.

        Returns
        -------
        pd.DataFrame
            Columns: threshold, total_cost, total_savings, net_benefit.
        """
        thresholds = np.arange(0.1, 0.91, 0.01)
        rows = []
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            result = calculate_cost(y_true, preds, amounts, self.cost_matrix)
            rows.append(
                {
                    "threshold": round(float(t), 2),
                    "total_cost": result["total_cost"],
                    "total_savings": result["total_savings"],
                    "net_benefit": result["net_benefit"],
                }
            )
        return pd.DataFrame(rows)

    def compare_strategies(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
    ) -> dict:
        """Compare default, cost-optimized, F1-optimized thresholds.

        Returns
        -------
        dict
            Each strategy maps to {'threshold': float, 'metrics': dict}.
        """
        strategies = {}
        for mode, label in [
            ("default", "default"),
            ("cost_optimized", "cost_optimized"),
            ("f1_optimized", "f1_optimized"),
        ]:
            t = optimize_threshold(y_true, y_scores, amounts, mode, self.cost_matrix)
            preds = (y_scores >= t).astype(int)
            metrics = self.analyze(y_true, preds, amounts)
            f1 = f1_score(y_true, preds, zero_division=0)
            tp = int(((y_true == 1) & (preds == 1)).sum())
            fp = int(((y_true == 0) & (preds == 1)).sum())
            fn = int(((y_true == 1) & (preds == 0)).sum())
            tn = int(((y_true == 0) & (preds == 0)).sum())

            strategies[label] = {
                "threshold": t,
                "metrics": metrics,
                "f1_score": f1,
                "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            }

        return strategies

    def savings_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        amounts: np.ndarray,
    ) -> dict:
        """Generate INR savings report vs no-model baseline.

        Returns
        -------
        dict
            baseline_cost, model_cost, savings, savings_pct, roi.
        """
        baseline_cost = float(amounts[y_true == 1].sum())
        analysis = self.analyze(y_true, y_pred, amounts)
        model_cost = analysis["total_cost"]
        savings = baseline_cost - model_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        roi = (savings / model_cost) if model_cost > 0 else float("inf")

        return {
            "baseline_cost": baseline_cost,
            "model_cost": model_cost,
            "savings": savings,
            "savings_pct": savings_pct,
            "roi": roi,
        }

    def plot_cost_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
        save_path: str = "evaluation/reports/cost_curve.png",
    ) -> str:
        """Plot and save cost curve visualization.

        Returns the save path.
        """
        df = self.cost_curve(y_true, y_scores, amounts)
        optimal_idx = df["net_benefit"].idxmax()
        optimal_row = df.loc[optimal_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        ax.plot(df["threshold"], df["total_cost"], label="Total Cost", linewidth=2)
        ax.plot(df["threshold"], df["total_savings"], label="Total Savings", linewidth=2)
        ax.plot(df["threshold"], df["net_benefit"], label="Net Benefit", linewidth=2, linestyle="--")

        ax.axvline(
            x=optimal_row["threshold"],
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f'Optimal Threshold: {optimal_row["threshold"]:.2f}',
        )
        ax.scatter(
            [optimal_row["threshold"]],
            [optimal_row["net_benefit"]],
            color="red",
            s=100,
            zorder=5,
        )

        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Cost (INR)", fontsize=12)
        ax.set_title("Cost Curve Across Classification Thresholds", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xlim(0.1, 0.9)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_threshold_comparison(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: np.ndarray,
        save_path: str = "evaluation/reports/threshold_comparison.png",
    ) -> str:
        """Plot and save threshold strategy comparison.

        Returns the save path.
        """
        strategies = self.compare_strategies(y_true, y_scores, amounts)

        labels = list(strategies.keys())
        total_costs = [strategies[s]["metrics"]["total_cost"] for s in labels]
        total_savings = [strategies[s]["metrics"]["total_savings"] for s in labels]
        net_benefits = [strategies[s]["metrics"]["net_benefit"] for s in labels]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        bars1 = ax.bar(x - width, total_costs, width, label="Total Cost", color="#e74c3c")
        bars2 = ax.bar(x, total_savings, width, label="Total Savings", color="#2ecc71")
        bars3 = ax.bar(x + width, net_benefits, width, label="Net Benefit", color="#3498db")

        ax.set_xlabel("Strategy", fontsize=12)
        ax.set_ylabel("Amount (INR)", fontsize=12)
        ax.set_title("Threshold Strategy Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in labels], fontsize=10)
        ax.legend(fontsize=10)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"Rs.{height:,.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path


if __name__ == "__main__":
    print("=" * 60)
    print("AI Risk Manager - Cost Analysis Module")
    print("=" * 60)

    data_path = "data/transactions.csv"
    model_path = "models/artifacts/stage1_model.pkl"

    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} transactions")

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_engine = artifact["feature_engine"]

    X = feature_engine.transform(df)
    feature_names = artifact["feature_names"]
    X = X[feature_names].values

    y_true = df["chargeback_label"].astype(int).values
    amounts = df["amount"].values

    y_scores = model.predict_proba(X)[:, 1]
    y_pred = (y_scores >= 0.5).astype(int)

    print("\nRunning cost analysis...")
    analyzer = EvaluationCostAnalyzer()

    analysis = analyzer.analyze(y_true, y_pred, amounts)
    print(f"\n  Total Cost:       INR {analysis['total_cost']:,.2f}")
    print(f"  Total Savings:    INR {analysis['total_savings']:,.2f}")
    print(f"  Net Benefit:      INR {analysis['net_benefit']:,.2f}")
    print(f"  Cost per Txn:     INR {analysis['cost_per_txn']:,.2f}")
    print(f"  Savings Rate:     {analysis['savings_rate']:.2%}")

    print("\n  Cost Breakdown:")
    for k, v in analysis["cost_breakdown"].items():
        print(f"    {k:20s}: INR {v:,.2f}")

    print("\nComparing threshold strategies...")
    strategies = analyzer.compare_strategies(y_true, y_scores, amounts)
    for name, data in strategies.items():
        print(f"\n  [{name}] threshold={data['threshold']:.2f}  "
              f"F1={data['f1_score']:.4f}  "
              f"Net Benefit=INR {data['metrics']['net_benefit']:,.2f}")

    print("\nGenerating savings report...")
    savings = analyzer.savings_report(y_true, y_pred, amounts)
    print(f"\n  Baseline Cost:    INR {savings['baseline_cost']:,.2f}")
    print(f"  Model Cost:       INR {savings['model_cost']:,.2f}")
    print(f"  Savings:          INR {savings['savings']:,.2f}")
    print(f"  Savings %:        {savings['savings_pct']:.2f}%")
    print(f"  ROI:              {savings['roi']:.2f}x")

    print("\nGenerating plots...")
    curve_path = analyzer.plot_cost_curve(y_true, y_scores, amounts)
    print(f"  Cost curve saved to: {curve_path}")

    comparison_path = analyzer.plot_threshold_comparison(y_true, y_scores, amounts)
    print(f"  Threshold comparison saved to: {comparison_path}")

    print("\nDone!")

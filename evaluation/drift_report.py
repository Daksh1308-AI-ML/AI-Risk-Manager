"""Drift impact report generator for AI Risk Manager.

Compares static vs adaptive model performance across 12 months of
concept-drift data and produces visualisations and a text summary.

Outputs are saved to ``evaluation/reports/``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from models.cost_matrix import calculate_cost
from models.feature_engine import FEATURES, FeatureEngine
from models.stage1_risk_scorer import Stage1RiskScorer

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

QUARTER_MAP = {m: f"Q{(m - 1) // 3 + 1}" for m in range(1, 13)}
SCENARIO_LABELS = {
    1: "Baseline", 2: "Baseline", 3: "Baseline",
    4: "Seasonal", 5: "Seasonal", 6: "Seasonal",
    7: "Adversarial", 8: "Adversarial", 9: "Adversarial",
    10: "Recovery", 11: "Recovery", 12: "Recovery",
}


class DriftReportGenerator:
    """Generate drift impact analysis reports and visualisations."""

    # ------------------------------------------------------------------
    # 1. Monthly summary
    # ------------------------------------------------------------------

    @staticmethod
    def generate_monthly_report(drift_data: pd.DataFrame) -> pd.DataFrame:
        """Compute per-month statistics from drift transaction data.

        Parameters
        ----------
        drift_data : pd.DataFrame
            Full 12-month drift dataset with at least ``month``,
            ``chargeback_label``, and ``amount`` columns.

        Returns
        -------
        pd.DataFrame
            One row per month with columns: month, fraud_rate, avg_amount,
            txn_count, drift_scenario.
        """
        rows = []
        for month, grp in drift_data.groupby("month"):
            rows.append({
                "month": int(month),
                "fraud_rate": float(grp["chargeback_label"].mean()),
                "avg_amount": float(grp["amount"].mean()),
                "txn_count": int(len(grp)),
                "drift_scenario": grp["drift_scenario"].iloc[0]
                if "drift_scenario" in grp.columns
                else SCENARIO_LABELS.get(int(month), "unknown"),
            })
        return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Static vs adaptive comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_static_vs_adaptive(
        baseline_data: pd.DataFrame,
        drift_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluate static and adaptive models month-by-month.

        **Static** – trained once on months 1-3, evaluated on every month.
        **Adaptive** – retrained cumulatively after each month, evaluated
        on the *next* month (simulating a real-world retraining cadence).

        Parameters
        ----------
        baseline_data : pd.DataFrame
            Data from months 1-3 used to train the initial static model.
        drift_data : pd.DataFrame
            All 12 months of drift data (superset of baseline_data).

        Returns
        -------
        pd.DataFrame
            Columns: month, static_f1, static_precision, static_recall,
            static_cost, adaptive_f1, adaptive_precision, adaptive_recall,
            adaptive_cost.
        """
        all_months = sorted(drift_data["month"].unique())

        # --- Static model (trained once on months 1-3) ---
        print("[DriftReport] Training static model on months 1-3 ...")
        static_fe = FeatureEngine()
        static_fe.fit(baseline_data)
        static_s1 = Stage1RiskScorer(threshold_mode="cost_optimized")
        static_s1.train(baseline_data, static_fe)
        print("[DriftReport] Static model trained.")

        # --- Evaluate static on every month ---
        static_monthly: dict[int, dict] = {}
        for month in all_months:
            month_df = drift_data[drift_data["month"] == month].reset_index(drop=True)
            static_monthly[month] = DriftReportGenerator._evaluate_model(
                static_s1, static_fe, month_df,
            )

        # --- Adaptive model (retrained cumulatively) ---
        adaptive_monthly: dict[int, dict] = {}
        cumulative_train = pd.DataFrame()
        previous_months = []

        for month in all_months:
            month_df = drift_data[drift_data["month"] == month].reset_index(drop=True)

            if previous_months:
                print(
                    f"[DriftReport] Training adaptive model on months "
                    f"{previous_months[0]}-{previous_months[-1]} ..."
                )
                adapt_fe = FeatureEngine()
                adapt_fe.fit(cumulative_train)
                adapt_s1 = Stage1RiskScorer(threshold_mode="cost_optimized")
                adapt_s1.train(cumulative_train, adapt_fe)
                adaptive_monthly[month] = DriftReportGenerator._evaluate_model(
                    adapt_s1, adapt_fe, month_df,
                )
            else:
                adaptive_monthly[month] = {
                    "f1": 0.0, "precision": 0.0, "recall": 0.0, "cost": 0.0,
                }

            cumulative_train = pd.concat(
                [cumulative_train, month_df], ignore_index=True,
            )
            previous_months.append(month)

        print("[DriftReport] Adaptive evaluation complete.")

        # --- Assemble result ---
        records = []
        for month in all_months:
            records.append({
                "month": month,
                "drift_scenario": SCENARIO_LABELS.get(month, "unknown"),
                "static_f1": static_monthly[month]["f1"],
                "static_precision": static_monthly[month]["precision"],
                "static_recall": static_monthly[month]["recall"],
                "static_cost": static_monthly[month]["cost"],
                "adaptive_f1": adaptive_monthly[month]["f1"],
                "adaptive_precision": adaptive_monthly[month]["precision"],
                "adaptive_recall": adaptive_monthly[month]["recall"],
                "adaptive_cost": adaptive_monthly[month]["cost"],
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 3. Drift decay line plot
    # ------------------------------------------------------------------

    @staticmethod
    def plot_drift_decay(monthly_metrics: pd.DataFrame) -> Path:
        """Line plot: F1 score over months for static vs adaptive.

        Parameters
        ----------
        monthly_metrics : pd.DataFrame
            Output of ``compare_static_vs_adaptive``.

        Returns
        -------
        Path
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        months = monthly_metrics["month"].values
        ax.plot(months, monthly_metrics["static_f1"], "o-", label="Static", linewidth=2)
        ax.plot(months, monthly_metrics["adaptive_f1"], "s-", label="Adaptive", linewidth=2)

        # Shade drift regions
        boundaries = [(4, 6, "Seasonal", "#ffcc00"), (7, 9, "Adversarial", "#ff6666"), (10, 12, "Recovery", "#66cc66")]
        for start, end, label, color in boundaries:
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=color, label=label)

        ax.set_xlabel("Month")
        ax.set_ylabel("F1 Score")
        ax.set_title("Model Performance Decay Under Concept Drift")
        ax.set_xticks(range(1, 13))
        ax.set_ylim(bottom=0)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

        path = REPORTS_DIR / "drift_decay.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[DriftReport] Saved drift decay plot -> {path}")
        return path

    # ------------------------------------------------------------------
    # 4. Fraud rate evolution bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_fraud_rate_evolution(drift_data: pd.DataFrame) -> Path:
        """Bar chart: fraud rate per month.

        Parameters
        ----------
        drift_data : pd.DataFrame
            Full drift dataset.

        Returns
        -------
        Path
            Path to the saved PNG.
        """
        monthly = DriftReportGenerator.generate_monthly_report(drift_data)

        scenario_colors = {
            "Baseline": "#4a90d9", "Seasonal": "#ffcc00",
            "Adversarial": "#ff6666", "Recovery": "#66cc66",
        }
        colors = [scenario_colors.get(s, "#999999") for s in monthly["drift_scenario"]]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(monthly["month"], monthly["fraud_rate"] * 100, color=colors, edgecolor="white")
        for bar, rate in zip(bars, monthly["fraud_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{rate:.1%}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Month")
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title("Fraud Rate Evolution Across Drift Scenarios")
        ax.set_xticks(range(1, 13))
        ax.grid(axis="y", alpha=0.3)

        # Legend from unique scenarios
        seen = set()
        for scenario, color in scenario_colors.items():
            if scenario not in seen:
                ax.bar([], [], color=color, label=scenario)
                seen.add(scenario)
        ax.legend()

        path = REPORTS_DIR / "fraud_rate.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[DriftReport] Saved fraud rate plot -> {path}")
        return path

    # ------------------------------------------------------------------
    # 5. Cost impact grouped bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def plot_cost_impact(comparison: pd.DataFrame) -> Path:
        """Grouped bar chart: total cost per quarter for static vs adaptive.

        Parameters
        ----------
        comparison : pd.DataFrame
            Output of ``compare_static_vs_adaptive``.

        Returns
        -------
        Path
            Path to the saved PNG.
        """
        comparison = comparison.copy()
        comparison["quarter"] = comparison["month"].map(QUARTER_MAP)

        quarterly = comparison.groupby("quarter").agg(
            static_cost=("static_cost", "sum"),
            adaptive_cost=("adaptive_cost", "sum"),
        ).reset_index()

        x = np.arange(len(quarterly))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, quarterly["static_cost"], width, label="Static", color="#4a90d9")
        ax.bar(x + width / 2, quarterly["adaptive_cost"], width, label="Adaptive", color="#66cc66")

        ax.set_xlabel("Quarter")
        ax.set_ylabel("Total Cost (INR)")
        ax.set_title("Financial Impact: Static vs Adaptive Model")
        ax.set_xticks(x)
        ax.set_xticklabels(quarterly["quarter"])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.ticklabel_format(style="plain", axis="y")

        path = REPORTS_DIR / "cost_impact.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[DriftReport] Saved cost impact plot -> {path}")
        return path

    # ------------------------------------------------------------------
    # 6. Full report orchestrator
    # ------------------------------------------------------------------

    @staticmethod
    def generate_full_report(drift_data: pd.DataFrame) -> Path:
        """Run all analyses and write a consolidated text report.

        Parameters
        ----------
        drift_data : pd.DataFrame
            Full 12-month drift dataset.

        Returns
        -------
        Path
            Path to the generated ``drift_report.txt``.
        """
        lines: list[str] = []
        sep = "=" * 72

        lines.append(sep)
        lines.append("  DRIFT IMPACT REPORT — AI Risk Manager")
        lines.append(sep)
        lines.append("")

        # --- 1. Monthly summary ---
        monthly_report = DriftReportGenerator.generate_monthly_report(drift_data)
        lines.append("1. MONTHLY SUMMARY")
        lines.append("-" * 72)
        lines.append(
            f"{'Month':<8}{'Scenario':<14}{'Fraud %':<10}"
            f"{'Avg Amount':>12}{'Txn Count':>10}"
        )
        lines.append("-" * 72)
        for _, row in monthly_report.iterrows():
            lines.append(
                f"{int(row['month']):<8}{row['drift_scenario']:<14}"
                f"{row['fraud_rate']:<10.2%}INR {row['avg_amount']:>8,.0f}"
                f"{int(row['txn_count']):>10}"
            )
        lines.append("")

        # --- 2. Static vs Adaptive ---
        comparison = DriftReportGenerator.compare_static_vs_adaptive(
            baseline_data=drift_data[drift_data["month"].isin([1, 2, 3])].reset_index(drop=True),
            drift_data=drift_data,
        )
        lines.append("2. STATIC VS ADAPTIVE MODEL COMPARISON")
        lines.append("-" * 72)
        lines.append(
            f"{'Month':<8}{'Static F1':>10}{'Adapt F1':>10}"
            f"{'Delta':>8}{'Static Cost':>14}{'Adapt Cost':>14}"
        )
        lines.append("-" * 72)
        for _, row in comparison.iterrows():
            delta = row["adaptive_f1"] - row["static_f1"]
            lines.append(
                f"{int(row['month']):<8}"
                f"{row['static_f1']:>10.4f}{row['adaptive_f1']:>10.4f}"
                f"{delta:>+8.4f}"
                f"INR {row['static_cost']:>10,.0f}"
                f"INR {row['adaptive_cost']:>10,.0f}"
            )
        lines.append("")

        # --- Summary statistics ---
        static_avg_f1 = comparison["static_f1"].mean()
        adaptive_avg_f1 = comparison["adaptive_f1"].mean()
        total_static_cost = comparison["static_cost"].sum()
        total_adaptive_cost = comparison["adaptive_cost"].sum()
        cost_savings = total_static_cost - total_adaptive_cost

        lines.append("3. SUMMARY STATISTICS")
        lines.append("-" * 72)
        lines.append(f"  Avg Static F1:         {static_avg_f1:.4f}")
        lines.append(f"  Avg Adaptive F1:       {adaptive_avg_f1:.4f}")
        lines.append(f"  F1 Improvement:        {adaptive_avg_f1 - static_avg_f1:+.4f}")
        lines.append(f"  Total Static Cost:     INR {total_static_cost:>12,.0f}")
        lines.append(f"  Total Adaptive Cost:   INR {total_adaptive_cost:>12,.0f}")
        lines.append(f"  Net Cost Savings:      INR {cost_savings:>12,.0f}")
        lines.append("")

        # --- Quarter breakdown ---
        comparison_copy = comparison.copy()
        comparison_copy["quarter"] = comparison_copy["month"].map(QUARTER_MAP)
        quarterly = comparison_copy.groupby("quarter").agg(
            static_cost=("static_cost", "sum"),
            adaptive_cost=("adaptive_cost", "sum"),
        ).reset_index()
        quarterly["savings"] = quarterly["static_cost"] - quarterly["adaptive_cost"]

        lines.append("4. QUARTERLY COST BREAKDOWN")
        lines.append("-" * 72)
        lines.append(
            f"{'Quarter':<10}{'Static Cost':>14}{'Adaptive Cost':>14}{'Savings':>14}"
        )
        lines.append("-" * 72)
        for _, row in quarterly.iterrows():
            lines.append(
                f"{row['quarter']:<10}"
                f"INR {row['static_cost']:>10,.0f}"
                f"INR {row['adaptive_cost']:>10,.0f}"
                f"INR {row['savings']:>10,.0f}"
            )
        lines.append("")

        lines.append(sep)
        lines.append("  END OF REPORT")
        lines.append(sep)

        # --- Write text report ---
        report_text = "\n".join(lines)
        report_path = REPORTS_DIR / "drift_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        print(f"[DriftReport] Text report saved -> {report_path}")

        # --- Generate plots ---
        DriftReportGenerator.plot_drift_decay(comparison)
        DriftReportGenerator.plot_fraud_rate_evolution(drift_data)
        DriftReportGenerator.plot_cost_impact(comparison)

        return report_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_model(
        scorer: Stage1RiskScorer,
        feature_engine: FeatureEngine,
        data: pd.DataFrame,
    ) -> dict:
        """Evaluate a Stage1RiskScorer on a dataset and return key metrics."""
        features_df = feature_engine.transform(data)
        X = features_df[FEATURES].values
        y_true = data["chargeback_label"].astype(int).values
        amounts = data["amount"].values.astype(np.float64)

        y_proba = scorer.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= scorer.threshold).astype(int)

        cost = calculate_cost(y_true, y_pred, amounts)
        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "cost": cost["total_cost"],
        }


if __name__ == "__main__":
    data_path = Path("data/drift_transactions.csv")
    if not data_path.exists():
        print(f"[ERROR] {data_path} not found. Run data/drift.py first.")
        raise SystemExit(1)

    print("[DriftReport] Loading drift data ...")
    df = pd.read_csv(data_path)
    print(f"[DriftReport] Loaded {len(df)} rows, {df['month'].nunique()} months.")

    report_path = DriftReportGenerator.generate_full_report(df)
    print(f"\n[DriftReport] Done. Report at: {report_path}")

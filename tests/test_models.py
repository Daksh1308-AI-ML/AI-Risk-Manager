"""Tests for model modules: cost_matrix, feature_engine, stage1, stage2."""

import pickle
import pytest
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, ".")

from models.cost_matrix import (
    COST_MATRIX,
    calculate_cost,
    optimize_threshold,
    CostAnalyzer,
    RBI_ZERO_LIABILITY_THRESHOLD,
    RBI_MAX_COMPENSATION,
    RBI_COMPENSATION_RATE,
)
from models.feature_engine import FeatureEngine, FEATURES
from models.stage1_risk_scorer import Stage1RiskScorer, THRESHOLDS
from models.stage2_fraud_classifier import (
    Stage2FraudClassifier,
    EVIDENCE_CHECKLISTS,
)
from data.schema import FraudLabel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Small synthetic transaction DataFrame for FeatureEngine and model tests."""
    n = 50
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="15min")
    customer_ids = rng.choice(["C001", "C002", "C003"], size=n)
    device_fps = rng.choice(["DEV_A", "DEV_B"], size=n)

    fraud_mask = rng.rand(n) < 0.3
    fraud_types = np.where(
        fraud_mask,
        rng.choice(
            ["friendly_fraud", "account_takeover", "technical_failure"],
            size=n,
        ),
        "genuine",
    )

    return pd.DataFrame({
        "transaction_id": [f"TXN{i:04d}" for i in range(n)],
        "timestamp": timestamps,
        "amount": rng.uniform(500, 50000, size=n).round(2),
        "payment_method": rng.choice(["upi", "credit_card"], size=n),
        "merchant_category": rng.choice(["electronics", "grocery"], size=n),
        "customer_id": customer_ids,
        "device_fingerprint": device_fps,
        "ip_address": [f"10.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}" for _ in range(n)],
        "is_new_device": rng.choice([0, 1], size=n),
        "is_new_address": rng.choice([0, 1], size=n),
        "account_age_days": rng.randint(1, 1000, size=n),
        "past_disputes": rng.randint(0, 5, size=n),
        "chargeback_label": fraud_mask.astype(bool),
        "fraud_type": fraud_types,
        "chargeback_reason": rng.choice(["not_received", "unauthorized", "duplicate"], size=n),
    })


@pytest.fixture
def fitted_engine(synthetic_df):
    """Fitted FeatureEngine instance."""
    engine = FeatureEngine()
    engine.fit(synthetic_df)
    return engine


@pytest.fixture
def features_df(fitted_engine, synthetic_df):
    """Transformed feature DataFrame."""
    return fitted_engine.transform(synthetic_df)


@pytest.fixture
def y_true():
    return np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])


@pytest.fixture
def y_pred():
    return np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])


@pytest.fixture
def amounts():
    return np.array([10000, 5000, 2000, 8000, 15000, 3000, 12000, 7000, 1000, 25000])


# ---------------------------------------------------------------------------
# Cost Matrix Tests
# ---------------------------------------------------------------------------

class TestCostMatrix:
    """Tests for cost matrix module."""

    def test_cost_matrix_structure(self):
        assert set(COST_MATRIX.keys()) == {
            "false_negative", "false_positive", "true_positive", "true_negative"
        }
        for quadrant in ["false_negative", "false_positive", "true_positive"]:
            assert isinstance(COST_MATRIX[quadrant], dict)
            assert len(COST_MATRIX[quadrant]) > 0
        assert COST_MATRIX["true_negative"]["cost"] == 0

    def test_rbi_constants(self):
        assert RBI_ZERO_LIABILITY_THRESHOLD == 50000
        assert RBI_MAX_COMPENSATION == 25000
        assert RBI_COMPENSATION_RATE == 0.85

    def test_calculate_cost_all_tp(self, y_true, amounts):
        y_pred = y_true.copy()
        result = calculate_cost(y_true, y_pred, amounts)

        tp_count = int((y_true == 1).sum())
        expected_tp = COST_MATRIX["true_positive"]["verification_cost"] * tp_count
        assert result["total_tp_cost"] == pytest.approx(expected_tp)
        assert result["total_fn_cost"] == pytest.approx(0.0)
        assert result["total_fp_cost"] == pytest.approx(0.0)
        assert result["total_tn_cost"] == pytest.approx(0.0)

    def test_calculate_cost_all_fp(self, y_true, amounts):
        y_pred = np.ones_like(y_true)
        result = calculate_cost(y_true, y_pred, amounts)

        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_amounts = amounts[fp_mask]
        fp = COST_MATRIX["false_positive"]
        lost_sale = (fp_amounts * fp["lost_sale_probability"]).sum()
        churn = (fp_amounts * fp["churn_probability"] * fp["churn_ltv_cost"]).sum()
        inv = fp["investigation_time_minutes"] / 60 * fp["hourly_rate"] * int(fp_mask.sum())
        review = fp["manual_review_cost"] * int(fp_mask.sum())
        expected_fp = lost_sale + churn + inv + review
        assert result["total_fp_cost"] == pytest.approx(expected_fp)

    def test_calculate_cost_all_fn(self, y_true, amounts):
        y_pred = np.zeros_like(y_true)
        result = calculate_cost(y_true, y_pred, amounts)

        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_amounts = amounts[fn_mask]
        fn = COST_MATRIX["false_negative"]
        chargeback = fn_amounts.sum()
        churn = (fn_amounts * fn["churn_probability"] * fn["churn_ltv_cost"]).sum()
        rbi = (fn_amounts * fn["rbi_penalty_probability"] * fn["rbi_penalty_amount"]).sum()
        fees = (fn["processing_fee"] + fn["operational_cost"]) * int(fn_mask.sum())
        expected_fn = chargeback + fees + churn + rbi
        assert result["total_fn_cost"] == pytest.approx(expected_fn)

    def test_calculate_cost_mixed(self, y_true, y_pred, amounts):
        result = calculate_cost(y_true, y_pred, amounts)

        assert result["total_cost"] == pytest.approx(
            result["total_fn_cost"] + result["total_fp_cost"] + result["total_tp_cost"]
        )
        assert result["net_benefit"] == pytest.approx(
            result["total_savings"] - result["total_cost"]
        )
        assert result["total_tn_cost"] == pytest.approx(0.0)
        for key in ["total_fn_cost", "total_fp_cost", "total_tp_cost", "total_savings"]:
            assert result[key] >= 0

    def test_calculate_cost_returns_dict_keys(self, y_true, y_pred, amounts):
        result = calculate_cost(y_true, y_pred, amounts)
        expected_keys = {
            "total_fn_cost", "total_fp_cost", "total_tp_cost", "total_tn_cost",
            "total_cost", "total_savings", "net_benefit",
        }
        assert set(result.keys()) == expected_keys

    def test_optimize_threshold_default(self, y_true, amounts):
        scores = np.random.RandomState(0).rand(len(y_true))
        threshold = optimize_threshold(y_true, scores, amounts, mode="default")
        assert threshold == 0.5

    def test_optimize_threshold_cost(self, y_true, amounts):
        scores = np.random.RandomState(1).rand(len(y_true))
        threshold = optimize_threshold(y_true, scores, amounts, mode="cost_optimized")
        assert 0.1 <= threshold <= 0.91

    def test_optimize_threshold_f1(self, y_true, amounts):
        scores = np.random.RandomState(2).rand(len(y_true))
        threshold = optimize_threshold(y_true, scores, amounts, mode="f1_optimized")
        assert 0.1 <= threshold <= 0.91

    def test_optimize_threshold_cost_minimizes(self, y_true, amounts):
        scores = np.random.RandomState(3).rand(len(y_true))
        best_t = optimize_threshold(y_true, scores, amounts, mode="cost_optimized")

        best_result = calculate_cost(y_true, (scores >= best_t).astype(int), amounts)
        baseline_result = calculate_cost(y_true, (scores >= 0.5).astype(int), amounts)
        assert best_result["total_cost"] <= baseline_result["total_cost"]

    def test_cost_analyzer_init(self):
        analyzer = CostAnalyzer()
        assert analyzer.cost_matrix is COST_MATRIX

    def test_cost_analyzer_analyze(self, y_true, y_pred, amounts):
        analyzer = CostAnalyzer()
        result = analyzer.analyze(y_true, y_pred, amounts)
        expected = calculate_cost(y_true, y_pred, amounts)
        assert result == expected

    def test_cost_analyzer_cost_curve(self, y_true, amounts):
        scores = np.random.RandomState(4).rand(len(y_true))
        analyzer = CostAnalyzer()
        curve = analyzer.cost_curve(y_true, scores, amounts, thresholds=np.arange(0.1, 0.9, 0.1))
        assert isinstance(curve, pd.DataFrame)
        assert set(curve.columns) == {"threshold", "total_cost", "total_savings", "net_benefit"}
        assert len(curve) == 8

    def test_cost_analyzer_compare_strategies(self, y_true, amounts):
        scores = np.random.RandomState(5).rand(len(y_true))
        analyzer = CostAnalyzer()
        strategies = analyzer.compare_strategies(y_true, scores, amounts)
        assert set(strategies.keys()) == {"default", "cost_optimized", "f1_optimized"}
        for key in strategies:
            assert "threshold" in strategies[key]
            assert "cost_breakdown" in strategies[key]


# ---------------------------------------------------------------------------
# Feature Engine Tests
# ---------------------------------------------------------------------------

class TestFeatureEngine:
    """Tests for feature engineering pipeline."""

    def test_fit_transform_output_shape(self, fitted_engine, synthetic_df):
        result = fitted_engine.transform(synthetic_df)
        assert result.shape[0] == len(synthetic_df)

    def test_all_features_present(self, fitted_engine, synthetic_df):
        result = fitted_engine.transform(synthetic_df)
        for feat in FEATURES:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_feature_names(self, fitted_engine):
        names = fitted_engine.get_feature_names()
        assert names == FEATURES
        assert len(names) == 20

    def test_no_nulls_in_output(self, fitted_engine, synthetic_df):
        result = fitted_engine.transform(synthetic_df)
        assert result[FEATURES].isnull().sum().sum() == 0

    def test_fit_transform_deterministic(self, synthetic_df):
        engine1 = FeatureEngine()
        out1 = engine1.fit_transform(synthetic_df)
        engine2 = FeatureEngine()
        out2 = engine2.fit_transform(synthetic_df)
        pd.testing.assert_frame_equal(out1[FEATURES], out2[FEATURES])

    def test_unfitted_raises_error(self, synthetic_df):
        engine = FeatureEngine()
        with pytest.raises(RuntimeError, match="not been fitted"):
            engine.transform(synthetic_df)

    def test_target_columns_preserved(self, fitted_engine, synthetic_df):
        result = fitted_engine.transform(synthetic_df)
        for col in ["chargeback_label", "fraud_type", "transaction_id"]:
            assert col in result.columns
        result_sorted = result.sort_values("transaction_id").reset_index(drop=True)
        orig_sorted = synthetic_df.sort_values("transaction_id").reset_index(drop=True)
        for col in ["chargeback_label", "fraud_type", "transaction_id"]:
            assert result_sorted[col].tolist() == orig_sorted[col].tolist()

    def test_custom_feature_names(self, synthetic_df):
        subset = FEATURES[:5]
        engine = FeatureEngine(feature_names=subset)
        result = engine.fit_transform(synthetic_df)
        assert list(result.columns[:5]) == subset

    def test_fit_returns_self(self, synthetic_df):
        engine = FeatureEngine()
        ret = engine.fit(synthetic_df)
        assert ret is engine

    def test_fit_transform_matches_separate_calls(self, synthetic_df):
        engine1 = FeatureEngine()
        out1 = engine1.fit_transform(synthetic_df)
        engine2 = FeatureEngine()
        engine2.fit(synthetic_df)
        out2 = engine2.transform(synthetic_df)
        pd.testing.assert_frame_equal(out1[FEATURES], out2[FEATURES])

    def test_velocity_features_non_negative(self, fitted_engine, synthetic_df):
        result = fitted_engine.transform(synthetic_df)
        for col in ["txn_count_1h", "txn_count_24h", "txn_count_7d"]:
            assert (result[col] >= 0).all()

    def test_binary_features_zero_one(self, fitted_engine, synthetic_df):
        result = fitted_engine.transform(synthetic_df)
        for col in ["is_new_device", "is_new_address", "is_weekend", "is_night", "ip_country_match"]:
            unique_vals = set(result[col].unique())
            assert unique_vals.issubset({0.0, 1.0}), f"{col} has unexpected values: {unique_vals}"


# ---------------------------------------------------------------------------
# Stage 1 Risk Scorer Tests
# ---------------------------------------------------------------------------

class TestStage1RiskScorer:
    """Tests for Stage 1 model."""

    def test_init_default(self):
        scorer = Stage1RiskScorer()
        assert scorer.threshold_mode == "cost_optimized"
        assert scorer.threshold == THRESHOLDS["cost_optimized"]

    @pytest.mark.parametrize("mode", ["default", "cost_optimized", "f1_optimized"])
    def test_init_valid_modes(self, mode):
        scorer = Stage1RiskScorer(threshold_mode=mode)
        assert scorer.threshold_mode == mode
        assert scorer.threshold == THRESHOLDS[mode]

    def test_init_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown threshold_mode"):
            Stage1RiskScorer(threshold_mode="invalid_mode")

    def test_train_and_predict(self, synthetic_df, fitted_engine):
        scorer = Stage1RiskScorer(threshold_mode="default")
        metrics = scorer.train(synthetic_df, fitted_engine)

        assert scorer.model is not None
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert "threshold" in metrics

    def test_predict_proba_shape(self, synthetic_df, fitted_engine):
        scorer = Stage1RiskScorer(threshold_mode="default")
        scorer.train(synthetic_df, fitted_engine)

        features = fitted_engine.transform(synthetic_df)
        proba = scorer.predict_proba(features)
        assert proba.shape == (len(synthetic_df),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_predictions_binary(self, synthetic_df, fitted_engine):
        scorer = Stage1RiskScorer(threshold_mode="default")
        scorer.train(synthetic_df, fitted_engine)

        features = fitted_engine.transform(synthetic_df)
        preds = scorer.predict(features)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0, 1})

    def test_feature_importance(self, synthetic_df, fitted_engine):
        scorer = Stage1RiskScorer(threshold_mode="default")
        scorer.train(synthetic_df, fitted_engine)

        importance = scorer.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert list(importance.columns) == ["feature", "importance"]
        assert len(importance) == len(FEATURES)

    def test_untrained_predict_raises(self):
        scorer = Stage1RiskScorer()
        dummy = pd.DataFrame(columns=FEATURES)
        with pytest.raises(RuntimeError, match="not been trained"):
            scorer.predict_proba(dummy)

    def test_untrained_feature_importance_raises(self):
        scorer = Stage1RiskScorer()
        with pytest.raises(RuntimeError, match="not been trained"):
            scorer.get_feature_importance()

    def test_save_and_load(self, synthetic_df, fitted_engine, tmp_path):
        scorer = Stage1RiskScorer(threshold_mode="default")
        scorer.train(synthetic_df, fitted_engine)

        save_path = tmp_path / "stage1_test.pkl"
        scorer.save(save_path)
        assert save_path.exists()

        loaded = Stage1RiskScorer.load(save_path)
        assert loaded.threshold == scorer.threshold
        assert loaded.threshold_mode == scorer.threshold_mode

        features = fitted_engine.transform(synthetic_df)
        preds_orig = scorer.predict(features)
        preds_loaded = loaded.predict(features)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_untrained_save_raises(self):
        scorer = Stage1RiskScorer()
        with pytest.raises(RuntimeError, match="No trained model"):
            scorer.save("/tmp/test.pkl")

    def test_train_metrics_in_range(self, synthetic_df, fitted_engine):
        scorer = Stage1RiskScorer(threshold_mode="default")
        metrics = scorer.train(synthetic_df, fitted_engine)
        for key in ["precision", "recall", "f1", "auc_roc", "auc_pr"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


# ---------------------------------------------------------------------------
# Stage 2 Fraud Classifier Tests
# ---------------------------------------------------------------------------

class TestStage2FraudClassifier:
    """Tests for Stage 2 model."""

    def test_init(self):
        clf = Stage2FraudClassifier()
        assert clf.model is None
        assert clf.class_names == []
        assert clf.metrics == {}

    def test_evidence_checklist(self):
        clf = Stage2FraudClassifier()
        for fraud_type in FraudLabel:
            checklist = clf.get_evidence_checklist(fraud_type.value)
            assert isinstance(checklist, list)
            assert len(checklist) > 0

    def test_evidence_checklist_all_types(self):
        assert len(EVIDENCE_CHECKLISTS) == 4
        for label in FraudLabel:
            assert label.value in EVIDENCE_CHECKLISTS
            assert len(EVIDENCE_CHECKLISTS[label.value]) == 5

    def test_evidence_checklist_unknown_returns_empty(self):
        clf = Stage2FraudClassifier()
        assert clf.get_evidence_checklist("unknown_type") == []

    def test_train_and_predict(self, synthetic_df, fitted_engine):
        clf = Stage2FraudClassifier()
        metrics = clf.train(synthetic_df, fitted_engine)

        assert clf.model is not None
        assert "classification_report" in metrics
        assert "n_fraud_cases" in metrics
        assert "n_total_cases" in metrics
        assert metrics["n_total_cases"] == len(synthetic_df)
        assert metrics["n_fraud_cases"] == int(synthetic_df["chargeback_label"].sum())

    def test_predict_output_structure(self, synthetic_df, fitted_engine):
        clf = Stage2FraudClassifier()
        clf.train(synthetic_df, fitted_engine)

        features = fitted_engine.transform(synthetic_df)
        fraud_mask = synthetic_df["chargeback_label"].values
        result = clf.predict(features, fraud_mask=fraud_mask)

        assert set(result.keys()) == {"fraud_types", "confidences", "evidence_checklists"}
        assert len(result["fraud_types"]) == len(synthetic_df)
        assert len(result["confidences"]) == len(synthetic_df)
        assert len(result["evidence_checklists"]) == len(synthetic_df)

    def test_predict_genuine_low_confidence(self, synthetic_df, fitted_engine):
        clf = Stage2FraudClassifier()
        clf.train(synthetic_df, fitted_engine)

        features = fitted_engine.transform(synthetic_df)
        all_false = np.zeros(len(synthetic_df), dtype=bool)
        result = clf.predict(features, fraud_mask=all_false)

        assert all(t == FraudLabel.GENUINE.value for t in result["fraud_types"])
        assert all(c == 0.0 for c in result["confidences"])
        assert all(c == [] for c in result["evidence_checklists"])

    def test_untrained_predict_raises(self):
        clf = Stage2FraudClassifier()
        dummy = pd.DataFrame(columns=FEATURES)
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(dummy)

    def test_save_and_load(self, synthetic_df, fitted_engine, tmp_path):
        clf = Stage2FraudClassifier()
        clf.train(synthetic_df, fitted_engine)

        save_path = str(tmp_path / "stage2_test.pkl")
        clf.save(save_path)

        loaded = Stage2FraudClassifier.load(save_path)
        assert loaded.class_names == clf.class_names
        assert loaded.model is not None

        features = fitted_engine.transform(synthetic_df)
        fraud_mask = synthetic_df["chargeback_label"].values
        result_orig = clf.predict(features, fraud_mask=fraud_mask)
        result_loaded = loaded.predict(features, fraud_mask=fraud_mask)
        assert result_orig["fraud_types"] == result_loaded["fraud_types"]

    def test_class_names_populated_after_train(self, synthetic_df, fitted_engine):
        clf = Stage2FraudClassifier()
        clf.train(synthetic_df, fitted_engine)
        assert len(clf.class_names) > 0
        assert "genuine" in clf.class_names

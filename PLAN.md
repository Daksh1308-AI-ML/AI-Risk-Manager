# Implementation Plan — Trishul

## Overview

This document outlines the phased implementation of the Triple Threat Chargeback Defense System.

**Timeline:** 7 days
**Stack:** Python, XGBoost, FastAPI, River (drift detection)
**Data:** Synthetic (10,000+ transactions)

---

## Phase 1: Foundation (Day 1)

### 1.1 Project Setup
- [ ] Initialize project structure
- [ ] Create `requirements.txt`
- [ ] Set up `.gitignore`
- [ ] Create `setup.py`

### 1.2 Data Schema
- [ ] Define `TransactionSchema` in `data/schema.py`
- [ ] Define `FraudLabel` enum (genuine, friendly_fraud, account_takeover, technical_failure)
- [ ] Define `ChargebackReason` enum
- [ ] Define `PaymentMethod` enum
- [ ] Define `MerchantCategory` enum

### 1.3 Synthetic Data Generator
- [ ] Implement base transaction generator (`data/generate.py`)
- [ ] Add fraud label distribution logic:
  - 60% friendly fraud
  - 25% genuine fraud
  - 10% account takeover
  - 5% technical failure
- [ ] Add behavioral features for each fraud type
- [ ] Generate 10,000 labeled transactions
- [ ] Export to CSV

**Deliverable:** `data/transactions.csv` with 10K rows

---

## Phase 2: Cost Matrix + Stage 1 Model (Day 2)

### 2.1 Cost Matrix
- [ ] Define RBI-aligned cost components (`models/cost_matrix.py`)
- [ ] Implement cost calculator function
- [ ] Add threshold optimization logic
- [ ] Define cost curves for FN, FP, TP, TN

### 2.2 Feature Engineering
- [ ] Implement `FeatureEngine` class (`models/feature_engine.py`)
- [ ] Add transaction velocity features (1h, 24h, 7d)
- [ ] Add amount deviation from customer norm
- [ ] Add device trust scoring
- [ ] Add geographic velocity
- [ ] Add account age risk factor
- [ ] Add historical dispute rate

### 2.3 Stage 1: Risk Scorer
- [ ] Implement XGBoost classifier (`models/stage1_risk_scorer.py`)
- [ ] Train on synthetic data
- [ ] Implement threshold optimization:
  - Default (0.5)
  - Cost-optimized (minimize total cost)
  - F1-optimized (maximize F1)
- [ ] Save model artifacts

**Deliverable:** Trained Stage 1 model + cost matrix

---

## Phase 3: Stage 2 Classifier + Pipeline (Day 3)

### 3.1 Stage 2: Fraud Type Classifier
- [ ] Implement Random Forest classifier (`models/stage2_fraud_classifier.py`)
- [ ] Add post-purchase behavioral features:
  - Login after purchase
  - Support contact
  - Return request
  - Account activity level
- [ ] Train 4-class classifier
- [ ] Generate classification report

### 3.2 Inference Pipeline
- [ ] Create unified inference class
- [ ] Stage 1 → Stage 2 flow
- [ ] Confidence calibration
- [ ] Evidence checklist generation per fraud type

**Deliverable:** End-to-end inference pipeline

---

## Phase 4: Drift Detection + Adaptive Training (Day 4)

### 4.1 Drift Simulation
- [ ] Implement time-sliced data generation (`data/drift.py`)
- [ ] Create 4 drift scenarios:
  - Month 1-3: Baseline
  - Month 4-6: Seasonal shift (Diwali)
  - Month 7-9: Adversarial shift (new technique)
  - Month 10-12: Partial recovery

### 4.2 Drift Detectors
- [ ] Implement ADWIN detector (`models/drift_detector.py`)
- [ ] Implement PSI (Population Stability Index)
- [ ] Implement Page-Hinkley test
- [ ] Add drift alerting logic

### 4.3 Adaptive Trainer
- [ ] Implement incremental learning (`models/adaptive_trainer.py`)
- [ ] Add retraining trigger logic
- [ ] Implement A/B comparison: static vs. adaptive
- [ ] Generate drift impact report

**Deliverable:** Drift detection system + adaptive retraining

---

## Phase 5: Evaluation + Cost Analysis (Day 5)

### 5.1 Standard Metrics
- [ ] Implement precision, recall, F1, AUC (`evaluation/metrics.py`)
- [ ] Add confusion matrix generation
- [ ] Add classification report

### 5.2 Cost Analysis
- [ ] Implement `CostAnalyzer` class (`evaluation/cost_analysis.py`)
- [ ] Calculate total cost across all quadrants
- [ ] Generate cost curves across threshold range
- [ ] Compare cost-optimized vs. F1-optimized thresholds

### 5.3 Held-Out Test Evaluation
- [ ] Split data: 80% train, 20% test
- [ ] Run full evaluation pipeline
- [ ] Generate comprehensive report
- [ ] Visualize results (matplotlib/seaborn)

### 5.4 Drift Report
- [ ] Implement drift decay visualization
- [ ] Compare static vs. adaptive performance
- [ ] Generate month-by-month performance report

**Deliverable:** Full evaluation report + visualizations

---

## Phase 6: FastAPI + Tests (Day 6)

### 6.1 API Setup
- [ ] Create FastAPI app (`api/main.py`)
- [ ] Define request/response schemas (`api/schemas.py`)
- [ ] Implement `/score` endpoint
- [ ] Implement `/classify` endpoint
- [ ] Implement `/evaluate` endpoint
- [ ] Implement `/drift/status` endpoint
- [ ] Implement `/drift/simulate` endpoint

### 6.2 Testing
- [ ] Unit tests for data generation (`tests/test_data.py`)
- [ ] Unit tests for models (`tests/test_models.py`)
- [ ] Integration tests for API (`tests/test_api.py`)
- [ ] Run pytest, achieve > 80% coverage

### 6.3 API Documentation
- [ ] Add OpenAPI/Swagger UI
- [ ] Add example requests/responses
- [ ] Add error handling

**Deliverable:** Working API + test suite

---

## Phase 7: Documentation + Presentation (Day 7)

### 7.1 Documentation
- [ ] Finalize README.md
- [ ] Complete ARCHITECTURE.md
- [ ] Update TODO.md with completion status
- [ ] Add inline code comments

### 7.2 Notebooks
- [ ] Create `01_data_exploration.ipynb`
- [ ] Create `02_model_training.ipynb`
- [ ] Create `03_drift_analysis.ipynb`

### 7.3 Presentation Materials
- [ ] Executive summary (1 page)
- [ ] Architecture diagram
- [ ] Key metrics dashboard
- [ ] Cost savings visualization
- [ ] Demo script

**Deliverable:** Complete documentation + presentation

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Precision | > 85% |
| Recall | > 80% |
| F1-Score | > 0.82 |
| Cost Savings vs. No Model | > 70% |
| Drift Resilience (month 12) | < 10% degradation |
| API Response Time | < 100ms |
| Test Coverage | > 80% |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Synthetic data unrealistic | Follow Indian e-commerce patterns, validate distributions |
| Drift simulation too artificial | Use realistic scenario progression |
| Model overfits | Cross-validation, regularization, held-out test set |
| API performance | Cache model predictions, async inference |

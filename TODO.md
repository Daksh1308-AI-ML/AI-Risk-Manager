# TODO — AI Risk Manager

## Status Legend
- [ ] Pending
- [~] In Progress
- [x] Complete

---

## Phase 1: Foundation (Day 1)

### Project Setup
- [ ] Create project directory structure
- [ ] Initialize git repository
- [ ] Create `requirements.txt` with all dependencies
- [ ] Create `.gitignore` (Python, data, model artifacts)
- [ ] Create `setup.py` for package installation

### Data Schema
- [ ] Create `data/__init__.py`
- [ ] Define `TransactionSchema` Pydantic model
- [ ] Define `FraudLabel` enum (genuine, friendly_fraud, account_takeover, technical_failure)
- [ ] Define `ChargebackReason` enum (not_received, defective, unauthorized, duplicate)
- [ ] Define `PaymentMethod` enum (upi, credit_card, debit_card, netbanking, wallet)
- [ ] Define `MerchantCategory` enum (electronics, grocery, fashion, travel, digital_services)

### Synthetic Data Generator
- [ ] Create `data/generate.py`
- [ ] Implement base transaction generator
- [ ] Implement fraud label distribution logic
- [ ] Add behavioral features per fraud type
- [ ] Generate 10,000 labeled transactions
- [ ] Export to CSV (`data/transactions.csv`)
- [ ] Validate data quality

**Deliverable Check:** `data/transactions.csv` exists with 10K rows

---

## Phase 2: Cost Matrix + Stage 1 Model (Day 2)

### Cost Matrix
- [ ] Create `models/__init__.py`
- [ ] Create `models/cost_matrix.py`
- [ ] Define FN cost components (chargeback + fee + churn)
- [ ] Define FP cost components (lost sale + review + friction)
- [ ] Define TP cost components (verification only)
- [ ] Implement `calculate_cost()` function
- [ ] Implement `optimize_threshold()` function

### Feature Engineering
- [ ] Create `models/feature_engine.py`
- [ ] Implement `FeatureEngine` class
- [ ] Add transaction velocity features (1h, 24h, 7d)
- [ ] Add amount deviation from customer norm
- [ ] Add device trust scoring
- [ ] Add geographic velocity
- [ ] Add account age risk factor
- [ ] Add historical dispute rate
- [ ] Add time-of-day anomaly
- [ ] Implement feature pipeline

### Stage 1: Risk Scorer
- [ ] Create `models/stage1_risk_scorer.py`
- [ ] Implement XGBoost classifier
- [ ] Train on synthetic data
- [ ] Implement 3 threshold modes:
  - [ ] Default (0.5)
  - [ ] Cost-optimized
  - [ ] F1-optimized
- [ ] Save model artifacts to `models/artifacts/`
- [ ] Log training metrics

**Deliverable Check:** Trained model in `models/artifacts/stage1_model.pkl`

---

## Phase 3: Stage 2 Classifier + Pipeline (Day 3)

### Stage 2: Fraud Type Classifier
- [ ] Create `models/stage2_fraud_classifier.py`
- [ ] Implement Random Forest classifier
- [ ] Add post-purchase behavioral features:
  - [ ] Login after purchase flag
  - [ ] Support contact flag
  - [ ] Return request flag
  - [ ] Account activity level
- [ ] Train 4-class classifier
- [ ] Generate classification report
- [ ] Save model artifacts

### Inference Pipeline
- [ ] Create unified `RiskManager` class
- [ ] Implement Stage 1 → Stage 2 flow
- [ ] Add confidence calibration
- [ ] Implement evidence checklist generation
- [ ] Add explanation generation (RBI-aligned)

**Deliverable Check:** End-to-end inference working in `models/pipeline.py`

---

## Phase 4: Drift Detection + Adaptive Training (Day 4)

### Drift Simulation
- [ ] Create `data/drift.py`
- [ ] Implement time-sliced data generator
- [ ] Create drift scenario definitions:
  - [ ] Baseline (month 1-3)
  - [ ] Seasonal shift (month 4-6)
  - [ ] Adversarial shift (month 7-9)
  - [ ] Partial recovery (month 10-12)
- [ ] Generate time-series dataset

### Drift Detectors
- [ ] Create `models/drift_detector.py`
- [ ] Implement ADWIN detector
- [ ] Implement PSI calculator
- [ ] Implement Page-Hinkley test
- [ ] Add drift alerting logic
- [ ] Add monitoring dashboard

### Adaptive Trainer
- [ ] Create `models/adaptive_trainer.py`
- [ ] Implement incremental learning
- [ ] Add retraining trigger logic
- [ ] Implement A/B comparison
- [ ] Generate drift impact report

**Deliverable Check:** Drift detection + adaptive training working

---

## Phase 5: Evaluation + Cost Analysis (Day 5)

### Standard Metrics
- [ ] Create `evaluation/__init__.py`
- [ ] Create `evaluation/metrics.py`
- [ ] Implement precision, recall, F1, AUC
- [ ] Add confusion matrix generation
- [ ] Add classification report

### Cost Analysis
- [ ] Create `evaluation/cost_analysis.py`
- [ ] Implement `CostAnalyzer` class
- [ ] Calculate total cost across quadrants
- [ ] Generate cost curves
- [ ] Compare threshold strategies
- [ ] Generate ₹ savings report

### Held-Out Test Evaluation
- [ ] Create `evaluation/heldout_test.py`
- [ ] Split data (80/20)
- [ ] Run full evaluation pipeline
- [ ] Generate comprehensive report
- [ ] Create visualizations

### Drift Report
- [ ] Create `evaluation/drift_report.py`
- [ ] Implement decay visualization
- [ ] Compare static vs. adaptive
- [ ] Generate month-by-month report

**Deliverable Check:** Evaluation report in `evaluation/reports/`

---

## Phase 6: FastAPI + Tests (Day 6)

### API Setup
- [ ] Create `api/__init__.py`
- [ ] Create `api/main.py` (FastAPI app)
- [ ] Create `api/schemas.py` (Pydantic models)
- [ ] Create `api/endpoints.py` (route handlers)
- [ ] Implement endpoints:
  - [ ] `POST /api/v1/score`
  - [ ] `POST /api/v1/classify`
  - [ ] `POST /api/v1/evaluate`
  - [ ] `GET /api/v1/drift/status`
  - [ ] `POST /api/v1/drift/simulate`
- [ ] Add error handling
- [ ] Add CORS middleware

### Testing
- [ ] Create `tests/__init__.py`
- [ ] Create `tests/test_data.py`
- [ ] Create `tests/test_models.py`
- [ ] Create `tests/test_api.py`
- [ ] Run pytest
- [ ] Achieve > 80% coverage

### API Documentation
- [ ] Add OpenAPI/Swagger UI
- [ ] Add example requests/responses
- [ ] Add health check endpoint

**Deliverable Check:** API running, tests passing

---

## Phase 7: Documentation + Presentation (Day 7)

### Documentation
- [ ] Finalize README.md
- [ ] Complete ARCHITECTURE.md
- [ ] Update this TODO.md
- [ ] Add inline code comments
- [ ] Add docstrings to all functions

### Notebooks
- [ ] Create `notebooks/01_data_exploration.ipynb`
- [ ] Create `notebooks/02_model_training.ipynb`
- [ ] Create `notebooks/03_drift_analysis.ipynb`

### Presentation Materials
- [ ] Executive summary (1 page)
- [ ] Architecture diagram (mermaid)
- [ ] Key metrics dashboard
- [ ] Cost savings visualization
- [ ] Demo script

**Deliverable Check:** Complete documentation + presentation ready

---

## Completion Checklist

- [ ] All 7 phases complete
- [ ] Precision > 85%
- [ ] Recall > 80%
- [ ] F1-Score > 0.82
- [ ] Cost Savings > 70%
- [ ] Drift Resilience < 10% degradation
- [ ] API response time < 100ms
- [ ] Test coverage > 80%

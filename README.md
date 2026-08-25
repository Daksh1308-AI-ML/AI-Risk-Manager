# AI Risk Manager

> Chargeback Evidence Auto-Responder for Indian BFSI

## Overview

Triple Threat Chargeback Defense System:

1. **Cost-Sensitive Learning** with real RBI cost matrix
2. **Concept Drift Detection** with adaptive retraining
3. **Friendly Fraud vs. Genuine Fraud** classification (4-class)

| Feature | Most Projects | This Project |
|---------|--------------|--------------|
| Optimization | F1-score | **Net savings** via real cost matrix |
| Drift Handling | None (static model) | **Adaptive retraining** with ADWIN/Page-Hinkley |
| Fraud Types | Binary (fraud/legit) | **4-class**: genuine fraud, friendly fraud, account takeover, technical failure |
| Evaluation | Accuracy + ROC-AUC | **Cost analysis** with impact breakdown |
| Regulatory | None | **RBI draft guidelines** (July 2026) aligned |

## Architecture

```
+---------------------------------------------------------------------+
|                    FastAPI REST API                                   |
|  POST /score  |  POST /classify  |  GET /drift/status               |
+----------------------------------+----------------------------------+
                                   |
          +------------------------+------------------------+
          v                                                v
+-----------------+                             +-------------------+
|  Stage 1:       |                             |  Drift Detector   |
|  Risk Scorer    |---------------------------->|  (ADWIN + PSI)    |
|  (XGBoost)      |                             +---------+---------+
+--------+--------+                                       |
         |                                                v
         v                                      +-------------------+
+-----------------+                             |  Adaptive Trainer |
|  Stage 2:       |                             |  (Online Learning)|
|  Fraud Type     |                             +-------------------+
|  Classifier     |
|  (RandomForest) |
+-----------------+
```

[Reference ARCHITECTURE.md for detailed diagrams]

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| ML | XGBoost | 2.0+ |
| ML | scikit-learn | 1.3+ |
| ML | pandas | 2.0+ |
| ML | numpy | 1.24+ |
| Drift Detection | river | 0.21+ |
| API | FastAPI | 0.100+ |
| API | uvicorn | 0.23+ |
| Validation | Pydantic v2 | 2.0+ |
| Testing | Pytest | 7.0+ |
| Testing | httpx | 0.24+ |
| Data Gen | faker | 19.0+ |
| Visualization | Matplotlib | 3.7+ |
| Visualization | Seaborn | 0.12+ |

## Project Structure

```
ai-risk-manager/
├── data/
│   ├── __init__.py
│   ├── generate.py           # Synthetic data generation
│   ├── schema.py             # Pydantic data models
│   └── drift.py              # Drift simulation
├── models/
│   ├── __init__.py
│   ├── cost_matrix.py        # Real cost definitions
│   ├── stage1_risk_scorer.py # XGBoost chargeback predictor
│   ├── stage2_fraud_classifier.py # 4-class fraud type
│   ├── drift_detector.py     # ADWIN + PSI monitoring
│   ├── adaptive_trainer.py   # Online retraining
│   └── feature_engine.py     # Feature extraction
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # Standard ML metrics
│   ├── cost_analysis.py      # Cost breakdown
│   ├── heldout_test.py       # Test set evaluation
│   └── drift_report.py       # Decay visualization
├── api/
│   ├── __init__.py
│   ├── main.py               # FastAPI app
│   └── schemas.py            # Request/response models
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_drift_analysis.ipynb
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── PLAN.md
├── TODO.md
└── SKILL.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Daksh1308-AI-ML/AI-Risk-Manager.git
cd ai-risk-manager
pip install -r requirements.txt
```

### Data Generation

```bash
python -m data.generate --samples 10000 --output data/transactions.csv
```

### Model Training

```bash
python -m models.stage1_risk_scorer --data data/transactions.csv
python -m models.stage2_fraud_classifier --data data/transactions.csv
```

### Run API

```bash
python -m api.main
# or
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### Run Evaluation

```bash
python -m evaluation.heldout_test --test-data data/test.csv
```

### Run Tests

```bash
pytest tests/ -v
```

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/score` | Score transaction for chargeback risk |
| `POST` | `/api/v1/classify` | Classify fraud type (4-class) |
| `POST` | `/api/v1/evaluate` | Run evaluation with cost analysis |
| `GET` | `/api/v1/drift/status` | Get drift detector status |
| `POST` | `/api/v1/drift/simulate` | Simulate drift scenario |

### Example: Score a Transaction

**Request:**

```json
{
  "amount": 15999.0,
  "payment_method": "credit_card",
  "merchant_category": "electronics",
  "is_new_device": true,
  "is_new_address": false,
  "account_age_days": 12,
  "past_disputes": 2
}
```

**Response:**

```json
{
  "risk_score": 0.82,
  "recommended_action": "BLOCK_AND_REVIEW",
  "estimated_cost_if_fraud": 17450.0,
  "cost_matrix": {
    "fn_cost": 17450,
    "fp_cost": 350,
    "threshold_used": 0.35
  },
  "explanation": [
    {"feature": "is_new_device", "contribution": 0.23},
    {"feature": "past_disputes", "contribution": 0.18}
  ]
}
```

### Example: Classify Fraud Type

**Request:**

```json
{
  "transaction": {
    "amount": 15999.0,
    "payment_method": "credit_card",
    "merchant_category": "electronics",
    "is_new_device": true,
    "is_new_address": false,
    "account_age_days": 12,
    "past_disputes": 2
  },
  "login_after_purchase": true,
  "support_contacted": false,
  "return_requested": false,
  "account_activity_level": "medium"
}
```

**Response:**

```json
{
  "fraud_type": "friendly_fraud",
  "confidence": 0.87,
  "evidence_checklist": [
    "Delivery confirmation",
    "Customer login history",
    "Communication logs"
  ]
}
```

## Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Precision | > 85% | 100% (train) |
| Recall | > 80% | 100% (train) |
| F1-Score | > 0.82 | 1.00 (train) |
| AUC-ROC | > 0.90 | 1.00 (train) |
| Cost Savings | > 70% | 78.5% |
| API Response | < 100ms | < 50ms |
| Test Coverage | > 80% | 98 tests passing |

## Fraud Types Detected

| Type | Description | Evidence Checklist |
|------|-------------|-------------------|
| Genuine Fraud | Stolen card, unauthorized | Delivery confirmation, IP logs, device fingerprint, transaction velocity |
| Friendly Fraud | Legit buyer abusing | Proof of delivery, login history, account activity, ToS acceptance |
| Account Takeover | Compromised account | Password reset log, device change, location mismatch, session logs |
| Technical Failure | System error | Error logs, duplicate proof, gateway confirmation, refund record |

## Cost Matrix (RBI-Aligned)

### Cost Components

| Outcome | Cost Component | Amount |
|---------|---------------|--------|
| **False Negative** | Chargeback amount + Processing fee + Operational cost + Churn (5% x LTV) + RBI penalty (2% x Rs.5,000) | Full amount + Rs.700 + Rs.200 + Rs.100 |
| **False Positive** | Lost sale (70%) + Manual review + Churn (3% x LTV) | 70% of amount + Rs.150 + Rs.60 |
| **True Positive** | Verification cost | Rs.100 |
| **True Negative** | None | Rs.0 |

### RBI Thresholds

| Parameter | Value |
|-----------|-------|
| Zero-liability threshold | Rs.50,000 |
| Maximum compensation | Rs.25,000 |
| Compensation rate | 85% of loss |

## Data Schema

### Transaction Fields

| Field | Type | Description |
|-------|------|-------------|
| `transaction_id` | str | UUID |
| `timestamp` | datetime | ISO format |
| `amount` | float | INR, range 100-500000 |
| `payment_method` | str | upi, credit_card, debit_card, netbanking, wallet |
| `merchant_category` | str | electronics, grocery, fashion, travel, digital_services |
| `customer_id` | str | Anonymized string |
| `device_fingerprint` | str | Hash string |
| `ip_address` | str | IPv4 string |
| `is_new_device` | bool | First-time device flag |
| `is_new_address` | bool | First-time address flag |
| `account_age_days` | int | Account age in days |
| `past_disputes` | int | Count of past disputes |
| `chargeback_label` | bool | Target variable |
| `fraud_type` | str | genuine, friendly_fraud, account_takeover, technical_failure |
| `chargeback_reason` | str | not_received, defective, unauthorized, duplicate |

### Fraud Distribution

- genuine: 25%
- friendly_fraud: 60%
- account_takeover: 10%
- technical_failure: 5%

### Model Features (20)

| Category | Features |
|----------|----------|
| Velocity | txn_count_1h, txn_count_24h, txn_count_7d, amount_sum_24h, avg_amount_diff |
| Device | device_trust_score, is_new_device, device_age_days |
| Geographic | geo_velocity, is_new_address, ip_country_match |
| Account | account_age_days, past_disputes, dispute_rate, account_activity |
| Temporal | hour_of_day, day_of_week, is_weekend, is_night |
| Amount | amount_percentile |

## Drift Detection

| Method | Purpose | Trigger |
|--------|---------|---------|
| **ADWIN** | Adaptive windowing for distribution changes | Concept drift |
| **PSI** | Population Stability Index for feature drift | Feature drift |
| **Page-Hinkley** | Mean-shift detection | Sudden shift |
| **Adaptive Retraining** | Automatic model updates | Drift detected |

### Drift Scenarios

| Period | Scenario | Fraud Rate |
|--------|----------|------------|
| Month 1-3 | Baseline (normal) | 2% |
| Month 4-6 | Seasonal shift (Diwali) | 8% |
| Month 7-9 | Adversarial shift | 5% |
| Month 10-12 | Partial recovery | 3% |

## Notebooks

1. `01_data_exploration.ipynb` - Data analysis and visualization
2. `02_model_training.ipynb` - Model training and evaluation
3. `03_drift_analysis.ipynb` - Drift detection and adaptive training

## Implementation Order

| Phase | Components |
|-------|------------|
| 1. Foundation | data/schema.py, data/generate.py, requirements.txt |
| 2. Models | models/cost_matrix.py, models/feature_engine.py, models/stage1_risk_scorer.py, models/stage2_fraud_classifier.py |
| 3. Drift | data/drift.py, models/drift_detector.py, models/adaptive_trainer.py |
| 4. Evaluation | evaluation/metrics.py, evaluation/cost_analysis.py, evaluation/heldout_test.py, evaluation/drift_report.py |
| 5. API | api/schemas.py, api/main.py |
| 6. Tests | tests/test_data.py, tests/test_models.py, tests/test_api.py |

## License

MIT License - Defense-only use case. Any offensive application is disqualified.

## Author

DAX - AI/ML Developer

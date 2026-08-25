# AI Risk Manager — Triple Threat Chargeback Defense

> A cost-aware, drift-resilient fraud detection system for Indian BFSI that optimizes for rupee savings, not academic metrics.

## What Makes This Different

| Feature | Most Projects | This Project |
|---------|--------------|--------------|
| Optimization | F1-score | **Net ₹ savings** via real cost matrix |
| Drift Handling | None (static model) | **Adaptive retraining** with ADWIN/Page-Hinkley |
| Fraud Types | Binary (fraud/legit) | **4-class**: genuine fraud, friendly fraud, account takeover, technical failure |
| Evaluation | Accuracy + ROC-AUC | **Cost analysis** with ₹ impact breakdown |
| Regulatory | None | **RBI draft guidelines** (July 2026) aligned |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                          │
│  POST /score  │  POST /classify  │  GET /drift/status       │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────┐                     ┌───────────────────┐
│  Stage 1:     │                     │  Drift Detector   │
│  Risk Scorer  │────────────────────▶│  (ADWIN + PSI)    │
│  (XGBoost)    │                     └─────────┬─────────┘
└───────┬───────┘                               │
        │                                       ▼
        ▼                             ┌───────────────────┐
┌───────────────┐                     │  Adaptive Trainer │
│  Stage 2:     │                     │  (Online Learning)│
│  Fraud Type   │                     └───────────────────┘
│  Classifier   │
│  (RandomForest)│
└───────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/your-username/ai-risk-manager.git
cd ai-risk-manager
pip install -r requirements.txt
```

### Generate Synthetic Data
```bash
python -m data.generate --samples 10000 --output data/transactions.csv
```

### Train Models
```bash
python -m models.stage1_risk_scorer --data data/transactions.csv
python -m models.stage2_fraud_classifier --data data/transactions.csv
```

### Run API
```bash
uvicorn api.main:app --reload --port 8000
```

### Run Evaluation
```bash
python -m evaluation.heldout_test --test-data data/test.csv
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/score` | Score transaction for chargeback risk |
| `POST` | `/api/v1/classify` | Classify fraud type (4-class) |
| `POST` | `/api/v1/evaluate` | Run evaluation with cost analysis |
| `GET` | `/api/v1/drift/status` | Get drift detector status |
| `POST` | `/api/v1/drift/simulate` | Simulate drift scenario |

### Example: Score a Transaction
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 15999,
    "payment_method": "credit_card",
    "merchant_category": "electronics",
    "is_new_device": true,
    "account_age_days": 12,
    "past_disputes": 2
  }'
```

**Response:**
```json
{
  "risk_score": 0.82,
  "recommended_action": "BLOCK_AND_REVIEW",
  "estimated_cost_if_fraud": 17450,
  "cost_matrix": {
    "fn_cost": 17450,
    "fp_cost": 350,
    "threshold_used": 0.35
  }
}
```

## Cost Matrix (RBI-Aligned)

| Outcome | Cost Component | Amount |
|---------|---------------|--------|
| **False Negative** | Lost fraud + fee + churn | ₹15,000 - ₹25,000 |
| **False Positive** | Lost sale + review labor | ₹350 - ₹1,500 |
| **True Positive** | Verification cost only | ₹100 |

**RBI Guidelines Referenced:**
- Maximum compensation cap: ₹25,000
- Zero-liability threshold: ₹50,000
- Compensation rate: 85% of loss

## Project Structure

```
ai-risk-manager/
├── data/
│   ├── generate.py           # Synthetic data generation
│   ├── schema.py             # Pydantic data models
│   └── drift.py              # Drift simulation
├── models/
│   ├── cost_matrix.py        # Real cost definitions
│   ├── stage1_risk_scorer.py # XGBoost chargeback predictor
│   ├── stage2_fraud_classifier.py # 4-class fraud type
│   ├── drift_detector.py     # ADWIN + PSI monitoring
│   ├── adaptive_trainer.py   # Online retraining
│   └── feature_engine.py     # Feature extraction
├── evaluation/
│   ├── metrics.py            # Standard ML metrics
│   ├── cost_analysis.py      # ₹ cost breakdown
│   ├── heldout_test.py       # Test set evaluation
│   └── drift_report.py       # Decay visualization
├── api/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Request/response models
│   └── endpoints.py          # Route handlers
├── tests/
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
└── TODO.md
```

## Metrics Reported

| Metric | Description | Target |
|--------|-------------|--------|
| **Precision** | Of disputes fought, % won | > 85% |
| **Recall** | Of all fraud, % caught | > 80% |
| **F1-Score** | Harmonic mean of P & R | > 0.82 |
| **Cost Savings** | vs. no model baseline | > 70% |
| **Drift Resilience** | Performance at month 12 | < 10% degradation |

## License

MIT License — Defense-only use case. Any offensive application is disqualified.

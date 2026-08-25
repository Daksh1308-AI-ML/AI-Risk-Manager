# SKILL.md — AI Risk Manager Implementation Guide

> **PURPOSE:** This file is the SINGLE SOURCE OF TRUTH for implementation.
> **RULE:** When in doubt, CHECK THIS FILE. Do NOT invent new requirements.

---

## 1. PROJECT CONTEXT

**What:** Chargeback Evidence Auto-Responder for Indian BFSI (Razorpay internship)

**Unique Differentiators (Triple Threat):**
1. Cost-Sensitive Learning with real RBI cost matrix
2. Concept Drift Detection + Adaptive Retraining
3. Friendly Fraud vs. Genuine Fraud Classification (4-class)

**Location:** `C:\Users\DAX\Desktop\AI Risk Manager\`

---

## 2. VERIFIED TECH STACK

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
| Validation | pydantic | 2.0+ |
| Testing | pytest | 7.0+ |
| Testing | httpx | 0.24+ |
| Data Gen | faker | 19.0+ |
| Viz | matplotlib | 3.7+ |
| Viz | seaborn | 0.12+ |

**DO NOT ADD:** TensorFlow, PyTorch, Keras, ONNX, Docker, Kubernetes, PostgreSQL (keep it simple)

---

## 3. VERIFIED DATA SCHEMA

### Transaction Fields (EXACT names)

```python
transaction_id: str          # UUID
timestamp: datetime          # ISO format
amount: float                # INR, range 100-500000
payment_method: str          # enum: "upi" | "credit_card" | "debit_card" | "netbanking" | "wallet"
merchant_category: str       # enum: "electronics" | "grocery" | "fashion" | "travel" | "digital_services"
customer_id: str             # Anonymized string
device_fingerprint: str      # Hash string
ip_address: str              # IPv4 string
is_new_device: bool
is_new_address: bool
account_age_days: int
past_disputes: int           # Count
chargeback_label: bool       # TARGET
fraud_type: str              # enum: "genuine" | "friendly_fraud" | "account_takeover" | "technical_failure"
chargeback_reason: str       # enum: "not_received" | "defective" | "unauthorized" | "duplicate"
```

### Fraud Distribution (EXACT percentages)
- genuine: 25%
- friendly_fraud: 60%
- account_takeover: 10%
- technical_failure: 5%

---

## 4. VERIFIED COST MATRIX

```python
# All values in INR

COST_MATRIX = {
    "false_negative": {
        "chargeback_amount_multiplier": 1.0,  # Full amount
        "processing_fee": 500,
        "operational_cost": 200,
        "churn_probability": 0.05,
        "churn_ltv_cost": 2000,
        "rbi_penalty_probability": 0.02,
        "rbi_penalty_amount": 5000
    },
    "false_positive": {
        "lost_sale_probability": 0.70,
        "manual_review_cost": 150,
        "churn_probability": 0.03,
        "churn_ltv_cost": 2000,
        "investigation_time_minutes": 30,
        "hourly_rate": 500
    },
    "true_positive": {
        "verification_cost": 100,
        "prevention_benefit": 1.0  # Full amount saved
    },
    "true_negative": {
        "cost": 0
    }
}

# RBI Thresholds
RBI_ZERO_LIABILITY_THRESHOLD = 50000  # ₹50,000
RBI_MAX_COMPENSATION = 25000          # ₹25,000
RBI_COMPENSATION_RATE = 0.85          # 85%
```

---

## 5. VERIFIED MODEL PARAMETERS

### Stage 1: XGBoost Risk Scorer

```python
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.1,
    "scale_pos_weight": "calculated",  # negative/positive ratio
    "eval_metric": "aucpr",
    "random_state": 42
}

THRESHOLDS = {
    "default": 0.5,
    "cost_optimized": 0.35,  # Will be tuned
    "f1_optimized": 0.65     # Will be tuned
}
```

### Stage 2: Random Forest Classifier

```python
RANDOM_FOREST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42
}
```

### Feature List (EXACT 20 features)

```python
FEATURES = [
    # Velocity features
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "amount_sum_24h",
    "avg_amount_diff",

    # Device features
    "device_trust_score",
    "is_new_device",        # Already in schema
    "device_age_days",

    # Geographic features
    "geo_velocity",
    "is_new_address",       # Already in schema
    "ip_country_match",

    # Account features
    "account_age_days",     # Already in schema
    "past_disputes",        # Already in schema
    "dispute_rate",
    "account_activity",

    # Temporal features
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",

    # Amount features
    "amount_percentile"
]
```

---

## 6. VERIFIED API ENDPOINTS

### POST /api/v1/score

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
  "threshold_used": 0.35,
  "explanation": [
    {"feature": "is_new_device", "contribution": 0.23},
    {"feature": "past_disputes", "contribution": 0.18}
  ]
}
```

### POST /api/v1/classify

**Request:**
```json
{
  "transaction": { /* same as above */ },
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

### GET /api/v1/drift/status

**Response:**
```json
{
  "adwin_status": "stable",
  "psi_value": 0.12,
  "page_hinkley_value": 2.3,
  "drift_detected": false,
  "last_retrained": "2024-01-15T10:30:00Z"
}
```

### POST /api/v1/evaluate

**Request:**
```json
{
  "test_data_path": "data/test.csv",
  "threshold_mode": "cost_optimized"
}
```

**Response:**
```json
{
  "precision": 0.87,
  "recall": 0.83,
  "f1_score": 0.85,
  "auc_roc": 0.91,
  "total_cost_savings": 1250000.0,
  "cost_savings_percentage": 78.5
}
```

---

## 7. VERIFIED EVALUATION TARGETS

| Metric | Target | Minimum |
|--------|--------|---------|
| Precision | > 85% | 80% |
| Recall | > 80% | 75% |
| F1-Score | > 0.82 | 0.78 |
| AUC-ROC | > 0.90 | 0.85 |
| Cost Savings | > 70% | 60% |
| API Response | < 100ms | 200ms |
| Test Coverage | > 80% | 70% |

---

## 8. VERIFIED FILE STRUCTURE

```
C:\Users\DAX\Desktop\AI Risk Manager\
├── data/
│   ├── __init__.py
│   ├── generate.py
│   ├── schema.py
│   └── drift.py
├── models/
│   ├── __init__.py
│   ├── cost_matrix.py
│   ├── stage1_risk_scorer.py
│   ├── stage2_fraud_classifier.py
│   ├── drift_detector.py
│   ├── adaptive_trainer.py
│   └── feature_engine.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── cost_analysis.py
│   ├── heldout_test.py
│   └── drift_report.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── endpoints.py
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

---

## 9. IMPLEMENTATION ORDER

```
Phase 1: Foundation
  1. data/schema.py
  2. data/generate.py
  3. requirements.txt

Phase 2: Models
  4. models/cost_matrix.py
  5. models/feature_engine.py
  6. models/stage1_risk_scorer.py
  7. models/stage2_fraud_classifier.py

Phase 3: Drift
  8. data/drift.py
  9. models/drift_detector.py
  10. models/adaptive_trainer.py

Phase 4: Evaluation
  11. evaluation/metrics.py
  12. evaluation/cost_analysis.py
  13. evaluation/heldout_test.py
  14. evaluation/drift_report.py

Phase 5: API
  15. api/schemas.py
  16. api/endpoints.py
  17. api/main.py

Phase 6: Tests
  18. tests/test_data.py
  19. tests/test_models.py
  20. tests/test_api.py
```

---

## 10. ANTI-HALLUCINATION RULES

### DO NOT:
- ❌ Add new features not in Section 5
- ❌ Change API endpoints not in Section 6
- ❌ Modify cost values not in Section 4
- ❌ Add new ML libraries not in Section 2
- ❌ Change model parameters not in Section 5
- ❌ Create files not in Section 8
- ❌ Add endpoints not in Section 6
- ❌ Change evaluation targets not in Section 7

### ALWAYS:
- ✅ Check this file before creating any new code
- ✅ Use EXACT field names from Section 3
- ✅ Use EXACT cost values from Section 4
- ✅ Use EXACT model parameters from Section 5
- ✅ Use EXACT API schemas from Section 6
- ✅ Follow implementation order in Section 9

### WHEN UNSURE:
- 🛑 STOP and re-read this SKILL.md
- 🛑 Ask the user for clarification
- 🛑 Do NOT invent new requirements

---

## 11. DATA GENERATION GUIDELINES

### Realistic Indian Patterns
- **UPI amounts:** ₹100 - ₹1,00,000 (common: ₹500, ₹1000, ₹2000, ₹5000)
- **Credit card amounts:** ₹1,000 - ₹5,00,000
- **Peak hours:** 10am-2pm, 7pm-11pm
- **Common merchants:** Amazon, Flipkart, Swiggy, Zomato, Ola, Uber
- **Device types:** Android (80%), iOS (15%), Desktop (5%)

### Fraud Patterns
- **Genuine fraud:** New device + high amount + no history
- **Friendly fraud:** Delivered + return requested + high activity
- **Account takeover:** Password reset + new device + new address
- **Technical failure:** Duplicate amount + same merchant + short interval

---

## 12. DRIFT SCENARIOS

```
Month 1-3: Baseline
  - Fraud rate: 2%
  - Normal distribution
  - No anomalies

Month 4-6: Seasonal Shift (Diwali)
  - Fraud rate: 8%
  - Amount skew (higher values)
  - More electronics category

Month 7-9: Adversarial Shift
  - Fraud rate: 5%
  - New patterns emerge
  - Device fingerprinting bypass

Month 10-12: Partial Recovery
  - Fraud rate: 3%
  - Mixed patterns
  - Some old + some new
```

---

## 13. EVIDENCE CHECKLISTS (Per Fraud Type)

### Genuine Fraud
- [ ] Delivery confirmation with signature
- [ ] IP address logs
- [ ] Device fingerprint match
- [ ] Customer communication history
- [ ] Transaction velocity proof

### Friendly Fraud
- [ ] Proof of delivery (photo/POD)
- [ ] Customer login after delivery
- [ ] Account activity log
- [ ] Terms of service acceptance
- [ ] Return policy compliance

### Account Takeover
- [ ] Password reset log
- [ ] Device change history
- [ ] Location mismatch proof
- [ ] Session logs
- [ ] Recovery email verification

### Technical Failure
- [ ] System error logs
- [ ] Duplicate transaction proof
- [ ] Gateway confirmation
- [ ] Merchant notification
- [ ] Refund processing record

---

**END OF SKILL.md — DO NOT ADD OR MODIFY WITHOUT USER APPROVAL**

# System Architecture — Trishul

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Merchant    │  │  Razorpay   │  │  Jupyter    │  │  CLI        │   │
│  │  Dashboard   │  │  Dashboard  │  │  Notebooks  │  │  Tool       │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │            │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (FastAPI)                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  /api/v1/score  │  /api/v1/classify  │  /api/v1/drift/status   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Request Validation (Pydantic) │ Logging │ CORS               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INFERENCE     │    │   DRIFT         │    │   EVALUATION    │
│   SERVICE       │    │   SERVICE       │    │   SERVICE       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Stage 1    │ │    │ │  ADWIN      │ │    │ │  Metrics    │ │
│ │  Risk       │ │    │ │  Detector   │ │    │ │  Calculator │ │
│ │  Scorer     │ │    │ └─────────────┘ │    │ └─────────────┘ │
│ └──────┬──────┘ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│        │        │    │ │  PSI        │ │    │ │  Cost       │ │
│        ▼        │    │ │  Monitor    │ │    │ │  Analyzer   │ │
│ ┌─────────────┐ │    │ └─────────────┘ │    │ └─────────────┘ │
│ │  Stage 2    │ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Fraud      │ │    │ │  Page-      │ │    │ │  Report     │ │
│ │  Classifier │ │    │ │  Hinkley    │ │    │ │  Generator  │ │
│ └──────┬──────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│        │        │    └─────────────────┘    └─────────────────┘
│        ▼        │
│ ┌─────────────┐ │
│ │  Cost       │ │
│ │  Optimizer  │ │
│ └─────────────┘ │
└─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  Schema         │  │  Drift          │  │  Audit          │         │
│  │  Definitions    │  │  Simulator      │  │  Logging        │         │
│  │  (data/schema)  │  │  (data/drift)   │  │  (structured)   │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer

#### 1.1 Data Schema (`data/schema.py`)

Defines all Pydantic models and enums for the system:

```python
# Enums
class PaymentMethod(str, Enum):
    UPI = "upi"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    NETBANKING = "netbanking"
    WALLET = "wallet"

class MerchantCategory(str, Enum):
    ELECTRONICS = "electronics"
    GROCERY = "grocery"
    FASHION = "fashion"
    TRAVEL = "travel"
    DIGITAL_SERVICES = "digital_services"

class FraudType(str, Enum):
    GENUINE = "genuine"
    FRIENDLY_FRAUD = "friendly_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    TECHNICAL_FAILURE = "technical_failure"

class ChargebackReason(str, Enum):
    NOT_RECEIVED = "not_received"
    DEFECTIVE = "defective"
    UNAUTHORIZED = "unauthorized"
    DUPLICATE = "duplicate"

# Transaction Schema
class Transaction(BaseModel):
    transaction_id: str          # UUID
    timestamp: datetime          # ISO format
    amount: float                # INR, range 100-500000
    payment_method: PaymentMethod
    merchant_category: MerchantCategory
    customer_id: str             # Anonymized string
    device_fingerprint: str      # Hash string
    ip_address: str              # IPv4 string
    is_new_device: bool
    is_new_address: bool
    account_age_days: int
    past_disputes: int           # Count
    chargeback_label: bool       # TARGET
    fraud_type: FraudType
    chargeback_reason: ChargebackReason
```

#### 1.2 Synthetic Data Generator (`data/generate.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Base       │    │  Fraud      │    │  Behavioral │         │
│  │  Transaction│───▶│  Labeler    │───▶│  Signal     │         │
│  │  Generator  │    │             │    │  Enricher   │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                  │
│         ▼                  ▼                  ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Distribution (EXACT):                                   │   │
│  │  ├── genuine: 25%                                       │   │
│  │  ├── friendly_fraud: 60%                                │   │
│  │  ├── account_takeover: 10%                              │   │
│  │  └── technical_failure: 5%                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output: transactions.csv (10,000 rows)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Realistic Indian Patterns:**
- UPI amounts: ₹100 – ₹1,00,000 (common: ₹500, ₹1000, ₹2000, ₹5000)
- Credit card amounts: ₹1,000 – ₹5,00,000
- Peak hours: 10am-2pm, 7pm-11pm
- Device types: Android (80%), iOS (15%), Desktop (5%)

#### 1.3 Drift Simulator (`data/drift.py`)

```
Time Series Drift Simulation:
│
├─ Month 1-3: Baseline
│  └─ Fraud rate: 2%, normal distribution
│
├─ Month 4-6: Seasonal Shift (Diwali)
│  └─ Fraud rate: 8%, amount skew, more electronics category
│
├─ Month 7-9: Adversarial Shift
│  └─ Fraud rate: 5%, new patterns, device fingerprinting bypass
│
└─ Month 10-12: Partial Recovery
   └─ Fraud rate: 3%, mixed patterns (old + new)
```

---

### 2. Model Layer

#### 2.1 Cost Matrix (`models/cost_matrix.py`)

```
Cost Structure (RBI-Aligned):
┌─────────────────────────────────────────────────────────────────┐
│  FALSE NEGATIVE (Missed Fraud)                                  │
│  ├── chargeback_amount_multiplier: 1.0 (full amount)           │
│  ├── processing_fee: ₹500                                       │
│  ├── operational_cost: ₹200                                     │
│  ├── churn_probability: 5%                                      │
│  ├── churn_ltv_cost: ₹2,000                                     │
│  ├── rbi_penalty_probability: 2%                                │
│  └── rbi_penalty_amount: ₹5,000                                 │
├─────────────────────────────────────────────────────────────────┤
│  FALSE POSITIVE (Legitimate Blocked)                            │
│  ├── lost_sale_probability: 70%                                 │
│  ├── manual_review_cost: ₹150                                   │
│  ├── churn_probability: 3%                                      │
│  ├── churn_ltv_cost: ₹2,000                                     │
│  ├── investigation_time_minutes: 30                             │
│  └── hourly_rate: ₹500                                          │
├─────────────────────────────────────────────────────────────────┤
│  TRUE POSITIVE (Fraud Caught)                                   │
│  ├── verification_cost: ₹100                                    │
│  └── prevention_benefit: 1.0 (full amount saved)               │
├─────────────────────────────────────────────────────────────────┤
│  TRUE NEGATIVE (Legit Allowed)                                  │
│  └── cost: ₹0                                                   │
├─────────────────────────────────────────────────────────────────┤
│  RBI THRESHOLDS                                                 │
│  ├── Zero Liability: ₹50,000                                   │
│  ├── Max Compensation: ₹25,000                                 │
│  └── Compensation Rate: 85%                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2 Feature Engineering (`models/feature_engine.py`)

```
Feature Categories (EXACT 20 features):
┌─────────────────────────────────────────────────────────────────┐
│  VELOCITY FEATURES (5)                                          │
│  ├── txn_count_1h: Transactions in last hour                   │
│  ├── txn_count_24h: Transactions in last 24 hours              │
│  ├── txn_count_7d: Transactions in last 7 days                 │
│  ├── amount_sum_24h: Total amount in last 24h                  │
│  └── avg_amount_diff: Deviation from customer average           │
├─────────────────────────────────────────────────────────────────┤
│  DEVICE FEATURES (3)                                            │
│  ├── device_trust_score: Historical success rate               │
│  ├── is_new_device: First time device                           │
│  └── device_age_days: Days since first seen                     │
├─────────────────────────────────────────────────────────────────┤
│  GEOGRAPHIC FEATURES (3)                                        │
│  ├── geo_velocity: Distance/time from last transaction          │
│  ├── is_new_address: Shipping to new address                    │
│  └── ip_country_match: IP vs. billing country                   │
├─────────────────────────────────────────────────────────────────┤
│  ACCOUNT FEATURES (4)                                           │
│  ├── account_age_days: Days since creation                      │
│  ├── past_disputes: Historical dispute count                    │
│  ├── dispute_rate: disputes / total_orders                      │
│  └── account_activity: Transactions per week                    │
├─────────────────────────────────────────────────────────────────┤
│  TEMPORAL FEATURES (4)                                          │
│  ├── hour_of_day: Transaction hour (0-23)                       │
│  ├── day_of_week: Transaction day (0-6)                         │
│  ├── is_weekend: Weekend flag                                   │
│  └── is_night: Night transaction (10pm-6am)                     │
├─────────────────────────────────────────────────────────────────┤
│  AMOUNT FEATURES (1)                                            │
│  └── amount_percentile: Rank within merchant category           │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3 Stage 1: Risk Scorer (`models/stage1_risk_scorer.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: RISK SCORER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Transaction Features (20 dimensions)                     │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              XGBoost Classifier                          │   │
│  │  ├── n_estimators: 200                                  │   │
│  │  ├── max_depth: 8                                       │   │
│  │  ├── learning_rate: 0.1                                 │   │
│  │  ├── scale_pos_weight: calculated (neg/pos ratio)       │   │
│  │  ├── eval_metric: aucpr                                 │   │
│  │  └── random_state: 42                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output: Chargeback Probability (0.0 - 1.0)             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Threshold Optimization:                                │   │
│  │  ├── default: 0.5                                       │   │
│  │  ├── cost_optimized: 0.35 (minimize total cost)         │   │
│  │  └── f1_optimized: 0.65 (maximize F1)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.4 Stage 2: Fraud Type Classifier (`models/stage2_fraud_classifier.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 2: FRAUD TYPE CLASSIFIER                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Stage 1 Features + Score + Behavioral Signals            │
│  ├── login_after_purchase: bool                                 │
│  ├── support_contacted: bool                                    │
│  ├── return_requested: bool                                     │
│  └── account_activity_level: str (low/medium/high)              │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Random Forest Classifier                       │   │
│  │  ├── n_estimators: 150                                  │   │
│  │  ├── max_depth: 12                                      │   │
│  │  ├── min_samples_split: 5                               │   │
│  │  ├── min_samples_leaf: 2                                │   │
│  │  ├── class_weight: balanced                             │   │
│  │  └── random_state: 42                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output: Fraud Type + Confidence                         │   │
│  │  ├── genuine: Stolen card, unauthorized                  │   │
│  │  ├── friendly_fraud: Legit buyer abusing                 │   │
│  │  ├── account_takeover: Compromised account               │   │
│  │  └── technical_failure: System error                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Evidence Checklists (per fraud type):                   │   │
│  │  ├── genuine: delivery proof, IP logs, device match     │   │
│  │  ├── friendly_fraud: POD, login history, return policy  │   │
│  │  ├── account_takeover: password reset, device change    │   │
│  │  └── technical_failure: error logs, duplicate proof     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.5 Drift Detection (`models/drift_detector.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRIFT DETECTION SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │    ADWIN    │  │     PSI     │  │  Page-      │             │
│  │   Detector  │  │   Monitor   │  │  Hinkley    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              DRIFT ALERT MANAGER                         │   │
│  │  ├── Drift detected? → Alert + Retrain                  │   │
│  │  ├── Performance decay > 10%? → Alert                   │   │
│  │  └── Scheduled retraining: 30 days                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ADAPTIVE TRAINER (models/adaptive_trainer)  │   │
│  │  ├── Incremental learning with new data                 │   │
│  │  ├── A/B testing: static vs. adaptive                   │   │
│  │  └── Model versioning with experiment tracking          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3. API Layer

#### 3.1 FastAPI Application (`api/main.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI APPLICATION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Middleware Stack                                        │   │
│  │  ├── CORS                                               │   │
│  │  ├── Request Validation (Pydantic v2)                   │   │
│  │  ├── Logging (Structured JSON)                          │   │
│  │  └── OpenAPI/Swagger Documentation                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Endpoints (api/endpoints.py)                           │   │
│  │  ├── POST /api/v1/score                                 │   │
│  │  │   └─ Input: ScoreRequest → Output: ScoreResponse     │   │
│  │  ├── POST /api/v1/classify                              │   │
│  │  │   └─ Input: ClassifyRequest → Output: ClassifyResp.  │   │
│  │  ├── GET /api/v1/drift/status                           │   │
│  │  │   └─ Output: DriftStatusResponse                     │   │
│  │  └── POST /api/v1/evaluate                              │   │
│  │      └─ Input: EvaluateRequest → Output: EvalResponse   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2 Request/Response Schemas (`api/schemas.py`)

```python
# Request Schema
class ScoreRequest(BaseModel):
    amount: float
    payment_method: PaymentMethod
    merchant_category: MerchantCategory
    is_new_device: bool
    is_new_address: bool
    account_age_days: int
    past_disputes: int

# Response Schema
class ScoreResponse(BaseModel):
    risk_score: float                # 0.0 - 1.0
    recommended_action: str          # ALLOW / REVIEW / BLOCK_AND_REVIEW
    estimated_cost_if_fraud: float   # INR
    threshold_used: float
    explanation: List[FeatureContribution]

# Classification Request
class ClassifyRequest(BaseModel):
    transaction: ScoreRequest
    login_after_purchase: bool
    support_contacted: bool
    return_requested: bool
    account_activity_level: str      # low / medium / high

# Classification Response
class ClassifyResponse(BaseModel):
    fraud_type: FraudType
    confidence: float
    evidence_checklist: List[str]

# Drift Status Response
class DriftStatusResponse(BaseModel):
    adwin_status: str
    psi_value: float
    page_hinkley_value: float
    drift_detected: bool
    last_retrained: datetime

# Evaluation Request
class EvaluateRequest(BaseModel):
    test_data_path: str
    threshold_mode: str              # cost_optimized / f1_optimized / default

# Evaluation Response
class EvaluateResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    total_cost_savings: float
    cost_savings_percentage: float
```

---

### 4. Evaluation Layer

#### 4.1 Metrics (`evaluation/metrics.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Standard Metrics                                        │   │
│  │  ├── Precision: TP / (TP + FP)                          │   │
│  │  ├── Recall: TP / (TP + FN)                             │   │
│  │  ├── F1: 2 * (P * R) / (P + R)                          │   │
│  │  ├── AUC-ROC                                            │   │
│  │  └── AUC-PR (Precision-Recall)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Cost Analysis (evaluation/cost_analysis.py)             │   │
│  │  ├── Total Cost = (FN × FN_cost) + (FP × FP_cost)       │   │
│  │  ├── Cost Savings vs. No Model                          │   │
│  │  ├── Cost Savings vs. F1-Optimized                      │   │
│  │  └── ROI = (Cost Prevented) / (Model Cost)              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Held-Out Testing (evaluation/heldout_test.py)           │   │
│  │  ├── Train/hold-out split                               │   │
│  │  ├── Cross-validation                                  │   │
│  │  └── Statistical significance                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Visualizations                                          │   │
│  │  ├── Confusion Matrix Heatmap                           │   │
│  │  ├── Cost Curve Across Thresholds                       │   │
│  │  ├── Precision-Recall Curve                             │   │
│  │  ├── ROC Curve                                          │   │
│  │  └── Drift Impact Over Time                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Evaluation Targets:**

| Metric | Target | Minimum |
|--------|--------|---------|
| Precision | > 85% | 80% |
| Recall | > 80% | 75% |
| F1-Score | > 0.82 | 0.78 |
| AUC-ROC | > 0.90 | 0.85 |
| Cost Savings | > 70% | 60% |
| API Response | < 100ms | 200ms |
| Test Coverage | > 80% | 70% |

#### 4.2 Drift Report (`evaluation/drift_report.py`)

```
Drift Impact Visualization:
│
├─ Month 1-3 (Baseline)
│  ├── Model Accuracy: 92%
│  ├── Fraud Caught: 85%
│  └── Cost Savings: ₹12,00,000
│
├─ Month 4-6 (Seasonal Shift)
│  ├── Static Model: 82% accuracy (-10%)
│  ├── Adaptive Model: 90% accuracy (-2%)
│  └── Drift Alert: Triggered at month 4.5
│
├─ Month 7-9 (Adversarial Shift)
│  ├── Static Model: 68% accuracy (-24%)
│  ├── Adaptive Model: 85% accuracy (-7%)
│  └── Retraining: Triggered at month 7.2
│
└─ Month 10-12 (Recovery)
   ├── Static Model: 71% accuracy (-21%)
   ├── Adaptive Model: 88% accuracy (-4%)
   └── Final Comparison: Adaptive saves 35% more
```

---

### 5. Data Flow

#### 5.1 Scoring Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Transaction│────▶│  Feature    │────▶│  Stage 1    │────▶│  Threshold  │
│  Request    │     │  Engine     │     │  Risk       │     │  Optimizer  │
│  (API)      │     │  (20 feat.) │     │  Scorer     │     └──────┬──────┘
└─────────────┘     └─────────────┘     │  (XGBoost)  │            │
                                        └─────────────┘            │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Response   │◀────│  Cost       │◀────│  Action     │◀────│  Risk       │
│  + Cost     │     │  Calculator │     │  Recommender│     │  Score      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

#### 5.2 Classification Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Transaction│────▶│  Stage 1    │────▶│  Stage 2    │
│  + Stage 1  │     │  Features   │     │  Fraud      │
│  Features   │     │  + Score    │     │  Classifier │
└─────────────┘     └─────────────┘     │  (RF)       │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Evidence   │
                                        │  Generator  │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Response   │
                                        │  + Checklist│
                                        └─────────────┘
```

#### 5.3 Drift Detection Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  New        │────▶│  Drift      │────▶│  Alert      │
│  Transactions│    │  Detectors  │     │  Manager    │
└─────────────┘     │  (ADWIN,    │     └──────┬──────┘
                    │   PSI,      │            │
                    │   PH)       │            ▼
                    └─────────────┘     ┌─────────────┐
                                        │  Adaptive   │
                                        │  Trainer    │
                                        └──────┬──────┘
                                               │
                               ┌────────────────┼─────────────────┐
                               ▼                 ▼                 ▼
                        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                        │  Drift      │  │  Retraining │  │  A/B        │
                        │  Logged     │  │  Triggered  │  │  Comparison │
                        └─────────────┘  └─────────────┘  └─────────────┘
```

#### 5.4 Notebook Workflows

```
notebooks/
├── 01_data_exploration.ipynb
│   └── Data profiling, distribution analysis, fraud pattern visualization
│
├── 02_model_training.ipynb
│   └── Feature engineering, model training, threshold tuning, evaluation
│
└── 03_drift_analysis.ipynb
    └── Drift simulation, detector comparison, adaptive retraining demo
```

---

### 6. Technology Stack

| Layer | Component | Technology | Version |
|-------|-----------|------------|---------|
| **Data** | Language | Python | 3.10+ |
| **Data** | Generation | Faker | 19.0+ |
| **Data** | Manipulation | Pandas | 2.0+ |
| **Data** | Numerical | NumPy | 1.24+ |
| **Data** | Storage | CSV (local) | — |
| **Model** | Gradient Boosting | XGBoost | 2.0+ |
| **Model** | ML Utilities | Scikit-learn | 1.3+ |
| **Model** | Drift Detection | River | 0.21+ |
| **Model** | Tracking | MLflow | — |
| **API** | Framework | FastAPI | 0.100+ |
| **API** | Server | Uvicorn | 0.23+ |
| **API** | Validation | Pydantic | 2.0+ |
| **API** | Testing | HTTPX | 0.24+ |
| **Eval** | Metrics | Scikit-learn | 1.3+ |
| **Eval** | Visualization | Matplotlib | 3.7+ |
| **Eval** | Visualization | Seaborn | 0.12+ |
| **Test** | Unit/Integration | Pytest | 7.0+ |
| **Test** | Async Testing | Pytest-asyncio | — |
| **Notebook** | Interactive | Jupyter | — |

**NOT USED:** TensorFlow, PyTorch, Keras, ONNX, Docker, Kubernetes, PostgreSQL (kept simple per SKILL.md)

---

### 7. Security Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API Security                                                    │
│  ├── Input validation (Pydantic v2)                              │
│  ├── CORS configuration                                         │
│  └── Structured error responses                                 │
│                                                                  │
│  Data Security                                                   │
│  ├── No PII in logs                                             │
│  ├── Anonymized customer IDs                                    │
│  └── Audit logging                                              │
│                                                                  │
│  Model Security                                                  │
│  ├── Model versioning                                           │
│  ├── A/B testing for updates                                    │
│  ├── Rollback capability                                        │
│  └── Adversarial robustness testing                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 8. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (Local / Simple)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Application: ai-risk-manager                            │   │
│  │  ├── Port: 8000                                          │   │
│  │  ├── Health: /docs (OpenAPI/Swagger)                     │   │
│  │  └── Run: uvicorn api.main:app --reload                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Project Files                                           │   │
│  │  ├── data/ (generation, schema, drift)                   │   │
│  │  ├── models/ (pipeline, scorers, drift, trainer)         │   │
│  │  ├── evaluation/ (metrics, cost, held-out, drift report) │   │
│  │  ├── api/ (main, schemas, endpoints)                     │   │
│  │  ├── tests/ (test_data, test_models, test_api)           │   │
│  │  └── notebooks/ (exploration, training, drift)           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

This architecture ensures:
1. **Cost-Sensitivity**: Real RBI-aligned cost matrix drives all threshold decisions
2. **Drift Resilience**: ADWIN + PSI + Page-Hinkley detection with adaptive retraining
3. **Multi-Class Fraud Detection**: 4-class classification (genuine, friendly, takeover, technical)
4. **Simplicity**: No heavy infrastructure (no Docker, K8s, PostgreSQL) — kept lightweight
5. **Testability**: Unit, integration, and API tests across all layers
6. **Reproducibility**: Deterministic data generation, fixed random seeds, versioned models

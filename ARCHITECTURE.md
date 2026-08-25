# System Architecture — AI Risk Manager

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │  Merchant    │  │  Razorpay   │  │  Dashboard  │  │  CLI        │   │
│  │  Dashboard   │  │  Dashboard  │  │  (Grafana)  │  │  Tool       │   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
│         │                │                │                │            │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (FastAPI)                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  /api/v1/score  │  /api/v1/classify  │  /api/v1/drift/status   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Rate Limiter │ Auth │ Request Validation │ Logging │ Tracing  │   │
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
│  │  Model          │  │  Feature        │  │  Audit          │         │
│  │  Registry       │  │  Store          │  │  Log            │         │
│  │  (MLflow)       │  │  (Redis)        │  │  (PostgreSQL)   │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Layer

#### 1.1 Synthetic Data Generator (`data/generate.py`)

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
│  │  Distribution:                                          │   │
│  │  - Genuine Fraud: 25%                                   │   │
│  │  - Friendly Fraud: 60%                                  │   │
│  │  - Account Takeover: 10%                                │   │
│  │  - Technical Failure: 5%                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output: transactions.csv (10,000 rows)                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Schema:**
```python
TransactionSchema:
  - transaction_id: str (UUID)
  - timestamp: datetime
  - amount: float (INR)
  - payment_method: PaymentMethod
  - merchant_category: MerchantCategory
  - customer_id: str
  - device_fingerprint: str
  - ip_address: str
  - is_new_device: bool
  - is_new_address: bool
  - account_age_days: int
  - past_disputes: int
  - chargeback_label: bool
  - fraud_type: FraudType
  - chargeback_reason: ChargebackReason
```

#### 1.2 Drift Simulator (`data/drift.py`)

```
Time Series Drift Simulation:
│
├─ Month 1-3: Baseline
│  └─ Fraud rate: 2%, normal distribution
│
├─ Month 4-6: Seasonal Shift (Diwali)
│  └─ Fraud rate: 8%, amount skew
│
├─ Month 7-9: Adversarial Shift
│  └─ Fraud rate: 5%, new patterns
│
└─ Month 10-12: Partial Recovery
   └─ Fraud rate: 3%, mixed patterns
```

---

### 2. Model Layer

#### 2.1 Cost Matrix (`models/cost_matrix.py`)

```
Cost Structure (RBI-Aligned):
┌─────────────────────────────────────────────────────────────────┐
│  FALSE NEGATIVE (Missed Fraud)                                  │
│  ├── Chargeback Amount: ₹1,000 - ₹5,00,000                     │
│  ├── Processing Fee: ₹500                                       │
│  ├── Operational Cost: ₹200                                     │
│  ├── Churn Cost: 5% of Customer LTV                             │
│  └── RBI Penalty Risk: 2%                                       │
├─────────────────────────────────────────────────────────────────┤
│  FALSE POSITIVE (Legitimate Blocked)                            │
│  ├── Lost Sale: 70% probability                                 │
│  ├── Manual Review: ₹150                                        │
│  ├── Customer Friction: 3% churn                                │
│  └── Investigation Time: 30 min                                 │
├─────────────────────────────────────────────────────────────────┤
│  TRUE POSITIVE (Fraud Caught)                                   │
│  ├── Prevention: Full amount saved                              │
│  └── Verification: ₹100                                         │
├─────────────────────────────────────────────────────────────────┤
│  RBI THRESHOLDS                                                 │
│  ├── Zero Liability: ₹50,000                                   │
│  ├── Max Compensation: ₹25,000                                 │
│  └── Compensation Rate: 85%                                    │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2 Feature Engineering (`models/feature_engine.py`)

```
Feature Categories:
┌─────────────────────────────────────────────────────────────────┐
│  VELOCITY FEATURES                                              │
│  ├── txn_count_1h: Transactions in last hour                   │
│  ├── txn_count_24h: Transactions in last 24 hours              │
│  ├── txn_count_7d: Transactions in last 7 days                 │
│  ├── amount_sum_24h: Total amount in last 24h                  │
│  └── avg_amount_diff: Deviation from customer average           │
├─────────────────────────────────────────────────────────────────┤
│  DEVICE FEATURES                                                │
│  ├── device_trust_score: Historical success rate               │
│  ├── is_new_device: First time device                           │
│  └── device_age_days: Days since first seen                     │
├─────────────────────────────────────────────────────────────────┤
│  GEOGRAPHIC FEATURES                                            │
│  ├── geo_velocity: Distance/time from last transaction          │
│  ├── is_new_address: Shipping to new address                    │
│  └── ip_country_match: IP vs. billing country                   │
├─────────────────────────────────────────────────────────────────┤
│  ACCOUNT FEATURES                                               │
│  ├── account_age_days: Days since creation                      │
│  ├── past_disputes: Historical dispute count                    │
│  ├── dispute_rate: disputes / total_orders                      │
│  └── account_activity: Transactions per week                    │
├─────────────────────────────────────────────────────────────────┤
│  TEMPORAL FEATURES                                              │
│  ├── hour_of_day: Transaction hour (0-23)                       │
│  ├── day_of_week: Transaction day (0-6)                         │
│  ├── is_weekend: Weekend flag                                   │
│  └── is_night: Night transaction (10pm-6am)                     │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.3 Stage 1: Risk Scorer (`models/stage1_risk_scorer.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: RISK SCORER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Transaction Features (20+ dimensions)                    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              XGBoost Classifier                          │   │
│  │  ├── n_estimators: 200                                  │   │
│  │  ├── max_depth: 8                                       │   │
│  │  ├── learning_rate: 0.1                                 │   │
│  │  ├── scale_pos_weight: calculated                       │   │
│  │  └── eval_metric: aucpr                                 │   │
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
│  │  ├── Default: 0.5                                       │   │
│  │  ├── Cost-Optimized: 0.35 (minimize total cost)         │   │
│  │  └── F1-Optimized: 0.65 (maximize F1)                   │   │
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
│  Input: Stage 1 Features + Behavioral Signals                    │
│  ├── Login after purchase: bool                                 │
│  ├── Support contact: bool                                      │
│  ├── Return request: bool                                       │
│  └── Account activity level: int                                │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Random Forest Classifier                       │   │
│  │  ├── n_estimators: 150                                  │   │
│  │  ├── max_depth: 12                                      │   │
│  │  ├── min_samples_split: 5                               │   │
│  │  └── class_weight: balanced                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Output: Fraud Type + Confidence                         │   │
│  │  ├── genuine_fraud: Stolen card, unauthorized           │   │
│  │  ├── friendly_fraud: Legit buyer abusing                │   │
│  │  ├── account_takeover: Compromised account              │   │
│  │  └── technical_failure: System error                    │   │
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
│  │              ADAPTIVE TRAINER                            │   │
│  │  ├── Incremental learning with new data                 │   │
│  │  ├── A/B testing: static vs. adaptive                   │   │
│  │  └── Model versioning with MLflow                       │   │
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
│  │  ├── Rate Limiter (100 req/min)                         │   │
│  │  ├── Request Validation (Pydantic)                      │   │
│  │  ├── Authentication (API Key)                           │   │
│  │  ├── Logging (Structured JSON)                          │   │
│  │  └── Tracing (OpenTelemetry)                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Endpoints                                              │   │
│  │  ├── POST /api/v1/score                                 │   │
│  │  │   └─ Input: Transaction → Output: Risk Score + Cost  │   │
│  │  ├── POST /api/v1/classify                              │   │
│  │  │   └─ Input: Transaction → Output: Fraud Type         │   │
│  │  ├── POST /api/v1/evaluate                              │   │
│  │  │   └─ Input: Dataset → Output: Evaluation Report      │   │
│  │  ├── GET /api/v1/drift/status                           │   │
│  │  │   └─ Output: Drift Detector Status                   │   │
│  │  └── POST /api/v1/drift/simulate                        │   │
│  │      └─ Input: Scenario → Output: Drift Report          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2 Request/Response Schemas (`api/schemas.py`)

```python
# Request Schema
ScoreRequest:
  amount: float
  payment_method: PaymentMethod
  merchant_category: MerchantCategory
  is_new_device: bool
  is_new_address: bool
  account_age_days: int
  past_disputes: int

# Response Schema
ScoreResponse:
  risk_score: float (0.0 - 1.0)
  recommended_action: str (ALLOW / REVIEW / BLOCK)
  estimated_cost_if_fraud: float (₹)
  cost_matrix: CostBreakdown
  explanation: List[FeatureContribution]

# Classification Response
ClassifyResponse:
  fraud_type: FraudType
  confidence: float
  evidence_checklist: List[str]
  recommended_evidence: List[Document]
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
│  │  Cost Analysis                                           │   │
│  │  ├── Total Cost = (FN × FN_cost) + (FP × FP_cost)       │   │
│  │  ├── Cost Savings vs. No Model                          │   │
│  │  ├── Cost Savings vs. F1-Optimized                      │   │
│  │  └── ROI = (Cost Prevented) / (Model Cost)              │   │
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
└─────────────┘     └─────────────┘     │  Scorer     │     └──────┬──────┘
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
└─────────────┘     └─────────────┘     └──────┬──────┘
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
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                       ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                       │  Drift      │  │  Retraining │  │  A/B        │
                       │  Logged     │  │  Triggered  │  │  Comparison │
                       └─────────────┘  └─────────────┘  └─────────────┘
```

---

### 6. Technology Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| **Data** | Generation | Faker, NumPy, Pandas |
| **Data** | Storage | CSV (local), PostgreSQL (prod) |
| **Model** | ML | XGBoost, Scikit-learn, Pandas |
| **Model** | Drift | River, Alibi-detect |
| **Model** | Tracking | MLflow |
| **API** | Framework | FastAPI, Uvicorn |
| **API** | Validation | Pydantic |
| **API** | Auth | API Key, JWT |
| **API** | Docs | OpenAPI/Swagger |
| **Eval** | Metrics | Scikit-learn, NumPy |
| **Eval** | Viz | Matplotlib, Seaborn, Plotly |
| **Test** | Unit | Pytest |
| **Test** | Integration | Pytest-asyncio, Httpx |
| **Infra** | Container | Docker |
| **Infra** | CI/CD | GitHub Actions |

---

### 7. Security Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API Security                                                    │
│  ├── API Key authentication                                     │
│  ├── Rate limiting (100 req/min)                                │
│  ├── Input validation (Pydantic)                                │
│  ├── CORS configuration                                         │
│  └── HTTPS enforcement                                          │
│                                                                  │
│  Data Security                                                   │
│  ├── No PII in logs                                             │
│  ├── Anonymized customer IDs                                    │
│  ├── Encrypted model artifacts                                  │
│  └── Audit logging                                              │
│                                                                  │
│  Model Security                                                  │
│  ├── Adversarial robustness testing                             │
│  ├── Model versioning                                           │
│  ├── A/B testing for updates                                    │
│  └── Rollback capability                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 8. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT (Docker)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Container: ai-risk-manager                              │   │
│  │  ├── Port: 8000                                          │   │
│  │  ├── Health: /health                                     │   │
│  │  └── Metrics: /metrics                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Volumes                                                │   │
│  │  ├── /app/models/artifacts (model files)                │   │
│  │  ├── /app/data (training data)                          │   │
│  │  └── /app/logs (application logs)                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

This architecture ensures:
1. **Scalability**: Stateless API, horizontal scaling
2. **Reliability**: Health checks, graceful degradation
3. **Observability**: Structured logging, metrics, tracing
4. **Security**: Auth, rate limiting, input validation
5. **Maintainability**: Clear separation of concerns
6. **Testability**: Unit, integration, and API tests

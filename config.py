"""
config.py — Central configuration for the BNPL Risk Scoring System.
All tuneable hyperparameters, paths, and thresholds live here.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")

RAW_DATA_PATH   = os.path.join(DATA_DIR, "loan_data.csv")      # LendingClub CSV
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_data.csv")
FEAT_DATA_PATH  = os.path.join(DATA_DIR, "featured_data.csv")

MODEL_PATH      = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_PATH    = os.path.join(MODEL_DIR, "feature_names.pkl")

# ─────────────────────────────────────────────
# BNPL FILTERING CRITERIA
# ─────────────────────────────────────────────
MAX_LOAN_AMOUNT = 5000      # BNPL loans are small (≤ $5,000)
MAX_TERM_MONTHS = 36        # Short-term: 36-month max (prefer 36 mo)
BNPL_TERM_LIST  = [36]      # We keep only 36-month term loans

# ─────────────────────────────────────────────
# TARGET LABEL ENCODING
# ─────────────────────────────────────────────
# LendingClub "loan_status" → binary default flag
DEFAULT_STATUSES = [
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
]
SAFE_STATUSES = [
    "Fully Paid",
    "Current",
    "In Grace Period",
    "Does not meet the credit policy. Status:Fully Paid",
]

# ─────────────────────────────────────────────
# FEATURE ENGINEERING PARAMETERS
# ─────────────────────────────────────────────
HIGH_UTIL_THRESHOLD     = 0.75   # revolving utilization spike flag
HIGH_DTI_THRESHOLD      = 35     # debt-to-income spike flag
RECENT_INQ_WINDOW       = 6      # months for "recent" hard inquiries
SPENDING_SPIKE_MULT     = 2.0    # loan_amt > MULT × monthly_income → spike

# ─────────────────────────────────────────────
# CLASS IMBALANCE
# ─────────────────────────────────────────────
SMOTE_RANDOM_STATE      = 42
SMOTE_SAMPLING_STRATEGY = "minority"   # upsample minority only

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
CV_FOLDS        = 5

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
}

# LightGBM hyperparameters
LGBM_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "is_unbalance":     True,
    "verbose":          -1,
}

# ─────────────────────────────────────────────
# RISK SCORING & DECISION
# ─────────────────────────────────────────────
DEFAULT_THRESHOLD   = 0.58   # calibrated threshold (Youden J on actual data)
HIGH_RISK_THRESHOLD = 0.80   # probability >= threshold → HIGH RISK (reject hard)

RISK_BANDS = {
    "LOW":       (0.00, 0.40),
    "MEDIUM":    (0.40, 0.58),
    "HIGH":      (0.58, 0.80),
    "VERY_HIGH": (0.80, 1.01),
}

# ─────────────────────────────────────────────
# CASH FLOW ANALYSIS
# ─────────────────────────────────────────────
CASH_FLOW_WEIGHTS = {
    "savings_ratio":   0.40,
    "income":          0.30,
    "expense_burden":  0.20,   # inverted — lower burden = higher score
    "dti":             0.10,   # inverted — lower DTI = higher score
}

# ─────────────────────────────────────────────
# FINAL COMPOSITE SCORING
# ─────────────────────────────────────────────
SCORE_WEIGHT_PD         = 0.60   # weight for (1 - PD)
SCORE_WEIGHT_CASHFLOW   = 0.40   # weight for cash_flow_score

# Default decision thresholds (overridden by bandit optimizer)
APPROVE_THRESHOLD       = 0.55
REJECT_THRESHOLD        = 0.40

# ─────────────────────────────────────────────
# DRIFT DETECTION (PSI)
# ─────────────────────────────────────────────
PSI_BUCKETS             = 10
PSI_DRIFT_THRESHOLD     = 0.20   # PSI > 0.20 → significant drift
PSI_WARN_THRESHOLD      = 0.10   # PSI > 0.10 → moderate drift

# ─────────────────────────────────────────────
# MULTI-ARMED BANDIT
# ─────────────────────────────────────────────
BANDIT_STRATEGIES = {
    "conservative": {"pd_threshold": 0.20, "description": "Low risk tolerance"},
    "moderate":     {"pd_threshold": 0.35, "description": "Balanced approach"},
    "aggressive":   {"pd_threshold": 0.50, "description": "High risk tolerance"},
}
BANDIT_ROUNDS           = 1000
BANDIT_LOAN_TERM_YEARS  = 3      # 36 months

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_HOST    = "0.0.0.0"
API_PORT    = 8000
API_VERSION = "v1"

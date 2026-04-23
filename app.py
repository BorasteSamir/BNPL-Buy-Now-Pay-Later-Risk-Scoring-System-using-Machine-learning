"""
app.py — Production-ready FastAPI REST API for BNPL Risk Scoring.

Endpoints
─────────
POST /api/v1/score          → Single applicant risk score + decision
POST /api/v1/score/batch    → Batch scoring (list of applicants)
GET  /api/v1/health         → Health check
GET  /api/v1/model/info     → Model metadata and feature list
GET  /api/v1/threshold      → Current decision threshold
PUT  /api/v1/threshold      → Update decision threshold at runtime

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
import pickle
import time
import uuid
from typing import List, Optional, Dict, Any

import numpy  as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADER (singleton)
# ═════════════════════════════════════════════════════════════════════════════
class ModelRegistry:
    """Thread-safe singleton model store with hot-swap support."""

    _model        = None
    _scaler       = None
    _feature_names: List[str] = []
    _threshold    = config.DEFAULT_THRESHOLD
    _loaded_at    = None

    @classmethod
    def load(cls):
        for path, label in [
            (config.MODEL_PATH,   "model"),
            (config.SCALER_PATH,  "scaler"),
            (config.FEATURE_PATH, "feature list"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {label} file: {path}\n"
                    "Run model_training.py first to train and save the model."
                )

        with open(config.MODEL_PATH,   "rb") as f: cls._model         = pickle.load(f)
        with open(config.SCALER_PATH,  "rb") as f: cls._scaler        = pickle.load(f)
        with open(config.FEATURE_PATH, "rb") as f: cls._feature_names = pickle.load(f)
        cls._loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.info(f"✅ Model loaded ({len(cls._feature_names)} features)")

    @classmethod
    def predict(cls, feature_df: pd.DataFrame) -> np.ndarray:
        """Return raw default probabilities for a pre-processed DataFrame."""
        X_sc = cls._scaler.transform(feature_df[cls._feature_names])
        return cls._model.predict_proba(X_sc)[:, 1]

    @classmethod
    def is_ready(cls) -> bool:
        return cls._model is not None


# Try loading at import time (fails gracefully if model not yet trained)
try:
    ModelRegistry.load()
except FileNotFoundError as e:
    logger.warning(f"Model not loaded at startup: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════
class ApplicantInput(BaseModel):
    """
    Financial and behavioural inputs for a single BNPL applicant.
    All values represent the state *at application time*.

    Tips
    ────
    • annual_inc     : total annual income in USD
    • dti            : debt-to-income ratio  (e.g. 18.5 means 18.5 %)
    • revol_util     : revolving credit utilisation (e.g. 65.3 means 65.3 %)
    • loan_amnt      : requested BNPL amount in USD
    • inq_last_6mths : number of hard credit inquiries in last 6 months
    • delinq_2yrs    : number of late payments in last 2 years
    • grade          : loan grade 1=A (best) to 7=G (worst)
    """
    # ── Core loan details ────────────────────────────────────────────────
    loan_amnt      : float = Field(...,  gt=0,    le=5000,  description="Requested loan amount (USD)")
    int_rate       : float = Field(13.0, gt=0,    le=40,    description="Interest rate (%)")
    installment    : float = Field(150,  gt=0,              description="Monthly instalment (USD)")
    grade          : int   = Field(3,    ge=1,    le=7,     description="Credit grade 1=A … 7=G")
    sub_grade      : int   = Field(13,   ge=1,    le=35,    description="Sub-grade 1=A1 … 35=G5")

    # ── Borrower profile ─────────────────────────────────────────────────
    annual_inc     : float = Field(...,  gt=0,              description="Annual gross income (USD)")
    emp_length     : float = Field(5,    ge=0,    le=10,    description="Employment length (years, 10=10+)")
    dti            : float = Field(...,  ge=0,    le=100,   description="Debt-to-income ratio (%)")
    home_ownership_MORTGAGE : int = Field(0, ge=0, le=1)
    home_ownership_OWN      : int = Field(0, ge=0, le=1)
    home_ownership_RENT     : int = Field(0, ge=0, le=1)

    # ── Credit behaviour ─────────────────────────────────────────────────
    delinq_2yrs    : float = Field(0,    ge=0,              description="Delinquencies in last 2 years")
    inq_last_6mths : float = Field(0,    ge=0,              description="Hard inquiries in last 6 months")
    open_acc       : float = Field(8,    ge=0,              description="Open credit lines")
    pub_rec        : float = Field(0,    ge=0,              description="Public derogatory records")
    revol_bal      : float = Field(5000, ge=0,              description="Revolving credit balance (USD)")
    revol_util     : float = Field(40,   ge=0,    le=100,   description="Revolving utilisation (%)")
    total_acc      : float = Field(15,   ge=0,              description="Total credit accounts")
    acc_now_delinq : float = Field(0,    ge=0,              description="Accounts currently delinquent")
    tot_coll_amt   : float = Field(0,    ge=0,              description="Total collection amount (USD)")
    tot_cur_bal    : float = Field(20000, ge=0,             description="Total current balance (USD)")

    # ── Extended account info ────────────────────────────────────────────
    num_actv_rev_tl: float = Field(4,    ge=0)
    num_rev_accts  : float = Field(8,    ge=0)
    num_il_tl      : float = Field(4,    ge=0)
    num_tl_90g_dpd_24m : float = Field(0, ge=0)
    num_tl_30dpd   : float = Field(0,    ge=0)
    num_tl_op_past_12m : float = Field(1, ge=0)
    pct_tl_nvr_dlq : float = Field(95,   ge=0,    le=100)
    percent_bc_gt_75: float = Field(20,  ge=0,    le=100)
    pub_rec_bankruptcies: float = Field(0, ge=0)
    tax_liens       : float = Field(0,   ge=0)
    total_bc_limit  : float = Field(8000, ge=0)
    total_bal_ex_mort: float = Field(15000, ge=0)
    total_il_high_credit_limit: float = Field(20000, ge=0)
    tot_hi_cred_lim : float = Field(30000, ge=0)
    num_bc_sats     : float = Field(3,   ge=0)
    num_bc_tl       : float = Field(4,   ge=0)
    num_op_rev_tl   : float = Field(5,   ge=0)
    num_rev_tl_bal_gt_0: float = Field(3, ge=0)
    num_sats        : float = Field(6,   ge=0)
    num_tl_120dpd_2m: float = Field(0,   ge=0)
    num_actv_bc_tl  : float = Field(2,   ge=0)

    # ── Verification & purpose flags (one-hot) ───────────────────────────
    verification_status_Source_Verified : int = Field(0, ge=0, le=1)
    verification_status_Verified        : int = Field(0, ge=0, le=1)
    addr_state_freq  : float = Field(0.05, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 2500,
                "int_rate": 14.5,
                "installment": 85,
                "grade": 3,
                "sub_grade": 13,
                "annual_inc": 48000,
                "emp_length": 4,
                "dti": 18.5,
                "home_ownership_MORTGAGE": 0,
                "home_ownership_OWN": 0,
                "home_ownership_RENT": 1,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 4200,
                "revol_util": 55.0,
                "total_acc": 12,
                "acc_now_delinq": 0,
                "tot_coll_amt": 0,
                "tot_cur_bal": 22000,
                "num_actv_rev_tl": 3,
                "num_rev_accts": 7,
                "num_il_tl": 3,
                "num_tl_90g_dpd_24m": 0,
                "num_tl_30dpd": 0,
                "num_tl_op_past_12m": 1,
                "pct_tl_nvr_dlq": 96,
                "percent_bc_gt_75": 15,
                "pub_rec_bankruptcies": 0,
                "tax_liens": 0,
                "total_bc_limit": 9000,
                "total_bal_ex_mort": 14000,
                "total_il_high_credit_limit": 18000,
                "tot_hi_cred_lim": 28000,
                "num_bc_sats": 3,
                "num_bc_tl": 4,
                "num_op_rev_tl": 5,
                "num_rev_tl_bal_gt_0": 3,
                "num_sats": 6,
                "num_tl_120dpd_2m": 0,
                "num_actv_bc_tl": 2,
                "verification_status_Source_Verified": 0,
                "verification_status_Verified": 1,
                "addr_state_freq": 0.04,
            }
        }


class ScoreResponse(BaseModel):
    request_id    : str
    default_prob  : float
    risk_band     : str
    decision      : str
    threshold_used: float
    confidence    : str
    flags         : Dict[str, Any]
    latency_ms    : float


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.01, le=0.99)


class BatchInput(BaseModel):
    applicants: List[ApplicantInput]


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDER  (mirrors feature_engineering.py logic for inference)
# ═════════════════════════════════════════════════════════════════════════════
def _safe_div_s(a, b, fill=0.0):
    return a / b if b != 0 else fill


def build_inference_features(inp: ApplicantInput) -> pd.DataFrame:
    """
    Reconstruct the full feature vector from raw API inputs.
    Must produce *exactly* the same columns as the training pipeline.
    """
    d = inp.dict()   # raw field values

    # ── Derived base values ───────────────────────────────────────────────
    monthly_inc = d["annual_inc"] / 12

    # ── Payment behaviour
    d["feat_has_delinq"]         = int(d["delinq_2yrs"] > 0)
    d["feat_delinq_severity"]    = float(np.log1p(d["delinq_2yrs"]))
    d["feat_currently_delinq"]   = int(d["acc_now_delinq"] > 0)
    d["feat_has_collections"]    = int(d["tot_coll_amt"] > 0)
    d["feat_log_coll_amt"]       = float(np.log1p(d["tot_coll_amt"]))
    d["feat_severe_late_accts"]  = float(d["num_tl_90g_dpd_24m"])
    d["feat_late_30dpd"]         = float(d["num_tl_30dpd"])
    d["feat_pct_clean_accts"]    = d["pct_tl_nvr_dlq"] / 100.0

    # ── Debt burden
    d["feat_dti_log"]            = float(np.log1p(max(d["dti"], 0)))
    d["feat_high_dti_flag"]      = int(d["dti"] >= config.HIGH_DTI_THRESHOLD)
    d["feat_monthly_inc"]        = monthly_inc
    d["feat_inst_to_inc"]        = _safe_div_s(d["installment"], monthly_inc)
    d["feat_high_inst_flag"]     = int(d["feat_inst_to_inc"] > 0.20)
    d["feat_bal_to_inc"]         = _safe_div_s(d["tot_cur_bal"], d["annual_inc"])
    d["feat_nonmort_debt_ratio"] = _safe_div_s(d["total_bal_ex_mort"], d["annual_inc"])

    # ── Credit utilisation
    util_frac = d["revol_util"] / 100.0
    d["feat_revol_util_pct"]     = util_frac
    d["feat_high_util_flag"]     = int(util_frac >= config.HIGH_UTIL_THRESHOLD)
    d["feat_util_squared"]       = util_frac ** 2
    d["feat_pct_bc_high_util"]   = d["percent_bc_gt_75"] / 100.0
    d["feat_revol_to_limit"]     = _safe_div_s(d["revol_bal"], d["total_bc_limit"])

    # ── Inquiry / activity
    d["feat_recent_inq"]         = float(d["inq_last_6mths"])
    d["feat_inq_spike_flag"]     = int(d["inq_last_6mths"] >= 3)
    d["feat_log_inq"]            = float(np.log1p(d["inq_last_6mths"]))
    d["feat_inq_to_open_accts"]  = _safe_div_s(d["inq_last_6mths"], d["open_acc"])
    d["feat_new_tl_12m"]         = float(d["num_tl_op_past_12m"])
    d["feat_acct_surge_flag"]    = int(d["num_tl_op_past_12m"] >= 3)

    # ── Loan sizing
    d["feat_loan_to_income"]     = _safe_div_s(d["loan_amnt"], d["annual_inc"])
    d["feat_spending_spike"]     = int(d["loan_amnt"] >= config.SPENDING_SPIKE_MULT * monthly_inc)
    d["feat_loan_to_total_bal"]  = _safe_div_s(d["loan_amnt"], d["tot_cur_bal"])
    total_cost                   = d["installment"] * 36
    d["feat_total_cost_to_inc"]  = _safe_div_s(total_cost, d["annual_inc"])

    # ── Credit depth
    d["feat_open_acc_log"]       = float(np.log1p(d["open_acc"]))
    d["feat_active_rev_ratio"]   = _safe_div_s(d["num_actv_rev_tl"], max(d["num_rev_accts"], 1))
    d["feat_install_share"]      = _safe_div_s(d["num_il_tl"], max(d["total_acc"], 1))
    d["feat_has_pub_rec"]        = int(d["pub_rec"] > 0)
    d["feat_has_bankruptcy"]     = int(d["pub_rec_bankruptcies"] > 0)
    d["feat_has_tax_lien"]       = int(d["tax_liens"] > 0)
    d["feat_grade_risk"]         = float(d["grade"])
    d["feat_subgrade_risk"]      = float(d["sub_grade"])
    d["feat_int_rate_risk"]      = d["int_rate"] / 30.0

    # ── Composite flags
    flag_keys = [
        "feat_has_delinq", "feat_currently_delinq", "feat_has_collections",
        "feat_high_dti_flag", "feat_high_util_flag", "feat_inq_spike_flag",
        "feat_spending_spike", "feat_has_pub_rec", "feat_has_bankruptcy",
        "feat_high_inst_flag", "feat_acct_surge_flag", "feat_severe_late_accts",
    ]
    d["feat_composite_risk_score"] = sum(d.get(k, 0) for k in flag_keys)
    d["feat_multi_flag"]           = int(d["feat_composite_risk_score"] >= 3)

    return pd.DataFrame([d])


# ═════════════════════════════════════════════════════════════════════════════
# RISK SCORING LOGIC
# ═════════════════════════════════════════════════════════════════════════════
def get_risk_band(prob: float) -> str:
    for band, (lo, hi) in config.RISK_BANDS.items():
        if lo <= prob < hi:
            return band
    return "VERY_HIGH"


def score_applicant(inp: ApplicantInput, threshold: float) -> dict:
    t0       = time.time()
    feat_df  = build_inference_features(inp)

    # Align columns to training feature list — fill zeros for any gap
    for col in ModelRegistry._feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    prob       = float(ModelRegistry.predict(feat_df)[0])
    risk_band  = get_risk_band(prob)
    decision   = "REJECT" if prob >= threshold else "APPROVE"
    confidence = (
        "HIGH"   if abs(prob - threshold) > 0.20
        else "MEDIUM" if abs(prob - threshold) > 0.10
        else "LOW"
    )

    # Human-readable flags
    d = inp.dict()
    flags = {
        "high_dti":        d["dti"] >= config.HIGH_DTI_THRESHOLD,
        "high_utilisation": (d["revol_util"] / 100) >= config.HIGH_UTIL_THRESHOLD,
        "recent_delinq":   d["delinq_2yrs"] > 0,
        "inq_spike":       d["inq_last_6mths"] >= 3,
        "spending_spike":  d["loan_amnt"] >= config.SPENDING_SPIKE_MULT * (d["annual_inc"] / 12),
        "collections":     d["tot_coll_amt"] > 0,
        "bankruptcy":      d["pub_rec_bankruptcies"] > 0,
    }

    latency_ms = (time.time() - t0) * 1000
    return {
        "request_id":    str(uuid.uuid4()),
        "default_prob":  round(prob, 4),
        "risk_band":     risk_band,
        "decision":      decision,
        "threshold_used": threshold,
        "confidence":    confidence,
        "flags":         flags,
        "latency_ms":    round(latency_ms, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="BNPL Risk Scoring API",
    description=(
        "Production-grade REST API for Buy Now Pay Later credit risk assessment.\n\n"
        "Returns a default probability, risk band (LOW / MEDIUM / HIGH / VERY_HIGH), "
        "and a binary APPROVE / REJECT decision."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for demo; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/health", tags=["System"])
def health():
    return {
        "status":     "ok" if ModelRegistry.is_ready() else "degraded",
        "model_ready": ModelRegistry.is_ready(),
        "loaded_at":   ModelRegistry._loaded_at,
    }


@app.get("/api/v1/model/info", tags=["System"])
def model_info():
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "n_features":    len(ModelRegistry._feature_names),
        "feature_names": ModelRegistry._feature_names,
        "threshold":     ModelRegistry._threshold,
        "risk_bands":    config.RISK_BANDS,
    }


@app.get("/api/v1/threshold", tags=["System"])
def get_threshold():
    return {"threshold": ModelRegistry._threshold}


@app.put("/api/v1/threshold", tags=["System"])
def update_threshold(body: ThresholdUpdate):
    old = ModelRegistry._threshold
    ModelRegistry._threshold = body.threshold
    logger.info(f"Threshold updated: {old} → {body.threshold}")
    return {"old_threshold": old, "new_threshold": body.threshold}


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
def score_single(applicant: ApplicantInput):
    """
    Score a single BNPL applicant.

    Returns
    -------
    • **default_prob**  — probability of default (0 → 1)
    • **risk_band**     — LOW / MEDIUM / HIGH / VERY_HIGH
    • **decision**      — APPROVE or REJECT
    • **flags**         — human-readable risk triggers
    """
    if not ModelRegistry.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready. Run model_training.py first.",
        )
    try:
        result = score_applicant(applicant, ModelRegistry._threshold)
        return ScoreResponse(**result)
    except Exception as e:
        logger.exception("Scoring error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/score/batch", tags=["Scoring"])
def score_batch(batch: BatchInput):
    """
    Score a list of BNPL applicants in one call (max 100).
    """
    if not ModelRegistry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    if len(batch.applicants) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 100")

    results = []
    for applicant in batch.applicants:
        try:
            r = score_applicant(applicant, ModelRegistry._threshold)
            results.append(r)
        except Exception as e:
            results.append({"error": str(e)})

    approved = sum(1 for r in results if r.get("decision") == "APPROVE")
    return {
        "total":    len(results),
        "approved": approved,
        "rejected": len(results) - approved,
        "approval_rate": round(approved / len(results), 4),
        "results":  results,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )

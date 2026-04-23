"""
dashboard.py — Interactive Streamlit dashboard for the BNPL Risk Scoring System.

Features:
  • Live single-applicant risk scoring with adjustable sliders
  • Risk gauge and probability visualisation
  • Feature importance chart (loaded from saved reports)
  • Batch simulation with downloadable results
  • Threshold tuning with real-time impact preview
  • SHAP waterfall explanation (if SHAP plots are available)

Run:
    streamlit run dashboard.py
"""

import os, sys, json, time, subprocess
import pandas as pd
import numpy  as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express       as px
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

API_BASE = f"http://localhost:{config.API_PORT}/api/{config.API_VERSION}"

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-START API SERVER
# ─────────────────────────────────────────────────────────────────────────────
def _api_is_up() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def _ensure_api_running():
    """Start the API server in the background if it is not already running."""
    if _api_is_up():
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "0.0.0.0", "--port", str(config.API_PORT)],
        cwd=base_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 30 s for the API to become ready
    for _ in range(30):
        time.sleep(1)
        if _api_is_up():
            return

_ensure_api_running()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BNPL Risk Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

    .main-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }

    .metric-card {
        background: #1e293b; border-radius: 12px; padding: 1.2rem;
        border: 1px solid #334155; text-align: center;
    }
    .metric-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .08em; }
    .metric-value { font-size: 2rem; font-weight: 700; margin-top: 0.3rem; }

    .approve-badge {
        background: #064e3b; color: #34d399; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 700; font-size: 1.4rem;
        display: inline-block; letter-spacing: .05em;
    }
    .reject-badge {
        background: #450a0a; color: #f87171; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 700; font-size: 1.4rem;
        display: inline-block; letter-spacing: .05em;
    }
    .flag-chip {
        background: #431407; color: #fb923c; border-radius: 999px;
        padding: 0.25rem 0.75rem; font-size: 0.78rem;
        display: inline-block; margin: 0.2rem;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #e2e8f0;
        border-bottom: 2px solid #334155; padding-bottom: 0.4rem;
        margin: 1rem 0 0.8rem 0;
    }
    .stSlider > div > div { background: #334155; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "offline"}


def call_score(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/score", json=payload, timeout=10)
        return r.json()
    except Exception:
        return None


def update_threshold(t: float):
    try:
        requests.put(f"{API_BASE}/threshold", json={"threshold": t}, timeout=5)
    except Exception:
        pass


def make_gauge(prob: float, threshold: float) -> go.Figure:
    color = (
        "#34d399" if prob < 0.25
        else "#fbbf24" if prob < threshold
        else "#f87171"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        title={"text": "Default Probability", "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar":  {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,   25],  "color": "#064e3b"},
                {"range": [25,  int(threshold * 100)], "color": "#422006"},
                {"range": [int(threshold * 100), 100], "color": "#450a0a"},
            ],
            "threshold": {
                "line": {"color": "#f8fafc", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=40, b=10, l=20, r=20),
        height=260,
        font={"color": "#e2e8f0"},
    )
    return fig


def make_bar(prob: float) -> go.Figure:
    bands = ["LOW\n0–25%", "MEDIUM\n25–40%", "HIGH\n40–65%", "VERY HIGH\n65–100%"]
    edges = [0, 0.25, 0.40, 0.65, 1.00]
    vals  = [min(max(prob - edges[i], 0), edges[i+1] - edges[i]) for i in range(4)]
    total = sum(vals)
    vals  = [v / total if total > 0 else 0 for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=bands,
        orientation="h",
        marker_color=["#34d399", "#fbbf24", "#fb923c", "#f87171"],
        text=[f"{v*100:.0f}%" for v in vals],
        textposition="inside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10),
        height=160,
        font={"color": "#e2e8f0"},
        xaxis={"visible": False},
        yaxis={"color": "#94a3b8"},
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    health = api_health()
    status_color = "🟢" if health["status"] == "ok" else "🔴"
    st.markdown(f"**API Status**: {status_color} {health['status'].upper()}")

    st.markdown("---")
    st.markdown("### Decision Threshold")
    threshold = st.slider(
        "Default probability threshold",
        min_value=0.10, max_value=0.80,
        value=config.DEFAULT_THRESHOLD,
        step=0.01,
        help="Applicants with probability ≥ threshold are REJECTED.",
    )
    if st.button("Apply Threshold"):
        update_threshold(threshold)
        st.success(f"Threshold updated to {threshold:.2f}")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🔍 Score Applicant", "📦 Batch Simulation", "📊 Model Reports"])


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">💳 BNPL Risk Scoring System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time creditworthiness assessment for Buy Now Pay Later financing</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SCORE APPLICANT
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔍 Score Applicant":

    # Initialize session state with default values
    if "loan_amnt" not in st.session_state:
        st.session_state.loan_amnt = 2000
    if "annual_inc" not in st.session_state:
        st.session_state.annual_inc = 48000
    if "installment" not in st.session_state:
        st.session_state.installment = 85.0
    if "dti" not in st.session_state:
        st.session_state.dti = 18.5
    if "revol_util" not in st.session_state:
        st.session_state.revol_util = 40.0
    if "grade" not in st.session_state:
        st.session_state.grade = 3
    if "int_rate" not in st.session_state:
        st.session_state.int_rate = 13.0
    if "delinq" not in st.session_state:
        st.session_state.delinq = 0
    if "inq" not in st.session_state:
        st.session_state.inq = 1
    if "revol_bal" not in st.session_state:
        st.session_state.revol_bal = 4200
    if "open_acc" not in st.session_state:
        st.session_state.open_acc = 8
    if "home_rent" not in st.session_state:
        st.session_state.home_rent = True

    col_inp, col_res = st.columns([1.3, 1], gap="large")

    with col_inp:
        # Preset Examples at the top
        st.markdown('<div class="section-header">Quick Examples</div>', unsafe_allow_html=True)
        ex1, ex2 = st.columns(2)
        
        if ex1.button("✅ Load APPROVE Example", use_container_width=True):
            # High income, low DTI, clean history
            st.session_state.loan_amnt = 1500
            st.session_state.annual_inc = 95000
            st.session_state.installment = 47.0
            st.session_state.dti = 8.0
            st.session_state.revol_util = 8.0
            st.session_state.grade = 1
            st.session_state.int_rate = 7.5
            st.session_state.delinq = 0
            st.session_state.inq = 0
            st.session_state.revol_bal = 2000
            st.session_state.open_acc = 12
            st.session_state.home_rent = False  # Owns home
            st.rerun()
            
        if ex2.button("❌ Load REJECT Example", use_container_width=True):
            # Low income, high DTI, bad history
            st.session_state.loan_amnt = 4800
            st.session_state.annual_inc = 18000
            st.session_state.installment = 185.0
            st.session_state.dti = 55.0
            st.session_state.revol_util = 95.0
            st.session_state.grade = 7
            st.session_state.int_rate = 28.0
            st.session_state.delinq = 5
            st.session_state.inq = 6
            st.session_state.revol_bal = 18000
            st.session_state.open_acc = 3
            st.session_state.home_rent = True
            st.rerun()

        st.markdown('<div class="section-header">Loan Request</div>', unsafe_allow_html=True)
        
        # Loan Amount - Green for small amounts, red for large
        loan_color = "🟢" if st.session_state.loan_amnt <= 2000 else "🟡" if st.session_state.loan_amnt <= 3500 else "🔴"
        loan_amnt = st.slider(
            f"{loan_color} Loan Amount ($)", 
            100, 5000, 
            value=st.session_state.loan_amnt, 
            step=50,
            key="loan_amnt",
            help="🟢 ≤$2000 (Good) | 🟡 $2000-3500 (OK) | 🔴 >$3500 (Risky)"
        )
        
        # Annual Income - Green for high income, red for low
        inc_color = "🟢" if st.session_state.annual_inc >= 60000 else "🟡" if st.session_state.annual_inc >= 35000 else "🔴"
        annual_inc = st.number_input(
            f"{inc_color} Annual Income ($)", 
            value=st.session_state.annual_inc, 
            step=1000,
            key="annual_inc",
            help="🟢 ≥$60k (Good) | 🟡 $35k-60k (OK) | 🔴 <$35k (Risky)"
        )
        
        # Monthly Installment
        installment = st.number_input(
            "Monthly Installment ($)", 
            value=st.session_state.installment, 
            step=5.0,
            key="installment"
        )

        st.markdown('<div class="section-header">Credit Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            # DTI - Green for low DTI, red for high
            dti_color = "🟢" if st.session_state.dti <= 15 else "🟡" if st.session_state.dti <= 25 else "🔴"
            dti = st.slider(
                f"{dti_color} DTI (%)", 
                0.0, 60.0, 
                value=st.session_state.dti, 
                step=0.5,
                key="dti",
                help="🟢 ≤15% (Excellent) | 🟡 15-25% (Good) | 🔴 >25% (Risky)"
            )
            
            # Grade - Green for A-C, red for E-G
            grade_color = "🟢" if st.session_state.grade <= 3 else "🟡" if st.session_state.grade <= 4 else "🔴"
            grade = st.selectbox(
                f"{grade_color} Grade (1=A → 7=G)", 
                [1,2,3,4,5,6,7], 
                index=st.session_state.grade-1,
                key="grade",
                help="🟢 A-C (Good) | 🟡 D (OK) | 🔴 E-G (Risky)"
            )
        
        with c2:
            # Revolving Utilization - Green for low util, red for high
            util_color = "🟢" if st.session_state.revol_util <= 30 else "🟡" if st.session_state.revol_util <= 60 else "🔴"
            revol_util = st.slider(
                f"{util_color} Revolving Util (%)", 
                0.0, 100.0, 
                value=st.session_state.revol_util, 
                step=1.0,
                key="revol_util",
                help="🟢 ≤30% (Good) | 🟡 30-60% (OK) | 🔴 >60% (High Risk)"
            )
            
            # Interest Rate - Green for low rates, red for high
            rate_color = "🟢" if st.session_state.int_rate <= 12 else "🟡" if st.session_state.int_rate <= 20 else "🔴"
            int_rate = st.slider(
                f"{rate_color} Interest Rate (%)", 
                5.0, 36.0, 
                value=st.session_state.int_rate, 
                step=0.5,
                key="int_rate",
                help="🟢 ≤12% (Prime) | 🟡 12-20% (Standard) | 🔴 >20% (Subprime)"
            )

        st.markdown('<div class="section-header">Behavioral Signals</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        
        with c3:
            # Delinquencies - Green for 0, red for any
            delinq_color = "🟢" if st.session_state.delinq == 0 else "🟡" if st.session_state.delinq <= 1 else "🔴"
            delinq = st.number_input(
                f"{delinq_color} Delinquencies (2 yr)", 
                min_value=0, max_value=20, 
                value=st.session_state.delinq,
                key="delinq",
                help="🟢 0 (Clean) | 🟡 1 (Minor) | 🔴 2+ (Major Risk)"
            )
            
            # Revolving Balance
            revol_bal = st.number_input(
                "Revolving Balance ($)", 
                min_value=0, 
                value=st.session_state.revol_bal, 
                step=100,
                key="revol_bal"
            )
        
        with c4:
            # Hard Inquiries - Green for 0-1, red for many
            inq_color = "🟢" if st.session_state.inq <= 1 else "🟡" if st.session_state.inq <= 2 else "🔴"
            inq = st.number_input(
                f"{inq_color} Hard Inquiries (6 mo)", 
                min_value=0, max_value=20, 
                value=st.session_state.inq,
                key="inq",
                help="🟢 0-1 (Good) | 🟡 2 (OK) | 🔴 3+ (Credit Seeking)"
            )
            
            # Open Accounts - Green for good mix
            acc_color = "🟢" if 8 <= st.session_state.open_acc <= 15 else "🟡" if 5 <= st.session_state.open_acc <= 20 else "🔴"
            open_acc = st.number_input(
                f"{acc_color} Open Accounts", 
                min_value=0, 
                value=st.session_state.open_acc,
                key="open_acc",
                help="🟢 8-15 (Good Mix) | 🟡 5-20 (OK) | 🔴 <5 or >20 (Thin/Overextended)"
            )

        # Home Ownership - Green for own/mortgage, yellow for rent
        home_color = "🟡" if st.session_state.home_rent else "🟢"
        home_rent = st.checkbox(
            f"{home_color} Renting (unchecked = owns home)", 
            value=st.session_state.home_rent,
            key="home_rent",
            help="🟢 Owns/Mortgage (Stable) | 🟡 Renting (Less Stable)"
        )

        score_btn = st.button("🚀 Score Applicant", type="primary", use_container_width=True)

    with col_res:
        st.markdown('<div class="section-header">Risk Decision</div>', unsafe_allow_html=True)

        if score_btn or "last_result" in st.session_state:
            if score_btn:
                payload = {
                    "loan_amnt": float(st.session_state.loan_amnt),
                    "int_rate":  float(st.session_state.int_rate),
                    "installment": float(st.session_state.installment),
                    "grade":     int(st.session_state.grade),
                    "sub_grade": int(st.session_state.grade) * 5 - 2,
                    "annual_inc": float(st.session_state.annual_inc),
                    "emp_length": 5,
                    "dti":       float(st.session_state.dti),
                    "home_ownership_MORTGAGE": 0 if st.session_state.home_rent else 1,
                    "home_ownership_OWN": 0,
                    "home_ownership_RENT": 1 if st.session_state.home_rent else 0,
                    "delinq_2yrs": float(st.session_state.delinq),
                    "inq_last_6mths": float(st.session_state.inq),
                    "open_acc":   float(st.session_state.open_acc),
                    "pub_rec":    0,
                    "revol_bal":  float(st.session_state.revol_bal),
                    "revol_util": float(st.session_state.revol_util),
                    "total_acc":  15,
                    "acc_now_delinq": 0,
                    "tot_coll_amt": 0,
                    "tot_cur_bal": 22000,
                    "num_actv_rev_tl": 3, "num_rev_accts": 7,
                    "num_il_tl": 3, "num_tl_90g_dpd_24m": 0,
                    "num_tl_30dpd": 0, "num_tl_op_past_12m": 1,
                    "pct_tl_nvr_dlq": 96, "percent_bc_gt_75": 15,
                    "pub_rec_bankruptcies": 0, "tax_liens": 0,
                    "total_bc_limit": 9000, "total_bal_ex_mort": 14000,
                    "total_il_high_credit_limit": 18000, "tot_hi_cred_lim": 28000,
                    "num_bc_sats": 3, "num_bc_tl": 4, "num_op_rev_tl": 5,
                    "num_rev_tl_bal_gt_0": 3, "num_sats": 6, "num_tl_120dpd_2m": 0,
                    "num_actv_bc_tl": 2,
                    "verification_status_Source_Verified": 0,
                    "verification_status_Verified": 1,
                    "addr_state_freq": 0.04,
                }

                with st.spinner("Scoring …"):
                    result = call_score(payload)

                if result:
                    st.session_state["last_result"] = result
                else:
                    st.error("API unavailable — start the API server first.")
                    result = None
            else:
                result = st.session_state["last_result"]

            if result:
                prob     = result["default_prob"]
                decision = result["decision"]
                band     = result["risk_band"]
                flags    = result["flags"]

                # Decision badge
                badge_cls = "approve-badge" if decision == "APPROVE" else "reject-badge"
                badge_txt = "✅ APPROVED" if decision == "APPROVE" else "❌ REJECTED"
                st.markdown(f'<div style="text-align:center;margin:0.5rem 0;">'
                            f'<span class="{badge_cls}">{badge_txt}</span></div>',
                            unsafe_allow_html=True)

                # Gauge
                st.plotly_chart(make_gauge(prob, threshold), use_container_width=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Default Prob", f"{prob*100:.1f}%")
                m2.metric("Risk Band", band)
                m3.metric("Confidence", result["confidence"])

                # Risk flags
                active_flags = [k for k, v in flags.items() if v]
                if active_flags:
                    st.markdown("**⚠️ Risk Flags:**")
                    flag_labels = {
                        "high_dti":         "High DTI",
                        "high_utilisation":  "High Util",
                        "recent_delinq":     "Delinquency",
                        "inq_spike":         "Inq Spike",
                        "spending_spike":    "Spend Spike",
                        "collections":       "Collections",
                        "bankruptcy":        "Bankruptcy",
                    }
                    chips = " ".join(
                        f'<span class="flag-chip">{flag_labels.get(f, f)}</span>'
                        for f in active_flags
                    )
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.success("✓ No major risk flags")
        else:
            st.info("Configure the applicant details on the left and click **Score Applicant**.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📦 Batch Simulation":
    st.markdown('<div class="section-header">Batch Applicant Simulation</div>', unsafe_allow_html=True)

    n = st.slider("Number of simulated applicants", 5, 100, 20)

    if st.button("▶️ Run Batch Simulation", type="primary"):
        import random
        random.seed(42)

        def rand_applicant():
            # Create more realistic mix - 65% good profiles, 35% risky
            profile_type = random.choices(["good", "average", "risky"], weights=[40, 25, 35])[0]
            
            if profile_type == "good":
                # Good profile - likely to be approved
                dti_v = round(random.uniform(5, 20), 1)
                util_v = round(random.uniform(5, 40), 1)
                inc_v = round(random.uniform(45000, 120000), 0)
                loan_v = round(random.uniform(500, 2500), 0)
                grade_v = random.choices([1, 2, 3, 4], weights=[20, 30, 30, 20])[0]
                delinq_v = random.choices([0, 1], weights=[85, 15])[0]
                inq_v = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
                
            elif profile_type == "average":
                # Average profile - mixed results
                dti_v = round(random.uniform(15, 35), 1)
                util_v = round(random.uniform(30, 70), 1)
                inc_v = round(random.uniform(30000, 65000), 0)
                loan_v = round(random.uniform(1500, 4000), 0)
                grade_v = random.choices([3, 4, 5], weights=[30, 40, 30])[0]
                delinq_v = random.choices([0, 1, 2], weights=[60, 30, 10])[0]
                inq_v = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
                
            else:  # risky
                # Risky profile - likely to be rejected
                dti_v = round(random.uniform(30, 60), 1)
                util_v = round(random.uniform(60, 95), 1)
                inc_v = round(random.uniform(18000, 40000), 0)
                loan_v = round(random.uniform(3000, 5000), 0)
                grade_v = random.choices([4, 5, 6, 7], weights=[20, 30, 30, 20])[0]
                delinq_v = random.choices([0, 1, 2, 3, 4, 5], weights=[20, 25, 25, 15, 10, 5])[0]
                inq_v = random.choices([1, 2, 3, 4, 5, 6], weights=[20, 25, 25, 15, 10, 5])[0]
            
            return {
                "loan_amnt": float(loan_v), 
                "int_rate": float(grade_v * 3 + random.uniform(5, 8)),
                "installment": round(float(loan_v) / 36 * random.uniform(1.1, 1.2), 2),
                "grade": grade_v, 
                "sub_grade": grade_v * 5 - random.randint(0, 4),
                "annual_inc": float(inc_v), 
                "emp_length": random.randint(0, 10),
                "dti": dti_v, 
                "home_ownership_MORTGAGE": random.choices([0, 1], weights=[70, 30])[0],
                "home_ownership_OWN": random.choices([0, 1], weights=[85, 15])[0],
                "home_ownership_RENT": 1,  # Will be overridden by above
                "delinq_2yrs": float(delinq_v), 
                "inq_last_6mths": float(inq_v),
                "open_acc": float(random.randint(4, 18)), 
                "pub_rec": float(random.choices([0, 1, 2], weights=[80, 15, 5])[0]),
                "revol_bal": float(random.randint(0, int(inc_v * 0.3))), 
                "revol_util": util_v,
                "total_acc": float(random.randint(8, 25)), 
                "acc_now_delinq": float(random.choices([0, 1], weights=[90, 10])[0]),
                "tot_coll_amt": float(random.choices([0, 0, 0, random.randint(100, 2000)], weights=[85, 5, 5, 5])[0]),
                "tot_cur_bal": float(random.randint(int(inc_v * 0.1), int(inc_v * 0.8))),
                "num_actv_rev_tl": random.randint(2, 6), 
                "num_rev_accts": random.randint(3, 12), 
                "num_il_tl": random.randint(1, 5),
                "num_tl_90g_dpd_24m": max(0, delinq_v - 1), 
                "num_tl_30dpd": random.randint(0, 2),
                "num_tl_op_past_12m": random.randint(0, 3), 
                "pct_tl_nvr_dlq": max(60, 100 - delinq_v * 8 - random.randint(0, 10)),
                "percent_bc_gt_75": int(util_v * random.uniform(0.6, 1.2)), 
                "pub_rec_bankruptcies": random.choices([0, 1], weights=[95, 5])[0],
                "tax_liens": random.choices([0, 1], weights=[98, 2])[0], 
                "total_bc_limit": random.randint(3000, 25000), 
                "total_bal_ex_mort": random.randint(2000, 30000),
                "total_il_high_credit_limit": random.randint(5000, 40000), 
                "tot_hi_cred_lim": random.randint(10000, 60000),
                "num_bc_sats": random.randint(1, 8), 
                "num_bc_tl": random.randint(2, 8), 
                "num_op_rev_tl": random.randint(2, 10),
                "num_rev_tl_bal_gt_0": random.randint(1, 6), 
                "num_sats": random.randint(3, 12), 
                "num_tl_120dpd_2m": random.choices([0, 1], weights=[95, 5])[0],
                "num_actv_bc_tl": random.randint(1, 5), 
                "verification_status_Source_Verified": random.choices([0, 1], weights=[70, 30])[0],
                "verification_status_Verified": random.choices([0, 1], weights=[60, 40])[0], 
                "addr_state_freq": round(random.uniform(0.01, 0.08), 3),
            }

        # Fix home ownership (only one can be 1)
        def fix_home_ownership(applicant):
            if applicant["home_ownership_MORTGAGE"] == 1:
                applicant["home_ownership_OWN"] = 0
                applicant["home_ownership_RENT"] = 0
            elif applicant["home_ownership_OWN"] == 1:
                applicant["home_ownership_MORTGAGE"] = 0
                applicant["home_ownership_RENT"] = 0
            else:
                applicant["home_ownership_MORTGAGE"] = 0
                applicant["home_ownership_OWN"] = 0
                applicant["home_ownership_RENT"] = 1
            return applicant

        applicants = [fix_home_ownership(rand_applicant()) for _ in range(n)]

        with st.spinner(f"Scoring {n} applicants …"):
            try:
                r = requests.post(f"{API_BASE}/score/batch",
                                  json={"applicants": applicants}, timeout=60)
                data = r.json()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        results = data["results"]

        # Summary row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", data["total"])
        c2.metric("✅ Approved", data["approved"])
        c3.metric("❌ Rejected", data["rejected"])
        c4.metric("Approval Rate", f"{data['approval_rate']*100:.1f}%")

        # Table
        rows = []
        for i, r_item in enumerate(results):
            rows.append({
                "#": i + 1,
                "Decision":   r_item.get("decision", "ERR"),
                "Prob (%)":   round(r_item.get("default_prob", 0) * 100, 1),
                "Risk Band":  r_item.get("risk_band", "–"),
                "Confidence": r_item.get("confidence", "–"),
            })
        df_res = pd.DataFrame(rows)

        def style_decision(val):
            return "color: #34d399" if val == "APPROVE" else "color: #f87171"

        st.dataframe(
            df_res.style.applymap(style_decision, subset=["Decision"]),
            use_container_width=True, height=400,
        )

        # Distribution chart
        prob_vals = [r_item.get("default_prob", 0) for r_item in results if "default_prob" in r_item]
        fig = px.histogram(
            x=prob_vals, nbins=20,
            labels={"x": "Default Probability", "count": "Applicants"},
            title="Distribution of Default Probabilities",
            color_discrete_sequence=["#6366f1"],
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Threshold={threshold}", annotation_position="top right")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        # Download
        csv = df_res.to_csv(index=False)
        st.download_button("⬇️ Download Results CSV", csv, "batch_results.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL REPORTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Reports":
    st.markdown('<div class="section-header">Modular Pipeline Reports</div>', unsafe_allow_html=True)

    image_files = {
        "Data Drift (PSI)":     "psi_drift.png",
        "ROC Curves":          "roc_curves.png",
        "Confusion Matrix":     "confusion_matrix.png",
        "Feature Importance":   "feature_importance.png",
        "Global SHAP":          "shap_summary.png",
        "Local SHAP Waterfall": "shap_waterfall.png",
        "Bandit Convergence":   "bandit_convergence.png",
        "Bandit Strategies":    "bandit_strategies.png",
        "Final Score Range":    "final_score_distribution.png",
        "Approval Decisions":   "decision_pie.png",
    }

    tabs = st.tabs(list(image_files.keys()))
    for tab, (label, fname) in zip(tabs, image_files.items()):
        with tab:
            path = os.path.join(config.REPORTS_DIR, fname)
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.info(f"Image not yet generated: {fname}\nRun the pipeline to produce it.")

"""
simulate_checkout.py — Real-time BNPL checkout flow simulator.

Simulates a merchant checkout where a customer requests BNPL financing.
Sends customer data to the running API and displays an instant decision.

Usage:
    python simulate_checkout.py                  # interactive mode
    python simulate_checkout.py --profile risky  # preset profiles
    python simulate_checkout.py --batch          # batch simulation of 10 customers

Profiles: safe | average | risky | borderline
"""

import sys, os, time, json, argparse, random
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

API_BASE = f"http://localhost:{config.API_PORT}/api/{config.API_VERSION}"


# ═════════════════════════════════════════════════════════════════════════════
# CUSTOMER PROFILES
# ═════════════════════════════════════════════════════════════════════════════
PROFILES = {
    "safe": {
        "_label": "Safe Customer — Low Risk",
        "loan_amnt": 1500,
        "int_rate": 9.5,
        "installment": 46,
        "grade": 1,
        "sub_grade": 2,
        "annual_inc": 72000,
        "emp_length": 8,
        "dti": 10.2,
        "home_ownership_MORTGAGE": 1,
        "home_ownership_OWN": 0,
        "home_ownership_RENT": 0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 2800,
        "revol_util": 18.0,
        "total_acc": 20,
        "acc_now_delinq": 0,
        "tot_coll_amt": 0,
        "tot_cur_bal": 45000,
        "num_actv_rev_tl": 4,
        "num_rev_accts": 9,
        "num_il_tl": 5,
        "num_tl_90g_dpd_24m": 0,
        "num_tl_30dpd": 0,
        "num_tl_op_past_12m": 1,
        "pct_tl_nvr_dlq": 100,
        "percent_bc_gt_75": 0,
        "pub_rec_bankruptcies": 0,
        "tax_liens": 0,
        "total_bc_limit": 15000,
        "total_bal_ex_mort": 8000,
        "total_il_high_credit_limit": 30000,
        "tot_hi_cred_lim": 60000,
        "num_bc_sats": 5,
        "num_bc_tl": 6,
        "num_op_rev_tl": 7,
        "num_rev_tl_bal_gt_0": 4,
        "num_sats": 8,
        "num_tl_120dpd_2m": 0,
        "num_actv_bc_tl": 3,
        "verification_status_Source_Verified": 0,
        "verification_status_Verified": 1,
        "addr_state_freq": 0.06,
    },
    "risky": {
        "_label": "Risky Customer — High Risk",
        "loan_amnt": 4500,
        "int_rate": 28.9,
        "installment": 195,
        "grade": 6,
        "sub_grade": 29,
        "annual_inc": 28000,
        "emp_length": 1,
        "dti": 41.5,
        "home_ownership_MORTGAGE": 0,
        "home_ownership_OWN": 0,
        "home_ownership_RENT": 1,
        "delinq_2yrs": 3,
        "inq_last_6mths": 5,
        "open_acc": 5,
        "pub_rec": 1,
        "revol_bal": 9800,
        "revol_util": 92.0,
        "total_acc": 8,
        "acc_now_delinq": 1,
        "tot_coll_amt": 1500,
        "tot_cur_bal": 31000,
        "num_actv_rev_tl": 5,
        "num_rev_accts": 6,
        "num_il_tl": 2,
        "num_tl_90g_dpd_24m": 2,
        "num_tl_30dpd": 1,
        "num_tl_op_past_12m": 4,
        "pct_tl_nvr_dlq": 60,
        "percent_bc_gt_75": 85,
        "pub_rec_bankruptcies": 1,
        "tax_liens": 0,
        "total_bc_limit": 4000,
        "total_bal_ex_mort": 28000,
        "total_il_high_credit_limit": 10000,
        "tot_hi_cred_lim": 15000,
        "num_bc_sats": 1,
        "num_bc_tl": 2,
        "num_op_rev_tl": 3,
        "num_rev_tl_bal_gt_0": 5,
        "num_sats": 4,
        "num_tl_120dpd_2m": 1,
        "num_actv_bc_tl": 4,
        "verification_status_Source_Verified": 0,
        "verification_status_Verified": 0,
        "addr_state_freq": 0.03,
    },
    "average": {
        "_label": "Average Customer — Medium Risk",
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
    },
    "borderline": {
        "_label": "Borderline Customer — Near Threshold",
        "loan_amnt": 3000,
        "int_rate": 19.2,
        "installment": 112,
        "grade": 4,
        "sub_grade": 19,
        "annual_inc": 38000,
        "emp_length": 2,
        "dti": 32.0,
        "home_ownership_MORTGAGE": 0,
        "home_ownership_OWN": 0,
        "home_ownership_RENT": 1,
        "delinq_2yrs": 1,
        "inq_last_6mths": 2,
        "open_acc": 6,
        "pub_rec": 0,
        "revol_bal": 6100,
        "revol_util": 70.0,
        "total_acc": 10,
        "acc_now_delinq": 0,
        "tot_coll_amt": 0,
        "tot_cur_bal": 18000,
        "num_actv_rev_tl": 4,
        "num_rev_accts": 6,
        "num_il_tl": 2,
        "num_tl_90g_dpd_24m": 0,
        "num_tl_30dpd": 0,
        "num_tl_op_past_12m": 2,
        "pct_tl_nvr_dlq": 88,
        "percent_bc_gt_75": 45,
        "pub_rec_bankruptcies": 0,
        "tax_liens": 0,
        "total_bc_limit": 6500,
        "total_bal_ex_mort": 16000,
        "total_il_high_credit_limit": 12000,
        "tot_hi_cred_lim": 20000,
        "num_bc_sats": 2,
        "num_bc_tl": 3,
        "num_op_rev_tl": 4,
        "num_rev_tl_bal_gt_0": 4,
        "num_sats": 5,
        "num_tl_120dpd_2m": 0,
        "num_actv_bc_tl": 3,
        "verification_status_Source_Verified": 1,
        "verification_status_Verified": 0,
        "addr_state_freq": 0.04,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
COLORS = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def print_checkout_header(label: str, loan_amnt: float):
    print("\n" + "═" * 60)
    print(c(f"  🛒  BNPL CHECKOUT — {label}", "bold"))
    print(f"  Requested amount: {c(f'${loan_amnt:,.2f}', 'cyan')}")
    print("═" * 60)


def print_decision(result: dict):
    prob     = result["default_prob"]
    band     = result["risk_band"]
    decision = result["decision"]
    flags    = result["flags"]

    # Decision banner
    if decision == "APPROVE":
        print(c(f"\n  ✅  DECISION: APPROVED", "green"))
    else:
        print(c(f"\n  ❌  DECISION: REJECTED", "red"))

    # Risk details
    band_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red", "VERY_HIGH": "red"}.get(band, "yellow")
    print(f"\n  Default Probability : {c(f'{prob*100:.1f}%', band_color)}")
    print(f"  Risk Band           : {c(band, band_color)}")
    print(f"  Decision Confidence : {result['confidence']}")
    print(f"  Threshold Used      : {result['threshold_used']}")
    print(f"  Latency             : {result['latency_ms']:.1f} ms")
    print(f"  Request ID          : {result['request_id']}")

    # Active risk flags
    active_flags = [k for k, v in flags.items() if v]
    if active_flags:
        print(c(f"\n  ⚠  Risk Flags Triggered:", "yellow"))
        flag_labels = {
            "high_dti":        "High Debt-to-Income Ratio",
            "high_utilisation":"High Credit Utilisation",
            "recent_delinq":   "Recent Payment Delinquency",
            "inq_spike":       "Inquiry Spike (3+ in 6 mo)",
            "spending_spike":  "Spending Spike vs Income",
            "collections":     "Collections History",
            "bankruptcy":      "Bankruptcy on Record",
        }
        for f in active_flags:
            print(f"      • {flag_labels.get(f, f)}")
    else:
        print(c("\n  ✓  No major risk flags detected", "green"))

    print("\n" + "─" * 60)


def call_api(payload: dict, endpoint: str = "/score") -> dict:
    """POST to the risk scoring API."""
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(c("\n  ⚠  API is not running.  Start it with:  uvicorn app:app --port 8000", "red"))
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(c(f"\n  API error: {e.response.text}", "red"))
        sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# CHECKOUT SIMULATION
# ═════════════════════════════════════════════════════════════════════════════
def simulate_single(profile_name: str):
    profile = PROFILES.get(profile_name)
    if not profile:
        print(f"Unknown profile '{profile_name}'. Choose from: {list(PROFILES)}")
        sys.exit(1)

    label    = profile.pop("_label", profile_name)
    payload  = profile.copy()

    print_checkout_header(label, payload["loan_amnt"])

    # Animate "processing" for realism
    print("\n  Processing application ", end="", flush=True)
    for _ in range(5):
        print(".", end="", flush=True)
        time.sleep(0.15)
    print()

    result = call_api(payload)
    print_decision(result)
    return result


def simulate_batch(n: int = 10):
    """Generate random applicants and score them in one batch call."""
    applicants = []
    for _ in range(n):
        # Random mix of profiles
        base  = random.choice(list(PROFILES.values())).copy()
        base.pop("_label", None)
        # Randomise a few fields for variety
        base["loan_amnt"]     = round(random.uniform(500, 5000), 2)
        base["annual_inc"]    = round(random.uniform(24000, 120000), 2)
        base["dti"]           = round(random.uniform(5, 45), 1)
        base["revol_util"]    = round(random.uniform(5, 95), 1)
        base["inq_last_6mths"]= random.randint(0, 6)
        base["delinq_2yrs"]   = random.randint(0, 4)
        applicants.append(base)

    print(f"\n{'═'*60}")
    print(c(f"  📦  BATCH SIMULATION — {n} applicants", "bold"))
    print(f"{'═'*60}")

    payload  = {"applicants": applicants}
    url      = f"{API_BASE}/score/batch"
    resp     = requests.post(url, json=payload, timeout=30)
    result   = resp.json()

    print(f"\n  Total       : {result['total']}")
    print(c(f"  Approved    : {result['approved']}", "green"))
    print(c(f"  Rejected    : {result['rejected']}", "red"))
    print(f"  Approval Rate: {result['approval_rate']*100:.1f}%")
    print()

    for i, r in enumerate(result["results"]):
        dec_str = c("✅ APPROVE", "green") if r["decision"] == "APPROVE" else c("❌ REJECT", "red")
        print(f"  [{i+1:02d}]  {dec_str}  | prob={r['default_prob']:.3f} | band={r['risk_band']}")

    print(f"\n{'═'*60}\n")


def interactive_mode():
    """Prompt user to enter values manually."""
    print(c("\n  🔍  BNPL RISK SCORING — Interactive Mode", "bold"))
    print("  (Press Enter to accept defaults)\n")

    try:
        loan_amnt  = float(input("  Loan amount ($, max 5000) [1500]: ").strip() or 1500)
        annual_inc = float(input("  Annual income ($) [48000]: ").strip() or 48000)
        dti        = float(input("  Debt-to-income ratio (%) [18.5]: ").strip() or 18.5)
        revol_util = float(input("  Revolving utilisation (%) [40]: ").strip() or 40)
        delinq     = int(input("  Delinquencies last 2 yrs [0]: ").strip() or 0)
        inq        = int(input("  Hard inquiries last 6 mo [1]: ").strip() or 1)
        grade      = int(input("  Credit grade (1=A…7=G) [3]: ").strip() or 3)
    except (ValueError, KeyboardInterrupt):
        print("\n  Cancelled.")
        sys.exit(0)

    # Use 'average' as baseline and patch with user values
    payload = PROFILES["average"].copy()
    payload.pop("_label", None)
    payload.update({
        "loan_amnt":       loan_amnt,
        "annual_inc":      annual_inc,
        "dti":             dti,
        "revol_util":      revol_util,
        "delinq_2yrs":     delinq,
        "inq_last_6mths":  inq,
        "grade":           grade,
        "installment":     round(loan_amnt / 36 * 1.15, 2),
    })

    print_checkout_header("Custom Application", loan_amnt)
    print("\n  Processing ", end="", flush=True)
    for _ in range(5):
        print(".", end="", flush=True)
        time.sleep(0.15)
    print()

    result = call_api(payload)
    print_decision(result)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNPL Checkout Simulator")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default=None,
                        help="Use a preset customer profile")
    parser.add_argument("--batch",  action="store_true",
                        help="Simulate a batch of 10 random applicants")
    parser.add_argument("--n",      type=int, default=10,
                        help="Number of applicants for batch mode")
    args = parser.parse_args()

    if args.batch:
        simulate_batch(args.n)
    elif args.profile:
        simulate_single(args.profile)
    else:
        interactive_mode()

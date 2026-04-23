"""
generate_sample_data.py — Create a realistic synthetic LendingClub-style
dataset for testing the BNPL pipeline without downloading from Kaggle.

Usage:
    python generate_sample_data.py            # default 50,000 rows
    python generate_sample_data.py --n 10000  # smaller, faster
"""

import os, sys, argparse
import numpy  as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def generate(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Latent risk factor (drives default probability)
    risk = rng.uniform(0, 1, n)   # 0 = safest, 1 = riskiest

    # ── Grade (correlated with risk)
    grade_int = np.clip((risk * 7).astype(int) + 1, 1, 7)
    grade_ltr = [chr(64 + g) for g in grade_int]
    sub_num   = rng.integers(1, 6, n)
    sub_grade = [f"{g}{s}" for g, s in zip(grade_ltr, sub_num)]

    # ── Income (log-normal, slightly negatively correlated with risk)
    annual_inc = np.exp(rng.normal(10.8 - risk * 0.6, 0.45, n)).round(2)
    monthly_inc = annual_inc / 12

    # ── Loan amount (small — BNPL filter ≤ $5,000)
    loan_amnt = (rng.uniform(200, 5000, n) * (1 - 0.3 * risk)).clip(200, 5000).round(2)
    int_rate  = (grade_int * 4.0 + 5.0 + rng.normal(0, 1.5, n)).clip(5, 36).round(2)
    installment = (loan_amnt / 36 * (1 + int_rate / 100 / 12 * 36)).round(2)

    # ── DTI (positively correlated with risk)
    dti = (risk * 45 + rng.normal(5, 4, n)).clip(0, 70).round(2)

    # ── Revolving utilisation (correlated with risk)
    revol_util = (risk * 80 + rng.normal(10, 8, n)).clip(0, 100).round(2)
    revol_bal  = (revol_util / 100 * rng.lognormal(8.5, 0.8, n)).round(2)

    # ── Delinquency / inquiry (correlated with risk)
    delinq_2yrs   = rng.poisson(risk * 2.5, n).clip(0, 15)
    inq_last_6mths = rng.poisson(risk * 3, n).clip(0, 10)
    acc_now_delinq = rng.poisson(risk * 0.5, n).clip(0, 5)
    tot_coll_amt   = (rng.exponential(risk * 2000, n)).round(2)

    # ── Account counts
    open_acc   = rng.integers(1, 20, n)
    total_acc  = open_acc + rng.integers(0, 20, n)
    pub_rec    = rng.poisson(risk * 0.3, n).clip(0, 5)
    pub_rec_bk = (rng.random(n) < risk * 0.05).astype(int)
    tax_liens  = (rng.random(n) < risk * 0.02).astype(int)

    # ── Extended fields
    num_actv_rev_tl   = rng.integers(0, 10, n)
    num_rev_accts     = num_actv_rev_tl + rng.integers(0, 6, n)
    num_il_tl         = rng.integers(0, 12, n)
    num_tl_90g        = rng.poisson(risk * 1.5, n).clip(0, 8)
    num_tl_30dpd      = rng.poisson(risk * 0.8, n).clip(0, 5)
    num_tl_op_12m     = rng.integers(0, 5, n)
    pct_tl_nvr_dlq    = (100 - risk * 40 + rng.normal(0, 5, n)).clip(50, 100).round(1)
    pct_bc_gt_75      = (risk * 80 + rng.normal(5, 8, n)).clip(0, 100).round(1)
    total_bc_limit    = rng.lognormal(9, 0.8, n).round(2)
    total_bal_ex_mort = rng.lognormal(9.5, 0.9, n).round(2)
    total_il_hi_cl    = rng.lognormal(10, 0.9, n).round(2)
    tot_cur_bal       = (total_bal_ex_mort + rng.lognormal(10, 0.5, n)).round(2)
    tot_hi_cred_lim   = (total_bc_limit + total_il_hi_cl + rng.lognormal(9, 0.7, n)).round(2)
    num_bc_sats       = rng.integers(0, 8, n)
    num_bc_tl         = num_bc_sats + rng.integers(0, 4, n)
    num_op_rev_tl     = rng.integers(0, 12, n)
    num_rev_gt_0      = rng.integers(0, 8, n)
    num_sats          = rng.integers(1, 15, n)
    num_120dpd        = rng.poisson(risk * 0.3, n).clip(0, 3)
    num_actv_bc       = rng.integers(0, 6, n)

    # ── Loan status (target)
    default_prob = (
        0.03
        + 0.004 * dti
        + 0.003 * revol_util
        + 0.06  * (grade_int - 1) / 6
        + 0.04  * (delinq_2yrs > 0)
        + 0.05  * (inq_last_6mths >= 3)
        + 0.08  * (acc_now_delinq > 0)
        + 0.05  * (tot_coll_amt > 0)
        - 0.000001 * annual_inc
        + rng.normal(0, 0.04, n)
    ).clip(0, 0.95)

    status = np.where(
        rng.random(n) < default_prob,
        rng.choice(config.DEFAULT_STATUSES[:3], n),
        rng.choice(config.SAFE_STATUSES[:2], n),
    )

    # ── Categorical fields
    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], n,
                                p=[0.45, 0.38, 0.14, 0.03])
    verification   = rng.choice(["Verified", "Source Verified", "Not Verified"], n,
                                p=[0.35, 0.35, 0.30])
    purpose        = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "other", "medical", "car", "vacation"],
        n, p=[0.45, 0.25, 0.10, 0.08, 0.05, 0.04, 0.03],
    )
    addr_state = rng.choice(
        ["CA", "TX", "NY", "FL", "IL", "OH", "PA", "GA", "NC", "MI"],
        n,
    )
    emp_length = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "4 years",
         "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
        n,
    )

    df = pd.DataFrame({
        "loan_amnt":    loan_amnt,
        "term":         "36 months",
        "int_rate":     [f"{r}%" for r in int_rate],
        "installment":  installment,
        "grade":        grade_ltr,
        "sub_grade":    sub_grade,
        "emp_length":   emp_length,
        "home_ownership": home_ownership,
        "annual_inc":   annual_inc,
        "verification_status": verification,
        "loan_status":  status,
        "purpose":      purpose,
        "addr_state":   addr_state,
        "dti":          dti,
        "delinq_2yrs":  delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc":     open_acc,
        "pub_rec":      pub_rec,
        "revol_bal":    revol_bal,
        "revol_util":   [f"{r}%" for r in revol_util],
        "total_acc":    total_acc,
        "acc_now_delinq": acc_now_delinq,
        "tot_coll_amt": tot_coll_amt,
        "tot_cur_bal":  tot_cur_bal,
        "num_actv_rev_tl": num_actv_rev_tl,
        "num_rev_accts": num_rev_accts,
        "num_il_tl":    num_il_tl,
        "num_tl_90g_dpd_24m": num_tl_90g,
        "num_tl_30dpd": num_tl_30dpd,
        "num_tl_op_past_12m": num_tl_op_12m,
        "pct_tl_nvr_dlq": pct_tl_nvr_dlq,
        "percent_bc_gt_75": pct_bc_gt_75,
        "pub_rec_bankruptcies": pub_rec_bk,
        "tax_liens":    tax_liens,
        "total_bc_limit": total_bc_limit,
        "total_bal_ex_mort": total_bal_ex_mort,
        "total_il_high_credit_limit": total_il_hi_cl,
        "tot_hi_cred_lim": tot_hi_cred_lim,
        "num_bc_sats":  num_bc_sats,
        "num_bc_tl":    num_bc_tl,
        "num_op_rev_tl": num_op_rev_tl,
        "num_rev_tl_bal_gt_0": num_rev_gt_0,
        "num_sats":     num_sats,
        "num_tl_120dpd_2m": num_120dpd,
        "num_actv_bc_tl": num_actv_bc,
    })

    default_rate = (df["loan_status"].isin(config.DEFAULT_STATUSES)).mean()
    print(f"Generated {len(df):,} synthetic loans | Default rate: {default_rate:.2%}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    df = generate(n=args.n, seed=args.seed)
    out = config.RAW_DATA_PATH
    df.to_csv(out, index=False)
    print(f"Saved → {out}")

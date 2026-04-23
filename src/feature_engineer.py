"""
Feature Engineer Module
Creates new features for BNPL risk modeling.

Feature groups:
  • Financial ratios (loan-to-income, payment burden, utilization efficiency)
  • Credit behaviour (account age, credit mix, delinquency rate)
  • Risk flags (high utilisation, high DTI, inquiry spikes, etc.)
  • Interaction features (grade × DTI, income × utilisation)
  • Derived features (log transforms, squares, bins)
  • NEW: debt_to_income_ratio, transaction_frequency,
         repayment_consistency, average_balance_trend
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates new features for BNPL risk assessment."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def create_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial ratio features."""
        # Loan-to-income ratio
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

        # Monthly payment burden
        df["payment_burden"] = df["installment"] / ((df["annual_inc"] / 12) + 1)

        # Credit utilization efficiency
        df["util_efficiency"] = df["revol_util"] / (df["dti"] + 1)

        # Available credit
        df["available_credit"] = np.where(
            df["revol_util"] > 0,
            df["revol_bal"] / (df["revol_util"] / 100) - df["revol_bal"],
            0
        )

        return df

    def create_credit_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create credit behavior features."""
        # Account age (proxy)
        df["avg_account_age"] = df["total_acc"] / (df["open_acc"] + 1)

        # Credit mix diversity
        df["credit_mix"] = (
            (df.get("num_bc_tl", 0) > 0).astype(int) +
            (df.get("num_il_tl", 0) > 0).astype(int) +
            (df.get("num_rev_accts", 0) > 0).astype(int)
        )

        # Recent activity ratio
        df["recent_activity"] = df.get("num_tl_op_past_12m", 0) / (df["total_acc"] + 1)

        # Delinquency rate
        df["delinq_rate"] = df["delinq_2yrs"] / (df["total_acc"] + 1)

        return df

    def create_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary risk flags."""
        # High utilization flag
        df["high_util_flag"] = (df["revol_util"] > self.config["HIGH_UTIL_THRESHOLD"] * 100).astype(int)

        # High DTI flag
        df["high_dti_flag"] = (df["dti"] > self.config["HIGH_DTI_THRESHOLD"]).astype(int)

        # Recent inquiry spike
        df["recent_inq_flag"] = (df["inq_last_6mths"] >= 3).astype(int)

        # Spending spike (loan amount vs monthly income)
        monthly_income = df["annual_inc"] / 12
        df["spending_spike_flag"] = (
            df["loan_amnt"] > self.config["SPENDING_SPIKE_MULT"] * monthly_income
        ).astype(int)

        # Collections flag
        df["collections_flag"] = (df.get("tot_coll_amt", 0) > 0).astype(int)

        # Bankruptcy flag
        df["bankruptcy_flag"] = (df.get("pub_rec_bankruptcies", 0) > 0).astype(int)

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # Grade × DTI interaction
        df["grade_dti_interaction"] = df["grade"] * df["dti"]

        # Income × Utilization interaction
        df["income_util_interaction"] = np.log1p(df["annual_inc"]) * df["revol_util"]

        # Loan amount × Grade interaction
        df["loan_grade_interaction"] = df["loan_amnt"] * df["grade"]

        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create log-transformed, squared, and binned features."""
        # Log transformations for skewed variables
        for col in ["annual_inc", "revol_bal", "tot_cur_bal"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col])

        # Squared terms for important features
        for col in ["dti", "revol_util", "inq_last_6mths"]:
            if col in df.columns:
                df[f"{col}_squared"] = df[col] ** 2

        # Binned features
        if "annual_inc" in df.columns:
            df["income_bin"] = pd.cut(
                df["annual_inc"],
                bins=[0, 30000, 50000, 75000, 100000, float('inf')],
                labels=[1, 2, 3, 4, 5]
            )
            df["income_bin"] = pd.to_numeric(df["income_bin"],
                                             errors='coerce').fillna(3).astype(int)

        return df

    # ── NEW: additional derived features per requirements ────────────────
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the four additional derived features required by spec:
          • debt_to_income_ratio  — loan amount relative to annual income
          • transaction_frequency — proxy: recent account openings / total accounts
          • repayment_consistency — % of trades never delinquent (normalised)
          • average_balance_trend — total balance / total credit limit (utilisation trend)
        """
        # Debt-to-income ratio (loan-specific, distinct from revolving DTI)
        df["debt_to_income_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1.0)

        # Transaction frequency proxy
        num_recent = df["num_tl_op_past_12m"] if "num_tl_op_past_12m" in df.columns else 0
        df["transaction_frequency"] = num_recent / (df["total_acc"] + 1)

        # Repayment consistency (pct_tl_nvr_dlq is 0–100 scale)
        if "pct_tl_nvr_dlq" in df.columns:
            df["repayment_consistency"] = df["pct_tl_nvr_dlq"] / 100.0
        else:
            df["repayment_consistency"] = 0.5

        # Average balance trend (aggregate utilisation)
        if "total_bal_ex_mort" in df.columns and "tot_hi_cred_lim" in df.columns:
            df["average_balance_trend"] = (
                df["total_bal_ex_mort"] / (df["tot_hi_cred_lim"] + 1.0)
            )
        else:
            df["average_balance_trend"] = 0.0

        return df

    # ── master entry point ───────────────────────────────────────────────
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        logger.info("Creating engineered features...")

        initial_cols = df.shape[1]

        df = self.create_financial_ratios(df)
        df = self.create_credit_behavior_features(df)
        df = self.create_risk_flags(df)
        df = self.create_interaction_features(df)
        df = self.create_derived_features(df)
        df = self.create_advanced_features(df)    # ← NEW

        # Handle any infinite or NaN values created
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        new_features = df.shape[1] - initial_cols
        logger.info(f"Created {new_features} new features. Total columns: {df.shape[1]}")

        return df
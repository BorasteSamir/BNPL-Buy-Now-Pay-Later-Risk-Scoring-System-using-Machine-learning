"""
Cash Flow Analyzer Module
Derives cash-flow proxies from LendingClub credit-profile data and produces
a composite cash_flow_score (0–1, higher = more financially stable).

Since the dataset lacks granular transaction-level data, we approximate:
  • monthly_income        = annual_inc / 12
  • monthly_expense_proxy = installment + revolving-interest burden
  • savings_ratio         = (income - expense) / income
  • expense_volatility    = DTI-based proxy
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CashFlowAnalyzer:
    """Derives cash-flow health indicators from credit profile data."""

    def __init__(self, config: Dict[str, Any]):
        self.weights = config.get("CASH_FLOW_WEIGHTS", {
            "savings_ratio": 0.40,
            "income": 0.30,
            "expense_burden": 0.20,
            "dti": 0.10,
        })

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _min_max_normalize(series: pd.Series) -> pd.Series:
        """Min-max normalization to [0, 1] range. Handles constant series."""
        smin, smax = series.min(), series.max()
        if smax == smin:
            return pd.Series(0.5, index=series.index)
        return (series - smin) / (smax - smin)

    # ── core computations ────────────────────────────────────────────────
    def compute_monthly_income(self, df: pd.DataFrame) -> pd.Series:
        """Monthly income from annual income."""
        return df["annual_inc"] / 12.0

    def compute_expense_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        Monthly expense proxy:
          installment  (fixed loan payment)
        + revolving interest burden  (revol_bal × int_rate / 100 / 12)
        """
        revol_interest = df["revol_bal"] * df["int_rate"] / 100.0 / 12.0
        return df["installment"] + revol_interest

    def compute_savings_ratio(self, income: pd.Series,
                              expense: pd.Series) -> pd.Series:
        """(income − expense) / income, clipped to [-1, 1]."""
        ratio = (income - expense) / (income + 1.0)  # +1 avoids div-by-zero
        return ratio.clip(-1.0, 1.0)

    def compute_expense_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Proxy for expense volatility using coefficient of variation
        of income-normalised installment across similar income bins.
        """
        monthly_inc = df["annual_inc"] / 12.0
        burden = df["installment"] / (monthly_inc + 1.0)

        # rolling-style proxy: deviation from the mean burden in income quintile
        df_tmp = pd.DataFrame({"burden": burden, "inc": monthly_inc})
        df_tmp["inc_bin"] = pd.qcut(df_tmp["inc"], q=5, labels=False,
                                    duplicates="drop")
        group_std = df_tmp.groupby("inc_bin")["burden"].transform("std").fillna(0)
        return group_std

    # ── main entry ───────────────────────────────────────────────────────
    def compute_cash_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all cash-flow columns to the DataFrame and return it.
        New columns:
          monthly_income, monthly_expense_proxy, savings_ratio,
          expense_volatility, cash_flow_score
        """
        logger.info("Computing cash-flow features...")

        income  = self.compute_monthly_income(df)
        expense = self.compute_expense_proxy(df)

        df["monthly_income"]        = income
        df["monthly_expense_proxy"] = expense
        df["savings_ratio"]         = self.compute_savings_ratio(income, expense)
        df["expense_volatility"]    = self.compute_expense_volatility(df)

        # ── composite cash_flow_score ────────────────────────────────────
        w = self.weights
        norm_savings  = self._min_max_normalize(df["savings_ratio"])
        norm_income   = self._min_max_normalize(np.log1p(income))
        norm_burden   = 1.0 - self._min_max_normalize(expense / (income + 1.0))
        norm_dti      = 1.0 - self._min_max_normalize(df["dti"].clip(0, 100))

        df["cash_flow_score"] = (
            w["savings_ratio"]  * norm_savings +
            w["income"]         * norm_income  +
            w["expense_burden"] * norm_burden  +
            w["dti"]            * norm_dti
        ).clip(0.0, 1.0)

        logger.info(f"Cash-flow score — mean: {df['cash_flow_score'].mean():.3f}, "
                     f"std: {df['cash_flow_score'].std():.3f}")
        return df

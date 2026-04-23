"""
Data Preprocessor Module
Handles data cleaning, filtering, and preprocessing
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def filter_bnpl_loans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for BNPL-relevant loans"""
        logger.info("Filtering for BNPL-relevant loans...")
        
        df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
        df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        
        before = len(df)
        df = df[df["loan_amnt"] <= self.config["MAX_LOAN_AMOUNT"]].copy()
        df = df[df["term"].isin(self.config["BNPL_TERM_LIST"])].copy()
        
        logger.info(f"BNPL filter: {before:,} → {len(df):,} rows kept")
        return df
    
    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build binary default target"""
        logger.info("Building binary default target...")
        
        all_statuses = self.config["DEFAULT_STATUSES"] + self.config["SAFE_STATUSES"]
        df = df[df["loan_status"].isin(all_statuses)].copy()
        
        df["default"] = df["loan_status"].apply(
            lambda s: 1 if s in self.config["DEFAULT_STATUSES"] else 0
        )
        
        dist = df["default"].value_counts()
        logger.info(f"Target distribution: Safe (0): {dist.get(0,0):,} | Default (1): {dist.get(1,0):,}")
        logger.info(f"Default rate: {df['default'].mean():.2%}")
        
        df.drop(columns=["loan_status"], inplace=True)
        return df
    
    def select_columns(self, df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
        """Select relevant columns"""
        available = [c for c in keep_cols if c in df.columns]
        missing = [c for c in keep_cols if c not in df.columns]
        
        if missing:
            logger.warning(f"Columns not found: {missing}")
            
        df = df[available].copy()
        logger.info(f"Columns retained: {df.shape[1]}")
        return df
    
    def handle_missing(self, df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("Handling missing values...")
        
        # Drop high-missingness columns
        miss_frac = df.isnull().mean()
        drop_cols = miss_frac[miss_frac > threshold].index.tolist()
        drop_cols = [c for c in drop_cols if c != "default"]
        
        if drop_cols:
            logger.info(f"Dropping {len(drop_cols)} columns with >{threshold:.0%} missing")
            df.drop(columns=drop_cols, inplace=True)
        
        # Impute remaining missing values
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Numeric: median
        for col in num_cols:
            if col != "default" and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical: mode
        for col in cat_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info(f"Missing after imputation: {df.isnull().sum().sum()}")
        return df
    
    def treat_outliers(self, df: pd.DataFrame, iqr_factor: float = 3.0) -> pd.DataFrame:
        """Treat outliers using IQR winsorization"""
        logger.info(f"Winsorizing outliers (IQR factor={iqr_factor})...")
        
        skew_cols = [
            "annual_inc", "revol_bal", "tot_cur_bal", "tot_coll_amt",
            "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
            "tot_hi_cred_lim", "loan_amnt", "installment"
        ]
        skew_cols = [c for c in skew_cols if c in df.columns]
        
        for col in skew_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
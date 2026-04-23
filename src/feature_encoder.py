"""
Feature Encoder Module
Handles encoding of categorical variables
"""

import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """Handles encoding of categorical features"""
    
    def __init__(self):
        self.encoders = {}
        
    def encode_grade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode grade as ordinal (A=1, B=2, ..., G=7)"""
        if "grade" not in df.columns:
            return df
            
        grade_map = {g: i + 1 for i, g in enumerate("ABCDEFG")}
        df["grade"] = df["grade"].str.strip().map(grade_map).fillna(4)
        return df
    
    def encode_sub_grade(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode sub_grade as ordinal (A1=1, A2=2, ..., G5=35)"""
        if "sub_grade" not in df.columns:
            return df
            
        def sub_grade_to_int(sg):
            sg = str(sg).strip()
            try:
                letter = sg[0]
                num = int(sg[1])
                return (ord(letter) - ord("A")) * 5 + num
            except Exception:
                return 18  # default mid-value
                
        df["sub_grade"] = df["sub_grade"].apply(sub_grade_to_int)
        return df
    
    def encode_emp_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode employment length"""
        if "emp_length" not in df.columns:
            return df
            
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10,
        }
        df["emp_length"] = df["emp_length"].str.strip().map(emp_map).fillna(5)
        return df
    
    def clean_percentage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean percentage columns (remove % sign)"""
        # Revolving utilization
        if "revol_util" in df.columns:
            revol_series = (
                df["revol_util"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
            df["revol_util"] = revol_series.fillna(revol_series.median())
        
        # Interest rate
        if "int_rate" in df.columns:
            df["int_rate"] = (
                df["int_rate"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(13.0)
            )
        
        return df
    
    def frequency_encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Frequency encode high-cardinality categorical"""
        if column not in df.columns:
            return df
            
        freq = df[column].value_counts(normalize=True)
        df[f"{column}_freq"] = df[column].map(freq).fillna(0)
        df.drop(columns=[column], inplace=True)
        return df
    
    def one_hot_encode(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """One-hot encode categorical columns"""
        existing_cols = [c for c in columns if c in df.columns]
        if existing_cols:
            df = pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype=int)
        return df
    
    def encode_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all encoding transformations"""
        logger.info("Encoding categorical variables...")
        
        df = self.encode_grade(df)
        df = self.encode_sub_grade(df)
        df = self.encode_emp_length(df)
        df = self.clean_percentage_columns(df)
        df = self.frequency_encode(df, "addr_state")
        df = self.one_hot_encode(df, ["home_ownership", "verification_status", "purpose"])
        
        return df
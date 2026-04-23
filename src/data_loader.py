"""
Data Loader Module
Handles loading and initial validation of raw data
"""

import os
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading raw data from various sources"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_dir = os.path.dirname(data_path)
        
    def _find_dataset_file(self) -> str:
        """Find the dataset file with flexible naming"""
        # Try exact path first
        if os.path.exists(self.data_path):
            return self.data_path
            
        # Try common variations in the data directory
        possible_names = [
            "loan_data.csv",
            "Loan_data.csv", 
            "loan.csv",
            "accepted_2007_to_2018Q4.csv",
            "lending_club.csv"
        ]
        
        for name in possible_names:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                logger.info(f"Found dataset file: {name}")
                return path
                
        return None
        
    def load_csv(self, skiprows: Optional[int] = None, usecols: Optional[list] = None) -> pd.DataFrame:
        """Load CSV file with validation"""
        # Try to find the dataset file
        actual_path = self._find_dataset_file()
        
        if actual_path is None:
            # List available files in data directory
            available_files = []
            if os.path.exists(self.data_dir):
                available_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            
            error_msg = (
                f"Dataset not found at '{self.data_path}'.\n"
                f"Searched in directory: {self.data_dir}\n"
                f"Available CSV files: {available_files if available_files else 'None'}\n\n"
                "Please download 'loan_data.csv' from Kaggle:\n"
                "https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
                f"And place it in: {self.data_dir}/"
            )
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading data from: {actual_path}")
        
        try:
            df = pd.read_csv(actual_path, low_memory=False, skiprows=skiprows, usecols=usecols)
            logger.info(f"Raw data shape: {df.shape}")
            
            # Clean footer rows if present
            if "loan_amnt" in df.columns:
                df = df[pd.to_numeric(df["loan_amnt"], errors="coerce").notna()].copy()
                logger.info(f"After cleaning: {df.shape}")
            
            return df
            
        except Exception as e:
            raise Exception(f"Error reading CSV file '{actual_path}': {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """Validate that required columns exist"""
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return False
        return True
"""
Model Trainer Module
Handles model training, evaluation, and selection.

Models:
  • XGBoost  (gradient-boosted trees)
  • LightGBM (gradient-boosted trees, fast training)

Outputs:
  • Best model selection by ROC-AUC
  • Classification metrics (accuracy, precision, recall, F1, AUC)
  • ROC curve overlay plot
  • Feature importance plot
"""

import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, accuracy_score,
                             roc_curve, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and target for training."""
        logger.info("Preparing data for training...")

        X = df.drop(columns=["default"])
        y = df["default"]
        feature_names = X.columns.tolist()

        logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X.values, y.values, feature_names

    def split_data(self, X: np.ndarray, y: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        return train_test_split(
            X, y,
            test_size=self.config["TEST_SIZE"],
            random_state=self.config["RANDOM_STATE"],
            stratify=y
        )

    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for handling class imbalance."""
        logger.info("Applying SMOTE for class balancing...")

        smote = SMOTE(
            random_state=self.config["SMOTE_RANDOM_STATE"],
            sampling_strategy=self.config["SMOTE_SAMPLING_STRATEGY"]
        )

        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {np.bincount(y_res)}")
        return X_res, y_res

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    # ── model training ───────────────────────────────────────────────────
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray
                      ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")

        scale_pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
        params = self.config.get("XGB_PARAMS", {})

        model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=self.config["RANDOM_STATE"],
            eval_metric=params.get("eval_metric", "logloss"),
        )
        model.fit(X_train, y_train)
        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray
                       ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        logger.info("Training LightGBM...")

        params = self.config.get("LGBM_PARAMS", {})

        model = lgb.LGBMClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            bagging_fraction=float(params.get("subsample", 0.8)),
            bagging_freq=5,
            feature_fraction=float(params.get("colsample_bytree", 0.8)),
            is_unbalance=params.get("is_unbalance", True),
            random_state=self.config["RANDOM_STATE"],
            verbose=-1,
        )
        model.fit(X_train, y_train)
        return model

    # ── evaluation ───────────────────────────────────────────────────────
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str) -> Dict[str, float]:
        """Evaluate model performance with full metrics."""
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc   = roc_auc_score(y_test, y_pred_proba)
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)

        report = classification_report(y_test, y_pred, output_dict=True)
        f1     = report["1"]["f1-score"]

        metrics = {
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1_score":  round(f1, 4),
            "roc_auc":   round(roc_auc, 4),
        }

        logger.info(
            f"{model_name} — AUC: {roc_auc:.4f}, Acc: {accuracy:.4f}, "
            f"Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}"
        )
        return metrics

    # ── master training pipeline ─────────────────────────────────────────
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost + LightGBM, evaluate, select best."""
        logger.info("Starting model training pipeline...")

        # Prepare data
        X, y, feature_names = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Apply SMOTE
        X_train_smote, y_train_smote = self.apply_smote(X_train, y_train)

        # Scale features (keep unscaled copies for tree models)
        X_train_scaled, X_test_scaled = self.scale_features(X_train_smote, X_test)

        # Train models (tree models use unscaled data)
        models = {
            "XGBoost":  self.train_xgboost(X_train_smote, y_train_smote),
            "LightGBM": self.train_lightgbm(X_train_smote, y_train_smote),
        }

        # Evaluate models
        results = {}
        roc_curves = {}
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[name] = (fpr, tpr, metrics["roc_auc"])

            results[name] = {"model": model, "metrics": metrics}

            # Track best model
            if metrics["roc_auc"] > self.best_score:
                self.best_score = metrics["roc_auc"]
                self.best_model = model
                self.best_model_name = name

        logger.info(f"🏆 Best model: {self.best_model_name} "
                     f"(ROC-AUC: {self.best_score:.4f})")

        return {
            "results":         results,
            "best_model":      self.best_model,
            "best_model_name": self.best_model_name,
            "scaler":          self.scaler,
            "feature_names":   feature_names,
            "X_test":          X_test,
            "y_test":          y_test,
            "roc_curves":      roc_curves,
        }

    # ── persistence ──────────────────────────────────────────────────────
    def save_model(self, model_path: str, scaler_path: str,
                   feature_path: str, model, scaler,
                   feature_names: List[str]):
        """Save trained model and preprocessing objects."""
        logger.info("Saving model artifacts...")

        for path in [model_path, scaler_path, feature_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(feature_path, "wb") as f:
            pickle.dump(feature_names, f)

        logger.info("Model artifacts saved successfully.")

    # ── visualisation ────────────────────────────────────────────────────
    @staticmethod
    def plot_roc_curves(roc_curves: dict, save_path: str = None):
        """Overlay ROC curves for all models."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"XGBoost": "#e74c3c", "LightGBM": "#2ecc71"}
        fig, ax = plt.subplots(figsize=(9, 7))

        for name, (fpr, tpr, auc) in roc_curves.items():
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})",
                    color=colors.get(name, "gray"), linewidth=2.5)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — XGBoost vs LightGBM",
                      fontsize=14, fontweight="bold")
        ax.legend(fontsize=12, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved → {save_path}")
        plt.close(fig)

    @staticmethod
    def plot_feature_importance(model, feature_names: list,
                                 top_n: int = 20,
                                 save_path: str = None):
        """Bar chart of top-N feature importances from the model."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            [feature_names[i] for i in idx][::-1],
            importances[idx][::-1],
            color="#3498db", edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title("Top-20 Feature Importances (Best Model)",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance chart saved → {save_path}")
        plt.close(fig)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
        """Confusion matrix heatmap."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Not Default", "Default"],
                    yticklabels=["Not Default", "Default"], ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved → {save_path}")
        plt.close(fig)
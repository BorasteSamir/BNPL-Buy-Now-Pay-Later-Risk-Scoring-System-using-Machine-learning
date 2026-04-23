"""
Model Explainability Module
Uses SHAP (SHapley Additive exPlanations) to explain model predictions.

Provides:
  • Global explanations — feature importance across all predictions
  • Local explanations  — per-applicant decision factors
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP-based model explainability for tree-based classifiers."""

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def _init_explainer(self, X_sample=None):
        """Lazy-initialise the SHAP TreeExplainer (with fallback)."""
        if self.explainer is None:
            import shap
            try:
                self.explainer = shap.TreeExplainer(self.model)
                self._use_tree = True
                logger.info("SHAP TreeExplainer initialised.")
            except Exception as e:
                logger.warning(f"TreeExplainer failed ({e}), using Explainer fallback.")
                # We need X_sample for the masker, and predict_proba for the callable
                if hasattr(self.model, "predict_proba"):
                    self.explainer = shap.Explainer(self.model.predict_proba, X_sample)
                else:
                    self.explainer = shap.Explainer(self.model.predict, X_sample)
                self._use_tree = False

    # ── compute SHAP values ──────────────────────────────────────────────
    def compute_shap_values(self, X: np.ndarray,
                            max_samples: int = 2000) -> np.ndarray:
        """
        Compute SHAP values for a sample of the data.
        Uses a subsample for performance on large datasets.
        """
        import shap

        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X_sample = X[idx]
            logger.info(f"Computing SHAP on subsample of {max_samples} rows...")
        else:
            X_sample = X
            logger.info(f"Computing SHAP on {len(X)} rows...")

        self._init_explainer(X_sample)

        try:
            self.shap_values = self.explainer.shap_values(X_sample)
        except Exception:
            # Fallback if shap_values() fails (some explainers only support __call__)
            shap_obj = self.explainer(X_sample)
            self.shap_values = shap_obj.values

        # For binary classifiers, shap_values may be a list [class0, class1]
        # or it might be 3D array for some explainers
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # class 1 = default
        elif len(self.shap_values.shape) == 3:
            self.shap_values = self.shap_values[:, :, 1]

        self._X_sample = X_sample
        logger.info("SHAP values computed.")
        return self.shap_values

    # ── global importance ────────────────────────────────────────────────
    def get_global_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return top-N features by mean |SHAP| value."""
        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first.")

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_shap": mean_abs,
        }).sort_values("mean_shap", ascending=False).head(top_n)

        return imp_df.reset_index(drop=True)

    # ── local explanation ────────────────────────────────────────────────
    def explain_single(self, idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction.
        Returns top positive and negative SHAP contributors.
        """
        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first.")

        sv = self.shap_values[idx]
        pairs = list(zip(self.feature_names, sv))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        top_risk    = [(f, round(v, 4)) for f, v in pairs if v > 0][:5]
        top_protect = [(f, round(v, 4)) for f, v in pairs if v < 0][:5]

        return {
            "top_risk_factors":       top_risk,
            "top_protective_factors": top_protect,
        }

    # ── plots ────────────────────────────────────────────────────────────
    def plot_summary(self, save_path: Optional[str] = None):
        """SHAP summary (beeswarm) plot — global feature importance."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first.")

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self._X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Feature Importance — Global View",
                  fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved → {save_path}")
        plt.close("all")

    def plot_waterfall(self, idx: int = 0, save_path: Optional[str] = None):
        """SHAP waterfall plot for a single prediction."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        if self.shap_values is None:
            raise ValueError("Call compute_shap_values() first.")

        base_value = getattr(self.explainer, "expected_value", None)
        if base_value is None:
            # Fallback if explainer has no expected_value
            if hasattr(self.model, "predict_proba"):
                try:
                    preds = self.model.predict_proba(self._X_sample)
                    base_value = preds[:, 1].mean() if preds.ndim > 1 else preds.mean()
                except Exception:
                    base_value = 0.5
            else:
                try:
                    base_value = self.model.predict(self._X_sample).mean()
                except Exception:
                    base_value = 0.5
        elif isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        explanation = shap.Explanation(
            values=self.shap_values[idx],
            base_values=base_value,
            data=self._X_sample[idx],
            feature_names=self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title("SHAP Waterfall — Single Applicant Explanation",
                  fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP waterfall saved → {save_path}")
        plt.close("all")

    def plot_feature_importance_bar(self, save_path: Optional[str] = None):
        """Classic bar chart of mean |SHAP| values."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imp = self.get_global_importance(20)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(imp["feature"], imp["mean_shap"], color="#3498db",
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Mean |SHAP| Value", fontsize=12)
        ax.set_title("Top-20 Features by SHAP Importance",
                      fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance bar chart saved → {save_path}")
        plt.close(fig)

"""
Drift Detector Module
Implements Population Stability Index (PSI) to compare feature distributions
between a reference (training) dataset and a new (production) dataset.

PSI interpretation:
  < 0.10  → No significant shift
  0.10–0.20 → Moderate shift — monitor
  > 0.20  → Significant shift — consider retraining
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift via Population Stability Index (PSI)."""

    def __init__(self, config: Dict[str, Any]):
        self.n_buckets       = config.get("PSI_BUCKETS", 10)
        self.drift_threshold = config.get("PSI_DRIFT_THRESHOLD", 0.20)
        self.warn_threshold  = config.get("PSI_WARN_THRESHOLD", 0.10)

    # ── single-feature PSI ───────────────────────────────────────────────
    @staticmethod
    def _psi_single(expected: np.ndarray, actual: np.ndarray,
                    n_buckets: int = 10) -> float:
        """
        Compute PSI for one feature.

        Steps:
          1. Determine bucket edges from the *expected* distribution.
          2. Bin both expected and actual into those buckets.
          3. PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
        """
        # Quantile-based bucket edges from training data
        edges = np.unique(
            np.percentile(expected, np.linspace(0, 100, n_buckets + 1))
        )
        if len(edges) < 3:
            return 0.0  # constant / near-constant feature

        exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
        act_counts = np.histogram(actual,   bins=edges)[0].astype(float)

        # Convert to proportions — add small epsilon to avoid log(0)
        eps = 1e-6
        exp_pct = exp_counts / exp_counts.sum() + eps
        act_pct = act_counts / act_counts.sum() + eps

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    # ── full-dataset PSI ─────────────────────────────────────────────────
    def compute_psi(self, reference_df: pd.DataFrame,
                    current_df: pd.DataFrame,
                    feature_cols: Optional[list] = None) -> pd.DataFrame:
        """
        Compute PSI for all numeric features between reference and current data.

        Returns a DataFrame with columns: feature, psi, status
        """
        if feature_cols is None:
            feature_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove target column if present
        feature_cols = [c for c in feature_cols if c != "default"]

        results = []
        for col in feature_cols:
            if col not in current_df.columns:
                continue
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            psi_value = self._psi_single(ref_vals, cur_vals, self.n_buckets)

            if psi_value > self.drift_threshold:
                status = "🔴 SIGNIFICANT DRIFT"
            elif psi_value > self.warn_threshold:
                status = "🟡 MODERATE DRIFT"
            else:
                status = "🟢 STABLE"

            results.append({
                "feature": col,
                "psi": round(psi_value, 6),
                "status": status,
            })

        psi_df = pd.DataFrame(results).sort_values("psi", ascending=False)
        return psi_df.reset_index(drop=True)

    def check_drift(self, reference_df: pd.DataFrame,
                    current_df: pd.DataFrame,
                    feature_cols: Optional[list] = None) -> Dict[str, Any]:
        """
        Run drift analysis and return structured report.

        Returns:
          {
            "psi_table":        pd.DataFrame,
            "drifted_features": list[str],
            "needs_retraining": bool,
            "summary":          str,
          }
        """
        logger.info("Running drift detection (PSI)...")

        psi_df = self.compute_psi(reference_df, current_df, feature_cols)

        drifted = psi_df[psi_df["psi"] > self.drift_threshold]["feature"].tolist()
        warned  = psi_df[(psi_df["psi"] > self.warn_threshold) &
                         (psi_df["psi"] <= self.drift_threshold)]["feature"].tolist()

        needs_retraining = len(drifted) > 0

        summary_lines = [
            f"Total features analysed : {len(psi_df)}",
            f"Significant drift (PSI>{self.drift_threshold}): {len(drifted)}  {drifted}",
            f"Moderate drift   (PSI>{self.warn_threshold}): {len(warned)}  {warned}",
            f"Retraining recommended  : {'YES ⚠️' if needs_retraining else 'NO ✅'}",
        ]
        summary = "\n".join(summary_lines)
        logger.info("\n" + summary)

        return {
            "psi_table": psi_df,
            "drifted_features": drifted,
            "needs_retraining": needs_retraining,
            "summary": summary,
        }

    # ── visualisation ────────────────────────────────────────────────────
    @staticmethod
    def plot_psi(psi_df: pd.DataFrame, save_path: Optional[str] = None):
        """Bar chart of PSI values per feature."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = psi_df.head(20)  # top-20 most drifted

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = []
        for _, row in top.iterrows():
            if "SIGNIFICANT" in row["status"]:
                colors.append("#e74c3c")
            elif "MODERATE" in row["status"]:
                colors.append("#f39c12")
            else:
                colors.append("#2ecc71")

        ax.barh(top["feature"], top["psi"], color=colors, edgecolor="white")
        ax.axvline(x=0.20, color="#e74c3c", linestyle="--", linewidth=1.5,
                   label="Drift threshold (0.20)")
        ax.axvline(x=0.10, color="#f39c12", linestyle="--", linewidth=1.5,
                   label="Warn threshold (0.10)")
        ax.set_xlabel("PSI Value", fontsize=12)
        ax.set_title("Population Stability Index — Feature Drift Report",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"PSI chart saved → {save_path}")
        plt.close(fig)

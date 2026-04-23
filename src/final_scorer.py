"""
Final Scoring Module
Combines credit risk PD and cash-flow score into a composite decision score.

  final_score = w1 × (1 − PD) + w2 × cash_flow_score

  Where:
    PD             = model-predicted probability of default (0–1)
    cash_flow_score = financial stability score (0–1, higher = better)

  Decision:
    APPROVE  if final_score ≥ approve_threshold
    REVIEW   if reject_threshold ≤ final_score < approve_threshold
    REJECT   if final_score < reject_threshold

  Thresholds are dynamically set by the bandit optimizer.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FinalScorer:
    """Composite risk scoring and loan decision engine."""

    def __init__(self, config: Dict[str, Any]):
        self.w_pd       = config.get("SCORE_WEIGHT_PD", 0.60)
        self.w_cf       = config.get("SCORE_WEIGHT_CASHFLOW", 0.40)
        self.approve_th = config.get("APPROVE_THRESHOLD", 0.65)
        self.reject_th  = config.get("REJECT_THRESHOLD", 0.35)

    def update_thresholds(self, approve: float, reject: float):
        """Update decision thresholds (e.g. from bandit optimizer)."""
        self.approve_th = approve
        self.reject_th  = reject
        logger.info(f"Thresholds updated — APPROVE≥{approve:.3f}, "
                     f"REJECT<{reject:.3f}")

    # ── core scoring ─────────────────────────────────────────────────────
    def compute_final_score(self, pd_scores: np.ndarray,
                            cash_flow_scores: np.ndarray) -> np.ndarray:
        """
        Compute composite final score.
        Higher score = better applicant (lower default risk + stronger cash flow).
        """
        final = self.w_pd * (1.0 - pd_scores) + self.w_cf * cash_flow_scores
        return np.clip(final, 0.0, 1.0)

    def make_decisions(self, final_scores: np.ndarray) -> np.ndarray:
        """
        Map final scores to APPROVE / REVIEW / REJECT decisions.
        """
        decisions = np.where(
            final_scores >= self.approve_th, "APPROVE",
            np.where(final_scores < self.reject_th, "REJECT", "REVIEW")
        )
        return decisions

    # ── full pipeline ────────────────────────────────────────────────────
    def score_and_decide(self, pd_scores: np.ndarray,
                         cash_flow_scores: np.ndarray) -> Dict[str, Any]:
        """
        End-to-end scoring: compute final score → make decisions → return report.
        """
        logger.info("Computing final composite scores...")

        final_scores = self.compute_final_score(pd_scores, cash_flow_scores)
        decisions    = self.make_decisions(final_scores)

        # Distribution summary
        unique, counts = np.unique(decisions, return_counts=True)
        dist = dict(zip(unique, counts))
        total = len(decisions)

        summary = {
            "total_applicants": total,
            "approved":  dist.get("APPROVE", 0),
            "review":    dist.get("REVIEW", 0),
            "rejected":  dist.get("REJECT", 0),
            "approve_rate": round(dist.get("APPROVE", 0) / total * 100, 1),
            "reject_rate":  round(dist.get("REJECT", 0) / total * 100, 1),
            "avg_score": round(float(final_scores.mean()), 4),
            "score_std": round(float(final_scores.std()), 4),
        }

        logger.info(f"Decisions — APPROVE: {summary['approved']} ({summary['approve_rate']}%) | "
                     f"REVIEW: {summary['review']} | "
                     f"REJECT: {summary['rejected']} ({summary['reject_rate']}%)")

        return {
            "final_scores": final_scores,
            "decisions":    decisions,
            "summary":      summary,
        }

    # ── visualisation ────────────────────────────────────────────────────
    def plot_score_distribution(self, final_scores: np.ndarray,
                                decisions: np.ndarray,
                                save_path: Optional[str] = None):
        """Histogram of final scores with threshold lines."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(final_scores, bins=50, color="#3498db", alpha=0.7,
                edgecolor="white", linewidth=0.5)
        ax.axvline(self.approve_th, color="#2ecc71", linestyle="--",
                   linewidth=2, label=f"Approve ≥ {self.approve_th:.2f}")
        ax.axvline(self.reject_th, color="#e74c3c", linestyle="--",
                   linewidth=2, label=f"Reject < {self.reject_th:.2f}")

        ax.fill_betweenx([0, ax.get_ylim()[1]], self.approve_th, 1.0,
                         alpha=0.08, color="#2ecc71")
        ax.fill_betweenx([0, ax.get_ylim()[1]], 0.0, self.reject_th,
                         alpha=0.08, color="#e74c3c")

        ax.set_xlabel("Final Composite Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Final Score Distribution with Decision Thresholds",
                      fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Score distribution saved → {save_path}")
        plt.close(fig)

    @staticmethod
    def plot_decision_pie(decisions: np.ndarray,
                          save_path: Optional[str] = None):
        """Pie chart of decision distribution."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        unique, counts = np.unique(decisions, return_counts=True)
        color_map = {"APPROVE": "#2ecc71", "REVIEW": "#f39c12", "REJECT": "#e74c3c"}
        colors = [color_map.get(d, "#95a5a6") for d in unique]

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            counts, labels=unique, autopct="%1.1f%%", colors=colors,
            textprops={"fontsize": 13}, startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for autotext in autotexts:
            autotext.set_fontweight("bold")
        ax.set_title("Loan Decision Distribution",
                      fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Decision pie chart saved → {save_path}")
        plt.close(fig)

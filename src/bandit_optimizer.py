"""
Multi-Armed Bandit Optimizer Module
Uses Thompson Sampling to dynamically select the best approval strategy
from three arms: conservative, moderate, aggressive.

Reward function:
  approved & repaid  → profit = loan_amnt × int_rate/100 × term_years
  approved & default → loss   = −loan_amnt
  rejected           → 0  (no profit, no loss)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BanditOptimizer:
    """Thompson Sampling bandit for optimizing BNPL approval thresholds."""

    def __init__(self, config: Dict[str, Any]):
        self.strategies   = config.get("BANDIT_STRATEGIES", {
            "conservative": {"pd_threshold": 0.20},
            "moderate":     {"pd_threshold": 0.35},
            "aggressive":   {"pd_threshold": 0.50},
        })
        self.n_rounds     = config.get("BANDIT_ROUNDS", 1000)
        self.term_years   = config.get("BANDIT_LOAN_TERM_YEARS", 3)
        self.random_state = config.get("RANDOM_STATE", 42)

        # Thompson Sampling state: Beta(alpha, beta) per arm
        self.arms = list(self.strategies.keys())
        self.alpha = {arm: 1.0 for arm in self.arms}  # successes + 1
        self.beta  = {arm: 1.0 for arm in self.arms}  # failures + 1

    # ── reward calculation ───────────────────────────────────────────────
    def _compute_reward(self, loan_amnt: float, int_rate: float,
                        is_default: int, is_approved: bool) -> float:
        """
        Compute profit/loss for a single loan decision.
        """
        if not is_approved:
            return 0.0  # rejected → no exposure

        if is_default:
            return -loan_amnt  # total loss on default
        else:
            # interest earned over loan term
            return loan_amnt * (int_rate / 100.0) * self.term_years

    # ── single round ─────────────────────────────────────────────────────
    def _select_arm(self, rng: np.random.Generator) -> str:
        """Thompson Sampling: sample from Beta posterior, pick highest."""
        samples = {
            arm: rng.beta(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.get)

    # ── main optimisation ────────────────────────────────────────────────
    def optimize(self, pd_scores: np.ndarray, actuals: np.ndarray,
                 loan_amnts: np.ndarray, int_rates: np.ndarray) -> Dict[str, Any]:
        """
        Run Thompson Sampling bandit over the dataset.

        Args:
            pd_scores:  array of predicted P(default) for each applicant
            actuals:    array of actual default labels (0/1)
            loan_amnts: array of loan amounts
            int_rates:  array of interest rates (%)

        Returns dict with:
            optimal_strategy, strategy_stats, history, thresholds
        """
        logger.info(f"Running bandit optimization ({self.n_rounds} rounds)...")
        rng = np.random.default_rng(self.random_state)
        n_samples = len(pd_scores)

        # Reset state
        self.alpha = {arm: 1.0 for arm in self.arms}
        self.beta  = {arm: 1.0 for arm in self.arms}

        # Track history
        history = {arm: {"selections": 0, "total_reward": 0.0, "rewards": []}
                   for arm in self.arms}
        arm_selections = []

        for t in range(self.n_rounds):
            # Pick a random applicant for this round
            idx = rng.integers(0, n_samples)

            # Select arm via Thompson Sampling
            chosen_arm = self._select_arm(rng)
            threshold  = self.strategies[chosen_arm]["pd_threshold"]

            # Decision: approve if PD < threshold
            approved = pd_scores[idx] < threshold

            # Compute reward
            reward = self._compute_reward(
                loan_amnts[idx], int_rates[idx], actuals[idx], approved
            )

            # Update Beta posterior
            if reward > 0:
                self.alpha[chosen_arm] += 1
            else:
                self.beta[chosen_arm] += 1

            # Record
            history[chosen_arm]["selections"] += 1
            history[chosen_arm]["total_reward"] += reward
            history[chosen_arm]["rewards"].append(reward)
            arm_selections.append(chosen_arm)

        # ── compute strategy-level stats ─────────────────────────────────
        strategy_stats = {}
        for arm in self.arms:
            n_sel = history[arm]["selections"]
            total = history[arm]["total_reward"]
            avg   = total / n_sel if n_sel > 0 else 0.0
            strategy_stats[arm] = {
                "selections":    n_sel,
                "total_reward":  round(total, 2),
                "avg_reward":    round(avg, 2),
                "pd_threshold":  self.strategies[arm]["pd_threshold"],
                "selection_pct": round(n_sel / self.n_rounds * 100, 1),
            }

        # Optimal = most selected arm
        optimal = max(strategy_stats, key=lambda a: strategy_stats[a]["selections"])
        optimal_threshold = self.strategies[optimal]["pd_threshold"]

        logger.info(f"Optimal strategy: {optimal} "
                     f"(threshold={optimal_threshold}, "
                     f"selected {strategy_stats[optimal]['selection_pct']}%)")

        return {
            "optimal_strategy":   optimal,
            "optimal_threshold":  optimal_threshold,
            "strategy_stats":     strategy_stats,
            "arm_selections":     arm_selections,
            "history":            history,
        }

    # ── derive dynamic decision thresholds ───────────────────────────────
    @staticmethod
    def derive_thresholds(optimal_threshold: float) -> Dict[str, float]:
        """
        From the bandit's optimal PD threshold, derive APPROVE / REVIEW / REJECT
        thresholds for the final composite score.

        Since final_score = w1*(1-PD) + w2*cash_flow, higher = better,
        we map the PD threshold to score thresholds:
          approve_threshold = 1 - optimal_pd_threshold + margin
          reject_threshold  = approve_threshold - band_width
        """
        # More aggressive PD threshold → lower bar for approval
        approve = max(0.30, 1.0 - optimal_threshold)
        reject  = max(0.15, approve - 0.25)

        return {
            "approve_threshold": round(approve, 3),
            "reject_threshold":  round(reject, 3),
        }

    # ── visualisation ────────────────────────────────────────────────────
    @staticmethod
    def plot_convergence(arm_selections: list, arms: list,
                         save_path: Optional[str] = None):
        """Plot arm selection frequency over time."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(arm_selections)
        window = max(20, n // 20)
        colors = {"conservative": "#2ecc71", "moderate": "#3498db",
                  "aggressive": "#e74c3c"}

        fig, ax = plt.subplots(figsize=(12, 5))
        for arm in arms:
            running = pd.Series([1 if s == arm else 0
                                 for s in arm_selections]).rolling(window).mean()
            ax.plot(running, label=arm.capitalize(), color=colors.get(arm, "gray"),
                    linewidth=2)

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Selection Frequency (rolling)", fontsize=12)
        ax.set_title("Multi-Armed Bandit — Strategy Convergence",
                      fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Bandit convergence plot saved → {save_path}")
        plt.close(fig)

    @staticmethod
    def plot_strategy_comparison(strategy_stats: dict,
                                  save_path: Optional[str] = None):
        """Bar chart comparing strategy performance."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arms = list(strategy_stats.keys())
        avg_rewards  = [strategy_stats[a]["avg_reward"] for a in arms]
        selections   = [strategy_stats[a]["selection_pct"] for a in arms]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Average reward
        ax1.bar([a.capitalize() for a in arms], avg_rewards, color=colors,
                edgecolor="white", linewidth=1.5)
        ax1.set_title("Average Reward per Strategy", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Avg Reward ($)", fontsize=11)
        for i, v in enumerate(avg_rewards):
            ax1.text(i, v + abs(v)*0.05, f"${v:,.0f}", ha="center", fontsize=10,
                     fontweight="bold")

        # Selection percentage
        ax2.bar([a.capitalize() for a in arms], selections, color=colors,
                edgecolor="white", linewidth=1.5)
        ax2.set_title("Selection Frequency (%)", fontsize=13, fontweight="bold")
        ax2.set_ylabel("% of Rounds", fontsize=11)
        for i, v in enumerate(selections):
            ax2.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10,
                     fontweight="bold")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Strategy comparison saved → {save_path}")
        plt.close(fig)

"""
Pipeline Orchestrator
Coordinates the entire ML pipeline from data loading to final scoring.

Stages:
  1. Data Preprocessing  (load → filter → clean → encode)
  2. Feature Engineering  (ratios, flags, interactions, advanced features)
  3. Cash Flow Analysis   (income, expense proxy, savings ratio, cash_flow_score)
  4. Model Training       (XGBoost + LightGBM, SMOTE, evaluation)
  5. Explainability       (SHAP summary + waterfall)
  6. Drift Detection      (PSI on train vs test)
  7. Bandit Optimization  (Thompson Sampling over 3 strategies)
  8. Final Scoring        (composite score → APPROVE / REVIEW / REJECT)
"""

import os
import sys
import logging
import warnings
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.feature_encoder import FeatureEncoder
from src.feature_engineer import FeatureEngineer
from src.cash_flow_analyzer import CashFlowAnalyzer
from src.model_trainer import ModelTrainer
from src.explainability import ModelExplainer
from src.drift_detector import DriftDetector
from src.bandit_optimizer import BanditOptimizer
from src.final_scorer import FinalScorer
import config

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """Orchestrates the complete ML pipeline."""

    def __init__(self):
        self.cfg = self._load_config()
        self.data_loader      = DataLoader(self.cfg["RAW_DATA_PATH"])
        self.preprocessor     = DataPreprocessor(self.cfg)
        self.encoder          = FeatureEncoder()
        self.feature_engineer = FeatureEngineer(self.cfg)
        self.cash_flow        = CashFlowAnalyzer(self.cfg)
        self.trainer          = ModelTrainer(self.cfg)
        self.drift_detector   = DriftDetector(self.cfg)
        self.bandit           = BanditOptimizer(self.cfg)
        self.final_scorer     = FinalScorer(self.cfg)

    # ── config ───────────────────────────────────────────────────────────
    def _load_config(self) -> Dict[str, Any]:
        """Load all configuration from config module."""
        return {
            # paths
            "RAW_DATA_PATH":  config.RAW_DATA_PATH,
            "CLEAN_DATA_PATH": config.CLEAN_DATA_PATH,
            "FEAT_DATA_PATH":  config.FEAT_DATA_PATH,
            "MODEL_PATH":     config.MODEL_PATH,
            "SCALER_PATH":    config.SCALER_PATH,
            "FEATURE_PATH":   config.FEATURE_PATH,
            "DATA_DIR":       config.DATA_DIR,
            "MODEL_DIR":      config.MODEL_DIR,
            "REPORTS_DIR":    config.REPORTS_DIR,
            # BNPL filtering
            "MAX_LOAN_AMOUNT":  config.MAX_LOAN_AMOUNT,
            "BNPL_TERM_LIST":   config.BNPL_TERM_LIST,
            "DEFAULT_STATUSES": config.DEFAULT_STATUSES,
            "SAFE_STATUSES":    config.SAFE_STATUSES,
            # feature engineering
            "HIGH_UTIL_THRESHOLD": config.HIGH_UTIL_THRESHOLD,
            "HIGH_DTI_THRESHOLD":  config.HIGH_DTI_THRESHOLD,
            "SPENDING_SPIKE_MULT": config.SPENDING_SPIKE_MULT,
            # class imbalance
            "SMOTE_RANDOM_STATE":      config.SMOTE_RANDOM_STATE,
            "SMOTE_SAMPLING_STRATEGY": config.SMOTE_SAMPLING_STRATEGY,
            # model training
            "TEST_SIZE":     config.TEST_SIZE,
            "RANDOM_STATE":  config.RANDOM_STATE,
            "XGB_PARAMS":    config.XGB_PARAMS,
            "LGBM_PARAMS":   config.LGBM_PARAMS,
            # cash flow
            "CASH_FLOW_WEIGHTS": config.CASH_FLOW_WEIGHTS,
            # final scoring
            "SCORE_WEIGHT_PD":       config.SCORE_WEIGHT_PD,
            "SCORE_WEIGHT_CASHFLOW": config.SCORE_WEIGHT_CASHFLOW,
            "APPROVE_THRESHOLD":     config.APPROVE_THRESHOLD,
            "REJECT_THRESHOLD":      config.REJECT_THRESHOLD,
            # drift
            "PSI_BUCKETS":         config.PSI_BUCKETS,
            "PSI_DRIFT_THRESHOLD": config.PSI_DRIFT_THRESHOLD,
            "PSI_WARN_THRESHOLD":  config.PSI_WARN_THRESHOLD,
            # bandit
            "BANDIT_STRATEGIES":      config.BANDIT_STRATEGIES,
            "BANDIT_ROUNDS":          config.BANDIT_ROUNDS,
            "BANDIT_LOAN_TERM_YEARS": config.BANDIT_LOAN_TERM_YEARS,
        }

    def _create_directories(self):
        """Create necessary directories."""
        for d in [self.cfg["DATA_DIR"], self.cfg["MODEL_DIR"],
                  self.cfg["REPORTS_DIR"]]:
            os.makedirs(d, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1 — Data Preprocessing
    # ══════════════════════════════════════════════════════════════════════
    def run_data_preprocessing(self):
        """Load raw data → filter → clean → encode."""
        logger.info("=" * 70)
        logger.info("STAGE 1 — DATA PREPROCESSING")
        logger.info("=" * 70)

        self._create_directories()

        KEEP_COLS = [
            "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
            "emp_length", "home_ownership", "annual_inc", "verification_status",
            "purpose", "addr_state", "loan_status",
            "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
            "revol_bal", "revol_util", "total_acc",
            "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
            "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
            "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
            "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m",
            "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
            "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
            "tax_liens", "tot_hi_cred_lim",
            "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
        ]
        
        # Load data passing specific columns to prevent OOM
        initial_cols = KEEP_COLS + ["term"]
        df = self.data_loader.load_csv(usecols=initial_cols)

        df = self.preprocessor.filter_bnpl_loans(df)
        df = self.preprocessor.build_target(df)
        df = self.preprocessor.select_columns(df, KEEP_COLS + ["default"])
        df = self.preprocessor.handle_missing(df)
        df = self.preprocessor.treat_outliers(df)
        df = self.encoder.encode_all(df)

        df.to_csv(self.cfg["CLEAN_DATA_PATH"], index=False)
        logger.info(f"Clean data saved → {self.cfg['CLEAN_DATA_PATH']} ({df.shape})")
        return df

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2 — Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    def run_feature_engineering(self, df=None):
        """Apply feature engineering transformations."""
        logger.info("=" * 70)
        logger.info("STAGE 2 — FEATURE ENGINEERING")
        logger.info("=" * 70)

        if df is None:
            import pandas as pd
            df = pd.read_csv(self.cfg["CLEAN_DATA_PATH"])
            logger.info(f"Loaded clean data: {df.shape}")

        df = self.feature_engineer.engineer_all_features(df)

        df.to_csv(self.cfg["FEAT_DATA_PATH"], index=False)
        logger.info(f"Featured data saved → {self.cfg['FEAT_DATA_PATH']}")
        return df

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3 — Cash Flow Analysis
    # ══════════════════════════════════════════════════════════════════════
    def run_cash_flow_analysis(self, df=None):
        """Compute cash-flow features and cash_flow_score."""
        logger.info("=" * 70)
        logger.info("STAGE 3 — CASH FLOW ANALYSIS")
        logger.info("=" * 70)

        if df is None:
            import pandas as pd
            df = pd.read_csv(self.cfg["FEAT_DATA_PATH"])

        df = self.cash_flow.compute_cash_flow_features(df)
        logger.info(f"Data shape after cash-flow features: {df.shape}")
        return df

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4 — Model Training
    # ══════════════════════════════════════════════════════════════════════
    def run_model_training(self, df=None):
        """Train XGBoost + LightGBM, evaluate, save best."""
        logger.info("=" * 70)
        logger.info("STAGE 4 — MODEL TRAINING")
        logger.info("=" * 70)

        if df is None:
            import pandas as pd
            df = pd.read_csv(self.cfg["FEAT_DATA_PATH"])

        # Drop cash-flow columns that are NOT used as model features
        # (cash_flow_score is used in final scoring, not as a model input)
        model_drop = ["monthly_income", "monthly_expense_proxy",
                      "savings_ratio", "expense_volatility", "cash_flow_score"]
        model_df = df.drop(columns=[c for c in model_drop if c in df.columns],
                           errors="ignore")

        results = self.trainer.train_all_models(model_df)

        # Save model
        self.trainer.save_model(
            self.cfg["MODEL_PATH"], self.cfg["SCALER_PATH"],
            self.cfg["FEATURE_PATH"],
            results["best_model"], results["scaler"], results["feature_names"]
        )

        # ── Visualisations ───────────────────────────────────────────────
        reports = self.cfg["REPORTS_DIR"]

        # ROC curves
        ModelTrainer.plot_roc_curves(
            results["roc_curves"],
            save_path=os.path.join(reports, "roc_curves.png"),
        )

        # Feature importance
        ModelTrainer.plot_feature_importance(
            results["best_model"], results["feature_names"],
            save_path=os.path.join(reports, "feature_importance.png"),
        )

        # Confusion matrix
        y_pred = results["best_model"].predict(results["X_test"])
        ModelTrainer.plot_confusion_matrix(
            results["y_test"], y_pred,
            save_path=os.path.join(reports, "confusion_matrix.png"),
        )

        return results

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5 — Explainability (SHAP)
    # ══════════════════════════════════════════════════════════════════════
    def run_explainability(self, model, feature_names, X_test):
        """SHAP explanations — global and local."""
        logger.info("=" * 70)
        logger.info("STAGE 5 — EXPLAINABILITY (SHAP)")
        logger.info("=" * 70)

        explainer = ModelExplainer(model, feature_names)
        explainer.compute_shap_values(X_test, max_samples=2000)

        reports = self.cfg["REPORTS_DIR"]

        # Global summary
        explainer.plot_summary(
            save_path=os.path.join(reports, "shap_summary.png"),
        )

        # Feature importance bar
        explainer.plot_feature_importance_bar(
            save_path=os.path.join(reports, "shap_importance.png"),
        )

        # Local explanation for first applicant
        explainer.plot_waterfall(
            idx=0,
            save_path=os.path.join(reports, "shap_waterfall.png"),
        )

        # Print top factors
        local = explainer.explain_single(idx=0)
        logger.info("Top risk factors for sample applicant:")
        for feat, val in local["top_risk_factors"]:
            logger.info(f"  ⬆ {feat}: {val:+.4f}")
        logger.info("Top protective factors:")
        for feat, val in local["top_protective_factors"]:
            logger.info(f"  ⬇ {feat}: {val:+.4f}")

        return explainer

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6 — Drift Detection (PSI)
    # ══════════════════════════════════════════════════════════════════════
    def run_drift_detection(self, df):
        """
        Simulate drift detection by comparing first 80% (reference)
        vs last 20% (current) of the dataset.
        """
        logger.info("=" * 70)
        logger.info("STAGE 6 — DRIFT DETECTION (PSI)")
        logger.info("=" * 70)

        split = int(len(df) * 0.8)
        ref_df = df.iloc[:split]
        cur_df = df.iloc[split:]

        drift_report = self.drift_detector.check_drift(ref_df, cur_df)

        # Plot
        DriftDetector.plot_psi(
            drift_report["psi_table"],
            save_path=os.path.join(self.cfg["REPORTS_DIR"], "psi_drift.png"),
        )

        # Print table
        logger.info("\nPSI Report (top 15):")
        logger.info("\n" + drift_report["psi_table"].head(15).to_string(index=False))

        if drift_report["needs_retraining"]:
            logger.warning("🚨 DRIFT THRESHOLD EXCEEDED: Triggering automated retraining pipeline (simulated)...")
            
        return drift_report

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 7 — Bandit Optimization
    # ══════════════════════════════════════════════════════════════════════
    def run_bandit_optimization(self, model, X_test, y_test, df_test_raw):
        """
        Run Thompson Sampling bandit over test set to find optimal
        approval strategy.
        """
        logger.info("=" * 70)
        logger.info("STAGE 7 — MULTI-ARMED BANDIT OPTIMIZATION")
        logger.info("=" * 70)

        pd_scores  = model.predict_proba(X_test)[:, 1]
        loan_amnts = df_test_raw["loan_amnt"].values if "loan_amnt" in df_test_raw.columns \
                     else np.full(len(X_test), 3000.0)
        int_rates  = df_test_raw["int_rate"].values if "int_rate" in df_test_raw.columns \
                     else np.full(len(X_test), 14.0)

        bandit_result = self.bandit.optimize(pd_scores, y_test,
                                              loan_amnts, int_rates)
        reports = self.cfg["REPORTS_DIR"]

        # Plots
        BanditOptimizer.plot_convergence(
            bandit_result["arm_selections"], self.bandit.arms,
            save_path=os.path.join(reports, "bandit_convergence.png"),
        )
        BanditOptimizer.plot_strategy_comparison(
            bandit_result["strategy_stats"],
            save_path=os.path.join(reports, "bandit_strategies.png"),
        )

        # Log results
        for arm, stats in bandit_result["strategy_stats"].items():
            logger.info(
                f"  {arm.capitalize():14s} — Threshold: {stats['pd_threshold']:.2f}, "
                f"Selected: {stats['selection_pct']}%, "
                f"Avg Reward: ${stats['avg_reward']:,.0f}"
            )

        return bandit_result

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 8 — Final Scoring
    # ══════════════════════════════════════════════════════════════════════
    def run_final_scoring(self, model, X_test, y_test, cash_flow_scores,
                          bandit_result):
        """
        Combine PD + cash_flow_score → final score → decision.
        Uses thresholds from bandit optimizer.
        """
        logger.info("=" * 70)
        logger.info("STAGE 8 — FINAL COMPOSITE SCORING")
        logger.info("=" * 70)

        pd_scores = model.predict_proba(X_test)[:, 1]

        # Update thresholds from bandit
        thresholds = BanditOptimizer.derive_thresholds(
            bandit_result["optimal_threshold"]
        )
        self.final_scorer.update_thresholds(
            thresholds["approve_threshold"],
            thresholds["reject_threshold"],
        )

        # Score and decide
        scoring_result = self.final_scorer.score_and_decide(
            pd_scores, cash_flow_scores
        )

        reports = self.cfg["REPORTS_DIR"]

        # Plots
        self.final_scorer.plot_score_distribution(
            scoring_result["final_scores"],
            scoring_result["decisions"],
            save_path=os.path.join(reports, "final_score_distribution.png"),
        )
        FinalScorer.plot_decision_pie(
            scoring_result["decisions"],
            save_path=os.path.join(reports, "decision_pie.png"),
        )

        # Summary
        s = scoring_result["summary"]
        logger.info(f"\n  Total applicants : {s['total_applicants']:,}")
        logger.info(f"  Approved         : {s['approved']:,} ({s['approve_rate']}%)")
        logger.info(f"  Under review     : {s['review']:,}")
        logger.info(f"  Rejected         : {s['rejected']:,} ({s['reject_rate']}%)")
        logger.info(f"  Avg final score  : {s['avg_score']:.4f}")

        return scoring_result

    # ══════════════════════════════════════════════════════════════════════
    # MASTER PIPELINE
    # ══════════════════════════════════════════════════════════════════════
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete 8-stage pipeline end-to-end."""
        logger.info("=" * 80)
        logger.info("🏦 BNPL RISK SCORING SYSTEM — FULL PIPELINE")
        logger.info("=" * 80)

        # STAGE 1 — Preprocessing
        df_clean = self.run_data_preprocessing()

        # STAGE 2 — Feature Engineering
        df_feat = self.run_feature_engineering(df_clean)

        # STAGE 3 — Cash Flow Analysis
        df_full = self.run_cash_flow_analysis(df_feat)

        # Save cash_flow_score column for later use in final scoring
        import pandas as pd
        cash_flow_all = df_full["cash_flow_score"].values.copy()

        # STAGE 4 — Model Training
        training_results = self.run_model_training(df_full)

        best_model     = training_results["best_model"]
        feature_names  = training_results["feature_names"]
        X_test         = training_results["X_test"]
        y_test         = training_results["y_test"]

        # STAGE 5 — Explainability
        self.run_explainability(best_model, feature_names, X_test)

        # STAGE 6 — Drift Detection
        # Use the model-input DataFrame (without cash-flow columns)
        model_drop = ["monthly_income", "monthly_expense_proxy",
                      "savings_ratio", "expense_volatility", "cash_flow_score"]
        df_model = df_full.drop(columns=[c for c in model_drop
                                         if c in df_full.columns],
                                errors="ignore")
        self.run_drift_detection(df_model)

        # STAGE 7 — Bandit Optimization
        # We need raw loan_amnt and int_rate for reward calculation
        # X_test is the test-split portion; match indices from df_full
        from sklearn.model_selection import train_test_split
        n = len(df_model)
        _, test_idx = train_test_split(
            range(n), test_size=self.cfg["TEST_SIZE"],
            random_state=self.cfg["RANDOM_STATE"],
            stratify=df_model["default"].values,
        )
        df_test_raw = df_full.iloc[test_idx].reset_index(drop=True)

        bandit_result = self.run_bandit_optimization(
            best_model, X_test, y_test, df_test_raw
        )

        # Cash-flow scores for test set
        cf_test = df_test_raw["cash_flow_score"].values \
                  if "cash_flow_score" in df_test_raw.columns \
                  else cash_flow_all[test_idx]

        # STAGE 8 — Final Scoring
        scoring_result = self.run_final_scoring(
            best_model, X_test, y_test, cf_test, bandit_result
        )

        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"📊 Reports saved to: {self.cfg['REPORTS_DIR']}")
        logger.info("=" * 80)

        return {
            "training_results": training_results,
            "bandit_result":    bandit_result,
            "scoring_result":   scoring_result,
        }


if __name__ == "__main__":
    pipeline = MLPipeline()
    results = pipeline.run_full_pipeline()

    print("\n" + "=" * 60)
    print("FINAL MODEL RESULTS")
    print("=" * 60)
    for model_name, result in results["training_results"]["results"].items():
        m = result["metrics"]
        print(f"\n{model_name}:")
        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1-Score  : {m['f1_score']:.4f}")
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")

    print(f"\n🏆 Best Model: {results['training_results']['best_model_name']}")
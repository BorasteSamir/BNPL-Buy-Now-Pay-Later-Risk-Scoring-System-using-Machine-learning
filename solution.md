# 💳 Architectural Solution Document: Modular BNPL Risk Scoring System

**Target Audience:** Academic Evaluators, Senior MLOps Engineers, Tech Leads
**Context:** Buy Now Pay Later (BNPL) Machine Learning Pipeline Implementation

---

## 1. Executive Overview & Objectives

The primary objective of this project is to construct a **production-ready, real-time risk assessment pipeline** tailored specifically for the Buy Now Pay Later (BNPL) financial structure. Unlike traditional multi-year personal lending, BNPL relies on micro-loans, high transaction velocity, and thin-file borrowers. 

Given the reliance on historical, tabular datasets (specifically the LendingClub loan dataset), the core challenge was generating organic proxy variables for cash-flow stability and establishing a self-optimizing risk boundary without access to true real-time behavioral/graph data.

**Key Deliverables Achieved:**
1. **Dynamic Risk Scoring:** Probability of Default (PD) modeling merged with a native Cash Flow stability score.
2. **AI Explainability:** Implementation of SHAP values to decode "black-box" tree ensembles.
3. **Automated Cutoffs:** Multi-armed bandit mechanisms mapping raw machine learning metric space into optimized threshold strategies driven by Expected Return on Investment (ROI).
4. **Reliability Monitoring:** Active Population Stability Index (PSI) drift tracking triggering simulated structural retraining.

---

## 2. Core Methodologies & Techniques

### A. Non-Linear Array Modeling (Gradient Boosting)
Financial risk profiles are inherently non-linear (e.g., high debt is survivable with high income, but fatal with erratic utilization). The pipeline harnesses **XGBoost** and **LightGBM**. Both models inherently process categorical sparsity, missing nodes, and multicollinearity without collapsing.

### B. Class Imbalance Resolution (SMOTE)
Because defaults represent a statistical minority (approx. 15-18% of the dataset), predictive models organically learn to collapse towards majority-class prediction (the "Safe" bucket), yielding immense False Negatives. The system integrates **Synthetic Minority Over-sampling Technique (SMOTE)** locally during the training bounds to structurally balance the decision trees.

### C. Game Theory Thresholds (Thompson Sampling)
Instead of statically selecting a threshold (e.g., PD > 0.50), the system dynamically optimizes via a **Multi-Armed Bandit**. Using *Thompson Sampling* (Beta distributions mapping successes/failures), it simulates 1,000 rounds of deployment across 'Conservative,' 'Moderate,' and 'Aggressive' probability boundaries bounds to establish the maximum financial profit curve (`Profit = Interest Earned - Default Loss`).

### D. Model Interpretability (SHAP & Permutation)
Regulatory compliance required explicit decision explainability. **SHAP (SHapley Additive exPlanations)** is integrated natively via `TreeExplainer`, mapping the exact local marginal contribution of every single variable ensuring transparent `REJECT` or `REVIEW` reasoning. 

### E. Distribution Monitoring (Population Stability Index / PSI)
Data drift ruins static ML models. The system compares real-time/inference dataset arrays against the original Reference distribution across 10 decile buckets. If PSI exceeds `0.20`, the pipeline initiates a hard stop and logs a mandatory retraining alert.

---

## 3. System Architecture & Data Flow

The architecture embraces a strictly decoupled, 8-Stage Sequential Pattern orchestrated by a central `MLPipeline` engine.

**Data Flow Sequence:**
1. **Ingestion (`loan_data.csv`)** ➔ Strict parsing of 45 necessary columns (Memory Optimization).
2. **Preprocessing** ➔ Constraints applied (BNPL-like loans only: ≤$5k, short-term).
3. **Sub-Modules** ➔ Cash-flow arrays & Feature logic generation.
4. **Model Tuning Matrix** ➔ Array standardization, SMOTE balancing, GBM hyperparameter training.
5. **Analytics Dump** ➔ SHAP, PSI, Bandit convergence mapped to visualizations (`reports/`).
6. **API Output** ➔ REST API loads `.pkl` files and maps real-time applicants into the active scoring boundary.

---

## 4. Module-Wise Breakdown

### `src/data_loader.py` & `src/data_preprocessor.py`
- **Responsibility:** Ingest array mappings cleanly out of local storage constraints.
- **Academic Merit:** Bypasses conventional Out-Of-Memory (OOM) failures by constraining Pandas pointer ingestion (`usecols`). Normalizes the target variable, converting multi-class string flags (`Charged Off`, `Late`) into a binary 1/0 default classification.

### `src/feature_engineer.py`
- **Responsibility:** Expanding the statistical footprint of the base dataset.
- **Functionality:** Generates continuous ratios critical for neural evaluation. Calculates bounding factors like `debt_to_income_ratio`, multi-tier `repayment_consistency`, and active penalty bounds. Converts generic categorical nodes using `OneHotEncoder`.

### `src/cash_flow_analyzer.py`
- **Responsibility:** Synthesize non-existent transaction data from available credit boundaries.
- **Functionality:** Calculates a bounded **Cash-Flow Score (0 to 1)** using inferred `monthly_income` vs proxy `monthly_expense`. Used sequentially to protect borrowers with high ML-predicted risk but high short-term liquidity.

### `src/model_trainer.py`
- **Responsibility:** Evaluates and serializes the predictive cores.
- **Functionality:** Simultaneously builds LightGBM and XGBClassifier models on 80/20 splits. Compares both using strictly bound ROC-AUC metrics, dumping standard classification metrics (`Precision`, `Recall`, `F1`, `Support`), and automatically serializes the `best_model.pkl` to disk.

### `src/explainability.py` & `src/drift_detector.py`
- **Responsibility:** Risk compliance.
- **Functionality:** `explainability.py` safely wraps SHAP plotting parameters. If the complex TreeExplainer API breaks due to array incompatibilities, it executes a programmatic fallback to a `PermutationExplainer`. `drift_detector.py` computes log-based array bounds to map distributional sliding, providing visual bar charts for review.

### `src/bandit_optimizer.py` & `src/final_scorer.py`
- **Responsibility:** Final business logic orchestration.
- **Functionality:** Dissects the output probabilities dynamically. `final_scorer.py` maps the formula `w_pd * (1 - Default_Probability) + w_cf * CashFlowScore` against the exact numerical cutoff threshold selected natively by the Thompson Sample reinforcement learning loop.

---

## 5. Integration Points

The repository utilizes independent microservice patterns:
- **`src/pipeline.py` (The Orchestrator):** A monolithic controller orchestrating exactly how arrays are passed sequentially between the sub-modules above. It acts as the singular state manager.
- **`app.py` (The Serving Layer):** A heavily validated `FastAPI` instance. It statically loads the global `.pkl` files upon boot into a Singleton `ModelRegistry` class. It exposes a JSON schema via Pydantic matching the exact feature-array required by the pipeline.
- **`dashboard.py` (The Dashboard UI):** A `Streamlit` instance. The dashboard communicates indirectly with the backend ML models. It scrapes the `reports/` folder directly to surface the 10 static analytic charts generated by the orchestration model, while bouncing HTTP POST requests to `app.py` to stream live "Score Applicant" data to visually move the pressure gauges. 

---

## 6. Assumptions & Design Decisions

### Data Constraints & Proxy Design
**Assumption:** The LendingClub dataset contains multi-year mortgage-like data alongside generic lending structure. 
**Decision:** We aggressively filter rows ensuring `loan_amnt <= 5000` and `term = 36` to mathematically proxy BNPL structures. We strictly assume `installment` constraints function interchangeably with BNPL staggered payments.

### Model Ensembling Bias vs Variance
**Assumption:** We assume high class imbalance (18% default rate) will crash standard accuracy metrics.
**Decision:** Selected `ROC-AUC` as the exclusive tournament metric in `model_trainer.py` instead of aggregate Accuracy. SMOTE was constrained strictly to the training split, preventing data-leakage during validation inference.

### Hardware Optimizations
**Decision:** To guarantee execution on standardized academic laptops running Python, we explicitly avoided native neural pipelines (TensorFlow/PyTorch) matching the tabular nature of the dataframe, strictly prioritizing vectorized operations inside Pandas and passing hard `usecols` maps to prevent Out-Of-Memory segmentation allocation faults.

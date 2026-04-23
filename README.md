# BNPL Risk Scoring System

A machine learning system that predicts the probability of loan default for Buy Now Pay Later (BNPL) applications. It scores each applicant in real time and returns an **APPROVE**, **REVIEW**, or **REJECT** decision.

---

## What This System Does

- Predicts the **probability of default (PD)** for a loan applicant
- Computes a **cash flow score** based on income, expenses, and savings
- Combines both into a **final risk score** to make a lending decision
- Detects **data drift** using Population Stability Index (PSI)
- Optimizes approval thresholds using a **Multi-Armed Bandit** algorithm
- Exposes a **REST API** for real-time scoring
- Provides an **interactive dashboard** to explore results

---

## Project Structure

```
BNPL_Risk_Scoring system/
│
├── src/                        # All core logic lives here
│   ├── data_loader.py          # Loads and validates raw CSV
│   ├── data_preprocessor.py    # Cleans and filters data
│   ├── feature_encoder.py      # Encodes categorical variables
│   ├── feature_engineer.py     # Creates new features
│   ├── model_trainer.py        # Trains XGBoost / LightGBM models
│   ├── cash_flow_analyzer.py   # Computes cash flow score
│   ├── drift_detector.py       # PSI-based drift detection
│   ├── bandit_optimizer.py     # Multi-armed bandit threshold optimizer
│   ├── explainability.py       # SHAP explanations
│   ├── final_scorer.py         # Combines PD + cash flow into final score
│   └── pipeline.py             # Orchestrates all modules
│
├── data/                       # Put your dataset here
│   └── loan_data.csv           # (you download this — see Step 3)
│
├── models/                     # Trained model files saved here automatically
├── reports/                    # Charts and plots saved here automatically
│
├── main.py                     # Run this to start everything
├── train_from_clean.py         # Run this if you already have clean_data.csv
├── dashboard.py                # Streamlit interactive dashboard
├── app.py                      # FastAPI REST API server
├── config.py                   # All settings and thresholds
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Requirements

Before you start, make sure you have the following installed on your computer:

- **Python 3.10 or 3.11** — [Download here](https://www.python.org/downloads/)
- **pip** — comes with Python
- **Git** — [Download here](https://git-scm.com/downloads) (only needed to clone the repo)
- At least **4 GB of free RAM** (the dataset is large)
- At least **3 GB of free disk space**

To check your Python version, open a terminal and run:
```bash
python --version
```

---

## Step-by-Step Setup

### Step 1 — Get the Code

Clone this repository to your computer:

```bash
git clone https://github.com/<username>/bnpl-risk-scoring.git
cd bnpl-risk-scoring
```

Or download the ZIP from GitHub and extract it.

---

### Step 2 — Create a Virtual Environment

A virtual environment keeps the project dependencies isolated from your system Python.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line. This means the virtual environment is active.

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> This will take 3–7 minutes the first time. It installs pandas, XGBoost, LightGBM, SHAP, FastAPI, Streamlit, and other packages.

---

### Step 4 — Download the Dataset

This project uses the **LendingClub loan dataset** from Kaggle.

1. Go to this link: [https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Create a free Kaggle account if you don't have one
3. Click the **Download** button
4. Extract the ZIP file
5. Find the file named `accepted_2007_to_2018Q4.csv` inside the extracted folder
6. Rename it to `loan_data.csv`
7. Place it inside the `data/` folder of this project

Your folder should look like this:
```
data/
└── loan_data.csv    ← file goes here (about 1.6 GB)
```

> The file is large (~1.6 GB). Make sure you have enough disk space before downloading.

---

### Step 5 — Run the Pipeline

Now you are ready to run the system. Simply run:

```bash
python main.py
```

This will show an interactive menu:

```
======================================================================
BNPL RISK SCORING SYSTEM
Buy Now Pay Later Credit Risk Assessment
======================================================================

SYSTEM STATUS
------------------------------
Dataset:       Found
Clean Data:    Missing
Featured Data: Missing
Trained Model: Missing

MAIN MENU
--------------------
1. Run ML Pipeline (Data → Features → Model)
2. Start API Server
3. Start Dashboard
4. Run Pipeline + Start API
5. Show Status
6. Exit

Select option (1-6):
```

- **Select option 1** to run the full pipeline for the first time
- This will preprocess the data, engineer features, and train the model
- It takes about **5–15 minutes** depending on your machine

---

### Step 6 — Start the API Server

After training is complete, start the REST API:

```bash
python main.py
# then select option 2
```

Or directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

- API will be live at: `http://localhost:8000`
- Interactive API docs at: `http://localhost:8000/docs`

---

### Step 7 — Open the Dashboard

Open a **new terminal**, activate the virtual environment again, then run:

```bash
streamlit run dashboard.py
```

- Dashboard opens automatically in your browser at: `http://localhost:8501`
- You can score individual applicants, run batch simulations, and view model reports

---

## Already Have clean_data.csv?

If you already have a preprocessed `clean_data.csv` file (e.g. from a previous run or shared by someone), you can skip the raw data preprocessing and go straight to training:

```bash
python train_from_clean.py
```

This will:
1. Load `clean_data.csv` from the `data/` folder
2. Run feature engineering
3. Train the model
4. Save all model artifacts to `models/`

---

## How the Pipeline Works

Here is what happens when you run the full pipeline:

```
loan_data.csv (raw, 1.6 GB)
        |
        v
[Data Loader]         — loads and validates the CSV
        |
        v
[Data Preprocessor]   — filters BNPL loans, builds default label,
                        handles missing values, treats outliers
        |
        v
[Feature Encoder]     — encodes grade, sub_grade, emp_length,
                        one-hot encodes home_ownership, purpose, etc.
        |
        v
[Feature Engineer]    — creates 24+ new features:
                        loan_to_income, payment_burden,
                        risk flags, interaction terms, etc.
        |
        v
[Cash Flow Analyzer]  — computes cash_flow_score from
                        income, expenses, savings ratio, DTI
        |
        v
[Model Trainer]       — applies SMOTE, trains XGBoost + LightGBM,
                        selects best model by ROC-AUC
        |
        v
[Drift Detector]      — computes PSI to detect data drift
        |
        v
[Bandit Optimizer]    — optimizes approval thresholds dynamically
        |
        v
[Final Scorer]        — combines PD + cash_flow_score into
                        final score → APPROVE / REVIEW / REJECT
        |
        v
[Explainability]      — SHAP values for model transparency
```

---

## Expected Output After Training

```
STARTING FEATURE ENGINEERING
Loaded clean data: (297324, 61)
Default rate: 10.44%
Created 24 new features. Total columns: 85

STARTING MODEL TRAINING
Features: 84, Samples: 297324
Class distribution: {0: 266294, 1: 31030}
After SMOTE: [213035, 213035]
Training Logistic Regression...
Training Random Forest...
Training XGBoost...

XGBoost  — ROC-AUC: 0.7123  Precision: 0.47  Recall: 0.63
LightGBM — ROC-AUC: 0.7340  Precision: 0.51  Recall: 0.61

Best Model: LightGBM
Model artifacts saved to models/
```

---

## Configuration

All settings are in `config.py`. You can change these without touching any other file:

| Setting | Default | What it does |
|---|---|---|
| `MAX_LOAN_AMOUNT` | 5000 | Only keep loans below this amount (BNPL filter) |
| `BNPL_TERM_LIST` | [36] | Only keep 36-month term loans |
| `DEFAULT_THRESHOLD` | 0.40 | Probability above this → REJECT |
| `APPROVE_THRESHOLD` | 0.65 | Final score above this → APPROVE |
| `REJECT_THRESHOLD` | 0.35 | Final score below this → REJECT |
| `PSI_DRIFT_THRESHOLD` | 0.20 | PSI above this triggers retraining alert |
| `TEST_SIZE` | 0.20 | 20% of data used for testing |
| `SCORE_WEIGHT_PD` | 0.60 | Weight of default probability in final score |
| `SCORE_WEIGHT_CASHFLOW` | 0.40 | Weight of cash flow score in final score |

---

## API Usage

Once the API server is running, you can score an applicant by sending a POST request:

```bash
curl -X POST "http://localhost:8000/api/v1/score" \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 2500,
    "annual_inc": 48000,
    "dti": 18.5,
    "int_rate": 14.5,
    "grade": 3,
    "revol_util": 55.0,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1
  }'
```

Response:
```json
{
  "default_prob": 0.23,
  "cash_flow_score": 0.71,
  "final_score": 0.67,
  "decision": "APPROVE",
  "risk_band": "MEDIUM",
  "confidence": "HIGH",
  "latency_ms": 4.2
}
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pandas'`**
- Your virtual environment is not activated
- Run `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
- Then run `pip install -r requirements.txt` again

**`FileNotFoundError: Dataset not found`**
- You haven't placed `loan_data.csv` in the `data/` folder
- Follow Step 4 above carefully

**`MemoryError` during SMOTE or training**
- Your machine doesn't have enough RAM
- Close other applications and try again
- Or reduce the dataset by editing `MAX_LOAN_AMOUNT` in `config.py` to a smaller value like `3000`

**`UnicodeEncodeError` in terminal**
- This is a Windows terminal encoding issue and does not affect results
- The pipeline still runs correctly

**API returns `model not found`**
- You haven't trained the model yet
- Run `python main.py` and select option 1 first

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas / numpy | Data processing |
| XGBoost / LightGBM | Credit risk model |
| imbalanced-learn | SMOTE for class imbalance |
| SHAP | Model explainability |
| FastAPI | REST API server |
| Streamlit | Interactive dashboard |
| Plotly / Matplotlib | Visualizations |
| scikit-learn | Preprocessing and evaluation |

---

## Dataset Credit

Dataset sourced from Kaggle:
**LendingClub Loan Data (2007–2018)**
[https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

- Contains over 2.2 million loan records
- Includes loan amount, interest rate, grade, employment info, credit history, and repayment status
- We filter it down to ~300,000 BNPL-relevant records (small loans, 36-month term)

---

## License

This project is for educational and research purposes.

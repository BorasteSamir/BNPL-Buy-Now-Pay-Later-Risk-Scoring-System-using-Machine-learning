"""
BNPL Risk Scoring System - Main Entry Point
Just run: python main.py

What happens automatically:
  1. Checks if model is trained — if not, runs the training pipeline
  2. Starts the FastAPI server in the background
  3. Waits for the API to be ready
  4. Opens the Streamlit dashboard in your browser
  5. Everything runs until you press Ctrl+C
"""

import os
import sys
import time
import subprocess
import threading
import webbrowser
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def print_banner():
    print("=" * 60)
    print("  BNPL RISK SCORING SYSTEM")
    print("  Buy Now Pay Later Credit Risk Assessment")
    print("=" * 60)

def check_model_exists():
    return (
        os.path.exists(config.MODEL_PATH) and
        os.path.exists(config.SCALER_PATH) and
        os.path.exists(config.FEATURE_PATH)
    )

def check_clean_data_exists():
    return os.path.exists(config.CLEAN_DATA_PATH)

def check_raw_data_exists():
    # Check for any common naming variation
    data_dir = config.DATA_DIR
    possible = ["loan_data.csv", "Loan_data.csv", "loan.csv",
                "accepted_2007_to_2018Q4.csv", "lending_club.csv"]
    return any(os.path.exists(os.path.join(data_dir, f)) for f in possible)

def wait_for_api(url: str, timeout: int = 30) -> bool:
    """Poll the API health endpoint until it responds or times out."""
    print("  Waiting for API to be ready", end="", flush=True)
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(" Ready!")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" Timed out.")
    return False

# ─────────────────────────────────────────────
# STEP 1 — TRAINING
# ─────────────────────────────────────────────

def run_training():
    """Run training from clean_data.csv if it exists, else full pipeline."""
    print("\n[1/3] Running Training Pipeline...")
    print("-" * 40)

    if check_clean_data_exists():
        print("  Found clean_data.csv — skipping preprocessing")
        print("  Running feature engineering + model training...")
        result = subprocess.run(
            [sys.executable, "train_from_clean.py"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    elif check_raw_data_exists():
        print("  Found raw dataset — running full pipeline...")
        result = subprocess.run(
            [sys.executable, "-c",
             "from src.pipeline import MLPipeline; MLPipeline().run_full_pipeline()"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
    else:
        print("\n  ERROR: No dataset found!")
        print(f"  Please place loan_data.csv in: {config.DATA_DIR}")
        print("  Download from: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        sys.exit(1)

    if result.returncode != 0:
        print("\n  ERROR: Training failed. Check the logs above.")
        sys.exit(1)

    print("  Training complete!")

# ─────────────────────────────────────────────
# STEP 2 — API SERVER (background thread)
# ─────────────────────────────────────────────

api_process = None

def start_api_background():
    """Start the FastAPI server as a background subprocess."""
    global api_process
    print("\n[2/3] Starting API Server...")
    print("-" * 40)

    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    print("  API server starting on http://localhost:8000")

# ─────────────────────────────────────────────
# STEP 3 — DASHBOARD
# ─────────────────────────────────────────────

dashboard_process = None

def start_dashboard():
    """Start the Streamlit dashboard as a background subprocess."""
    global dashboard_process
    print("\n[3/3] Starting Dashboard...")
    print("-" * 40)

    dashboard_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py",
         "--server.port", "8501",
         "--server.headless", "false"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("  Dashboard starting on http://localhost:8501")

# ─────────────────────────────────────────────
# SHUTDOWN
# ─────────────────────────────────────────────

def shutdown():
    """Gracefully stop all background processes."""
    print("\n\nShutting down...")
    if api_process:
        api_process.terminate()
        print("  API server stopped.")
    if dashboard_process:
        dashboard_process.terminate()
        print("  Dashboard stopped.")
    print("Goodbye!")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Create required directories
    os.makedirs(config.DATA_DIR,    exist_ok=True)
    os.makedirs(config.MODEL_DIR,   exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    print_banner()

    # Step 1 — Train if model doesn't exist
    if check_model_exists():
        print("\n[1/3] Model already trained — skipping training.")
    else:
        run_training()

    # Step 2 — Start API server in background
    start_api_background()

    # Wait until API is ready before opening dashboard
    api_ready = wait_for_api("http://localhost:8000/api/v1/health", timeout=60)
    if not api_ready:
        print("  WARNING: API did not respond in time. Dashboard may show offline.")

    # Step 3 — Start dashboard
    start_dashboard()

    # Give dashboard a moment to start then open browser
    time.sleep(3)
    webbrowser.open("http://localhost:8501")

    print("\n" + "=" * 60)
    print("  Everything is running!")
    print("  API:       http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("  Dashboard: http://localhost:8501")
    print("=" * 60)
    print("\n  Press Ctrl+C to stop everything.\n")

    # Keep running until Ctrl+C
    try:
        while True:
            # Restart API if it crashes
            if api_process and api_process.poll() is not None:
                print("  API server stopped unexpectedly. Restarting...")
                start_api_background()
            time.sleep(5)
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()

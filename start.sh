#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

echo "PULSE: Starting local launcher"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "PULSE: Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "PULSE: Installing dependencies"
python -m pip install --upgrade pip setuptools wheel >/dev/null
python -m pip install --prefer-binary -r requirements.txt

if [[ ! -f "data/system_metrics.csv" ]]; then
  echo "PULSE: system_metrics.csv not found -> collecting quick starter data"
  python data_collection.py --samples 30 --delay 1 --scenario normal
fi

if [[ ! -f "data/system_metrics_labeled.csv" ]]; then
  echo "PULSE: labeled dataset not found -> preprocessing (forecast mode)"
  python preprocess.py --label-mode forecast --horizon-steps 5 --window-sizes 3,5
fi

if [[ ! -f "models/logistic_regression.pkl" || ! -f "models/random_forest.pkl" || ! -f "models/svm.pkl" ]]; then
  echo "PULSE: one or more model files missing -> training models"
  python train.py --test-size 0.2 --horizon-steps 5 || {
    echo "PULSE: Training failed. Check train.py before launching app."
    exit 1
  }
fi

echo "PULSE: Launching Streamlit app at http://localhost:8501"
exec streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false

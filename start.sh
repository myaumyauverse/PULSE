#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-quick}"

cd "$PROJECT_DIR"

echo "PULSE: Starting local launcher"
echo "PULSE: Bootstrap mode = $BOOTSTRAP_MODE"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "PULSE: Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "PULSE: Installing dependencies"
python -m pip install --upgrade pip setuptools wheel >/dev/null
python -m pip install --prefer-binary -r requirements.txt

need_collection=0
need_preprocess=0
need_train=0

if [[ ! -f "data/system_metrics.csv" ]]; then
  need_collection=1
fi

if [[ ! -f "data/system_metrics_labeled.csv" ]]; then
  need_preprocess=1
fi

if [[ ! -f "models/logistic_regression.pkl" || ! -f "models/random_forest.pkl" || ! -f "models/svm.pkl" || ! -f "models/training_summary.json" ]]; then
  need_train=1
fi

if [[ "$need_collection" -eq 1 || "$need_preprocess" -eq 1 || "$need_train" -eq 1 ]]; then
  if [[ "$BOOTSTRAP_MODE" == "full" ]]; then
    if command -v stress-ng >/dev/null 2>&1; then
      echo "PULSE: running full bootstrap via auto_collect_and_train.sh"
      chmod +x auto_collect_and_train.sh
      ./auto_collect_and_train.sh
    else
      echo "PULSE: BOOTSTRAP_MODE=full requested but stress-ng is missing."
      echo "PULSE: Install stress-ng for full mode or use BOOTSTRAP_MODE=quick."
      exit 1
    fi
  else
    if [[ "$need_collection" -eq 1 ]]; then
      echo "PULSE: system_metrics.csv not found -> collecting quick starter data"
      python data_collection.py --samples 120 --delay 1 --scenario normal
    fi

    if [[ "$need_preprocess" -eq 1 || "$need_collection" -eq 1 ]]; then
      echo "PULSE: preprocessing dataset (forecast mode)"
      python preprocess.py --label-mode forecast --horizon-steps 5 --window-sizes 3,5
    fi

    if [[ "$need_train" -eq 1 || "$need_preprocess" -eq 1 || "$need_collection" -eq 1 ]]; then
      echo "PULSE: training models"
      python train.py --test-size 0.2 --horizon-steps 5 --split-type stratified --random-state 42 || {
        echo "PULSE: Training failed. Check train.py before launching app."
        exit 1
      }
      echo "PULSE: evaluating models"
      python evaluate.py --test-size 0.2 || true
    fi
  fi
fi

echo "PULSE: Launching Streamlit app at http://localhost:8501"
exec streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false

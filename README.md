# PULSE — Predictive Usage & Load Spike Estimator

A beginner-friendly machine learning project that predicts system spikes using real-time metrics with a Streamlit dashboard.

## Overview

PULSE collects system metrics (CPU, RAM, processes), preprocesses them, trains ML models, and provides real-time spike predictions through a local web dashboard.

## Features

- **Real-time System Metrics**: Captures CPU, RAM, and process count continuously
- **ML Models**: Trains Logistic Regression, Random Forest, and SVM
- **Live Predictions**: Predicts system spikes with clear alert status (✅ stable or 🚨 spike alert)
- **Interactive Dashboard**: Monochrome Streamlit UI with live metrics, model comparison, ensemble confidence, activity feed, and history download
- **Clean Git Workflow**: Feature branches for each development phase

## Project Structure

```
PULSE/
├── data/
│   ├── system_metrics.csv           # Raw collected metrics
│   └── system_metrics_labeled.csv    # Preprocessed data with spike labels
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── data_collection.py      # Phase 1: Collect system metrics
├── preprocess.py           # Phase 2: Create spike labels
├── train.py                # Phase 3: Train models
├── evaluate.py             # Phase 4: Evaluate model metrics
├── predict_live.py         # Phase 5: Real-time predictions
├── app.py                  # Phase 6: Streamlit dashboard
├── start.sh                # One-command local launcher
├── requirements.txt        # Project dependencies
├── .gitignore              # Ignore local environment/cache files
└── README.md              # This file
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/myaumyauverse/PULSE.git
cd PULSE
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Quick start (recommended)
```bash
chmod +x start.sh
./start.sh
```
This script prepares the environment, checks data/model artifacts, and launches Streamlit locally.

## Usage

### Phase 1: Collect System Metrics (Diverse Scenarios)
```bash
python data_collection.py --samples 100 --delay 2 --scenario normal
```
- `--samples`: Number of data points to collect (default: 30)
- `--delay`: Seconds between samples (default: 2)
- `--scenario`: Dataset tag (`idle`, `normal`, `cpu_stress`, `ram_stress`, `mixed`)

Recommended balanced collection for better model accuracy:
```bash
# baseline
python data_collection.py --samples 120 --delay 2 --scenario idle
python data_collection.py --samples 200 --delay 2 --scenario normal

# while running stress workload
python data_collection.py --samples 150 --delay 1 --scenario cpu_stress
python data_collection.py --samples 150 --delay 1 --scenario ram_stress
python data_collection.py --samples 150 --delay 1 --scenario mixed
```

### Phase 2: Preprocess Data (Forecast Labels + Temporal Features)
```bash
python preprocess.py --label-mode forecast --horizon-steps 5 --window-sizes 3,5
```
Creates forecast target and temporal features:
- `spike=1` means CPU is expected to cross threshold at `t + horizon`
- Adds lag/rolling/delta features for CPU, RAM, and process count

Legacy simple mode is still available:
```bash
python preprocess.py --label-mode simple --cpu-threshold 80
```

### Phase 3: Train Models (Chronological Train/Test Split)
```bash
python train.py --test-size 0.2 --horizon-steps 5
```
Trains and saves:
- Logistic Regression
- Random Forest (100 estimators)
- SVM (RBF kernel)

Training now uses chronological split:
- oldest 80% rows -> train
- latest 20% rows -> held-out test

### Phase 4: Evaluate Models
```bash
python evaluate.py --test-size 0.2
```
Displays train and held-out test metrics:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix
- Train-test F1 gap (overfitting signal)

### Phase 5: Real-Time Forecast Prediction
```bash
python predict_live.py --samples 5
```
Makes 5 live near-future forecasts using temporal feature window and soft-voting ensemble

### Phase 6: Launch Dashboard (Recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with:
- Live metrics display
- Spike prediction alert
- Model comparison table and ensemble confidence meter
- Metrics history chart + downloadable CSV
- Recent activity feed

## How It Works

**Forecast Label Rule (default mode):**
```
spike_t = 1 if CPU_percent at (t + horizon) > threshold else 0
```

**Model Training:**
- Features: current + temporal features (lags, rolling means/std, deltas)
- Target: Future spike (0 or 1)
- Strategy: chronological split training with held-out testing

**Real-Time Forecasting:**
- Captures current system state continuously
- Builds live temporal feature row from recent history
- Loads trained models
- Averages model spike probabilities (soft voting)
- Alert: If average > 0.5 -> high near-future spike likelihood

## Viva Explanation Points

1. **Data Collection**: Uses psutil to capture real-time metrics with timestamps
2. **Preprocessing**: Simple rule-based labeling (CPU > 80 = spike)
3. **Model Selection**: Three diverse models (linear, ensemble, SVM) for robustness
4. **Evaluation**: Standard ML metrics (accuracy, precision, recall)
5. **Real-Time**: Live predictions without retraining
6. **UI**: Interactive dashboard for visualization and monitoring

## Git Workflow

All features were developed using feature branches:

```bash
main (stable)
  ├── feature/data-collection    → merged to dev
  ├── feature/preprocessing      → merged to dev
  ├── feature/model-training     → merged to dev
  ├── feature/evaluation         → merged to dev
  ├── feature/prediction         → merged to dev
  └── feature/ui-streamlit       → merged to dev → merged to main
```

## Troubleshooting

### Models not found
```bash
python train.py  # Train models first
```

### No spike data
```bash
python data_collection.py  # Collect more data with higher system load
```

### Streamlit not starting
```bash
pip install --upgrade streamlit
```

### Download button text not visible
Hard refresh the browser after CSS updates:
```bash
# In browser
Ctrl+Shift+R
```

## Performance Notes

- Training with <100 samples uses synthetic spike augmentation (for demo purposes)
- Real-time metrics collection has ~1 second overhead
- Dashboard updates every 3 seconds by default (configurable)

## Future Enhancements

- Time-series forecasting (LSTM)
- Anomaly detection
- Alert notifications
- Historical data export
- Multi-machine monitoring

---

**Created for**: ML Lab Viva  
**Repository**: https://github.com/myaumyauverse/PULSE  
**Python Version**: 3.8+

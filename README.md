# PULSE — Predictive Usage & Load Spike Estimator

A beginner-friendly machine learning project that predicts system spikes using real-time metrics with a Streamlit dashboard.

## Overview

PULSE collects system metrics (CPU, RAM, processes), preprocesses them, trains ML models, and provides real-time spike predictions through a local web dashboard.

## Features

- **Real-time System Metrics**: Captures CPU, RAM, and process count continuously
- **ML Models**: Trains Logistic Regression, Random Forest, and SVM
- **Live Predictions**: Predicts system spikes with clear alert status (✅ stable or 🚨 spike alert)
- **Interactive Dashboard**: Streamlit UI with live metrics and history graphs
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
├── requirements.txt        # Project dependencies
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

## Usage

### Phase 1: Collect System Metrics
```bash
python data_collection.py --samples 100 --delay 2
```
- `--samples`: Number of data points to collect (default: 30)
- `--delay`: Seconds between samples (default: 2)

### Phase 2: Preprocess Data
```bash
python preprocess.py
```
Creates spike labels: `spike=1` if CPU > 80%, else `spike=0`

### Phase 3: Train Models
```bash
python train.py
```
Trains and saves:
- Logistic Regression
- Random Forest (100 estimators)
- SVM (RBF kernel)

### Phase 4: Evaluate Models
```bash
python evaluate.py
```
Displays: Accuracy, Precision, Recall, Confusion Matrix

### Phase 5: Real-Time Prediction
```bash
python predict_live.py --samples 5
```
Makes 5 live predictions with spike alerts

### Phase 6: Launch Dashboard (Recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with:
- Live metrics display
- Spike prediction alert
- 50-sample history graph

## How It Works

**Spike Label Rule:**
```
spike = 1 if CPU_percent > 80 else 0
```

**Model Training:**
- Features: [CPU%, RAM%, Process Count]
- Target: Spike (0 or 1)
- Strategy: Trains on preprocessed data, saves models as pickle files

**Real-Time Prediction:**
- Captures current system state
- Loads trained models
- Averages predictions from all 3 models
- Alert: If average > 0.5 → spike alert, else stable

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

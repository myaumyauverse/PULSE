import joblib
import pandas as pd
import psutil
import streamlit as st
from pathlib import Path


st.set_page_config(page_title="PULSE", layout="wide")

# Title and header
st.markdown("# 🔥 PULSE — Predictive Usage & Load Spike Estimator")
st.markdown("---")

# Sidebar for controls
st.sidebar.markdown("## Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh metrics", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)

# Column layout for metrics
col1, col2, col3 = st.columns(3)

# Get current system metrics
cpu_percent = psutil.cpu_percent(interval=1)
ram_percent = psutil.virtual_memory().percent
process_count = len(psutil.pids())

# Display live metrics
with col1:
    st.metric(label="CPU Usage", value=f"{cpu_percent:.1f}%", delta=None)

with col2:
    st.metric(label="RAM Usage", value=f"{ram_percent:.1f}%", delta=None)

with col3:
    st.metric(label="Process Count", value=process_count, delta=None)

st.markdown("---")

# Prediction section
st.markdown("## 🚨 Spike Prediction")

models_dir = Path("models")
models = {
    "Logistic Regression": models_dir / "logistic_regression.pkl",
    "Random Forest": models_dir / "random_forest.pkl",
    "SVM": models_dir / "svm.pkl",
}

predictions = []
for model_name, model_path in models.items():
    if model_path.exists():
        model = joblib.load(model_path)
        X = [[cpu_percent, ram_percent, process_count]]
        pred = model.predict(X)[0]
        predictions.append(pred)

if predictions:
    avg_pred = sum(predictions) / len(predictions)
    if avg_pred > 0.5:
        st.error("🚨 **PULSE ALERT: HIGH chance of system spike incoming!**")
        alert_status = "SPIKE DETECTED"
    else:
        st.success("✅ **PULSE Status: System stable**")
        alert_status = "STABLE"
else:
    st.warning("⚠️ Models not found. Please train models first.")
    alert_status = "N/A"

st.markdown("---")

# History graph
st.markdown("## 📊 Metrics History")

data_path = Path("data/system_metrics_labeled.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    if len(df) > 0:
        st.line_chart(df[["cpu_percent", "ram_percent"]].tail(50))
    else:
        st.info("No historical data yet. Run data collection first.")
else:
    st.info("No historical data yet. Run data collection first.")

st.markdown("---")

# Footer
st.markdown(
    """
    **PULSE Project** | Data Collection → Preprocessing → Model Training → Evaluation → Prediction → Streamlit UI
    
    For more info, visit: https://github.com/myaumyauverse/PULSE
    """
)

if auto_refresh:
    st.rerun()

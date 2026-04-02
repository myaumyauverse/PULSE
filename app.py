from datetime import datetime
import json
from pathlib import Path

import joblib
import pandas as pd
import psutil
import streamlit as st

from feature_engineering import BASE_FEATURE_COLUMNS, build_live_feature_frame


FEATURE_COLUMNS = ["cpu_percent", "ram_percent", "process_count"]
MODELS_DIR = Path("models")
SUMMARY_FILE = MODELS_DIR / "training_summary.json"
MODEL_FILES = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
    "Random Forest": MODELS_DIR / "random_forest.pkl",
    "SVM": MODELS_DIR / "svm.pkl",
}

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #ffffff;
  --bg-soft: #f7f7f7;
  --bg-muted: #efefef;
  --surface: #ffffff;
  --surface-elevated: #fafafa;

  --text-strong: #0a0a0a;
  --text: #171717;
  --text-muted: #525252;
  --text-soft: #737373;
  --text-inverse: #ffffff;

  --border: #e5e5e5;
  --border-strong: #d4d4d4;
  --divider: #ebebeb;

  --primary: #111111;
  --primary-hover: #000000;
  --primary-active: #262626;
  --secondary: #ffffff;
  --secondary-hover: #f5f5f5;
  --secondary-active: #ebebeb;

  --success: #1f1f1f;
  --warning: #2a2a2a;
  --danger: #000000;
  --info: #303030;

  --focus-ring: #000000;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.06);
  --shadow-md: 0 6px 18px rgba(0, 0, 0, 0.08);
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

[data-testid="stAppViewContainer"],
.main,
.block-container,
header[data-testid="stHeader"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    color: var(--text) !important;
}

.stCaptionContainer,
[data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
}

[data-baseweb="notification"],
[data-baseweb="notification"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text-strong);
    letter-spacing: -0.02em;
}

[data-testid="stSidebar"] {
    background: var(--surface-elevated);
    border-right: 1px solid var(--border);
    color: var(--text);
}

[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-sm);
    padding: 8px 10px;
}

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: var(--shadow-sm);
    padding: 16px;
}

.kpi-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 12px;
    margin-bottom: 8px;
    background: var(--bg-muted);
}

.status-good { border-left: 4px solid var(--success); }
.status-alert { border-left: 4px solid var(--danger); }

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
    background: var(--surface);
}

.stDownloadButton > button,
button[kind="primary"] {
    background: var(--primary);
    color: var(--text-inverse);
    border: 1px solid var(--primary);
    border-radius: 8px;
}

.stDownloadButton > button:hover,
button[kind="primary"]:hover {
    background: var(--primary-hover);
    border-color: var(--primary-hover);
}

.stDownloadButton > button,
.stDownloadButton > button * {
    color: var(--text-inverse) !important;
    fill: var(--text-inverse) !important;
}

.stDownloadButton [data-testid="stMarkdownContainer"],
.stDownloadButton [data-testid="stMarkdownContainer"] p,
.stDownloadButton [data-testid="stMarkdownContainer"] span {
    color: var(--text-inverse) !important;
}

.stDownloadButton > button:active,
button[kind="primary"]:active {
    background: var(--primary-active);
    border-color: var(--primary-active);
}

button:focus-visible {
    box-shadow: 0 0 0 2px var(--focus-ring);
}

code {
    font-family: 'IBM Plex Mono', monospace !important;
}
</style>
"""


@st.cache_resource
def load_models() -> dict:
    models = {}
    for name, path in MODEL_FILES.items():
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_training_summary() -> dict:
    if SUMMARY_FILE.exists():
        return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    return {"feature_columns": BASE_FEATURE_COLUMNS, "horizon_steps": 0, "calibrated_threshold": 0.5}


def get_current_metrics() -> dict:
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_percent": psutil.virtual_memory().percent,
        "process_count": len(psutil.pids()),
    }


def compare_models(
    models: dict,
    x_live: pd.DataFrame,
    calibrated_threshold: float,
) -> tuple[pd.DataFrame, float, int]:
    rows = []

    for model_name, model in models.items():
        if hasattr(model, "predict_proba"):
            spike_prob = float(model.predict_proba(x_live)[0][1])
            pred = int(spike_prob >= calibrated_threshold)
        else:
            pred = int(model.predict(x_live)[0])
            spike_prob = float(pred)

        rows.append(
            {
                "model": model_name,
                "prediction": "Spike" if pred == 1 else "Stable",
                "spike_probability": round(spike_prob, 4),
            }
        )

    comparison_df = pd.DataFrame(rows)
    if comparison_df.empty:
        return comparison_df, 0.0, 0

    ensemble_score = float(comparison_df["spike_probability"].mean())
    ensemble_pred = int(ensemble_score >= calibrated_threshold)
    return comparison_df, ensemble_score, ensemble_pred


def log_activity(message: str, level: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    st.session_state.activity.insert(0, {"time": now, "message": message, "level": level})
    st.session_state.activity = st.session_state.activity[:12]


st.set_page_config(page_title="PULSE", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown("# PULSE Control Deck")
st.caption("Monochrome monitoring UI with model comparison and ensemble confidence")
st.markdown("---")

st.sidebar.markdown("## Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)

st.sidebar.markdown("## Stress Test Quick Guide")
st.sidebar.code("stress-ng --cpu 4 --timeout 45s", language="bash")
st.sidebar.code("stress-ng --vm 2 --vm-bytes 50% --timeout 45s", language="bash")

if "live_history" not in st.session_state:
    st.session_state.live_history = pd.DataFrame(columns=["time", "cpu_percent", "ram_percent"])

if "activity" not in st.session_state:
    st.session_state.activity = []

if "last_alert_state" not in st.session_state:
    st.session_state.last_alert_state = None


st.caption("Live Decision Engine | In-place refresh with no full page rerender")

run_every = f"{refresh_interval}s" if auto_refresh else None


@st.fragment(run_every=run_every)
def live_dashboard() -> None:
    metrics = get_current_metrics()
    if "raw_live_history" not in st.session_state:
        st.session_state.raw_live_history = []
    st.session_state.raw_live_history.append(metrics)

    models = load_models()
    summary = load_training_summary()
    feature_columns = summary.get("feature_columns", FEATURE_COLUMNS)
    horizon_steps = int(summary.get("horizon_steps", 0))
    calibrated_threshold = float(summary.get("calibrated_threshold", 0.5))
    x_live = build_live_feature_frame(
        history_rows=st.session_state.raw_live_history,
        feature_columns=feature_columns,
        window_sizes=[3, 5],
    )

    if x_live is not None:
        comparison_df, ensemble_score, ensemble_pred = compare_models(
            models,
            x_live,
            calibrated_threshold,
        )
    else:
        comparison_df, ensemble_score, ensemble_pred = pd.DataFrame(), 0.0, 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU Usage", f"{metrics['cpu_percent']:.1f}%")
    with col2:
        st.metric("RAM Usage", f"{metrics['ram_percent']:.1f}%")
    with col3:
        st.metric("Process Count", int(metrics["process_count"]))

    st.markdown("---")
    st.subheader("Spike Prediction")

    if not models:
        st.warning("Models not found. Run training first.")
    elif x_live is None:
        st.info("Warming up temporal window for forecast features...")
    else:
        if ensemble_pred == 1:
            if horizon_steps > 0:
                st.markdown(
                    f'<div class="panel status-alert">⚠️ <b>PULSE Alert:</b> High chance of spike within next {horizon_steps} steps</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="panel status-alert">⚠️ <b>PULSE Alert:</b> High chance of spike</div>', unsafe_allow_html=True)
            if st.session_state.last_alert_state != "alert":
                log_activity("Forecast alert triggered by ensemble", "danger")
                st.session_state.last_alert_state = "alert"
        else:
            st.markdown('<div class="panel status-good">✅ <b>PULSE Status:</b> Near-future forecast stable</div>', unsafe_allow_html=True)
            if st.session_state.last_alert_state != "stable":
                log_activity("Near-future stable by ensemble decision", "success")
                st.session_state.last_alert_state = "stable"

        st.markdown("### Ensemble Confidence")
        st.progress(int(min(max(ensemble_score, 0.0), 1.0) * 100))
        if horizon_steps > 0:
            st.caption(
                f"Ensemble score: {ensemble_score:.2f} (threshold = {calibrated_threshold:.2f}), forecast horizon = {horizon_steps} steps"
            )
        else:
            st.caption(f"Ensemble score: {ensemble_score:.2f} (threshold = {calibrated_threshold:.2f})")

        st.subheader("Model Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        prob_list = ", ".join(f"{v:.2f}" for v in comparison_df["spike_probability"].tolist())
        st.info(
            "Ensemble process: average spike probability across all models. "
            f"mean([{prob_list}]) = {ensemble_score:.2f}. "
            f"Threshold {calibrated_threshold:.2f} -> {'Spike' if ensemble_pred == 1 else 'Stable'}."
        )

    st.markdown("---")
    left, right = st.columns([2, 1.1], gap="large")

    now = datetime.now().strftime("%H:%M:%S")
    new_row = pd.DataFrame(
        [{"time": now, "cpu_percent": metrics["cpu_percent"], "ram_percent": metrics["ram_percent"]}]
    )
    st.session_state.live_history = pd.concat([st.session_state.live_history, new_row], ignore_index=True).tail(120)

    history_df = st.session_state.live_history.copy()
    history_df_indexed = history_df.set_index("time")
    history_csv = history_df.to_csv(index=False).encode("utf-8")

    with left:
        st.subheader("Metrics History")
        st.line_chart(history_df_indexed[["cpu_percent", "ram_percent"]], use_container_width=True)

    with right:
        st.subheader("Recent Activity")
        st.download_button(
            "Download Metrics History (CSV)",
            data=history_csv,
            file_name="pulse_metrics_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if not st.session_state.activity:
            st.caption("No events yet.")
        else:
            for row in st.session_state.activity[:8]:
                icon = "✅" if row["level"] == "success" else "⚠️" if row["level"] == "danger" else "ℹ️"
                st.markdown(f"{icon} **{row['time']}**  {row['message']}")


live_dashboard()

st.markdown("---")
st.markdown(
    "**PULSE Project** | Data Collection -> Preprocessing -> Model Training -> Evaluation -> Prediction -> Streamlit UI"
)

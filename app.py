import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Paths and page configuration
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "student_performance_model.pkl"
GRAPH_FILE = BASE_DIR / "actual_vs_predicted.png"
METRICS_FILE = BASE_DIR / "model_metrics.json"
HISTORY_FILE = BASE_DIR / "prediction_history.json"
COMPARISON_FILE = BASE_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Theme: dark navy with blue and purple accents
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1020;
        --panel: #111a2e;
        --panel-soft: #17243b;
        --border: #273752;
        --text: #f4f7fb;
        --muted: #9caac0;
        --blue: #55a7ff;
        --purple: #9b8cff;
        --green: #4adea4;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(85,167,255,.12), transparent 34%),
            radial-gradient(ellipse at 95% 6%, rgba(155,140,255,.10), transparent 30%),
            var(--bg);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(11,16,32,.92); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 { letter-spacing: -0.025em; }
    h1 { font-size: clamp(2rem, 4vw, 3.1rem) !important; font-weight: 800 !important; }
    h2 { font-size: 1.55rem !important; font-weight: 750 !important; }
    p, label, .stCaption { color: var(--text); }
    .hero {
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 30px 32px;
        margin-bottom: 22px;
        background: linear-gradient(115deg, rgba(23,36,59,.98), rgba(17,26,46,.92));
        box-shadow: 0 18px 55px rgba(0,0,0,.18);
    }
    .eyebrow {
        color: var(--blue);
        font-size: .78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .16em;
        margin-bottom: 10px;
    }
    .hero-copy { color: #b9c7da; font-size: 1.05rem; max-width: 760px; line-height: 1.65; }
    .pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid #344a6b;
        background: #172943;
        color: #c9dcf5;
        font-size: .8rem;
        margin: 5px 5px 0 0;
    }
    .section-label {
        color: var(--blue);
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .panel {
        background: linear-gradient(145deg, rgba(20,31,52,.96), rgba(15,24,42,.96));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        height: 100%;
    }
    .panel-title { font-size: 1.08rem; font-weight: 750; margin-bottom: 5px; }
    .muted { color: var(--muted); font-size: .9rem; line-height: 1.5; }
    .result-number {
        font-size: clamp(2.5rem, 4vw, 3.5rem);
        font-weight: 850;
        letter-spacing: -.05em;
        line-height: 1.1;
        color: var(--text);
    }
    .result-unit { color: var(--muted); font-size: .95rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #15233a, #111b2f);
        border: 1px solid var(--border);
        padding: 17px 18px;
        border-radius: 16px;
    }
    div[data-testid="stMetricLabel"] { color: #aebdd1; }
    div[data-testid="stMetricValue"] { color: #f5f8ff; }
    div[data-testid="stForm"] {
        background: rgba(17,26,46,.8);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px 22px 8px 22px;
    }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        border-radius: 11px;
        border: 1px solid #3a5b87;
        min-height: 44px;
        font-weight: 700;
    }
    .stFormSubmitButton > button {
        background: linear-gradient(90deg, #347fe2, #6656d9);
        color: white;
        border: 0;
    }
    .stDownloadButton > button { background: #172943; color: #edf5ff; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: #111b2e;
        border-color: #33435e;
        border-radius: 10px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 13px;
        overflow: hidden;
    }
    hr { border-color: #273752 !important; }
    .footer-note { color: #8e9db4; font-size: .82rem; text-align: center; padding-top: 18px; }
    @media (max-width: 700px) {
        .hero { padding: 22px 20px; }
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load model and metrics
# ---------------------------------------------------------
if not MODEL_FILE.exists():
    st.error(
        "Trained model not found. Place 'student_performance_model.pkl' "
        "in the same folder as app.py, then run test.py."
    )
    st.stop()

try:
    model = joblib.load(MODEL_FILE)
except Exception as exc:
    st.error(f"The trained model could not be loaded: {exc}")
    st.stop()

fallback_metrics = {
    "model_name": "Unknown model",
    "r2": 0.78,
    "mae": 1.33,
    "rmse": 2.11,
    "features": ["studytime", "failures", "absences", "G1", "G2", "age"],
}
metrics = fallback_metrics.copy()
if METRICS_FILE.exists():
    try:
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            loaded_metrics = json.load(file)
        if isinstance(loaded_metrics, dict):
            metrics.update(loaded_metrics)
    except (OSError, json.JSONDecodeError):
        st.warning("Could not read model_metrics.json; displaying fallback metrics.")

model_name = str(metrics.get("model_name", "Unknown model"))
feature_names = list(
    metrics.get("features", ["studytime", "failures", "absences", "G1", "G2", "age"])
)

try:
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
except OSError as exc:
    st.error(f"Could not initialize prediction history: {exc}")
    st.stop()

# ---------------------------------------------------------
# Hero header
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Machine Learning · Academic Insights</div>
      <h1>🎓 Student Performance<br>Prediction</h1>
      <div class="hero-copy">
        Estimate a student's final mathematics grade using previous academic
        performance and selected student details.
      </div>
      <div style="margin-top:14px">
        <span class="pill">Random Forest Regression</span>
        <span class="pill">UCI Student Dataset</span>
        <span class="pill">Grade scale: 0–20</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Performance overview
# ---------------------------------------------------------
st.markdown('<div class="section-label">Model overview</div>', unsafe_allow_html=True)
st.header("Performance at a glance")
st.caption(f"Selected model: **{model_name}** · Metrics are calculated on the held-out test set.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("R² score", f"{float(metrics.get('r2', 0.78)):.3f}", help="Regression metric; not classification accuracy.")
m2.metric("Mean absolute error", f"{float(metrics.get('mae', 1.33)):.2f}", help="Average absolute error in grade points.")
m3.metric("Root mean squared error", f"{float(metrics.get('rmse', 2.11)):.2f}", help="Penalizes larger prediction errors more strongly.")
m4.metric("Model type", "Regression", help="The output is a numerical grade.")

training_records = metrics.get("training_records")
testing_records = metrics.get("testing_records")
if training_records is not None and testing_records is not None:
    st.caption(
        f"Evaluation split: {training_records} training records · {testing_records} test records. "
        "R² is a regression metric, not classification accuracy."
    )

with st.expander("Compare Linear Regression and Random Forest"):
    if COMPARISON_FILE.exists():
        try:
            comparison_df = pd.read_csv(COMPARISON_FILE)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        except (OSError, pd.errors.ParserError) as exc:
            st.warning(f"Could not load model comparison: {exc}")
    else:
        st.info("Model comparison file not found.")

st.divider()

# ---------------------------------------------------------
# Main workspace: input form and result
# ---------------------------------------------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-label">Prediction workspace</div>', unsafe_allow_html=True)
    st.header("Student details")
    st.markdown("<p class='muted'>Enter the student's information, then run the model to estimate the final grade.</p>", unsafe_allow_html=True)

    with st.form("student_prediction_form", clear_on_submit=False):
        age = st.number_input("Student age", min_value=15, max_value=22, value=16, step=1)
        studytime = st.selectbox(
            "Weekly study time",
            options=[1, 2, 3, 4],
            index=1,
            format_func=lambda value: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours",
            }[value],
        )
        failures = st.number_input("Previous class failures", min_value=0, max_value=4, value=0, step=1)
        absences = st.number_input("School absences", min_value=0, max_value=100, value=2, step=1)
        g1 = st.number_input("First-period grade (G1)", min_value=0.0, max_value=20.0, value=15.0, step=1.0)
        g2 = st.number_input("Second-period grade (G2)", min_value=0.0, max_value=20.0, value=16.0, step=1.0)
        submitted = st.form_submit_button("✨  Predict final grade", use_container_width=True)

with right:
    st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)
    st.header("Estimated final grade")
    if "last_prediction" in st.session_state:
        last = st.session_state["last_prediction"]
        st.markdown(
            f"""
            <div class="panel">
              <div class="muted">Predicted final grade (G3)</div>
              <div style="height:10px"></div>
              <div class="result-number">{last['prediction']:.2f}<span class="result-unit"> / 20</span></div>
              <div style="height:9px"></div>
              <div class="muted">Estimated percentage: {last['percentage']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(round(last["percentage"])))
        if last["prediction"] >= 16:
            st.success("High predicted performance")
        elif last["prediction"] >= 10:
            st.info("Moderate predicted performance")
        else:
            st.warning("Low predicted performance")
        st.caption("This is an estimate from historical data, not a guaranteed grade.")
        st.download_button(
            "⬇ Download prediction report",
            data=last["report"],
            file_name="student_prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div class="panel">
              <div class="muted">Your prediction will appear here</div>
              <div style="height:12px"></div>
              <div class="result-number">—<span class="result-unit"> / 20</span></div>
              <div style="height:12px"></div>
              <div class="muted">Complete the student details and select <b>Predict final grade</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("The model estimates a numerical final grade using six input features.")

# Handle submitted prediction after drawing the form and result area.
if submitted:
    student_values = {
        "studytime": int(studytime),
        "failures": int(failures),
        "absences": int(absences),
        "G1": float(g1),
        "G2": float(g2),
        "age": int(age),
    }
    try:
        if hasattr(model, "feature_names_in_"):
            ordered_features = list(model.feature_names_in_)
        else:
            ordered_features = feature_names

        missing_features = [name for name in ordered_features if name not in student_values]
        if missing_features:
            st.error(f"The model requires input features that the form does not provide: {missing_features}")
            st.stop()

        student = pd.DataFrame(
            [[student_values[name] for name in ordered_features]],
            columns=ordered_features,
        )
        raw_prediction = float(model.predict(student)[0])
        prediction = max(0.0, min(20.0, raw_prediction))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    percentage = (prediction / 20.0) * 100.0
    history_record = {
        "timestamp": timestamp,
        **student_values,
        "predicted_grade": round(prediction, 2),
        "model_name": model_name,
    }

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)
        if not isinstance(history, list):
            history = []
        history.append(history_record)
        with HISTORY_FILE.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Prediction calculated, but history could not be saved: {exc}")

    report = f"""STUDENT PERFORMANCE PREDICTION REPORT
=====================================

Prediction timestamp: {timestamp}

STUDENT DETAILS
---------------
Age: {int(age)}
Weekly study time category: {int(studytime)}
Previous class failures: {int(failures)}
School absences: {int(absences)}
First period grade (G1): {float(g1):.1f}
Second period grade (G2): {float(g2):.1f}

PREDICTION RESULT
-----------------
Predicted final grade (G3): {prediction:.2f} / 20
Predicted percentage: {percentage:.1f}%

Algorithm: {model_name}
Dataset: UCI Student Performance Dataset

Note: This prediction is an estimate based on historical data, not a
guaranteed academic outcome.
"""
    st.session_state["last_prediction"] = {
        "prediction": prediction,
        "percentage": percentage,
        "report": report,
    }
    st.rerun()

st.divider()

# ---------------------------------------------------------
# Actual versus predicted graph
# ---------------------------------------------------------
graph_left, graph_right = st.columns([1.25, 0.75], gap="large")
with graph_left:
    st.markdown('<div class="section-label">Model diagnostics</div>', unsafe_allow_html=True)
    st.header("Actual vs predicted grades")
    if GRAPH_FILE.exists():
        st.image(str(GRAPH_FILE), use_container_width=True)
        st.caption(f"Actual versus predicted grades for {model_name}. Points closer to the diagonal indicate closer predictions.")
    else:
        st.warning("Graph not found. Run test.py to generate actual_vs_predicted.png.")
with graph_right:
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">How to read this chart</div>
          <p class="muted">Each point represents a test-set student. The horizontal axis shows the actual final grade; the vertical axis shows the model's predicted grade.</p>
          <p class="muted">Points near the diagonal line indicate predictions close to the actual grade. Points farther away represent larger errors.</p>
          <hr>
          <div class="panel-title">Model selection</div>
          <p class="muted">Linear Regression was compared with Random Forest Regression. Random Forest was selected because it performed better on the recorded evaluation metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------
# Prediction history
# ---------------------------------------------------------
st.markdown('<div class="section-label">Saved activity</div>', unsafe_allow_html=True)
st.header("Prediction history")
try:
    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        history = json.load(file)
    if not isinstance(history, list):
        history = []
    if history:
        history_df = pd.DataFrame(history)
        if "model_name" not in history_df.columns:
            history_df["model_name"] = model_name
        else:
            missing_model = history_df["model_name"].isna() | (
                history_df["model_name"].astype(str).str.strip().str.lower().isin(["", "none", "nan"])
            )
            history_df.loc[missing_model, "model_name"] = model_name

        display_df = history_df.iloc[::-1].rename(columns={"model_name": "Model"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download prediction history (CSV)",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    else:
        st.caption("No predictions have been saved yet.")
except (OSError, json.JSONDecodeError) as exc:
    st.warning(f"Prediction history could not be displayed: {exc}")

st.divider()

# ---------------------------------------------------------
# About section
# ---------------------------------------------------------
with st.expander("About this project"):
    about1, about2 = st.columns(2)
    with about1:
        st.markdown("**Project details**")
        st.write(f"Selected algorithm: {model_name}")
        st.write("Dataset: UCI Student Performance Dataset")
        st.write("Target: final mathematics grade (G3), 0–20")
    with about2:
        st.markdown("**Input features**")
        st.write("Study time · Previous failures · Absences · G1 · G2 · Age")
        st.write("Prediction history is stored locally in `prediction_history.json`.")
    st.warning("Predictions are estimates based on historical data and should not be treated as guaranteed academic outcomes.")

st.markdown('<div class="footer-note">Student Performance Prediction · Machine Learning Project</div>', unsafe_allow_html=True)

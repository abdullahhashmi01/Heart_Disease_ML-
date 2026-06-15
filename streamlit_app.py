import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #24243e 100%);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(220,53,69,0.15) 0%, rgba(255,87,87,0.08) 100%);
    border: 1px solid rgba(220,53,69,0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6b6b, #ee5a24, #ff9ff3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    color: rgba(255,255,255,0.55);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* Section Headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #ff6b6b;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,107,107,0.25);
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255,107,107,0.3);
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #ff6b6b;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Predict Button */
.stButton > button {
    background: linear-gradient(135deg, #dc3545, #ee5a24) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(220,53,69,0.4) !important;
}

/* Result Cards */
.result-positive {
    background: linear-gradient(135deg, rgba(220,53,69,0.18), rgba(238,90,36,0.12));
    border: 1px solid rgba(220,53,69,0.5);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-negative {
    background: linear-gradient(135deg, rgba(46,213,115,0.15), rgba(0,200,150,0.10));
    border: 1px solid rgba(46,213,115,0.4);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.result-desc {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
}

/* Input fields */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.88rem !important;
}

/* Pipeline Status */
.pipeline-step {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-size: 0.9rem;
}

/* Info box */
.info-box {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #ff6b6b;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: rgba(255,255,255,0.65);
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper: Load Model & Preprocessor ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load saved model and preprocessor from artifacts directory."""
    model_path = Path("artifacts/model.pkl")
    preprocessor_path = Path("artifacts/preprocessor.pkl")

    model, preprocessor = None, None

    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    if preprocessor_path.exists():
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

    return model, preprocessor


# ─── Helper: Prediction ─────────────────────────────────────────────────────────
def predict(model, preprocessor, input_df):
    if preprocessor is not None:
        X = preprocessor.transform(input_df)
    else:
        X = input_df.values

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return prediction, proba


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🫀</div>
        <div style='font-family: Space Grotesk, sans-serif; font-weight:700;
                    font-size:1.1rem; color:#ff6b6b; margin-top:0.3rem;'>
            Heart Disease ML
        </div>
        <div style='color:rgba(255,255,255,0.35); font-size:0.75rem; margin-top:0.2rem;'>
            UCI Dataset · Random Forest
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🔮 Predict", "📊 Model Performance", "📂 Pipeline Info", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    # Artifact status
    st.markdown("<div style='font-size:0.78rem; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.7rem;'>Artifact Status</div>", unsafe_allow_html=True)

    artifacts = {
        "model.pkl": Path("artifacts/model.pkl").exists(),
        "preprocessor.pkl": Path("artifacts/preprocessor.pkl").exists(),
        "train.csv": Path("artifacts/train.csv").exists(),
        "test.csv": Path("artifacts/test.csv").exists(),
    }
    for name, exists in artifacts.items():
        icon = "🟢" if exists else "🔴"
        st.markdown(f"<div style='font-size:0.82rem; color:rgba(255,255,255,0.6); margin:0.2rem 0;'>{icon} {name}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div style='font-size:0.75rem; color:rgba(255,255,255,0.25); text-align:center;'>Built by Abdullah Hashmi</div>", unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🫀 Heart Disease Predictor</div>
    <div class='hero-subtitle'>
        End-to-End ML Pipeline &nbsp;·&nbsp; UCI Clinical Dataset &nbsp;·&nbsp; MLflow + DagsHub Tracking
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if "🔮 Predict" in page:

    model, preprocessor = load_artifacts()

    if model is None:
        st.warning("⚠️ Model not found. Please run `python app.py` first to generate `artifacts/model.pkl`.")
        st.code("python app.py", language="bash")
        st.stop()

    st.markdown("<div class='section-header'>Patient Clinical Attributes</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Fill in the patient's clinical data below. All fields are required for accurate prediction.</div>", unsafe_allow_html=True)

    # ── Input Form ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 20, 80, 50, help="Patient age in years")
        sex = st.selectbox("Sex", ["Male", "Female"])
        dataset = st.selectbox("Dataset Origin", ["Cleveland", "Hungarian", "Switzerland", "Long Beach VA"])

    with col2:
        st.markdown("**Symptoms & Tests**")
        cp = st.selectbox("Chest Pain Type", [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        restecg = st.selectbox("Resting ECG Results", [
            "normal",
            "lv hypertrophy",
            "st-t abnormality"
        ])

    with col3:
        st.markdown("**Stress Test Results**")
        thalch = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["False", "True"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak ST Segment", ["upsloping", "flat", "downsloping"])
        ca = st.slider("Number of Major Vessels (ca)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", [
            "normal",
            "fixed defect",
            "reversable defect"
        ])

    st.divider()

    # ── Predict Button ──────────────────────────────────────────────────────────
    if st.button("🔮 Predict Heart Disease Risk"):
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex.lower(),
            "dataset": dataset,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs == "True",
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang == "True",
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": float(ca),
            "thal": thal,
        }])

        with st.spinner("Analyzing clinical data..."):
            time.sleep(0.8)  # slight UX delay for dramatic effect
            try:
                prediction, proba = predict(model, preprocessor, input_data)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

        # ── Result Display ────────────────────────────────────────────────────
        col_r1, col_r2 = st.columns([2, 1])

        with col_r1:
            if prediction == 1:
                st.markdown("""
                <div class='result-positive'>
                    <div class='result-title' style='color:#ff6b6b;'>⚠️ Heart Disease Detected</div>
                    <div class='result-desc'>The model predicts a positive risk for heart disease based on the provided clinical attributes. Please consult a cardiologist for a full diagnostic evaluation.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-negative'>
                    <div class='result-title' style='color:#2ed573;'>✅ Low Risk Detected</div>
                    <div class='result-desc'>The model predicts a low likelihood of heart disease. Regular health check-ups and a healthy lifestyle are still recommended.</div>
                </div>
                """, unsafe_allow_html=True)

        with col_r2:
            if proba is not None:
                risk_pct = round(proba[1] * 100, 1)
                safe_pct = round(proba[0] * 100, 1)
                st.metric("Risk Probability", f"{risk_pct}%")
                st.metric("Safe Probability", f"{safe_pct}%")
                st.progress(proba[1])

        st.markdown("<div class='info-box'>⚕️ <b>Disclaimer:</b> This prediction is generated by a machine learning model trained on the UCI Heart Disease dataset. It is not a medical diagnosis. Always consult a qualified healthcare professional.</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "📊 Model Performance" in page:

    st.markdown("<div class='section-header'>Evaluation Summary</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='metric-grid'>
        <div class='metric-card'>
            <div class='metric-value'>0.9166</div>
            <div class='metric-label'>ROC AUC</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>85.9%</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>89.5%</div>
            <div class='metric-label'>Precision</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>86.2%</div>
            <div class='metric-label'>Recall</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>All Model Comparison</div>", unsafe_allow_html=True)

    results_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest 🏆", "Gradient Boosting", "XGBoost"],
        "Accuracy": [0.8424, 0.8152, 0.8587, 0.8261, 0.8478],
        "ROC AUC": [0.9040, 0.8179, 0.9166, 0.9024, 0.9117],
        "Precision": [0.8774, 0.8713, 0.8952, 0.8738, 0.8785],
        "Recall": [0.8532, 0.8073, 0.8624, 0.8257, 0.8624],
    })

    st.dataframe(
        results_df.style.highlight_max(
            subset=["Accuracy", "ROC AUC", "Precision", "Recall"],
            color="rgba(220,53,69,0.3)"
        ).format({
            "Accuracy": "{:.4f}",
            "ROC AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='section-header'>Best Model: Random Forest Classifier</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='info-box'><b>Optimal Hyperparameters (GridSearchCV 5-Fold)</b><br><br>🔧 max_depth: 10<br>🔧 min_samples_split: 5<br>🔧 n_estimators: 100</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='info-box'><b>Cross-Validation Strategy</b><br><br>📐 5-Fold Stratified CV<br>🎯 Optimization Metric: ROC AUC<br>🧪 Models Compared: 5</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>ROC AUC Comparison</div>", unsafe_allow_html=True)
    st.bar_chart(
        results_df.set_index("Model")["ROC AUC"],
        color="#dc3545"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif "📂 Pipeline Info" in page:

    st.markdown("<div class='section-header'>End-to-End Pipeline Workflow</div>", unsafe_allow_html=True)

    steps = [
        ("1", "Data Ingestion", "data_ingestion.py", "Loads raw UCI dataset → saves raw.csv → 80/20 stratified split → train.csv + test.csv", "✅"),
        ("2", "Data Transformation", "data_transformation.py", "Numerical: median imputation + standard scaling | Categorical: mode imputation + one-hot encoding → preprocessor.pkl", "✅"),
        ("3", "Model Training", "model_trainer.py", "5-Fold GridSearchCV for 5 models → best model selected → logged to MLflow/DagsHub → model.pkl", "✅"),
    ]

    for num, name, file, desc, status in steps:
        st.markdown(f"""
        <div class='pipeline-step'>
            <div style='background:rgba(220,53,69,0.2); border-radius:50%; width:32px; height:32px;
                        display:flex; align-items:center; justify-content:center;
                        font-family: Space Grotesk; font-weight:700; color:#ff6b6b; flex-shrink:0;'>{num}</div>
            <div>
                <div style='font-weight:600; color:#fff; margin-bottom:0.2rem;'>{name} &nbsp;
                    <span style='color:rgba(255,255,255,0.3); font-size:0.8rem; font-family:monospace;'>{file}</span>
                    <span style='margin-left:0.5rem;'>{status}</span>
                </div>
                <div style='color:rgba(255,255,255,0.5); font-size:0.83rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Feature Engineering</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='info-box'><b>Numerical Features</b><br><br>age · trestbps · chol · thalch · oldpeak · ca<br><br>→ Median Imputation<br>→ Standard Scaling</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='info-box'><b>Categorical Features</b><br><br>sex · dataset · cp · fbs · restecg · exang · slope · thal<br><br>→ Mode Imputation<br>→ One-Hot Encoding (handle_unknown=ignore)<br>→ Standard Scaling</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Run the Pipeline</div>", unsafe_allow_html=True)
    st.code("""# 1. Clone the repository
git clone https://github.com/abdullahhashmi01/Heart_Disease_ML-
cd Heart_Disease_ML-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set MLflow / DagsHub credentials
export MLFLOW_TRACKING_USERNAME="abdullahhashmi01"
export MLFLOW_TRACKING_PASSWORD="your_dagshub_token_here"

# 4. Run the full pipeline
python app.py

# 5. Launch this Streamlit app
streamlit run streamlit_app.py
""", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:

    st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    This is a production-grade End-to-End Machine Learning project that predicts the likelihood of clinical heart disease using clinical attributes from the <b>UCI Heart Disease Dataset</b>.
    <br><br>
    The pipeline implements clean software engineering patterns with modular components, structured feature engineering, advanced hyperparameter tuning via cross-validation grids, and automated experiment tracking via <b>MLflow + DagsHub</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Tech Stack</div>", unsafe_allow_html=True)

    techs = [
        ("🐍", "Python", "Core language"),
        ("🤖", "Scikit-learn", "ML models & pipelines"),
        ("⚡", "XGBoost", "Gradient boosting"),
        ("📈", "MLflow", "Experiment tracking"),
        ("🐙", "DagsHub", "Remote MLflow backend"),
        ("🌀", "Streamlit", "This web interface"),
        ("🐼", "Pandas / NumPy", "Data processing"),
    ]

    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(techs):
        with cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom:1rem;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-weight:600; color:#fff; margin-top:0.4rem; font-size:0.9rem;'>{name}</div>
                <div class='metric-label'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Author</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        👤 <b>Abdullah Hashmi</b><br>
        📧 theabdullahhashmi90@gmail.com<br>
        🔗 <a href='https://github.com/abdullahhashmi01' style='color:#ff6b6b;'>github.com/abdullahhashmi01</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='info-box'>⚕️"
    " <b>Medical Disclaimer:</b> This application is built for educational and research purposes only. Predictions made by this model should not be used" \
    " as a substitute for professional medical advice, diagnosis, or treatment.</div>", unsafe_allow_html=True)


from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"

st.set_page_config(
    page_title="MediCost AI | Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b27 40%, #0f1922 100%);
        min-height: 100vh;
    }

    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00d4ff22, #7c3aed22);
        border: 1px solid #7c3aed55;
        border-radius: 999px;
        padding: 0.3rem 1.1rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: #a78bfa;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e2e8f0, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
        margin: 0 0 0.6rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 560px;
        margin: 0 auto 1rem;
        font-weight: 400;
        line-height: 1.6;
    }

    .hero-note {
        color: #7c8aa5;
        font-size: 0.86rem;
        max-width: 600px;
        margin: 0 auto 2rem;
        line-height: 1.5;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 20px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }

    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #7c3aed;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(255, 255, 255, 0.13) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #38bdf8) !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        padding: 0.85rem 2rem;
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 24px rgba(124, 58, 237, 0.4);
        margin-top: 0.5rem;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.55);
    }

    .result-box {
        background: linear-gradient(135deg, #7c3aed18, #38bdf818);
        border: 1px solid #7c3aed55;
        border-radius: 20px;
        padding: 2.2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-label {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.4rem;
    }

    .result-amount {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }

    .result-note {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.75rem;
    }

    .info-grid {
        display: flex;
        gap: 1rem;
        margin-top: 1.2rem;
        flex-wrap: wrap;
    }

    .info-tile {
        flex: 1;
        min-width: 120px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }

    .info-tile .tile-val {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .info-tile .tile-key {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .bmi-normal { color: #4ade80; }
    .bmi-overweight { color: #fbbf24; }
    .bmi-obese { color: #f87171; }
    .bmi-under { color: #60a5fa; }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #4b5a73;
        font-size: 0.78rem;
    }

    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def bmi_category(bmi: float) -> tuple[str, str]:
    """Return the BMI category label and matching CSS class."""
    if bmi < 18.5:
        return "Underweight", "bmi-under"
    if bmi < 25.0:
        return "Normal Weight", "bmi-normal"
    if bmi < 30.0:
        return "Overweight", "bmi-overweight"
    return "Obese", "bmi-obese"


def format_region(value: str) -> str:
    """Format dataset region codes into reader-friendly labels."""
    labels = {
        "northeast": "Northeast",
        "northwest": "Northwest",
        "southeast": "Southeast",
        "southwest": "Southwest",
    }
    return labels.get(value, value.title())


def build_input_frame(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
) -> pd.DataFrame:
    """Build a single-row dataframe in the same shape used for training."""
    input_df = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )
    return input_df[EXPECTED_COLUMNS]


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained pipeline or stop the app with a friendly message."""
    if not MODEL_PATH.exists():
        st.error(
            "The trained model file `model.pkl` was not found. Run `python main.py` "
            "to generate it, then restart the app.",
            icon="🚨",
        )
        st.stop()

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(
            "The model file could not be loaded. It may be corrupted or created "
            "with incompatible package versions.",
            icon="🚨",
        )
        st.caption(f"Technical details: {exc}")
        st.stop()


model = load_model()

st.markdown(
    """
    <div class="hero-header">
        <div class="hero-badge">AI-Powered Estimate</div>
        <div class="hero-title">MediCost AI</div>
        <div class="hero-subtitle">
            Estimate annual medical insurance charges from the same inputs used
            to train the model.
        </div>
        <div class="hero-note">
            This tool is intended for demonstration and learning. Predictions are
            generated from a pre-trained machine learning pipeline and should not
            be treated as professional financial or medical advice.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-label">👤 Personal Information</div>',
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    age = st.slider(
        "Age (years)",
        min_value=18,
        max_value=64,
        value=30,
        step=1,
        help="Age range supported by the training dataset.",
    )
with col2:
    sex = st.selectbox(
        "Biological Sex",
        options=["male", "female"],
        format_func=lambda value: "Male" if value == "male" else "Female",
        help="Category used by the original dataset.",
    )

children = st.select_slider(
    "Number of Dependents Covered",
    options=[0, 1, 2, 3, 4, 5],
    value=0,
    help="Number of children or dependents covered by the plan.",
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-label">❤️ Health Metrics</div>',
    unsafe_allow_html=True,
)

bmi = st.slider(
    "Body Mass Index (BMI)",
    min_value=10.0,
    max_value=55.0,
    value=25.0,
    step=0.1,
    help="Healthy BMI is often considered to be between 18.5 and 24.9.",
)

cat_label, cat_css = bmi_category(bmi)
st.markdown(
    f'<p style="font-size:0.83rem;margin-top:-0.3rem;">Weight Category: '
    f'<span class="{cat_css}"><b>{cat_label}</b></span></p>',
    unsafe_allow_html=True,
)

smoker = st.radio(
    "Smoking Status",
    options=["no", "yes"],
    format_func=lambda value: "Non-smoker" if value == "no" else "Smoker",
    horizontal=True,
    help="Smoking status from the original dataset.",
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-label">📍 Geographic Region (US)</div>',
    unsafe_allow_html=True,
)

region = st.selectbox(
    "Residential Region",
    options=["northeast", "northwest", "southeast", "southwest"],
    format_func=format_region,
    help="Region category used by the training dataset.",
)
st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("Predict Insurance Cost", use_container_width=True)

if predict_btn:
    input_df = build_input_frame(age, sex, bmi, children, smoker, region)

    try:
        with st.spinner("Generating estimate..."):
            prediction = float(model.predict(input_df)[0])
    except Exception as exc:
        st.error(
            "Prediction failed. Please confirm the saved model matches the app's "
            "expected input fields and package versions.",
            icon="🚨",
        )
        st.caption(f"Technical details: {exc}")
    else:
        if prediction < 0:
            st.warning(
                "The model returned a negative estimate, which is not a valid "
                "insurance charge. Retraining or output constraints may be needed."
            )

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Estimated Annual Insurance Cost</div>
                <div class="result-amount">${max(prediction, 0.0):,.2f}</div>
                <div class="result-note">
                    Generated by the saved machine learning pipeline in
                    <code>model.pkl</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-grid">
                <div class="info-tile">
                    <div class="tile-val">{age}</div>
                    <div class="tile-key">Age</div>
                </div>
                <div class="info-tile">
                    <div class="tile-val">{sex.title()}</div>
                    <div class="tile-key">Sex</div>
                </div>
                <div class="info-tile">
                    <div class="tile-val">{bmi:.1f}</div>
                    <div class="tile-key">BMI | {cat_label}</div>
                </div>
                <div class="info-tile">
                    <div class="tile-val">{children}</div>
                    <div class="tile-key">Dependents</div>
                </div>
                <div class="info-tile">
                    <div class="tile-val">{"Yes" if smoker == "yes" else "No"}</div>
                    <div class="tile-key">Smoker</div>
                </div>
                <div class="info-tile">
                    <div class="tile-val">{format_region(region)}</div>
                    <div class="tile-key">Region</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">MediCost AI | Built with Streamlit and scikit-learn | '
    'Educational use only.</div>',
    unsafe_allow_html=True,
)

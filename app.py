
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# LOAD MODEL + DATA
# =========================================================
MODEL_PATH = "best_model.joblib"
DATA_PATH = "StudentPerformanceFactors.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    model = load_model()
    df = load_data()
except Exception as e:
    st.error(
        "Model ya dataset load nahi ho raha. "
        "Ensure karo ki best_model.joblib aur StudentPerformanceFactors.csv "
        "app.py ke same folder mein hain."
    )
    st.stop()

# =========================================================
# CSS — MODERN 3D / GLASS / ANIMATED UI
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #070d1d;
    --panel: rgba(16, 27, 52, 0.78);
    --panel2: rgba(11, 20, 39, 0.88);
    --border: rgba(255,255,255,0.10);
    --text: #f5f7ff;
    --muted: #91a4c7;
    --blue: #4f7cff;
    --purple: #8b5cf6;
    --cyan: #22d3ee;
    --green: #22c55e;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(79,124,255,.18), transparent 28%),
        radial-gradient(circle at 92% 18%, rgba(34,211,238,.12), transparent 25%),
        radial-gradient(circle at 50% 95%, rgba(139,92,246,.13), transparent 30%),
        #070d1d;
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {
    max-width: 1240px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

/* Animated ambient orbs */
.stApp::before,
.stApp::after {
    content: "";
    position: fixed;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    filter: blur(70px);
    opacity: .16;
    pointer-events: none;
    z-index: 0;
    animation: floatOrb 9s ease-in-out infinite alternate;
}
.stApp::before {
    background: #4f7cff;
    left: -100px;
    top: 20%;
}
.stApp::after {
    background: #22d3ee;
    right: -100px;
    top: 55%;
    animation-delay: 2s;
}
@keyframes floatOrb {
    from { transform: translate3d(0,0,0) scale(1); }
    to { transform: translate3d(35px,-30px,0) scale(1.15); }
}

/* HERO */
.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 46px;
    border-radius: 30px;
    margin-bottom: 22px;
    background:
        linear-gradient(135deg, rgba(46,88,220,.96), rgba(105,52,238,.96));
    border: 1px solid rgba(255,255,255,.20);
    box-shadow:
        0 25px 70px rgba(34,66,180,.30),
        inset 0 1px 0 rgba(255,255,255,.20);
    transform: perspective(1100px) rotateX(0deg);
}
.hero::before {
    content: "";
    position: absolute;
    width: 310px;
    height: 310px;
    right: -80px;
    top: -150px;
    border-radius: 50%;
    background: rgba(255,255,255,.12);
    box-shadow: 0 0 80px rgba(255,255,255,.08);
    animation: heroPulse 6s ease-in-out infinite alternate;
}
.hero::after {
    content: "";
    position: absolute;
    width: 170px;
    height: 170px;
    left: 45%;
    bottom: -120px;
    border-radius: 50%;
    background: rgba(34,211,238,.15);
}
@keyframes heroPulse {
    to { transform: translate(-20px, 20px) scale(1.12); }
}
.hero-content { position: relative; z-index: 2; }
.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.20);
    color: #dce7ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.1px;
    margin-bottom: 14px;
}
.hero h1 {
    font-size: clamp(32px, 4vw, 54px);
    line-height: 1.04;
    margin: 0;
    color: white;
    font-weight: 800;
    letter-spacing: -1.8px;
}
.hero p {
    max-width: 720px;
    margin: 16px 0 0;
    color: #dbe6ff;
    font-size: 16px;
    line-height: 1.7;
}

/* INFO BAR */
.info-bar {
    padding: 18px 22px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: rgba(12,22,43,.70);
    box-shadow: 0 12px 35px rgba(0,0,0,.15);
    margin-bottom: 28px;
    color: #b9c9e5;
}
.info-bar strong { color: white; }

/* SECTION */
.section-title {
    margin: 30px 0 4px;
    font-size: 25px;
    font-weight: 800;
    color: #f7f9ff;
}
.section-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 18px;
}

/* INPUT AREA */

/* Streamlit widgets */
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: rgba(9,18,36,.86) !important;
    border-color: rgba(255,255,255,.10) !important;
    border-radius: 12px !important;
}
label, .stNumberInput label, .stSelectbox label {
    color: #cbd8ef !important;
    font-weight: 600 !important;
}
div[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,.06) !important;
    color: #d9e3f5 !important;
}

/* BUTTON */
div.stButton > button {
    width: 100%;
    min-height: 58px;
    border: 0 !important;
    border-radius: 17px !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: .2px;
    background: linear-gradient(100deg, #3b6ff5, #7c3aed) !important;
    box-shadow:
        0 12px 35px rgba(79,124,255,.30),
        inset 0 1px 0 rgba(255,255,255,.18);
    transition: all .22s ease;
}
div.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 18px 45px rgba(79,124,255,.42);
}
div.stButton > button:active {
    transform: translateY(0) scale(.99);
}

/* RESULT */
.result-card {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 34px 25px 30px;
    margin: 25px 0 30px;
    border-radius: 28px;
    background:
        radial-gradient(circle at 50% -20%, rgba(34,211,238,.22), transparent 45%),
        linear-gradient(145deg, rgba(14,39,70,.96), rgba(8,18,38,.98));
    border: 1px solid rgba(34,211,238,.22);
    box-shadow:
        0 25px 70px rgba(0,0,0,.28),
        inset 0 1px 0 rgba(255,255,255,.06);
}
.result-label {
    color: #8ea6ce;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}
.result-score {
    font-size: 64px;
    line-height: 1;
    font-weight: 800;
    margin: 12px 0 8px;
    color: white;
    text-shadow: 0 0 30px rgba(79,124,255,.25);
}
.result-score span {
    font-size: 22px;
    color: #91a4c7;
    font-weight: 600;
}
.result-status {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(34,211,238,.09);
    border: 1px solid rgba(34,211,238,.18);
    color: #a9e9f4;
    font-size: 13px;
    font-weight: 700;
}

/* METRIC CARDS */
.metric-card {
    min-height: 150px;
    padding: 24px;
    border-radius: 22px;
    border: 1px solid var(--border);
    background: linear-gradient(145deg, rgba(18,31,57,.90), rgba(8,17,34,.95));
    box-shadow: 0 16px 40px rgba(0,0,0,.18);
    text-align: center;
}
.metric-name {
    color: #91a7cb;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
.metric-value {
    color: white;
    font-size: 31px;
    font-weight: 800;
    margin-top: 10px;
}
.metric-info {
    color: #6f83a8;
    font-size: 12px;
    margin-top: 6px;
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 30px 0 10px;
    color: #566b91;
    font-size: 12px;
}

/* Mobile */
@media (max-width: 800px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .hero { padding: 30px 25px; }
    .result-score { font-size: 50px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
<div class="hero">
  <div class="hero-content">
    <div class="badge">✦ MACHINE LEARNING • PREDICTIVE ANALYTICS</div>
    <h1>🎓 Student Performance<br>Predictor</h1>
    <p>
      Predict a student's expected examination score using academic,
      personal, learning and environmental factors.
    </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-bar">
  <strong>How it works:</strong>
  Enter the student's information below → the trained Machine Learning
  model processes the inputs → the app estimates the expected exam score.
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# INPUTS
# =========================================================
st.markdown('<div class="section-title">🧑‍🎓 Student Profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Provide the student details carefully for a more reliable prediction.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    hours = st.number_input(
        "📚 Hours Studied (per week)",
        min_value=1, max_value=44, value=10, step=1,
        help="Approximate weekly study hours."
    )

    attendance = st.number_input(
        "📅 Attendance (%)",
        min_value=60, max_value=100, value=80, step=1
    )

    parental_involvement = st.selectbox(
        "👨‍👩‍👧 Parental Involvement",
        ["Low", "Medium", "High"]
    )

    resources = st.selectbox(
        "📖 Access to Resources",
        ["Low", "Medium", "High"]
    )

    extracurricular = st.selectbox(
        "⚽ Extracurricular Activities",
        ["No", "Yes"]
    )

    sleep = st.number_input(
        "😴 Sleep Hours",
        min_value=4, max_value=10, value=7, step=1
    )

    previous_scores = st.number_input(
        "📈 Previous Scores",
        min_value=50, max_value=100, value=70, step=1
    )

    motivation = st.selectbox(
        "🔥 Motivation Level",
        ["Low", "Medium", "High"]
    )

    internet = st.selectbox(
        "🌐 Internet Access",
        ["No", "Yes"]
    )

    tutoring = st.number_input(
        "👨‍🏫 Tutoring Sessions",
        min_value=0, max_value=8, value=2, step=1
    )


with right:
    family_income = st.selectbox(
        "💰 Family Income",
        ["Low", "Medium", "High"]
    )

    teacher_quality = st.selectbox(
        "🏫 Teacher Quality",
        ["Low", "Medium", "High"]
    )

    school_type = st.selectbox(
        "🏢 School Type",
        ["Public", "Private"]
    )

    peer_influence = st.selectbox(
        "👥 Peer Influence",
        ["Negative", "Neutral", "Positive"]
    )

    physical_activity = st.number_input(
        "🏃 Physical Activity (hours/week)",
        min_value=0, max_value=6, value=3, step=1
    )

    learning_disabilities = st.selectbox(
        "🧠 Learning Disabilities",
        ["No", "Yes"]
    )

    parental_education = st.selectbox(
        "🎓 Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )

    distance = st.selectbox(
        "📍 Distance From Home",
        ["Near", "Moderate", "Far"]
    )

    gender = st.selectbox(
        "👤 Gender",
        ["Male", "Female"]
    )

    st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PREDICTION
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

predict_clicked = st.button("✨  PREDICT EXAM SCORE")

if predict_clicked:
    input_data = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation,
        "Internet_Access": internet,
        "Tutoring_Sessions": tutoring,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance,
        "Gender": gender,
    }])

    try:
        prediction = float(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Keep displayed score within the normal exam scale.
    prediction = float(np.clip(prediction, 0, 100))

    if prediction < 60:
        status = "Needs Improvement"
        status_icon = "📘"
    elif prediction < 75:
        status = "Average Performance"
        status_icon = "📊"
    elif prediction < 90:
        status = "Good Performance"
        status_icon = "🚀"
    else:
        status = "Excellent Performance"
        status_icon = "🏆"

    st.markdown(
        f"""
<div class="result-card">
  <div class="result-label">PREDICTED EXAM SCORE</div>
  <div class="result-score">{prediction:.2f}<span> / 100</span></div>
  <div class="result-status">{status_icon} {status}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # =====================================================
    # SCORE ANALYSIS — PROFESSIONAL INTERACTIVE CHARTS
    # =====================================================
    st.markdown(
        '<div class="section-title">📊 Score Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">Interactive visual analysis of the predicted academic performance.</div>',
        unsafe_allow_html=True,
    )

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                number={
                    "suffix": " / 100",
                    "font": {"size": 38, "color": "#ffffff"}
                },
                title={
                    "text": "Performance Score",
                    "font": {"size": 18, "color": "#c8d6ee"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#6f83a8",
                        "tickfont": {"color": "#91a4c7"}
                    },
                    "bar": {"color": "#4f7cff", "thickness": 0.30},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 60], "color": "rgba(239,68,68,.16)"},
                        {"range": [60, 75], "color": "rgba(234,179,8,.16)"},
                        {"range": [75, 90], "color": "rgba(34,211,238,.13)"},
                        {"range": [90, 100], "color": "rgba(34,197,94,.13)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 3},
                        "thickness": 0.8,
                        "value": prediction
                    }
                }
            )
        )
        gauge.update_layout(
            height=360,
            margin=dict(l=25, r=25, t=70, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    with chart_right:
        benchmark = pd.DataFrame({
            "Category": [
                "Needs Improvement",
                "Average",
                "Good",
                "Excellent",
                "Prediction"
            ],
            "Score": [55, 67.5, 82.5, 95, prediction]
        })

        fig = px.bar(
            benchmark,
            x="Category",
            y="Score",
            text="Score",
        )
        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
            marker_color=["#ef4444", "#eab308", "#22d3ee", "#22c55e", "#7c3aed"],
            cliponaxis=False,
        )
        fig.update_layout(
            height=360,
            margin=dict(l=25, r=25, t=70, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title={
                "text": "Performance Benchmark",
                "x": 0.5,
                "font": {"size": 18, "color": "#c8d6ee"}
            },
            xaxis={
                "title": "",
                "tickfont": {"color": "#91a4c7"},
                "gridcolor": "rgba(255,255,255,0)"
            },
            yaxis={
                "title": "Score",
                "range": [0, 105],
                "tickfont": {"color": "#91a4c7"},
                "title_font": {"color": "#91a4c7"},
                "gridcolor": "rgba(255,255,255,.06)"
            },
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # =====================================================
    # INPUT PROFILE
    # =====================================================
    st.markdown(
        '<div class="section-title">🧩 Prediction Profile</div>',
        unsafe_allow_html=True,
    )

    profile = pd.DataFrame({
        "Factor": [
            "Study Hours", "Attendance", "Previous Score",
            "Sleep", "Tutoring", "Physical Activity"
        ],
        "Value": [
            hours, attendance, previous_scores,
            sleep, tutoring, physical_activity
        ]
    })

    profile_fig = px.bar(
        profile,
        x="Value",
        y="Factor",
        orientation="h",
        text="Value",
    )
    profile_fig.update_traces(
        marker_color="#4f7cff",
        textposition="outside",
        cliponaxis=False,
    )
    profile_fig.update_layout(
        height=330,
        margin=dict(l=30, r=40, t=30, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "gridcolor": "rgba(255,255,255,.06)",
            "tickfont": {"color": "#91a4c7"}
        },
        yaxis={
            "tickfont": {"color": "#c8d6ee"}
        },
        showlegend=False,
    )
    st.plotly_chart(profile_fig, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# MODEL PERFORMANCE
# =========================================================
st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Evaluation metrics from the trained regression models used in this project.</div>',
    unsafe_allow_html=True,
)

# Current project results
linear_r2 = 0.7696
linear_mae = 0.45
linear_rmse = 1.80

tree_r2 = 0.5444

m1, m2, m3 = st.columns(3, gap="large")

with m1:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-name">R² Score</div>
  <div class="metric-value">{linear_r2*100:.2f}%</div>
  <div class="metric-info">Goodness of Fit</div>
</div>
""",
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-name">MAE</div>
  <div class="metric-value">{linear_mae:.2f}</div>
  <div class="metric-info">Mean Absolute Error</div>
</div>
""",
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-name">RMSE</div>
  <div class="metric-value">{linear_rmse:.2f}</div>
  <div class="metric-info">Root Mean Squared Error</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

compare = pd.DataFrame({
    "Model": ["Decision Tree", "Linear Regression"],
    "R² Score (%)": [tree_r2 * 100, linear_r2 * 100]
})

comparison_fig = px.bar(
    compare,
    x="R² Score (%)",
    y="Model",
    orientation="h",
    text="R² Score (%)",
)
comparison_fig.update_traces(
    marker_color=["#22d3ee", "#7c3aed"],
    texttemplate="%{text:.2f}%",
    textposition="outside",
    cliponaxis=False,
)
comparison_fig.update_layout(
    height=330,
    margin=dict(l=30, r=60, t=45, b=35),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title={
        "text": "Model Comparison",
        "x": 0.5,
        "font": {"size": 19, "color": "#f5f7ff"}
    },
    xaxis={
        "range": [0, 100],
        "gridcolor": "rgba(255,255,255,.06)",
        "tickfont": {"color": "#91a4c7"}
    },
    yaxis={"tickfont": {"color": "#c8d6ee"}},
    showlegend=False,
)
st.plotly_chart(comparison_fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    """
<div class="info-bar" style="text-align:center;">
  🏆 <strong>Best Model: Linear Regression</strong>
  &nbsp; • &nbsp;
  R² Score: <strong>76.96%</strong>
  &nbsp; • &nbsp;
  Compared with: <strong>Decision Tree</strong>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="footer">
  © 2026 Tanmoy Majumder &nbsp;|&nbsp;
  Student Performance Prediction &nbsp;|&nbsp;
  Machine Learning Project &nbsp;|&nbsp;
  Built with Python + Scikit-learn + Streamlit
</div>
""",
    unsafe_allow_html=True,
)

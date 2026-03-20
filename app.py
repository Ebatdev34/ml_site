# ================================================================
#  ML UNIVERSE  —  by Ebatdev
#  Full ML Pipeline · Premium Dark UI
#  Data Upload · Explorationi · Training · Results · Export
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, r2_score,
                              mean_absolute_error, mean_squared_error)

st.set_page_config(
    page_title="ML Universe",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
#  STYLE  —  Deep Space · Precision Instrument
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #060610 !important;
    color: #c4d4e8;
    font-family: 'DM Mono', monospace;
}

/* Subtle dot grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(rgba(100,160,255,0.06) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #04040e !important;
    border-right: 1px solid rgba(100,160,255,0.08) !important;
}
section[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace !important; }

.block-container { padding: 2rem 2.5rem 4rem !important; position: relative; z-index: 1; }

h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

/* Logo */
.ml-logo {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 6px;
    color: #64a0ff;
    text-transform: uppercase;
    text-align: center;
    padding: 20px 0 4px;
    text-shadow: 0 0 20px rgba(100,160,255,0.4);
}
.ml-tagline {
    font-size: 10px;
    letter-spacing: 2px;
    color: rgba(196,212,232,0.25);
    text-align: center;
    margin-bottom: 24px;
    text-transform: uppercase;
}

/* Page header */
.page-header {
    border-bottom: 1px solid rgba(100,160,255,0.12);
    padding-bottom: 14px;
    margin-bottom: 28px;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: #e8f0fc;
    margin: 0;
    line-height: 1.1;
}
.page-sub {
    font-size: 11px;
    color: rgba(196,212,232,0.35);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Step indicator */
.step-row {
    display: flex;
    gap: 6px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid rgba(100,160,255,0.12);
    color: rgba(196,212,232,0.3);
}
.step-active {
    background: rgba(100,160,255,0.08);
    border-color: rgba(100,160,255,0.35);
    color: #64a0ff;
}
.step-done {
    background: rgba(0,220,130,0.06);
    border-color: rgba(0,220,130,0.2);
    color: #00dc82;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.card-blue  { border-left: 3px solid #64a0ff; }
.card-green { border-left: 3px solid #00dc82; }
.card-pink  { border-left: 3px solid #ff64a0; }
.card-gold  { border-left: 3px solid #ffcc64; }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #64a0ff;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 10px;
    letter-spacing: 2px;
    color: rgba(196,212,232,0.3);
    text-transform: uppercase;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-blue  { background: rgba(100,160,255,0.1); color: #64a0ff; border: 1px solid rgba(100,160,255,0.2); }
.badge-green { background: rgba(0,220,130,0.1);  color: #00dc82; border: 1px solid rgba(0,220,130,0.2); }
.badge-pink  { background: rgba(255,100,160,0.1); color: #ff64a0; border: 1px solid rgba(255,100,160,0.2); }
.badge-gold  { background: rgba(255,204,100,0.1); color: #ffcc64; border: 1px solid rgba(255,204,100,0.2); }

/* Table */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* Progress */
.prog-wrap {
    background: rgba(255,255,255,0.04);
    border-radius: 3px;
    height: 3px;
    margin: 10px 0;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    background: linear-gradient(90deg, #64a0ff, #00dc82);
    border-radius: 3px;
    transition: width 0.4s;
}

/* Info / Success / Warning boxes */
.info-box {
    background: rgba(100,160,255,0.05);
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #a0c4ff;
    margin: 8px 0;
}
.success-box {
    background: rgba(0,220,130,0.05);
    border: 1px solid rgba(0,220,130,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #00dc82;
    margin: 8px 0;
}
.warn-box {
    background: rgba(255,204,100,0.05);
    border: 1px solid rgba(255,204,100,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #ffcc64;
    margin: 8px 0;
}
.error-box {
    background: rgba(255,100,100,0.05);
    border: 1px solid rgba(255,100,100,0.15);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #ff6464;
    margin: 8px 0;
}

/* Best model highlight */
.best-model {
    background: linear-gradient(135deg, rgba(100,160,255,0.06), rgba(0,220,130,0.06));
    border: 1px solid rgba(0,220,130,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Section label */
.section-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(196,212,232,0.25);
    margin: 20px 0 12px;
}

/* Streamlit widget overrides */
.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #c4d4e8 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: rgba(100,160,255,0.4) !important;
    color: #64a0ff !important;
    background: rgba(100,160,255,0.06) !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: #c4d4e8 !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(100,160,255,0.3) !important;
    box-shadow: 0 0 0 1px rgba(100,160,255,0.1) !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: #c4d4e8 !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: rgba(100,160,255,0.3) !important;
}

label { color: rgba(196,212,232,0.45) !important; font-size: 11px !important; letter-spacing: 1px !important; }
.stCaption { color: rgba(196,212,232,0.3) !important; font-size: 11px !important; }
hr { border-color: rgba(255,255,255,0.05) !important; }

.stProgress > div > div { background: linear-gradient(90deg, #64a0ff, #00dc82) !important; }

section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(100,160,255,0.12) !important;
}

div[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
}

div[data-testid="stCheckbox"] label {
    color: #c4d4e8 !important;
    font-size: 13px !important;
    letter-spacing: 0 !important;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Plotly chart backgrounds */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ================================================================
#  PLOTLY THEME
# ================================================================

PLOT_THEME = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(255,255,255,0.02)',
    font=dict(family='DM Mono, monospace', color='#c4d4e8', size=12),
    colorway=['#64a0ff','#00dc82','#ff64a0','#ffcc64','#a064ff','#64dcff'],
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.08)'),
    margin=dict(l=40,r=20,t=40,b=40),
)

def styled_fig(fig):
    fig.update_layout(**PLOT_THEME)
    return fig

# ================================================================
#  SESSION STATE
# ================================================================

for key in ['data','processed','task_type','target_col','models','results','step']:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ['models','results'] else {}

if st.session_state.step is None:
    st.session_state.step = 1

# ================================================================
#  HELPERS
# ================================================================

def detect_task(series: pd.Series) -> str:
    if series.dtype == 'object' or series.nunique() <= 20:
        return 'classification'
    return 'regression'

def preprocess(df: pd.DataFrame, target: str, task: str):
    df = df.copy().dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    encoded_cols, scaled_cols = [], []

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            encoded_cols.append(col)

    if task == 'classification' and y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.astype(str))

    X = X.fillna(X.median(numeric_only=True))

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        scaled_cols = numeric_cols

    return {
        'X': X, 'y': y,
        'feature_names': X.columns.tolist(),
        'encoded': encoded_cols,
        'scaled': scaled_cols,
    }

def train_model(processed, model_key, task, test_size=0.2, random_state=42, cv=5):
    X, y = processed['X'], processed['y']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models_map = {
        'logistic_regression':        LogisticRegression(max_iter=1000, random_state=random_state),
        'decision_tree':              DecisionTreeClassifier(random_state=random_state),
        'random_forest':              RandomForestClassifier(n_estimators=100, random_state=random_state),
        'linear_regression':          LinearRegression(),
        'decision_tree_regressor':    DecisionTreeRegressor(random_state=random_state),
        'random_forest_regressor':    RandomForestRegressor(n_estimators=100, random_state=random_state),
    }
    model = models_map[model_key]
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    scoring = 'accuracy' if task == 'classification' else 'r2'
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    result = {'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}

    if task == 'classification':
        avg = 'binary' if len(np.unique(y)) == 2 else 'weighted'
        result.update({
            'accuracy': accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred, average=avg, zero_division=0),
            'recall': recall_score(y_te, y_pred, average=avg, zero_division=0),
            'f1': f1_score(y_te, y_pred, average=avg, zero_division=0),
            'confusion': confusion_matrix(y_te, y_pred),
        })
    else:
        result.update({
            'r2': r2_score(y_te, y_pred),
            'mae': mean_absolute_error(y_te, y_pred),
            'mse': mean_squared_error(y_te, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_te, y_pred)),
        })

    if hasattr(model, 'feature_importances_'):
        result['importance'] = model.feature_importances_
    elif hasattr(model, 'coef_'):
        result['importance'] = np.abs(model.coef_).flatten()[:len(processed['feature_names'])]

    return result

def step_done(n):
    return st.session_state.step > n

def step_active(n):
    return st.session_state.step == n

# ================================================================
#  SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("<div class='ml-logo'>ML Universe</div>", unsafe_allow_html=True)
    st.markdown("<div class='ml-tagline'>Democratizing Machine Learning</div>", unsafe_allow_html=True)

    page = st.selectbox("", [
        "01 · Upload Data",
        "02 · Explore",
        "03 · Train Models",
        "04 · Results",
        "05 · Export",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Pipeline status
    st.markdown("<div style='font-size:10px;letter-spacing:2px;color:rgba(196,212,232,0.25);text-transform:uppercase;margin-bottom:10px;'>Pipeline Status</div>", unsafe_allow_html=True)

    steps = [
        ("Data Loaded", st.session_state.data is not None),
        ("Preprocessed", st.session_state.processed is not None),
        ("Models Trained", bool(st.session_state.results)),
    ]
    for label, done in steps:
        icon = "✓" if done else "○"
        color = "#00dc82" if done else "rgba(196,212,232,0.2)"
        st.markdown(f"<div style='font-size:12px;color:{color};padding:3px 0;'>{icon}  {label}</div>", unsafe_allow_html=True)

    if st.session_state.data is not None:
        st.markdown("---")
        df = st.session_state.data
        st.markdown(f"""
        <div style='font-size:11px;color:rgba(196,212,232,0.35);line-height:1.8;'>
        Rows: <span style='color:#64a0ff;'>{df.shape[0]:,}</span><br>
        Cols: <span style='color:#64a0ff;'>{df.shape[1]}</span><br>
        Missing: <span style='color:#ffcc64;'>{df.isnull().sum().sum():,}</span><br>
        Task: <span style='color:#00dc82;'>{st.session_state.task_type or "—"}</span>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
#  STEP INDICATOR (shared)
# ================================================================

def show_steps(current):
    steps = ["Upload","Explore","Train","Results","Export"]
    html = "<div class='step-row'>"
    for i, s in enumerate(steps, 1):
        if i < current:
            cls = "step-done"
            icon = "✓"
        elif i == current:
            cls = "step-active"
            icon = str(i)
        else:
            cls = "step-item"
            icon = str(i)
        html += f"<div class='step-item {cls}'>{icon} · {s}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ================================================================
#  PAGE 01 · UPLOAD
# ================================================================

if page == "01 · Upload Data":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Upload Dataset</div>
        <div class='page-sub'>CSV files · auto-detect task type</div>
    </div>
    """, unsafe_allow_html=True)
    show_steps(1)

    uploaded = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        help="Upload a CSV with column headers. First, select your target column."
    )

    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            st.session_state.data = data

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"<div class='metric-card'><div class='metric-num'>{data.shape[0]:,}</div><div class='metric-label'>Rows</div></div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='metric-card'><div class='metric-num'>{data.shape[1]}</div><div class='metric-label'>Columns</div></div>", unsafe_allow_html=True)
            with c3:
                st.markdown(f"<div class='metric-card'><div class='metric-num'>{data.isnull().sum().sum():,}</div><div class='metric-label'>Missing</div></div>", unsafe_allow_html=True)
            with c4:
                mem = data.memory_usage(deep=True).sum() / 1024
                st.markdown(f"<div class='metric-card'><div class='metric-num'>{mem:.1f}KB</div><div class='metric-label'>Size</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns([3,2])

            with col_left:
                st.markdown("<div class='section-label'>Data Preview</div>", unsafe_allow_html=True)
                st.dataframe(data.head(8), use_container_width=True, height=260)

            with col_right:
                st.markdown("<div class='section-label'>Column Types</div>", unsafe_allow_html=True)
                col_info = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes.astype(str),
                    'Nulls': data.isnull().sum(),
                    'Unique': data.nunique(),
                })
                st.dataframe(col_info, use_container_width=True, height=260)

            st.markdown("<div class='section-label'>Select Target Column</div>", unsafe_allow_html=True)
            target = st.selectbox("Target column", data.columns.tolist(), label_visibility="collapsed")

            if target:
                task = detect_task(data[target])
                st.session_state.task_type = task
                st.session_state.target_col = target

                task_color = "badge-blue" if task == "classification" else "badge-pink"
                st.markdown(f"""
                <div class='card card-green' style='margin-top:12px;'>
                    <span class='badge {task_color}'>{task}</span>
                    <div style='margin-top:10px;font-size:13px;color:#c4d4e8;'>
                        Target: <strong style='color:#e8f0fc;'>{target}</strong> ·
                        {data[target].nunique()} unique values ·
                        {data[target].isnull().sum()} missing
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if task == 'classification':
                    vc = data[target].value_counts().head(10)
                    fig = go.Figure(go.Bar(
                        x=vc.index.astype(str), y=vc.values,
                        marker=dict(color='#64a0ff', opacity=0.8)
                    ))
                    fig.update_layout(title="Class Distribution", **PLOT_THEME)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = go.Figure(go.Histogram(
                        x=data[target].dropna(),
                        nbinsx=30,
                        marker=dict(color='#64a0ff', opacity=0.8)
                    ))
                    fig.update_layout(title="Target Distribution", **PLOT_THEME)
                    st.plotly_chart(fig, use_container_width=True)

                if st.button("⚡ Preprocess Data →", use_container_width=False):
                    with st.spinner("Processing..."):
                        processed = preprocess(data, target, task)
                        st.session_state.processed = processed
                        st.session_state.step = 2
                    st.markdown("<div class='success-box'>✓ Data preprocessed. Head to Explore or Train.</div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"<div class='error-box'>Error reading file: {e}</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:48px;color:rgba(196,212,232,0.25);'>
            <div style='font-size:32px;margin-bottom:12px;'>📂</div>
            <div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:rgba(232,240,252,0.4);'>Drop a CSV to begin</div>
            <div style='font-size:12px;margin-top:8px;'>Supports classification and regression tasks</div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
#  PAGE 02 · EXPLORE
# ================================================================

elif page == "02 · Explore":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Data Exploration</div>
        <div class='page-sub'>Understand your data before training</div>
    </div>
    """, unsafe_allow_html=True)
    show_steps(2)

    if st.session_state.data is None:
        st.markdown("<div class='warn-box'>⚠ Upload a dataset first.</div>", unsafe_allow_html=True)
        st.stop()
    if st.session_state.processed is None:
        st.markdown("<div class='warn-box'>⚠ Preprocess your data first (Upload page).</div>", unsafe_allow_html=True)
        st.stop()

    data = st.session_state.data
    processed = st.session_state.processed

    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🔗 Correlations", "🔬 Features"])

    with tab1:
        st.markdown("<div class='section-label'>Descriptive Statistics</div>", unsafe_allow_html=True)
        st.dataframe(data.describe().round(3), use_container_width=True)

        st.markdown("<div class='section-label'>Preprocessing Applied</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='card card-blue'>
                <div class='card-label'>Encoded Columns</div>
                <div style='font-size:13px;color:#e8f0fc;'>{', '.join(processed['encoded']) if processed['encoded'] else 'None'}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='card card-green'>
                <div class='card-label'>Scaled Columns</div>
                <div style='font-size:13px;color:#e8f0fc;'>{len(processed['scaled'])} numeric features standardized</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        numeric = processed['X'].select_dtypes(include=[np.number])
        if numeric.shape[1] > 1:
            corr = numeric.corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale=[[0,'#ff64a0'],[0.5,'#060610'],[1,'#64a0ff']],
                zmid=0,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont=dict(size=9),
            ))
            fig.update_layout(title="Feature Correlation Matrix", height=500, **PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<div class='info-box'>Not enough numeric features for correlation analysis.</div>", unsafe_allow_html=True)

    with tab3:
        feature_names = processed['feature_names']
        st.markdown(f"<div class='section-label'>{len(feature_names)} Features After Preprocessing</div>", unsafe_allow_html=True)

        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            with cols[i % 3]:
                st.markdown(f"<div class='card' style='padding:10px 14px;margin-bottom:8px;'><span style='font-size:12px;color:#64a0ff;'>●</span> <span style='font-size:12px;'>{feat}</span></div>", unsafe_allow_html=True)

# ================================================================
#  PAGE 03 · TRAIN
# ================================================================

elif page == "03 · Train Models":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Train Models</div>
        <div class='page-sub'>Select algorithms · configure · run</div>
    </div>
    """, unsafe_allow_html=True)
    show_steps(3)

    if st.session_state.processed is None:
        st.markdown("<div class='warn-box'>⚠ Upload and preprocess data first.</div>", unsafe_allow_html=True)
        st.stop()

    task = st.session_state.task_type

    MODELS = {
        'classification': {
            'Logistic Regression': 'logistic_regression',
            'Decision Tree': 'decision_tree',
            'Random Forest': 'random_forest',
        },
        'regression': {
            'Linear Regression': 'linear_regression',
            'Decision Tree': 'decision_tree_regressor',
            'Random Forest': 'random_forest_regressor',
        }
    }
    available = MODELS.get(task, {})

    task_color = "badge-blue" if task == "classification" else "badge-pink"
    st.markdown(f"<span class='badge {task_color}'>{task} task</span><br><br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2,1])

    with col_left:
        st.markdown("<div class='section-label'>Select Models</div>", unsafe_allow_html=True)
        selected = st.multiselect("Models", list(available.keys()),
                                  default=list(available.keys()),
                                  label_visibility="collapsed")

    with col_right:
        st.markdown("<div class='section-label'>Configuration</div>", unsafe_allow_html=True)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("CV folds", 3, 10, 5)
        random_state = st.number_input("Random seed", value=42)

    if not selected:
        st.markdown("<div class='warn-box'>⚠ Select at least one model.</div>", unsafe_allow_html=True)
        st.stop()

    if st.button("🚀 Train Selected Models", use_container_width=False):
        results = {}
        progress = st.progress(0)
        status = st.empty()

        for i, name in enumerate(selected):
            status.markdown(f"<div class='info-box'>Training {name}...</div>", unsafe_allow_html=True)
            try:
                res = train_model(
                    st.session_state.processed,
                    available[name],
                    task,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    cv=int(cv_folds)
                )
                results[name] = res
            except Exception as e:
                st.markdown(f"<div class='error-box'>Error training {name}: {e}</div>", unsafe_allow_html=True)
            progress.progress((i+1) / len(selected))

        status.markdown(f"<div class='success-box'>✓ Trained {len(results)} model(s). Head to Results.</div>", unsafe_allow_html=True)
        st.session_state.results = results
        st.session_state.step = 4

    # Show previously trained results summary
    if st.session_state.results:
        st.markdown("<div class='section-label'>Last Training Run</div>", unsafe_allow_html=True)
        for name in st.session_state.results:
            r = st.session_state.results[name]
            if task == 'classification':
                key_metric = f"Accuracy: {r['accuracy']:.3f}"
            else:
                key_metric = f"R²: {r['r2']:.3f}"
            st.markdown(f"""
            <div class='card card-green'>
                <span class='badge badge-green'>trained</span>
                <span style='margin-left:10px;font-size:14px;color:#e8f0fc;font-family:Syne,sans-serif;font-weight:700;'>{name}</span>
                <span style='margin-left:12px;font-size:12px;color:rgba(196,212,232,0.5);'>{key_metric} · CV: {r['cv_mean']:.3f} ± {r['cv_std']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)

# ================================================================
#  PAGE 04 · RESULTS
# ================================================================

elif page == "04 · Results":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Results & Analysis</div>
        <div class='page-sub'>Compare models · visualize performance</div>
    </div>
    """, unsafe_allow_html=True)
    show_steps(4)

    if not st.session_state.results:
        st.markdown("<div class='warn-box'>⚠ Train models first.</div>", unsafe_allow_html=True)
        st.stop()

    task = st.session_state.task_type
    results = st.session_state.results

    # Build comparison table
    rows = []
    for name, r in results.items():
        if task == 'classification':
            rows.append({
                'Model': name,
                'Accuracy': round(r['accuracy'], 4),
                'Precision': round(r['precision'], 4),
                'Recall': round(r['recall'], 4),
                'F1': round(r['f1'], 4),
                'CV Mean': round(r['cv_mean'], 4),
                'CV Std': round(r['cv_std'], 4),
            })
        else:
            rows.append({
                'Model': name,
                'R²': round(r['r2'], 4),
                'MAE': round(r['mae'], 4),
                'MSE': round(r['mse'], 4),
                'RMSE': round(r['rmse'], 4),
                'CV Mean': round(r['cv_mean'], 4),
                'CV Std': round(r['cv_std'], 4),
            })

    df_results = pd.DataFrame(rows)
    best_metric = 'Accuracy' if task == 'classification' else 'R²'
    best_idx = df_results[best_metric].idxmax()
    best_model_name = df_results.loc[best_idx, 'Model']
    best_score = df_results.loc[best_idx, best_metric]

    # Best model highlight
    st.markdown(f"""
    <div class='best-model'>
        <div style='font-size:10px;letter-spacing:2px;color:rgba(0,220,130,0.6);text-transform:uppercase;margin-bottom:8px;'>Best Model</div>
        <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e8f0fc;'>{best_model_name}</div>
        <div style='font-size:13px;color:#00dc82;margin-top:4px;'>{best_metric}: {best_score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison table
    st.markdown("<div class='section-label'>Model Comparison</div>", unsafe_allow_html=True)
    st.dataframe(df_results, use_container_width=True)

    # Bar chart comparison
    metric_cols = [c for c in df_results.columns if c not in ['Model','CV Std']]
    fig = go.Figure()
    colors = ['#64a0ff','#00dc82','#ff64a0','#ffcc64']
    for i, name in enumerate(df_results['Model']):
        vals = df_results[df_results['Model']==name][metric_cols].values.flatten()
        fig.add_trace(go.Bar(name=name, x=metric_cols, y=vals,
                             marker_color=colors[i % len(colors)], opacity=0.85))
    fig.update_layout(barmode='group', title="Metrics Comparison", **PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # Per-model details
    st.markdown("<div class='section-label'>Per-Model Details</div>", unsafe_allow_html=True)
    for name, r in results.items():
        with st.expander(f"📈 {name}"):
            c1, c2 = st.columns(2)

            with c1:
                if task == 'classification':
                    metrics = [("Accuracy", r['accuracy']),("Precision", r['precision']),
                               ("Recall", r['recall']),("F1", r['f1'])]
                else:
                    metrics = [("R²", r['r2']),("MAE", r['mae']),("MSE", r['mse']),("RMSE", r['rmse'])]

                m_cols = st.columns(2)
                for i, (label, val) in enumerate(metrics):
                    with m_cols[i % 2]:
                        st.markdown(f"<div class='metric-card'><div class='metric-num' style='font-size:20px;'>{val:.4f}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class='card card-blue' style='margin-top:12px;'>
                    <div class='card-label'>Cross-Validation</div>
                    <div style='font-size:16px;color:#64a0ff;'>{r['cv_mean']:.4f} <span style='font-size:12px;color:rgba(196,212,232,0.4);'>± {r['cv_std']:.4f}</span></div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                if task == 'classification' and 'confusion' in r:
                    cm = r['confusion']
                    fig_cm = go.Figure(go.Heatmap(
                        z=cm, text=cm, texttemplate="%{text}",
                        colorscale=[[0,'#060610'],[1,'#64a0ff']],
                        showscale=False,
                    ))
                    fig_cm.update_layout(title="Confusion Matrix", height=280, **PLOT_THEME)
                    st.plotly_chart(fig_cm, use_container_width=True)

                if 'importance' in r:
                    feat_names = st.session_state.processed['feature_names']
                    imp = r['importance'][:len(feat_names)]
                    pairs = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)[:10]
                    names_sorted, vals_sorted = zip(*pairs)
                    fig_imp = go.Figure(go.Bar(
                        x=list(vals_sorted), y=list(names_sorted),
                        orientation='h',
                        marker=dict(color='#00dc82', opacity=0.8)
                    ))
                    fig_imp.update_layout(title="Feature Importance", height=280, **PLOT_THEME)
                    st.plotly_chart(fig_imp, use_container_width=True)

# ================================================================
#  PAGE 05 · EXPORT
# ================================================================

elif page == "05 · Export":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Export Results</div>
        <div class='page-sub'>Download your results · share your findings</div>
    </div>
    """, unsafe_allow_html=True)
    show_steps(5)

    if not st.session_state.results:
        st.markdown("<div class='warn-box'>⚠ Train models first.</div>", unsafe_allow_html=True)
        st.stop()

    task = st.session_state.task_type
    rows = []
    for name, r in st.session_state.results.items():
        row = {'Model': name, 'Task': task}
        if task == 'classification':
            row.update({'Accuracy': r['accuracy'],'Precision': r['precision'],
                        'Recall': r['recall'],'F1': r['f1']})
        else:
            row.update({'R2': r['r2'],'MAE': r['mae'],'MSE': r['mse'],'RMSE': r['rmse']})
        row.update({'CV_Mean': r['cv_mean'],'CV_Std': r['cv_std']})
        rows.append(row)

    df_export = pd.DataFrame(rows)
    st.dataframe(df_export, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv_buf = io.StringIO()
        df_export.to_csv(csv_buf, index=False)
        st.download_button(
            "💾 Download CSV",
            data=csv_buf.getvalue(),
            file_name=f"ml_universe_results_{datetime.now().strftime('%Y%m%d_%H%M') if False else 'results'}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        json_data = df_export.to_json(orient="records", indent=2)
        st.download_button(
            "📄 Download JSON",
            data=json_data,
            file_name="ml_universe_results.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card card-blue'>
        <div class='card-label'>Next Steps</div>
        <div style='font-size:13px;color:#c4d4e8;line-height:2;margin-top:8px;'>
            ● More data → better generalization<br>
            ● Feature engineering → more signal<br>
            ● Hyperparameter tuning → squeeze performance<br>
            ● Cross-validation stability → trust your metrics<br>
            ● Ensemble methods → combine weak learners
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================
#  FOOTER
# ================================================================

st.markdown("""
<div style='text-align:center;margin-top:40px;font-size:10px;letter-spacing:2px;
color:rgba(196,212,232,0.15);text-transform:uppercase;'>
ML Universe · Built by Ebatdev · Democratizing Machine Learning
</div>
""", unsafe_allow_html=True)

# Fix datetime import for export page
from datetime import datetime

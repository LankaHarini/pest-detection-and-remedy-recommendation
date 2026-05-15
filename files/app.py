"""
app.py - AI Pest Detection & Crop Yield Management System
==========================================================
Production-ready Streamlit application.

Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import uuid
import datetime
import io
import base64
from pathlib import Path

from database.db_manager import (
    save_detection_session, get_recent_detections,
    get_pest_statistics, get_detection_summary,
    save_yield_prediction, get_recent_predictions, get_yield_by_crop
)
from utils.pest_detector import PestDetector, PEST_CLASSES, CLASS_COLORS
from utils.yield_model import predict_yield, get_recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AgriShield — AI Pest & Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --bg:       #0b0f0a;
  --surface:  #111810;
  --border:   #1e2b1b;
  --green:    #4ade80;
  --lime:     #a3e635;
  --amber:    #fbbf24;
  --red:      #f87171;
  --text:     #e8f0e3;
  --muted:    #6b7f65;
}

html, body, [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    color: var(--green) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* Buttons */
.stButton > button {
    background: var(--green) !important;
    color: #0b0f0a !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--lime) !important;
    transform: translateY(-1px) !important;
}

/* Selectbox, slider, etc. */
.stSelectbox > div > div, .stNumberInput > div > div > input, .stTextInput > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* Risk badges */
.risk-low    { background:#14532d; color:#4ade80; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.risk-medium { background:#713f12; color:#fbbf24; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.risk-high   { background:#7f1d1d; color:#f87171; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }

/* Detection card */
.det-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.det-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.conf-bar-wrap {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    flex: 1;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 4px;
    background: var(--green);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
    background: transparent !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Demo badge */
.demo-badge {
    background: #1c1c00;
    border: 1px solid var(--amber);
    color: var(--amber);
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.8rem;
    display: inline-block;
    margin-bottom: 8px;
}

/* Header */
.app-header {
    padding: 24px 0 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--green), var(--lime));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 4px 0 0;
}

/* Table */
.dataframe { background: var(--surface) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

if "detector" not in st.session_state:
    st.session_state.detector = PestDetector()
if "last_detections" not in st.session_state:
    st.session_state.last_detections = []
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = None
if "last_risk" not in st.session_state:
    st.session_state.last_risk = "Low"

detector: PestDetector = st.session_state.detector


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RISK_CSS = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}
RISK_ICONS = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

PEST_HEX = {
    "Rat":         "#ff6f00",
    "Grasshopper": "#00e676",
    "Locust":      "#ff1744",
    "Aphid":       "#e040fb",
    "Caterpillar": "#ffd600",
}

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def risk_badge(level: str) -> str:
    return f'<span class="{RISK_CSS.get(level, "risk-low")}">{RISK_ICONS.get(level,"")} {level}</span>'

def gauge_chart(value: float, max_val: float, title: str, unit: str = "tons"):
    pct = min(value / max_val, 1.0) * 100
    color = "#4ade80" if pct > 60 else "#fbbf24" if pct > 30 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"color": "#e8f0e3", "size": 28}},
        title={"text": title, "font": {"color": "#6b7f65", "size": 14}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#6b7f65"},
            "bar":  {"color": color},
            "bgcolor": "#111810",
            "bordercolor": "#1e2b1b",
            "steps": [
                {"range": [0, max_val * 0.4], "color": "#1a0f0f"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "#131a0f"},
                {"range": [max_val * 0.7, max_val], "color": "#0f1a12"},
            ],
            "threshold": {
                "line": {"color": "#4ade80", "width": 3},
                "thickness": 0.75,
                "value": max_val * 0.8
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="#0b0f0a", plot_bgcolor="#0b0f0a",
        height=260, margin=dict(t=40, b=20, l=20, r=20),
        font={"color": "#e8f0e3"}
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px;">
      <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(135deg,#4ade80,#a3e635);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">🌾 AgriShield</div>
      <div style="color:#6b7f65;font-size:0.75rem;margin-top:2px;">AI-Powered Farm Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["🔍 Pest Detection", "📊 Yield Management", "📈 Analytics Dashboard", "⚙️ Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model status
    model_status = "✅ Loaded" if detector.model_loaded else "⚠️ Demo Mode"
    status_color = "#4ade80" if detector.model_loaded else "#fbbf24"
    st.markdown(f"""
    <div style="font-size:0.8rem;color:#6b7f65;">Model Status</div>
    <div style="color:{status_color};font-weight:600;font-size:0.9rem;">{model_status}</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick stats
    summary = get_detection_summary()
    st.markdown(f"""
    <div style="font-size:0.75rem;color:#6b7f65;margin-bottom:8px;">QUICK STATS</div>
    <div style="display:flex;justify-content:space-between;margin:4px 0;">
      <span style="color:#6b7f65;font-size:0.8rem;">Sessions</span>
      <span style="color:#4ade80;font-weight:600;">{summary.get('total_sessions',0) or 0}</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin:4px 0;">
      <span style="color:#6b7f65;font-size:0.8rem;">Pests Found</span>
      <span style="color:#4ade80;font-weight:600;">{summary.get('total_pests',0) or 0}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#6b7f65;line-height:1.6;">
    <b style="color:#4ade80;">Detects:</b><br>
    🐀 Rat &nbsp;|&nbsp; 🦗 Grasshopper<br>
    🪲 Locust &nbsp;|&nbsp; 🔵 Aphid<br>
    🐛 Caterpillar
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Pest Detection
# ─────────────────────────────────────────────────────────────────────────────

if "🔍 Pest Detection" in page:
    st.markdown("""
    <div class="app-header">
      <h1 class="app-title">Pest Detection</h1>
      <p class="app-sub">Upload a field image or use live camera — detects 5 pest classes in real-time</p>
    </div>
    """, unsafe_allow_html=True)

    if not detector.model_loaded:
        st.markdown('<div class="demo-badge">⚠ DEMO MODE — No trained model found. Detections are simulated.</div>', unsafe_allow_html=True)

    source_tab, camera_tab = st.tabs(["📁 Upload Image", "📷 Live Camera"])

    # ── Upload Image ──────────────────────────────────────────────────────────
    with source_tab:
        col_upload, col_settings = st.columns([3, 1])
        with col_settings:
            st.markdown("**Detection Settings**")
            conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.40, 0.05)
            detector.conf_threshold = conf_thresh

        with col_upload:
            uploaded = st.file_uploader(
                "Drop image here",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                label_visibility="collapsed"
            )

        if uploaded:
            pil_img = Image.open(uploaded)
            col_orig, col_result = st.columns(2)

            with col_orig:
                st.markdown("**Original Image**")
                st.image(pil_img, use_container_width=True)

            # Run detection
            with st.spinner("Analyzing image..."):
                cv_img = pil_to_cv(pil_img)
                annotated, detections = detector.detect(cv_img)
                risk = detector.get_risk_level(detections)

            with col_result:
                st.markdown("**Detection Result**")
                st.image(cv_to_pil(annotated), use_container_width=True)

            # Save to DB
            session_id = str(uuid.uuid4())
            save_detection_session(session_id, detections, uploaded.name, "upload", risk)
            st.session_state.last_detections = detections
            st.session_state.last_session_id = session_id
            st.session_state.last_risk = risk

            # ── Detection Summary ─────────────────────────────────────────────
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            counts = detector.count_by_class(detections)

            m1.metric("Total Pests", len(detections))
            m2.metric("Classes Found", sum(1 for c in counts.values() if c > 0))
            m3.metric("Avg Confidence",
                      f"{np.mean([d['confidence'] for d in detections]):.1%}" if detections else "—")
            m4.metric("Risk Level", risk)

            st.markdown(f"**Overall Risk: {risk_badge(risk)}**", unsafe_allow_html=True)

            # Detection list
            if detections:
                st.markdown("**Detected Pests**")
                for det in detections:
                    hex_color = PEST_HEX.get(det["class_name"], "#4ade80")
                    conf_pct = int(det["confidence"] * 100)
                    st.markdown(f"""
                    <div class="det-card">
                      <div class="det-dot" style="background:{hex_color};"></div>
                      <div style="flex:1;">
                        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#e8f0e3;">{det['class_name']}</div>
                        <div class="conf-bar-wrap">
                          <div class="conf-bar" style="width:{conf_pct}%;background:{hex_color};"></div>
                        </div>
                      </div>
                      <div style="color:{hex_color};font-weight:700;min-width:48px;text-align:right;">{conf_pct}%</div>
                      <div style="color:#6b7f65;font-size:0.75rem;">[{det['bbox'][0]},{det['bbox'][1]}]</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No pests detected in this image!")

            # Class distribution chart
            if any(v > 0 for v in counts.values()):
                st.markdown("**Class Distribution**")
                fig = px.bar(
                    x=list(counts.keys()),
                    y=list(counts.values()),
                    color=list(counts.keys()),
                    color_discrete_map=PEST_HEX,
                    labels={"x": "Pest Class", "y": "Count"},
                )
                fig.update_layout(
                    paper_bgcolor="#0b0f0a", plot_bgcolor="#111810",
                    font={"color": "#e8f0e3"}, showlegend=False,
                    height=280, margin=dict(t=20, b=40, l=40, r=20),
                    xaxis=dict(gridcolor="#1e2b1b"),
                    yaxis=dict(gridcolor="#1e2b1b"),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Camera Tab ────────────────────────────────────────────────────────────
    with camera_tab:
        st.markdown("**Live Camera Detection**")
        cam_img = st.camera_input("Point camera at crop field")

        if cam_img:
            pil_img = Image.open(cam_img)
            cv_img = pil_to_cv(pil_img)

            with st.spinner("Analyzing..."):
                annotated, detections = detector.detect(cv_img)
                risk = detector.get_risk_level(detections)

            st.image(cv_to_pil(annotated), caption="Detection Result", use_container_width=True)

            session_id = str(uuid.uuid4())
            save_detection_session(session_id, detections, "camera_capture", "camera", risk)
            st.session_state.last_detections = detections
            st.session_state.last_session_id = session_id
            st.session_state.last_risk = risk

            col1, col2, col3 = st.columns(3)
            col1.metric("Pests", len(detections))
            col2.metric("Risk", risk)
            col3.metric("Session", session_id[:8])

            st.markdown(f"**Risk Level: {risk_badge(risk)}**", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Yield Management
# ─────────────────────────────────────────────────────────────────────────────

elif "📊 Yield Management" in page:
    st.markdown("""
    <div class="app-header">
      <h1 class="app-title">Yield Management</h1>
      <p class="app-sub">Enter farm parameters to predict crop yield and assess pest impact</p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_results = st.columns([1, 1.4])

    with col_form:
        st.markdown("### Farm Parameters")

        crop_type = st.selectbox("Crop Type",
            ["Wheat", "Rice", "Corn/Maize", "Soybean", "Cotton",
             "Sugarcane", "Potato", "Tomato", "Onion", "Other"])

        c1, c2 = st.columns(2)
        farm_size = c1.number_input("Farm Size", min_value=0.1, max_value=10000.0, value=5.0, step=0.5)
        farm_unit = c2.selectbox("Unit", ["acres", "hectares"])

        c3, c4 = st.columns(2)
        fertilizer = c3.number_input("Fertilizer Usage", min_value=0.0, max_value=500.0, value=100.0, step=5.0)
        fert_unit = c4.selectbox("Fertilizer Unit", ["kg/acre", "kg/hectare"])

        c5, c6 = st.columns(2)
        rainfall = c5.number_input("Annual Rainfall", min_value=0.0, max_value=5000.0, value=600.0, step=10.0)
        rain_unit = c6.selectbox("Rainfall Unit", ["mm", "inches"])

        with st.expander("Advanced Parameters"):
            temperature = st.slider("Avg Temperature (°C)", -10.0, 50.0, 25.0, 0.5)
            soil_ph = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
            irrigation = st.checkbox("Irrigation Available", value=False)

        # Link to last detection
        st.markdown("---")
        st.markdown("**Pest Risk Input**")
        if st.session_state.last_detections:
            auto_risk = st.session_state.last_risk
            st.markdown(f"From last detection: {risk_badge(auto_risk)}", unsafe_allow_html=True)
            use_detection = st.checkbox("Use detected pest risk", value=True)
            if use_detection:
                pest_risk = auto_risk
            else:
                pest_risk = st.select_slider("Manual Risk Level", ["Low", "Medium", "High"], value="Low")
        else:
            pest_risk = st.select_slider("Pest Risk Level", ["Low", "Medium", "High"], value="Low")
            st.caption("Run pest detection first to auto-fill risk level")

        predict_btn = st.button("🌾 Predict Yield", type="primary", use_container_width=True)

    with col_results:
        if predict_btn:
            with st.spinner("Computing yield prediction..."):
                result = predict_yield(
                    crop_type=crop_type,
                    farm_size=farm_size,
                    farm_size_unit=farm_unit,
                    fertilizer_usage=fertilizer,
                    fertilizer_unit=fert_unit,
                    rainfall=rainfall,
                    rainfall_unit=rain_unit,
                    temperature=temperature,
                    soil_ph=soil_ph,
                    irrigation=irrigation,
                    pest_risk=pest_risk,
                )

                # Save to DB
                if st.session_state.last_session_id:
                    result["session_id"] = st.session_state.last_session_id
                save_yield_prediction(result)

            st.markdown("### Yield Prediction Results")

            # Gauge charts
            g1, g2 = st.columns(2)
            max_yield = result["predicted_yield"] * 1.5 or 10

            with g1:
                st.plotly_chart(
                    gauge_chart(result["predicted_yield"], max_yield, "Predicted Yield", "tons"),
                    use_container_width=True
                )
            with g2:
                st.plotly_chart(
                    gauge_chart(result["adjusted_yield"], max_yield, "Pest-Adjusted Yield", "tons"),
                    use_container_width=True
                )

            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Raw Yield", f"{result['predicted_yield']:.1f} tons")
            m2.metric("After Pest Loss", f"{result['adjusted_yield']:.1f} tons",
                      delta=f"-{result['pest_loss_tons']:.1f} tons",
                      delta_color="inverse")
            m3.metric("Yield/Acre", f"{result['yield_per_acre']:.2f} tons")

            st.markdown(f"**Pest Impact Risk: {risk_badge(pest_risk)}**", unsafe_allow_html=True)
            if result['pest_loss_pct'] > 0:
                st.warning(f"⚠️ Pest pressure will reduce yield by **{result['pest_loss_pct']:.1f}%** "
                           f"({result['pest_loss_tons']:.2f} tons lost)")

            # Factor breakdown
            with st.expander("📊 Factor Breakdown"):
                factors = result["factors"]
                factor_names = ["Fertilizer", "Rainfall", "Temperature", "Soil pH"]
                factor_vals = [
                    factors["fertilizer"] * 100,
                    factors["rainfall"] * 100,
                    factors["temperature"] * 100,
                    factors["soil_ph"] * 100,
                ]
                colors = ["#4ade80" if v >= 80 else "#fbbf24" if v >= 60 else "#f87171"
                          for v in factor_vals]

                fig = go.Figure(go.Bar(
                    x=factor_vals, y=factor_names, orientation="h",
                    marker_color=colors, text=[f"{v:.0f}%" for v in factor_vals],
                    textposition="outside"
                ))
                fig.update_layout(
                    paper_bgcolor="#0b0f0a", plot_bgcolor="#111810",
                    font={"color": "#e8f0e3"}, height=200,
                    margin=dict(t=10, b=10, l=20, r=60),
                    xaxis=dict(range=[0, 130], gridcolor="#1e2b1b"),
                    yaxis=dict(gridcolor="#1e2b1b"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Recommendations
            st.markdown("### Recommendations")
            recs = get_recommendations(result)
            for title, rec in recs:
                st.markdown(f"""
                <div style="background:#111810;border:1px solid #1e2b1b;border-radius:8px;
                            padding:12px 16px;margin:6px 0;">
                  <div style="font-family:'Syne',sans-serif;font-weight:700;color:#4ade80;
                              margin-bottom:4px;">{title}</div>
                  <div style="color:#e8f0e3;font-size:0.875rem;">{rec}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:400px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1e2b1b;border-radius:12px;color:#6b7f65;text-align:center;">
              <div>
                <div style="font-size:3rem;margin-bottom:16px;">🌾</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;">
                  Fill in farm parameters<br>and click Predict Yield
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page: Analytics Dashboard
# ─────────────────────────────────────────────────────────────────────────────

elif "📈 Analytics Dashboard" in page:
    st.markdown("""
    <div class="app-header">
      <h1 class="app-title">Analytics Dashboard</h1>
      <p class="app-sub">Detection history, pest trends, and yield analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Overview KPIs ─────────────────────────────────────────────────────────
    summary = get_detection_summary()
    pest_stats = get_pest_statistics()
    recent_dets = get_recent_detections(50)
    recent_preds = get_recent_predictions(20)
    yield_by_crop = get_yield_by_crop()

    risk_counts = summary.get("risk_counts", {})

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Sessions",   summary.get("total_sessions", 0) or 0)
    k2.metric("Pests Detected",   summary.get("total_pests", 0) or 0)
    k3.metric("Low Risk",         risk_counts.get("Low", 0))
    k4.metric("Medium Risk",      risk_counts.get("Medium", 0))
    k5.metric("High Risk",        risk_counts.get("High", 0))

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # ── Pest Frequency Chart ──────────────────────────────────────────────────
    with col_a:
        st.markdown("#### Pest Detection Frequency")
        if pest_stats:
            fig = go.Figure(go.Bar(
                x=[p["pest_class"] for p in pest_stats],
                y=[p["total_detections"] for p in pest_stats],
                marker_color=[PEST_HEX.get(p["pest_class"], "#4ade80") for p in pest_stats],
                text=[p["total_detections"] for p in pest_stats],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="#0b0f0a", plot_bgcolor="#111810",
                font={"color": "#e8f0e3"}, height=300,
                margin=dict(t=20, b=40, l=40, r=20),
                xaxis=dict(gridcolor="#1e2b1b"),
                yaxis=dict(gridcolor="#1e2b1b"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detection data yet. Run some pest detections first!")

    # ── Risk Distribution Pie ─────────────────────────────────────────────────
    with col_b:
        st.markdown("#### Risk Level Distribution")
        if any(risk_counts.values()):
            fig = go.Figure(go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=0.5,
                marker=dict(colors=["#4ade80", "#fbbf24", "#f87171"]),
            ))
            fig.update_layout(
                paper_bgcolor="#0b0f0a", plot_bgcolor="#0b0f0a",
                font={"color": "#e8f0e3"}, height=300,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk data yet.")

    # ── Avg Confidence per Class ──────────────────────────────────────────────
    if pest_stats:
        st.markdown("#### Average Detection Confidence by Pest Class")
        fig = go.Figure()
        for p in pest_stats:
            fig.add_trace(go.Bar(
                x=[p["pest_class"]],
                y=[round(p["avg_confidence"] * 100, 1)],
                name=p["pest_class"],
                marker_color=PEST_HEX.get(p["pest_class"], "#4ade80"),
                text=[f"{round(p['avg_confidence']*100,1)}%"],
                textposition="outside",
            ))
        fig.update_layout(
            paper_bgcolor="#0b0f0a", plot_bgcolor="#111810",
            font={"color": "#e8f0e3"}, height=280, showlegend=False,
            margin=dict(t=20, b=40, l=40, r=20),
            yaxis=dict(range=[0, 110], gridcolor="#1e2b1b"),
            xaxis=dict(gridcolor="#1e2b1b"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Yield by Crop ─────────────────────────────────────────────────────────
    if yield_by_crop:
        st.markdown("#### Average Predicted Yield by Crop Type")
        fig = go.Figure()
        crops = [r["crop_type"] for r in yield_by_crop]
        raw   = [r["avg_yield"] or 0 for r in yield_by_crop]
        adj   = [r["avg_adjusted_yield"] or 0 for r in yield_by_crop]

        fig.add_trace(go.Bar(name="Predicted",      x=crops, y=raw, marker_color="#4ade80"))
        fig.add_trace(go.Bar(name="Pest-Adjusted",  x=crops, y=adj, marker_color="#f87171"))

        fig.update_layout(
            barmode="group",
            paper_bgcolor="#0b0f0a", plot_bgcolor="#111810",
            font={"color": "#e8f0e3"}, height=300,
            margin=dict(t=20, b=40, l=40, r=20),
            legend=dict(bgcolor="#111810", bordercolor="#1e2b1b"),
            xaxis=dict(gridcolor="#1e2b1b"),
            yaxis=dict(gridcolor="#1e2b1b", title="Yield (tons)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent Detection History ──────────────────────────────────────────────
    st.markdown("#### Recent Detection Sessions")
    if recent_dets:
        for det in recent_dets[:10]:
            risk = det.get("risk_level", "Low")
            ts   = det.get("created_at", "")[:16]
            n    = det.get("total_pests", 0)
            src  = det.get("source_type", "upload")
            sid  = det.get("session_id", "")[:8]
            st.markdown(f"""
            <div style="background:#111810;border:1px solid #1e2b1b;border-radius:6px;
                        padding:10px 16px;margin:4px 0;display:flex;align-items:center;gap:16px;">
              <div style="font-family:'DM Mono',monospace;color:#6b7f65;font-size:0.75rem;min-width:120px;">{ts}</div>
              <div style="flex:1;color:#e8f0e3;">Session <b>{sid}</b> · {src}</div>
              <div style="color:#4ade80;font-weight:700;">{n} pest{"s" if n!=1 else ""}</div>
              <div>{risk_badge(risk)}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No detection sessions yet.")


# ─────────────────────────────────────────────────────────────────────────────
# Page: Model Info
# ─────────────────────────────────────────────────────────────────────────────

elif "⚙️ Model Info" in page:
    st.markdown("""
    <div class="app-header">
      <h1 class="app-title">Model Information</h1>
      <p class="app-sub">YOLOv8 architecture details, training setup, and dataset structure</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Architecture")
        st.markdown("""
        <div style="background:#111810;border:1px solid #1e2b1b;border-radius:8px;padding:20px;">
          <table style="width:100%;border-collapse:collapse;color:#e8f0e3;font-size:0.875rem;">
            <tr><td style="color:#6b7f65;padding:6px 0;">Base Architecture</td><td style="color:#4ade80;font-weight:600;">YOLOv8s / YOLOv8m</td></tr>
            <tr><td style="color:#6b7f65;padding:6px 0;">Input Resolution</td><td>640 × 640</td></tr>
            <tr><td style="color:#6b7f65;padding:6px 0;">Classes</td><td>5 pest classes</td></tr>
            <tr><td style="color:#6b7f65;padding:6px 0;">Training Epochs</td><td>50+ (recommended 100)</td></tr>
            <tr><td style="color:#6b7f65;padding:6px 0;">Optimizer</td><td>AdamW</td></tr>
            <tr><td style="color:#6b7f65;padding:6px 0;">Augmentation</td><td>Mosaic, MixUp, Flip, HSV</td></tr>
            <tr><td style="color:#6b7f65;padding:6px 0;">Framework</td><td>Ultralytics YOLOv8</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Pest Classes")
        for cls in PEST_CLASSES:
            hex_c = PEST_HEX.get(cls, "#4ade80")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
              <div style="width:14px;height:14px;border-radius:50%;background:{hex_c};"></div>
              <span style="color:#e8f0e3;">{cls}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Dataset Structure")
        st.code("""
pest_system/
├── app.py                    # Streamlit app
├── train.py                  # Training script
├── requirements.txt
├── dataset/
│   ├── data.yaml             # YOLOv8 config
│   ├── images/
│   │   ├── train/            # Training images
│   │   ├── val/              # Validation images
│   │   └── test/             # Test images
│   └── labels/
│       ├── train/            # YOLO format labels
│       ├── val/
│       └── test/
├── models/
│   └── pest_detector.pt      # Trained weights
├── database/
│   ├── db_manager.py
│   └── pest_system.db
└── utils/
    ├── pest_detector.py
    └── yield_model.py
        """, language="text")

        st.markdown("### Training Command")
        st.code("""
# Basic training (50 epochs)
python train.py

# Advanced options
python train.py \\
  --model yolov8m.pt \\
  --epochs 100 \\
  --imgsz 640 \\
  --batch 16

# YOLO label format per image:
# class_id cx cy width height
# (all normalized 0-1)
# Example: 0 0.5 0.5 0.3 0.4
        """, language="bash")

    st.markdown("### YOLO Annotation Format")
    st.markdown("""
    Each `.txt` label file corresponds to an image:
    ```
    # <class_id> <center_x> <center_y> <width> <height>  (all 0-1 normalized)
    0 0.523 0.481 0.142 0.201     # Rat
    2 0.782 0.321 0.087 0.112     # Locust
    4 0.234 0.671 0.063 0.084     # Caterpillar
    ```
    Use **Roboflow**, **LabelImg**, or **CVAT** to annotate images.
    """)

    st.markdown("### Recommended Datasets")
    st.markdown("""
    - [Roboflow Universe — Pest Detection](https://universe.roboflow.com/browse/agriculture)
    - [IP102 Insect Pest Dataset](https://github.com/xpwu95/IP102) — 102 insect classes
    - [Kaggle Agricultural Pests](https://www.kaggle.com/search?q=agricultural+pest+detection)
    """)

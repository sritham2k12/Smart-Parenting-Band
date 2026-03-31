"""
Smart Parenting Health Monitor — Streamlit Dashboard
Full-featured monitoring dashboard with real-time updates,
charts, emotion AI, alert panel, and LLM insights.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from database import (init_db, fetch_recent_readings,
                      fetch_emotion_history, fetch_anomalies,
                      get_summary_stats, insert_single_reading,
                      insert_emotion_prediction)
from data_simulator import generate_live_reading, generate_dataset
from data_processor import prepare_training_data, prepare_inference_payload
from emotion_engine import (predict_emotion, rule_based_predict,
                            load_model, train_emotion_model,
                            get_feature_importance)
from alerts import check_alerts, evaluate_trend_alerts, get_alert_summary
from llm_engine import generate_insights


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParentAI Monitor",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Dark theme overrides ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem !important;
    font-weight: 600;
    color: #58a6ff;
}
[data-testid="stMetricLabel"] {
    color: #8b949e;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* ── Alert boxes ── */
.alert-warning {
    background: rgba(248,81,73,0.12);
    border-left: 4px solid #f85149;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #ffa198;
    font-size: 0.9rem;
}
.alert-caution {
    background: rgba(210,153,34,0.12);
    border-left: 4px solid #d2a21f;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #f0c040;
    font-size: 0.9rem;
}
.alert-normal {
    background: rgba(63,185,80,0.12);
    border-left: 4px solid #3fb950;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #7ee787;
    font-size: 0.9rem;
}

/* ── Emotion badge ── */
.emotion-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 100px;
    font-weight: 600;
    font-size: 1rem;
    text-transform: capitalize;
    letter-spacing: 0.05em;
}

/* ── Insight card ── */
.insight-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}
.insight-card h4 {
    color: #58a6ff;
    margin-bottom: 8px;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.insight-card p {
    color: #c9d1d9;
    line-height: 1.6;
    font-size: 0.95rem;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Live dot ── */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #3fb950;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.4); }
}

/* ── Hide Streamlit branding ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
COLORS = {
    "bg":         "#0d1117",
    "surface":    "#161b22",
    "border":     "#30363d",
    "text":       "#e6edf3",
    "muted":      "#8b949e",
    "blue":       "#58a6ff",
    "green":      "#3fb950",
    "yellow":     "#d29922",
    "red":        "#f85149",
    "purple":     "#bc8cff",
    "teal":       "#39d0d3",
}

EMOTION_COLORS = {
    "calm":     "#3fb950",
    "excited":  "#d29922",
    "stressed": "#f85149",
    "fatigue":  "#bc8cff",
    "sleep":    "#58a6ff",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_setup():
    """Auto-initialize and seed if DB is empty."""
    init_db()
    stats = get_summary_stats()
    if not stats["total_readings"]:
        with st.spinner("⚙️ First run — seeding 7 days of simulation data..."):
            df = generate_dataset(days=7, interval_minutes=5)
            from database import insert_sensor_data
            insert_sensor_data(df)
            X, y, scaler, le = prepare_training_data(df)
            train_emotion_model(X, y, le)
        st.success("Setup complete!")
        st.rerun()


def get_emotion_color(emotion: str) -> str:
    return EMOTION_COLORS.get(emotion, COLORS["blue"])


def emotion_html(emotion: str, confidence: float) -> str:
    col = get_emotion_color(emotion)
    return (f'<span class="emotion-badge" '
            f'style="background:{col}22;color:{col};border:1.5px solid {col}40;">'
            f'{emotion} &nbsp; <small>{confidence:.0%}</small></span>')


# ─── Plotly theme defaults ────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="Space Grotesk", color=COLORS["muted"], size=12),
    xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
               tickcolor=COLORS["muted"]),
    yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
               tickcolor=COLORS["muted"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
)


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_vitals_chart(df: pd.DataFrame) -> go.Figure:
    """4-panel time-series: HR, Temperature, SpO2, Activity."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").tail(60)

    df["time"] = df["timestamp"].dt.strftime("%H:%M")

    df["heart_rate_smooth"] = df["heart_rate"].rolling(5).mean()
    df["temperature_smooth"] = df["temperature"].rolling(5).mean()
    df["spo2_smooth"] = df["spo2"].rolling(5).mean()
    df["activity_smooth"] = df["activity"].rolling(5).mean()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "❤️ Heart Rate (bpm)",
            "🌡️ Temperature (°C)",
            "🫁 SpO2 (%)",
            "⚡ Activity Level"
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.10
    )

    def add_line(row, col, y_col, color, ref_val=None, ref_label=None):
        fig.add_trace(go.Scatter(
            x=df["time"], y=df[y_col],
            mode="lines", line=dict(color=color, width=2.5),
            name=y_col, showlegend=False,
            hovertemplate=f"<b>{y_col}</b>: %{{y:.1f}}<br>%{{x}}<extra></extra>"
        ), row=row, col=col)
        if ref_val:
            fig.add_hline(y=ref_val, line_dash="dot",
                          line_color=COLORS["red"], opacity=0.5,
                          annotation_text=ref_label, row=row, col=col)

    add_line(1, 1, "heart_rate_smooth",  COLORS["red"],    150, "Alert")
    add_line(1, 2, "temperature_smooth", COLORS["yellow"],  38.5, "Fever")
    add_line(2, 1, "spo2_smooth",        COLORS["teal"],    92,  "Low")
    add_line(2, 2, "activity_smooth",    COLORS["green"])

    # Anomaly markers
    anom = df[df.get("anomaly", pd.Series(False, index=df.index)) == True] \
           if "anomaly" in df.columns else pd.DataFrame()
    if not anom.empty:
        for (row, col, col_name) in [(1,1,"heart_rate"),(1,2,"temperature")]:
            fig.add_trace(go.Scatter(
                x=anom["time"], y=anom[col_name],
                mode="markers",
                marker=dict(color=COLORS["red"], size=8, symbol="x"),
                name="Anomaly", showlegend=False,
                hovertemplate="⚠️ Anomaly<extra></extra>"
            ), row=row, col=col)

    fig.update_layout(
        **PLOT_LAYOUT,
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Vital Signs — 24h Overview",
                   font=dict(color=COLORS["text"], size=15))
    )
    # Individual axis tweaks
    fig.update_yaxes(range=[50, 180], row=1, col=1)
    fig.update_yaxes(range=[36, 40],  row=1, col=2)
    fig.update_yaxes(range=[85, 102], row=2, col=1)
    fig.update_yaxes(range=[0, 105],  row=2, col=2)
    fig.update_xaxes(
        tickangle=0,
        nticks=6,
        tickfont=dict(size=10)
    )
    return fig


def build_emotion_timeline(em_df: pd.DataFrame) -> go.Figure:
    """Scatter timeline colored by predicted emotion."""
    if em_df.empty:
        return go.Figure(layout=dict(**PLOT_LAYOUT,
                         title=dict(text="No emotion data yet")))

    em_df["timestamp"] = pd.to_datetime(em_df["timestamp"])
    em_df = em_df.sort_values(by="timestamp").tail(60)
    em_df["time"] = em_df["timestamp"].dt.strftime("%H:%M")

    fig = go.Figure()
    for emotion, grp in em_df.groupby("predicted_emotion"):
        fig.add_trace(go.Scatter(
            x=grp["time"], y=grp["confidence"],
            mode="markers",
            name=emotion.capitalize(),
            marker=dict(color=get_emotion_color(emotion), size=7, opacity=0.8),
            hovertemplate=f"<b>{emotion}</b><br>Confidence: %{{y:.0%}}<br>%{{x}}<extra></extra>"
        ))

    fig.update_layout(**PLOT_LAYOUT,
                      height=250,
                      title=dict(text="Emotion Predictions Timeline",
                                 font=dict(color=COLORS["text"], size=14)))

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))

    fig.update_yaxes(title="Confidence", tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(tickangle=0, nticks=6)
    return fig


def build_emotion_donut(em_df: pd.DataFrame) -> go.Figure:
    """Donut chart of emotion distribution."""
    if em_df.empty:
        return go.Figure()

    dist = em_df["predicted_emotion"].value_counts()
    colors = [get_emotion_color(e) for e in dist.index]

    fig = go.Figure(go.Pie(
        labels=[e.capitalize() for e in dist.index],
        values=dist.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
        textinfo="percent",
        textfont=dict(size=13, family="Space Grotesk"),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="Space Grotesk", color=COLORS["muted"]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        annotations=[dict(text="Emotion<br>Mix", x=0.5, y=0.5,
                          font=dict(size=13, color=COLORS["muted"]),
                          showarrow=False)]
    )
    return fig


def build_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation matrix of vitals."""
    cols = ["heart_rate", "temperature", "activity", "spo2", "hrv"]
    corr = df[cols].corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, COLORS["blue"]], [0.5, COLORS["surface"]], [1, COLORS["red"]]],
        zmid=0,
        text=corr.values,
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{x} × %{y}</b><br>r = %{z}<extra></extra>"
    ))
    fig.update_layout(**PLOT_LAYOUT, height=320,
                      title=dict(text="Vital Signs Correlation",
                                 font=dict(color=COLORS["text"], size=14)))
    fig.update_xaxes(tickangle=0, nticks=6)
    return fig


def build_hr_distribution(df: pd.DataFrame) -> go.Figure:
    """Heart rate distribution with normal range band."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").tail(60)

    fig = go.Figure()
    fig.add_vrect(x0=70, x1=130, fillcolor=COLORS["green"],
                  opacity=0.07, line_width=0, annotation_text="Normal range")
    fig.add_trace(go.Histogram(
        x=df["heart_rate"],
        nbinsx=40,
        marker_color=COLORS["red"],
        opacity=0.75,
        name="Heart Rate",
        hovertemplate="HR: %{x:.0f} bpm<br>Count: %{y}<extra></extra>"
    ))
    fig.update_layout(**PLOT_LAYOUT,
                      height=250,
                      title=dict(text="Heart Rate Distribution",
                                 font=dict(color=COLORS["text"], size=14)))

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))

    fig.update_xaxes(title="BPM", tickangle=0, nticks=6)
    fig.update_yaxes(title="Count")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <div style='font-size:2.5rem;'>👶</div>
            <div style='font-size:1.2rem; font-weight:700; color:#e6edf3;'>ParentAI Monitor</div>
            <div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>
                Smart Child Health Dashboard
            </div>
        </div>
        <hr style='border-color:#30363d; margin:12px 0 20px;'>
        """, unsafe_allow_html=True)

        st.markdown("### ⚙️ Settings")
        hours = st.slider("Data window (hours)", 1, 168, 24)
        refresh_rate = st.select_slider(
            "Auto-refresh", options=[5, 10, 30, 60, 120], value=30
        )
        live_mode = st.toggle("🔴 Live Monitoring", value=True)
        st.session_state["live_mode"] = live_mode

        st.divider()
        st.markdown("### 🧠 AI Settings")
        show_insights = st.toggle("Generate AI Insights", value=True)
        insight_hours = st.slider("Insight window (hours)", 1, 72, 12)

        st.divider()
        st.markdown("### 🛠️ Actions")
        col1, col2 = st.columns(2)
        retrain = col1.button("🔄 Retrain", use_container_width=True)
        seed    = col2.button("📦 Reseed", use_container_width=True)

        if seed:
            with st.spinner("Generating data..."):
                df = generate_dataset(days=7, interval_minutes=5)
                from database import insert_sensor_data
                insert_sensor_data(df)
            st.success("7-day data seeded!")

        if retrain:
            with st.spinner("Retraining model..."):
                df_all = fetch_recent_readings(hours=168)
                if len(df_all) > 200:
                    X, y, scaler, le = prepare_training_data(df_all)
                    train_emotion_model(X, y, le)
                    st.success("Model retrained!")
                else:
                    st.warning("Need more data — click Reseed first.")

        # Export button
        st.divider()
        st.markdown("### 📤 Export")
        df_export = fetch_recent_readings(hours=hours)
        if not df_export.empty:
            csv = df_export.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"health_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()
        st.markdown(
            "<div style='color:#8b949e;font-size:0.75rem;text-align:center;'>"
            "v1.0 · Smart Parenting AI · 2024</div>",
            unsafe_allow_html=True
        )

    return hours, refresh_rate, live_mode, show_insights, insight_hours


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ensure_setup()
    hours, refresh_rate, live_mode, show_insights, insight_hours = render_sidebar()

    # ── Page Title ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='display:flex; align-items:center; gap:12px; padding:8px 0 20px;'>
        <div style='font-size:2rem;'>👶</div>
        <div>
            <div style='font-size:1.6rem; font-weight:700; color:#e6edf3; line-height:1.1;'>
                Smart Parenting Health Monitor
            </div>
            <div style='color:#8b949e; font-size:0.85rem;'>
                AI · IoT · Emotion Detection · Real-Time Analytics
            </div>
        </div>
        <div style='margin-left:auto; display:flex; align-items:center; gap:8px;'>
            <span class='live-dot'></span>
            <span style='color:#3fb950; font-size:0.85rem; font-weight:600;'>LIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live reading ────────────────────────────────────────────────────────
    if live_mode:
        live = generate_live_reading()
        model = load_model()
        if model:
            X = prepare_inference_payload(live)
            emotion, confidence = predict_emotion(X)
        else:
            emotion, confidence = rule_based_predict(live)
        live["predicted_emotion"] = emotion
        live["confidence"]        = confidence

        alerts = check_alerts(live)
        alert_sum = get_alert_summary(alerts)

        # Save to DB
        rid = insert_single_reading(live)
        if rid:
            insert_emotion_prediction(rid, live["timestamp"], emotion, confidence)
    else:
        live, alerts, alert_sum = None, [], {"level": "normal", "count": 0}

    # ── Alert Banner ────────────────────────────────────────────────────────
    if alerts:
        level = alert_sum["level"]
        cls   = "alert-warning" if level == "warning" else "alert-caution"
        for a in alerts:
            st.markdown(
                f'<div class="{cls}">{a.icon} <strong>{a.metric.replace("_"," ").title()}</strong>'
                f' — {a.message}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="alert-normal">🟢 All vitals within normal range — child is doing great!</div>',
            unsafe_allow_html=True
        )

    # ── Live Metrics Row ────────────────────────────────────────────────────
    if live:
        st.markdown('<p class="section-title">📡 Live Sensor Feed</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("❤️ Heart Rate",  f"{live['heart_rate']:.0f}", "bpm")
        c2.metric("🌡️ Temperature", f"{live['temperature']:.1f}", "°C")
        c3.metric("⚡ Activity",     f"{live['activity']:.0f}",   "/100")
        c4.metric("🫁 SpO2",        f"{live['spo2']:.0f}",       "%")
        c5.metric("💓 HRV",         f"{live['hrv']:.0f}",        "ms")
        c6.metric("🕐 Last Updated", datetime.now().strftime("%H:%M:%S"), "")

        em_col = get_emotion_color(emotion)
        st.markdown(
            f'<div style="margin:8px 0 20px;">'
            f'<span style="color:#8b949e;font-size:0.9rem;">Current Emotion: </span>'
            f'{emotion_html(emotion, confidence)}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.map(pd.DataFrame({
            "lat": [live["lat"]],
            "lon": [live["lon"]]
        }))

    # ── Load Historical Data ────────────────────────────────────────────────
    df = fetch_recent_readings(hours=hours)
    df = df.tail(100)  # keep last 100 data points for cleaner charts
    df["heart_rate_smooth"] = df["heart_rate"].rolling(5, min_periods=1).mean()
    em_df = fetch_emotion_history(hours=hours).tail(100)

    if df.empty:
        st.warning("No historical data found. Click **📦 Reseed** in the sidebar.")
        return

    # ── Aggregate Metrics Row ──────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Period Statistics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Heart Rate",  f"{df['heart_rate'].mean():.0f} bpm",
              f"σ {df['heart_rate'].std():.1f}")
    c2.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C",
              f"max {df['temperature'].max():.1f}")
    c3.metric("Avg Activity",    f"{df['activity'].mean():.0f}/100",
              f"peak {df['activity'].max():.0f}")
    c4.metric("Avg SpO2",        f"{df['spo2'].mean():.1f}%",
              f"min {df['spo2'].min():.0f}")
    anomaly_count = int(df.get("anomaly", pd.Series(0)).sum()) \
                    if "anomaly" in df.columns else 0
    c5.metric("Anomalies",       str(anomaly_count),
              "⚠️ detected" if anomaly_count else "✅ none")

    st.divider()

    # ── Vitals Chart ────────────────────────────────────────────────────────
    st.plotly_chart(build_vitals_chart(df), use_container_width=True, config={"displayModeBar": False})

    # ── Emotion Row ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">🧠 Emotion AI Analytics</p>', unsafe_allow_html=True)
    ec1, ec2 = st.columns([2, 1])

    with ec1:
        st.plotly_chart(build_emotion_timeline(em_df),
                        use_container_width=True, config={"displayModeBar": False})
    with ec2:
        st.plotly_chart(build_emotion_donut(em_df),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Secondary Analytics ─────────────────────────────────────────────────
    st.divider()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.plotly_chart(build_hr_distribution(df),
                        use_container_width=True, config={"displayModeBar": False})
    with sc2:
        st.plotly_chart(build_correlation_heatmap(df),
                        use_container_width=True, config={"displayModeBar": False})

    # ── Feature Importance ──────────────────────────────────────────────────
    fi = get_feature_importance()
    if fi:
        st.divider()
        st.markdown('<p class="section-title">🔍 Model Feature Importance</p>',
                    unsafe_allow_html=True)
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(8)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(color=COLORS["blue"], opacity=0.85),
            hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>"
        ))
        fig_fi.update_layout(**PLOT_LAYOUT, height=280,
                              title=dict(text="Random Forest — Feature Importance",
                                         font=dict(color=COLORS["text"], size=14)))
        st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})

    # ── AI Insights Panel ───────────────────────────────────────────────────
    if show_insights:
        st.divider()
        st.markdown('<p class="section-title">💡 AI Parenting Insights</p>',
                    unsafe_allow_html=True)

        gen = st.button("✨ Generate Insights", type="primary", use_container_width=False)
        if gen or "insights" not in st.session_state:
            with st.spinner("🤖 Analyzing health patterns..."):
                df_ins = fetch_recent_readings(hours=insight_hours)
                em_ins = fetch_emotion_history(hours=insight_hours)
                anom_ins = fetch_anomalies(hours=insight_hours)

                stats = {
                    "avg_hr":       round(df_ins["heart_rate"].mean(), 1) if not df_ins.empty else 0,
                    "avg_temp":     round(df_ins["temperature"].mean(), 2) if not df_ins.empty else 0,
                    "avg_activity": round(df_ins["activity"].mean(), 1) if not df_ins.empty else 0,
                    "avg_spo2":     round(df_ins["spo2"].mean(), 1) if not df_ins.empty else 0,
                }
                dist = em_ins["predicted_emotion"].value_counts(normalize=True).to_dict() \
                       if not em_ins.empty else {"calm": 1.0}
                anomaly_types = anom_ins["anomaly_type"].dropna().tolist() \
                                if not anom_ins.empty else []

                st.session_state["insights"] = generate_insights(stats, dist, anomaly_types)

        insight = st.session_state.get("insights", {})
        if insight:
            level = insight.get("alert_level", "normal")
            src   = insight.get("source", "simulated")
            badge = "🟢 Normal" if level == "normal" else \
                    "🟡 Caution" if level == "caution" else "🔴 Warning"
            source_badge = "🤖 OpenAI GPT" if src == "openai" else "⚡ Smart Engine"

            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**Alert Level:** {badge}")
            col_b.markdown(f"<div style='text-align:right;color:#8b949e;font-size:0.8rem;'>{source_badge}</div>",
                           unsafe_allow_html=True)

            def insight_card(title, content, icon=""):
                st.markdown(
                    f'<div class="insight-card">'
                    f'<h4>{icon} {title}</h4>'
                    f'<p>{content}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            insight_card("Overall Summary",    insight.get("summary", ""), "📋")
            insight_card("Health Insights",    insight.get("health_insights", ""), "🩺")
            insight_card("Emotion Analysis",   insight.get("emotion_analysis", ""), "🧠")
            insight_card("Parenting Tips",     insight.get("parenting_tips", ""), "💛")

    # ── Recent Data Table ───────────────────────────────────────────────────
    st.divider()
    with st.expander("📋 Raw Sensor Data (last 50 records)"):
        show_cols = ["timestamp","heart_rate","temperature","activity","spo2","hrv"]
        if "anomaly" in df.columns:
            show_cols.append("anomaly")
        st.dataframe(
            df[show_cols].tail(50).sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True
        )

    # ── Auto-refresh ────────────────────────────────────────────────────────
    if st.session_state.get("live_mode", False):
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()

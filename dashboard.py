import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import config

st.set_page_config(page_title="DriftGuard - SOC Dashboard", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
.stApp { background-color: #0d1117; color: #e6edf3; }
[data-testid="stMetricValue"] { color: #58a6ff; }
[data-testid="stMetricLabel"] { color: #8b949e; }
.stDataFrame { background-color: #161b22; }
div[data-baseweb="select"] { background-color: #161b22; }
h1, h2, h3 { color: #e6edf3; }
.alert-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.risk-high { color: #f85149; font-weight: 600; }
.risk-medium { color: #d29922; font-weight: 600; }
.risk-low { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    alerts = pd.read_csv(config.EXPLAINED_ALERTS_PATH)
    logs = pd.read_csv(config.FEATURES_PATH)
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])
    return alerts, logs

alerts, logs = load_data()

def risk_tier(pct):
    if pct >= 0.995:
        return "high"
    if pct >= 0.98:
        return "medium"
    return "low"

alerts["risk_tier"] = alerts["score_percentile"].apply(risk_tier)

st.title("🛡️ DriftGuard: Behavioral Anomaly Detection")
st.caption("AI-powered analyst dashboard for real-time access anomaly triage")

col1, col2, col3, col4, col5 = st.columns(5)
total_alerts = len(alerts)
true_attacks = (alerts["predicted_attack_type"] != "normal").sum()
false_positives = (alerts["predicted_attack_type"] == "normal").sum()
avg_conf = alerts.loc[alerts["predicted_attack_type"] != "normal", "confidence"].mean()
high_risk = (alerts["risk_tier"] == "high").sum()

col1.metric("Total Alerts", total_alerts)
col2.metric("Likely Real Attacks", true_attacks)
col3.metric("Likely False Positives", false_positives)
col4.metric("Avg Attack Confidence", f"{avg_conf:.0%}" if pd.notna(avg_conf) else "N/A")
col5.metric("High Risk Alerts", high_risk)

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Real Attack Type Distribution (excludes false positives)")
    dist = alerts[alerts["predicted_attack_type"] != "normal"]["predicted_attack_type"].value_counts().reset_index()
    dist.columns = ["attack_type", "count"]
    fig = px.bar(dist, x="count", y="attack_type", orientation="h", color="count",
                 color_continuous_scale=["#30363d", "#f85149"])
    fig.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font_color="#e6edf3", showlegend=False, height=350,
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Detection Source")
    source_dist = alerts["flagged_by"].value_counts().reset_index()
    source_dist.columns = ["source", "count"]
    fig2 = go.Figure(data=[go.Pie(
        labels=source_dist["source"], values=source_dist["count"], hole=0.55,
        marker=dict(colors=["#58a6ff", "#f85149"]),
    )])
    fig2.update_layout(
        paper_bgcolor="#0d1117", font_color="#e6edf3", height=350,
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

trend_left, trend_right = st.columns(2)

with trend_left:
    st.subheader("Risk Score Distribution")
    fig3 = px.histogram(alerts, x="score_percentile", nbins=40, color_discrete_sequence=["#58a6ff"])
    fig3.add_vline(x=0.98, line_dash="dash", line_color="#f85149",
                    annotation_text="alert threshold", annotation_font_color="#e6edf3")
    fig3.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        height=320, xaxis_title="Risk percentile", yaxis_title="Alert count", bargap=0.05,
    )
    st.plotly_chart(fig3, use_container_width=True)

with trend_right:
    st.subheader("Alerts Over Time")
    alerts_with_time = alerts.merge(logs[["session_id", "timestamp"]], on="session_id", how="left")
    alerts_with_time["date"] = alerts_with_time["timestamp"].dt.date
    daily = alerts_with_time.groupby(["date", "predicted_attack_type"]).size().reset_index(name="count")
    fig4 = px.area(daily, x="date", y="count", color="predicted_attack_type",
                    color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_layout(
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
        height=320, legend=dict(orientation="h", yanchor="bottom", y=-0.4), legend_title="",
    )
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("Highest Risk Entities")
entity_risk = alerts[alerts["predicted_attack_type"] != "normal"].groupby("entity_id").agg(
    alert_count=("session_id", "count"),
    max_risk=("score_percentile", "max"),
    top_attack=("predicted_attack_type", lambda x: x.value_counts().idxmax()),
).sort_values("max_risk", ascending=False).head(10).reset_index()

fig5 = px.bar(entity_risk, x="max_risk", y="entity_id", orientation="h", color="top_attack",
              color_discrete_sequence=px.colors.qualitative.Set2,
              hover_data=["alert_count"])
fig5.update_layout(
    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", font_color="#e6edf3",
    height=380, yaxis={"categoryorder": "total ascending"}, legend_title="Top Attack Type",
)
st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.subheader("Model Performance by Attack Type (held-out test set)")
st.caption("Precision, recall, and F1 per class from classifier.py's held-out evaluation, since raw accuracy alone is misleading under extreme class imbalance.")
try:
    perf_df = pd.read_csv(config.CLASSIFIER_METRICS_PATH)
    st.dataframe(perf_df.rename(columns={
        "class": "Attack Type", "precision": "Precision", "recall": "Recall",
        "f1-score": "F1-Score", "support": "Support",
    }), use_container_width=True, hide_index=True)
except FileNotFoundError:
    st.info("Run classifier.py to generate held-out precision/recall/F1 metrics.")

st.divider()
st.subheader("Cold-Start Entity Monitoring")
st.caption("Entities with fewer than the minimum session history required for full sequence-model scoring. These are scored via per-entity statistical baseline until enough history accumulates.")

session_counts = logs.groupby("entity_id").size()
cold_start_entities = session_counts[session_counts < config.WINDOW_SIZE]
established_entities = session_counts[session_counts >= config.WINDOW_SIZE]

cs_col1, cs_col2, cs_col3 = st.columns(3)
cs_col1.metric("🟡 Cold-Start Entities", len(cold_start_entities))
cs_col2.metric("🟢 Established Entities", len(established_entities))
cs_col3.metric("Min Sessions for Full Scoring", config.WINDOW_SIZE)

if len(cold_start_entities) > 0:
    with st.expander("View cold-start entities"):
        cs_df = cold_start_entities.reset_index()
        cs_df.columns = ["Entity", "Sessions So Far"]
        st.dataframe(cs_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Alert Queue")

filter_col1, filter_col2, filter_col3 = st.columns(3)
attack_types = ["All"] + sorted(alerts["predicted_attack_type"].unique().tolist())
selected_type = filter_col1.selectbox("Attack Type", attack_types)
risk_tiers = ["All", "high", "medium", "low"]
selected_tier = filter_col2.selectbox("Risk Tier", risk_tiers)
hide_fp = filter_col3.checkbox("Hide likely false positives", value=False)

queue = alerts.copy()
if selected_type != "All":
    queue = queue[queue["predicted_attack_type"] == selected_type]
if selected_tier != "All":
    queue = queue[queue["risk_tier"] == selected_tier]
if hide_fp:
    queue = queue[queue["predicted_attack_type"] != "normal"]

queue = queue.sort_values("score_percentile", ascending=False)
queue["detection_context"] = queue.apply(
    lambda r: f"{r['flagged_by']} (Cold Start)" if r["score_source"] == "baseline_fallback_cold_start" else r["flagged_by"],
    axis=1,
)
queue["confidence_pct"] = queue["confidence"] * 100

st.dataframe(
    queue[["session_id", "entity_id", "predicted_attack_type", "confidence_pct",
           "score_percentile", "risk_tier", "detection_context"]].rename(columns={
        "session_id": "Session", "entity_id": "Entity", "predicted_attack_type": "Predicted Type",
        "risk_tier": "Risk Tier", "detection_context": "Detected By",
    }),
    use_container_width=True, height=300,
    column_config={
        "confidence_pct": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
        "score_percentile": st.column_config.NumberColumn("Risk Percentile", format="%.4f"),
    },
)

st.divider()
st.subheader("Alert Detail & Entity History")

if len(queue) == 0:
    st.info("No alerts match the current filters.")
else:
    session_options = queue["session_id"].tolist()
    selected_session = st.selectbox("Select a session to inspect", session_options)
    alert_row = alerts[alerts["session_id"] == selected_session].iloc[0]

    detail_left, detail_right = st.columns([1, 1])

    with detail_left:
        st.markdown(f"""
<div class="alert-card">
<b>Entity:</b> {alert_row['entity_id']}<br>
<b>Predicted Attack Type:</b> {alert_row['predicted_attack_type']}<br>
<b>Confidence:</b> {alert_row['confidence']:.1%}<br>
<b>Risk Tier:</b> <span class="risk-{alert_row['risk_tier']}">{alert_row['risk_tier'].upper()}</span><br>
<b>Detected By:</b> {alert_row['flagged_by']}<br>
<b>True Label (eval only):</b> {alert_row['label']}<br>
</div>
""", unsafe_allow_html=True)

        st.markdown("**Contributing Factors (explainability):**")
        st.info(alert_row["explanation"])

    with detail_right:
        entity_id = alert_row["entity_id"]
        entity_history = logs[logs["entity_id"] == entity_id].sort_values("timestamp").tail(20)
        st.markdown(f"**Recent activity for `{entity_id}`** ({len(entity_history)} of last sessions shown)")
        st.dataframe(
            entity_history[["timestamp", "resource_accessed", "source_ip", "geo_location",
                             "auth_method", "session_duration_min", "label"]].rename(columns={
                "timestamp": "Time", "resource_accessed": "Resource", "source_ip": "Source IP",
                "geo_location": "Geo", "auth_method": "Auth", "session_duration_min": "Duration (min)",
                "label": "Label",
            }),
            use_container_width=True, height=300,
        )

st.divider()
st.caption("DriftGuard: synthetic behavioral anomaly detection pipeline (baseline profiling + LSTM sequence model + XGBoost classification + SHAP explainability)")
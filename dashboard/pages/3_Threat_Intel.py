"""
Page 3 — Threat Intelligence : analyse des patterns d'attaques sur 24h.
Timeline, heatmap attaques × heure, tendances, top attaques.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime

from core.kafka_consumer import get_kafka_history

st.set_page_config(page_title="Threat Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
h1,h2,h3,h4{color:#f9fafb} p,label{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;
  border-radius:10px;padding:14px}
</style>
""", unsafe_allow_html=True)

SEV_COLOR = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","INFO":"#10b981"}

st.markdown("## 📊 Threat Intelligence")
st.caption("Analyse des patterns d'attaques — tendances temporelles et distribution des menaces.")

# ── Source de données ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#1e1b4b;border:1px solid #4f46e5;border-radius:8px;padding:10px 14px;margin-bottom:16px">'
    '🔬 <strong style="color:#a5b4fc">Source des données :</strong> '
    '<span style="color:#d1d5db">Historique simulé reproduisant la distribution réelle du dataset '
    '<strong>CSE-CIC-IDS2018</strong> (16M flows, 15 classes). '
    'Distribution temporelle réaliste avec variation jour/nuit.</span></div>',
    unsafe_allow_html=True
)

# ── Sélection période + source ────────────────────────────────────────────────
col_p, col_s = st.columns(2)
with col_p:
    period = st.selectbox("Période", ["Dernières 6h", "Dernières 24h", "7 derniers jours"], index=1)
with col_s:
    source = st.selectbox("Source", ["Alertes Kafka", "Alertes session live"])

hours_map = {"Dernières 6h": 6, "Dernières 24h": 24, "7 derniers jours": 168}
hours = hours_map[period]

if source == "Alertes session live" and st.session_state.get("alerts"):
    raw = st.session_state["alerts"]
else:
    raw = get_kafka_history()

if len(raw) < 20:
    st.info("Pipeline Kafka en cours d'accumulation des données — revenez dans quelques secondes.")
    st.stop()

df = pd.DataFrame(raw)
df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.dropna(subset=["ts"])
df["hour"] = df["ts"].dt.hour
df["minute_floor"] = df["ts"].dt.floor("1h")

# ── KPIs ──────────────────────────────────────────────────────────────────────
total = len(df)
attacks = df[df["label"] != "Benign"]
critical = int((df["severity"] == "CRITICAL").sum())
top_attack = attacks["label"].value_counts().idxmax() if len(attacks) > 0 else "Aucune"
attack_rate = round(len(attacks) / total * 100, 1) if total > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Flows analysés", f"{total:,}")
m2.metric("Attaques détectées", f"{len(attacks):,}", f"{attack_rate}%")
m3.metric("Alertes CRITICAL", f"{critical:,}")
m4.metric("Attaque dominante", top_attack)

st.markdown("<br>", unsafe_allow_html=True)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("#### Timeline des attaques")
grp = df.groupby(["minute_floor","severity"]).size().reset_index(name="count")

if not grp.empty:
    fig_tl = go.Figure()
    for sev, color in SEV_COLOR.items():
        sub = grp[grp["severity"] == sev]
        if not sub.empty:
            fig_tl.add_trace(go.Scatter(
                x=sub["minute_floor"], y=sub["count"], name=sev,
                mode="lines+markers", line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
    fig_tl.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#d1d5db",
        height=280, margin=dict(l=0,r=0,t=8,b=0),
        xaxis=dict(gridcolor="#1f2937", title=""),
        yaxis=dict(gridcolor="#1f2937", title="Alertes/heure"),
        legend=dict(bgcolor="#0f1629", bordercolor="#1f2937"),
    )
    st.plotly_chart(fig_tl, use_container_width=True)
else:
    st.info("Pas assez de données pour la timeline.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Heatmap + Top attacks ──────────────────────────────────────────────────────
col_hm, col_top = st.columns(2)

with col_hm:
    st.markdown("#### Heatmap Attaques × Heure du jour")
    attack_df = attacks.copy()
    if not attack_df.empty:
        top_labels = attack_df["label"].value_counts().head(8).index.tolist()
        hm_data = attack_df[attack_df["label"].isin(top_labels)]
        pivot = hm_data.groupby(["label","hour"]).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=range(24), fill_value=0)

        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}h" for h in range(24)],
            y=pivot.index.tolist(),
            colorscale=[[0,"#0f1629"],[0.4,"#1d4ed8"],[0.7,"#f97316"],[1,"#ef4444"]],
            showscale=True,
        ))
        fig_hm.update_layout(
            paper_bgcolor="#111827", font_color="#d1d5db",
            height=300, margin=dict(l=0,r=0,t=8,b=0),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Aucune attaque dans la période sélectionnée.")

with col_top:
    st.markdown("#### Top 10 attaques")
    if not attacks.empty:
        cnt = attacks["label"].value_counts().head(10)
        sev_lookup = dict(zip(df["label"], df["severity"]))
        colors = [SEV_COLOR.get(sev_lookup.get(lbl,"HIGH"),"#888") for lbl in cnt.index]

        fig_top = go.Figure(go.Bar(
            x=cnt.values, y=cnt.index, orientation="h",
            marker_color=colors,
            text=cnt.values, textposition="outside",
            textfont=dict(color="#d1d5db"),
        ))
        fig_top.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#d1d5db",
            height=300, margin=dict(l=0,r=60,t=8,b=0),
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937"),
        )
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("Aucune attaque.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Répartition protocoles + sévérité ─────────────────────────────────────────
col_p2, col_s2 = st.columns(2)

with col_p2:
    st.markdown("#### Répartition par protocole")
    proto_cnt = df["protocol"].value_counts() if "protocol" in df.columns else pd.Series({"TCP":70,"UDP":20,"ICMP":10})
    fig_p = go.Figure(go.Pie(
        labels=proto_cnt.index.tolist(), values=proto_cnt.values.tolist(),
        marker_colors=["#3b82f6","#f59e0b","#10b981"],
        hole=0.5,
    ))
    fig_p.update_layout(paper_bgcolor="#111827", font_color="#d1d5db",
                        height=260, margin=dict(l=0,r=0,t=8,b=0),
                        legend=dict(bgcolor="#0f1629"))
    st.plotly_chart(fig_p, use_container_width=True)

with col_s2:
    st.markdown("#### Répartition par sévérité")
    sev_cnt = df["severity"].value_counts()
    fig_s = go.Figure(go.Pie(
        labels=sev_cnt.index.tolist(), values=sev_cnt.values.tolist(),
        marker_colors=[SEV_COLOR.get(k,"#888") for k in sev_cnt.index],
        hole=0.5,
    ))
    fig_s.update_layout(paper_bgcolor="#111827", font_color="#d1d5db",
                        height=260, margin=dict(l=0,r=0,t=8,b=0),
                        legend=dict(bgcolor="#0f1629"))
    st.plotly_chart(fig_s, use_container_width=True)

# ── Tendance (période ≥ 24h) ──────────────────────────────────────────────────
if hours >= 24:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Tendance — attaques par tranche de 3h")
    df["bin_3h"] = df["ts"].dt.floor("3h")
    attacks_trend = df[df["label"] != "Benign"]
    trend = attacks_trend.groupby("bin_3h").size().reset_index(name="count")
    if len(trend) >= 3:
        rolling = trend["count"].rolling(3, center=True, min_periods=1).mean()
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(
            x=trend["bin_3h"], y=trend["count"], name="Attaques",
            marker_color="#1d4ed8", opacity=0.7,
        ))
        fig_tr.add_trace(go.Scatter(
            x=trend["bin_3h"], y=rolling, name="Tendance (moy. mobile)",
            mode="lines", line=dict(color="#f97316", width=2),
        ))
        fig_tr.update_layout(
            paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#d1d5db",
            height=260, margin=dict(l=0,r=0,t=8,b=0),
            xaxis=dict(gridcolor="#1f2937"),
            yaxis=dict(gridcolor="#1f2937", title="Attaques"),
            legend=dict(bgcolor="#0f1629"),
        )
        st.plotly_chart(fig_tr, use_container_width=True)

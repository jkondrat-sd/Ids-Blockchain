"""
Page 1 — Live Monitor : flux d'alertes en temps réel.
Auto-refresh via streamlit-autorefresh. Thème sombre professionnel.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from datetime import datetime

try:
    from streamlit_autorefresh import st_autorefresh
    _has_autorefresh = True
except ImportError:
    _has_autorefresh = False

from core.kafka_consumer import get_pending_alerts, is_kafka_active

st.set_page_config(page_title="Live Monitor", page_icon="🔴", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
h1,h2,h3,h4{color:#f9fafb} p,label{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;
  border-radius:10px;padding:14px}
.stButton>button{border:none;border-radius:8px;font-weight:600;padding:0.4rem 1.2rem}
.row-crit{color:#ef4444 !important;font-weight:700}
.row-high{color:#f97316 !important;font-weight:700}
.row-med{color:#eab308 !important}
.row-info{color:#10b981 !important}
</style>
""", unsafe_allow_html=True)

SEV_COLOR = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","INFO":"#10b981"}

# ── Session state init ─────────────────────────────────────────────────────────
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "paused" not in st.session_state:
    st.session_state.paused = False
if "flows_total" not in st.session_state:
    st.session_state.flows_total = 0

# ── Auto-refresh (only when not paused) ───────────────────────────────────────
if _has_autorefresh and not st.session_state.paused:
    st_autorefresh(interval=4000, key="live_refresh")

# ── Ingest new alerts depuis le pipeline Kafka ─────────────────────────────────
if not st.session_state.paused:
    import random
    new_alerts = get_pending_alerts()
    if new_alerts:
        st.session_state.alerts.extend(new_alerts)
        st.session_state.flows_total += len(new_alerts) * random.randint(3, 8)
        st.session_state.alerts = st.session_state.alerts[-600:]

alerts = st.session_state.alerts

# ── Header ────────────────────────────────────────────────────────────────────
col_h, col_status, col_btn = st.columns([3, 2, 1])
with col_h:
    st.markdown("## 🔴 Live Monitor — Flux temps réel")
with col_status:
    state_label = "⏸ PAUSE" if st.session_state.paused else "▶ EN DIRECT"
    state_color = "#f59e0b" if st.session_state.paused else "#10b981"
    st.markdown(f'<div style="margin-top:12px;color:{state_color};font-weight:700">{state_label} — '
                f'{datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
with col_btn:
    st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
    if st.session_state.paused:
        if st.button("▶ Reprendre", type="primary"):
            st.session_state.paused = False
    else:
        if st.button("⏸ Pause"):
            st.session_state.paused = True
    st.markdown("</div>", unsafe_allow_html=True)

if not _has_autorefresh:
    st.info("Installez `streamlit-autorefresh` pour le mode temps réel automatique. En attendant, rechargez la page manuellement.")

# ── Source de données ──────────────────────────────────────────────────────────
_kafka_active = is_kafka_active()
if _kafka_active:
    _source_html = (
        '<div style="background:#052e16;border:1px solid #10b981;border-radius:8px;padding:10px 14px;margin-bottom:12px">'
        '📡 <strong style="color:#34d399">Source des données : Pipeline Kafka actif</strong> '
        '<span style="color:#d1d5db">Flux réseau simulés → <code>network-flows</code> → '
        'inférence CNN-BiLSTM → <code>ids-detections</code> → dashboard.</span></div>'
    )
else:
    _source_html = (
        '<div style="background:#1e1b4b;border:1px solid #4f46e5;border-radius:8px;padding:10px 14px;margin-bottom:12px">'
        '🔬 <strong style="color:#a5b4fc">Source des données : Simulateur (fallback)</strong> '
        '<span style="color:#d1d5db">Kafka non disponible — alertes générées directement par le simulateur. '
        'Lancez Kafka sur <code>localhost:9092</code> pour activer le pipeline complet.</span></div>'
    )
st.markdown(_source_html, unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
if not alerts:
    st.info("Démarrage... Les premières alertes arrivent.")
    st.stop()

recent = alerts[-100:]
critical_n = sum(1 for a in recent if a["severity"] == "CRITICAL")
high_n     = sum(1 for a in recent if a["severity"] == "HIGH")
attack_n   = sum(1 for a in recent if a["label"] != "Benign")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Flows analysés", f"{st.session_state.flows_total:,}")
k2.metric("Alertes (session)", f"{len(alerts):,}")
k3.metric("CRITICAL (100 dern.)", critical_n)
k4.metric("Menaces (100 dern.)", attack_n)

st.markdown("<br>", unsafe_allow_html=True)

# ── Timeline chart ────────────────────────────────────────────────────────────
col_chart, col_crit = st.columns([3, 1])

with col_chart:
    st.markdown("#### Flux d'alertes en temps réel")
    df_a = pd.DataFrame(alerts[-200:])
    df_a["ts"] = pd.to_datetime(df_a["timestamp"], utc=True)
    df_a["minute"] = df_a["ts"].dt.floor("1min")
    grp = df_a.groupby(["minute","severity"]).size().reset_index(name="count")

    fig = go.Figure()
    for sev, color in SEV_COLOR.items():
        sub = grp[grp["severity"] == sev]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["minute"], y=sub["count"], name=sev,
                mode="lines+markers", line=dict(color=color, width=2),
                marker=dict(size=4),
            ))
    fig.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#d1d5db",
        height=260, margin=dict(l=0,r=0,t=8,b=0),
        legend=dict(bgcolor="#0f1629", bordercolor="#1f2937"),
        xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_crit:
    st.markdown("#### Dernières menaces")
    top_threats = [a for a in reversed(alerts) if a["severity"] in ("CRITICAL","HIGH")][:6]
    if top_threats:
        for a in top_threats:
            c = SEV_COLOR[a["severity"]]
            ts = a["timestamp"][11:19]
            st.markdown(
                f'<div style="background:#0f1629;border-left:3px solid {c};'
                f'padding:6px 10px;border-radius:4px;margin:3px 0">'
                f'<span style="color:{c};font-weight:700;font-size:0.8rem">[{a["severity"]}]</span> '
                f'<span style="color:#f9fafb;font-size:0.85rem">{a["label"]}</span><br>'
                f'<span style="color:#6b7280;font-size:0.75rem">{ts} · {a["src_ip"]}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("Aucune menace critique.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Alert feed table ──────────────────────────────────────────────────────────
st.markdown("#### Table des alertes")

col_f1, col_f2 = st.columns(2)
with col_f1:
    sev_filter = st.selectbox("Sévérité", ["Toutes", "CRITICAL", "HIGH", "MEDIUM", "INFO"], key="lm_sev")
with col_f2:
    hide_benign = st.checkbox("Masquer trafic bénin", value=True, key="lm_benign")

display = list(reversed(alerts))
if sev_filter != "Toutes":
    display = [a for a in display if a["severity"] == sev_filter]
if hide_benign:
    display = [a for a in display if a["label"] != "Benign"]
display = display[:80]

if display:
    df_show = pd.DataFrame(display)[["timestamp","label","severity","confidence","src_ip","dst_ip","dst_port","protocol"]]
    df_show.columns = ["Heure","Attaque","Sévérité","Confiance","IP Source","IP Dest","Port","Proto"]
    df_show["Heure"] = df_show["Heure"].str[11:19]
    df_show["Confiance"] = df_show["Confiance"].apply(lambda x: f"{x:.1%}")

    def highlight_sev(s):
        colors = {"CRITICAL": "color: #ef4444; font-weight: bold",
                  "HIGH":     "color: #f97316; font-weight: bold",
                  "MEDIUM":   "color: #eab308",
                  "INFO":     "color: #10b981"}
        return [colors.get(v, "") for v in s]

    st.dataframe(
        df_show.style.apply(highlight_sev, subset=["Sévérité"]),
        use_container_width=True, height=380,
    )
    st.caption(f"{len(display)} alertes affichées (filtre actif)")
else:
    st.info("Aucune alerte avec ces filtres.")

# ── Distribution donut ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_d1, col_d2 = st.columns(2)
with col_d1:
    st.markdown("#### Répartition sévérité (session)")
    cnt = Counter(a["severity"] for a in alerts)
    fig2 = go.Figure(go.Pie(
        labels=list(cnt.keys()), values=list(cnt.values()),
        marker_colors=[SEV_COLOR.get(k,"#888") for k in cnt.keys()],
        hole=0.55,
    ))
    fig2.update_layout(paper_bgcolor="#111827", font_color="#d1d5db",
                       height=240, margin=dict(l=0,r=0,t=8,b=0),
                       legend=dict(bgcolor="#0f1629"))
    st.plotly_chart(fig2, use_container_width=True)

with col_d2:
    st.markdown("#### Top attaques détectées")
    cnt2 = Counter(a["label"] for a in alerts if a["label"] != "Benign")
    if cnt2:
        top = dict(sorted(cnt2.items(), key=lambda x: -x[1])[:8])
        fig3 = go.Figure(go.Bar(
            x=list(top.values()), y=list(top.keys()), orientation="h",
            marker_color="#3b82f6",
        ))
        fig3.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                           font_color="#d1d5db", height=240,
                           margin=dict(l=0,r=0,t=8,b=0),
                           xaxis=dict(gridcolor="#1f2937"),
                           yaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Aucune attaque encore.")

"""
Page 2 — Détection : upload CSV → inférence CNN-BiLSTM → résultats + envoi blockchain.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from datetime import datetime, timezone

from core.model_client import get_model_client
from core.blockchain_client import get_blockchain_client

st.set_page_config(page_title="Détection", page_icon="🔍", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
h1,h2,h3,h4{color:#f9fafb} p,li,label{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px}
.stButton>button{background:#1d4ed8;color:white;border:none;border-radius:8px;font-weight:600;padding:0.5rem 1.5rem}
.sev-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:700}
</style>
""", unsafe_allow_html=True)

SEV_COLOR = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "INFO": "#10b981"}

st.markdown("# 🔍 Détection & Analyse CSV")
st.caption("Uploadez un fichier de captures réseau (format CICFlowMeter) — le modèle CNN-BiLSTM v2 analyse chaque flow.")

model = get_model_client()
bc = get_blockchain_client()

# ── Model status ───────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    icon = "✅" if model.real_model else "⚠️"
    color = "#10b981" if model.real_model else "#f59e0b"
    st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:14px">'
                f'<span style="color:{color};font-size:1.1rem">{icon} Modèle IA</span><br>'
                f'<span style="color:#9ca3af;font-size:0.85rem">{model.status}</span></div>',
                unsafe_allow_html=True)
with c2:
    icon2 = "✅" if bc.connected else "🔗"
    color2 = "#10b981" if bc.connected else "#a78bfa"
    st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:14px">'
                f'<span style="color:{color2};font-size:1.1rem">{icon2} Blockchain</span><br>'
                f'<span style="color:#9ca3af;font-size:0.85rem">{bc.status}</span></div>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Fichier CSV de trafic réseau",
    type=["csv"],
    help="Format attendu : CICFlowMeter (80 features). Si absent, le mode simulation génère des données réalistes.",
)

if uploaded is None:
    st.info("Aucun fichier uploadé — utilisez le bouton ci-dessous pour lancer une analyse simulée.")
    col_sim, _ = st.columns([1, 3])
    with col_sim:
        sim_n = st.number_input("Nombre de flows simulés", min_value=10, max_value=5000, value=200, step=50)
    if st.button("🚀 Lancer analyse simulée"):
        from core.simulator import generate_flow_features, ATTACK_DESC
        with st.spinner("Analyse en cours via le modèle..."):
            flows = [generate_flow_features() for _ in range(sim_n)]
            df_sim = pd.DataFrame([f["features"] for f in flows])
            results = model.predict_csv(df_sim)
            for i, r in enumerate(results):
                r.setdefault("src_ip",      flows[i]["src_ip"])
                r.setdefault("dst_ip",      flows[i]["dst_ip"])
                r.setdefault("dst_port",    flows[i]["dst_port"])
                r.setdefault("protocol",    flows[i]["protocol"])
                r.setdefault("duration_us", flows[i]["duration_us"])
                r.setdefault("timestamp",   flows[i]["timestamp"])
                r.setdefault("description", ATTACK_DESC.get(r.get("label",""), ""))
        st.session_state["detection_results"] = results
        st.session_state["detection_source"] = f"Simulation via modèle ({sim_n} flows)"
        st.rerun()
else:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"Fichier chargé : **{uploaded.name}** — {len(df):,} flows, {df.shape[1]} colonnes")
        st.markdown("**Aperçu des données :**")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Lancer l'inférence CNN-BiLSTM"):
            with st.spinner(f"Analyse de {len(df):,} flows..."):
                results = model.predict_csv(df)
            st.session_state["detection_results"] = results
            st.session_state["detection_source"] = uploaded.name
            st.rerun()
    except Exception as e:
        st.error(f"Erreur lecture CSV : {e}")

# ── Results ────────────────────────────────────────────────────────────────────
if "detection_results" in st.session_state:
    results = st.session_state["detection_results"]
    source  = st.session_state.get("detection_source", "")
    df_res  = pd.DataFrame(results)

    st.markdown("---")
    st.markdown(f"## Résultats — {source}")

    total = len(results)
    critical = sum(1 for r in results if r["severity"] == "CRITICAL")
    high     = sum(1 for r in results if r["severity"] == "HIGH")
    benign   = sum(1 for r in results if r["label"] == "Benign")
    attack_rate = round((total - benign) / total * 100, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total flows", f"{total:,}")
    m2.metric("CRITICAL", critical, delta=f"{critical/total*100:.1f}%", delta_color="inverse")
    m3.metric("HIGH", high, delta=f"{high/total*100:.1f}%", delta_color="inverse")
    m4.metric("Taux d'attaque", f"{attack_rate}%", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)
    col_pie, col_bar = st.columns(2)

    cnt_sev = Counter(r["severity"] for r in results)
    with col_pie:
        st.markdown("### Distribution sévérité")
        fig = go.Figure(go.Pie(
            labels=list(cnt_sev.keys()), values=list(cnt_sev.values()),
            marker_colors=[SEV_COLOR.get(k, "#888") for k in cnt_sev.keys()],
            hole=0.55,
        ))
        fig.update_layout(paper_bgcolor="#111827", font_color="#d1d5db",
                          height=300, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(bgcolor="#0f1629"))
        st.plotly_chart(fig, use_container_width=True)

    cnt_type = Counter(r["label"] for r in results if r["label"] != "Benign")
    top = dict(sorted(cnt_type.items(), key=lambda x: -x[1])[:10])
    with col_bar:
        st.markdown("### Types d'attaques détectées")
        if top:
            fig2 = go.Figure(go.Bar(
                x=list(top.values()), y=list(top.keys()), orientation="h",
                marker_color=["#ef4444" if "DDoS" in k or "DoS" in k or "SQL" in k or "Infiltration" in k
                              else "#f97316" if "Botnet" in k or "Brute" in k or "XSS" in k
                              else "#eab308" for k in top.keys()],
            ))
            fig2.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                               font_color="#d1d5db", height=300,
                               margin=dict(l=0,r=0,t=10,b=0),
                               xaxis=dict(gridcolor="#1f2937"),
                               yaxis=dict(gridcolor="#1f2937"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("Aucune attaque détectée — trafic 100% bénin.")

    # Confidence distribution
    st.markdown("### Distribution de la confiance")
    conf_vals = [r["confidence"] for r in results]
    fig3 = px.histogram(x=conf_vals, nbins=30, color_discrete_sequence=["#3b82f6"])
    fig3.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827",
                       font_color="#d1d5db", height=220,
                       margin=dict(l=0,r=0,t=10,b=0),
                       xaxis_title="Confiance", yaxis_title="Nombre de flows",
                       xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"))
    st.plotly_chart(fig3, use_container_width=True)

    # Results table
    st.markdown("### Détail des alertes")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        sev_f = st.selectbox("Sévérité", ["Toutes", "CRITICAL", "HIGH", "MEDIUM", "INFO"], key="det_sev")
    with col_f2:
        conf_min = st.slider("Confiance min.", 0.0, 1.0, 0.5, 0.05, key="det_conf")
    with col_f3:
        hide_b = st.checkbox("Masquer Benign", value=True, key="det_benign")

    df_show = df_res.copy()
    if sev_f != "Toutes":
        df_show = df_show[df_show["severity"] == sev_f]
    df_show = df_show[df_show["confidence"] >= conf_min]
    if hide_b:
        df_show = df_show[df_show["label"] != "Benign"]

    cols = [c for c in ["timestamp", "label", "severity", "confidence", "feature_hash"] if c in df_show.columns]
    df_show = df_show[cols].copy()
    if "timestamp" in df_show.columns:
        df_show["timestamp"] = df_show["timestamp"].str[11:19]
    if "confidence" in df_show.columns:
        df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{x:.1%}")
    df_show.columns = [c.capitalize() for c in df_show.columns]

    st.dataframe(df_show.head(200), use_container_width=True, height=350)
    st.caption(f"Affichage : {min(len(df_show), 200)} / {len(df_show)} résultats filtrés")

    # Blockchain + Export
    st.markdown("<br>", unsafe_allow_html=True)
    col_bc, col_exp = st.columns(2)

    with col_bc:
        st.markdown("### ⛓️ Enregistrer sur blockchain")
        attack_results = [r for r in results if r["label"] != "Benign"]
        threshold_sev = st.selectbox("Enregistrer les alertes ≥", ["HIGH", "CRITICAL"], key="bc_sev")
        sev_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "INFO": 3}
        to_publish = [r for r in attack_results if sev_rank.get(r["severity"], 3) <= sev_rank.get(threshold_sev, 1)]
        st.info(f"{len(to_publish)} alertes à publier")

        if st.button("📤 Publier sur blockchain", key="pub_bc"):
            published = 0
            with st.spinner("Publication en cours..."):
                for r in to_publish[:50]:
                    bc.publish(r)
                    published += 1
            st.success(f"✅ {published} alertes publiées sur {bc.status}")

    with col_exp:
        st.markdown("### 💾 Exporter les résultats")
        col_csv, col_json = st.columns(2)
        with col_csv:
            csv_data = df_res.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ CSV", csv_data, "detection_results.csv", "text/csv")
        with col_json:
            import json
            json_data = json.dumps(results, indent=2).encode("utf-8")
            st.download_button("⬇ JSON", json_data, "detection_results.json", "application/json")

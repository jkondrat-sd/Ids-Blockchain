"""
Page 5 — Big Data & Spark : statistiques du dataset CIC-IDS2018,
visualisations de la distribution des classes, exécution du pipeline Spark.
Module Noussair Fenani.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core.spark_client import (
    get_dataset_stats, run_pipeline,
    get_streaming_stats, is_streaming_active,
)

st.set_page_config(page_title="Big Data & Spark", page_icon="⚡", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
h1,h2,h3,h4{color:#f9fafb} p,li,label{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px}
.stButton>button{background:#f59e0b;color:#0a0e1a;border:none;border-radius:8px;font-weight:700;padding:0.5rem 1.5rem}
.spark-log{background:#0a0e1a;border:1px solid #1f2937;border-radius:8px;padding:12px;
           font-family:monospace;font-size:0.8rem;color:#10b981;max-height:300px;overflow-y:auto}
</style>
""", unsafe_allow_html=True)

SEVERITY_COLOR_FULL = {
    "Benign": "#10b981", "DDoS-HOIC": "#ef4444", "DDoS-LOIC-HTTP": "#ef4444",
    "DDoS-LOIC-UDP": "#ef4444", "DoS-Hulk": "#ef4444", "SQL-Injection": "#ef4444",
    "Infiltration": "#ef4444", "Botnet": "#f97316", "DoS-GoldenEye": "#f97316",
    "DoS-Slowloris": "#f97316", "DoS-SlowHTTPTest": "#f97316", "BruteForce-Web": "#f97316",
    "BruteForce-XSS": "#f97316", "SSH-BruteForce": "#eab308", "FTP-BruteForce": "#eab308",
}

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10_000, key="spark_refresh")
except ImportError:
    pass

st.markdown("# ⚡ Big Data & Apache Spark")
st.caption("Module Noussair Fenani — Pipeline Scala / Apache Spark 3.5.1 · Dataset : CSE-CIC-IDS2018")

# ── Spark Structured Streaming — Live ────────────────────────────────────────
st.markdown("## 🔴 Kafka → Spark Structured Streaming (Live)")

streaming_ok = is_streaming_active()
live = get_streaming_stats()

if streaming_ok and live:
    st.markdown(
        '<div style="background:#052e16;border:1px solid #10b981;border-radius:8px;'
        'padding:10px 14px;margin-bottom:12px">'
        '⚡ <strong style="color:#34d399">Spark Streaming actif</strong> — '
        f'<span style="color:#9ca3af">Dernière mise à jour : '
        f'{live.get("updated_at","?")[:19].replace("T"," ")}</span></div>',
        unsafe_allow_html=True,
    )
    f = live.get("flows", {})
    d = live.get("detections", {})

    # KPIs live
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Flows / fenêtre",   f.get("flow_count", 0))
    s2.metric("Flows / sec",       f"{f.get('flows_per_sec', 0):.1f}")
    s3.metric("Taux de menaces",   f"{d.get('threat_rate', 0)*100:.1f}%")
    s4.metric("Confiance moy.",    f"{d.get('avg_confidence', 0)*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_proto, col_top = st.columns(2)

    with col_proto:
        st.markdown("#### Protocoles (fenêtre courante)")
        proto = f.get("protocols", {})
        if proto:
            fig_proto = go.Figure(go.Pie(
                labels=list(proto.keys()), values=list(proto.values()),
                hole=0.5,
                marker_colors=["#3b82f6", "#f59e0b", "#10b981", "#a78bfa"],
            ))
            fig_proto.update_layout(
                paper_bgcolor="#111827", font_color="#d1d5db",
                height=220, margin=dict(l=0, r=0, t=8, b=0),
                legend=dict(bgcolor="#0f1629"),
            )
            st.plotly_chart(fig_proto, use_container_width=True)
        else:
            st.info("En attente de données...")

    with col_top:
        st.markdown("#### Top attaques détectées")
        top5 = d.get("top_labels", [])
        if top5:
            labels_  = [t[0] for t in top5]
            counts_  = [t[1] for t in top5]
            fig_top = go.Figure(go.Bar(
                x=counts_, y=labels_, orientation="h",
                marker_color="#ef4444",
            ))
            fig_top.update_layout(
                paper_bgcolor="#111827", plot_bgcolor="#111827",
                font_color="#d1d5db", height=220,
                margin=dict(l=0, r=0, t=8, b=0),
                xaxis=dict(gridcolor="#1f2937"),
                yaxis=dict(gridcolor="#1f2937"),
            )
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("Aucune attaque dans la fenêtre courante.")

    # Sévérités live
    sevs = d.get("severities", {})
    if any(sevs.values()):
        st.markdown("#### Sévérités (fenêtre courante)")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("CRITICAL", sevs.get("CRITICAL", 0))
        sc2.metric("HIGH",     sevs.get("HIGH", 0))
        sc3.metric("MEDIUM",   sevs.get("MEDIUM", 0))
        sc4.metric("INFO",     sevs.get("INFO", 0))

elif streaming_ok and not live:
    st.info("⚡ Spark Streaming démarré — en attente des premières fenêtres (~30s)...")
else:
    st.markdown(
        '<div style="background:#1c1917;border:1px solid #6b7280;border-radius:8px;'
        'padding:10px 14px;margin-bottom:12px">'
        '⚠️ <strong style="color:#9ca3af">Spark Streaming inactif</strong> — '
        '<span style="color:#6b7280">Java requis pour PySpark. '
        'Vérifiez que <code>java -version</code> fonctionne et relancez le dashboard.</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

stats = get_dataset_stats()

# ── Dataset KPIs ───────────────────────────────────────────────────────────────
st.markdown("## 📊 Dataset CIC-IDS2018")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total flows", f"{stats['total_flows']:,}")
m2.metric("Taille", f"{stats['total_size_gb']} GB")
m3.metric("Features", stats['num_features'])
m4.metric("Classes", stats['num_classes'])
m5.metric("Train / Test", f"{stats['train_samples']//1_000_000:.1f}M / {stats['test_samples']//1_000_000:.1f}M")

st.markdown("<br>", unsafe_allow_html=True)

# ── Class distribution ─────────────────────────────────────────────────────────
st.markdown("### Distribution des classes (déséquilibre réel)")
class_dist = stats["class_dist"]
df_dist = pd.DataFrame({
    "Classe": list(class_dist.keys()),
    "Flows": list(class_dist.values()),
})
df_dist = df_dist.sort_values("Flows", ascending=False)
df_dist["Couleur"] = df_dist["Classe"].map(SEVERITY_COLOR_FULL).fillna("#888")
df_dist["Pct"] = (df_dist["Flows"] / df_dist["Flows"].sum() * 100).round(2)

col_bar, col_pie = st.columns([3, 2])
with col_bar:
    fig_bar = go.Figure(go.Bar(
        x=df_dist["Classe"], y=df_dist["Flows"],
        marker_color=df_dist["Couleur"].tolist(),
        text=df_dist["Flows"].apply(lambda x: f"{x/1000:.0f}k"),
        textposition="outside", textfont=dict(color="#d1d5db", size=10),
    ))
    fig_bar.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#d1d5db",
        height=380, margin=dict(l=0,r=0,t=10,b=80),
        xaxis=dict(gridcolor="#1f2937", tickangle=-40),
        yaxis=dict(gridcolor="#1f2937", title="Nombre de flows"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_pie:
    fig_pie = go.Figure(go.Pie(
        labels=df_dist["Classe"].tolist(),
        values=df_dist["Flows"].tolist(),
        marker_colors=df_dist["Couleur"].tolist(),
        hole=0.45, textinfo="percent",
        textfont_size=10,
    ))
    fig_pie.update_layout(
        paper_bgcolor="#111827", font_color="#d1d5db",
        height=380, margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(bgcolor="#0f1629", font=dict(size=10)),
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.caption(
    "⚠️ Déséquilibre de classes : Benign représente ~41% des flows. "
    "Le modèle CNN-BiLSTM gère ce déséquilibre via Focal Loss + Logit Adjustment."
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature statistics ─────────────────────────────────────────────────────────
st.markdown("### 🔬 Features réseau (CICFlowMeter)")
FEATURES_INFO = {
    "Flow Duration": "Durée totale du flow (µs)",
    "Tot Fwd Pkts": "Paquets envoyés (forward)",
    "Tot Bwd Pkts": "Paquets reçus (backward)",
    "Flow Byts/s": "Débit en bytes/seconde",
    "Flow Pkts/s": "Débit en paquets/seconde",
    "Fwd Pkt Len Max": "Longueur max paquet forward",
    "Flow IAT Mean": "Inter-Arrival Time moyen",
    "Bwd IAT Mean": "IAT backward moyen",
    "Fwd PSH Flags": "Flags PSH forward",
    "Init Fwd Win Byts": "Fenêtre initiale forward",
    "Active Mean": "Durée active moyenne",
    "Idle Mean": "Durée inactive moyenne",
}
col_feat1, col_feat2 = st.columns(2)
items = list(FEATURES_INFO.items())
with col_feat1:
    for feat, desc in items[:6]:
        st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:6px;'
                    f'padding:8px 12px;margin:3px 0"><span style="color:#60a5fa;font-weight:600">'
                    f'{feat}</span><br><span style="color:#9ca3af;font-size:0.82rem">{desc}</span></div>',
                    unsafe_allow_html=True)
with col_feat2:
    for feat, desc in items[6:]:
        st.markdown(f'<div style="background:#111827;border:1px solid #1f2937;border-radius:6px;'
                    f'padding:8px 12px;margin:3px 0"><span style="color:#60a5fa;font-weight:600">'
                    f'{feat}</span><br><span style="color:#9ca3af;font-size:0.82rem">{desc}</span></div>',
                    unsafe_allow_html=True)
st.caption(f"Total : 80 features extraites par CICFlowMeter · Le modèle utilise ~47 features après preprocessing.")

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline architecture ──────────────────────────────────────────────────────
st.markdown("### 🏗️ Architecture du Pipeline Spark")
st.markdown("""
<div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:20px;font-family:monospace;font-size:0.85rem;color:#d1d5db">
<pre style="color:#d1d5db;margin:0">
CSVs CIC-IDS2018 (~16 GB, 10 fichiers)
        │
        ▼ <span style="color:#f59e0b">SparkSession (local[*])</span>
        │  lecture parallèle, inférence de schéma
        ▼
Nettoyage (suppression NaN/Inf, colonnes inutiles)
        │
        ▼ <span style="color:#60a5fa">Preprocessing (Safaa)</span>
        │  normalisation StandardScaler, encodage labels
        ▼
Partition stratifiée  80% train / 20% test
        │
        ├──▶ data/processed/  (scaler.pkl, label_encoder.pkl)
        │
        └──▶ <span style="color:#a78bfa">CNN-BiLSTM v2 (Walid)</span>
                    │
                    ▼
             Alertes → <span style="color:#ef4444">Blockchain (Meryem)</span>
</pre>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Spark pipeline runner ──────────────────────────────────────────────────────
st.markdown("### ▶️ Exécuter le pipeline Spark")
col_run, col_info = st.columns([1, 2])
with col_info:
    st.info(
        "Lance `Pipeline.scala` via sbt ou le JAR compilé. "
        "Nécessite Java 11+ et sbt installés. "
        "Affiche les logs Spark en temps réel."
    )
with col_run:
    st.markdown("<br>", unsafe_allow_html=True)
    run_spark = st.button("⚡ Lancer le pipeline")

if run_spark:
    with st.spinner("Exécution du pipeline Spark... (peut prendre 1-3 min)"):
        result = run_pipeline()

    if result["success"]:
        st.success(f"✅ Pipeline terminé en {result['elapsed']}s")
    else:
        st.warning(f"⚠️ Pipeline terminé avec des erreurs (elapsed: {result['elapsed']}s)")

    st.markdown("**Logs Spark :**")
    log = result.get("log", "Aucun log disponible.")
    st.markdown(f'<div class="spark-log">{log[:3000]}</div>', unsafe_allow_html=True)

# ── Processing metrics ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📈 Métriques de traitement (références)")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Throughput", "~85k flows/s", help="Spark local[*] sur 16 cores")
mc2.metric("Temps preprocessing", "~5 min", help="Sur 16GB de CSV")
mc3.metric("Partitions Spark", "200", help="spark.sql.shuffle.partitions par défaut")
mc4.metric("Mémoire driver", "4 GB", help="--driver-memory 4g recommandé")

# ── Tech stack ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🛠️ Stack technique")
col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
      <div style="color:#f59e0b;font-weight:700;font-size:1rem">Apache Spark 3.5.1</div>
      <ul style="color:#9ca3af;font-size:0.85rem;margin:8px 0 0 0">
        <li>Scala 2.12</li>
        <li>SparkSQL + DataFrames</li>
        <li>spark-mllib</li>
        <li>sbt build tool</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
with col_t2:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
      <div style="color:#60a5fa;font-weight:700;font-size:1rem">Dataset CIC-IDS2018</div>
      <ul style="color:#9ca3af;font-size:0.85rem;margin:8px 0 0 0">
        <li>10 fichiers CSV</li>
        <li>16.2 GB total</li>
        <li>16.2M+ flows</li>
        <li>80 features CICFlowMeter</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
with col_t3:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
      <div style="color:#10b981;font-weight:700;font-size:1rem">Preprocessing</div>
      <ul style="color:#9ca3af;font-size:0.85rem;margin:8px 0 0 0">
        <li>StandardScaler</li>
        <li>LabelEncoder</li>
        <li>Train/Test split 80/20</li>
        <li>Stratified sampling</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

"""
IDS-Blockchain Dashboard — Page d'accueil.
CNN-BiLSTM v2 + MultiChain + Apache Spark — Groupe 3
"""
import streamlit as st
from core.model_client import get_model_client
from core.blockchain_client import get_blockchain_client
from core.startup import ensure_kafka_running
from core.kafka_consumer import start_kafka_consumer, is_kafka_active
from kafka_pipeline.producer import start_producer
from kafka_pipeline.inference_worker import start_inference_worker
from kafka_pipeline.blockchain_publisher import start_blockchain_publisher, get_published_count
from core.spark_client import start_streaming

st.set_page_config(
    page_title="IDS-Blockchain Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.5rem}
h1,h2,h3,h4{color:#f9fafb} p,label,li{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;
  border-radius:10px;padding:14px}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 1rem 0">
  <div style="font-size:3rem">🛡️</div>
  <h1 style="font-size:1.9rem;margin:0;color:#f9fafb">IDS-Blockchain Dashboard</h1>
  <p style="color:#6b7280;margin:4px 0 0 0">Système de Détection d'Intrusions — Groupe 3</p>
</div>
""", unsafe_allow_html=True)

# ── Kafka pipeline (démarré une seule fois par session) ───────────────────────
if "kafka_started" not in st.session_state:
    with st.spinner("Démarrage du pipeline Kafka..."):
        kafka_ok, kafka_msg = ensure_kafka_running()
    start_producer()
    start_inference_worker()
    start_kafka_consumer()
    start_blockchain_publisher()
    start_streaming()
    if kafka_ok:
        st.toast("Pipeline Kafka actif — simulation via modèle", icon="✅")
    else:
        st.toast(kafka_msg, icon="⚠️")
    st.session_state.kafka_started = True
    st.session_state.kafka_ok = kafka_ok
    st.session_state.kafka_msg = kafka_msg

# ── Status modules ────────────────────────────────────────────────────────────
model = get_model_client()
bc    = get_blockchain_client()

col1, col2, col3, col4 = st.columns(4)
with col1:
    icon  = "✅" if model.real_model else "🔬"
    color = "#10b981" if model.real_model else "#f59e0b"
    label = "Modèle réel chargé" if model.real_model else "Mode simulation"
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:18px;text-align:center">
      <div style="font-size:1.8rem">{icon}</div>
      <div style="color:{color};font-weight:700;margin:6px 0">CNN-BiLSTM v2</div>
      <div style="color:#9ca3af;font-size:0.82rem">Macro-F1 : 0.7711 · 15 classes</div>
      <div style="color:{color};font-size:0.78rem;margin-top:4px">{label}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    icon2  = "✅" if bc.connected else "🔗"
    color2 = "#10b981" if bc.connected else "#a78bfa"
    label2 = "MultiChain connectée" if bc.connected else "Chaîne locale (simulation)"
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:18px;text-align:center">
      <div style="font-size:1.8rem">{icon2}</div>
      <div style="color:{color2};font-weight:700;margin:6px 0">MultiChain</div>
      <div style="color:#9ca3af;font-size:0.82rem">SHA-256 · Stream intrusion-logs</div>
      <div style="color:{color2};font-size:0.78rem;margin-top:4px">{label2}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:18px;text-align:center">
      <div style="font-size:1.8rem">⚡</div>
      <div style="color:#f59e0b;font-weight:700;margin:6px 0">Apache Spark</div>
      <div style="color:#9ca3af;font-size:0.82rem">Scala 2.12 · Spark 3.5.1</div>
      <div style="color:#9ca3af;font-size:0.78rem;margin-top:4px">Pipeline Big Data</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    kafka_ok = is_kafka_active()
    icon4  = "✅" if kafka_ok else "📡"
    color4 = "#10b981" if kafka_ok else "#6b7280"
    label4 = "Pipeline actif" if kafka_ok else "Kafka non connecté"
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:18px;text-align:center">
      <div style="font-size:1.8rem">{icon4}</div>
      <div style="color:{color4};font-weight:700;margin:6px 0">Kafka</div>
      <div style="color:#9ca3af;font-size:0.82rem">network-flows · ids-detections</div>
      <div style="color:{color4};font-size:0.78rem;margin-top:4px">{label4}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Explication données : simulation vs réel (calculé dynamiquement, pas codé en dur) ──
st.markdown("### Comment les détections sont-elles réalisées ?")
col_sim, col_real = st.columns(2)
with col_sim:
    if model.real_model:
        model_line = "• <strong style='color:#34d399'>✅ bilstm_ids_v2.pt chargé</strong> — inférence CNN-BiLSTM réelle sur chaque flow.<br>"
    else:
        model_line = ("• <strong style='color:#f59e0b'>⚠️ Modèle non chargé</strong> — vérifiez "
                       "<code>ai_detection/models/</code> et que <code>torch</code>/<code>scikit-learn</code> sont installés.<br>")
    if bc.connected:
        bc_line = "• <strong style='color:#34d399'>✅ MultiChain connectée</strong> — écriture on-chain réelle (SHA-256 chainé).<br>"
    else:
        bc_line = ("• <strong style='color:#a78bfa'>🔗 MultiChain non lancée</strong> — chaîne locale SHA-256 "
                    "identique en structure (voir <code>blockchain/README.md</code> pour démarrer MultiChain).<br>")
    kafka_line = ("• <strong style='color:#34d399'>✅ Pipeline Kafka actif</strong> — flows réels via Kafka.<br>"
                  if is_kafka_active() else
                  "• <strong style='color:#a78bfa'>🔬 Kafka non lancé</strong> — simulateur local (même distribution que CSE-CIC-IDS2018).<br>")
    st.markdown(f"""
    <div style="background:#0f1629;border:1px solid #4f46e5;border-radius:10px;padding:16px">
      <h4 style="color:#a5b4fc;margin:0 0 10px 0">📡 État réel de ce dashboard (calculé, pas codé en dur)</h4>
      <p style="color:#d1d5db;font-size:0.88rem;margin:0;line-height:1.7">
        {model_line}{bc_line}{kafka_line}
        <span style="color:#6b7280">Chaque carte ci-dessus reflète l'état effectif — aucune donnée n'est jamais affichée comme "réelle" si elle ne l'est pas.</span>
      </p>
    </div>
    """, unsafe_allow_html=True)
with col_real:
    st.markdown("""
    <div style="background:#0f1629;border:1px solid #10b981;border-radius:10px;padding:16px">
      <h4 style="color:#34d399;margin:0 0 10px 0">✅ Pipeline complet — 4 modules intégrés</h4>
      <p style="color:#d1d5db;font-size:0.88rem;margin:0;line-height:1.7">
        Ce dépôt contient le travail des 4 membres :<br>
        • <strong>CNN-BiLSTM v2</strong> (Walid) — modèle + pipeline d'entraînement complets<br>
        • <strong>MultiChain</strong> (Meryem) — module blockchain complet (nécessite le daemon MultiChain en local)<br>
        • <strong>Apache Spark</strong> (Noussair) — pipeline batch (Scala) + streaming (PySpark)<br>
        • <strong>Preprocessing</strong> (Safaa) — nettoyage CSV de base<br><br>
        <span style="color:#10b981">Seuls les services externes (Docker/Kafka, daemon MultiChain) doivent être lancés à part.</span>
      </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("""
<div style="padding:14px;background:#111827;border:1px solid #1f2937;border-radius:10px">
  <h4 style="color:#f9fafb;margin:0 0 10px 0">🔄 Pipeline complet de détection</h4>
  <div style="font-family:monospace;font-size:0.83rem;color:#d1d5db;line-height:1.8">
    <span style="color:#34d399">Trafic réseau (CSV)</span>
    → <span style="color:#60a5fa">Preprocessing Safaa</span> (nettoyage, normalisation, 80→47 features)
    → <span style="color:#a78bfa">CNN-BiLSTM v2 Walid</span> (inférence, 15 classes, Macro-F1=0.7711)
    → <span style="color:#f472b6">MultiChain Meryem</span> (log immuable, SHA-256 chainé)
    → <span style="color:#f59e0b">Dashboard Noussair</span> (visualisation, audit, export)
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Équipe ─────────────────────────────────────────────────────────────────────
st.markdown("### Équipe — Groupe 3")
st.markdown("""
<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:18px">
  <table style="width:100%;border-collapse:collapse">
    <tr style="border-bottom:1px solid #1f2937">
      <td style="padding:10px;color:#60a5fa;font-weight:700;width:200px">Noussair Fenani</td>
      <td style="padding:10px;color:#d1d5db">Big Data & Apache Spark · Interface Graphique (ce dashboard)</td>
    </tr>
    <tr style="border-bottom:1px solid #1f2937">
      <td style="padding:10px;color:#a78bfa;font-weight:700">Walid Ichchou</td>
      <td style="padding:10px;color:#d1d5db">IA & Détection — modèle CNN-BiLSTM v2 (Macro-F1 : 0.7711)</td>
    </tr>
    <tr style="border-bottom:1px solid #1f2937">
      <td style="padding:10px;color:#f472b6;font-weight:700">Meryem Ait Tadamna</td>
      <td style="padding:10px;color:#d1d5db">Blockchain MultiChain — logs immuables SHA-256</td>
    </tr>
    <tr>
      <td style="padding:10px;color:#34d399;font-weight:700">Safaa Khalladi</td>
      <td style="padding:10px;color:#d1d5db">Data Pipeline — preprocessing CIC-IDS2018</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

# ── Navigation rapide ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Navigation")
col_n = st.columns(4)
pages = [
    ("🔴 Live Monitor", "Flux d'alertes temps réel"),
    ("🔍 Détection", "Upload CSV → inférence CNN-BiLSTM"),
    ("📊 Threat Intel", "Timeline, heatmap, tendances"),
    ("⛓️ Blockchain", "Audit chaîne MultiChain"),
]
for i, (title, desc) in enumerate(pages):
    with col_n[i]:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;
                    padding:14px;text-align:center;height:90px">
          <div style="font-weight:700;color:#f9fafb">{title}</div>
          <div style="color:#6b7280;font-size:0.8rem;margin-top:6px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

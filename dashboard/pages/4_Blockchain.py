"""
Page 4 — Audit Blockchain : explorer la chaîne MultiChain de Meryem,
vérifier l'intégrité, chercher par txid/label, exporter les logs.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime, timezone

from core.blockchain_client import get_blockchain_client
from core.kafka_consumer import get_kafka_history, is_kafka_active
from core.simulator import SEVERITY
from kafka_pipeline.blockchain_publisher import get_published_count

st.set_page_config(page_title="Audit Blockchain", page_icon="⛓️", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
h1,h2,h3,h4{color:#f9fafb} p,li,label{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px}
.stButton>button{background:#7c3aed;color:white;border:none;border-radius:8px;font-weight:600;padding:0.5rem 1.5rem}
.block-card{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;margin:4px 0}
.hash-text{font-family:monospace;font-size:0.8rem;color:#60a5fa;word-break:break-all}
</style>
""", unsafe_allow_html=True)

SEV_COLOR = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "INFO": "#10b981"}

st.markdown("# ⛓️ Audit Blockchain")
st.caption("Module Meryem Ait Tadamna — MultiChain `ids-chain` · Stream `intrusion-logs`")

bc = get_blockchain_client()

# ── Connection status ──────────────────────────────────────────────────────────
if bc.connected:
    st.markdown('<div style="background:#052e16;border:1px solid #10b981;border-radius:8px;padding:12px">'
                '✅ <strong style="color:#10b981">MultiChain connectée</strong> — '
                'Lecture en direct depuis le stream <code>intrusion-logs</code></div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div style="background:#1e1b4b;border:1px solid #7c3aed;border-radius:8px;padding:12px">'
                '🔗 <strong style="color:#a78bfa">Chaîne locale (simulation)</strong> — '
                'MultiChain non disponible · Les données publiées via "Détection" sont visibles ici</div>',
                unsafe_allow_html=True)

# ── Kafka auto-publish status ──────────────────────────────────────────────────
if is_kafka_active():
    published = get_published_count()
    st.markdown(
        f'<div style="background:#1c1917;border:1px solid #f59e0b;border-radius:8px;'
        f'padding:10px 14px;margin-top:8px">'
        f'📡 <strong style="color:#fbbf24">Publication automatique Kafka active</strong> — '
        f'<span style="color:#d1d5db">Toute alerte <strong>CRITICAL</strong> ou <strong>HIGH</strong> '
        f'reçue sur <code>ids-detections</code> est publiée automatiquement sur la blockchain. '
        f'Publiées cette session : <strong style="color:#fbbf24">{published}</strong></span></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Load chain data ────────────────────────────────────────────────────────────
chain_data = bc.get_alerts(count=200)

if not chain_data:
    st.info("Aucun bloc dans la chaîne. Publiez des alertes depuis la page Détection.")
    st.markdown("**Pour peupler la chaîne en démo :**")
    if st.button("📥 Injecter 20 alertes de test"):
        history = get_kafka_history()
        attacks = [a for a in history if a.get("label") != "Benign"][:20]
        for a in attacks:
            bc.publish({
                "label": a["label"], "severity": a["severity"],
                "confidence": a["confidence"], "timestamp": a["timestamp"],
                "feature_hash": a["feature_hash"],
            })
        st.success("20 alertes injectées !")
        st.rerun()
    st.stop()

# ── Chain integrity ────────────────────────────────────────────────────────────
col_integ, col_stats = st.columns([1, 2])

with col_integ:
    st.markdown("### 🔐 Intégrité de la chaîne")
    if st.button("🔍 Vérifier l'intégrité"):
        ok, msg = bc.verify_chain()
        if ok:
            st.success(f"✅ {msg} — Aucune altération détectée")
        else:
            st.error(f"🚨 ALERTE : {msg}")

with col_stats:
    st.markdown("### 📊 Statistiques")
    total_b = len(chain_data)
    on_chain = sum(1 for b in chain_data if b.get("on_chain", False))
    critical = sum(1 for b in chain_data if b.get("severity") == "CRITICAL")
    confirmed = sum(1 for b in chain_data if b.get("confirmations", 0) > 0)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Blocs totaux", total_b)
    mc2.metric("On-chain", on_chain)
    mc3.metric("CRITICAL", critical)
    mc4.metric("Confirmés", confirmed)

st.markdown("<br>", unsafe_allow_html=True)

# ── Timeline of publications ───────────────────────────────────────────────────
st.markdown("### 📈 Timeline des publications blockchain")
df_bc = pd.DataFrame(chain_data)
if "timestamp" in df_bc.columns:
    df_bc["ts"] = pd.to_datetime(df_bc["timestamp"], errors="coerce", utc=True)
    df_bc = df_bc.dropna(subset=["ts"])
    df_bc["hour"] = df_bc["ts"].dt.floor("1h")
    grp = df_bc.groupby(["hour", "severity"]).size().reset_index(name="count") if "severity" in df_bc.columns else None

    if grp is not None and not grp.empty:
        fig = go.Figure()
        for sev, color in SEV_COLOR.items():
            sub = grp[grp["severity"] == sev]
            if not sub.empty:
                fig.add_trace(go.Bar(x=sub["hour"], y=sub["count"], name=sev,
                                     marker_color=color))
        fig.update_layout(
            barmode="stack", paper_bgcolor="#111827", plot_bgcolor="#111827",
            font_color="#d1d5db", height=250, margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(gridcolor="#1f2937"), yaxis=dict(gridcolor="#1f2937"),
            legend=dict(bgcolor="#0f1629"),
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Search ─────────────────────────────────────────────────────────────────────
st.markdown("### 🔎 Recherche dans la chaîne")
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    search_txid = st.text_input("Recherche par TXID", placeholder="abc123...")
with col_s2:
    all_labels = sorted(set(b.get("label", "") for b in chain_data if b.get("label")))
    label_filter = st.selectbox("Filtrer par type", ["Tous"] + all_labels)
with col_s3:
    sev_filter = st.selectbox("Filtrer par sévérité", ["Toutes", "CRITICAL", "HIGH", "MEDIUM", "INFO"])

filtered = list(chain_data)
if search_txid:
    filtered = [b for b in filtered if search_txid.lower() in str(b.get("txid", "")).lower()]
if label_filter != "Tous":
    filtered = [b for b in filtered if b.get("label") == label_filter]
if sev_filter != "Toutes":
    filtered = [b for b in filtered if b.get("severity") == sev_filter]

# ── Block explorer table ───────────────────────────────────────────────────────
st.markdown(f"### 📦 Explorateur de blocs ({len(filtered)} blocs)")

if filtered:
    display_cols = []
    for b in filtered[:100]:
        row = {
            "TXID": str(b.get("txid", b.get("block_hash", "")))[:16] + "...",
            "Timestamp": str(b.get("timestamp", ""))[:19],
            "Attaque": b.get("label", "N/A"),
            "Sévérité": b.get("severity", "N/A"),
            "Confiance": f'{b.get("confidence", 0):.1%}' if b.get("confidence") else "N/A",
            "Confirmations": b.get("confirmations", 0),
            "On-chain": "✅" if b.get("on_chain") else "🔗",
        }
        display_cols.append(row)

    df_display = pd.DataFrame(display_cols)

    def color_sev(val):
        colors = {"CRITICAL": "color:#ef4444;font-weight:700",
                  "HIGH": "color:#f97316;font-weight:700",
                  "MEDIUM": "color:#eab308", "INFO": "color:#10b981"}
        return colors.get(val, "")

    st.dataframe(
        df_display.style.map(color_sev, subset=["Sévérité"]),
        use_container_width=True, height=400,
    )

    # Block detail
    st.markdown("### 🔬 Détail d'un bloc")
    if len(filtered) > 0:
        block_idx = st.selectbox("Sélectionner un bloc",
                                 range(min(len(filtered), 50)),
                                 format_func=lambda i: f"Bloc {i} — {filtered[i].get('label','?')} @ {str(filtered[i].get('timestamp',''))[:19]}")
        b = filtered[block_idx]
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            <div class="block-card">
              <b style="color:#a78bfa">TXID</b><br>
              <span class="hash-text">{b.get('txid', b.get('block_hash', 'N/A'))}</span><br><br>
              <b style="color:#a78bfa">Attaque</b><br>
              <span style="color:{SEV_COLOR.get(b.get('severity','INFO'),'#888')};font-weight:700">
                {b.get('label','N/A')} [{b.get('severity','?')}]
              </span><br><br>
              <b style="color:#a78bfa">Timestamp</b><br>
              <span style="color:#d1d5db">{b.get('timestamp','N/A')}</span><br><br>
              <b style="color:#a78bfa">Confiance</b><br>
              <span style="color:#d1d5db">{f"{b.get('confidence',0):.1%}" if b.get('confidence') else 'N/A'}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown(f"""
            <div class="block-card">
              <b style="color:#a78bfa">Payload Hash</b><br>
              <span class="hash-text">{b.get('payload_hash', 'N/A')}</span><br><br>
              <b style="color:#a78bfa">Prev Hash</b><br>
              <span class="hash-text">{b.get('prev_hash', 'N/A')}</span><br><br>
              <b style="color:#a78bfa">Block Hash</b><br>
              <span class="hash-text">{b.get('block_hash', 'N/A')}</span><br><br>
              <b style="color:#a78bfa">On-chain</b><br>
              <span style="color:{'#10b981' if b.get('on_chain') else '#f59e0b'}">
                {"✅ Confirmé" if b.get('on_chain') else "🔗 Local"}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        export_data = json.dumps(filtered, indent=2, default=str).encode("utf-8")
        st.download_button("⬇ Export JSON", export_data, "blockchain_logs.json", "application/json")
    with col_e2:
        df_export = pd.DataFrame([{
            "txid": b.get("txid", ""), "timestamp": b.get("timestamp", ""),
            "label": b.get("label", ""), "severity": b.get("severity", ""),
            "confidence": b.get("confidence", ""), "on_chain": b.get("on_chain", False),
        } for b in filtered])
        st.download_button("⬇ Export CSV", df_export.to_csv(index=False).encode(), "blockchain_logs.csv", "text/csv")
else:
    st.info("Aucun bloc correspondant aux filtres.")

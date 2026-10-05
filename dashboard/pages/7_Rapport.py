"""
Page 7 — Rapport d'incidents : résumé exécutif, table filtrée, export CSV/JSON.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime, timezone
from collections import Counter

from core.simulator import ATTACK_DESC
from core.blockchain_client import get_blockchain_client
from core.kafka_consumer import get_kafka_history

st.set_page_config(page_title="Rapport d'incidents", page_icon="📋", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0a0e1a}
[data-testid="stSidebar"]{background:#0f1629;border-right:1px solid #1f2937}
h1,h2,h3,h4{color:#f9fafb} p,label{color:#d1d5db}
div[data-testid="metric-container"]{background:#111827;border:1px solid #1f2937;
  border-radius:10px;padding:14px}
.stButton>button{border:none;border-radius:8px;font-weight:600;padding:0.4rem 1.2rem}
.stDownloadButton>button{background:#059669;color:white;border:none;border-radius:8px;
  font-weight:600;padding:0.4rem 1.2rem}
</style>
""", unsafe_allow_html=True)

SEV_COLOR = {"CRITICAL":"#ef4444","HIGH":"#f97316","MEDIUM":"#eab308","INFO":"#10b981"}

st.markdown("## 📋 Rapport d'Incidents")
st.caption("Résumé exécutif exportable sur la période sélectionnée.")

# ── Source de données ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#1e1b4b;border:1px solid #4f46e5;border-radius:8px;padding:10px 14px;margin-bottom:16px">'
    '🔬 <strong style="color:#a5b4fc">Source des données :</strong> '
    '<span style="color:#d1d5db">Toutes les détections ci-dessous proviennent du <strong>simulateur '
    'IDS-Blockchain</strong> qui reproduit fidèlement la distribution réelle de CSE-CIC-IDS2018. '
    'Avec le modèle CNN-BiLSTM réel (bilstm_ids_v2.pt), les données seront des inférences réelles. '
    'Les exports restent identiques dans les deux cas.</span></div>',
    unsafe_allow_html=True
)

# ── Sélecteurs ────────────────────────────────────────────────────────────────
col_p, col_src = st.columns(2)
with col_p:
    period = st.selectbox("Période", ["Dernière heure", "Dernières 24h", "7 derniers jours"])
with col_src:
    source_opt = st.selectbox("Source", ["Alertes Kafka", "Alertes session live", "Blockchain"])

with st.spinner("Chargement..."):
    if source_opt == "Alertes session live" and st.session_state.get("alerts"):
        raw = st.session_state["alerts"]
    elif source_opt == "Blockchain":
        bc = get_blockchain_client()
        raw = bc.get_alerts(count=500)
        if not raw:
            raw = get_kafka_history()
            if raw:
                st.info("Blockchain vide — affichage de l'historique Kafka.")
    else:
        raw = get_kafka_history()

if not raw:
    st.warning("Aucune donnée disponible pour cette période.")
    st.stop()

df = pd.DataFrame(raw)

# Normalise les colonnes : assure la présence des champs minimum
for col in ["label","severity","confidence","timestamp"]:
    if col not in df.columns:
        df[col] = "N/A"

df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

total = len(df)
attacks = df[df["label"] != "Benign"]
critical  = int((df["severity"] == "CRITICAL").sum())
high      = int((df["severity"] == "HIGH").sum())
medium    = int((df["severity"] == "MEDIUM").sum())
attack_rate = round(len(attacks) / total * 100, 1) if total > 0 else 0
top_label = attacks["label"].value_counts().idxmax() if len(attacks) > 0 else "Aucune"
top_count = int(attacks["label"].value_counts().iloc[0]) if len(attacks) > 0 else 0
avg_conf  = round(float(df["confidence"].mean()) * 100, 1)
now_str   = datetime.now().strftime("%d/%m/%Y à %H:%M")

# ── Résumé exécutif ────────────────────────────────────────────────────────────
st.markdown("### 📄 Résumé exécutif")
st.markdown(f"""
<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:24px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px">
    <div>
      <h3 style="color:#f9fafb;margin:0">Rapport de Sécurité — IDS Blockchain</h3>
      <p style="color:#6b7280;margin:4px 0 0 0;font-size:0.85rem">Généré le {now_str} · Période : {period}</p>
    </div>
    <span style="background:#1d4ed8;color:white;padding:4px 16px;border-radius:20px;font-size:0.85rem;font-weight:700">
      Groupe 3
    </span>
  </div>
  <hr style="border-color:#1f2937;margin:0 0 16px 0">
  <p style="color:#d1d5db;line-height:1.8;margin:0">
    Le système IDS a analysé <strong style="color:#60a5fa">{total:,} flows réseau</strong> et détecté
    <strong style="color:#ef4444">{len(attacks):,} événements malveillants</strong>
    (taux d'attaque : <strong style="color:#f97316">{attack_rate}%</strong>).<br>
    Attaque dominante : <strong style="color:#ef4444">{top_label}</strong> ({top_count:,} occurrences).<br>
    Confiance moyenne du modèle : <strong style="color:#10b981">{avg_conf}%</strong>.<br>
    <span style="color:#ef4444;font-weight:700">{critical:,} alertes CRITICAL</span> ·
    <span style="color:#f97316;font-weight:700">{high:,} HIGH</span> ·
    <span style="color:#eab308;font-weight:700">{medium:,} MEDIUM</span>.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Flows analysés", f"{total:,}")
m2.metric("CRITICAL", f"{critical:,}")
m3.metric("HIGH", f"{high:,}")
m4.metric("Taux d'attaque", f"{attack_rate}%")
m5.metric("Confiance moy.", f"{avg_conf}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Filtres + Table ────────────────────────────────────────────────────────────
st.markdown("### Table des incidents")
c1, c2, c3 = st.columns(3)
with c1:
    sev_f  = st.selectbox("Sévérité", ["Toutes","CRITICAL","HIGH","MEDIUM","INFO"], key="r_sev")
with c2:
    hide_b = st.checkbox("Masquer Benign", value=True, key="r_benign")
with c3:
    conf_min = st.slider("Confiance min.", 0.0, 1.0, 0.0, 0.05, key="r_conf")

df_f = df.copy()
if sev_f != "Toutes":
    df_f = df_f[df_f["severity"] == sev_f]
if hide_b:
    df_f = df_f[df_f["label"] != "Benign"]
df_f = df_f[df_f["confidence"] >= conf_min].head(1000)

display_cols = [c for c in ["timestamp","label","severity","confidence","src_ip","dst_ip","dst_port","protocol"] if c in df_f.columns]
df_table = df_f[display_cols].copy()
if "timestamp" in df_table.columns:
    df_table["timestamp"] = df_table["timestamp"].str[:19]
if "confidence" in df_table.columns:
    df_table["confidence"] = df_table["confidence"].apply(lambda x: f"{float(x):.1%}")

rename = {"timestamp":"Timestamp","label":"Attaque","severity":"Sévérité",
          "confidence":"Confiance","src_ip":"IP Source","dst_ip":"IP Dest",
          "dst_port":"Port","protocol":"Proto"}
df_table = df_table.rename(columns=rename)

def _color_sev(s):
    m = {"CRITICAL":"color:#ef4444;font-weight:bold",
         "HIGH":"color:#f97316;font-weight:bold",
         "MEDIUM":"color:#eab308","INFO":"color:#10b981"}
    return [m.get(v,"") for v in s]

if "Sévérité" in df_table.columns:
    st.dataframe(df_table.style.apply(_color_sev, subset=["Sévérité"]),
                 use_container_width=True, height=380)
else:
    st.dataframe(df_table, use_container_width=True, height=380)

st.caption(f"{len(df_table):,} incidents filtrés sur {total:,} flows.")

# ── Résumé par type d'attaque ──────────────────────────────────────────────────
if len(attacks) > 0 and "label" in attacks.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Résumé par type d'attaque")
    rows = []
    for label, grp in attacks.groupby("label"):
        sev = grp["severity"].iloc[0] if "severity" in grp.columns else "?"
        avg_c = float(grp["confidence"].mean()) if "confidence" in grp.columns else 0.0
        rows.append({
            "Type d'attaque":  label,
            "Sévérité":        sev,
            "Occurrences":     len(grp),
            "% du total":      f"{len(grp)/total*100:.2f}%",
            "Confiance moy.":  f"{avg_c:.1%}",
            "Description":     ATTACK_DESC.get(label, ""),
        })
    df_sum = pd.DataFrame(rows).sort_values("Occurrences", ascending=False)
    st.dataframe(
        df_sum.style.apply(_color_sev, subset=["Sévérité"]),
        use_container_width=True, hide_index=True
    )

# ── Intégrité blockchain ───────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Intégrité Blockchain")
bc = get_blockchain_client()
ok, msg = bc.verify_chain()
chain_len = bc.chain_length()
if ok:
    st.markdown(f'<div style="background:#052e16;border:1px solid #10b981;border-radius:8px;padding:12px">'
                f'✅ <strong style="color:#10b981">Chaîne intègre</strong> — {msg} · {chain_len} blocs enregistrés.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f'<div style="background:#1f0000;border:1px solid #ef4444;border-radius:8px;padding:12px">'
                f'🚨 <strong style="color:#ef4444">ALERTE INTÉGRITÉ : {msg}</strong></div>',
                unsafe_allow_html=True)

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Export du rapport")

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "period": period,
    "source": source_opt,
    "summary": {
        "total_flows": total,
        "total_attacks": int(len(attacks)),
        "critical": critical, "high": high, "medium": medium,
        "attack_rate_pct": attack_rate,
        "avg_confidence_pct": avg_conf,
        "top_attack": top_label,
        "blockchain_integrity": bool(ok),
        "blockchain_blocks": chain_len,
    },
    "incidents": df_f.to_dict(orient="records"),
}

col_e1, col_e2 = st.columns(2)
with col_e1:
    json_bytes = json.dumps(report, indent=2, default=str).encode("utf-8")
    st.download_button(
        "⬇ Télécharger rapport JSON",
        json_bytes,
        f"rapport_ids_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        "application/json",
        use_container_width=True,
    )
with col_e2:
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Télécharger incidents CSV",
        csv_bytes,
        f"incidents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
    )

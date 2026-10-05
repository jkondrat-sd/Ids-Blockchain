# Dashboard IDS — Interface Graphique + Pipeline Kafka
**Auteur :** Noussair Fenani

---

## Ce que fait ce module

Interface graphique centrale du projet IDS-Blockchain. Regroupe et visualise
en temps réel les résultats des 4 modules du groupe via un pipeline Kafka automatique.

```
Flows réseau
    │  Kafka topic: network-flows
    ▼
Inference Worker ──→ Modèle CNN-BiLSTM (Walid)
    │  Kafka topic: ids-detections
    ├──→ Blockchain Publisher ──→ MultiChain (Meryem)
    ├──→ Spark Streaming      ──→ stats JSON
    └──→ Dashboard Consumer   ──→ 7 pages Streamlit
```

---

## Prérequis

| Outil         | Version    | Obligatoire |
|---------------|------------|-------------|
| Python        | 3.10+      | Oui         |
| Docker Desktop| Dernière   | Non (fallback simulation) |
| Java JDK      | 11+        | Non (Spark Streaming désactivé sans Java) |

---

## Installation

```bash
cd dashboard
pip install -r requirements.txt
```

---

## Démarrage

```bash
cd dashboard
streamlit run app.py
```

Le dashboard s'ouvre sur **http://localhost:8501**

Kafka démarre automatiquement via Docker si disponible.
Sans Docker → mode simulation pure (toutes les pages restent fonctionnelles).

### Démarrer Kafka manuellement (optionnel)

```bash
# Depuis la racine du projet
docker compose up -d
```

---

## Pages disponibles

| Page               | Description |
|--------------------|-------------|
| Accueil            | Statut des 4 modules, explication du pipeline |
| 1. Live Monitor    | Flux alertes temps réel, KPIs, timeline par sévérité |
| 2. Détection       | Upload CSV → inférence CNN-BiLSTM → export résultats |
| 3. Threat Intel    | Tendances 24h/7j, heatmap attaques × heure |
| 4. Blockchain      | Explorateur MultiChain, vérification intégrité SHA-256 |
| 5. Big Data        | Stats CIC-IDS2018, bouton lancer pipeline Spark |
| 6. Modèle IA       | F1 par classe, matrice de confusion 15x15 |
| 7. Rapport         | Rapport exportable PDF / CSV / JSON |

---

## Activer le vrai modèle (mode réel)

Le dashboard bascule automatiquement dès que ces fichiers existent :

```
ai_detection/models/bilstm_ids_v2.pt        ← Walid fournit ce fichier
ai_detection/models/web_splitter_v2.pkl     ← Walid fournit ce fichier
ai_detection/data/processed/scaler.pkl      ← Safaa fournit ce fichier
```

Aucune modification du code n'est nécessaire.

---

## Structure des fichiers

```
dashboard/
├── app.py                        ← Page d'accueil + initialisation
├── requirements.txt
├── core/
│   ├── simulator.py              ← Générateur alertes (mode démo)
│   ├── model_client.py           ← Wrapper CNN-BiLSTM v2 (Walid)
│   ├── blockchain_client.py      ← Wrapper MultiChain (Meryem)
│   ├── spark_client.py           ← Wrapper pipeline Spark
│   ├── startup.py                ← Démarrage auto Kafka via Docker
│   └── kafka_consumer.py         ← Buffer alertes + persistance JSON
├── kafka_pipeline/
│   ├── producer.py               ← Publie flows → topic network-flows
│   ├── inference_worker.py       ← Appelle modèle → topic ids-detections
│   └── blockchain_publisher.py   ← Filtre CRITICAL/HIGH → blockchain
├── spark/
│   └── streaming_pipeline.py     ← PySpark Streaming (Kafka → JSON)
└── pages/
    ├── 1_Live_Monitor.py
    ├── 2_Detection.py
    ├── 3_Threat_Intel.py
    ├── 4_Blockchain.py
    ├── 5_BigData.py
    ├── 6_Model_Performance.py
    └── 7_Rapport.py
```

---

## Variables d'environnement

| Variable         | Défaut           | Description |
|------------------|------------------|-------------|
| KAFKA_BOOTSTRAP  | localhost:9092   | Adresse du broker Kafka |

---

## Dépendances Python

```
streamlit>=1.32.0        plotly>=5.18.0
streamlit-autorefresh    pandas>=2.0.0
kafka-python>=2.0.2      numpy>=1.24.0
pyspark>=3.5.0           torch>=2.0.0
scikit-learn>=1.3.0      fpdf2>=2.7.0
```

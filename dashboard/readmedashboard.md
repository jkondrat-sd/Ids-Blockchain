# 🛡️ IDS-Blockchain Dashboard

**Auteur :** Noussair Fenani  
**Rôle :** Big Data & Apache Spark · Interface Graphique  
**Projet :** Système de Détection d'Intrusions Adaptatif — Groupe 3  

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du dashboard](#2-architecture-du-dashboard)
3. [Installation et lancement](#3-installation-et-lancement)
4. [Pages du dashboard](#4-pages-du-dashboard)
5. [Modules core](#5-modules-core)
6. [Données : simulation vs mode réel](#6-données--simulation-vs-mode-réel)
7. [Connexion aux autres modules du projet](#7-connexion-aux-autres-modules-du-projet)
8. [Stack technique](#8-stack-technique)

---

## 1. Vue d'ensemble

Le dashboard IDS-Blockchain est l'interface graphique centrale du projet. Il regroupe et visualise en temps réel les résultats de chacun des 4 modules du groupe :

| Module | Responsable | Ce que le dashboard affiche |
|--------|-------------|----------------------------|
| Modèle IA (CNN-BiLSTM v2) | Walid Ichchou | Détections, F1 par classe, matrice de confusion |
| Blockchain (MultiChain) | Meryem Ait Tadamna | Audit de la chaîne, blocs, intégrité SHA-256 |
| Big Data (Apache Spark) | Noussair Fenani | Stats dataset, pipeline, métriques de traitement |
| Data Pipeline | Safaa Khalladi | Distribution des classes, features CICFlowMeter |

Le dashboard est construit avec **Streamlit** (Python). Il fonctionne en **mode simulation** quand les fichiers du modèle et de la blockchain ne sont pas disponibles, et bascule automatiquement en **mode réel** dès que ces fichiers sont présents.

---

## 2. Architecture du dashboard

```
dashboard/
├── app.py                        ← Page d'accueil (home)
├── requirements.txt              ← Dépendances Python
├── readmedashboard.md            ← Ce fichier
│
├── core/                         ← Modules partagés (logique métier)
│   ├── __init__.py
│   ├── simulator.py              ← Générateur d'alertes réalistes (mode démo)
│   ├── model_client.py           ← Wrapper CNN-BiLSTM v2 (Walid)
│   ├── blockchain_client.py      ← Wrapper MultiChain (Meryem)
│   └── spark_client.py           ← Wrapper pipeline Spark (Noussair)
│
└── pages/                        ← Pages Streamlit (navigation automatique)
    ├── 1_Live_Monitor.py         ← Flux d'alertes en temps réel
    ├── 2_Detection.py            ← Upload CSV + inférence CNN-BiLSTM
    ├── 3_Threat_Intel.py         ← Analyse des menaces et tendances
    ├── 4_Blockchain.py           ← Audit de la chaîne MultiChain
    ├── 5_BigData.py              ← Big Data & Apache Spark
    ├── 6_Model_Performance.py    ← Performance du modèle IA
    └── 7_Rapport.py              ← Rapport d'incidents exportable
```

---

## 3. Installation et lancement

### Prérequis

- Python 3.10 ou supérieur
- pip

### Installation des dépendances

```bash
cd dashboard
pip install -r requirements.txt
```

Le fichier `requirements.txt` contient :

```
streamlit>=1.32.0
streamlit-autorefresh>=1.0.1
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
scikit-learn>=1.3.0
fpdf2>=2.7.0
```

### Lancement

```bash
cd dashboard
streamlit run app.py
```

Le dashboard s'ouvre automatiquement sur `http://localhost:8501`.  
La navigation entre les pages se fait via la barre latérale à gauche.

---

## 4. Pages du dashboard

### Page d'accueil — `app.py`

- Présentation de l'équipe et des 4 modules du projet
- Status en temps réel de chaque module (modèle chargé ou non, blockchain connectée ou non)
- Explication complète du pipeline de détection (du trafic réseau jusqu'à la blockchain)
- Explication de la différence entre mode simulation et mode réel

---

### Page 1 — Live Monitor (`1_Live_Monitor.py`)

**Objectif :** afficher un flux d'alertes IDS en temps réel.

**Fonctionnalités :**
- Auto-refresh toutes les 4 secondes (via `streamlit-autorefresh`)
- Bouton Pause / Reprendre pour geler le flux
- 4 KPI cards : flows analysés, alertes totales, alertes CRITICAL, menaces récentes
- Graphe linéaire en temps réel : alertes par minute, colorées par sévérité (CRITICAL/HIGH/MEDIUM/INFO)
- Panneau latéral : 6 dernières menaces critiques avec IP source et timestamp
- Table des alertes avec filtre par sévérité et option "masquer Benign"
- Donut : répartition des sévérités sur la session
- Bar chart : top 8 attaques détectées

**Niveaux de sévérité :**

| Sévérité | Couleur | Classes concernées |
|----------|---------|-------------------|
| CRITICAL | Rouge `#ef4444` | DDoS-HOIC, DDoS-LOIC-HTTP, DDoS-LOIC-UDP, DoS-Hulk, SQL-Injection, Infiltration |
| HIGH | Orange `#f97316` | Botnet, DoS-GoldenEye, DoS-Slowloris, DoS-SlowHTTPTest, BruteForce-Web, BruteForce-XSS |
| MEDIUM | Jaune `#eab308` | SSH-BruteForce, FTP-BruteForce |
| INFO | Vert `#10b981` | Benign (trafic normal) |

---

### Page 2 — Détection (`2_Detection.py`)

**Objectif :** analyser un fichier CSV de trafic réseau avec le modèle CNN-BiLSTM.

**Fonctionnalités :**
- Upload d'un fichier CSV au format CICFlowMeter (80 features)
- Ou lancement d'une analyse simulée (choix du nombre de flows)
- Inférence batch via `IDSPredictor` (Walid)
- Affichage des résultats :
  - 4 KPI : total flows, CRITICAL, HIGH, taux d'attaque
  - Pie chart : distribution des sévérités
  - Bar chart : types d'attaques détectées
  - Histogramme : distribution des scores de confiance
  - Table détaillée avec filtres (sévérité, confiance min, masquer Benign)
- Publication des alertes sur MultiChain (Meryem) avec seuil configurable
- Export CSV et JSON des résultats

**Format CSV attendu :**  
Les colonnes doivent correspondre aux features listées dans `ai_detection/data/processed/feature_names.json`. Les données ne doivent pas être normalisées — le scaling est appliqué automatiquement par `IDSPredictor`.

---

### Page 3 — Threat Intelligence (`3_Threat_Intel.py`)

**Objectif :** analyser les patterns d'attaques et les tendances temporelles.

**Fonctionnalités :**
- Sélecteur de période : 6h / 24h / 7 jours
- Source : historique simulé ou alertes de la session live
- 4 KPI : flows analysés, attaques, CRITICAL, attaque dominante
- Timeline : courbes par sévérité (alertes par heure)
- Heatmap : top 8 types d'attaques × heure du jour (0h–23h)
- Bar chart : top 10 attaques avec couleur par sévérité
- Donut : répartition par protocole (TCP / UDP / ICMP)
- Donut : répartition par sévérité
- Graphe de tendance : attaques par tranche de 3h avec moyenne mobile

---

### Page 4 — Audit Blockchain (`4_Blockchain.py`)

**Objectif :** explorer et vérifier la chaîne MultiChain de Meryem.

**Fonctionnalités :**
- Banner de connexion : MultiChain en direct ou chaîne locale simulée
- Bouton "Vérifier l'intégrité" → appelle `verify_chain()` → affiche OK ou ALERTE
- 4 KPI : blocs totaux, on-chain, CRITICAL, confirmés
- Timeline : publications blockchain par heure et par sévérité
- Recherche : par TXID, par type d'attaque, par sévérité
- Explorateur de blocs : table avec TXID, timestamp, attaque, sévérité, confiance, on-chain
- Détail d'un bloc : payload_hash, prev_hash, block_hash, status on-chain
- Export JSON et CSV des logs blockchain
- Bouton "Injecter 20 alertes de test" pour peupler la chaîne en démo

**Structure d'un bloc enregistré :**
```json
{
  "block_idx":    3,
  "txid":         "f7c3a1b2...",
  "payload": {
    "label":        "DDoS-HOIC",
    "severity":     "CRITICAL",
    "confidence":   0.97,
    "timestamp":    "2026-04-27T10:23:45Z",
    "feature_hash": "a3f1c2e4..."
  },
  "payload_hash": "f7c3a1...",
  "prev_hash":    "8e2b4d...",
  "block_hash":   "c2d5f8...",
  "on_chain":     true
}
```

---

### Page 5 — Big Data & Spark (`5_BigData.py`)

**Objectif :** visualiser le dataset CIC-IDS2018 et le pipeline Apache Spark.

**Fonctionnalités :**
- 5 KPI : total flows (16.2M), taille (16.2 GB), features (80), classes (15), train/test split
- Bar chart : distribution des 15 classes (déséquilibre de classes visible)
- Pie chart : répartition par classe
- Liste des 12 features réseau clés avec descriptions
- Schéma du pipeline complet (CSV → Spark → Preprocessing → CNN-BiLSTM → Blockchain)
- Bouton "Lancer le pipeline" : exécute `Pipeline.scala` via sbt ou JAR, affiche les logs
- Métriques de traitement : throughput (~85k flows/s), temps preprocessing (~5 min), partitions Spark
- Cards de la stack technique : Spark 3.5.1, Dataset CIC-IDS2018, Preprocessing

**Distribution des classes du dataset :**

| Classe | Flows | % |
|--------|-------|---|
| Benign | 6 747 813 | 41.6% |
| DDoS-HOIC | 1 082 885 | 6.7% |
| SSH-BruteForce | 187 589 | 1.2% |
| Botnet | 286 191 | 1.8% |
| Infiltration | 18 893 | 0.1% |
| ... | ... | ... |

---

### Page 6 — Performance du Modèle IA (`6_Model_Performance.py`)

**Objectif :** afficher les métriques complètes du modèle CNN-BiLSTM v2 (Walid).

**Fonctionnalités :**
- 5 KPI globaux : Accuracy (90.76%), Macro-F1 (0.7711), Weighted-F1 (0.8955), ~1.2M paramètres, ~30 min d'entraînement
- Bar chart : F1-Score par classe (code couleur vert/orange/rouge selon performance)
- Ligne de référence : Macro-F1 moyen (0.7711)
- Bar chart groupé : Précision vs Rappel par classe
- Slider : seuil de confiance configurable (0.50–0.99)
- Matrice de confusion : heatmap 15×15 normalisée
- Table comparaison des 3 modèles : CNN-BiLSTM v2 (production), BiLSTM Advanced, BiLSTM Baseline
- Analyse des classes difficiles : Infiltration (F1=0.30) et Web Attacks (F1=0.21–0.47)
- Schéma de l'architecture CNN-BiLSTM v2

**Résultats du modèle production (CNN-BiLSTM v2) :**

| Métrique | Valeur |
|----------|--------|
| Accuracy | 90.76% |
| Macro-F1 | 0.7711 |
| Weighted-F1 | 0.8955 |
| Classe la plus facile | DDoS-HOIC, SSH-BruteForce (F1=1.00) |
| Classe la plus difficile | SQL-Injection (F1=0.21), Infiltration (F1=0.30) |

---

### Page 7 — Rapport d'Incidents (`7_Rapport.py`)

**Objectif :** générer un rapport exportable sur une période donnée.

**Fonctionnalités :**
- Sélecteur de période : 1h / 24h / 7 jours
- Source : historique simulé, alertes session live, ou blockchain
- Résumé exécutif automatique avec les chiffres clés
- 5 KPI : flows analysés, CRITICAL, HIGH, taux d'attaque, confiance moyenne
- Table des incidents avec filtres : sévérité, Benign, confiance minimum, nombre max de lignes
- Résumé par type d'attaque : occurrences, % du total, confiance moyenne, description
- Vérification de l'intégrité blockchain en bas de page
- Export JSON : rapport complet (résumé + incidents)
- Export CSV : incidents filtrés uniquement

---

## 5. Modules core

### `core/simulator.py`

Génère des alertes IDS réalistes en mode démo.  
La distribution des classes reproduit exactement celle du dataset CSE-CIC-IDS2018.

**Fonctions principales :**
- `generate_alert(ts=None)` — génère une alerte unique avec label, sévérité, confiance, IP source/dest, port, protocole
- `generate_history(hours, avg_per_hour)` — génère un historique complet sur N heures

**Distribution simulée (basée sur CIC-IDS2018) :**
```python
CLASS_WEIGHTS = {
    "Benign":       0.53,   # trafic normal
    "DDoS-HOIC":    0.08,   # attaque dominante
    "SSH-BruteForce": 0.04,
    "Botnet":       0.03,
    ...
}
```

---

### `core/model_client.py`

Wrapper autour de `IDSPredictor` (module Walid).

**Comportement :**
- Si `ai_detection/models/bilstm_ids_v2.pt` et `web_splitter_v2.pkl` existent → charge le vrai modèle PyTorch
- Sinon → utilise le simulateur pour générer des prédictions réalistes

**Méthodes :**
- `predict_csv(df)` → retourne une liste de dicts `{label, severity, confidence, timestamp, feature_hash}`
- `status` → `"Modèle réel chargé"` ou `"Mode simulation"`

---

### `core/blockchain_client.py`

Wrapper autour des modules de Meryem (`blockchain_reader.py`, `blockchain_interface/api.py`).

**Comportement :**
- Essaie de se connecter à MultiChain via `blockchain_reader.get_all_intrusions()`
- Si MultiChain ne tourne pas → utilise une chaîne locale en mémoire (SHA-256 chainée identique à l'algorithme de Meryem)

**Méthodes :**
- `get_alerts(count)` → retourne les N dernières alertes de la chaîne
- `publish(prediction)` → publie une alerte sur MultiChain ou la chaîne locale
- `verify_chain()` → retourne `(True, message)` si intègre, `(False, message)` si fraude détectée
- `chain_length()` → nombre de blocs dans la chaîne
- `status` → `"MultiChain connectée"` ou `"Chaîne locale (simulation)"`

---

### `core/spark_client.py`

Wrapper autour du pipeline Scala de Noussair (`big_data_spark/src/main/scala/Pipeline.scala`).

**Comportement :**
- Si le JAR compilé existe (`target/scala-2.12/ids-bigdata-pipeline_2.12-1.0.jar`) → exécute `java -jar`
- Sinon → essaie `sbt run` dans le dossier `big_data_spark/`
- Si ni Java ni sbt n'est disponible → retourne un message d'erreur explicite

**Fonctions :**
- `run_pipeline()` → exécute le pipeline et retourne `{success, log, elapsed}`
- `get_dataset_stats()` → retourne les statistiques du dataset CIC-IDS2018

---

## 6. Données : simulation vs mode réel

### Mode simulation (actuel)

Le simulateur génère des alertes avec :
- La **même distribution statistique** que le vrai dataset CIC-IDS2018
- Des IP source/destination aléatoires (subnets réalistes : 192.168.x, 10.0.x, etc.)
- Des ports cohérents avec chaque type d'attaque (port 22 pour SSH-BruteForce, port 3306 pour SQL-Injection, etc.)
- Des scores de confiance dans les plages réelles du modèle (ex: CRITICAL entre 0.87 et 0.99)

⚠️ **Les données simulées ne sont pas issues d'une vraie capture réseau.** Elles servent à démontrer le fonctionnement complet du dashboard.

### Mode réel (après fusion des branches)

Le dashboard bascule automatiquement en mode réel dès que les fichiers suivants sont présents :

| Fichier requis | Module | Effet |
|----------------|--------|-------|
| `ai_detection/models/bilstm_ids_v2.pt` | Walid | Inférence CNN-BiLSTM réelle |
| `ai_detection/models/web_splitter_v2.pkl` | Walid | Classification Web Attacks |
| `ai_detection/data/processed/scaler.pkl` | Safaa | Normalisation des features |
| `ai_detection/data/processed/feature_names.json` | Safaa | Sélection des features |
| MultiChain en cours d'exécution | Meryem | Logs blockchain réels |

Aucune modification du code du dashboard n'est nécessaire pour passer en mode réel.

---

## 7. Connexion aux autres modules du projet

### Connexion avec le module IA (Walid)

Le dashboard importe et utilise directement :
```python
# core/model_client.py
from src.inference.predict import IDSPredictor

predictor = IDSPredictor(
    checkpoint_path = "ai_detection/models/bilstm_ids_v2.pt",
    splitter_path   = "ai_detection/models/web_splitter_v2.pkl",
    processed_dir   = "ai_detection/data/processed/",
)
result = predictor.predict(feature_vector)
# {"label": "DDoS-HOIC", "severity": "CRITICAL", "confidence": 0.97, ...}
```

### Connexion avec le module Blockchain (Meryem)

Le dashboard utilise :
```python
# core/blockchain_client.py
import blockchain_reader
import blockchain_interface.api as api

# Lecture
entries = blockchain_reader.get_all_intrusions(count=100)

# Écriture
block = api.log_prediction(prediction)

# Vérification
ok = api.verify_chain()
```

### Connexion avec le pipeline Spark (Noussair)

```python
# core/spark_client.py
import subprocess
result = subprocess.run(
    ["java", "-jar", "big_data_spark/target/scala-2.12/ids-bigdata-pipeline_2.12-1.0.jar"],
    capture_output=True, text=True, timeout=120,
)
```

---

## 8. Stack technique

| Technologie | Version | Usage |
|-------------|---------|-------|
| Python | 3.10+ | Langage principal du dashboard |
| Streamlit | ≥ 1.32.0 | Framework UI web |
| streamlit-autorefresh | ≥ 1.0.1 | Auto-refresh de la page Live Monitor |
| Plotly | ≥ 5.18.0 | Graphes interactifs (timelines, heatmaps, pie charts) |
| Pandas | ≥ 2.0.0 | Manipulation des données tabulaires |
| NumPy | ≥ 1.24.0 | Calculs numériques |
| PyTorch | ≥ 2.0.0 | Chargement et inférence du modèle CNN-BiLSTM |
| scikit-learn | ≥ 1.3.0 | Scaler, LabelEncoder, web_splitter |
| fpdf2 | ≥ 2.7.0 | Génération de rapports PDF (optionnel) |
| Apache Spark | 3.5.1 | Big Data pipeline (Scala) |
| MultiChain | 2.3.3 | Blockchain privée pour les logs d'intrusion |

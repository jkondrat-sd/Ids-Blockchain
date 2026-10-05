# Pipeline Big Data — Apache Spark (Batch)
**Auteur :** Noussair Fenani

---

## Ce que fait ce module

Pipeline Scala/Spark qui prétraite le dataset CSE-CIC-IDS2018 avant l'entraînement
du modèle CNN-BiLSTM. Prend en entrée les CSV bruts, produit un dataset propre
et normalisé au format Parquet.

```
CSVs CIC-IDS2018 (~16 GB)
    │
    ▼  [1] Lecture Spark (local[*])
    ▼  [2] Sélection 47 features + nettoyage NaN/Inf
    ▼  [3] Normalisation labels ("DDOS attack-HOIC" → "DDoS-HOIC")
    ▼  [4] VectorAssembler + StandardScaler (μ=0, σ=1)
    ▼  [5] StringIndexer (label → index 0-14)
    ▼  [6] Split 80% train / 20% test → Parquet
```

---

## Prérequis

| Outil     | Version  | Installation |
|-----------|----------|--------------|
| Java JDK  | 11+      | https://adoptium.net |
| sbt       | 1.9+     | https://www.scala-sbt.org/download |
| Scala     | 2.12.18  | Téléchargé automatiquement par sbt |
| Spark     | 3.5.1    | Téléchargé automatiquement par sbt |

### Vérifier les prérequis

```bash
java -version    # doit afficher 11.x ou supérieur
sbt --version    # doit afficher 1.9.x
```

---

## Démarrage

### Option 1 — Lancement direct (recommandé)

```bash
cd big_data_spark
sbt run
```

Sans dataset → génère automatiquement **100 000 flows synthétiques** avec la
distribution réelle de CIC-IDS2018 pour démontrer toutes les étapes.

### Option 2 — Avec le vrai dataset CIC-IDS2018

```bash
cd big_data_spark
sbt "run --input /chemin/vers/cic-ids2018 --output output/preprocessed"
```

### Option 3 — Compiler un fat JAR autonome

```bash
cd big_data_spark
sbt assembly
# Produit : target/scala-2.12/ids-bigdata-pipeline_2.12-1.0.jar
java -jar target/scala-2.12/ids-bigdata-pipeline_2.12-1.0.jar
```

---

## Sortie attendue

```
==============================================================
  IDS-BigData Pipeline - CSE-CIC-IDS2018
  Auteur : Noussair Fenani - Big Data & Apache Spark
==============================================================
[1/6] Dataset absent - donnees synthetiques (100 000 flows)
      -> 100,000 flows  (48 colonnes)
[2/6] Selection 47 features + nettoyage NaN/Inf...
      -> 100,000 flows retenus
[3/6] Normalisation des labels...
        Benign                   53,000  (53.0%)
        DDoS-HOIC                 8,000  ( 8.0%)
        ...
[4/6] VectorAssembler + StandardScaler...
[5/6] StringIndexer (label -> index)...
        [ 0] Benign
        [ 1] DDoS-HOIC
        ...
[6/6] Split 80/20 + sauvegarde Parquet...
==============================================================
  RESULTATS
==============================================================
  Total  : 100,000 flows
  Train  : 80,012 -> output/preprocessed/train
  Test   : 19,988 -> output/preprocessed/test
  Feats  : 47 / 47
  Temps  : 18.3 s  |  Throughput : 5,464 flows/s
==============================================================
  Pipeline termine avec succes.
```

---

## Arguments disponibles

| Argument   | Défaut              | Description |
|------------|---------------------|-------------|
| `--input`  | `data/cic-ids2018`  | Dossier contenant les CSV bruts |
| `--output` | `output/preprocessed` | Dossier de sortie Parquet |

---

## Features sélectionnées (47/80)

Les 47 features retenues correspondent à `ai_detection/data/processed/feature_names.json` :

```
Flow Duration, Tot Fwd Pkts, Tot Bwd Pkts, Fwd Pkt Len Max, Fwd Pkt Len Min,
Fwd Pkt Len Mean, Fwd Pkt Len Std, Bwd Pkt Len Max, Bwd Pkt Len Min,
Bwd Pkt Len Mean, Bwd Pkt Len Std, Flow Byts/s, Flow Pkts/s,
Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, ...
```

---

## Structure des fichiers

```
big_data_spark/
├── build.sbt                          ← Config Spark 3.5.1 + sbt-assembly
├── project/
│   ├── build.properties               ← sbt version 1.9.9
│   └── plugins.sbt                    ← sbt-assembly 2.1.5
└── src/main/scala/
    └── Pipeline.scala                 ← Pipeline complet (6 étapes)
```

---

## Connexion avec les autres modules

Ce pipeline produit les données utilisées par :
- **Walid (CNN-BiLSTM)** — reçoit les Parquet train/test pour l'entraînement
- **Safaa (Data Pipeline)** — fournit `feature_names.json` et `scaler.pkl`
- **Dashboard (Noussair)** — le bouton "Lancer le pipeline" sur la page Big Data appelle ce JAR

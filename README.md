# IDS-Blockchain — Adaptive Network Intrusion Detection

Multi-class network intrusion detection on the **CSE-CIC-IDS2018** dataset, with a streaming pipeline, a tamper-evident blockchain audit log, and a monitoring dashboard.

A CNN-BiLSTM model classifies each network flow into one of 15 traffic classes (benign plus 14 attack families). High-severity alerts are hashed and written to a private MultiChain ledger so the alert history cannot be silently altered. A Streamlit dashboard shows detections, threat trends, the audit chain and model performance.

This was a four-person academic project. Team credits are [below](#team).

## Results

Measured with the checkpoint in this repository (`ai_detection/models/bilstm_ids_v2.pt`) on the held-out test split of 212,330 flows:

| Metric | Value |
|---|---|
| Macro-F1 (15 classes) | **0.7711** |
| Accuracy | 91.6% |
| Benign recall | 98.6% |
| Model size | ~0.98M parameters |

The hardest classes are Infiltration, which overlaps heavily with benign traffic in feature space, and the three web-attack classes, which have only a few hundred samples each in the whole dataset.

## Architecture

```
Network traffic (CICFlowMeter CSVs, ~16 GB, 16.2M flows)
        │
        ▼
data_pipeline/      first-pass cleaning and exploration (pandas)
big_data_spark/     distributed preprocessing (Scala, Spark 3.5)
        │
        ▼
ai_detection/       CNN-BiLSTM classifier (PyTorch), 15 classes
        │           label + severity + confidence per flow
        ▼
blockchain/         SHA-256 chained alert blocks on MultiChain
        │
        ▼
dashboard/          Streamlit UI, Kafka pipeline, Spark Structured Streaming
```

### Detection model

```
Input: one flow = 47 features, reshaped to (T=12, C=4)
 └─ 2× Conv1d (k=3) + GELU
 └─ BiLSTM (hidden=128, 2 layers)
 └─ Multi-head attention pooling (4 heads)
 └─ MLP head → 13 classes
 └─ Logistic-regression splitter restores the 3 web-attack classes → 15 classes
```

Design decisions worth noting:

- **One flow is one sample.** CICFlowMeter already aggregates packets into per-flow statistics, so there are no sliding windows over unrelated flows.
- **Class imbalance is handled with logit adjustment** (Menon et al., 2021) and focal loss, not by stacking oversampling, weighted sampling and class weights. Stacking all three gave the model a uniform prior and collapsed benign recall to 0.29.
- **Inference adds the training-set class prior back to the logits.** This is required: without it the same checkpoint scores macro-F1 0.64 instead of 0.77.
- **Rare web attacks are merged for training** and separated afterwards by a small dedicated classifier, since three classes with a few hundred samples each are too rare to learn jointly.

The full reasoning behind each choice is in [docs/LSTM_METHODOLOGY.md](docs/LSTM_METHODOLOGY.md).

## Repository layout

| Path | Contents |
|---|---|
| `ai_detection/` | Preprocessing, training scripts, evaluation, inference API, trained checkpoint |
| `big_data_spark/` | Scala/Spark batch pipeline (sbt project) |
| `data_pipeline/` | Initial cleaning and visualisation scripts |
| `blockchain/` | MultiChain logger, reader and `log_prediction()` interface |
| `dashboard/` | Streamlit app (7 pages), Kafka producer/worker, Spark streaming job |
| `docs/` | Methodology write-up and LaTeX report sources |
| `data/` | Dataset download instructions (raw data is not in git) |

## Quick start

Requires Python 3.10+. Docker, Java and MultiChain are optional: the dashboard detects what is available and falls back to a local simulator and a local hash chain for the rest.

### Dashboard

```bash
pip install -r dashboard/requirements.txt
cd dashboard
streamlit run app.py
```

On Windows, `start.bat` does the same. The dashboard interface is in French.

### Inference on a CSV of flows

```bash
pip install -r ai_detection/requirements.txt
python ai_detection/src/inference/predict.py --input flows.csv
```

```
[0000] Infiltration              CRITICAL  conf=0.502
[0001] DDoS-HOIC                 CRITICAL  conf=0.974
[0002] DoS-Hulk                  CRITICAL  conf=0.993
[0003] Benign                    INFO      conf=0.603
```

From Python:

```python
import sys
sys.path.insert(0, "ai_detection")
from src.inference.predict import load_predictor

predictor = load_predictor()
result = predictor.predict(feature_vector)   # 47 unscaled features, see feature_names.json
# {"label": "DDoS-HOIC", "severity": "CRITICAL", "confidence": 0.97, "timestamp": ..., "feature_hash": ...}
```

### Retraining

Download the dataset as described in [data/README.md](data/README.md), then from `ai_detection/`:

```bash
python src/data_processing/preprocess.py
python src/training/train_v2.py
```

Training was done on a single 8 GB GPU. See [ai_detection/README.md](ai_detection/README.md) for details.

### Optional services

```bash
docker compose up -d                       # Kafka broker for the real-time pipeline
cd big_data_spark && sbt run               # Spark batch pipeline (Java 11+, sbt 1.9+)
python blockchain/multichain_setup.py      # MultiChain stream setup (MultiChain 2.3.3)
```

## Documentation

- [docs/LSTM_METHODOLOGY.md](docs/LSTM_METHODOLOGY.md): preprocessing and training methodology ([PDF](docs/LSTM_METHODOLOGY.pdf))
- [ai_detection/README.md](ai_detection/README.md): model component
- [dashboard/README.md](dashboard/README.md): dashboard and Kafka pipeline (French)
- [blockchain/README.md](blockchain/README.md): blockchain module (French)
- [big_data_spark/README.md](big_data_spark/README.md): Spark pipeline (French)
- [docs/report/](docs/report/): LaTeX sources of the project report

## Team

| Member | Contribution |
|---|---|
| Walid Ichchou | Detection model: preprocessing, CNN-BiLSTM training, evaluation, inference API |
| Noussair Fenani | Spark batch pipeline, Streamlit dashboard, Kafka and Spark streaming integration |
| Meryem Ait Tadamna | Blockchain audit layer (MultiChain) |
| Safaa Khalladi | Data pipeline and initial preprocessing |

## Dataset

CSE-CIC-IDS2018, Canadian Institute for Cybersecurity. Sharafaldin, Lashkari and Ghorbani, *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP 2018.

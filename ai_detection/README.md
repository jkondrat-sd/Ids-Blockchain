# LSTM Intrusion Detection System — IDS-Blockchain

Multi-class network intrusion detection on the CSE-CIC-IDS2018 dataset using a CNN-BiLSTM model. Detects 15 attack classes in real-time and feeds tamper-evident alert blocks to a blockchain audit layer.

**Best result:** macro-F1 **0.7711** (CNN-BiLSTM v2, 15 classes, natural-prior inference)

---

## Architecture

```
Raw CSVs (CIC-IDS-2018, ~16 GB)
        │
        ▼
src/data_processing/preprocess.py
        │  per-flow feature cleaning, scaling, stratified split
        ▼
data/processed/  (X_*.npy, y_*.npy, scaler.pkl, label_encoder.pkl …)
        │
        ▼
src/training/train_v2.py          ← CNN-BiLSTM v2 (production)
        │  focal loss + logit-adjustment + EMA
        ▼
models/bilstm_ids_v2.pt + web_splitter_v2.pkl
        │
        ├──▶ dashboard/app.py        (Streamlit — benchmark + live scan + blockchain tab)
        │
        └──▶ src/inference/predict.py
                    │
                    ▼
             blockchain_interface/api.py  → smart contract (log_prediction)
```

**Core design rule:** one network flow = one sample. No sliding windows. CICFlowMeter already reduces raw packets to ~80 per-flow statistical features. The feature vector is reshaped to `(T, 4)` for the LSTM — not a real time series.

---

## Folder Structure

```
├── src/
│   ├── data_processing/preprocess.py   ← preprocessing pipeline
│   ├── training/
│   │   ├── train_v2.py                 ← CNN-BiLSTM (best, production)
│   │   ├── train_advanced.py           ← deeper BiLSTM (Infiltration focus)
│   │   └── train_v3.py                 ← CNN-LSTM surgical test
│   ├── evaluation/
│   │   ├── eval_v2_natural.py          ← natural-prior diagnostic
│   │   └── sweep_v2_tau.py             ← tau hyperparameter sweep
│   ├── inference/predict.py            ← IDSPredictor class + CLI
│   └── utils/preprocessing_utils.py
├── dashboard/
│   ├── app.py                          ← main dashboard (blockchain Tab 6)
│   └── ids_console.py                  ← SOC monitoring console
├── blockchain_interface/api.py         ← log_prediction() stub
├── configs/default.yaml                ← all hyperparameters
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   └── 02_training.ipynb
├── models/                             ← checkpoints (tracked in git)
├── data/
│   ├── raw/                            ← gitignored; paste CSVs here
│   └── processed/                      ← inference artifacts tracked; tensors gitignored
└── outputs/                            ← gitignored; training plots/reports
```

---

## Setup

```bash
# 1. Create and activate virtualenv (Python 3.10)
python -m venv tf-gpu
tf-gpu\Scripts\activate          # Windows
source tf-gpu/Scripts/activate   # Linux / Git Bash

# 2. Install dependencies
pip install -r requirements.txt

# 3. GPU (CUDA 12.1) — replace the torch line above with:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> **VRAM limit:** keep `BATCH_SIZE <= 2048` with AMP, `<= 1024` without. Exceeding this causes GPU TDR BSODs on 8 GB cards. If a BSOD occurs, scan `tf-gpu/Lib/site-packages/` for null-byte corruption before debugging anything else.

---

## How to Train

Place the 10 CIC-IDS-2018 CSV files in `data/raw/` (filenames: `*TrafficForML_CICFlowMeter.csv`).

```bash
# Step 1 — Preprocess (~5 min, writes data/processed/)
python src/data_processing/preprocess.py

# Step 2 — Train CNN-BiLSTM v2 (best model, ~30 min on GPU)
python src/training/train_v2.py
# Writes: models/bilstm_ids_v2.pt, models/web_splitter_v2.pkl, outputs/v2_*

# Alternative: exploratory notebooks
jupyter nbconvert --to notebook --execute notebooks/01_preprocessing.ipynb --output notebooks/01_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_training.ipynb --output notebooks/02_training.ipynb
```

### Model variants

| Script | Checkpoint | Macro-F1 | Notes |
|---|---|---|---|
| `train_v2.py` | `bilstm_ids_v2.pt` | **0.7711** | Production — CNN-BiLSTM + logit-adjustment |
| `train_advanced.py` | `bilstm_ids_advanced.pt` | ~0.74 | 3-layer BiLSTM, targets Infiltration |
| `02_training.ipynb` | `bilstm_ids.pt` | 0.7366 | Baseline BiLSTM |

---

## How to Run Inference

```bash
# Single CSV file -> predictions printed to stdout
python src/inference/predict.py --input my_flows.csv

# Save predictions to JSON
python src/inference/predict.py --input my_flows.csv --output predictions.json
```

CSV must contain the feature columns in `data/processed/feature_names.json` (unscaled — the predictor applies the scaler internally).

Python API:
```python
from src.inference.predict import load_predictor

predictor = load_predictor()
result = predictor.predict(feature_vector)
# result = {"label": "DDoS-HOIC", "severity": "CRITICAL",
#           "confidence": 0.97, "timestamp": "...", "feature_hash": "..."}
```

---

## How the Dashboard Works

```bash
# Main dashboard (benchmark + live scan + blockchain audit tab)
streamlit run dashboard/app.py

# SOC monitoring console (alert inspector + synthetic log generator)
streamlit run dashboard/ids_console.py
```

`dashboard/app.py` Tab 6 ("Blockchain Audit") implements a local SHA-256 chained ledger and is wired to `blockchain_interface/api.py`.

---

## Blockchain Integration

### What the ML side provides (`blockchain_interface/api.py`)

```python
from blockchain_interface.api import log_prediction
from src.inference.predict import load_predictor

predictor = load_predictor()
prediction = predictor.predict(feature_vector)
block = log_prediction(prediction)
```

`log_prediction()` returns:
```json
{
  "block_idx":    0,
  "payload":      {"label": "DDoS-HOIC", "severity": "CRITICAL", "confidence": 0.97,
                   "timestamp": "2026-04-17T12:00:00+00:00", "feature_hash": "abc..."},
  "payload_hash": "sha256(payload)",
  "prev_hash":    "0000...0000",
  "block_hash":   "sha256(prev_hash + payload_hash + block_idx)",
  "on_chain":     false
}
```

### What the blockchain teammate must implement

Open `blockchain_interface/api.py` and fill in `_submit_to_chain()`:

```python
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

tx = contract.functions.logAlert(
    bytes.fromhex(block_hash),
    bytes.fromhex(payload_hash),
    bytes.fromhex(prev_hash),
).transact({"from": w3.eth.accounts[0]})
w3.eth.wait_for_transaction_receipt(tx)
```

Only hashes go on-chain. The full `payload` dict is stored off-chain (IPFS, your DB).

---

## Key Results

```
Model:        CNN-BiLSTM v2 (bilstm_ids_v2.pt)
Dataset:      CSE-CIC-IDS2018, 15 classes
Accuracy:     0.9076
Macro-F1:     0.7711
Weighted-F1:  0.8955
```

| Class | F1 |
|---|---|
| Botnet | 0.99 |
| DDoS-HOIC / LOIC-HTTP | 1.00 |
| DoS-GoldenEye / Hulk / Slowloris | 0.98–1.00 |
| SSH-BruteForce | 1.00 |
| Infiltration | 0.30 (hardest — 18k samples, overlaps Benign) |
| BruteForce-Web/XSS, SQL-Injection | 0.21–0.47 (low support) |

---

## Known Issues

- **Rebalancing rule:** never stack SMOTE + `WeightedRandomSampler` + class-weighted focal loss simultaneously. Doing so collapses Benign recall to ~0.29. See `PIPELINE.md §6` for the full autopsy.
- **VRAM:** `BATCH_SIZE > 2048` triggers GPU TDR on 8 GB cards. After a BSOD, null bytes corrupt `.py` files in the venv — scan with `grep -rla $'\x00' tf-gpu/Lib/site-packages/` before debugging.

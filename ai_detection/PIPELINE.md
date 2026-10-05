# CSE-CIC-IDS2018 — BiLSTM Intrusion Detection Pipeline

> **Note on paths.** This document was written when the model code lived in a single `lstm/` folder. In this repository the same files are under `ai_detection/` (`lstm/train_v2.py` is `ai_detection/src/training/train_v2.py`, `lstm/processed/` is `ai_detection/data/processed/`, `lstm/models/` is `ai_detection/models/`, `lstm/dashboard.py` was replaced by `dashboard/app.py`).

Full documentation of the preprocessing + training pipeline for the multi-class
LSTM IDS. Everything lives inside `lstm/`; no other folder in this repo is
authoritative.

---

## 1. Folder layout

```
lstm/
├── 01_preprocessing.ipynb     # data cleaning + feature eng. + splits + scaling
├── 02_training.ipynb          # BiLSTM + focal loss + evaluation
├── train_advanced.py          # script for the "flawless demo" retrain
├── dashboard.py               # Streamlit probe dashboard
├── PIPELINE.md                # this file
├── processed/                 # outputs of 01_preprocessing.ipynb
│   ├── X_train.npy / X_val.npy / X_test.npy
│   ├── y_train.npy / y_val.npy / y_test.npy
│   ├── class_weights.npy
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── class_names.json
│   ├── feature_names.json
│   └── class_distribution.png
├── models/
│   └── bilstm_ids.pt          # trained checkpoint (state_dict + config)
└── outputs/                   # reports + plots from 02_training.ipynb
    ├── classification_report.txt
    ├── classification_report_prior_adjusted.txt
    ├── confusion_matrix.png
    ├── confusion_matrix_prior_adjusted.png
    ├── per_attack_f1.png
    ├── per_attack_f1_prior_adjusted.png
    ├── pr_per_class.png
    ├── roc_per_class.png
    ├── training_curves.png
    └── history.json
```

Raw CSVs stay outside this folder (`../*TrafficForML_CICFlowMeter.csv`) — the
preprocessing notebook discovers them by glob.

---

## 2. The design rule that drives everything

**One flow = one sample. No sliding windows.**

CICFlowMeter already aggregates packet sequences into ~80 per-flow statistical
features (duration, packet counts, byte counts, flag counts, inter-arrival-time
stats, …). A row is already "the temporal content of one flow collapsed into
features." An earlier pipeline (`lstm_ids_gpu_full_pipeline_*.ipynb`, now deleted)
built sliding LSTM windows of size 15 over **row-shuffled** data. Adjacent rows
had no temporal relationship, so the "sequence" was random unrelated flows and
the LSTM had nothing real to learn — it hallucinated confident predictions.

The fix: feed one flow at a time, but reshape its feature vector as a
pseudo-sequence so an LSTM can still learn local feature interactions:

```
feature vector F  ──pad to multiple of C──►  (T = F/C, C)   where C = 4
```

This is the approach used in Ramyaelineni 2020, Wang 2023 and Gocs & Johanyak
2024. Pair it with a class-weighted loss and stratified splits to cope with
heavy imbalance.

---

## 3. Preprocessing — `01_preprocessing.ipynb`

### 3.1 Inputs

* 10 CSE-CIC-IDS2018 CSVs (`*TrafficForML_CICFlowMeter.csv`), ~16M flows.
* Located one level above the project root (`../` relative to `lstm/`).

### 3.2 Step-by-step

| # | Step | Why |
|---|------|-----|
| 1 | **Stream CSVs in chunks** (`CHUNK_SIZE = 200_000`) | Full dataset is >12 GB; reading all at once OOMs. |
| 2 | **Strip column whitespace**, intersect columns across all CSVs | CICFlowMeter column names differ subtly between files. |
| 3 | **Drop metadata cols** (`Timestamp, Flow ID, Src/Dst IP, Src/Dst Port, Protocol`) | Leakage. IPs & timestamps let the model memorize attackers instead of learning traffic patterns. |
| 4 | **Drop 10 known zero-variance columns** (`Bwd PSH Flags`, `Fwd URG Flags`, …) | Confirmed zero-variance in Gocs & Johanyak 2024; add nothing. |
| 5 | **Coerce features to numeric**, drop `Inf` / `NaN` / negative-duration rows, drop duplicates | CICFlowMeter emits a few Inf values in rate columns; duplicates skew training. |
| 6 | **Normalize labels** via `LABEL_MAP` | Merges typos (`Infilteration → Infiltration`, `SSH-Bruteforce → SSH-BruteForce`, `Bot → Botnet`, `Brute Force -Web → BruteForce-Web`, …). |
| 7 | **Cap rows per class** — `BENIGN_CAP = 400_000`, `MAX_ROWS_PER_CLASS = 200_000`, drop classes with < 50 rows | Keeps RAM under 6 GB and reduces (not removes) imbalance. Benign kept larger because it's realistic noise, not artificial. |
| 8 | **Drop residual zero-variance** after capping | Safety net. |
| 9 | **Drop features with \|corr\| > 0.98** | Removes redundant byte-count / packet-count twins that blow up the LSTM input dim for no signal. |
| 10 | **Winsorize** every feature at its 1st / 99th percentile | CIC-IDS2018 has extreme outliers (flow durations > 10⁸ µs). Clipping stops the scaler from squashing useful range. |
| 11 | **Label-encode** classes (`LabelEncoder`) → integer targets. |
| 12 | **Stratified 70 / 15 / 15 split** (`train / val / test`) | Stratification is mandatory — some classes have <100 flows. |
| 13 | **`StandardScaler` fit on train only**, transform val/test | No information leakage from val/test into the scaler. |
| 14 | **`compute_class_weight('balanced')`** on train labels | Saved to `class_weights.npy`, consumed by the training loss. |
| 15 | **Save** tensors + scaler + encoder + feature list + class list to `processed/` | Training notebook loads these and never touches raw CSVs. |

### 3.3 Final feature count

After the pipeline, the flow feature vector sits at roughly **47 numeric
features** (exact count depends on correlation cut-off). The training notebook
pads this to a multiple of 4 and reshapes to `(T, 4)`.

### 3.4 Artifacts written

```
processed/X_train.npy    (N_train, F) float32
processed/X_val.npy      (N_val,   F) float32
processed/X_test.npy     (N_test,  F) float32
processed/y_*.npy        int64 label indices
processed/class_weights.npy  float32, shape (num_classes,)
processed/scaler.pkl         fitted StandardScaler
processed/label_encoder.pkl  fitted LabelEncoder
processed/class_names.json
processed/feature_names.json
processed/class_distribution.png
```

---

## 4. Training — `02_training.ipynb`

### 4.1 Data loading

```python
CHUNK  = 4
PAD_TO = ceil(num_features / CHUNK) * CHUNK
T_STEPS = PAD_TO / CHUNK
# X reshaped from (N, F) → (N, T_STEPS, CHUNK)
```

`FlowDataset` right-pads the feature vector with zeros and reshapes. One flow
becomes one `(T, 4)` tensor. Yields `(x, y)` pairs on demand; tensors stay on
CPU and move to GPU inside the training loop.

### 4.2 Model — `BiLSTMClassifier`

```
Input (B, T, 4)
 └─ BiLSTM(input=4, hidden=128, layers=2, bidirectional=True, dropout=0.30)
     → output (B, T, 256)
 └─ Attention pooling
     scores = v · tanh(W · h_t)          # shape (B, T, 1)
     a = softmax(scores, dim=T)
     pooled = Σ_t a_t · h_t              # (B, 256)
 └─ Head
     Linear(256 → 64) → ReLU → Dropout(0.30) → Linear(64 → num_classes)
```

Parameter count ≈ **500 K**. Tiny by deep-learning standards — the whole point
is that the feature space is already distilled by CICFlowMeter.

### 4.3 Loss — focal loss with class-weighted alpha

```python
focal_alpha = sqrt(class_weights) / mean(sqrt(class_weights))
FocalLoss(alpha=focal_alpha, gamma=2.0)
```

* `gamma=2` is the standard focal-loss down-weighting of easy examples.
* `alpha=sqrt(weights)` is a **gentler** rebalancing than raw inverse frequency.
  Raw weights over-correct on CIC-IDS2018 and collapse Benign recall. See §6.

### 4.4 Optimizer + schedule

* **AdamW**, `lr=1e-3`, `weight_decay=1e-5`.
* **OneCycleLR**, `max_lr=1e-3`, over `EPOCHS × steps_per_epoch` iters.
* **Early stopping** on val macro-F1, `patience=8`.

### 4.5 Rebalancing policy (Option B)

```
USE_SAMPLER      = False    # no WeightedRandomSampler
TARGET_MIN       = 3_000    # modest SMOTE floor for rare classes
FOCAL_GAMMA      = 2.0
focal_alpha      = sqrt(class_weights) / mean
```

**Rule: pick at most two of {SMOTE, sampler, class-weighted focal}.** See §6
for the autopsy of why.

### 4.6 Training loop

```
for epoch in 1..30:
    train one pass (focal loss + AdamW + OneCycleLR step per batch)
    eval on val -> macro-F1
    if macro-F1 > best: save best_state, reset patience
    else: patience -= 1; stop if 0
load best_state
evaluate on test
```

Outputs: classification report, confusion matrix, per-class PR curves, per-class
ROC curves, per-attack F1 bar chart, history JSON, training curves, checkpoint.

### 4.7 Checkpoint format

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': CHUNK,            # 4
    'hidden':    HIDDEN,           # 128
    'num_layers': LSTM_LAYERS,     # 2
    'num_classes': num_classes,    # 15
    'dropout':   DROPOUT,          # 0.30
    'pad_to':    PAD_TO,
    'chunk':     CHUNK,
    'num_features': num_features,
}, 'models/bilstm_ids.pt')
```

The dashboard and `train_advanced.py` both consume this format.

---

## 4b. Advanced architecture — `train_advanced.py`

The baseline notebook plateaus around macro-F1 0.74 with **Infiltration** stuck
at F1 ≈ 0.30 — that single class is the demo blocker (18 k test flows, dense
support, just hard to separate from Benign). `train_advanced.py` is the
"flawless demo" pass that targets it, while reusing the exact same processed
tensors from `lstm/processed/`. It is **not** a replacement for the baseline —
it writes its own checkpoint at `lstm/models/bilstm_ids_advanced.pt`.

### 4b.1 Architecture diagram

```
Input (B, T=12, 4)
 │
 ├─ Linear projection 4 → 192                              # lift to model dim
 │
 ├─ BiLSTM × 3 layers, hidden = 192, bidirectional         # deeper temporal mixer
 │     dropout 0.30 between layers
 │     output shape (B, T, 384)                            #  = 192 × 2
 │
 ├─ Residual:  LayerNorm( BiLSTM_out + Linear(proj → 384) )
 │     short-circuit so the projection survives 3 LSTM layers
 │
 ├─ Multi-head self-attention pool, 4 heads, dim 384       # pooled (B, 384)
 │     learned query attends across all T timesteps
 │     LayerNorm on the pooled vector
 │
 └─ Head:  Linear(384 → 128) → GELU → Dropout(0.30) → Linear(128 → 15)
```

Parameter count ≈ **1.7 M** — still tiny, but ~3× the baseline.

### 4b.2 Why each piece

| Component | Why it's there |
|---|---|
| **3 BiLSTM layers (vs 2)** | The baseline 2-layer net saturates well before overfitting. Adding a third layer gives the network capacity to learn the subtle Infiltration ↔ Benign boundary without blowing past VRAM. |
| **Hidden 192 (vs 128)** | Wider hidden state lets each timestep carry more disentangled feature interactions. 192 is the sweet spot: still 8 GB-friendly under AMP, divisible by 4 for the multi-head attention. |
| **Linear input projection 4 → 192** | The pseudo-sequence is only 4 features wide per step; projecting to model-dim before the LSTM means the recurrence operates on a richer representation than raw padded floats. |
| **Residual connection around the BiLSTM** | 3-layer LSTMs are notoriously hard to train end-to-end — gradients vanish through the recurrence. The skip from the projection to the post-LSTM tensor keeps the optimizer honest. |
| **Multi-head self-attention pool (vs single-query attention)** | The baseline uses one learned attention vector. With 4 heads the model can pool different "views" of the sequence in parallel — e.g. one head focused on duration features, another on byte-rate features. This is the same idea as Transformer pooling tokens. |
| **LayerNorm after residual and after pool** | Stabilizes the signal feeding the head; without it the deeper net's loss curves are visibly noisier. |
| **GELU in the head (vs ReLU)** | Smoother gradient near zero — small but consistent macro-F1 win on imbalanced multi-class problems. |

### 4b.3 Two-stage training

```
Stage 1: binary Benign-vs-Attack
    └─ 10 epochs, OneCycleLR, vanilla focal loss (alpha = [1, 1])
    └─ goal: teach the BiLSTM backbone "what is normal traffic"

Stage 2a: 15-class warm-start, backbone FROZEN
    └─ 2 epochs, only the head trains
    └─ goal: align the new 15-class head with the pretrained backbone
              before any backbone updates can wreck it

Stage 2b: 15-class fine-tune, backbone UNFROZEN
    └─ 38 epochs, half learning rate (LR × 0.5), CosineAnnealingLR
    └─ early stopping on val macro-F1, patience 10
    └─ goal: refine the full network on the real objective
```

The stage-1 backbone is copied tensor-by-tensor into stage 2 — only the final
classifier head is re-initialized. This warm-start cuts ~5 epochs off
convergence and gives the model a much better starting prior on
Benign-vs-Attack than random init.

### 4b.4 Training tricks

| Trick | What it does |
|---|---|
| **AMP** (`torch.amp` mixed precision) | Halves VRAM, lets us run `BATCH_SIZE = 2048` safely on the 8 GB RTX 2070 Max-Q. No measurable F1 cost. |
| **EMA of weights** (`decay = 0.999`) | Eval and best-checkpoint use the exponential moving average of parameters, not the raw end-of-epoch weights. Smooths out noisy late-training oscillation; the EMA usually beats the raw weights by 0.5–1 F1 point. |
| **Grad-norm clipping at 1.0** | Standard safety net for deep BiLSTMs — avoids the occasional gradient spike that AMP scaling can amplify. |
| **OneCycleLR (stage 1) + Cosine annealing (stage 2b)** | OneCycle for the warmup phases, cosine for the long fine-tune — same recipe as most modern image classifiers. |

### 4b.5 Rebalancing — same Option B rule as the notebook

```python
SMOTE_FLOOR        = 3_000   # rare classes only, modest floor
USE_SAMPLER        = False   # never re-enable while focal alpha is on
FOCAL_GAMMA        = 2.0
focal_alpha        = sqrt(class_weights) / mean
INFILTRATION_BOOST = 1.0     # MUST stay 1.0 — see §6
LABEL_SMOOTH       = 0.0     # MUST stay 0.0 — see §6
```

**Anything that softens the train-time prior re-creates the triple-stack bug.**
A `SMOTE_FLOOR` above 3 000, an Infiltration alpha boost, label smoothing on
the 15-class head, or re-enabling `WeightedRandomSampler` will all collapse
Benign recall the same way — see §6 for the autopsy and the 2026-04-13
diagnostic run that confirmed it twice.

### 4b.6 Outputs

```
lstm/models/bilstm_ids_advanced.pt         # checkpoint, same format as §4.7
lstm/outputs/advanced_classification_report.txt
lstm/outputs/advanced_confusion_matrix.png
lstm/outputs/advanced_history.json         # stage1 / stage2_warm / stage2_main
```

The dashboard auto-detects this checkpoint when it is present.

---

## 5. Current results (as of 2026-04-13)

From `outputs/classification_report.txt` (Option B retrain, no prior hack):

```
accuracy           0.9076
macro-F1           0.7366
weighted-F1        0.8955
```

Per-class F1 highlights:

| Class              | F1   | Notes |
|--------------------|------|-------|
| Benign             | 0.85 | OK — triple-stack bug fixed |
| Botnet             | 0.99 | production-ready |
| DDoS-HOIC          | 1.00 | production-ready |
| DDoS-LOIC-HTTP     | 1.00 | production-ready |
| DDoS-LOIC-UDP      | 0.89 | OK |
| DoS-GoldenEye      | 1.00 | production-ready |
| DoS-Hulk           | 1.00 | production-ready |
| DoS-Slowloris      | 0.98 | production-ready |
| SSH-BruteForce     | 1.00 | production-ready |
| **Infiltration**   | **0.30** | **blocker — 18k test samples, model can't separate from Benign** |
| BruteForce-Web     | 0.47 | low-support noise |
| BruteForce-XSS     | 0.21 | low-support noise |
| SQL-Injection      | 0.35 | low-support noise |
| FTP-BruteForce     | 0.57 | low-support noise |
| DoS-SlowHTTPTest   | 0.43 | low-support noise |

The demo blocker is **Infiltration**, not the rare web classes — 18 k real
samples at 0.30 F1 is a learning problem, not a data-scarcity problem. See
`train_advanced.py` for the "flawless demo" pass targeting it.

---

## 6. The triple-stack rebalancing bug (history)

On 2026-04-13 the first full run used `TARGET_MIN=10_000`, `USE_SAMPLER=True`
and focal `alpha = sqrt(weights)`. Three rebalancing layers stacked gave the
model an effectively uniform training prior, so at test time (~28% Benign) it
refused to predict Benign: **Benign recall = 0.29, precision = 0.99**, and the
tiny classes (SQL-Injection, BruteForce-Web/XSS, Infiltration) spammed false
positives to "catch" the leaked Benign flows.

A Bayes prior correction at inference (add `log P_natural(y)` to the logits,
estimated from the val set) lifted macro-F1 from 0.6555 → 0.7045 and Benign
recall from 0.29 → 0.99 — **proving the training prior was the cause, not the
model's capacity.** The notebook was then switched to Option B (sampler off,
TARGET_MIN=3 000, gentler alpha) and retrained — that is the current result.

Rule encoded in the training notebook:
* never re-enable `WeightedRandomSampler` while SMOTE + class-weighted focal
  loss are also on;
* Bayes prior correction at inference is a **diagnostic**, not a cure —
  `classification_report_prior_adjusted.txt` is kept only for comparison;
* if Benign recall ever drops while its precision stays high, that's the
  signature of a training / deployment prior mismatch — fix the training prior.

---

## 7. Running the pipeline from scratch

```bash
# 1. activate the venv (torch 2.5.1+cu121 on RTX 2070 Max-Q)
source tf-gpu/Scripts/activate           # Windows: tf-gpu\Scripts\activate

# 2. preprocessing (runs once, ~5 min, writes lstm/processed/)
jupyter nbconvert --to notebook --execute lstm/01_preprocessing.ipynb \
    --output 01_preprocessing.ipynb

# 3. baseline training (~30 min, writes lstm/models/ + lstm/outputs/)
jupyter nbconvert --to notebook --execute lstm/02_training.ipynb \
    --output 02_training.ipynb

# 4. advanced retrain targeting the weak classes
python lstm/train_advanced.py

# 5. probe dashboard
streamlit run lstm/dashboard.py
```

---

## 8. Known environment hazards

* **GPU TDR BSODs corrupt `.py` files in site-packages with null bytes.** After
  any crash, scan `tf-gpu/Lib/site-packages/` for `\x00` in `.py` files before
  debugging anything else. Recovery: delete the corrupted package folder and
  reinstall torch via the cu121 index.
* RTX 2070 Max-Q has 8 GB VRAM. Keep `BATCH_SIZE ≤ 1024` without AMP, or
  ≤ 2048 with AMP, otherwise the TDR triggers again.

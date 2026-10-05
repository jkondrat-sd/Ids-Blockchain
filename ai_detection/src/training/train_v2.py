"""
train_v2.py — literature-driven rewrite of train_advanced.py.

Deltas from train_advanced.py (rationale in PIPELINE.md §4c):
  * merge {BruteForce-Web, BruteForce-XSS, SQL-Injection} into a single
    `WebAttack` class for training (13 classes instead of 15). With only
    12/34/86 test flows for these three, flat classification is mathematically
    hopeless — literature consistently reports near-0 F1 on web attacks.
    A tiny LogisticRegression splitter restores the 3-way breakdown at
    inference, so the 15-class macro-F1 is still reported for apples-to-apples
    comparison with the notebook baseline (0.7366).
  * drop SMOTE. Replace it with training-time logit adjustment
    (Menon et al. 2021, NeurIPS) — add tau * log(pi_c) to the logits before
    the loss. Cleaner than synthetic oversampling, no inflation artifacts.
  * focal loss with gamma=2.0, **alpha uniform** (no sqrt-weight boost).
    Respects the "<= 2 rebalancers" memory rule: logit-adjustment is the
    sole class-rebalancing lever.
  * CNN front-end (2x Conv1d, k=3) before the LSTM. Literature consistently
    shows CNN-LSTM > pure BiLSTM for CIC-IDS-2018.
  * shallower LSTM (2 layers, hidden=128) — 3x192 was overparameterized for
    ~12 timesteps.
  * optional seed ensemble (NUM_SEEDS, default 1). Seeds 42, 7, 2024 if >1.

Writes:
    lstm/models/bilstm_ids_v2.pt             (first seed only, for dashboard)
    lstm/models/bilstm_ids_v2_seed{S}.pt     (all seeds, if ensemble)
    lstm/models/web_splitter_v2.pkl          (3-way logistic splitter)
    lstm/outputs/v2_classification_report_13.txt   (training metric)
    lstm/outputs/v2_classification_report_15.txt   (vs baseline)
    lstm/outputs/v2_confusion_matrix_15.png
    lstm/outputs/v2_history.json
"""

from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 2048
EPOCHS = 30
LR = 1.5e-3
WEIGHT_DECAY = 2e-5
HIDDEN = 128
LSTM_LAYERS = 2
CNN_CH = 128
DROPOUT = 0.30
N_HEADS = 4
CHUNK = 4
FOCAL_GAMMA = 2.0
LA_TAU = 1.0                 # logit-adjustment strength (Menon et al.)
EMA_DECAY = 0.999
PATIENCE = 8
SEEDS = [42]                 # set to [42, 7, 2024] for a 3-seed ensemble

WEB_CLASSES = ("BruteForce-Web", "BruteForce-XSS", "SQL-Injection")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE} batch={BATCH_SIZE} hidden={HIDDEN} seeds={SEEDS}")


# --------------------------------------------------------------------------- #
# Label remapping (15 -> 13)
# --------------------------------------------------------------------------- #


def build_label_map(class_names):
    web_idx = [class_names.index(c) for c in WEB_CLASSES]
    merged_names = [c for c in class_names if c not in WEB_CLASSES]
    merged_names.append("WebAttack")
    web_merged_idx = len(merged_names) - 1
    label_map = np.zeros(len(class_names), dtype=np.int64)
    for i, c in enumerate(class_names):
        label_map[i] = web_merged_idx if c in WEB_CLASSES else merged_names.index(c)
    return merged_names, label_map, web_idx, web_merged_idx


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_processed():
    X_train = np.load(PROC_DIR / "X_train.npy")
    X_val = np.load(PROC_DIR / "X_val.npy")
    X_test = np.load(PROC_DIR / "X_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_val = np.load(PROC_DIR / "y_val.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")
    class_names = json.loads((PROC_DIR / "class_names.json").read_text())
    return X_train, X_val, X_test, y_train, y_val, y_test, class_names


class FlowDataset(Dataset):
    def __init__(self, X, y, pad_to, chunk):
        n, f = X.shape
        if f < pad_to:
            X = np.pad(X, ((0, 0), (0, pad_to - f)), mode="constant")
        self.X = torch.from_numpy(
            X.reshape(n, pad_to // chunk, chunk).astype(np.float32)
        )
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(out.squeeze(1))


class CNNLSTM(nn.Module):
    def __init__(self, chunk, cnn_ch, hidden, lstm_layers, num_classes,
                 dropout, n_heads):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(chunk, cnn_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cnn_ch // 2, cnn_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            cnn_ch, hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pool = MultiHeadAttentionPool(hidden * 2, n_heads=n_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, chunk) -> (B, chunk, T)
        h = self.conv(x.transpose(1, 2))
        # (B, cnn_ch, T) -> (B, T, cnn_ch)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        return self.head(self.pool(h))


# --------------------------------------------------------------------------- #
# Loss (focal + logit-adjustment)
# --------------------------------------------------------------------------- #


class LAFocal(nn.Module):
    """Focal loss with training-time logit adjustment.

    Training: minimize CE-focal( logits + tau*log(pi_c), y ).
    Inference: drop the adjustment -> argmax(logits) gives class-balanced
    predictions, which is what macro-F1 rewards.
    """

    def __init__(self, log_pi, tau=1.0, gamma=2.0):
        super().__init__()
        self.register_buffer("log_pi", torch.as_tensor(log_pi, dtype=torch.float32))
        self.tau = tau
        self.gamma = gamma

    def forward(self, logits, target):
        adj = logits + self.tau * self.log_pi.unsqueeze(0)
        log_probs = F.log_softmax(adj, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-9)
        loss = -((1 - pt) ** self.gamma) * log_pt
        return loss.mean()


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        model.load_state_dict(self._backup)
        del self._backup


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(xb)
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(yb.numpy())
    return np.concatenate(all_logits), np.concatenate(all_y)


def evaluate_macro(model, loader, device):
    logits, y = collect_logits(model, loader, device)
    y_hat = logits.argmax(1)
    return f1_score(y, y_hat, average="macro", zero_division=0), logits, y


def train_one_seed(seed, X_tr, y_tr, X_va, y_va, pad_to, num_classes, log_pi):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = FlowDataset(X_tr, y_tr, pad_to, CHUNK)
    val_ds = FlowDataset(X_va, y_va, pad_to, CHUNK)
    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=0, pin_memory=pin)

    model = CNNLSTM(
        chunk=CHUNK, cnn_ch=CNN_CH, hidden=HIDDEN, lstm_layers=LSTM_LAYERS,
        num_classes=num_classes, dropout=DROPOUT, n_heads=N_HEADS,
    ).to(DEVICE)
    criterion = LAFocal(log_pi=log_pi, tau=LA_TAU, gamma=FOCAL_GAMMA).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    ema = EMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_f1, stall, best_state = -1.0, 0, None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            total += loss.item() * len(yb)
            n += len(yb)
        train_loss = total / n

        ema.apply_to(model)
        val_f1, _, _ = evaluate_macro(model, val_loader, DEVICE)
        ema.restore(model)

        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_f1": val_f1, "dt": dt})
        print(f"[seed {seed}] epoch {epoch:2d}/{EPOCHS}  train {train_loss:.4f}  "
              f"val-F1 {val_f1:.4f}  ({dt:.1f}s)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                print(f"[seed {seed}] early stop at epoch {epoch} "
                      f"(best val-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_f1


# --------------------------------------------------------------------------- #
# Web-attack splitter (3-way logistic regression on original features)
# --------------------------------------------------------------------------- #


def train_web_splitter(X, y_orig, web_idx, seed):
    mask = np.isin(y_orig, web_idx)
    Xw, yw = X[mask], y_orig[mask]
    uniq, cnt = np.unique(yw, return_counts=True)
    print(f"[splitter] training on {len(Xw)} web flows: "
          f"{dict(zip(uniq.tolist(), cnt.tolist()))}")
    if len(uniq) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=seed)
    clf.fit(Xw, yw)
    return clf


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_processed()
    merged_names, label_map, web_idx, web_merged_idx = build_label_map(class_names)
    print(f"[remap] 15 -> 13 classes, merged: {WEB_CLASSES} -> 'WebAttack' (idx {web_merged_idx})")

    # remap labels to 13-class space
    y_train_m = label_map[y_train]
    y_val_m = label_map[y_val]
    y_test_m = label_map[y_test]
    num_classes = len(merged_names)

    # class priors from TRAINING distribution (13-class)
    counts = np.bincount(y_train_m, minlength=num_classes).astype(np.float64)
    pi = counts / counts.sum()
    log_pi = np.log(pi + 1e-12).astype(np.float32)
    print("[prior] pi (13-class training):")
    for c, p in zip(merged_names, pi):
        print(f"  {c:18s} {p:.6f}  (n={int(p * counts.sum()):,})")

    num_features = X_train.shape[1]
    pad_to = ((num_features + CHUNK - 1) // CHUNK) * CHUNK
    t_steps = pad_to // CHUNK
    print(f"[data] features={num_features} pad_to={pad_to} t_steps={t_steps}")
    print(f"[data] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    # train N seeds
    seed_logits_val, seed_logits_test = [], []
    histories = {}
    first_model = None
    for seed in SEEDS:
        print(f"\n========= Training seed {seed} =========")
        model, hist, best_f1 = train_one_seed(
            seed, X_train, y_train_m, X_val, y_val_m,
            pad_to, num_classes, log_pi,
        )
        histories[seed] = hist
        # collect logits on val + test
        val_ds = FlowDataset(X_val, y_val_m, pad_to, CHUNK)
        test_ds = FlowDataset(X_test, y_test_m, pad_to, CHUNK)
        pin = DEVICE.type == "cuda"
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                pin_memory=pin)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                                 pin_memory=pin)
        v_logits, _ = collect_logits(model, val_loader, DEVICE)
        t_logits, _ = collect_logits(model, test_loader, DEVICE)
        seed_logits_val.append(v_logits)
        seed_logits_test.append(t_logits)

        ckpt_path = MODEL_DIR / (
            "bilstm_ids_v2.pt" if seed == SEEDS[0]
            else f"bilstm_ids_v2_seed{seed}.pt"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "arch": "CNNLSTM",
            "chunk": CHUNK,
            "cnn_ch": CNN_CH,
            "hidden": HIDDEN,
            "lstm_layers": LSTM_LAYERS,
            "num_classes": num_classes,
            "dropout": DROPOUT,
            "n_heads": N_HEADS,
            "pad_to": pad_to,
            "num_features": num_features,
            "class_names": merged_names,
            "orig_class_names": class_names,
            "label_map": label_map.tolist(),
            "web_idx": web_idx,
            "web_merged_idx": web_merged_idx,
            "seed": seed,
            "best_val_f1_13c": best_f1,
        }, ckpt_path)
        if first_model is None:
            first_model = model

    # ensemble logits (mean)
    val_logits = np.mean(seed_logits_val, axis=0)
    test_logits = np.mean(seed_logits_test, axis=0)
    val_pred_m = val_logits.argmax(1)
    test_pred_m = test_logits.argmax(1)

    # ========= 13-class metrics (training metric) =========
    f1_13 = f1_score(y_test_m, test_pred_m, average="macro", zero_division=0)
    report_13 = classification_report(
        y_test_m, test_pred_m, target_names=merged_names,
        digits=4, zero_division=0,
    )
    print(f"\n========= Test evaluation (13-class) =========")
    print(f"test macro-F1 (13-class): {f1_13:.4f}")
    print(report_13)
    (OUT_DIR / "v2_classification_report_13.txt").write_text(
        f"test macro-F1 (13-class): {f1_13:.4f}\n\n{report_13}"
    )

    # ========= Train web splitter + reconstruct 15-class predictions =========
    splitter = train_web_splitter(X_train, y_train, web_idx, SEEDS[0])
    if splitter is not None:
        with open(MODEL_DIR / "web_splitter_v2.pkl", "wb") as f:
            pickle.dump({"clf": splitter, "web_idx": web_idx,
                         "web_merged_idx": web_merged_idx}, f)

    # merged_idx -> original non-web class idx
    reverse_map = np.zeros(num_classes, dtype=np.int64)
    for orig_i, merged_i in enumerate(label_map):
        if class_names[orig_i] not in WEB_CLASSES:
            reverse_map[merged_i] = orig_i

    test_pred_15 = reverse_map[test_pred_m].copy()
    web_mask = test_pred_m == web_merged_idx
    if web_mask.any():
        if splitter is not None:
            test_pred_15[web_mask] = splitter.predict(X_test[web_mask])
        else:
            # fallback: assign most common web class in train
            uniq, cnt = np.unique(y_train[np.isin(y_train, web_idx)],
                                  return_counts=True)
            test_pred_15[web_mask] = uniq[cnt.argmax()] if len(uniq) else web_idx[0]

    f1_15 = f1_score(y_test, test_pred_15, average="macro", zero_division=0)
    report_15 = classification_report(
        y_test, test_pred_15, target_names=class_names,
        digits=4, zero_division=0,
    )
    print(f"\n========= Test evaluation (15-class, reconstructed) =========")
    print(f"test macro-F1 (15-class): {f1_15:.4f}   "
          f"(baseline notebook: 0.7366, advanced v1: 0.6750)")
    print(report_15)
    (OUT_DIR / "v2_classification_report_15.txt").write_text(
        f"test macro-F1 (15-class): {f1_15:.4f}\n\n{report_15}"
    )

    # ========= History + confusion matrix =========
    (OUT_DIR / "v2_history.json").write_text(json.dumps({
        "seeds": SEEDS,
        "histories": {str(s): h for s, h in histories.items()},
        "test_macro_f1_13c": f1_13,
        "test_macro_f1_15c": f1_15,
    }, indent=2))

    try:
        import matplotlib.pyplot as plt
        cm = confusion_matrix(y_test, test_pred_15, labels=list(range(len(class_names))))
        cm_n = cm.astype(float)
        cm_n /= cm_n.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"CNN-BiLSTM v2 — test macro-F1 (15c) {f1_15:.4f}")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if cm_n[i, j] > 0.01:
                    ax.text(j, i, f"{cm_n[i, j]:.2f}",
                            ha="center", va="center",
                            color="white" if cm_n[i, j] > 0.5 else "black",
                            fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "v2_confusion_matrix_15.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped confusion matrix: {e}")

    print("\n[done] checkpoints + splitter written to lstm/models/")
    print("[done] reports written to lstm/outputs/v2_*")


if __name__ == "__main__":
    main()

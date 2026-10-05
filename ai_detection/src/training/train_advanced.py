"""
Advanced training pass for the CIC-IDS2018 BiLSTM IDS.

Goal: go from the 02_training.ipynb baseline (macro-F1 ~0.74, Infiltration F1
~0.30) to a demo-grade model. This script does NOT replace the baseline
notebook — it reuses the same processed tensors and writes its own checkpoint.

What's different from 02_training.ipynb:

  * deeper BiLSTM (3 layers, 192 hidden) with residual connection before the
    attention pool
  * self-attention pool instead of single-query attention (multi-head, 4 heads)
  * single-stage 15-class training from random init (an earlier two-stage
    Benign-vs-Attack warmup biased the backbone toward a coarse binary
    representation and regressed Infiltration + rare-class precision)
  * AMP (torch.amp) — halves VRAM, lets us run batch 2048 safely on 8 GB
  * EMA of weights for eval (stability)
  * SMOTE tier floor = 3 000 for rare classes only, no sampler
  * Option B rebalancing: focal alpha = sqrt(class_weights) / mean, gamma 2.0,
    no label smoothing, no per-class alpha boosts
  * one AdamW + one OneCycleLR across the full run (no mid-training optimizer
    reset, no schedule restart)

Run:
    python lstm/train_advanced.py

Writes:
    lstm/models/bilstm_ids_advanced.pt
    lstm/outputs/advanced_classification_report.txt
    lstm/outputs/advanced_confusion_matrix.png
    lstm/outputs/advanced_history.json
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

SEED = 42
BATCH_SIZE = 2048
EPOCHS = 40
LR = 1.5e-3
WEIGHT_DECAY = 2e-5
HIDDEN = 192
LSTM_LAYERS = 3
DROPOUT = 0.30
N_HEADS = 4
CHUNK = 4
FOCAL_GAMMA = 2.0
LABEL_SMOOTH = 0.0
EMA_DECAY = 0.999
SMOTE_FLOOR = 3_000
PATIENCE = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[config] device={DEVICE} batch={BATCH_SIZE} hidden={HIDDEN} layers={LSTM_LAYERS}")


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
    class_weights = np.load(PROC_DIR / "class_weights.npy")
    class_names = json.loads((PROC_DIR / "class_names.json").read_text())
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights, class_names


def smote_rare_classes(X, y, floor, class_names, seed):
    """Light SMOTE on rare classes only — we don't touch classes above the floor."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[smote] imbalanced-learn not installed — skipping SMOTE")
        return X, y

    counts = np.bincount(y, minlength=len(class_names))
    targets = {c: floor for c, n in enumerate(counts) if 0 < n < floor}
    if not targets:
        return X, y
    print(f"[smote] lifting {len(targets)} rare classes to {floor} each")
    k = min(5, min(counts[c] for c in targets) - 1)
    if k < 1:
        print("[smote] not enough samples for SMOTE — skipping")
        return X, y
    sm = SMOTE(sampling_strategy=targets, k_neighbors=k, random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res.astype(np.float32), y_res


class FlowDataset(Dataset):
    def __init__(self, X, y, pad_to, chunk):
        n, f = X.shape
        if f < pad_to:
            X = np.pad(X, ((0, 0), (0, pad_to - f)), mode="constant")
        self.X = torch.from_numpy(X.reshape(n, pad_to // chunk, chunk).astype(np.float32))
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


class AdvancedBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        n_heads: int,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.residual = nn.Linear(hidden, hidden * 2)
        self.norm = nn.LayerNorm(hidden * 2)
        self.pool = MultiHeadAttentionPool(hidden * 2, n_heads=n_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        h, _ = self.lstm(z)
        h = self.norm(h + self.residual(z))
        pooled = self.pool(h)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# --------------------------------------------------------------------------- #
# Loss & EMA
# --------------------------------------------------------------------------- #


class FocalLossSmooth(nn.Module):
    def __init__(self, alpha, gamma: float = 2.0, smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_cls = logits.size(1)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (n_cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        probs = log_probs.exp()
        pt = (probs * true_dist).sum(dim=1).clamp(min=1e-9)
        log_pt = (log_probs * true_dist).sum(dim=1)
        at = self.alpha[target]
        loss = -at * (1 - pt) ** self.gamma * log_pt
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


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, n = 0.0, 0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            n += len(yb)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(yb.cpu().numpy())
    y_hat = np.concatenate(preds)
    y_ref = np.concatenate(trues)
    macro = f1_score(y_ref, y_hat, average="macro", zero_division=0)
    return total_loss / n, macro, y_hat, y_ref


def train_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    epochs, patience, ema, tag,
):
    best_f1, stall = -1.0, 0
    best_state = None
    history = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        model.train()
        t0, total_loss, n = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            ema.update(model)
            total_loss += loss.item() * len(yb)
            n += len(yb)

        train_loss = total_loss / n

        # eval with EMA weights
        ema.apply_to(model)
        val_loss, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device, num_classes=None
        )
        ema.restore(model)

        dt = time.time() - t0
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_f1": val_f1, "dt": dt}
        )
        print(
            f"[{tag}] epoch {epoch:2d}/{epochs}  train {train_loss:.4f}  "
            f"val {val_loss:.4f}  val-F1 {val_f1:.4f}  ({dt:.1f}s)"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                print(f"[{tag}] early stop at epoch {epoch} (best val-F1 {best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_f1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights, class_names = (
        load_processed()
    )
    num_classes = len(class_names)
    num_features = X_train.shape[1]
    pad_to = ((num_features + CHUNK - 1) // CHUNK) * CHUNK
    t_steps = pad_to // CHUNK
    print(f"[data] features={num_features} pad_to={pad_to} t_steps={t_steps}")
    print(f"[data] classes={num_classes} ({class_names})")
    print(f"[data] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    X_train, y_train = smote_rare_classes(
        X_train, y_train, SMOTE_FLOOR, class_names, SEED
    )
    print(f"[data] train after SMOTE floor={SMOTE_FLOOR}: {len(X_train):,}")

    train_ds = FlowDataset(X_train, y_train, pad_to, CHUNK)
    val_ds = FlowDataset(X_val, y_val, pad_to, CHUNK)
    test_ds = FlowDataset(X_test, y_test, pad_to, CHUNK)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=pin
    )

    # ---------- 15-class training (single stage, random init) ------------- #
    print("\n========= 15-class training =========")
    model = AdvancedBiLSTM(
        input_dim=CHUNK,
        hidden=HIDDEN,
        num_layers=LSTM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
        n_heads=N_HEADS,
    ).to(DEVICE)

    alpha = np.sqrt(class_weights).astype(np.float32)
    alpha = alpha / alpha.mean()
    criterion = FocalLossSmooth(alpha, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTH).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    ema = EMA(model, EMA_DECAY)

    model, hist_main, best_f1 = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE,
        epochs=EPOCHS, patience=PATIENCE, ema=ema, tag="train",
    )
    print(f"[train] best val macro-F1: {best_f1:.4f}")

    # ---------- Test evaluation ------------------------------------------- #
    print("\n========= Test evaluation =========")
    _, test_f1, y_hat, y_ref = evaluate(
        model, test_loader, criterion, DEVICE, num_classes
    )
    print(f"test macro-F1: {test_f1:.4f}")
    report = classification_report(
        y_ref, y_hat, target_names=class_names, digits=4, zero_division=0
    )
    print(report)

    # ---------- Save ------------------------------------------------------ #
    ckpt_path = MODEL_DIR / "bilstm_ids_advanced.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": "AdvancedBiLSTM",
            "input_dim": CHUNK,
            "hidden": HIDDEN,
            "num_layers": LSTM_LAYERS,
            "num_classes": num_classes,
            "dropout": DROPOUT,
            "n_heads": N_HEADS,
            "pad_to": pad_to,
            "chunk": CHUNK,
            "num_features": num_features,
            "class_names": class_names,
        },
        ckpt_path,
    )
    (OUT_DIR / "advanced_classification_report.txt").write_text(report)
    (OUT_DIR / "advanced_history.json").write_text(
        json.dumps(
            {
                "train": hist_main,
                "test_macro_f1": test_f1,
            },
            indent=2,
        )
    )

    try:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_ref, y_hat, labels=list(range(num_classes)))
        cm_n = cm.astype(float)
        cm_n /= cm_n.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Advanced BiLSTM — test macro-F1 {test_f1:.4f}")
        for i in range(num_classes):
            for j in range(num_classes):
                if cm_n[i, j] > 0.01:
                    ax.text(
                        j, i, f"{cm_n[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm_n[i, j] > 0.5 else "black",
                        fontsize=7,
                    )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "advanced_confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[plot] skipped confusion matrix: {e}")

    print(f"\n[done] checkpoint: {ckpt_path}")
    print(f"[done] report    : {OUT_DIR / 'advanced_classification_report.txt'}")


if __name__ == "__main__":
    main()

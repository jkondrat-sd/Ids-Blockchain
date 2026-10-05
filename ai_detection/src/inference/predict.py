"""
predict.py — standalone inference wrapper for the CNN-BiLSTM IDS v2 model.

Loads bilstm_ids_v2.pt + web_splitter_v2.pkl + preprocessing artifacts from
data/processed/ and exposes IDSPredictor for single-flow or batch prediction.
The blockchain_interface/api.py calls IDSPredictor.predict() and formats the
result for on-chain logging.

Usage (CLI):
    python src/inference/predict.py --input sample.csv

Usage (Python):
    from src.inference.predict import IDSPredictor, load_predictor
    predictor = load_predictor()
    result = predictor.predict(feature_vector)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEVERITY = {
    "Benign":           "INFO",
    "DDoS-HOIC":        "CRITICAL",
    "DDoS-LOIC-HTTP":   "CRITICAL",
    "DDoS-LOIC-UDP":    "CRITICAL",
    "DoS-Hulk":         "CRITICAL",
    "SQL-Injection":    "CRITICAL",
    "Infiltration":     "CRITICAL",
    "Botnet":           "HIGH",
    "DoS-GoldenEye":    "HIGH",
    "DoS-Slowloris":    "HIGH",
    "DoS-SlowHTTPTest": "HIGH",
    "BruteForce-Web":   "HIGH",
    "BruteForce-XSS":   "HIGH",
    "SSH-BruteForce":   "MEDIUM",
    "FTP-BruteForce":   "MEDIUM",
}

WEB_CLASSES = ("BruteForce-Web", "BruteForce-XSS", "SQL-Injection")


# --------------------------------------------------------------------------- #
# Legacy model variant (kept so older checkpoints still load)
# --------------------------------------------------------------------------- #

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, num_classes,
                 cnn_ch, n_heads, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_ch, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(cnn_ch),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(cnn_ch),
        )
        self.lstm = nn.LSTM(cnn_ch, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.MultiheadAttention(hidden * 2, n_heads,
                                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)
        pooled, _ = self.attn(out, out, out)
        pooled = pooled.mean(dim=1)
        return self.head(pooled)


# --------------------------------------------------------------------------- #
# Production architecture (must match the CNNLSTM class in
# src/training/train_v2.py and train_v3.py exactly)
#
# This is the architecture of the committed bilstm_ids_v2.pt: GELU-only conv
# front-end and a learned-query attention pool. The CNNLSTM class above is an
# older variant (Conv+BatchNorm+ReLU, self-attention + mean-pool).
# IDSPredictor picks whichever of the two matches the checkpoint's
# state_dict keys, so either layout loads correctly.
# --------------------------------------------------------------------------- #

class _MultiHeadAttentionPool(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, dim) / math.sqrt(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(out.squeeze(1))


class CNNLSTMv3(nn.Module):
    def __init__(self, chunk, cnn_ch, hidden, lstm_layers, num_classes,
                 dropout, n_heads):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(chunk, cnn_ch // 2, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(cnn_ch // 2, cnn_ch, kernel_size=3, padding=1), nn.GELU(),
        )
        self.lstm = nn.LSTM(
            cnn_ch, hidden, num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pool = _MultiHeadAttentionPool(hidden * 2, n_heads=n_heads)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.head(self.pool(h))


# --------------------------------------------------------------------------- #
# IDSPredictor
# --------------------------------------------------------------------------- #

@dataclass
class Prediction:
    label: str
    severity: str
    confidence: float
    timestamp: str
    feature_hash: str


class IDSPredictor:
    """
    Wraps the CNN-BiLSTM v2 model for single-flow or batch inference.
    Natural-prior logit adjustment (LA_K=1.0) is applied at inference.
    """

    def __init__(self, checkpoint_path: Path, splitter_path: Path,
                 processed_dir: Path, device: torch.device = DEVICE):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self.device = device
        self.chunk = ckpt["chunk"]
        self.pad_to = ckpt["pad_to"]
        self.num_features = ckpt["num_features"]
        self.la_tau = ckpt.get("la_tau", 1.0)

        self.orig_class_names: list[str] = ckpt["orig_class_names"]
        # Some committed checkpoints predate the natural-prior refactor and store the
        # merged 13-class names under "class_names" instead of "merged_class_names".
        self.merged_class_names: list[str] = ckpt.get(
            "merged_class_names", ckpt.get("class_names", self.orig_class_names)
        )

        num_classes = ckpt.get("num_classes", len(self.merged_class_names))
        if "log_pi_train" in ckpt:
            log_pi = ckpt["log_pi_train"]
        elif (processed_dir / "log_pi_train.json").exists():
            # Checkpoint predates the natural-prior refactor: the training-set class
            # log-frequencies are shipped alongside the other preprocessing artifacts.
            with open(processed_dir / "log_pi_train.json") as f:
                log_pi = json.load(f)["log_pi_train"]
        else:
            # No saved training-set class log-frequencies at all: disable the
            # logit adjustment (zero vector == no-op) instead of fabricating a prior.
            log_pi = [0.0] * num_classes
        self.log_pi: torch.Tensor = torch.tensor(log_pi, dtype=torch.float32, device=device)

        state_dict = ckpt["model_state_dict"]
        is_v3_arch = any(k.startswith("conv.") or k.startswith("pool.") for k in state_dict)

        if is_v3_arch:
            self.model = CNNLSTMv3(
                chunk=self.chunk,
                cnn_ch=ckpt.get("cnn_ch", 128),
                hidden=ckpt["hidden"],
                lstm_layers=ckpt.get("num_layers", ckpt.get("lstm_layers")),
                num_classes=num_classes,
                dropout=ckpt["dropout"],
                n_heads=ckpt.get("n_heads", 4),
            ).to(device)
        else:
            self.model = CNNLSTM(
                # "input_dim" is the per-timestep channel count fed to the Conv1d front-end,
                # which always equals the chunk size — some checkpoints don't
                # store it explicitly.
                input_dim=ckpt.get("input_dim", self.chunk),
                hidden=ckpt["hidden"],
                num_layers=ckpt.get("num_layers", ckpt.get("lstm_layers")),
                num_classes=num_classes,
                cnn_ch=ckpt.get("cnn_ch", 128),
                n_heads=ckpt.get("n_heads", 4),
                dropout=ckpt["dropout"],
            ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        with open(splitter_path, "rb") as f:
            spl = pickle.load(f)
        # train_v2.py saves {"clf": LogisticRegression, "web_idx": ..., "web_merged_idx": ...}
        self.web_splitter = spl["clf"] if isinstance(spl, dict) and "clf" in spl else spl

        with open(processed_dir / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(processed_dir / "feature_names.json") as f:
            self.feature_names: list[str] = json.load(f)

    def _to_sequence(self, x: np.ndarray) -> torch.Tensor:
        """Pad feature vector to multiple of chunk, reshape to (1, T, chunk)."""
        pad = np.zeros(self.pad_to - x.shape[-1], dtype=np.float32)
        x_padded = np.concatenate([x, pad])
        t_steps = self.pad_to // self.chunk
        return torch.tensor(
            x_padded.reshape(1, t_steps, self.chunk), dtype=torch.float32
        ).to(self.device)

    def predict(self, feature_vector: np.ndarray) -> dict:
        """
        Predict the class of a single network flow.

        Args:
            feature_vector: 1-D numpy array of shape (num_features,),
                            already in the same feature space as training
                            (i.e. the ~47 features output by preprocess.py,
                            BEFORE scaling — scaling is applied here).

        Returns:
            {
              "label":        str   — attack class name (15-class),
              "severity":     str   — INFO / MEDIUM / HIGH / CRITICAL,
              "confidence":   float — softmax probability of predicted class,
              "timestamp":    str   — ISO-8601 UTC,
              "feature_hash": str   — SHA-256 of the raw feature vector (for audit),
            }
        """
        feature_vector = feature_vector.astype(np.float32)
        feature_hash = hashlib.sha256(feature_vector.tobytes()).hexdigest()

        scaled = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        x = self._to_sequence(scaled)

        with torch.no_grad():
            logits = self.model(x)[0]
            logits_adj = logits + self.la_tau * self.log_pi
            probs = F.softmax(logits_adj, dim=-1).cpu().numpy()

        merged_idx = int(np.argmax(probs))
        merged_label = self.merged_class_names[merged_idx]

        if merged_label == "WebAttack":
            # The splitter was fit on scaled features and predicts original class indices.
            web_idx = int(self.web_splitter.predict(scaled.reshape(1, -1))[0])
            label = self.orig_class_names[web_idx]
        else:
            label = merged_label

        confidence = float(probs[merged_idx])

        return {
            "label":        label,
            "severity":     SEVERITY.get(label, "UNKNOWN"),
            "confidence":   round(confidence, 4),
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "feature_hash": feature_hash,
        }

    def predict_batch(self, X: np.ndarray) -> list[dict]:
        """Predict a batch of flows. X shape: (N, num_features)."""
        return [self.predict(row) for row in X]


# --------------------------------------------------------------------------- #
# Convenience loader
# --------------------------------------------------------------------------- #

def load_predictor(
    checkpoint: Path = MODEL_DIR / "bilstm_ids_v2.pt",
    splitter: Path = MODEL_DIR / "web_splitter_v2.pkl",
    processed_dir: Path = PROC_DIR,
) -> IDSPredictor:
    return IDSPredictor(checkpoint, splitter, processed_dir)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli():
    import pandas as pd
    parser = argparse.ArgumentParser(description="Run CNN-BiLSTM IDS inference")
    parser.add_argument("--input", required=True, help="CSV file of network flows")
    parser.add_argument("--output", default=None, help="Output JSON file (optional)")
    args = parser.parse_args()

    predictor = load_predictor()
    df = pd.read_csv(args.input)
    df.columns = df.columns.str.strip()

    # Keep only the model's feature columns
    X = df[predictor.feature_names].values.astype(np.float32)
    results = predictor.predict_batch(X)

    for i, r in enumerate(results):
        print(f"[{i:04d}] {r['label']:25s} {r['severity']:8s}  conf={r['confidence']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved {len(results)} predictions → {args.output}")


if __name__ == "__main__":
    _cli()

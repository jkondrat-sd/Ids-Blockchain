"""
eval_v2_natural.py — Diagnostic A.

Re-score the existing bilstm_ids_v2.pt checkpoint under two inference rules:

    (a) raw argmax(logits)                          -- what train_v2.py reported
    (b) argmax(logits + tau * log pi_train)         -- natural-prior correction

If (b) lifts the 13-class macro-F1 above the baseline 0.7366, the CNN-LSTM
backbone learned useful features and the LA miscalibration is purely at the
output layer. In that case we can keep the checkpoint and ship rule (b).
If (b) is still below 0.7366, LA has damaged the representation itself and
we need v3.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.training.train_v2 import CHUNK, CNNLSTM, FlowDataset, LA_TAU, WEB_CLASSES, build_label_map, collect_logits

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096


def main():
    ckpt = torch.load(MODEL_DIR / "bilstm_ids_v2.pt", map_location=DEVICE, weights_only=False)
    class_names = ckpt["orig_class_names"]
    merged_names, label_map, web_idx, web_merged_idx = build_label_map(class_names)
    num_classes = ckpt["num_classes"]
    pad_to = ckpt["pad_to"]
    print(f"[ckpt] arch={ckpt['arch']} classes={num_classes} pad_to={pad_to} seed={ckpt.get('seed')}")

    model = CNNLSTM(
        chunk=ckpt["chunk"], cnn_ch=ckpt["cnn_ch"], hidden=ckpt["hidden"],
        lstm_layers=ckpt["lstm_layers"], num_classes=num_classes,
        dropout=ckpt["dropout"], n_heads=ckpt["n_heads"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_test = np.load(PROC_DIR / "X_test.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_test_m = label_map[y_test]
    y_train_m = label_map[y_train]

    pi = np.bincount(y_train_m, minlength=num_classes).astype(np.float64)
    pi /= pi.sum()
    log_pi = np.log(pi + 1e-12).astype(np.float32)

    test_ds = FlowDataset(X_test, y_test_m, pad_to, CHUNK)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        pin_memory=DEVICE.type == "cuda")
    print("[eval] collecting logits...")
    logits, _ = collect_logits(model, loader, DEVICE)

    # Rule (a): raw argmax  (this is what train_v2.py already reported)
    pred_a = logits.argmax(1)
    f1_a_13 = f1_score(y_test_m, pred_a, average="macro", zero_division=0)

    # Rule (b): natural-prior correction
    # We trained with CE( logits + tau*log pi, y ), so logits learned to fill in the gap.
    # Undoing the shift at eval moves predictions back toward the training prior.
    logits_b = logits + LA_TAU * log_pi[None, :]
    pred_b = logits_b.argmax(1)
    f1_b_13 = f1_score(y_test_m, pred_b, average="macro", zero_division=0)

    print(f"\n[rule a] raw argmax                13-class macro-F1: {f1_a_13:.4f}")
    print(f"[rule b] + tau*log(pi_train)        13-class macro-F1: {f1_b_13:.4f}")

    # Apply splitter for 15-class reconstruction under both rules
    splitter_path = MODEL_DIR / "web_splitter_v2.pkl"
    splitter = None
    if splitter_path.exists():
        with open(splitter_path, "rb") as f:
            splitter = pickle.load(f)["clf"]

    reverse_map = np.zeros(num_classes, dtype=np.int64)
    for orig_i, merged_i in enumerate(label_map):
        if class_names[orig_i] not in WEB_CLASSES:
            reverse_map[merged_i] = orig_i

    def reconstruct_15(pred_m):
        pred_15 = reverse_map[pred_m].copy()
        mask = pred_m == web_merged_idx
        if mask.any() and splitter is not None:
            pred_15[mask] = splitter.predict(X_test[mask])
        return pred_15

    pred_a_15 = reconstruct_15(pred_a)
    pred_b_15 = reconstruct_15(pred_b)
    f1_a_15 = f1_score(y_test, pred_a_15, average="macro", zero_division=0)
    f1_b_15 = f1_score(y_test, pred_b_15, average="macro", zero_division=0)

    print(f"[rule a] raw argmax                15-class macro-F1: {f1_a_15:.4f}")
    print(f"[rule b] + tau*log(pi_train)        15-class macro-F1: {f1_b_15:.4f}")
    print(f"\n  baseline notebook: 0.7366   advanced v1: 0.6750   v2 (rule a): 0.6407")

    # Per-class report for rule (b) — 15-class, the comparable metric
    print("\n======= 15-class report under rule (b) =======")
    report_b_15 = classification_report(
        y_test, pred_b_15, target_names=class_names, digits=4, zero_division=0,
    )
    print(report_b_15)

    (OUT_DIR / "v2_natural_classification_report_15.txt").write_text(
        f"test macro-F1 15c (rule b, +log pi): {f1_b_15:.4f}\n"
        f"test macro-F1 13c (rule b, +log pi): {f1_b_13:.4f}\n\n{report_b_15}"
    )
    (OUT_DIR / "v2_natural_summary.json").write_text(json.dumps({
        "rule_a_raw_argmax": {"f1_13c": float(f1_a_13), "f1_15c": float(f1_a_15)},
        "rule_b_natural_prior": {"f1_13c": float(f1_b_13), "f1_15c": float(f1_b_15)},
        "baseline_notebook": 0.7366, "advanced_v1": 0.6750,
    }, indent=2))

    print("\n[done] reports at lstm/outputs/v2_natural_*")


if __name__ == "__main__":
    main()

"""
sweep_v2_tau.py — find the best inference-time prior-correction strength
for the bilstm_ids_v2.pt checkpoint.

We evaluated two extremes in eval_v2_natural.py:
  k=0.0  (raw argmax)               -> 15c F1 0.6407
  k=1.0  (full natural-prior undo)  -> 15c F1 0.7711

Rule (b) at k=1.0 over-corrected some classes (FTP-BruteForce -> 0,
Infiltration -> 0.15). Intermediate k should rescue them without giving
up the web-attack wins. This script sweeps k on the val set and reports
the test F1 at the val-best k (no peeking at test for selection).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from src.training.train_v2 import CHUNK, CNNLSTM, FlowDataset, WEB_CLASSES, build_label_map, collect_logits

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

    model = CNNLSTM(
        chunk=ckpt["chunk"], cnn_ch=ckpt["cnn_ch"], hidden=ckpt["hidden"],
        lstm_layers=ckpt["lstm_layers"], num_classes=num_classes,
        dropout=ckpt["dropout"], n_heads=ckpt["n_heads"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_val = np.load(PROC_DIR / "X_val.npy")
    X_test = np.load(PROC_DIR / "X_test.npy")
    y_val = np.load(PROC_DIR / "y_val.npy")
    y_test = np.load(PROC_DIR / "y_test.npy")
    y_train = np.load(PROC_DIR / "y_train.npy")
    y_val_m = label_map[y_val]
    y_test_m = label_map[y_test]
    y_train_m = label_map[y_train]

    pi = np.bincount(y_train_m, minlength=num_classes).astype(np.float64)
    pi /= pi.sum()
    log_pi = np.log(pi + 1e-12).astype(np.float32)

    pin = DEVICE.type == "cuda"
    val_loader = DataLoader(FlowDataset(X_val, y_val_m, pad_to, CHUNK),
                            batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(FlowDataset(X_test, y_test_m, pad_to, CHUNK),
                             batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)
    print("[eval] collecting val logits...")
    val_logits, _ = collect_logits(model, val_loader, DEVICE)
    print("[eval] collecting test logits...")
    test_logits, _ = collect_logits(model, test_loader, DEVICE)

    # splitter for 15-class reconstruction
    splitter = None
    sp_path = MODEL_DIR / "web_splitter_v2.pkl"
    if sp_path.exists():
        with open(sp_path, "rb") as f:
            splitter = pickle.load(f)["clf"]

    reverse_map = np.zeros(num_classes, dtype=np.int64)
    for orig_i, merged_i in enumerate(label_map):
        if class_names[orig_i] not in WEB_CLASSES:
            reverse_map[merged_i] = orig_i

    def to_15(pred_m, X):
        pred_15 = reverse_map[pred_m].copy()
        mask = pred_m == web_merged_idx
        if mask.any() and splitter is not None:
            pred_15[mask] = splitter.predict(X[mask])
        return pred_15

    # sweep k on the val set (13-class metric — cheap, no splitter needed)
    ks = np.linspace(0.0, 1.5, 31)
    val_f1_13 = []
    for k in ks:
        pred = (val_logits + k * log_pi[None, :]).argmax(1)
        f1 = f1_score(y_val_m, pred, average="macro", zero_division=0)
        val_f1_13.append(f1)
    val_f1_13 = np.array(val_f1_13)

    print("\n  k      val-F1(13c)")
    for k, f in zip(ks, val_f1_13):
        marker = "  <-- best" if f == val_f1_13.max() else ""
        print(f"  {k:4.2f}   {f:.4f}{marker}")

    k_star = float(ks[val_f1_13.argmax()])
    print(f"\n[sweep] best k on val: {k_star:.2f}  (val 13c F1 {val_f1_13.max():.4f})")

    # apply k* on test — both 13c and 15c (splitter reconstruction)
    pred_test_m = (test_logits + k_star * log_pi[None, :]).argmax(1)
    f1_test_13 = f1_score(y_test_m, pred_test_m, average="macro", zero_division=0)
    pred_test_15 = to_15(pred_test_m, X_test)
    f1_test_15 = f1_score(y_test, pred_test_15, average="macro", zero_division=0)

    print(f"\n[test @ k={k_star:.2f}] 13-class macro-F1: {f1_test_13:.4f}")
    print(f"[test @ k={k_star:.2f}] 15-class macro-F1: {f1_test_15:.4f}")
    print("  reference: baseline 0.7366, v1 0.6750, v2 raw 0.6407, v2 k=1.0 0.7711")

    report = classification_report(y_test, pred_test_15, target_names=class_names,
                                   digits=4, zero_division=0)
    print(f"\n======= 15-class report @ k={k_star:.2f} =======")
    print(report)

    (OUT_DIR / f"v2_sweep_report_15.txt").write_text(
        f"k*={k_star:.2f}\nval F1(13c)={val_f1_13.max():.4f}\n"
        f"test F1(13c)={f1_test_13:.4f}\ntest F1(15c)={f1_test_15:.4f}\n\n{report}"
    )
    (OUT_DIR / "v2_sweep_summary.json").write_text(json.dumps({
        "ks": ks.tolist(),
        "val_f1_13c": val_f1_13.tolist(),
        "k_star": k_star,
        "test_f1_13c": float(f1_test_13),
        "test_f1_15c": float(f1_test_15),
    }, indent=2))

    print(f"\n[done] reports at lstm/outputs/v2_sweep_*")


if __name__ == "__main__":
    main()

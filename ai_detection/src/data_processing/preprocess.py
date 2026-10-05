"""
CSE-CIC-IDS2018 — Preprocessing Pipeline for LSTM IDS

Extracted from 01_preprocessing.ipynb. Produces per-flow feature tensors
for the multi-class BiLSTM/CNN-BiLSTM intrusion detection model.

Design rule: one flow = one sample. No sliding windows. CICFlowMeter already
aggregates packets into ~80 per-flow statistical features. We clean, cap, scale,
and split those features. The training script reshapes each flow's feature
vector into a pseudo-sequence (T, C) for the LSTM.

Usage:
    python src/data_processing/preprocess.py

Outputs:
    data/processed/X_train.npy, X_val.npy, X_test.npy   (N, F) float32
    data/processed/y_train.npy, y_val.npy, y_test.npy    int64 label indices
    data/processed/class_weights.npy                      float32 (num_classes,)
    data/processed/scaler.pkl                             fitted StandardScaler
    data/processed/label_encoder.pkl                      fitted LabelEncoder
    data/processed/class_names.json
    data/processed/feature_names.json
    data/processed/class_distribution.png
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# =========================================================================== #
# Paths
# =========================================================================== #

ROOT        = Path(__file__).resolve().parents[2]       # repo root
DATA_DIR    = ROOT / "data" / "raw"                     # place CIC-IDS-2018 CSVs here
OUT_DIR     = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================================== #
# Pipeline hyper-parameters
# =========================================================================== #

SEED               = 42
CHUNK_SIZE         = 200_000    # rows per read_csv chunk
MAX_ROWS_PER_CLASS = 200_000    # cap non-benign classes
BENIGN_CAP         = 400_000    # cap benign (kept bigger: realistic noise)
MIN_CLASS_ROWS     = 50         # drop classes below this threshold
CORR_THRESHOLD     = 0.98       # |r| above -> drop redundant feature

# =========================================================================== #
# Column & label maps
# =========================================================================== #

# Metadata / identifier columns that cause leakage or are non-numeric
DROP_META = {
    "Timestamp", "Flow ID", "Src IP", "Dst IP",
    "Src Port", "Dst Port", "Protocol",
}

# 10 zero-variance columns confirmed in literature (Gocs & Johanyak 2024)
ZERO_VAR_COLS = {
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "CWE Flag Count",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
}

# Canonical label names (merge typos & alternate spellings)
LABEL_MAP = {
    "Benign": "Benign", "benign": "Benign", "BENIGN": "Benign",
    "FTP-BruteForce": "FTP-BruteForce",
    "SSH-Bruteforce": "SSH-BruteForce", "SSH-BruteForce": "SSH-BruteForce",
    "DoS attacks-GoldenEye": "DoS-GoldenEye",
    "DoS attacks-Slowloris": "DoS-Slowloris",
    "DoS attacks-Hulk": "DoS-Hulk",
    "DoS attacks-SlowHTTPTest": "DoS-SlowHTTPTest",
    "DDoS attacks-LOIC-HTTP": "DDoS-LOIC-HTTP",
    "DDOS attack-HOIC": "DDoS-HOIC", "DDoS attack-HOIC": "DDoS-HOIC",
    "DDOS attack-LOIC-UDP": "DDoS-LOIC-UDP", "DDoS attack-LOIC-UDP": "DDoS-LOIC-UDP",
    "Brute Force -Web": "BruteForce-Web",
    "Brute Force -XSS": "BruteForce-XSS",
    "SQL Injection": "SQL-Injection",
    "Infilteration": "Infiltration", "Infiltration": "Infiltration",
    "Bot": "Botnet",
}

# =========================================================================== #
# Cleaning functions
# =========================================================================== #


def detect_feature_columns(files):
    """Return the intersection of numeric columns across all CSVs, minus metadata."""
    common = None
    for f in files:
        cols = [c.strip() for c in pd.read_csv(f, nrows=0).columns]
        common = set(cols) if common is None else (common & set(cols))
    if "Label" not in common:
        raise ValueError("Label column missing in at least one CSV")
    keep = sorted(
        c for c in common
        if c not in DROP_META and c not in ZERO_VAR_COLS and c != "Label"
    )
    keep.append("Label")
    return keep


def clean_chunk(df):
    """Clean one chunk: normalize labels, coerce numerics, drop bad rows."""
    df.columns = [c.strip() for c in df.columns]
    df["Label"] = df["Label"].astype(str).str.strip().map(LABEL_MAP)
    df = df.dropna(subset=["Label"])

    feat_cols = [c for c in df.columns if c != "Label"]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if "Flow Duration" in df.columns:
        df = df[df["Flow Duration"] >= 0]

    df = df.drop_duplicates()
    df[feat_cols] = df[feat_cols].astype(np.float32)
    return df


def load_balanced(files, use_cols, chunk_size, attack_cap, benign_cap):
    """Stream CSVs, clean each chunk, and keep a capped buffer per class."""
    buffers = {}
    for i, f in enumerate(files, 1):
        t0, kept = time.time(), 0
        for chunk in pd.read_csv(
            f, usecols=use_cols, chunksize=chunk_size, low_memory=False
        ):
            cleaned = clean_chunk(chunk)
            if cleaned.empty:
                continue
            kept += len(cleaned)
            for lbl, grp in cleaned.groupby("Label"):
                cap = benign_cap if lbl == "Benign" else attack_cap
                if lbl not in buffers:
                    buffers[lbl] = grp
                else:
                    buffers[lbl] = pd.concat(
                        [buffers[lbl], grp], ignore_index=True
                    )
                if len(buffers[lbl]) > cap:
                    buffers[lbl] = buffers[lbl].sample(
                        n=cap, random_state=SEED
                    ).reset_index(drop=True)
            del cleaned
            gc.collect()
        print(
            f"  [{i:02d}/{len(files)}] {f.name:55s} "
            f"kept={kept:>9,}  ({time.time()-t0:5.1f}s)"
        )

    # Drop rare classes
    rare = [k for k, v in buffers.items() if len(v) < MIN_CLASS_ROWS]
    for k in rare:
        print(f'  - dropping rare class "{k}" ({len(buffers[k])} rows)')
        del buffers[k]

    return buffers


# =========================================================================== #
# Main pipeline
# =========================================================================== #


def main():
    np.random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  CSE-CIC-IDS2018 PREPROCESSING PIPELINE")
    print("=" * 70)

    # ---- Discover CSVs ----
    csv_files = sorted(DATA_DIR.glob("*TrafficForML_CICFlowMeter.csv"))
    print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")
    for p in csv_files:
        print(f"  - {p.name}")
    assert len(csv_files) >= 10, f"Expected 10 CSVs, found {len(csv_files)}"

    # ---- Detect columns ----
    use_cols = detect_feature_columns(csv_files)
    print(f"\nFeature columns kept: {len(use_cols) - 1}   (+ Label)")

    # ---- Load & clean ----
    print("\nLoading & cleaning all CSVs in chunks...")
    buffers = load_balanced(
        csv_files, use_cols, CHUNK_SIZE, MAX_ROWS_PER_CLASS, BENIGN_CAP
    )

    print("\n=== Class distribution after capping ===")
    for lbl in sorted(buffers, key=lambda k: -len(buffers[k])):
        print(f"  {lbl:22s} {len(buffers[lbl]):>9,}")

    df = (
        pd.concat(buffers.values(), ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    del buffers
    gc.collect()
    print(f"\nCombined dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"RAM: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # ---- Class distribution plot ----
    counts = df["Label"].value_counts()
    fig, ax = plt.subplots(figsize=(11, 5))
    counts.plot.bar(color="steelblue", ax=ax)
    ax.set_title("Class distribution after capping (log scale)")
    ax.set_yscale("log")
    ax.set_ylabel("# flows (log)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=120)
    plt.close()

    # ---- Feature cleanup ----
    X_df = df.drop(columns=["Label"])
    y_text = df["Label"].astype(str)

    # Drop remaining zero-variance
    std = X_df.std()
    zero_var = std[std == 0].index.tolist()
    if zero_var:
        print(f"Zero-variance dropped: {zero_var}")
        X_df = X_df.drop(columns=zero_var)

    # Drop highly correlated features
    corr = X_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
    if to_drop:
        print(f"Highly correlated dropped ({len(to_drop)}): {to_drop}")
        X_df = X_df.drop(columns=to_drop)

    # Winsorize at 1st/99th percentile
    lo = X_df.quantile(0.01)
    hi = X_df.quantile(0.99)
    X_df = X_df.clip(lower=lo, upper=hi, axis=1)

    feature_names = X_df.columns.tolist()
    print(f"\nFinal feature count: {len(feature_names)}")
    print(f"Samples            : {len(X_df):,}")

    # ---- Label encoding & split ----
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = label_encoder.classes_.tolist()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    X = X_df.values.astype(np.float32)
    del df, X_df, y_text
    gc.collect()

    # Stratified 70 / 15 / 15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    del X, X_tmp, y_tmp
    gc.collect()

    print(f"Train: {X_train.shape}  {dict(Counter(y_train))}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

    # ---- Z-score scaling (fit on train only) ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    for arr in (X_train, X_val, X_test):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Scaled train mean={X_train.mean():+.4f}  std={X_train.std():.4f}")
    print(f"Scaled val   mean={X_val.mean():+.4f}  std={X_val.std():.4f}")

    # ---- Class weights ----
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y_train
    ).astype(np.float32)
    print("\nClass weights:")
    for name, w in zip(class_names, class_weights):
        print(f"  {name:22s} {w:7.3f}")

    # ---- Save artifacts ----
    np.save(OUT_DIR / "X_train.npy", X_train)
    np.save(OUT_DIR / "X_val.npy", X_val)
    np.save(OUT_DIR / "X_test.npy", X_test)
    np.save(OUT_DIR / "y_train.npy", y_train)
    np.save(OUT_DIR / "y_val.npy", y_val)
    np.save(OUT_DIR / "y_test.npy", y_test)
    np.save(OUT_DIR / "class_weights.npy", class_weights)

    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(OUT_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(OUT_DIR / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved artifacts to: {OUT_DIR}")
    for p in sorted(OUT_DIR.iterdir()):
        if p.is_file():
            print(f"  - {p.name:25s} {p.stat().st_size / 1e6:8.2f} MB")

    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

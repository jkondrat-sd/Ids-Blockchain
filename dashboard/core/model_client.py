"""
model_client.py — Wrapper around IDSPredictor (CNN-BiLSTM v2).
Tries to load the real model; falls back to the simulator when unavailable.
"""
from __future__ import annotations
import sys
import json
import hashlib
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = ROOT / "ai_detection"
MODEL_DIR = AI_ROOT / "models"
PROC_DIR = AI_ROOT / "data" / "processed"

SEVERITY = {
    "Benign": "INFO", "DDoS-HOIC": "CRITICAL", "DDoS-LOIC-HTTP": "CRITICAL",
    "DDoS-LOIC-UDP": "CRITICAL", "DoS-Hulk": "CRITICAL", "SQL-Injection": "CRITICAL",
    "Infiltration": "CRITICAL", "Botnet": "HIGH", "DoS-GoldenEye": "HIGH",
    "DoS-Slowloris": "HIGH", "DoS-SlowHTTPTest": "HIGH", "BruteForce-Web": "HIGH",
    "BruteForce-XSS": "HIGH", "SSH-BruteForce": "MEDIUM", "FTP-BruteForce": "MEDIUM",
}


class ModelClient:
    def __init__(self):
        self.real_model = False
        self.predictor = None
        self.feature_names: list[str] = []
        self._load()

    def _load(self):
        ckpt = MODEL_DIR / "bilstm_ids_v2.pt"
        spl = MODEL_DIR / "web_splitter_v2.pkl"
        if not (ckpt.exists() and spl.exists() and PROC_DIR.exists()):
            return
        try:
            sys.path.insert(0, str(AI_ROOT))
            from src.inference.predict import IDSPredictor
            self.predictor = IDSPredictor(ckpt, spl, PROC_DIR)
            feat_path = PROC_DIR / "feature_names.json"
            if feat_path.exists():
                self.feature_names = json.loads(feat_path.read_text())
            self.real_model = True
        except Exception:
            self.real_model = False

    def predict_csv(self, df: pd.DataFrame) -> list[dict]:
        if self.real_model and self.predictor and self.feature_names:
            df.columns = df.columns.str.strip()
            available = [c for c in self.feature_names if c in df.columns]
            if len(available) == len(self.feature_names):
                X = df[self.feature_names].values.astype(np.float32)
                return self.predictor.predict_batch(X)
        return self._simulate_csv(df)

    def _simulate_csv(self, df: pd.DataFrame) -> list[dict]:
        from core.simulator import generate_alert
        results = []
        for _ in range(len(df)):
            a = generate_alert()
            results.append({
                "label": a["label"], "severity": a["severity"],
                "confidence": a["confidence"], "timestamp": a["timestamp"],
                "feature_hash": a["feature_hash"],
            })
        return results

    @property
    def status(self) -> str:
        return "Modèle réel chargé" if self.real_model else "Mode simulation"


_client: ModelClient | None = None

def get_model_client() -> ModelClient:
    global _client
    if _client is None:
        _client = ModelClient()
    return _client

"""
simulator.py — Realistic IDS alert generator for demo/offline mode.
Mirrors the class distribution and confidence patterns of CIC-IDS2018.
"""
from __future__ import annotations
import random
import uuid
from datetime import datetime, timezone, timedelta

CLASSES = [
    "Benign", "DDoS-HOIC", "DDoS-LOIC-HTTP", "DDoS-LOIC-UDP",
    "DoS-Hulk", "DoS-GoldenEye", "DoS-Slowloris", "DoS-SlowHTTPTest",
    "Botnet", "Infiltration", "BruteForce-Web", "BruteForce-XSS",
    "SQL-Injection", "SSH-BruteForce", "FTP-BruteForce",
]

CLASS_WEIGHTS = {
    "Benign": 0.53, "DDoS-HOIC": 0.08, "DDoS-LOIC-HTTP": 0.05,
    "DDoS-LOIC-UDP": 0.05, "DoS-Hulk": 0.05, "DoS-GoldenEye": 0.03,
    "DoS-Slowloris": 0.02, "DoS-SlowHTTPTest": 0.02, "Botnet": 0.03,
    "Infiltration": 0.02, "BruteForce-Web": 0.02, "BruteForce-XSS": 0.02,
    "SQL-Injection": 0.01, "SSH-BruteForce": 0.04, "FTP-BruteForce": 0.02,
}

SEVERITY = {
    "Benign": "INFO", "DDoS-HOIC": "CRITICAL", "DDoS-LOIC-HTTP": "CRITICAL",
    "DDoS-LOIC-UDP": "CRITICAL", "DoS-Hulk": "CRITICAL", "SQL-Injection": "CRITICAL",
    "Infiltration": "CRITICAL", "Botnet": "HIGH", "DoS-GoldenEye": "HIGH",
    "DoS-Slowloris": "HIGH", "DoS-SlowHTTPTest": "HIGH", "BruteForce-Web": "HIGH",
    "BruteForce-XSS": "HIGH", "SSH-BruteForce": "MEDIUM", "FTP-BruteForce": "MEDIUM",
}

CONF_RANGE = {
    "CRITICAL": (0.87, 0.99), "HIGH": (0.75, 0.95),
    "MEDIUM": (0.65, 0.90), "INFO": (0.91, 0.99),
}

ATTACK_DESC = {
    "DDoS-HOIC": "High Orbit Ion Cannon — volumetric HTTP flood",
    "DDoS-LOIC-HTTP": "Low Orbit Ion Cannon — HTTP layer flood",
    "DDoS-LOIC-UDP": "Low Orbit Ion Cannon — UDP flood",
    "DoS-Hulk": "HTTP Unbearable Load King — randomized flood",
    "DoS-GoldenEye": "GoldenEye — HTTP keep-alive DoS",
    "DoS-Slowloris": "Slowloris — slow HTTP header exhaustion",
    "DoS-SlowHTTPTest": "Slow HTTP body POST attack",
    "Botnet": "Command & control traffic (ARES botnet)",
    "Infiltration": "Lateral movement / data exfiltration",
    "BruteForce-Web": "Web login brute force",
    "BruteForce-XSS": "Cross-site scripting injection",
    "SQL-Injection": "SQL injection attempt",
    "SSH-BruteForce": "SSH credential brute force",
    "FTP-BruteForce": "FTP credential brute force",
    "Benign": "Normal network traffic",
}

_SUBNETS = ["192.168.1", "10.0.0", "172.16.0", "203.0.113", "198.51.100"]

def _rip():
    s = random.choice(_SUBNETS)
    return f"{s}.{random.randint(1, 254)}"

def _port(label: str) -> int:
    if "SSH" in label: return 22
    if "FTP" in label: return 21
    if "SQL" in label: return 3306
    if "DDoS" in label or "DoS" in label: return random.choice([80, 443, 8080])
    if "Web" in label or "XSS" in label: return random.choice([80, 443])
    return random.randint(1024, 65535)

def generate_alert(ts: str | None = None) -> dict:
    labels = list(CLASS_WEIGHTS.keys())
    weights = list(CLASS_WEIGHTS.values())
    label = random.choices(labels, weights=weights, k=1)[0]
    sev = SEVERITY[label]
    lo, hi = CONF_RANGE[sev]
    return {
        "label":       label,
        "severity":    sev,
        "confidence":  round(random.uniform(lo, hi), 4),
        "timestamp":   ts or datetime.now(timezone.utc).isoformat(),
        "src_ip":      _rip(),
        "dst_ip":      _rip(),
        "dst_port":    _port(label),
        "protocol":    random.choice(["TCP", "TCP", "UDP", "ICMP"]),
        "duration_us": random.randint(0, 500_000),
        "feature_hash": f"{random.getrandbits(128):032x}",
        "description": ATTACK_DESC.get(label, ""),
    }

# Noms exacts attendus par le modèle CNN-BiLSTM (feature_names.json)
FLOW_FEATURE_NAMES = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Mean", "Fwd IAT Std",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "Down/Up Ratio", "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std",
]

def generate_flow_features(ts: str | None = None) -> dict:
    """Raw unlabeled network flow — sent to Kafka for model inference."""
    label = random.choices(list(CLASS_WEIGHTS.keys()), weights=list(CLASS_WEIGHTS.values()), k=1)[0]
    duration = random.randint(0, 500_000)
    features = {name: round(random.uniform(0, 1e5), 4) for name in FLOW_FEATURE_NAMES}
    features["Flow Duration"] = float(duration)
    features["SYN Flag Count"] = float(random.randint(0, 5))
    features["ACK Flag Count"] = float(random.randint(0, 20))
    features["PSH Flag Count"] = float(random.randint(0, 10))
    features["FIN Flag Count"] = float(random.randint(0, 3))
    features["RST Flag Count"] = float(random.randint(0, 2))
    return {
        "flow_id": uuid.uuid4().hex[:16],
        "timestamp": ts or datetime.now(timezone.utc).isoformat(),
        "src_ip": _rip(),
        "dst_ip": _rip(),
        "dst_port": _port(label),
        "protocol": random.choice(["TCP", "TCP", "UDP", "ICMP"]),
        "duration_us": duration,
        "features": features,
    }

def generate_history(hours: int = 24, avg_per_hour: int = 150) -> list[dict]:
    alerts = []
    now = datetime.now(timezone.utc)
    for h in range(hours, 0, -1):
        t = now - timedelta(hours=h)
        n = random.randint(int(avg_per_hour * 0.6), int(avg_per_hour * 1.4))
        for _ in range(n):
            ts = (t + timedelta(seconds=random.randint(0, 3599))).isoformat()
            alerts.append(generate_alert(ts=ts))
    return alerts

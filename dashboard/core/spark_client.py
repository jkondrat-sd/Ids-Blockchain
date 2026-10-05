"""
spark_client.py — Wrapper Spark : stats statiques + Structured Streaming via subprocess.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
SPARK_DIR  = ROOT / "big_data_spark"
JAR        = SPARK_DIR / "target" / "scala-2.12" / "ids-bigdata-pipeline_2.12-1.0.jar"
STATS_FILE = Path(__file__).resolve().parents[1] / "data" / "spark_stats.json"
PIPELINE   = Path(__file__).resolve().parents[1] / "spark" / "streaming_pipeline.py"

DATASET_STATS = {
    "total_flows":    16_232_943,
    "total_size_gb":  16.2,
    "num_features":   80,
    "num_classes":    15,
    "train_samples":  12_986_354,
    "test_samples":    3_246_589,
    "class_dist": {
        "Benign":           6_747_813,
        "DDoS-HOIC":        1_082_885,
        "DDoS-LOIC-HTTP":     576_191,
        "DDoS-LOIC-UDP":      540_104,
        "DoS-Hulk":           461_912,
        "SSH-BruteForce":     187_589,
        "DoS-GoldenEye":      141_933,
        "DoS-Slowloris":      139_890,
        "Botnet":             286_191,
        "DoS-SlowHTTPTest":   139_890,
        "FTP-BruteForce":     198_514,
        "Infiltration":        18_893,
        "BruteForce-Web":      39_791,
        "BruteForce-XSS":      35_794,
        "SQL-Injection":       25_553,
    },
}

# ── Batch pipeline (Scala JAR / sbt) ─────────────────────────────────────────

def run_pipeline() -> dict:
    if not SPARK_DIR.exists():
        return {"success": False, "log": "Dossier big_data_spark introuvable.", "elapsed": 0}
    start = time.time()
    try:
        if JAR.exists():
            result = subprocess.run(
                ["java", "-jar", str(JAR)],
                capture_output=True, text=True, timeout=120,
            )
            elapsed = round(time.time() - start, 2)
            return {"success": result.returncode == 0,
                    "log": result.stdout + result.stderr, "elapsed": elapsed}
        sbt = subprocess.run(
            ["sbt", "run"], capture_output=True, text=True,
            timeout=180, cwd=str(SPARK_DIR),
        )
        elapsed = round(time.time() - start, 2)
        return {"success": sbt.returncode == 0,
                "log": sbt.stdout + sbt.stderr, "elapsed": elapsed}
    except FileNotFoundError:
        return {"success": False, "log": "sbt / java introuvable. Installez le JDK et sbt.", "elapsed": 0}
    except subprocess.TimeoutExpired:
        return {"success": False, "log": "Timeout — pipeline > 3 min.", "elapsed": 180}
    except Exception as e:
        return {"success": False, "log": str(e), "elapsed": 0}


def get_dataset_stats() -> dict:
    return DATASET_STATS

# ── Structured Streaming (PySpark subprocess) ─────────────────────────────────

_spark_process: subprocess.Popen | None = None


def _java_available() -> bool:
    try:
        subprocess.run(["java", "-version"], capture_output=True, timeout=3)
        return True
    except Exception:
        return False


def start_streaming() -> bool:
    """Lance streaming_pipeline.py comme subprocess PySpark. Retourne True si démarré."""
    global _spark_process
    if _spark_process is not None and _spark_process.poll() is None:
        return True  # Déjà en cours
    if not _java_available():
        logger.warning("Java absent — Spark Streaming désactivé")
        return False
    try:
        _spark_process = subprocess.Popen(
            [sys.executable, str(PIPELINE)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(PIPELINE.parent),
        )
        logger.info(f"Spark Streaming démarré (PID {_spark_process.pid})")
        return True
    except Exception as exc:
        logger.warning(f"Erreur démarrage Spark Streaming: {exc}")
        return False


def get_streaming_stats() -> dict | None:
    """Lit spark_stats.json. Retourne None si absent ou données > 60s."""
    try:
        if not STATS_FILE.exists():
            return None
        data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        updated = data.get("updated_at")
        if updated:
            age = (datetime.now() - datetime.fromisoformat(updated)).total_seconds()
            if age > 60:
                return None
        return data
    except Exception:
        return None


def is_streaming_active() -> bool:
    return _spark_process is not None and _spark_process.poll() is None

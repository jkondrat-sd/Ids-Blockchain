"""
kafka_consumer.py — Consomme ids-detections, alimente le buffer live et persiste sur disque.
Toute alerte simulée passe par ce consumer (aucun appel direct au simulateur dans les pages).

Si aucun broker Kafka n'est joignable au démarrage, ce module bascule automatiquement
sur un simulateur local (_simulate_loop) qui alimente exactement le même buffer/fichier,
pour que Live Monitor / Threat Intel / Rapport fonctionnent même sans Docker/Kafka.
"""
from __future__ import annotations
import json
import random
import socket
import threading
import logging
import os
import time
from collections import deque
from pathlib import Path

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "ids-detections"
GROUP_ID = "ids-dashboard"

_DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "kafka_alerts.json"
MAX_HISTORY = 5000
FLUSH_EVERY = 10        # messages
FLUSH_INTERVAL = 5.0    # secondes

logger = logging.getLogger(__name__)

_alert_buffer: deque = deque(maxlen=600)
_lock = threading.Lock()
_active = False
_running = False

# ── Persistance JSON ──────────────────────────────────────────────────────────

_write_batch: list[dict] = []
_last_flush: float = 0.0
_file_lock = threading.Lock()


def _flush_to_file() -> None:
    global _write_batch, _last_flush
    with _file_lock:
        if not _write_batch:
            return
        batch = _write_batch[:]
        _write_batch = []
    try:
        _DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if _DATA_FILE.exists():
            existing = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        existing.extend(batch)
        if len(existing) > MAX_HISTORY:
            existing = existing[-MAX_HISTORY:]
        _DATA_FILE.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        _last_flush = time.monotonic()
    except Exception as exc:
        logger.warning(f"Erreur écriture kafka_alerts.json : {exc}")


def get_kafka_history() -> list[dict]:
    """Retourne tout l'historique persisté sur disque (jusqu'à 5000 alertes)."""
    try:
        if _DATA_FILE.exists():
            return json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Erreur lecture kafka_alerts.json : {exc}")
    return []

# ── Consumer thread ───────────────────────────────────────────────────────────

def _consume_loop() -> None:
    global _active, _write_batch, _last_flush
    try:
        from kafka import KafkaConsumer
    except ImportError:
        logger.warning("kafka-python non installé — dashboard consumer désactivé")
        return

    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BOOTSTRAP,
            group_id=GROUP_ID,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            request_timeout_ms=10_000,
            session_timeout_ms=10_000,
        )
        _active = True
        _last_flush = time.monotonic()
        logger.info(f"Dashboard consumer démarré ← topic '{TOPIC}'")
    except Exception as exc:
        logger.warning(f"Kafka dashboard consumer: connexion échouée ({exc})")
        return

    while _running:
        records = consumer.poll(timeout_ms=500)
        new_alerts = []
        for _, messages in records.items():
            for msg in messages:
                new_alerts.append(msg.value)

        if new_alerts:
            with _lock:
                for a in new_alerts:
                    _alert_buffer.append(a)
            with _file_lock:
                _write_batch.extend(new_alerts)

        now = time.monotonic()
        if len(_write_batch) >= FLUSH_EVERY or (now - _last_flush) >= FLUSH_INTERVAL:
            _flush_to_file()

    _flush_to_file()
    consumer.close()
    _active = False
    logger.info("Dashboard consumer arrêté")


def _kafka_reachable(timeout_ms: int = 1500) -> bool:
    """Sonde TCP brute (voir core/startup.py._kafka_reachable pour le pourquoi :
    KafkaAdminClient ignore request_timeout_ms au bootstrap et peut bloquer ~30s)."""
    host, _, port_s = BOOTSTRAP.partition(":")
    try:
        port = int(port_s or 9092)
        with socket.create_connection((host, port), timeout=timeout_ms / 1000):
            return True
    except OSError:
        return False


def _simulate_loop() -> None:
    """Fallback sans Kafka : génère des alertes réalistes directement dans le buffer.
    N'active PAS `_active` (réservé au vrai pipeline Kafka) — l'UI affiche correctement
    'Simulateur (fallback)' tout en ayant des données à montrer."""
    global _write_batch, _last_flush
    from core.simulator import generate_alert, generate_history

    seed = generate_history(hours=24, avg_per_hour=80)
    with _lock:
        for a in seed[-_alert_buffer.maxlen:]:
            _alert_buffer.append(a)
    with _file_lock:
        _write_batch.extend(seed)
    _last_flush = time.monotonic()
    _flush_to_file()
    logger.info(f"Simulateur local démarré — {len(seed)} alertes d'historique générées")

    while _running:
        time.sleep(random.uniform(1.5, 3.0))
        batch = [generate_alert() for _ in range(random.randint(1, 4))]
        with _lock:
            for a in batch:
                _alert_buffer.append(a)
        with _file_lock:
            _write_batch.extend(batch)
        now = time.monotonic()
        if len(_write_batch) >= FLUSH_EVERY or (now - _last_flush) >= FLUSH_INTERVAL:
            _flush_to_file()

    _flush_to_file()
    logger.info("Simulateur local arrêté")


def start_kafka_consumer() -> None:
    global _running
    if _running:
        return
    _running = True
    if _kafka_reachable():
        t = threading.Thread(target=_consume_loop, daemon=True, name="kafka-dashboard-consumer")
    else:
        t = threading.Thread(target=_simulate_loop, daemon=True, name="kafka-dashboard-simulator")
    t.start()


def get_pending_alerts() -> list[dict]:
    """Retourne et vide le buffer d'alertes accumulées depuis le dernier appel."""
    with _lock:
        items = list(_alert_buffer)
        _alert_buffer.clear()
    return items


def is_kafka_active() -> bool:
    return _active

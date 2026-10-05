"""
blockchain_publisher.py — Consomme ids-detections et publie automatiquement
les alertes CRITICAL et HIGH sur la blockchain sans intervention manuelle.
"""
from __future__ import annotations
import json
import threading
import logging
import os

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "ids-detections"
GROUP_ID = "ids-blockchain-publisher"
AUTO_SEVERITIES = {"CRITICAL", "HIGH"}

logger = logging.getLogger(__name__)

_running = False
_published_count = 0
_lock = threading.Lock()


def _publish_loop() -> None:
    global _published_count
    try:
        from kafka import KafkaConsumer
    except ImportError:
        logger.warning("kafka-python non installé — blockchain publisher désactivé")
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
    except Exception as exc:
        logger.warning(f"Kafka blockchain publisher: connexion échouée ({exc})")
        return

    from core.blockchain_client import get_blockchain_client
    bc = get_blockchain_client()
    logger.info(f"Blockchain publisher démarré ← '{TOPIC}' (auto-publish: {AUTO_SEVERITIES})")

    while _running:
        records = consumer.poll(timeout_ms=500)
        for _, messages in records.items():
            for msg in messages:
                alert = msg.value
                if alert.get("severity") not in AUTO_SEVERITIES:
                    continue
                try:
                    block = bc.publish({
                        "label":        alert.get("label"),
                        "severity":     alert.get("severity"),
                        "confidence":   alert.get("confidence"),
                        "timestamp":    alert.get("timestamp"),
                        "feature_hash": alert.get("feature_hash", ""),
                    })
                    with _lock:
                        _published_count += 1
                    logger.info(
                        f"Blockchain ← [{alert.get('severity')}] {alert.get('label')} "
                        f"→ txid {block.get('txid', '?')}"
                    )
                except Exception as exc:
                    logger.error(f"Erreur publication blockchain: {exc}")

    consumer.close()
    logger.info("Blockchain publisher arrêté")


def start_blockchain_publisher() -> None:
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_publish_loop, daemon=True, name="kafka-blockchain-publisher")
    t.start()


def get_published_count() -> int:
    with _lock:
        return _published_count

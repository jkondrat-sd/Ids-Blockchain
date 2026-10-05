"""
producer.py — Génère des flux réseau bruts (simulés) et les publie sur Kafka.
Topic: network-flows
"""
from __future__ import annotations
import json
import time
import random
import threading
import logging
import os

from core.simulator import generate_flow_features

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = "network-flows"
BATCH_INTERVAL = 2.0

logger = logging.getLogger(__name__)

_running = False


def _produce_loop() -> None:
    try:
        from kafka import KafkaProducer
    except ImportError:
        logger.warning("kafka-python non installé — producer désactivé")
        return

    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=5000,
            connections_max_idle_ms=10_000,
        )
    except Exception as exc:
        logger.warning(f"Kafka producer: connexion échouée ({exc})")
        return

    logger.info(f"Kafka producer démarré → topic '{TOPIC}' sur {BOOTSTRAP}")
    while _running:
        n = random.randint(5, 10)
        for _ in range(n):
            flow = generate_flow_features()
            producer.send(TOPIC, flow)
        producer.flush()
        time.sleep(BATCH_INTERVAL)

    producer.close()
    logger.info("Kafka producer arrêté")


def start_producer() -> None:
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_produce_loop, daemon=True, name="kafka-producer")
    t.start()

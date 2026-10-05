"""
inference_worker.py — Consomme network-flows, passe par le modèle, publie sur ids-detections.
Topic entrée : network-flows
Topic sortie : ids-detections
"""
from __future__ import annotations
import json
import threading
import logging
import os

import pandas as pd

from core.model_client import get_model_client
from core.simulator import ATTACK_DESC

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
IN_TOPIC = "network-flows"
OUT_TOPIC = "ids-detections"
BATCH_SIZE = 10

logger = logging.getLogger(__name__)

_running = False


def _inference_loop() -> None:
    try:
        from kafka import KafkaConsumer, KafkaProducer
    except ImportError:
        logger.warning("kafka-python non installé — inference worker désactivé")
        return

    try:
        consumer = KafkaConsumer(
            IN_TOPIC,
            bootstrap_servers=BOOTSTRAP,
            group_id="ids-inference",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            request_timeout_ms=10_000,
            session_timeout_ms=10_000,
        )
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except Exception as exc:
        logger.warning(f"Kafka inference worker: connexion échouée ({exc})")
        return

    model = get_model_client()
    logger.info(f"Inference worker démarré — '{IN_TOPIC}' → '{OUT_TOPIC}' (modèle: {model.status})")

    while _running:
        raw = consumer.poll(timeout_ms=1000, max_records=BATCH_SIZE)
        if not raw:
            continue

        batch_features: list[dict] = []
        batch_meta: list[dict] = []

        for _, messages in raw.items():
            for msg in messages:
                flow = msg.value
                batch_features.append(flow.get("features", {}))
                batch_meta.append({
                    "timestamp":   flow.get("timestamp", ""),
                    "src_ip":      flow.get("src_ip", ""),
                    "dst_ip":      flow.get("dst_ip", ""),
                    "dst_port":    flow.get("dst_port", 0),
                    "protocol":    flow.get("protocol", "TCP"),
                    "duration_us": flow.get("duration_us", 0),
                })

        if not batch_features:
            continue

        try:
            df = pd.DataFrame(batch_features)
            predictions = model.predict_csv(df)
            for pred, meta in zip(predictions, batch_meta):
                alert = {
                    "label":        pred["label"],
                    "severity":     pred["severity"],
                    "confidence":   pred["confidence"],
                    "timestamp":    meta["timestamp"] or pred.get("timestamp", ""),
                    "src_ip":       meta["src_ip"],
                    "dst_ip":       meta["dst_ip"],
                    "dst_port":     meta["dst_port"],
                    "protocol":     meta["protocol"],
                    "duration_us":  meta["duration_us"],
                    "feature_hash": pred.get("feature_hash", ""),
                    "description":  ATTACK_DESC.get(pred["label"], ""),
                }
                producer.send(OUT_TOPIC, alert)
            producer.flush()
            logger.debug(f"Batch de {len(predictions)} flows traités et publiés sur '{OUT_TOPIC}'")
        except Exception as exc:
            logger.error(f"Erreur lors de l'inférence: {exc}")

    consumer.close()
    producer.close()
    logger.info("Inference worker arrêté")


def start_inference_worker() -> None:
    global _running
    if _running:
        return
    _running = True
    t = threading.Thread(target=_inference_loop, daemon=True, name="kafka-inference")
    t.start()

"""
startup.py — Démarre Kafka automatiquement via Docker si nécessaire et crée les topics.
Appelé une seule fois au chargement du dashboard (app.py).
"""
from __future__ import annotations
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPICS = ["network-flows", "ids-detections"]
COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.yml"
KAFKA_TIMEOUT = 45  # secondes max d'attente

logger = logging.getLogger(__name__)


def _kafka_reachable(bootstrap: str = BOOTSTRAP, timeout_ms: int = 2000) -> bool:
    """Sonde TCP brute plutôt que KafkaAdminClient : ce dernier ignore
    request_timeout_ms pendant le bootstrap initial et peut bloquer ~30s
    même quand aucun broker n'écoute (bug connu de kafka-python)."""
    host, _, port_s = bootstrap.partition(":")
    try:
        port = int(port_s or 9092)
        with socket.create_connection((host, port), timeout=timeout_ms / 1000):
            return True
    except OSError:
        return False


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _start_docker_kafka() -> bool:
    """Lance docker compose up -d depuis la racine du projet."""
    if not COMPOSE_FILE.exists():
        logger.warning(f"docker-compose.yml introuvable : {COMPOSE_FILE}")
        return False

    compose_dir = str(COMPOSE_FILE.parent)

    # Essaie d'abord `docker compose` (v2), puis `docker-compose` (v1)
    for cmd in [
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--remove-orphans"],
        ["docker-compose", "-f", str(COMPOSE_FILE), "up", "-d", "--remove-orphans"],
    ]:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=30, cwd=compose_dir,
            )
            if result.returncode == 0:
                logger.info("Kafka démarré via Docker Compose")
                return True
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning(f"Erreur docker compose: {exc}")

    return False


def _wait_for_kafka(timeout: int = KAFKA_TIMEOUT) -> bool:
    logger.info(f"Attente de Kafka ({timeout}s max)...")
    for i in range(timeout):
        if _kafka_reachable():
            logger.info(f"Kafka prêt après {i + 1}s")
            return True
        time.sleep(1)
    return False


def _create_topics() -> None:
    try:
        from kafka import KafkaAdminClient
        from kafka.admin import NewTopic
        from kafka.errors import TopicAlreadyExistsError

        client = KafkaAdminClient(bootstrap_servers=BOOTSTRAP, request_timeout_ms=5000)
        existing = set(client.list_topics())

        to_create = [
            NewTopic(name=t, num_partitions=1, replication_factor=1)
            for t in TOPICS if t not in existing
        ]
        if to_create:
            client.create_topics(to_create, validate_only=False)
            logger.info(f"Topics créés : {[t.name for t in to_create]}")
        else:
            logger.info("Topics déjà existants")

        client.close()
    except Exception as exc:
        logger.warning(f"Création topics échouée (auto-create activé sur le broker): {exc}")


def ensure_kafka_running() -> tuple[bool, str]:
    """
    Garantit que Kafka est disponible.
    Retourne (succès: bool, message: str) pour affichage dans le dashboard.
    """
    # 1. Déjà disponible ?
    if _kafka_reachable():
        _create_topics()
        return True, "Kafka déjà actif sur localhost:9092"

    # 2. Essayer Docker
    if not _docker_available():
        return False, "Kafka non disponible et Docker absent — mode simulateur activé"

    logger.info("Kafka absent — démarrage via Docker Compose...")
    started = _start_docker_kafka()
    if not started:
        return False, "Docker Compose a échoué — mode simulateur activé"

    # 3. Attendre que Kafka soit prêt
    ready = _wait_for_kafka()
    if not ready:
        return False, f"Kafka n'a pas démarré dans les {KAFKA_TIMEOUT}s — mode simulateur activé"

    _create_topics()
    return True, "Kafka démarré automatiquement via Docker"

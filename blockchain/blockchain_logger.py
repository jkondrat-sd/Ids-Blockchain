"""
blockchain_logger.py — Fichier de test pour vérifier l'écriture sur MultiChain.

Ce fichier sert uniquement à tester que la blockchain fonctionne correctement
en simulant des alertes comme si elles venaient du module IA.

Pour l'intégration réelle avec Walid, voir : blockchain_interface/api.py
"""

import json
import datetime
import subprocess

from config import MULTICHAIN_CLI, CHAIN_NAME, STREAM_NAME, STREAM_KEY


def run_multichain_cli(args):
    """Lance une commande multichain-cli et retourne la sortie."""
    try:
        completed = subprocess.run(
            [MULTICHAIN_CLI, CHAIN_NAME] + args,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"ERREUR multichain-cli : {e.stderr.strip()}")
        raise
    except FileNotFoundError:
        print(f"ERREUR : multichain-cli introuvable à '{MULTICHAIN_CLI}'.")
        raise


def publier_alerte(alerte: dict) -> str:
    """
    Publie une alerte dans le stream MultiChain.

    alerte : dictionnaire contenant les infos de l'intrusion.
    Retourne le txid (identifiant de la transaction sur la blockchain).
    """

    # Ajoute l'horodatage si absent
    if "timestamp" not in alerte:
        alerte["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Encode en JSON puis en hexadécimal (format requis par MultiChain)
    hex_data = json.dumps(alerte, ensure_ascii=False).encode("utf-8").hex()

    print(f"[TEST] Publication de l'alerte '{alerte['label']}' dans le stream '{STREAM_NAME}'...")
    txid = run_multichain_cli(["publish", STREAM_NAME, STREAM_KEY, hex_data])
    print(f"[TEST] OK — TXID : {txid}")
    return txid


if __name__ == "__main__":
    # ── Alertes de test ───────────────────────────────────────────
    # Ces alertes simulent ce que le module IA enverrait en production.
    # Lance ce fichier pour vérifier que l'écriture dans le stream fonctionne.

    alertes_test = [
        {
            "label"        : "DDoS",
            "severity"     : "CRITICAL",
            "confidence"   : 0.97,
            "timestamp"    : "2026-04-19T10:00:00Z",
            "feature_hash" : "a3f1c2e4b5d6789012345678abcdef01",
        },
        {
            "label"        : "PortScan",
            "severity"     : "MEDIUM",
            "confidence"   : 0.85,
            "timestamp"    : "2026-04-19T10:01:00Z",
            "feature_hash" : "b7e2d3f5a6c8901234567890bcdef012",
        },
        {
            "label"        : "BruteForce",
            "severity"     : "HIGH",
            "confidence"   : 0.91,
            "timestamp"    : "2026-04-19T10:02:00Z",
            "feature_hash" : "c8f3e4a7b9d0123456789012cdef0123",
        },
    ]

    print(f"=== Test d'écriture dans le stream '{STREAM_NAME}' ===\n")
    for alerte in alertes_test:
        try:
            txid = publier_alerte(alerte)
        except Exception as e:
            print(f"[TEST] ECHEC pour '{alerte['label']}' : {e}\n")

    print("\n=== Fin du test. Lance blockchain_reader.py pour vérifier. ===")

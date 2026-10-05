"""
blockchain_reader.py — Lecture et vérification des logs stockés sur la blockchain.

Ce fichier permet de :
  1. Récupérer toutes les intrusions enregistrées dans le stream.
  2. Récupérer une intrusion précise via son txid.
  3. Afficher un résumé lisible des logs.

Pourquoi est-ce utile ?
  → Cela permet à l'équipe (ou au dashboard) de consulter l'historique
    complet des intrusions détectées, avec la garantie que ces données
    n'ont pas été altérées (propriété d'immuabilité de la blockchain).
"""

import json

from config import MULTICHAIN_CLI, CHAIN_NAME, STREAM_NAME, STREAM_KEY


# ──────────────────────────────────────────────────────────────
# Fonction utilitaire (même que dans blockchain_logger.py)
# ──────────────────────────────────────────────────────────────

def run_multichain_cli(args):
    """
    Exécute une commande multichain-cli et retourne la sortie brute.
    Voir blockchain_logger.py pour les détails.
    """
    import subprocess
    try:
        completed = subprocess.run(
            [MULTICHAIN_CLI, CHAIN_NAME] + args,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"ERREUR multichain-cli {args}: {e.stderr.strip()}")
        raise
    except FileNotFoundError:
        print(f"ERREUR : multichain-cli introuvable à '{MULTICHAIN_CLI}'.")
        raise


# ──────────────────────────────────────────────────────────────
# Décodage des données stockées sur la blockchain
# ──────────────────────────────────────────────────────────────

def decode_hex_data(hex_str: str) -> dict:
    """
    Décode une chaîne hexadécimale en dictionnaire Python.

    Lors de l'enregistrement (blockchain_logger.py), on a converti
    le JSON en hexadécimal. Ici on fait l'opération inverse :
      hex → bytes UTF-8 → chaîne JSON → dictionnaire Python.
    """
    try:
        raw_bytes = bytes.fromhex(hex_str)
        json_str  = raw_bytes.decode("utf-8")
        return json.loads(json_str)
    except Exception as e:
        # Si le décodage échoue, on retourne un dict avec l'hex brut
        return {"raw_hex": hex_str, "decode_error": str(e)}


def parse_stream_item(item: dict) -> dict:
    """
    Transforme une entrée brute du stream MultiChain en log lisible.

    MultiChain retourne chaque entrée sous cette forme :
    {
        "txid"  : "abc123...",
        "key"   : "intrusion",
        "data"  : "7b226c6162656c22...",   ← données en hex
        "confirmations": 5,
        ...
    }

    Cette fonction extrait et décode les données utiles.
    """
    txid          = item.get("txid", "")
    confirmations = item.get("confirmations", 0)

    # Les données sont dans item["data"] (format hex)
    hex_data = item.get("data", "")
    log_data = decode_hex_data(hex_data)

    return {
        "txid"          : txid,
        "confirmations" : confirmations,  # nombre de blocs confirmant la transaction
        "log"           : log_data,       # les données décodées (label, IP, etc.)
    }


# ──────────────────────────────────────────────────────────────
# Récupération des entrées du stream
# ──────────────────────────────────────────────────────────────

def get_all_intrusions(count: int = 50, start: int = -50) -> list:
    """
    Récupère les dernières entrées du stream 'intrusion-logs'.

    count : nombre maximum d'entrées à récupérer (défaut : 50).
    start : position de départ. -50 = les 50 dernières entrées.
            0 = depuis le début du stream.

    Retourne : liste de logs décodés, du plus récent au plus ancien.

    Commande MultiChain utilisée :
      multichain-cli ids-chain liststreamitems intrusion-logs false <count> <start>
      → "false" = ne pas inclure les données de la blockchain (blocs, etc.)
    """
    print(f"[BLOCKCHAIN] Récupération des {count} dernières intrusions...")

    raw_output = run_multichain_cli([
        "liststreamitems",
        STREAM_NAME,
        "false",       # verbose = false (moins de données inutiles)
        str(count),
        str(start),
    ])

    # La sortie est un JSON array
    items = json.loads(raw_output)

    # Décoder chaque entrée
    decoded = [parse_stream_item(item) for item in items]

    print(f"[BLOCKCHAIN] {len(decoded)} intrusion(s) récupérée(s).")
    return decoded


def get_intrusion_by_txid(txid: str) -> dict:
    """
    Récupère une intrusion spécifique par son identifiant de transaction (txid).

    txid : la chaîne hexadécimale retournée lors de l'enregistrement
           (valeur retournée par receive_and_log dans blockchain_logger.py).

    Utile pour vérifier qu'une entrée précise est bien sur la blockchain,
    par exemple après un incident de sécurité.

    Commande MultiChain :
      multichain-cli ids-chain gettxoutdata <txid>
    """
    print(f"[BLOCKCHAIN] Recherche de la transaction {txid}...")

    # getrawtransaction retourne les détails complets d'une transaction
    raw_output = run_multichain_cli(["getrawtransaction", txid, "1"])
    tx_data    = json.loads(raw_output)

    # Les données du stream sont dans vout[0].data (dépend de la version MultiChain)
    # On cherche le champ "data" dans les sorties de la transaction
    for vout in tx_data.get("vout", []):
        script_asm = vout.get("scriptPubKey", {}).get("asm", "")
        if "OP_RETURN" in script_asm:
            # Les données sont encodées après OP_RETURN
            hex_data = script_asm.split(" ")[-1]
            return {
                "txid"          : txid,
                "confirmations" : tx_data.get("confirmations", 0),
                "log"           : decode_hex_data(hex_data),
            }

    return {"txid": txid, "error": "Données introuvables dans la transaction."}


def get_intrusions_by_label(label: str) -> list:
    """
    Filtre les intrusions par type d'attaque (label).

    label : ex. "DDoS", "PortScan", "BruteForce"...

    Récupère toutes les entrées du stream et ne garde que celles
    dont le label correspond. Utile pour analyser un type d'attaque précis.
    """
    all_entries = get_all_intrusions(count=200, start=-200)

    filtered = [
        entry for entry in all_entries
        if entry.get("log", {}).get("label", "").lower() == label.lower()
    ]

    print(f"[BLOCKCHAIN] {len(filtered)} intrusion(s) de type '{label}' trouvée(s).")
    return filtered


# ──────────────────────────────────────────────────────────────
# Affichage lisible
# ──────────────────────────────────────────────────────────────

def print_intrusion_summary(entries: list):
    """
    Affiche un résumé tabulaire des intrusions récupérées.
    Utile pour le débogage ou pour un rapport rapide dans le terminal.
    """
    if not entries:
        print("Aucune intrusion enregistrée.")
        return

    print("\n" + "=" * 70)
    print(f"{'TXID[:12]':<14} {'Timestamp':<25} {'Label':<15} {'Confiance':>9} {'IP Source'}")
    print("=" * 70)

    for entry in entries:
        txid  = entry.get("txid", "")[:12]      # on affiche les 12 premiers chars
        log   = entry.get("log", {})
        ts    = log.get("timestamp", "N/A")[:19] # sans les millisecondes
        label = log.get("label", "N/A")
        conf  = log.get("confidence", 0.0)
        src   = log.get("src_ip", "N/A")

        print(f"{txid:<14} {ts:<25} {label:<15} {conf:>9.2%} {src}")

    print("=" * 70)
    print(f"Total : {len(entries)} intrusion(s)\n")


# ──────────────────────────────────────────────────────────────
# Test rapide (à exécuter directement)
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Récupère les 20 dernières intrusions et les affiche
    entries = get_all_intrusions(count=20)
    print_intrusion_summary(entries)

"""
config.py — Configuration centralisée du module blockchain.

Au lieu de coder en dur les chemins et noms dans chaque fichier,
on les définit ici une seule fois. Si tu changes de machine ou
de nom de chaîne, tu modifies uniquement ce fichier.
"""

import os

# ──────────────────────────────────────────────
# Chemin vers l'exécutable multichain-cli
# On essaie d'abord la variable d'environnement MULTICHAIN_CLI,
# sinon on utilise le chemin par défaut sur cette machine.
# ──────────────────────────────────────────────
MULTICHAIN_CLI = os.environ.get(
    "MULTICHAIN_CLI",
    r"C:\multichain\multichain-cli.exe"
)

# Nom de la blockchain MultiChain qu'on a créée
CHAIN_NAME = os.environ.get("CHAIN_NAME", "ids-chain")

# Nom du stream qui stocke les logs d'intrusion
STREAM_NAME = os.environ.get("STREAM_NAME", "intrusion-logs")

# Clé utilisée pour identifier les entrées dans le stream.
# MultiChain associe chaque entrée à une "key" (comme un tag).
# On utilise "intrusion" pour toutes les détections d'attaques.
STREAM_KEY = "intrusion"

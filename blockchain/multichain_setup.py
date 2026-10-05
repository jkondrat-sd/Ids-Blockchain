import subprocess
import sys

# Chemin vers multichain-cli
MULTICHAIN_CLI = r"C:\multichain\multichain-cli.exe"
CHAIN_NAME = "ids-chain"
STREAM_NAME = "intrusion-logs"


def run_multichain_cli(args):
    """Exécute une commande multichain-cli et retourne la sortie standard."""
    try:
        completed = subprocess.run(
            [MULTICHAIN_CLI, CHAIN_NAME] + args,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Affichage du message d'erreur renvoyé par multichain-cli
        print(f"ERREUR lors de l'exécution de multichain-cli {args}: {e.stderr.strip()}")
        raise
    except FileNotFoundError:
        print(f"ERREUR : multichain-cli introuvable à '{MULTICHAIN_CLI}'.")
        raise


def check_chain_running():
    """Vérifie que la blockchain tourne en demandant des infos basiques."""
    try:
        print("Étape 1 : Vérification du statut de la blockchain ids-chain...")
        output = run_multichain_cli(["getinfo"])
        print("OK : la blockchain ids-chain est en fonctionnement.")
        print(output)
    except Exception:
        print("Échec : impossible de vérifier que la blockchain ids-chain tourne.")
        raise


def create_stream():
    """Crée un stream si non existant."""
    try:
        print(f"Étape 2 : Création du stream '{STREAM_NAME}'...")
        # La commande create stream échoue si le stream existe déjà
        output = run_multichain_cli(["create", "stream", STREAM_NAME, "true"])
        print(f"OK : stream '{STREAM_NAME}' créé avec succès.")
        print(output)
    except subprocess.CalledProcessError as e:
        # Si le stream existe déjà, on le considère comme un succès
        if "already exists" in e.stderr.lower():
            print(f"Info : le stream '{STREAM_NAME}' existe déjà.")
        else:
            print(f"Échec : impossible de créer le stream '{STREAM_NAME}'.")
            raise


def subscribe_stream():
    """S'abonne au stream."""
    try:
        print(f"Étape 3 : Abonnement au stream '{STREAM_NAME}'...")
        output = run_multichain_cli(["subscribe", STREAM_NAME])
        print(f"OK : abonné au stream '{STREAM_NAME}' avec succès.")
        print(output)
    except subprocess.CalledProcessError as e:
        if "already subscribed" in e.stderr.lower() or "already subscribed" in e.stdout.lower():
            print(f"Info : déjà abonné au stream '{STREAM_NAME}'.")
        else:
            print(f"Échec : impossible de s'abonner au stream '{STREAM_NAME}'.")
            raise


if __name__ == "__main__":
    try:
        check_chain_running()
        create_stream()
        subscribe_stream()
        print("\nToutes les étapes sont terminées avec succès.")
    except Exception as e:
        print("\nScript arrêté en raison d'une erreur.")
        print(str(e))
        sys.exit(1)

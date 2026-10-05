# Module Blockchain — IDS-Blockchain

**Réalisé par : Meryem Ait Tadamna**
**Groupe 3 — Projet Système de Détection d'Intrusion (IDS)**

---

## Objectif

Ce module est responsable de l'**enregistrement immuable** des intrusions détectées par le module IA (LSTM).

Chaque fois que l'IA détecte une attaque réseau, ce module publie l'alerte sur une blockchain **MultiChain**. Une fois enregistrée, cette alerte ne peut plus être modifiée ni supprimée — ce qui garantit l'intégrité totale des logs de sécurité.

---

## Pourquoi la blockchain ?

Une base de données classique peut être compromise : un attaquant peut effacer les traces de son passage.

Avec MultiChain, chaque entrée est **enchaînée cryptographiquement** à la précédente via un hash SHA-256. Si quelqu'un tente de modifier un enregistrement, toute la chaîne devient invalide et la fraude est immédiatement détectable.

```
Bloc 0 → Bloc 1 → Bloc 2 → Bloc 3
  ↑          ↑        ↑        ↑
contient  contient contient contient
hash(0)   hash(1)  hash(2)  hash(3)
```

---

## Architecture du module

```
blockchain/
├── config.py                      # Paramètres centralisés (chemins, noms)
├── multichain_setup.py            # Initialisation du stream MultiChain (lancé une fois)
├── blockchain_logger.py           # Formate et publie les alertes sur la blockchain
├── blockchain_reader.py           # Lit et vérifie les alertes enregistrées
└── blockchain_interface/
    ├── __init__.py
    └── api.py                     # Point d'entrée appelé par le module IA (Walid)
```

---

## Flux de données

```
Module IA (IDSPredictor)
        |
        | appelle log_prediction(prediction)
        ↓
blockchain_interface/api.py
        |
        | construit un bloc SHA-256 chainé
        | appelle _submit_to_chain()
        ↓
MultiChain (stream: intrusion-logs)
        |
        | stocké de façon permanente
        ↓
blockchain_reader.py  ←  consultation / vérification
```

---

## Structure d'un bloc enregistré

Chaque alerte est stockée sous cette forme sur la blockchain :

```json
{
  "block_idx":    3,
  "payload": {
    "timestamp":    "2026-04-18T10:23:45Z",
    "label":        "DDoS",
    "severity":     "CRITICAL",
    "confidence":   0.97,
    "feature_hash": "a3f1c2e4b5d6..."
  },
  "payload_hash": "f7c3a1...",
  "prev_hash":    "8e2b4d...",
  "block_hash":   "c2d5f8...",
  "on_chain":     true
}
```

| Champ | Description |
|-------|-------------|
| `label` | Type d'attaque détecté (DDoS, PortScan, BruteForce...) |
| `severity` | Niveau de criticité : INFO / MEDIUM / HIGH / CRITICAL |
| `confidence` | Score de confiance du modèle LSTM (0.0 à 1.0) |
| `feature_hash` | Empreinte SHA-256 du vecteur de features réseau |
| `prev_hash` | Hash du bloc précédent — ce qui forme la chaîne |
| `block_hash` | Empreinte unique de ce bloc |
| `on_chain` | `true` si la publication sur MultiChain a réussi |

---

## Utilisation

### 1. Initialisation (une seule fois)

Avant tout, démarrer la blockchain MultiChain et lancer :

```bash
python multichain_setup.py
```

Cela crée le stream `intrusion-logs` sur la chaîne `ids-chain`.

### 2. Enregistrer une intrusion (appelé par le module IA)

```python
from blockchain_interface.api import log_prediction

prediction = {
    "label":        "DDoS",
    "severity":     "CRITICAL",
    "confidence":   0.97,
    "timestamp":    "2026-04-18T10:23:45Z",
    "feature_hash": "a3f1c2e4b5d6...",
}

block = log_prediction(prediction)
print(block["on_chain"])   # True si enregistré avec succès
```

### 3. Lire les intrusions enregistrées

```bash
python blockchain_reader.py
```

Ou depuis le code :

```python
from blockchain_reader import get_all_intrusions, print_intrusion_summary

entries = get_all_intrusions(count=20)
print_intrusion_summary(entries)
```

### 4. Vérifier l'intégrité de la chaîne

```python
from blockchain_interface.api import verify_chain

print(verify_chain())   # True = chaîne intacte, False = fraude détectée
```

---

## Configuration

Tous les paramètres sont dans `config.py`. Ils peuvent aussi être définis comme variables d'environnement :

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `MULTICHAIN_CLI` | `C:\multichain\multichain-cli.exe` | Chemin vers multichain-cli |
| `CHAIN_NAME` | `ids-chain` | Nom de la blockchain |
| `STREAM_NAME` | `intrusion-logs` | Nom du stream de logs |

---

## Technologies utilisées

- **MultiChain 2.3.3** — blockchain privée pour le stockage immuable
- **Python 3** — implémentation du module
- **SHA-256** — algorithme de hachage pour le chaînage des blocs

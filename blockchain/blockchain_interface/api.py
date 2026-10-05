"""
blockchain_interface/api.py — Bridge between the LSTM-IDS and the blockchain layer.

The ML side calls log_prediction() with a prediction dict from IDSPredictor.predict().
This module formats it into a tamper-evident block (SHA-256 chained) and submits
the hashes to MultiChain for immutable storage.

How the integration works
--------------------------
  1. Walid's IDSPredictor calls log_prediction(prediction).
  2. This module builds a SHA-256 chained block from the prediction.
  3. _submit_to_chain() publishes the block hashes to the MultiChain stream
     "intrusion-logs" using multichain-cli.
  4. The block is also kept in memory for fast local verification (verify_chain).

Note: We use MultiChain (not Ethereum/web3) so there is no gas cost.
      We store the full block data on-chain (not just hashes).

Data schema received from the ML module (IDSPredictor.predict())
-----------------------------------------------------------------
prediction = {
    "label":        str,    # e.g. "DDoS-HOIC"
    "severity":     str,    # "INFO" | "MEDIUM" | "HIGH" | "CRITICAL"
    "confidence":   float,  # 0.0 – 1.0
    "timestamp":    str,    # ISO-8601 UTC
    "feature_hash": str,    # SHA-256 of the raw feature vector
}

Block schema stored on MultiChain
-----------------------------------
{
    "block_idx":    int,    # monotonically increasing index
    "payload":      dict,   # the full prediction dict
    "payload_hash": str,    # SHA-256(json(payload))
    "prev_hash":    str,    # hash of the previous block (links the chain)
    "block_hash":   str,    # SHA-256(prev_hash + payload_hash + block_idx)
    "on_chain":     bool,   # True if MultiChain submission succeeded
}
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import os

# ── Import config from the parent blockchain/ directory ──────────────────────
# This file lives in blockchain/blockchain_interface/api.py
# config.py lives in blockchain/config.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MULTICHAIN_CLI, CHAIN_NAME, STREAM_NAME


# ─────────────────────────────────────────────────────────────────────────────
# In-memory chain state
# Used for local integrity checks via verify_chain().
# In production, replace with a persistent database.
# ─────────────────────────────────────────────────────────────────────────────

_chain: list[dict] = []
_GENESIS_HASH = "0" * 64   # first block has no predecessor, so prev_hash = 64 zeros


def _prev_hash() -> str:
    """Returns the block_hash of the last block, or the genesis hash if empty."""
    return _chain[-1]["block_hash"] if _chain else _GENESIS_HASH


def _sha256(data: str) -> str:
    """Computes the SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# MultiChain helper
# ─────────────────────────────────────────────────────────────────────────────

def _run_multichain_cli(args: list) -> str:
    """
    Runs a multichain-cli command and returns the output.

    Example: _run_multichain_cli(["publish", "intrusion-logs", "key", "hexdata"])
    → executes: multichain-cli ids-chain publish intrusion-logs key hexdata
    """
    result = subprocess.run(
        [MULTICHAIN_CLI, CHAIN_NAME] + args,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def log_prediction(prediction: dict) -> dict:
    """
    Called by Walid's IDSPredictor after each positive detection.

    Steps:
      1. Compute SHA-256 of the prediction JSON (payload_hash).
      2. Compute the block_hash by chaining: prev_hash + payload_hash + block_idx.
         → This links every block to the previous one, making the chain tamper-evident.
      3. Submit the full block to MultiChain via _submit_to_chain().
      4. Add the block to the local in-memory chain.

    Returns the block dict with on_chain=True if submission succeeded.
    """
    block_idx    = len(_chain)
    payload_str  = json.dumps(prediction, sort_keys=True)
    payload_hash = _sha256(payload_str)
    prev         = _prev_hash()
    block_hash   = _sha256(prev + payload_hash + str(block_idx))

    block = {
        "block_idx":    block_idx,
        "payload":      prediction,
        "payload_hash": payload_hash,
        "prev_hash":    prev,
        "block_hash":   block_hash,
        "on_chain":     False,
    }

    # Submit to MultiChain; update on_chain flag based on success
    on_chain        = _submit_to_chain(block_hash, payload_hash, prev, block)
    block["on_chain"] = on_chain
    _chain.append(block)
    return block


def verify_chain() -> bool:
    """
    Verifies the integrity of the local in-memory chain.

    Re-computes each block_hash from scratch and compares it to the stored value.
    Returns False if any block has been tampered with.

    Note: this checks local state only. To verify what's on MultiChain,
          use blockchain_reader.py.
    """
    prev = _GENESIS_HASH
    for block in _chain:
        expected = _sha256(prev + block["payload_hash"] + str(block["block_idx"]))
        if expected != block["block_hash"]:
            return False
        prev = block["block_hash"]
    return True


def get_chain() -> list[dict]:
    """Returns a copy of the local in-memory chain."""
    return list(_chain)


# ─────────────────────────────────────────────────────────────────────────────
# MultiChain submission — THIS IS WHAT WALID LEFT FOR YOU TO IMPLEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _submit_to_chain(
    block_hash:   str,
    payload_hash: str,
    prev_hash:    str,
    block:        dict,
) -> bool:
    """
    Publishes a block to the MultiChain stream 'intrusion-logs'.

    Unlike Ethereum (where only hashes go on-chain to save gas), MultiChain
    has no gas cost, so we store the full block data for easy retrieval.

    What gets stored on MultiChain:
      - Key   : block_hash  (makes each entry uniquely searchable by block ID)
      - Data  : full block JSON, hex-encoded (MultiChain requires hex format)

    The chain of prev_hash fields means that if someone tampers with an old
    entry, all subsequent block_hashes become invalid — same principle as Bitcoin.

    Returns True if the transaction was published successfully, False otherwise.
    """
    try:
        # Serialize the full block to JSON, then encode as hex
        # MultiChain's publish command requires data in hexadecimal format
        block_json = json.dumps(block, sort_keys=True)
        hex_data   = block_json.encode("utf-8").hex()

        # Publish to the stream
        # Command: multichain-cli ids-chain publish intrusion-logs <block_hash> <hex_data>
        # The block_hash is used as the stream key so entries are searchable by block ID
        txid = _run_multichain_cli(["publish", STREAM_NAME, block_hash, hex_data])

        print(f"[BLOCKCHAIN] Block #{block['block_idx']} publié. "
              f"Label: {block['payload'].get('label', '?')} | TXID: {txid}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[BLOCKCHAIN] Erreur MultiChain lors de la publication : {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"[BLOCKCHAIN] multichain-cli introuvable à '{MULTICHAIN_CLI}'.")
        return False
    except Exception as e:
        print(f"[BLOCKCHAIN] Erreur inattendue : {e}")
        return False

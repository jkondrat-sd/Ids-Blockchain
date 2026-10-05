"""
blockchain_client.py — Wrapper around Meryem's blockchain module (MultiChain).
Falls back to a local in-memory SHA-256 chain when MultiChain is not running.
"""
from __future__ import annotations
import sys
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOCKCHAIN_ROOT = ROOT / "blockchain"


class BlockchainClient:
    def __init__(self):
        self.connected = False
        self._reader = None
        self._api = None
        self._local_chain: list[dict] = []
        self._genesis = "0" * 64
        self._connect()

    def _connect(self):
        sys.path.insert(0, str(BLOCKCHAIN_ROOT))
        try:
            import blockchain_reader as br
            br.get_all_intrusions(count=1)
            self._reader = br
            from blockchain_interface import api as ba
            self._api = ba
            self.connected = True
        except Exception:
            self.connected = False

    def get_alerts(self, count: int = 100) -> list[dict]:
        if self.connected and self._reader:
            try:
                entries = self._reader.get_all_intrusions(count=count)
                return [
                    {**e.get("log", {}), "txid": e.get("txid", ""),
                     "confirmations": e.get("confirmations", 0)}
                    for e in entries
                ]
            except Exception:
                pass
        return list(reversed(self._local_chain[-count:]))

    def publish(self, prediction: dict) -> dict:
        if self.connected and self._api:
            try:
                block = self._api.log_prediction(prediction)
                # log_prediction() de Meryem ne retourne pas txid/confirmations/timestamp
                if "txid" not in block:
                    block["txid"] = block.get("block_hash", "")[:16]
                if "confirmations" not in block:
                    block["confirmations"] = 0
                if "timestamp" not in block:
                    block["timestamp"] = prediction.get("timestamp", datetime.now(timezone.utc).isoformat())
                return block
            except Exception:
                pass
        return self._local_publish(prediction)

    def _local_publish(self, prediction: dict) -> dict:
        prev = self._local_chain[-1]["block_hash"] if self._local_chain else self._genesis
        idx = len(self._local_chain)
        payload_str = json.dumps(prediction, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        block_hash = hashlib.sha256((prev + payload_hash + str(idx)).encode()).hexdigest()
        block = {
            "block_idx": idx, "payload": prediction,
            "payload_hash": payload_hash, "prev_hash": prev,
            "block_hash": block_hash, "on_chain": False,
            "txid": block_hash[:16], "confirmations": 0,
            "timestamp": prediction.get("timestamp", datetime.now(timezone.utc).isoformat()),
            **prediction,
        }
        self._local_chain.append(block)
        return block

    def verify_chain(self) -> tuple[bool, str]:
        if self.connected and self._api:
            try:
                ok = self._api.verify_chain()
                return ok, "MultiChain" if ok else "FRAUDE DÉTECTÉE sur MultiChain"
            except Exception:
                pass
        prev = self._genesis
        for b in self._local_chain:
            expected = hashlib.sha256(
                (prev + b["payload_hash"] + str(b["block_idx"])).encode()
            ).hexdigest()
            if expected != b["block_hash"]:
                return False, f"Bloc #{b['block_idx']} corrompu (chaîne locale)"
            prev = b["block_hash"]
        return True, "Chaîne locale intacte"

    def chain_length(self) -> int:
        if self.connected and self._api:
            try:
                return len(self._api.get_chain())
            except Exception:
                pass
        return len(self._local_chain)

    def get_block(self, txid: str) -> dict | None:
        if self.connected and self._reader:
            try:
                return self._reader.get_intrusion_by_txid(txid)
            except Exception:
                pass
        for b in self._local_chain:
            if b.get("txid", "").startswith(txid[:8]):
                return b
        return None

    @property
    def status(self) -> str:
        return "MultiChain connectée" if self.connected else "Chaîne locale (simulation)"


_client: BlockchainClient | None = None

def get_blockchain_client() -> BlockchainClient:
    global _client
    if _client is None:
        _client = BlockchainClient()
    return _client

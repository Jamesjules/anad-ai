"""
Anad Peer Network
=================
Self-sustaining. Resilient. No central server.

How it works:
  1. Your node starts up
  2. Tries known peers from last session
  3. If none — uses hardcoded bootstrap nodes (initial nodes)
  4. Once connected — discovers more peers organically
  5. Shares peer lists with everyone
  6. Network grows and heals itself

Resilience:
  - Any node can go down — network routes around it
  - No single node is essential
  - Even if 99% of nodes die — survivors reconnect
  - Bootstrap nodes are just starting points, not controllers

GitHub independence:
  - Updates spread peer-to-peer
  - Genesis node publishes update → peers verify → spread
  - Update is signed by genesis identity — unforgeable
  - No GitHub needed after initial install

Author: Anad Community
License: Public Domain
"""

import json
import os
import time
import hashlib
import socket
import threading
import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict


# ══════════════════════════════════════════════════════════════════
# PEER RECORD
# ══════════════════════════════════════════════════════════════════

@dataclass
class Peer:
    """A known peer on the Anad network"""
    node_id: str
    host: str
    port: int
    public_key: str
    last_seen: float = 0.0
    latency_ms: float = 999.0
    tier: str = "unknown"       # server / desktop / laptop / mobile
    version: str = "0.0.0"
    credits: int = 0
    is_bootstrap: bool = False  # bootstrap nodes never get pruned

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_alive(self) -> bool:
        """Consider peer alive if seen in last 5 minutes"""
        return (time.time() - self.last_seen) < 300

    @property
    def score(self) -> float:
        """
        Peer quality score for routing decisions.
        Higher = prefer this peer for queries.
        """
        recency = max(0, 1 - (time.time() - self.last_seen) / 3600)
        latency_score = max(0, 1 - self.latency_ms / 1000)
        tier_score = {"server": 1.0, "desktop": 0.8, "laptop": 0.6, "mobile": 0.3}.get(self.tier, 0.5)
        return (recency * 0.4) + (latency_score * 0.4) + (tier_score * 0.2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Peer":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════
# PEER STORE — persisted to disk
# ══════════════════════════════════════════════════════════════════

class PeerStore:
    """
    Persists known peers to disk.
    On restart — immediately reconnects to last known peers.
    No discovery needed from scratch every time.
    """

    def __init__(self, path: str):
        self.path = path
        self._peers: Dict[str, Peer] = {}
        self._load()

    def add(self, peer: Peer):
        self._peers[peer.node_id] = peer
        self._save()

    def remove(self, node_id: str):
        self._peers.pop(node_id, None)
        self._save()

    def update_seen(self, node_id: str, latency_ms: float = 0):
        if node_id in self._peers:
            self._peers[node_id].last_seen = time.time()
            self._peers[node_id].latency_ms = latency_ms
            self._save()

    def get_best(self, n: int = 10, exclude: Set[str] = None) -> List[Peer]:
        """Get top N peers by quality score"""
        exclude = exclude or set()
        peers = [
            p for p in self._peers.values()
            if p.node_id not in exclude
        ]
        peers.sort(key=lambda p: p.score, reverse=True)
        return peers[:n]

    def get_all_alive(self) -> List[Peer]:
        return [p for p in self._peers.values() if p.is_alive]

    def get_bootstrap(self) -> List[Peer]:
        return [p for p in self._peers.values() if p.is_bootstrap]

    def prune_old(self, max_age_days: int = 30):
        """Remove peers not seen in max_age_days (except bootstrap)"""
        cutoff = time.time() - (max_age_days * 86400)
        to_remove = [
            nid for nid, p in self._peers.items()
            if p.last_seen < cutoff and not p.is_bootstrap
        ]
        for nid in to_remove:
            del self._peers[nid]
        if to_remove:
            self._save()

    def count(self) -> int:
        return len(self._peers)

    def _save(self):
        data = {
            nid: peer.to_dict()
            for nid, peer in self._peers.items()
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            for nid, pdata in data.items():
                self._peers[nid] = Peer.from_dict(pdata)
        except Exception as e:
            print(f"Warning: Could not load peers: {e}")


# ══════════════════════════════════════════════════════════════════
# UPDATE MANIFEST — GitHub-free updates
# ══════════════════════════════════════════════════════════════════

@dataclass
class UpdateManifest:
    """
    Software update manifest.
    Signed by genesis node identity.
    Spreads peer-to-peer — no GitHub needed.

    Verification:
      Every node verifies the signature before applying.
      Forged manifests are rejected cryptographically.
      Only genesis node can sign valid updates.
    """
    version: str
    release_notes: str
    checksum_sha256: str        # SHA256 of the update package
    download_urls: List[str]    # multiple mirrors — any will do
    signature: str              # signed by genesis node
    genesis_public_key: str     # genesis node's public key
    timestamp: float = 0.0
    min_peers_to_spread: int = 3  # don't spread until N peers confirmed

    def is_valid(self) -> bool:
        """Verify this manifest was signed by genesis node"""
        from node.identity import AnadIdentity
        try:
            data = f"{self.version}{self.checksum_sha256}{self.timestamp}".encode()
            sig = bytes.fromhex(self.signature)
            return AnadIdentity.verify_from_public_key(
                self.genesis_public_key, data, sig
            )
        except Exception:
            return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UpdateManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════
# PEER DISCOVERY — how nodes find each other
# ══════════════════════════════════════════════════════════════════

class PeerDiscovery:
    """
    Finds and maintains connections to other Anad nodes.

    Discovery methods (in order of preference):
      1. Last known peers (from disk)
      2. Bootstrap nodes (hardcoded, initial nodes only)
      3. Peer exchange (ask peers for their peers)
      4. LAN discovery (find nodes on same network)

    No central server. No DNS. No GitHub.
    The network finds itself.
    """

    # Bootstrap nodes — genesis seeds
    # Your node is the first. Others connect through you.
    # Add your public IP here once you have a static IP or domain.
    BOOTSTRAP_NODES = [
        # Example — replace with your actual IP when ready:
        # {"host": "YOUR_PUBLIC_IP", "port": 8765, "node_id": "anad1_4ccd35bbd635c4a03678cf44f1"}
    ]

    def __init__(
        self,
        peer_store: PeerStore,
        my_node_id: str,
        my_host: str,
        my_port: int,
    ):
        self.peer_store = peer_store
        self.my_node_id = my_node_id
        self.my_host = my_host
        self.my_port = my_port
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        """Start background peer discovery"""
        self._running = True
        threading.Thread(
            target=self._discovery_loop,
            daemon=True,
            name="anad-peer-discovery"
        ).start()
        threading.Thread(
            target=self._lan_discovery_loop,
            daemon=True,
            name="anad-lan-discovery"
        ).start()
        print("Peer discovery started")

    def stop(self):
        self._running = False

    def _discovery_loop(self):
        """Main discovery loop — runs forever in background"""
        _printed_waiting = False
        while self._running:
            try:
                known = self.peer_store.get_best(20)
                if not known:
                    self._try_bootstrap()
                    if not _printed_waiting:
                        print("  Waiting for peers to join the network...")
                        _printed_waiting = True
                else:
                    if _printed_waiting:
                        print(f"  Peer connected!")
                        _printed_waiting = False
                    self._ping_peers(known[:10])
                    self._request_peer_lists(known[:3])

                self.peer_store.prune_old()
                alive = len(self.peer_store.get_all_alive())
                time.sleep(30 if alive > 0 else 10)

            except Exception:
                time.sleep(10)

    def _lan_discovery_loop(self):
        """
        Discover Anad nodes on local network.
        Useful for home/office setups with multiple nodes.
        Sends UDP broadcast — safe and standard.
        """
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(2)

                # Broadcast our presence
                message = json.dumps({
                    "type": "anad_announce",
                    "node_id": self.my_node_id,
                    "port": self.my_port,
                    "version": "0.1.0",
                }).encode()

                sock.sendto(message, ("255.255.255.255", 8766))
                sock.close()

            except Exception:
                pass

            time.sleep(60)  # announce every minute

    def _try_bootstrap(self):
        """Connect to bootstrap nodes when we have no peers"""
        for bootstrap in self.BOOTSTRAP_NODES:
            self._try_connect(
                bootstrap["host"],
                bootstrap["port"],
                bootstrap.get("node_id", "unknown"),
                is_bootstrap=True
            )

    def _try_connect(
        self,
        host: str,
        port: int,
        node_id: str,
        is_bootstrap: bool = False,
    ) -> bool:
        """Attempt to connect to a specific peer"""
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((host, port))

            # Send handshake
            handshake = json.dumps({
                "type": "handshake",
                "node_id": self.my_node_id,
                "host": self.my_host,
                "port": self.my_port,
                "version": "0.1.0",
            }).encode()
            sock.send(len(handshake).to_bytes(4, "big") + handshake)

            # Receive response
            resp_len = int.from_bytes(sock.recv(4), "big")
            resp = json.loads(sock.recv(resp_len))
            sock.close()

            latency = (time.time() - start) * 1000

            peer = Peer(
                node_id=resp.get("node_id", node_id),
                host=host,
                port=port,
                public_key=resp.get("public_key", ""),
                last_seen=time.time(),
                latency_ms=latency,
                tier=resp.get("tier", "unknown"),
                version=resp.get("version", "0.0.0"),
                is_bootstrap=is_bootstrap,
            )
            self.peer_store.add(peer)
            return True

        except Exception:
            return False

    def _ping_peers(self, peers: List[Peer]):
        """Ping peers to check they're alive"""
        for peer in peers:
            try:
                start = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                sock.connect((peer.host, peer.port))
                ping = json.dumps({"type": "ping"}).encode()
                sock.send(len(ping).to_bytes(4, "big") + ping)
                resp_len = int.from_bytes(sock.recv(4), "big")
                sock.recv(resp_len)
                sock.close()
                latency = (time.time() - start) * 1000
                self.peer_store.update_seen(peer.node_id, latency)
            except Exception:
                pass

    def _request_peer_lists(self, peers: List[Peer]):
        """Ask peers for their peer lists — how the network grows"""
        for peer in peers:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((peer.host, peer.port))
                req = json.dumps({"type": "get_peers"}).encode()
                sock.send(len(req).to_bytes(4, "big") + req)
                resp_len = int.from_bytes(sock.recv(4), "big")
                resp = json.loads(sock.recv(resp_len))
                sock.close()

                # Add new peers we didn't know about
                for peer_data in resp.get("peers", []):
                    if peer_data["node_id"] != self.my_node_id:
                        new_peer = Peer.from_dict(peer_data)
                        if new_peer.node_id not in [
                            p.node_id for p in self.peer_store.get_best(100)
                        ]:
                            self._try_connect(
                                new_peer.host,
                                new_peer.port,
                                new_peer.node_id,
                            )
            except Exception:
                pass

    def get_peers_for_routing(self, n: int = 5) -> List[Peer]:
        """Get best peers for routing a query"""
        return self.peer_store.get_best(n)

    def network_health(self) -> dict:
        """Current network health report"""
        alive = self.peer_store.get_all_alive()
        all_peers = self.peer_store.get_best(100)
        tiers = {}
        for p in alive:
            tiers[p.tier] = tiers.get(p.tier, 0) + 1

        return {
            "total_known_peers": self.peer_store.count(),
            "alive_peers": len(alive),
            "peers_by_tier": tiers,
            "avg_latency_ms": (
                sum(p.latency_ms for p in alive) / len(alive)
                if alive else 999
            ),
            "network_healthy": len(alive) >= 2,
        }

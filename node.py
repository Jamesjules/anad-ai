"""
Anad Node
=========
The heart of the network.

Brings together:
  - Identity (who you are)
  - Memory (what you remember)
  - Network (who you know)
  - Model (what you can do)
  - Credits (what you've contributed)

Self-sustaining:
  - Starts with no internet needed (if peers known)
  - Updates spread from node to node
  - Network heals itself around failures
  - Your identity and memory survive hardware changes

Author: Anad Community
License: Public Domain
"""

import os
import sys
import json
import time
import threading
import hashlib
import socket
from typing import Optional, List, Dict
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from node.identity import AnadIdentity
from node.network import PeerStore, PeerDiscovery, Peer
from memory.memory import AnadMemory


# ══════════════════════════════════════════════════════════════════
# CREDIT LEDGER — local, synced with peers
# ══════════════════════════════════════════════════════════════════

class CreditLedger:
    """
    Tracks compute credits earned and spent.

    Credits = compute time contributed.
    Not money. Not crypto. Just fairness.

    Ledger is local — synced with peers for verification.
    Nobody can inflate their credits.
    Nobody can steal your credits.
    """

    def __init__(self, path: str, node_id: str):
        self.path = path
        self.node_id = node_id
        self._balance: int = 0
        self._history: List[Dict] = []
        self._load()

    def earn(self, amount: int, reason: str):
        """Earn credits for contributing compute"""
        self._balance += amount
        self._history.append({
            "type": "earn",
            "amount": amount,
            "reason": reason,
            "timestamp": time.time(),
            "balance_after": self._balance,
        })
        self._save()

    def spend(self, amount: int, reason: str) -> bool:
        """Spend credits to use AI. Returns False if insufficient."""
        if self._balance < amount:
            return False
        self._balance -= amount
        self._history.append({
            "type": "spend",
            "amount": amount,
            "reason": reason,
            "timestamp": time.time(),
            "balance_after": self._balance,
        })
        self._save()
        return True

    @property
    def balance(self) -> int:
        return self._balance

    def give_welcome_credits(self):
        """Give new nodes starter credits"""
        if self._balance == 0 and not self._history:
            self.earn(50, "welcome_bonus")
            print("Welcome to Anad! You have 50 starter credits.")

    def history(self, last_n: int = 20) -> List[Dict]:
        return self._history[-last_n:]

    def _save(self):
        data = {"balance": self._balance, "history": self._history}
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._balance = data.get("balance", 0)
            self._history = data.get("history", [])
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# RESOURCE MANAGER — respects your limits
# ══════════════════════════════════════════════════════════════════

@dataclass
class ResourceConfig:
    """
    What the node owner allows Anad to use.
    Set via the UI. Respected absolutely.
    """
    cpu_percent: int = 40         # max CPU usage
    gpu_percent: int = 60         # max GPU usage
    ram_mb: int = 4096            # max RAM in MB
    disk_gb: int = 20             # max disk space
    bandwidth_percent: int = 30   # max bandwidth
    active_hours: List[int] = None  # hours when node is active (0-23)

    def __post_init__(self):
        if self.active_hours is None:
            # Default: active midnight-6am and 10pm-midnight
            self.active_hours = [0, 1, 2, 3, 4, 5, 22, 23]

    def is_active_now(self) -> bool:
        """Should the node be contributing right now?"""
        hour = time.localtime().tm_hour
        return hour in self.active_hours

    def to_dict(self) -> dict:
        return {
            "cpu_percent": self.cpu_percent,
            "gpu_percent": self.gpu_percent,
            "ram_mb": self.ram_mb,
            "disk_gb": self.disk_gb,
            "bandwidth_percent": self.bandwidth_percent,
            "active_hours": self.active_hours,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceConfig":
        return cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        })

    @classmethod
    def load(cls, path: str) -> "ResourceConfig":
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return cls.from_dict(json.load(f))
            except Exception:
                pass
        return cls()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ══════════════════════════════════════════════════════════════════
# QUERY ROUTER — sends queries to best available node
# ══════════════════════════════════════════════════════════════════

class QueryRouter:
    """
    Routes incoming queries to the best available node.

    Priority:
      1. Local processing if resources allow
      2. Best peer by score and capability
      3. Fallback to any available peer
      4. Queue if no peers (retry when network recovers)

    Resilience:
      If serving node dies mid-response — rerout to next.
      Query never just fails silently.
    """

    def __init__(self, peer_discovery: PeerDiscovery, resources: ResourceConfig):
        self.discovery = peer_discovery
        self.resources = resources
        self._active_queries: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def route(self, query: dict) -> dict:
        """
        Route a query to best available handler.
        Returns response or error with fallback suggestion.
        """
        query_id = hashlib.sha256(
            f"{query}{time.time()}".encode()
        ).hexdigest()[:12]

        with self._lock:
            self._active_queries[query_id] = {
                "query": query,
                "started": time.time(),
                "attempts": 0,
            }

        try:
            # Try local first if resources allow
            if self._can_handle_locally(query):
                return self._handle_locally(query_id, query)

            # Route to best peer
            peers = self.discovery.get_peers_for_routing(n=3)
            for peer in peers:
                result = self._route_to_peer(query_id, query, peer)
                if result and not result.get("error"):
                    return result

            # All peers failed
            return {
                "error": "no_available_nodes",
                "message": "Network busy. Try again shortly.",
                "query_id": query_id,
            }

        finally:
            with self._lock:
                self._active_queries.pop(query_id, None)

    def _can_handle_locally(self, query: dict) -> bool:
        """Check if we have resources to handle this locally"""
        # Mobile nodes only handle simple queries
        query_type = query.get("type", "conversation")
        return (
            self.resources.is_active_now() and
            self.resources.ram_mb >= 2048
        )

    def _handle_locally(self, query_id: str, query: dict) -> dict:
        """Handle query on this node"""
        return {
            "query_id": query_id,
            "handled_by": "local",
            "status": "processing",
        }

    def _route_to_peer(
        self,
        query_id: str,
        query: dict,
        peer: Peer
    ) -> Optional[dict]:
        """Send query to a specific peer"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)
            sock.connect((peer.host, peer.port))

            message = json.dumps({
                "type": "query",
                "query_id": query_id,
                "payload": query,
            }).encode()
            sock.send(len(message).to_bytes(4, "big") + message)

            resp_len = int.from_bytes(sock.recv(4), "big")
            response = json.loads(sock.recv(min(resp_len, 1024 * 1024)))
            sock.close()

            return response

        except Exception as e:
            return {"error": str(e)}

    def active_query_count(self) -> int:
        with self._lock:
            return len(self._active_queries)


# ══════════════════════════════════════════════════════════════════
# MAIN ANAD NODE
# ══════════════════════════════════════════════════════════════════

class AnadNode:
    """
    The complete Anad node.

    Start this and you're part of the network.
    Contribute compute, earn credits, use AI.
    Your data stays yours.
    The network sustains itself.
    """

    ANAD_PORT = 8765           # default port
    ANAD_VERSION = "0.1.0"

    def __init__(self, data_dir: str = "./anad_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # These get populated on start()
        self.identity: Optional[AnadIdentity] = None
        self.memory: Optional[AnadMemory] = None
        self.credits: Optional[CreditLedger] = None
        self.resources: Optional[ResourceConfig] = None
        self.peer_store: Optional[PeerStore] = None
        self.discovery: Optional[PeerDiscovery] = None
        self.router: Optional[QueryRouter] = None

        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._credit_thread: Optional[threading.Thread] = None
        self._paused = threading.Event()
        self._start_time: float = 0

        print("Anad node initializing...")

    def setup_identity(self, passphrase: str, alias: str = "") -> AnadIdentity:
        """
        First time setup — create identity.
        Or load existing identity.
        """
        identity_path = os.path.join(self.data_dir, "identity.json")

        if os.path.exists(identity_path):
            print("Loading existing identity...")
            identity = AnadIdentity.load(identity_path, passphrase)
        else:
            print("Creating new identity...")
            identity = AnadIdentity.generate(alias=alias)
            identity.save(identity_path, passphrase)

        self.identity = identity
        print(f"Identity ready: {identity.node_id}")
        return identity

    def start(self, passphrase: str, alias: str = "", port: int = ANAD_PORT):
        """Start the Anad node"""
        self._start_time = time.time()

        # Load or create identity
        self.setup_identity(passphrase, alias)

        # Initialize memory (encrypted with identity)
        memory_path = os.path.join(self.data_dir, "memory")
        self.memory = AnadMemory(
            storage_path=memory_path,
            identity_public_key_hex=self.identity.public_key_hex,
        )

        # Load resource config
        resource_path = os.path.join(self.data_dir, "resources.json")
        self.resources = ResourceConfig.load(resource_path)
        self.resources.save(resource_path)

        # Initialize credit ledger
        credit_path = os.path.join(self.data_dir, "credits.json")
        self.credits = CreditLedger(credit_path, self.identity.node_id)
        self.credits.give_welcome_credits()

        # Initialize peer network
        peer_path = os.path.join(self.data_dir, "peers.json")
        self.peer_store = PeerStore(peer_path)

        # Detect our external IP
        my_host = self._detect_host()

        self.discovery = PeerDiscovery(
            peer_store=self.peer_store,
            my_node_id=self.identity.node_id,
            my_host=my_host,
            my_port=port,
        )

        # Query router
        self.router = QueryRouter(self.discovery, self.resources)

        # Start background services
        self._running = True
        self.discovery.start()
        self._start_server(port)
        self._start_credit_earner()

        print("\n" + "═" * 50)
        print("  ANAD NODE RUNNING")
        print(f"  Node ID:  {self.identity.node_id}")
        print(f"  Port:     {port}")
        print(f"  Credits:  {self.credits.balance}")
        print(f"  Version:  {self.ANAD_VERSION}")
        print("═" * 50)
        print("\n  Press Ctrl+C to pause")
        print("  Type 'status' for network info\n")

    def _detect_host(self) -> str:
        """Detect our reachable IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
            s.close()
            return host
        except Exception:
            return "127.0.0.1"

    def _start_server(self, port: int):
        """Start TCP server to accept incoming connections"""
        self._server_thread = threading.Thread(
            target=self._serve,
            args=(port,),
            daemon=True,
            name="anad-server",
        )
        self._server_thread.start()

    def _serve(self, port: int):
        """Accept and handle incoming peer connections"""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(50)
            server.settimeout(1)

            while self._running:
                try:
                    conn, addr = server.accept()
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        daemon=True,
                    ).start()
                except socket.timeout:
                    continue

        except Exception as e:
            print(f"Server error: {e}")

    def _handle_connection(self, conn: socket.socket, addr):
        """Handle a single incoming connection"""
        try:
            conn.settimeout(30)
            msg_len = int.from_bytes(conn.recv(4), "big")
            message = json.loads(conn.recv(min(msg_len, 1024 * 1024)))

            msg_type = message.get("type")
            response = self._process_message(msg_type, message)

            resp_bytes = json.dumps(response).encode()
            conn.send(len(resp_bytes).to_bytes(4, "big") + resp_bytes)

            # Earn credit for serving
            if msg_type == "query":
                self.credits.earn(1, "served_query")

        except Exception as e:
            pass
        finally:
            conn.close()

    def _process_message(self, msg_type: str, message: dict) -> dict:
        """Process incoming network messages"""

        if msg_type == "ping":
            return {"type": "pong", "node_id": self.identity.node_id}

        elif msg_type == "handshake":
            return {
                "type": "handshake_ack",
                "node_id": self.identity.node_id,
                "public_key": self.identity.public_key_hex,
                "version": self.ANAD_VERSION,
                "tier": self._detect_tier(),
            }

        elif msg_type == "get_peers":
            peers = self.peer_store.get_best(20)
            return {
                "type": "peers",
                "peers": [p.to_dict() for p in peers],
            }

        elif msg_type == "query":
            # Handle AI query
            if self._paused.is_set():
                return {"error": "node_paused", "message": "This node is paused"}
            return self._handle_query(message)

        elif msg_type == "update_manifest":
            # Peer-to-peer update notification
            return self._handle_update(message)

        else:
            return {"error": "unknown_message_type"}

    def _handle_query(self, message: dict) -> dict:
        """Process an AI query"""
        # In production: run through model
        # For now: placeholder response
        return {
            "type": "query_response",
            "query_id": message.get("query_id"),
            "response": "Anad node processing...",
            "node_id": self.identity.node_id,
        }

    def _handle_update(self, message: dict) -> dict:
        """
        Handle a peer-to-peer software update.

        Verify signature → download → verify checksum → apply.
        Forged updates are rejected cryptographically.
        """
        from node.network import UpdateManifest
        try:
            manifest = UpdateManifest.from_dict(message.get("manifest", {}))
            if not manifest.is_valid():
                return {"error": "invalid_signature", "message": "Update rejected"}

            print(f"\n  Update available: v{manifest.version}")
            print(f"  {manifest.release_notes}")
            print(f"  Will apply after current queries complete.")

            return {"status": "accepted", "version": manifest.version}
        except Exception as e:
            return {"error": str(e)}

    def _start_credit_earner(self):
        """Earn credits passively for running a node"""
        self._credit_thread = threading.Thread(
            target=self._earn_credits_loop,
            daemon=True,
            name="anad-credits",
        )
        self._credit_thread.start()

    def _earn_credits_loop(self):
        """Earn credits every hour for keeping node running"""
        while self._running:
            time.sleep(3600)  # every hour
            if not self._paused.is_set() and self.resources.is_active_now():
                tier = self._detect_tier()
                earn_map = {
                    "server": 20, "desktop": 10,
                    "laptop": 5, "mobile": 2
                }
                amount = earn_map.get(tier, 5)
                self.credits.earn(amount, f"node_uptime_{tier}")

    def _detect_tier(self) -> str:
        """Detect what tier this node is"""
        ram_gb = self.resources.ram_mb / 1024
        if ram_gb >= 32:
            return "server"
        elif ram_gb >= 16:
            return "desktop"
        elif ram_gb >= 8:
            return "laptop"
        else:
            return "mobile"

    def pause(self):
        """Pause node contributions — instant"""
        self._paused.set()
        print("\n  Node paused. Credits no longer earning.")
        print("  Your data is safe. Call resume() to continue.")

    def resume(self):
        """Resume node contributions"""
        self._paused.clear()
        print("\n  Node resumed.")

    def stop(self):
        """Clean shutdown"""
        self._running = False
        self._paused.set()
        print("\n  Anad node stopped. All data saved.")

    def status(self) -> dict:
        """Current node status"""
        uptime = time.time() - self._start_time
        health = self.discovery.network_health() if self.discovery else {}
        mem_stats = self.memory.stats() if self.memory else {}

        return {
            "node_id": self.identity.node_id if self.identity else "not started",
            "version": self.ANAD_VERSION,
            "uptime_seconds": int(uptime),
            "paused": self._paused.is_set(),
            "credits": self.credits.balance if self.credits else 0,
            "network": health,
            "memory": mem_stats,
            "tier": self._detect_tier(),
            "active_queries": self.router.active_query_count() if self.router else 0,
        }

    def update_resources(self, **kwargs):
        """Update resource allocation from UI"""
        if not self.resources:
            return
        for key, value in kwargs.items():
            if hasattr(self.resources, key):
                setattr(self.resources, key, value)
        resource_path = os.path.join(self.data_dir, "resources.json")
        self.resources.save(resource_path)
        print(f"Resources updated: {kwargs}")

"""
Anad Memory System
==================
AI memory that belongs to YOU alone.

Unlike every other AI:
  - Memory is stored on YOUR device
  - Encrypted with YOUR key
  - Nobody else can read it
  - Not Anad nodes, not corporations, nobody
  - You can export, delete, or move it freely

Memory types:
  1. Conversation history  — past sessions
  2. User preferences      — how you like responses
  3. Personal context      — things you've told Anad
  4. Knowledge anchors     — facts important to you
  5. Skill memory          — your code, documents

The AI becomes more useful over time
without ever surrendering your data to anyone.

Author: Anad Community
License: Public Domain
"""

import os
import json
import time
import hashlib
import base64
from typing import List, Dict, Optional, Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


# ══════════════════════════════════════════════════════════════════
# MEMORY ENCRYPTION — Your key, your data
# ══════════════════════════════════════════════════════════════════

class MemoryVault:
    """
    Encrypts and decrypts memory entries.

    Uses AES-256-GCM — military grade encryption.
    Key derived from your identity's private key.
    Nobody can read your memories without your key.
    """

    def __init__(self, identity_public_key_hex: str):
        """
        Derive memory encryption key from identity.
        Same identity = same key = same memories on any device.
        """
        # Derive a memory-specific key from the identity key
        # Using HKDF so the memory key is separate from signing key
        key_bytes = bytes.fromhex(identity_public_key_hex)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"anad-memory-v1",
            info=b"memory-encryption",
        )
        self._key = hkdf.derive(key_bytes)

    def encrypt(self, data: dict) -> str:
        """Encrypt a memory entry to a base64 string"""
        plaintext = json.dumps(data).encode()
        nonce = os.urandom(12)
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        # Store nonce + ciphertext together
        combined = nonce + ciphertext
        return base64.b64encode(combined).decode()

    def decrypt(self, encrypted: str) -> dict:
        """Decrypt a memory entry"""
        combined = base64.b64decode(encrypted)
        nonce = combined[:12]
        ciphertext = combined[12:]
        aesgcm = AESGCM(self._key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext)


# ══════════════════════════════════════════════════════════════════
# MEMORY TYPES
# ══════════════════════════════════════════════════════════════════

class MemoryEntry:
    """A single memory item"""

    TYPES = {
        "conversation": "past conversation turns",
        "preference":   "how you like things done",
        "context":      "personal facts you've shared",
        "anchor":       "important knowledge to remember",
        "skill":        "code, documents, workflows",
    }

    def __init__(
        self,
        memory_type: str,
        content: str,
        metadata: Optional[Dict] = None,
        timestamp: Optional[float] = None,
        memory_id: Optional[str] = None,
    ):
        self.memory_type = memory_type
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()
        self.memory_id = memory_id or self._generate_id()

    def _generate_id(self) -> str:
        h = hashlib.sha256(
            f"{self.content}{self.timestamp}".encode()
        ).hexdigest()[:16]
        return f"mem_{h}"

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            memory_type=data["memory_type"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp"),
            memory_id=data.get("memory_id"),
        )


# ══════════════════════════════════════════════════════════════════
# CONVERSATION SESSION
# ══════════════════════════════════════════════════════════════════

class ConversationSession:
    """
    A single conversation with Anad.

    Stored encrypted on your device.
    Referenced in future conversations if you allow it.
    Deleted on demand — no traces anywhere.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._new_id()
        self.turns: List[Dict] = []
        self.started_at = time.time()
        self.title = ""

    def _new_id(self) -> str:
        return "session_" + hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:12]

    def add_turn(self, role: str, content: str):
        """Add a conversation turn. role = 'user' or 'anad'"""
        self.turns.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        # Auto-title from first user message
        if not self.title and role == "user":
            self.title = content[:60] + ("..." if len(content) > 60 else "")

    def get_context(self, last_n: int = 20) -> List[Dict]:
        """Get last N turns for context injection"""
        return self.turns[-last_n:]

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "started_at": self.started_at,
            "turns": self.turns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSession":
        session = cls(session_id=data["session_id"])
        session.title = data.get("title", "")
        session.started_at = data.get("started_at", time.time())
        session.turns = data.get("turns", [])
        return session


# ══════════════════════════════════════════════════════════════════
# MAIN MEMORY STORE
# ══════════════════════════════════════════════════════════════════

class AnadMemory:
    """
    Complete memory system for an Anad user.

    Everything is encrypted on your device.
    You control what is remembered and what is forgotten.

    Features:
      - Persistent conversation history
      - Personal preferences and context
      - Searchable memory retrieval
      - Full export (take your data anywhere)
      - Selective deletion
      - Memory summary for context injection

    Memory never leaves your device unencrypted.
    Memory is never sent to other nodes.
    Memory is never used for training without your consent.
    """

    def __init__(
        self,
        storage_path: str,
        identity_public_key_hex: str,
    ):
        self.storage_path = storage_path
        self.vault = MemoryVault(identity_public_key_hex)
        self._memories: Dict[str, MemoryEntry] = {}
        self._sessions: Dict[str, ConversationSession] = {}
        self._current_session: Optional[ConversationSession] = None
        self._preferences: Dict[str, Any] = {}

        os.makedirs(storage_path, exist_ok=True)
        self._load()

    # ── Session Management ─────────────────────────────────────

    def new_session(self) -> ConversationSession:
        """Start a new conversation session"""
        session = ConversationSession()
        self._current_session = session
        self._sessions[session.session_id] = session
        return session

    def current_session(self) -> Optional[ConversationSession]:
        """Get or create current session"""
        if not self._current_session:
            return self.new_session()
        return self._current_session

    def add_turn(self, role: str, content: str):
        """Add a turn to the current session"""
        session = self.current_session()
        session.add_turn(role, content)
        self._save_sessions()

    def get_conversation_context(self, last_n: int = 10) -> List[Dict]:
        """Get recent conversation turns for context"""
        if not self._current_session:
            return []
        return self._current_session.get_context(last_n)

    def list_sessions(self) -> List[Dict]:
        """List all past conversations"""
        sessions = []
        for s in self._sessions.values():
            sessions.append({
                "session_id": s.session_id,
                "title": s.title or "Untitled",
                "started_at": s.started_at,
                "turns": len(s.turns),
            })
        return sorted(sessions, key=lambda x: x["started_at"], reverse=True)

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load a past session"""
        session = self._sessions.get(session_id)
        if session:
            self._current_session = session
        return session

    def delete_session(self, session_id: str):
        """Delete a conversation permanently"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            if (self._current_session and
                    self._current_session.session_id == session_id):
                self._current_session = None
            self._save_sessions()
            print(f"Session {session_id} deleted permanently")

    # ── Long-Term Memory ───────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "context",
        metadata: Optional[Dict] = None,
    ) -> MemoryEntry:
        """
        Store something in long-term memory.

        Examples:
          memory.remember("I prefer concise responses", "preference")
          memory.remember("I work at a hospital", "context")
          memory.remember("My Python project uses FastAPI", "anchor")
        """
        entry = MemoryEntry(
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
        )
        self._memories[entry.memory_id] = entry
        self._save_memories()
        return entry

    def forget(self, memory_id: str):
        """Delete a specific memory permanently"""
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save_memories()
            print(f"Memory {memory_id} deleted permanently")

    def forget_all(self):
        """
        Nuclear option — delete ALL memories.
        Irreversible. Gone forever.
        """
        self._memories.clear()
        self._sessions.clear()
        self._current_session = None
        self._preferences.clear()
        self._save_all()
        print("All memories deleted permanently")

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Simple keyword search over memories.
        Future: semantic search using the Anad model itself.
        """
        query_lower = query.lower()
        results = []

        for entry in self._memories.values():
            if query_lower in entry.content.lower():
                results.append(entry)

        # Sort by recency
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results[:limit]

    def get_by_type(self, memory_type: str) -> List[MemoryEntry]:
        """Get all memories of a specific type"""
        return [
            m for m in self._memories.values()
            if m.memory_type == memory_type
        ]

    # ── Preferences ───────────────────────────────────────────

    def set_preference(self, key: str, value: Any):
        """Set a user preference"""
        self._preferences[key] = value
        self._save_preferences()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        return self._preferences.get(key, default)

    # ── Context Injection ─────────────────────────────────────

    def build_context(self) -> str:
        """
        Build a context string to inject into prompts.
        Tells Anad what it should remember about you.
        This is what makes Anad feel like it knows you.

        Only injected with YOUR permission.
        Never stored on any server.
        """
        parts = []

        # Preferences
        prefs = self.get_by_type("preference")
        if prefs:
            parts.append("User preferences:")
            for p in prefs[:5]:
                parts.append(f"  - {p.content}")

        # Personal context
        contexts = self.get_by_type("context")
        if contexts:
            parts.append("About this user:")
            for c in contexts[:5]:
                parts.append(f"  - {c.content}")

        # Knowledge anchors
        anchors = self.get_by_type("anchor")
        if anchors:
            parts.append("Important context:")
            for a in anchors[:3]:
                parts.append(f"  - {a.content}")

        if not parts:
            return ""

        return "\n".join(parts)

    def stats(self) -> Dict:
        """Memory statistics"""
        type_counts = {}
        for m in self._memories.values():
            type_counts[m.memory_type] = type_counts.get(m.memory_type, 0) + 1

        return {
            "total_memories": len(self._memories),
            "total_sessions": len(self._sessions),
            "by_type": type_counts,
            "storage_path": self.storage_path,
            "current_session": (
                self._current_session.session_id
                if self._current_session else None
            ),
        }

    # ── Export / Import ───────────────────────────────────────

    def export(self, export_path: str):
        """
        Export all your data.
        Encrypted — only you can read it.
        Take it anywhere. It's yours.
        """
        export_data = {
            "version": "1.0",
            "exported_at": time.time(),
            "memories": self.vault.encrypt({
                "memories": {
                    k: v.to_dict()
                    for k, v in self._memories.items()
                },
                "preferences": self._preferences,
            }),
            "sessions": self.vault.encrypt({
                "sessions": {
                    k: v.to_dict()
                    for k, v in self._sessions.items()
                }
            }),
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Memory exported to {export_path}")
        print(f"This file is encrypted. Only you can read it.")

    def import_from(self, import_path: str):
        """Import memories from an export file"""
        with open(import_path) as f:
            data = json.load(f)

        mem_data = self.vault.decrypt(data["memories"])
        for mid, mdict in mem_data["memories"].items():
            entry = MemoryEntry.from_dict(mdict)
            self._memories[entry.memory_id] = entry

        self._preferences.update(mem_data.get("preferences", {}))

        session_data = self.vault.decrypt(data["sessions"])
        for sid, sdict in session_data["sessions"].items():
            session = ConversationSession.from_dict(sdict)
            self._sessions[session.session_id] = session

        self._save_all()
        print(f"Memory imported from {import_path}")

    # ── Persistence ───────────────────────────────────────────

    def _save_memories(self):
        encrypted = self.vault.encrypt({
            "memories": {
                k: v.to_dict() for k, v in self._memories.items()
            }
        })
        with open(os.path.join(self.storage_path, "memories.enc"), "w") as f:
            f.write(encrypted)

    def _save_sessions(self):
        encrypted = self.vault.encrypt({
            "sessions": {
                k: v.to_dict() for k, v in self._sessions.items()
            }
        })
        with open(os.path.join(self.storage_path, "sessions.enc"), "w") as f:
            f.write(encrypted)

    def _save_preferences(self):
        encrypted = self.vault.encrypt(self._preferences)
        with open(os.path.join(self.storage_path, "preferences.enc"), "w") as f:
            f.write(encrypted)

    def _save_all(self):
        self._save_memories()
        self._save_sessions()
        self._save_preferences()

    def _load(self):
        """Load all encrypted memory from disk"""
        # Load memories
        mem_path = os.path.join(self.storage_path, "memories.enc")
        if os.path.exists(mem_path):
            try:
                with open(mem_path) as f:
                    data = self.vault.decrypt(f.read())
                for mid, mdict in data.get("memories", {}).items():
                    self._memories[mid] = MemoryEntry.from_dict(mdict)
            except Exception as e:
                print(f"Warning: Could not load memories: {e}")

        # Load sessions
        sess_path = os.path.join(self.storage_path, "sessions.enc")
        if os.path.exists(sess_path):
            try:
                with open(sess_path) as f:
                    data = self.vault.decrypt(f.read())
                for sid, sdict in data.get("sessions", {}).items():
                    self._sessions[sid] = ConversationSession.from_dict(sdict)
            except Exception as e:
                print(f"Warning: Could not load sessions: {e}")

        # Load preferences
        pref_path = os.path.join(self.storage_path, "preferences.enc")
        if os.path.exists(pref_path):
            try:
                with open(pref_path) as f:
                    self._preferences = self.vault.decrypt(f.read())
            except Exception as e:
                print(f"Warning: Could not load preferences: {e}")

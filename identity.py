"""
Anad Identity System
====================
Your identity on the Anad network.

NOT tied to:
  - Your email
  - Your IP address
  - Your hardware
  - Any corporation

TIED TO:
  - Your cryptographic keypair
  - A file YOU control
  - Moves with you across any hardware

Philosophy:
  You are your key. Nothing else.
  No one can revoke your identity.
  No one can impersonate you without your private key.
  Lose your key = lose your identity. Guard it.

Author: Anad Community
License: Public Domain
"""

import os
import json
import hashlib
import base64
import time
from typing import Optional, Tuple
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


# ══════════════════════════════════════════════════════════════════
# ANAD IDENTITY
# ══════════════════════════════════════════════════════════════════

class AnadIdentity:
    """
    Your permanent identity on the Anad network.

    Based on Ed25519 keypair — same cryptography used by
    Signal, Tor, and modern SSH.

    Your node_id is derived from your public key.
    Same public key = same node_id = same identity.
    Any hardware. Any location. Forever.

    Security model:
      Private key is encrypted with your passphrase
      Even if someone steals your identity file,
      they cannot use it without your passphrase.
      Your passphrase never leaves your device.
    """

    def __init__(
        self,
        private_key: Ed25519PrivateKey,
        node_id: str,
        created_at: float,
        alias: str = "",
    ):
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self.node_id = node_id
        self.created_at = created_at
        self.alias = alias  # human readable name, optional

    @classmethod
    def generate(cls, alias: str = "") -> "AnadIdentity":
        """
        Generate a brand new identity.
        Called once — when someone first joins Anad.
        """
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Node ID = first 32 chars of sha256(public_key)
        pub_bytes = public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw
        )
        node_id = "anad1_" + hashlib.sha256(pub_bytes).hexdigest()[:26]

        print(f"New Anad identity generated")
        print(f"Node ID: {node_id}")
        print(f"This is your permanent identity. Keep your key file safe.")

        return cls(
            private_key=private_key,
            node_id=node_id,
            created_at=time.time(),
            alias=alias,
        )

    @property
    def public_key_hex(self) -> str:
        """Your public identity — safe to share with anyone"""
        pub_bytes = self._public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw
        )
        return pub_bytes.hex()

    def sign(self, data: bytes) -> bytes:
        """
        Sign data with your private key.
        Used to prove messages come from you.
        """
        return self._private_key.sign(data)

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify a signature from this identity"""
        try:
            self._public_key.verify(signature, data)
            return True
        except Exception:
            return False

    @staticmethod
    def verify_from_public_key(
        public_key_hex: str,
        data: bytes,
        signature: bytes
    ) -> bool:
        """Verify a signature given just a public key hex string"""
        try:
            pub_bytes = bytes.fromhex(public_key_hex)
            pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
            pub_key.verify(signature, data)
            return True
        except Exception:
            return False

    def save(self, path: str, passphrase: str):
        """
        Save identity to disk, encrypted with passphrase.

        Even if someone steals this file,
        they cannot use your identity without your passphrase.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Derive encryption key from passphrase using scrypt
        salt = os.urandom(32)
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
        key = kdf.derive(passphrase.encode())

        # Encrypt private key
        private_bytes = self._private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption()
        )

        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        encrypted = aesgcm.encrypt(nonce, private_bytes, None)

        identity_data = {
            "version": "1.0",
            "node_id": self.node_id,
            "public_key": self.public_key_hex,
            "alias": self.alias,
            "created_at": self.created_at,
            "encrypted_private_key": base64.b64encode(encrypted).decode(),
            "salt": base64.b64encode(salt).decode(),
            "nonce": base64.b64encode(nonce).decode(),
        }

        with open(path, "w") as f:
            json.dump(identity_data, f, indent=2)

        print(f"Identity saved to {path}")
        print(f"Keep this file and your passphrase safe.")

    @classmethod
    def load(cls, path: str, passphrase: str) -> "AnadIdentity":
        """Load identity from disk using passphrase"""
        with open(path) as f:
            data = json.load(f)

        # Derive key from passphrase
        salt = base64.b64decode(data["salt"])
        kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
        key = kdf.derive(passphrase.encode())

        # Decrypt private key
        nonce = base64.b64decode(data["nonce"])
        encrypted = base64.b64decode(data["encrypted_private_key"])

        try:
            aesgcm = AESGCM(key)
            private_bytes = aesgcm.decrypt(nonce, encrypted, None)
        except Exception:
            raise ValueError("Wrong passphrase or corrupted identity file")

        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)

        return cls(
            private_key=private_key,
            node_id=data["node_id"],
            created_at=data["created_at"],
            alias=data.get("alias", ""),
        )

    def to_public_info(self) -> dict:
        """
        Public information safe to share with peers.
        Never includes private key.
        """
        return {
            "node_id": self.node_id,
            "public_key": self.public_key_hex,
            "alias": self.alias,
        }

    def __repr__(self):
        alias_str = f" ({self.alias})" if self.alias else ""
        return f"AnadIdentity({self.node_id}{alias_str})"

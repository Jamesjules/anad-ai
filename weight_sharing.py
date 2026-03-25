"""
Anad Weight Sharing Protocol
==============================
Trained weights distributed peer-to-peer.
No node repeats training another already did.

How it works:
  1. Genesis node trains on data → produces weights
  2. Weights signed with genesis identity
  3. Peers request weights → verify signature → apply
  4. Each node trains only on NEW data not in index
  5. Incremental updates merged back to master
  6. Network collectively improves the model

Key guarantee:
  Weights are cryptographically signed.
  Forged or tampered weights are rejected.
  You always know exactly which node produced them.

Author: Anad Community
License: Public Domain
"""

import os
import json
import hashlib
import time
import zipfile
import tempfile
from typing import Optional, List
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════
# WEIGHT MANIFEST — describes a set of weights
# ══════════════════════════════════════════════════════════════════

@dataclass
class WeightManifest:
    """
    Describes a trained model checkpoint.
    Signed by the node that produced it.
    """
    version: str              # e.g. "0.1.0"
    model_name: str           # e.g. "anad-nano"
    step: int                 # training step
    loss: float               # training loss at this step
    data_checksums: List[str] # what data was trained on
    producer_node_id: str     # who trained this
    producer_public_key: str  # their public key
    signature: str            # signed by producer
    file_checksum: str        # sha256 of weight file
    file_size_bytes: int
    timestamp: float = 0.0
    parent_version: str = ""  # what weights this was built on

    def verify(self) -> bool:
        """Verify this manifest was signed by claimed producer"""
        try:
            from node.identity import AnadIdentity
            data = (
                f"{self.version}{self.step}"
                f"{self.file_checksum}{self.timestamp}"
            ).encode()
            sig = bytes.fromhex(self.signature)
            return AnadIdentity.verify_from_public_key(
                self.producer_public_key, data, sig
            )
        except Exception:
            return False

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "WeightManifest":
        return cls(**{k: v for k, v in d.items()
                     if k in cls.__dataclass_fields__})

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WeightManifest":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ══════════════════════════════════════════════════════════════════
# WEIGHT PACKAGE — weights + manifest bundled together
# ══════════════════════════════════════════════════════════════════

class WeightPackage:
    """
    Bundles model weights + manifest into a single shareable file.
    Nodes request this file from peers.
    Verified before applying.
    """

    @staticmethod
    def create(
        model_dir: str,
        manifest: WeightManifest,
        output_path: str,
    ):
        """
        Pack model weights + manifest into a zip.
        This is what gets distributed to peers.
        """
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            zf.writestr(
                "manifest.json",
                json.dumps(manifest.to_dict(), indent=2)
            )

            # Add all weight files
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith((".npy", ".json")):
                        full_path = os.path.join(root, file)
                        arc_name = os.path.relpath(full_path, model_dir)
                        zf.write(full_path, arc_name)

        # Compute final checksum
        checksum = WeightPackage._checksum(output_path)
        print(f"  Weight package created: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        print(f"  Checksum: {checksum[:16]}...")
        return checksum

    @staticmethod
    def verify_and_extract(
        package_path: str,
        extract_dir: str,
    ) -> Optional[WeightManifest]:
        """
        Verify a weight package then extract it.
        Returns manifest if valid, None if tampered.
        """
        # Verify file checksum
        actual_checksum = WeightPackage._checksum(package_path)

        with zipfile.ZipFile(package_path, "r") as zf:
            # Read manifest
            try:
                manifest_data = json.loads(zf.read("manifest.json"))
                manifest = WeightManifest.from_dict(manifest_data)
            except Exception as e:
                print(f"  Invalid package: bad manifest ({e})")
                return None

            # Verify checksum matches
            if manifest.file_checksum != actual_checksum:
                print(f"  Rejected: checksum mismatch (file may be corrupted)")
                return None

            # Verify signature
            if not manifest.verify():
                print(f"  Rejected: invalid signature (possible tampering)")
                return None

            # Extract weights
            os.makedirs(extract_dir, exist_ok=True)
            for item in zf.namelist():
                if item != "manifest.json":
                    zf.extract(item, extract_dir)

        print(f"  Weights verified and extracted to {extract_dir}")
        print(f"  Producer: {manifest.producer_node_id[:24]}...")
        print(f"  Version: {manifest.version} step {manifest.step}")
        return manifest

    @staticmethod
    def _checksum(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()


# ══════════════════════════════════════════════════════════════════
# WEIGHT STORE — manages local weight versions
# ══════════════════════════════════════════════════════════════════

class WeightStore:
    """
    Manages weight versions on this node.

    Keeps track of:
      - Which version is currently active
      - What versions are available
      - What data each version was trained on
      - Which peers have which versions
    """

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._index_path = os.path.join(store_dir, "weight_index.json")
        self._index: dict = self._load_index()

    def save_version(
        self,
        model_dir: str,
        manifest: WeightManifest,
    ) -> str:
        """Save a new weight version"""
        version_dir = os.path.join(
            self.store_dir,
            f"v{manifest.version}_step{manifest.step}"
        )
        os.makedirs(version_dir, exist_ok=True)

        # Copy model files
        import shutil
        for item in os.listdir(model_dir):
            src = os.path.join(model_dir, item)
            dst = os.path.join(version_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        # Save manifest
        manifest.save(os.path.join(version_dir, "manifest.json"))

        # Create shareable package
        package_path = os.path.join(
            self.store_dir,
            f"anad_{manifest.version}_step{manifest.step}.zip"
        )
        checksum = WeightPackage.create(version_dir, manifest, package_path)

        # Update index
        self._index[f"v{manifest.version}_step{manifest.step}"] = {
            "version": manifest.version,
            "step": manifest.step,
            "loss": manifest.loss,
            "package": package_path,
            "checksum": checksum,
            "timestamp": manifest.timestamp,
            "data_count": len(manifest.data_checksums),
        }
        self._index["latest"] = f"v{manifest.version}_step{manifest.step}"
        self._save_index()

        return package_path

    def get_latest_manifest(self) -> Optional[dict]:
        """Get info about the latest version"""
        latest_key = self._index.get("latest")
        if latest_key:
            return self._index.get(latest_key)
        return None

    def get_package_path(self, version_key: str) -> Optional[str]:
        """Get path to a specific version's package"""
        info = self._index.get(version_key)
        if info:
            return info.get("package")
        return None

    def list_versions(self) -> List[dict]:
        """List all available versions"""
        return [
            v for k, v in self._index.items()
            if k != "latest" and isinstance(v, dict)
        ]

    def _load_index(self) -> dict:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_index(self):
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)


# ══════════════════════════════════════════════════════════════════
# FEDERATED TRAINING COORDINATOR
# ══════════════════════════════════════════════════════════════════

class FederatedCoordinator:
    """
    Coordinates distributed training across nodes.

    Rules:
      1. Each node only trains on data it hasn't seen
      2. Trained weights shared with all peers
      3. New nodes download latest weights — no retraining
      4. Incremental updates merged periodically
      5. All contributions tracked and credited

    This is how the network trains collectively
    without duplicating work.
    """

    def __init__(
        self,
        weight_store: WeightStore,
        data_index_path: str,
        node_identity,
    ):
        self.weight_store = weight_store
        self.data_index_path = data_index_path
        self.identity = node_identity

    def sign_manifest(
        self,
        manifest: WeightManifest,
    ) -> WeightManifest:
        """Sign a weight manifest with this node's identity"""
        data = (
            f"{manifest.version}{manifest.step}"
            f"{manifest.file_checksum}{manifest.timestamp}"
        ).encode()
        signature = self.identity.sign(data)
        manifest.signature = signature.hex()
        manifest.producer_node_id = self.identity.node_id
        manifest.producer_public_key = self.identity.public_key_hex
        return manifest

    def prepare_weights_for_sharing(
        self,
        model_dir: str,
        version: str,
        step: int,
        loss: float,
        data_checksums: List[str],
    ) -> str:
        """
        After training, prepare weights for distribution.
        Signs, packages, and stores for peer requests.
        """
        # Compute file checksum before signing
        temp_package = tempfile.mktemp(suffix=".zip")
        manifest = WeightManifest(
            version=version,
            model_name="anad-nano",
            step=step,
            loss=loss,
            data_checksums=data_checksums,
            producer_node_id=self.identity.node_id,
            producer_public_key=self.identity.public_key_hex,
            signature="",
            file_checksum="",
            file_size_bytes=0,
            timestamp=time.time(),
        )

        # Create temp package to get checksum
        WeightPackage.create(model_dir, manifest, temp_package)
        manifest.file_checksum = WeightPackage._checksum(temp_package)
        manifest.file_size_bytes = os.path.getsize(temp_package)
        os.remove(temp_package)

        # Sign the manifest
        manifest = self.sign_manifest(manifest)

        # Save final package
        package_path = self.weight_store.save_version(model_dir, manifest)

        print(f"\n  Weights ready for sharing")
        print(f"  Version: {version} step {step}")
        print(f"  Loss: {loss:.4f}")
        print(f"  Data trained: {len(data_checksums)} records")
        print(f"  Package: {package_path}")

        return package_path

    def should_download_from_peer(
        self,
        peer_manifest: dict,
        my_step: int,
    ) -> bool:
        """
        Should we download weights from a peer?
        Yes if peer is further along in training.
        """
        peer_step = peer_manifest.get("step", 0)
        peer_loss = peer_manifest.get("loss", 999)
        return peer_step > my_step

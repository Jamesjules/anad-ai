"""
Anad Genesis Node Setup
========================
Makes your node the official first node of the network.

What this does:
  1. Records your node as genesis in the network config
  2. Updates bootstrap list with your node ID
  3. Publishes your public key as the signing authority
  4. All future nodes will connect to you first
  5. All weight updates you sign become trusted network-wide

Your node ID: anad1_4ccd35bbd635c4a03678cf44f1

Run once:
    python genesis.py

Author: Anad Community
License: Public Domain
"""

import os
import sys
import json
import time
import socket
import getpass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from node.identity import AnadIdentity


GENESIS_CONFIG_PATH = "./genesis.json"
NETWORK_CONFIG_PATH = "./node/network.py"


def get_public_ip() -> str:
    """Detect public IP for bootstrap configuration"""
    # Try to get external IP
    try:
        import urllib.request
        with urllib.request.urlopen("https://api.ipify.org", timeout=5) as r:
            return r.read().decode().strip()
    except Exception:
        pass
    # Fall back to local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def setup_genesis():
    print("\n" + "═" * 55)
    print("  ANAD GENESIS NODE SETUP")
    print("  Making your node the first node of the network")
    print("═" * 55 + "\n")

    # Load identity
    identity_path = "./anad_data/identity.json"
    if not os.path.exists(identity_path):
        print("  No identity found. Run python main.py first.")
        return

    passphrase = getpass.getpass("  Passphrase: ")
    try:
        identity = AnadIdentity.load(identity_path, passphrase)
    except ValueError:
        print("  Wrong passphrase.")
        return

    print(f"\n  Node ID:    {identity.node_id}")
    print(f"  Public key: {identity.public_key_hex[:32]}...")

    # Detect IP
    print("\n  Detecting your IP address...")
    public_ip = get_public_ip()
    print(f"  Detected:   {public_ip}")
    print()
    custom_ip = input(f"  Use this IP? (Enter to confirm, or type different IP): ").strip()
    if custom_ip:
        public_ip = custom_ip

    port = input("  Port (default 8765): ").strip() or "8765"

    # Sign genesis record
    genesis_data = {
        "node_id":    identity.node_id,
        "public_key": identity.public_key_hex,
        "host":       public_ip,
        "port":       int(port),
        "timestamp":  time.time(),
        "version":    "0.1.0",
        "role":       "genesis",
        "message":    "Anad genesis node — public AI for humanity",
    }

    # Sign with private key
    sign_bytes = json.dumps({
        k: v for k, v in genesis_data.items()
        if k != "signature"
    }, sort_keys=True).encode()
    signature = identity.sign(sign_bytes)
    genesis_data["signature"] = signature.hex()

    # Save genesis config
    with open(GENESIS_CONFIG_PATH, "w") as f:
        json.dump(genesis_data, f, indent=2)
    print(f"\n  Genesis config saved to {GENESIS_CONFIG_PATH}")

    # Update network.py bootstrap list
    bootstrap_entry = (
        f'        {{"host": "{public_ip}", '
        f'"port": {port}, '
        f'"node_id": "{identity.node_id}"}}'
    )

    network_path = "./node/network.py"
    with open(network_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace bootstrap nodes section
    old = (
        '    BOOTSTRAP_NODES = [\n'
        '        # Example — replace with your actual IP when ready:\n'
        '        # {"host": "YOUR_PUBLIC_IP", "port": 8765, "node_id": "anad1_4ccd35bbd635c4a03678cf44f1"}\n'
        '    ]'
    )
    new = (
        f'    BOOTSTRAP_NODES = [\n'
        f'        # Genesis node — anad1_4ccd35bbd635c4a03678cf44f1\n'
        f'{bootstrap_entry}\n'
        f'    ]'
    )

    if old in content:
        content = content.replace(old, new)
        with open(network_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Bootstrap list updated in node/network.py")
    else:
        print(f"  Note: Update BOOTSTRAP_NODES in node/network.py manually:")
        print(f"  {bootstrap_entry}")

    # Print summary
    print("\n" + "═" * 55)
    print("  GENESIS NODE CONFIGURED")
    print("═" * 55)
    print(f"""
  Your node is now the genesis node of Anad.

  Node ID:   {identity.node_id}
  Host:      {public_ip}:{port}
  Role:      Genesis (trusted signing authority)

  What this means:
    ✓ All new nodes will connect to you first
    ✓ Weight updates you sign are trusted network-wide
    ✓ Your public key is the root of trust for Anad

  Next steps:
    1. Push to GitHub so new nodes find your bootstrap:
       git add .
       git commit -m "feat: genesis node configuration"
       git push

    2. Share your genesis.json publicly so others
       can verify your identity is legitimate

    3. Make sure port {port} is open on your router
       (port forwarding) for other nodes to reach you

    4. Run python main.py to keep your genesis node online
""")


if __name__ == "__main__":
    setup_genesis()

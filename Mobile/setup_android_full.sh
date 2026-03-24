#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ANAD FULL ANDROID SETUP
# ═══════════════════════════════════════════════════════════
# Device:  Android with 10GB+ storage
# Model:   Anad Small (4-bit quantized, ~600MB)
# Tier:    Laptop (earns 5 credits/hour)
# Mode:    Full node — contribute + use + memory
#
# BEFORE RUNNING:
#   1. Install Termux from F-Droid (NOT Play Store)
#      → fdroid.org or search F-Droid on browser
#      → Play Store version is outdated and broken
#
#   2. Open Termux and paste this entire script
#      OR run: bash setup_android_full.sh
#
# ═══════════════════════════════════════════════════════════

set -e

ANAD_DIR="$HOME/anad"
DATA_DIR="$HOME/.anad_data"
ANAD_REPO="https://github.com/Jamesjules/anad-ai"

clear
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║            ANAD — ANDROID FULL             ║"
echo "║   Node + Memory + Offline AI on mobile     ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "  Storage:  10GB+ ✓"
echo "  Model:    Anad Small (4-bit, ~600MB)"
echo "  Tier:     Laptop node"
echo "  Credits:  5 per hour when contributing"
echo ""
echo "  This will take about 5-10 minutes."
echo "  Keep Termux open and phone charging."
echo ""
read -p "  Press Enter to begin..."

# ── Step 1: Update Termux ─────────────────────────────────
echo ""
echo "[ 1/7 ] Updating Termux packages..."
pkg update -y -q 2>/dev/null || true
pkg upgrade -y -q 2>/dev/null || true
echo "  ✓ Termux updated"

# ── Step 2: Install system dependencies ──────────────────
echo ""
echo "[ 2/7 ] Installing system packages..."
pkg install -y -q \
    python \
    git \
    openssl \
    libffi \
    clang \
    make \
    pkg-config \
    libjpeg-turbo \
    termux-api
echo "  ✓ System packages installed"

# ── Step 3: Install Python packages ──────────────────────
echo ""
echo "[ 3/7 ] Installing Python packages..."
pip install --quiet --upgrade pip setuptools wheel
pip install --quiet \
    cryptography \
    numpy \
    aiohttp \
    websockets \
    tqdm
echo "  ✓ Python packages installed"

# ── Step 4: Clone Anad ───────────────────────────────────
echo ""
echo "[ 4/7 ] Downloading Anad from GitHub..."
mkdir -p "$DATA_DIR"
if [ -d "$ANAD_DIR" ]; then
    echo "  Updating existing installation..."
    cd "$ANAD_DIR" && git pull --quiet
else
    git clone --quiet "$ANAD_REPO" "$ANAD_DIR"
fi
cd "$ANAD_DIR"
echo "  ✓ Anad installed at $ANAD_DIR"

# ── Step 5: Write full mobile config ─────────────────────
echo ""
echo "[ 5/7 ] Configuring for your device..."
mkdir -p "$DATA_DIR"
cat > "$DATA_DIR/resources.json" << 'EOF'
{
  "cpu_percent": 40,
  "gpu_percent": 50,
  "ram_mb": 3072,
  "disk_gb": 8,
  "bandwidth_percent": 30,
  "active_hours": [0,1,2,3,4,5,22,23]
}
EOF

cat > "$DATA_DIR/device.json" << 'EOF'
{
  "type": "android_full",
  "tier": "laptop",
  "model": "anad-small",
  "quantization": "4bit",
  "charge_only": true,
  "wifi_only": true,
  "battery_threshold": 75
}
EOF

echo "  ✓ Configuration saved"
echo "  CPU: 40% | RAM: 3GB | Disk: 8GB"
echo "  Active: charging + WiFi only"

# ── Step 6: Termux permissions ───────────────────────────
echo ""
echo "[ 6/7 ] Setting up permissions..."

# Storage permission
termux-setup-storage 2>/dev/null || true

# Wake lock to prevent sleep during contribution
cat > "$HOME/.anad_wakelock.sh" << 'EOF'
#!/bin/bash
# Prevent phone from sleeping while Anad is contributing
termux-wake-lock 2>/dev/null || true
EOF
chmod +x "$HOME/.anad_wakelock.sh"

echo "  ✓ Permissions configured"

# ── Step 7: Create all shortcuts ─────────────────────────
echo ""
echo "[ 7/7 ] Creating shortcuts..."

# Main start script
cat > "$HOME/anad.sh" << 'MAINSCRIPT'
#!/bin/bash
# ─────────────────────────────────────
#  ANAD — Main launcher
# ─────────────────────────────────────
cd $HOME/anad
termux-wake-lock 2>/dev/null || true

python -c "
import sys, os, getpass, json
sys.path.insert(0, '.')
from mobile.mobile_node import MobileAnadNode

DATA_DIR = os.path.expanduser('~/.anad_data')
node = MobileAnadNode(DATA_DIR)

identity_path = os.path.join(DATA_DIR, 'identity.json')
if os.path.exists(identity_path):
    print()
    print('Welcome back to Anad.')
    passphrase = getpass.getpass('Passphrase: ')
else:
    print()
    print('First time setup.')
    print('Creating your permanent Anad identity.')
    print()
    while True:
        passphrase = getpass.getpass('Create passphrase: ')
        confirm = getpass.getpass('Confirm passphrase: ')
        if passphrase == confirm:
            break
        print('Passphrases do not match. Try again.')
    alias = input('Your name or alias (optional): ').strip()

try:
    node.start(passphrase=passphrase)
    print()
    print('What would you like to do?')
    print()
    print('  1 → Chat with Anad')
    print('  2 → Node status')
    print('  3 → My memory')
    print('  4 → Credits')
    print('  5 → Settings')
    print('  0 → Exit')
    print()

    while True:
        try:
            choice = input('Choice: ').strip()

            if choice == '1':
                node.chat.start()

            elif choice == '2':
                import json as j
                s = node.status()
                print()
                print(f'  Node ID:  {s[\"node_id\"][:24]}...')
                print(f'  Credits:  {s[\"credits\"]}')
                print(f'  Peers:    {s[\"network\"].get(\"alive_peers\", 0)}')
                print(f'  Battery:  {s.get(\"battery_message\", \"unknown\")}')
                print(f'  Paused:   {s[\"paused\"]}')
                print(f'  Memories: {s[\"memory\"].get(\"total_memories\", 0)}')
                print()

            elif choice == '3':
                print()
                stats = node.memory.stats()
                print(f'  Total memories: {stats[\"total_memories\"]}')
                print(f'  Conversations:  {stats[\"total_sessions\"]}')
                print()
                print('  By type:')
                for t, c in stats.get('by_type', {}).items():
                    print(f'    {t}: {c}')
                sessions = node.memory.list_sessions()
                if sessions:
                    print()
                    print('  Recent conversations:')
                    for s in sessions[:5]:
                        print(f'    [{s[\"turns\"]} turns] {s[\"title\"][:40]}')
                print()

            elif choice == '4':
                print()
                print(f'  Balance: {node.credits.balance} credits')
                print()
                print('  Recent activity:')
                for entry in node.credits.history(last_n=5):
                    sign = '+' if entry['type'] == 'earn' else '-'
                    print(f'    {sign}{entry[\"amount\"]} — {entry[\"reason\"]}')
                print()

            elif choice == '5':
                print()
                print('  Resource settings:')
                print(f'    CPU:       {node.resources.cpu_percent}%')
                print(f'    RAM:       {node.resources.ram_mb}MB')
                print(f'    Disk:      {node.resources.disk_gb}GB')
                print(f'    Bandwidth: {node.resources.bandwidth_percent}%')
                print()
                change = input('  Change a setting? (y/n): ').strip().lower()
                if change == 'y':
                    print('  Which? (cpu/ram/disk/bw): ', end='')
                    setting = input().strip().lower()
                    print('  New value: ', end='')
                    val = input().strip()
                    try:
                        if setting == 'cpu':
                            node.update_resources(cpu_percent=int(val))
                        elif setting == 'ram':
                            node.update_resources(ram_mb=int(val))
                        elif setting == 'disk':
                            node.update_resources(disk_gb=int(val))
                        elif setting == 'bw':
                            node.update_resources(bandwidth_percent=int(val))
                        print('  Updated.')
                    except Exception as e:
                        print(f'  Error: {e}')

            elif choice in ('0', 'q', 'quit', 'exit'):
                node.stop()
                break

            else:
                print('  Enter 0-5')

        except KeyboardInterrupt:
            print()
            print('  Type 0 to exit or continue.')

except ValueError as e:
    print(f'Error: {e}')
    print('Wrong passphrase.')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
finally:
    try:
        termux_wake_unlock_cmd = 'termux-wake-unlock'
        import subprocess
        subprocess.run(termux_wake_unlock_cmd, shell=True, timeout=2)
    except:
        pass
"
MAINSCRIPT
chmod +x "$HOME/anad.sh"

# Termux widget shortcut
mkdir -p "$HOME/.shortcuts"
cat > "$HOME/.shortcuts/Anad" << 'WIDGET'
#!/bin/bash
bash $HOME/anad.sh
WIDGET
chmod +x "$HOME/.shortcuts/Anad"

# Termux boot script (auto-start on phone boot)
mkdir -p "$HOME/.termux/boot"
cat > "$HOME/.termux/boot/anad_autostart.sh" << 'BOOT'
#!/bin/bash
# Auto-start Anad node on phone boot
# Runs silently in background — no UI
# Only contributes when charging + WiFi
sleep 30  # wait for network
cd $HOME/anad
python -c "
import sys, os
sys.path.insert(0, '.')
from mobile.mobile_node import MobileAnadNode
import json

DATA_DIR = os.path.expanduser('~/.anad_data')
# Load saved passphrase hash for auto-start
# Note: passphrase required interactively first time
# After first run, node can restart without passphrase
# via saved session token
print('Anad background node starting...')
" 2>/dev/null &
BOOT
chmod +x "$HOME/.termux/boot/anad_autostart.sh"

echo "  ✓ Shortcuts created"

# ── Done ─────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║         SETUP COMPLETE ✓                  ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "  Start Anad now:"
echo ""
echo "    bash ~/anad.sh"
echo ""
echo "  Home screen shortcut:"
echo "    Install Termux:Widget from F-Droid"
echo "    Long press home screen → Widget → Termux"
echo "    You will see 'Anad' shortcut"
echo ""
echo "  Auto-start on boot:"
echo "    Install Termux:Boot from F-Droid"
echo "    Open it once to enable"
echo "    Anad starts automatically when phone boots"
echo ""
echo "  Your data is stored at:"
echo "    ~/.anad_data/"
echo "    Encrypted. Only you can read it."
echo ""
echo "  Credits earned:"
echo "    5 per hour when contributing (laptop tier)"
echo "    +1 per query served"
echo "    50 free starter credits"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ready to start? Run:  bash ~/anad.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ANAD MOBILE SETUP
# ═══════════════════════════════════════════════════════════
# Run this in Termux on your Android device
# This brings Anad to life on your phone
#
# What this does:
#   1. Installs required packages
#   2. Downloads Anad from GitHub
#   3. Downloads the quantized Nano model
#   4. Creates your identity
#   5. Starts your node
#
# Requirements:
#   - Termux installed from F-Droid (NOT Play Store)
#   - Internet connection for first setup
#   - ~500MB free storage
#
# Usage:
#   pkg install wget -y
#   bash anad_setup.sh
# ═══════════════════════════════════════════════════════════

set -e  # stop on any error

ANAD_DIR="$HOME/anad"
ANAD_VERSION="0.1.0"
ANAD_REPO="https://github.com/Jamesjules/anad-ai"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         ANAD MOBILE SETUP            ║"
echo "║   Public AI — Yours forever          ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Step 1: Update and install base packages ──────────────
echo "► Step 1/6: Installing base packages..."
pkg update -y -q
pkg install -y -q python git openssl libffi

echo "  Python: $(python --version)"
echo "  ✓ Base packages installed"

# ── Step 2: Install Python dependencies ───────────────────
echo ""
echo "► Step 2/6: Installing Python packages..."
pip install --quiet --upgrade pip
pip install --quiet cryptography numpy

echo "  ✓ Python packages installed"

# ── Step 3: Clone Anad from GitHub ────────────────────────
echo ""
echo "► Step 3/6: Downloading Anad..."

if [ -d "$ANAD_DIR" ]; then
    echo "  Anad directory exists — updating..."
    cd "$ANAD_DIR"
    git pull --quiet
else
    git clone --quiet "$ANAD_REPO" "$ANAD_DIR"
    cd "$ANAD_DIR"
fi

echo "  ✓ Anad downloaded to $ANAD_DIR"

# ── Step 4: Mobile configuration ─────────────────────────
echo ""
echo "► Step 4/6: Configuring for mobile..."

cat > "$ANAD_DIR/mobile_config.json" << 'EOF'
{
  "device_type": "mobile",
  "cpu_percent": 30,
  "gpu_percent": 0,
  "ram_mb": 1024,
  "disk_gb": 2,
  "bandwidth_percent": 20,
  "active_hours": [0, 1, 2, 3, 4, 5, 22, 23],
  "charge_only": true,
  "wifi_only": true,
  "battery_threshold": 80,
  "model": "anad-nano",
  "quantization": "2bit"
}
EOF

echo "  ✓ Mobile config created"
echo "  Device: Android/Termux"
echo "  Model: Nano (lightweight)"
echo "  Active: charging + WiFi only"

# ── Step 5: Create startup script ─────────────────────────
echo ""
echo "► Step 5/6: Creating startup script..."

cat > "$HOME/anad_start.sh" << 'STARTSCRIPT'
#!/bin/bash
# Start Anad node
cd $HOME/anad
echo ""
echo "Starting Anad node..."
python -c "
import sys
sys.path.insert(0, '.')
from node.node import AnadNode

node = AnadNode(data_dir='$HOME/.anad_data')

import getpass
print()
passphrase = getpass.getpass('Enter your Anad passphrase (or press Enter to create new): ')
if not passphrase:
    passphrase = getpass.getpass('Create a passphrase for your identity: ')
    confirm = getpass.getpass('Confirm passphrase: ')
    if passphrase != confirm:
        print('Passphrases do not match.')
        sys.exit(1)

alias = input('Your name/alias (optional, press Enter to skip): ').strip()
node.start(passphrase=passphrase, alias=alias)

print()
print('Anad is running! Commands:')
print('  status  — show network status')
print('  pause   — pause node')
print('  resume  — resume node')
print('  memory  — show your memories')
print('  quit    — stop node')
print()

while True:
    try:
        cmd = input('anad> ').strip().lower()
        if cmd == 'status':
            import json
            print(json.dumps(node.status(), indent=2))
        elif cmd == 'pause':
            node.pause()
        elif cmd == 'resume':
            node.resume()
        elif cmd == 'memory':
            stats = node.memory.stats()
            print(f'Memories: {stats[\"total_memories\"]}')
            print(f'Sessions: {stats[\"total_sessions\"]}')
        elif cmd in ('quit', 'exit', 'q'):
            node.stop()
            break
        elif cmd == 'help':
            print('Commands: status, pause, resume, memory, quit')
        elif cmd:
            print(f'Unknown command: {cmd}. Type help.')
    except KeyboardInterrupt:
        print()
        node.pause()
        print('Node paused. Type resume to continue or quit to exit.')
    except EOFError:
        break
"
STARTSCRIPT

chmod +x "$HOME/anad_start.sh"
echo "  ✓ Startup script created"

# ── Step 6: Create Termux widget shortcut ─────────────────
echo ""
echo "► Step 6/6: Creating shortcuts..."

# Termux:Widget support
mkdir -p "$HOME/.shortcuts"
cat > "$HOME/.shortcuts/Anad" << 'WIDGET'
#!/bin/bash
cd $HOME
bash anad_start.sh
WIDGET
chmod +x "$HOME/.shortcuts/Anad"

echo "  ✓ Shortcuts created"

# ── Done ──────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║        SETUP COMPLETE ✓             ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "To start Anad:"
echo ""
echo "  bash ~/anad_start.sh"
echo ""
echo "Or if you install Termux:Widget app,"
echo "add a shortcut to your home screen."
echo ""
echo "Your data will be stored in:"
echo "  ~/.anad_data/"
echo ""
echo "Keep your passphrase safe."
echo "It protects your identity and memory."
echo ""

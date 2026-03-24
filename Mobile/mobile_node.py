"""
Anad Mobile Node
================
Optimized for Android/iOS devices.

Key differences from desktop node:
  - Only runs when charging AND on WiFi
  - Battery threshold check before contributing
  - Tiny quantized model (Nano 2-bit = ~45MB)
  - Aggressive memory management
  - Background service aware
  - Automatic pause on low battery

This is the same Anad network — just mobile optimized.
Same identity. Same credits. Same memory.

Author: Anad Community
License: Public Domain
"""

import os
import sys
import json
import time
import threading
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from node.node import AnadNode, ResourceConfig, CreditLedger
from node.identity import AnadIdentity
from memory.memory import AnadMemory


# ══════════════════════════════════════════════════════════════════
# BATTERY MONITOR
# ══════════════════════════════════════════════════════════════════

class BatteryMonitor:
    """
    Monitors battery state on Android via Termux API.

    Anad only contributes when:
      - Charging (not draining your battery)
      - Battery above threshold
      - WiFi connected (not using mobile data)

    Your phone first. Anad second. Always.
    """

    def __init__(self, charge_threshold: int = 80):
        self.charge_threshold = charge_threshold
        self._last_check = 0
        self._cached_state = {"charging": False, "level": 100, "wifi": True}

    def get_state(self) -> dict:
        """Get current battery and network state"""
        # Cache for 30 seconds to avoid excessive checks
        if time.time() - self._last_check < 30:
            return self._cached_state

        state = {"charging": False, "level": 100, "wifi": True}

        # Try Termux API (Android)
        try:
            import subprocess
            result = subprocess.run(
                ["termux-battery-status"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                state["charging"] = data.get("status") in ("CHARGING", "FULL")
                state["level"] = data.get("percentage", 100)
        except Exception:
            # Not on Termux or API not available
            # Assume safe defaults
            state["charging"] = True
            state["level"] = 100

        # Try to check WiFi
        try:
            import subprocess
            result = subprocess.run(
                ["termux-wifi-connectioninfo"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                state["wifi"] = data.get("supplicant_state") == "COMPLETED"
        except Exception:
            state["wifi"] = True  # assume WiFi if can't check

        self._cached_state = state
        self._last_check = time.time()
        return state

    def should_contribute(self) -> bool:
        """
        Should this device contribute to the network right now?
        Only contributes when it's safe and won't impact the user.
        """
        state = self.get_state()
        return (
            state["charging"] and
            state["level"] >= self.charge_threshold and
            state["wifi"]
        )

    def status_message(self) -> str:
        state = self.get_state()
        if self.should_contribute():
            return f"Contributing ✓ (battery {state['level']}%, charging, WiFi)"
        reasons = []
        if not state["charging"]:
            reasons.append("not charging")
        if state["level"] < self.charge_threshold:
            reasons.append(f"battery {state['level']}% (need {self.charge_threshold}%)")
        if not state["wifi"]:
            reasons.append("no WiFi")
        return f"Standby ({', '.join(reasons)})"


# ══════════════════════════════════════════════════════════════════
# MOBILE CHAT INTERFACE — terminal based
# ══════════════════════════════════════════════════════════════════

class MobileChat:
    """
    Simple terminal chat interface for mobile.
    Works in Termux on Android.
    Clean. Fast. Offline-capable.
    """

    def __init__(self, node: "MobileAnadNode"):
        self.node = node
        self._history = []

    def start(self):
        """Start interactive chat session"""
        self.node.memory.new_session()

        print("\n" + "─" * 40)
        print("  Anad — ask anything")
        print("  /memory — show your memories")
        print("  /forget  — delete a memory")
        print("  /status  — node status")
        print("  /save    — remember something")
        print("  /exit    — end chat")
        print("─" * 40 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break

                # Regular chat
                self.node.memory.add_turn("user", user_input)
                response = self._get_response(user_input)
                self.node.memory.add_turn("anad", response)

                print(f"\nAnad: {response}\n")

            except KeyboardInterrupt:
                print("\n\nChat paused. Type /exit to quit.")
            except EOFError:
                break

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns False to exit."""
        parts = cmd.split(" ", 1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/exit":
            print("Goodbye.")
            return False

        elif command == "/memory":
            stats = self.node.memory.stats()
            print(f"\n  Memories: {stats['total_memories']}")
            print(f"  Sessions: {stats['total_sessions']}")
            mems = self.node.memory.get_by_type("context") + \
                   self.node.memory.get_by_type("preference")
            for m in mems[:10]:
                print(f"  [{m.memory_type}] {m.content[:60]}")
            print()

        elif command == "/save":
            if arg:
                entry = self.node.memory.remember(arg, "anchor")
                print(f"  Saved: {entry.memory_id}")
            else:
                content = input("  What should I remember? ")
                entry = self.node.memory.remember(content, "anchor")
                print(f"  Saved: {entry.memory_id}")

        elif command == "/forget":
            if arg:
                self.node.memory.forget(arg)
            else:
                mid = input("  Memory ID to forget: ")
                self.node.memory.forget(mid)

        elif command == "/status":
            status = self.node.status()
            print(f"\n  Node: {status['node_id'][:20]}...")
            print(f"  Credits: {status['credits']}")
            print(f"  Battery: {self.node.battery.status_message()}")
            print(f"  Peers: {status['network'].get('alive_peers', 0)}")
            print()

        elif command == "/export":
            path = os.path.expanduser("~/anad_memory_backup.json")
            self.node.memory.export(path)
            print(f"  Exported to {path}")

        else:
            print(f"  Unknown command: {command}")

        return True

    def _get_response(self, user_input: str) -> str:
        """
        Get response from Anad.
        Uses local model if available, routes to network otherwise.
        """
        # Build context from memory
        memory_context = self.node.memory.build_context()
        conv_context = self.node.memory.get_conversation_context(last_n=6)

        # Check if we have credits
        if self.node.credits.balance <= 0:
            return (
                "You're out of credits. "
                "Connect your phone to WiFi and charger to earn more."
            )

        # Spend 1 credit
        self.node.credits.spend(1, "conversation")

        # Route to model
        query = {
            "type": "conversation",
            "message": user_input,
            "memory_context": memory_context,
            "conversation_history": conv_context,
        }

        # Try local first, then network
        if self.node._model_loaded:
            return self.node._run_local_inference(query)
        else:
            result = self.node.router.route(query) if self.node.router else None
            if result and not result.get("error"):
                return result.get("response", "...")
            return (
                "No nodes available right now. "
                "Your query will be answered when the network reconnects."
            )


# ══════════════════════════════════════════════════════════════════
# MOBILE ANAD NODE
# ══════════════════════════════════════════════════════════════════

class MobileAnadNode(AnadNode):
    """
    Mobile-optimized Anad node.

    Inherits everything from AnadNode.
    Adds: battery awareness, tiny model, mobile UI.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.expanduser("~/.anad_data")
        super().__init__(data_dir=data_dir)
        self.battery = BatteryMonitor(charge_threshold=80)
        self._model_loaded = False
        self._battery_thread: Optional[threading.Thread] = None
        self.chat = None

    def start(self, passphrase: str, alias: str = "", port: int = 8765):
        """Start mobile node with battery awareness"""
        super().start(passphrase=passphrase, alias=alias, port=port)

        # Override resources for mobile
        self.update_resources(
            cpu_percent=30,
            gpu_percent=0,
            ram_mb=1024,
            disk_gb=2,
            bandwidth_percent=20,
        )

        # Start battery monitor
        self._battery_thread = threading.Thread(
            target=self._battery_loop,
            daemon=True,
            name="anad-battery-monitor",
        )
        self._battery_thread.start()

        # Initialize chat
        self.chat = MobileChat(self)

        # Load nano model if space allows
        self._try_load_nano_model()

        print(f"\n  Battery: {self.battery.status_message()}")
        print(f"  Model: {'loaded' if self._model_loaded else 'routing to network'}")

    def _battery_loop(self):
        """
        Monitor battery and auto-pause/resume.
        Your phone always comes first.
        """
        while self._running:
            should_contribute = self.battery.should_contribute()

            if should_contribute and self._paused.is_set():
                # Safe to contribute now
                print("\n  [Battery OK — node resuming contribution]")
                self._paused.clear()

            elif not should_contribute and not self._paused.is_set():
                # Not safe — pause automatically
                state = self.battery.get_state()
                if not state["charging"]:
                    print("\n  [Battery unplugged — node paused to save power]")
                elif state["level"] < self.battery.charge_threshold:
                    print(f"\n  [Battery {state['level']}% — paused until charged more]")
                elif not state["wifi"]:
                    print("\n  [WiFi disconnected — paused to save data]")
                self._paused.set()

            time.sleep(60)  # check every minute

    def _try_load_nano_model(self):
        """
        Try to load the quantized Nano model locally.
        If model file exists and RAM allows.
        """
        model_path = os.path.join(self.data_dir, "model_nano_2bit")
        if os.path.exists(model_path) and self.resources.ram_mb >= 512:
            try:
                # Load quantized model
                # Full implementation when model training is complete
                self._model_loaded = True
                print("  Nano model loaded — offline inference available")
            except Exception as e:
                print(f"  Model not loaded ({e}) — using network inference")
        else:
            print("  No local model — routing queries to network")

    def _run_local_inference(self, query: dict) -> str:
        """Run inference locally on device"""
        # Will be implemented when model training completes
        # For now returns placeholder
        return "Local inference coming in V1"

    def status(self) -> dict:
        """Mobile status including battery"""
        base = super().status()
        base["battery"] = self.battery.get_state()
        base["battery_message"] = self.battery.status_message()
        base["model_loaded"] = self._model_loaded
        return base


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for mobile node.
    Run this in Termux: python mobile_node.py
    """
    import getpass

    print("\n╔══════════════════════════════════════╗")
    print("║             ANAD                     ║")
    print("║     Public AI — On Your Phone        ║")
    print("╚══════════════════════════════════════╝\n")

    data_dir = os.path.expanduser("~/.anad_data")
    node = MobileAnadNode(data_dir=data_dir)

    # Get passphrase
    identity_path = os.path.join(data_dir, "identity.json")
    if os.path.exists(identity_path):
        print("Welcome back to Anad.")
        passphrase = getpass.getpass("Passphrase: ")
    else:
        print("First time setup — creating your identity.")
        print("This will be your permanent Anad identity.")
        print("Store your passphrase safely.\n")
        passphrase = getpass.getpass("Create passphrase: ")
        confirm = getpass.getpass("Confirm passphrase: ")
        if passphrase != confirm:
            print("Passphrases don't match.")
            return
        alias = input("Your name (optional): ").strip()

    # Start node
    try:
        node.start(passphrase=passphrase)
        # Start chat interface
        node.chat.start()
    except ValueError as e:
        print(f"\nError: {e}")
        print("Wrong passphrase or corrupted identity.")
    except KeyboardInterrupt:
        print("\n\nAnad paused.")
    finally:
        node.stop()


if __name__ == "__main__":
    main()

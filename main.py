"""
Anad — Main Entry Point
=======================
Run this to start your Anad node.

Usage:
    python main.py

Author: Anad Community
License: Public Domain
"""

import sys
import os

# Make sure imports work from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from node.node import AnadNode
import getpass


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║              ANAD                    ║")
    print("║   Public AI — Owned by everyone      ║")
    print("╚══════════════════════════════════════╝\n")

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anad_data")
    node = AnadNode(data_dir=data_dir)

    identity_path = os.path.join(data_dir, "identity.json")

    if os.path.exists(identity_path):
        print("Welcome back.")
        passphrase = getpass.getpass("Passphrase: ")
    else:
        print("First time setup — creating your identity.\n")
        while True:
            passphrase = getpass.getpass("Create passphrase: ")
            confirm = getpass.getpass("Confirm passphrase: ")
            if passphrase == confirm:
                break
            print("Passphrases don't match. Try again.\n")
        alias = input("Your name or alias (optional): ").strip()

    try:
        node.start(passphrase=passphrase)

        print("\nCommands:")
        print("  status  — network status")
        print("  pause   — pause node")
        print("  resume  — resume node")
        print("  memory  — your memories")
        print("  credits — your balance")
        print("  quit    — stop\n")

        while True:
            try:
                cmd = input("anad> ").strip().lower()

                if cmd == "status":
                    import json
                    s = node.status()
                    print(f"\n  Node ID:  {s['node_id'][:28]}...")
                    print(f"  Credits:  {s['credits']}")
                    print(f"  Peers:    {s['network'].get('alive_peers', 0)}")
                    print(f"  Paused:   {s['paused']}\n")

                elif cmd == "pause":
                    node.pause()

                elif cmd == "resume":
                    node.resume()

                elif cmd == "memory":
                    stats = node.memory.stats()
                    print(f"\n  Memories: {stats['total_memories']}")
                    print(f"  Sessions: {stats['total_sessions']}\n")

                elif cmd == "credits":
                    print(f"\n  Balance: {node.credits.balance} credits")
                    for e in node.credits.history(last_n=5):
                        sign = "+" if e["type"] == "earn" else "-"
                        print(f"  {sign}{e['amount']} — {e['reason']}")
                    print()

                elif cmd in ("quit", "exit", "q"):
                    node.stop()
                    break

                elif cmd:
                    print("  Unknown command. Type status, pause, resume, memory, credits, quit")

            except KeyboardInterrupt:
                print("\n  Ctrl+C — type quit to exit or resume to continue")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Wrong passphrase or corrupted identity file.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

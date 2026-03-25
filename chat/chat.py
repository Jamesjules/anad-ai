"""
Anad Chat
=========
Talk to Anad directly from your terminal.
Uses your trained model weights.
Memory persists across sessions.

Usage:
    python chat.py

Commands during chat:
    /help     — show commands
    /memory   — show what Anad remembers about you
    /save     — save something to memory
    /forget   — delete a memory
    /clear    — start fresh conversation
    /history  — show past sessions
    /status   — model and node status
    /settings — adjust response style
    /exit     — quit

Author: Anad Community
License: Public Domain
"""

import os
import sys
import json
import time
import getpass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.inference import AnadInference
from memory.memory import AnadMemory
from node.identity import AnadIdentity


# ══════════════════════════════════════════════════════════════════
# RESPONSE FORMATTER
# ══════════════════════════════════════════════════════════════════

def format_response(text: str, width: int = 70) -> str:
    """Wrap long responses for clean terminal display"""
    if not text:
        return ""
    words = text.split()
    lines = []
    current = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return "\n  ".join(lines)


# ══════════════════════════════════════════════════════════════════
# CHAT SESSION
# ══════════════════════════════════════════════════════════════════

class AnadChat:
    """
    Full chat interface with memory, history, and settings.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        data_dir: str = "./anad_data",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.inference = AnadInference(checkpoint_dir)
        self.memory = None
        self.identity = None
        self.session_turns = []

        # Response settings
        self.temperature = 0.8
        self.max_tokens  = 200
        self.show_stats  = False

    def start(self):
        """Start the chat interface"""
        self._print_banner()

        # Load identity and memory
        self._load_identity()

        # Load model
        print("  Loading Anad model...\n")
        if not self.inference.load():
            print("  Run 'python train.py' first to train the model.")
            return

        # Start conversation
        self._new_session()
        self._chat_loop()

    def _print_banner(self):
        print("\n" + "═" * 52)
        print("  अनाद — Anad")
        print("  Public AI. Your memory. No corporate control.")
        print("═" * 52 + "\n")

    def _load_identity(self):
        """Load user identity for memory encryption"""
        identity_path = os.path.join(self.data_dir, "identity.json")
        if not os.path.exists(identity_path):
            print("  No identity found.")
            print("  Run 'python main.py' to create your identity first.\n")
            # Use anonymous memory
            self.memory = AnadMemory(
                storage_path=os.path.join(self.data_dir, "memory"),
                identity_public_key_hex="0" * 64,
            )
            return

        try:
            passphrase = getpass.getpass("  Passphrase: ")
            self.identity = AnadIdentity.load(identity_path, passphrase)
            self.memory = AnadMemory(
                storage_path=os.path.join(self.data_dir, "memory"),
                identity_public_key_hex=self.identity.public_key_hex,
            )
            print(f"  Identity: {self.identity.node_id[:24]}...")
            stats = self.memory.stats()
            if stats["total_memories"] > 0:
                print(f"  Memories: {stats['total_memories']} remembered")
            print()
        except ValueError:
            print("  Wrong passphrase. Using anonymous mode.\n")
            self.memory = AnadMemory(
                storage_path=os.path.join(self.data_dir, "memory_anon"),
                identity_public_key_hex="0" * 64,
            )

    def _new_session(self):
        """Start a new conversation session"""
        self.session_turns = []
        if self.memory:
            self.memory.new_session()

    def _build_prompt(self, user_input: str) -> str:
        """
        Build the full prompt including memory context
        and conversation history.
        """
        parts = []

        # Memory context
        if self.memory:
            ctx = self.memory.build_context()
            if ctx:
                parts.append(f"Context about this user:\n{ctx}\n")

        # Recent conversation history
        if self.session_turns:
            history = self.session_turns[-6:]  # last 3 exchanges
            for turn in history:
                role = "User" if turn["role"] == "user" else "Anad"
                parts.append(f"{role}: {turn['content']}")

        # Current input
        parts.append(f"User: {user_input}")
        parts.append("Anad:")

        return "\n".join(parts)

    def _get_response(self, user_input: str) -> tuple:
        """Get response from model. Returns (response, time_taken)"""
        prompt = self._build_prompt(user_input)
        t0 = time.time()
        response = self.inference.generate(
            prompt=prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            top_k=50,
        )
        elapsed = time.time() - t0

        # Clean up response — remove any echoed prompt
        for marker in ["User:", "Anad:", "Context about"]:
            if marker in response:
                response = response[:response.index(marker)].strip()

        return response.strip() or "...", elapsed

    def _chat_loop(self):
        """Main conversation loop"""
        print("  Chat with Anad. Type /help for commands.\n")
        print("─" * 52)

        while True:
            try:
                user_input = input("\n  You: ").strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Get response
                print("\n  Anad: ", end="", flush=True)
                response, elapsed = self._get_response(user_input)
                print(format_response(response))

                if self.show_stats:
                    print(f"\n  [{elapsed:.1f}s | temp={self.temperature}]")

                # Save to memory and session
                self.session_turns.append({"role": "user",    "content": user_input})
                self.session_turns.append({"role": "anad",    "content": response})

                if self.memory:
                    self.memory.add_turn("user", user_input)
                    self.memory.add_turn("anad", response)

                    # Auto-detect things to remember
                    self._auto_remember(user_input)

            except KeyboardInterrupt:
                print("\n\n  (Ctrl+C — type /exit to quit)")
            except EOFError:
                break

        print("\n  Goodbye.\n")

    def _auto_remember(self, user_input: str):
        """
        Automatically detect and save important information
        the user shares in conversation.
        """
        triggers = [
            ("my name is", "context"),
            ("i am a", "context"),
            ("i work", "context"),
            ("i live in", "context"),
            ("i prefer", "preference"),
            ("i like", "preference"),
            ("i don't like", "preference"),
            ("please always", "preference"),
            ("please never", "preference"),
            ("remember that", "anchor"),
            ("don't forget", "anchor"),
        ]
        lower = user_input.lower()
        for trigger, mtype in triggers:
            if trigger in lower:
                self.memory.remember(user_input, mtype)
                break

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns False to exit."""
        parts = cmd.split(" ", 1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "/exit":
            return False

        elif command == "/help":
            print("""
  Commands:
    /help      — this list
    /memory    — show what Anad remembers
    /save      — save something to memory
    /forget    — delete a memory
    /clear     — start fresh conversation
    /history   — past conversations
    /status    — model info
    /temp N    — set temperature (0.1-1.5)
    /tokens N  — set max response length
    /stats     — toggle timing display
    /export    — export all your memories
    /exit      — quit
""")

        elif command == "/memory":
            if not self.memory:
                print("  No memory system loaded.")
                return True
            stats = self.memory.stats()
            print(f"\n  Total memories: {stats['total_memories']}")
            print(f"  Conversations:  {stats['total_sessions']}\n")
            all_mems = []
            for mtype in ["preference", "context", "anchor"]:
                mems = self.memory.get_by_type(mtype)
                for m in mems[:5]:
                    all_mems.append((mtype, m))
            if all_mems:
                print("  What Anad remembers about you:")
                for mtype, m in all_mems:
                    print(f"    [{mtype}] {m.content[:60]}")
            else:
                print("  Nothing remembered yet.")
                print("  Say things like 'I prefer concise answers' to build memory.")
            print()

        elif command == "/save":
            content = arg or input("  What to remember: ").strip()
            if content:
                entry = self.memory.remember(content, "anchor")
                print(f"  Saved. (id: {entry.memory_id})")

        elif command == "/forget":
            mid = arg or input("  Memory ID to forget: ").strip()
            if mid:
                self.memory.forget(mid)
                print(f"  Forgotten.")

        elif command == "/clear":
            self._new_session()
            print("  Conversation cleared. Memory intact.")

        elif command == "/history":
            sessions = self.memory.list_sessions()
            if not sessions:
                print("  No past conversations.")
            else:
                print(f"\n  Past {min(10, len(sessions))} conversations:")
                for s in sessions[:10]:
                    ts = time.strftime("%b %d %H:%M", time.localtime(s["started_at"]))
                    print(f"    [{ts}] {s['title'][:45]} ({s['turns']} turns)")
                print()

        elif command == "/status":
            ckpt = self.inference._find_latest_checkpoint()
            print(f"\n  Model:      Anad Nano")
            print(f"  Checkpoint: {os.path.basename(ckpt) if ckpt else 'none'}")
            print(f"  Device:     {self.inference.device.upper()}")
            print(f"  Temp:       {self.temperature}")
            print(f"  Max tokens: {self.max_tokens}")
            if self.identity:
                print(f"  Node ID:    {self.identity.node_id[:28]}...")
            print()

        elif command == "/temp":
            try:
                self.temperature = float(arg)
                print(f"  Temperature set to {self.temperature}")
            except ValueError:
                print("  Usage: /temp 0.8  (0.1 = focused, 1.5 = creative)")

        elif command == "/tokens":
            try:
                self.max_tokens = int(arg)
                print(f"  Max tokens set to {self.max_tokens}")
            except ValueError:
                print("  Usage: /tokens 200")

        elif command == "/stats":
            self.show_stats = not self.show_stats
            print(f"  Stats display: {'on' if self.show_stats else 'off'}")

        elif command == "/export":
            path = os.path.join(self.data_dir, "memory_export.json")
            self.memory.export(path)
            print(f"  Exported to {path}")

        else:
            print(f"  Unknown command: {command}. Type /help")

        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chat with Anad")
    parser.add_argument("--checkpoints", default="./checkpoints")
    parser.add_argument("--data",        default="./anad_data")
    parser.add_argument("--temp",        type=float, default=0.8)
    parser.add_argument("--tokens",      type=int,   default=200)
    args = parser.parse_args()

    chat = AnadChat(
        checkpoint_dir=args.checkpoints,
        data_dir=args.data,
    )
    chat.temperature = args.temp
    chat.max_tokens  = args.tokens
    chat.start()


if __name__ == "__main__":
    main()

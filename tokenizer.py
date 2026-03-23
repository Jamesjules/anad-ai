"""
ANAD Tokenizer
==============
Built from scratch. No corporate dependencies.
Supports all Indian languages natively.
Then all world languages.

Philosophy:
- Humanity first, not English first
- Transparent, auditable, open
- No hidden behavior
- BPE (Byte Pair Encoding) based

Author: Anad Community
License: Public Domain — belongs to everyone
"""

import re
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# SPECIAL TOKENS
# These are universal — same across all languages
# ─────────────────────────────────────────────
SPECIAL_TOKENS = {
    "<PAD>":   0,   # Padding
    "<UNK>":   1,   # Unknown token
    "<BOS>":   2,   # Beginning of sequence
    "<EOS>":   3,   # End of sequence
    "<SEP>":   4,   # Separator
    "<MASK>":  5,   # Mask for training
    "<USER>":  6,   # User turn marker
    "<ANAD>":  7,   # Anad response marker
    "<DOC>":   8,   # Document start
    "<CODE>":  9,   # Code block start
    "<NL>":    10,  # Newline token
}

# ─────────────────────────────────────────────
# LANGUAGE SCRIPTS
# Unicode ranges for each script
# Indian languages prioritized
# ─────────────────────────────────────────────
SCRIPT_RANGES = {
    # Indian Scripts — First Class Citizens
    "gujarati":   (0x0A80, 0x0AFF),
    "devanagari": (0x0900, 0x097F),  # Hindi, Sanskrit, Marathi
    "bengali":    (0x0980, 0x09FF),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
    "kannada":    (0x0C80, 0x0CFF),
    "malayalam":  (0x0D00, 0x0D7F),
    "punjabi":    (0x0A00, 0x0A7F),  # Gurmukhi
    "odia":       (0x0B00, 0x0B7F),
    "urdu":       (0x0600, 0x06FF),  # Arabic script

    # Other Major Scripts
    "latin":      (0x0000, 0x007F),  # English and European
    "arabic":     (0x0600, 0x06FF),
    "chinese":    (0x4E00, 0x9FFF),
    "japanese":   (0x3040, 0x30FF),
    "korean":     (0xAC00, 0xD7AF),
    "cyrillic":   (0x0400, 0x04FF),  # Russian
    "greek":      (0x0370, 0x03FF),
}


class AnadTokenizer:
    """
    Anad BPE Tokenizer

    Byte Pair Encoding tokenizer built from scratch.
    Trained on multilingual data with Indian language priority.

    Usage:
        tokenizer = AnadTokenizer()
        tokenizer.train(texts, vocab_size=32000)
        tokens = tokenizer.encode("Hello world")
        text = tokenizer.decode(tokens)
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.special_tokens = SPECIAL_TOKENS.copy()

        # Initialize with special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Load special tokens into vocabulary first"""
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token

    def _get_base_vocab(self) -> Dict[str, int]:
        """
        Build base vocabulary from raw bytes
        This ensures every possible input can be tokenized
        No unknown characters — ever
        """
        base = self.special_tokens.copy()
        idx = len(base)

        # All 256 bytes as base tokens
        # This is the foundation — handles ANY language
        for i in range(256):
            token = f"<byte_{i:03d}>"
            base[token] = idx
            idx += 1

        # End of word marker
        base["</w>"] = idx
        idx += 1

        return base

    def _text_to_bytes(self, text: str) -> List[str]:
        """Convert text to byte-level tokens"""
        encoded = text.encode("utf-8")
        return [f"<byte_{b:03d}>" for b in encoded]

    def _get_word_frequencies(
        self, texts: List[str]
    ) -> Dict[Tuple[str, ...], int]:
        """Count word frequencies in training corpus"""
        freq = defaultdict(int)

        for text in texts:
            # Split on whitespace, preserve all characters
            words = text.split()
            for word in words:
                # Convert to byte tokens with end marker
                byte_tokens = tuple(self._text_to_bytes(word)) + ("</w>",)
                freq[byte_tokens] += 1

        return freq

    def _get_pair_frequencies(
        self, word_freq: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent token pairs"""
        pair_freq = defaultdict(int)

        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += freq

        return pair_freq

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_freq: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Merge a token pair across entire vocabulary"""
        new_word_freq = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word, freq in word_freq.items():
            # Find and merge all occurrences of pair
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freq[tuple(new_word)] = freq

        return new_word_freq

    def train(self, texts: List[str], vocab_size: Optional[int] = None):
        """
        Train tokenizer on text corpus

        Args:
            texts: List of training texts
                   Should include texts in all target languages
                   Indian languages should be well represented
            vocab_size: Target vocabulary size (default: self.vocab_size)
        """
        if vocab_size:
            self.vocab_size = vocab_size

        print(f"Training Anad tokenizer...")
        print(f"Target vocab size: {self.vocab_size}")
        print(f"Training on {len(texts)} texts")

        # Start with byte-level base vocabulary
        self.vocab = self._get_base_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Get word frequencies
        print("Computing word frequencies...")
        word_freq = self._get_word_frequencies(texts)

        # BPE training loop
        num_merges = self.vocab_size - len(self.vocab)
        print(f"Performing {num_merges} BPE merges...")

        for i in range(num_merges):
            # Get pair frequencies
            pair_freq = self._get_pair_frequencies(word_freq)

            if not pair_freq:
                print(f"No more pairs to merge at step {i}")
                break

            # Find most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            best_freq = pair_freq[best_pair]

            if best_freq < 2:
                print(f"All remaining pairs appear only once. Stopping.")
                break

            # Create new merged token
            new_token = "".join(best_pair)
            new_idx = len(self.vocab)

            # Add to vocabulary
            self.vocab[new_token] = new_idx
            self.reverse_vocab[new_idx] = new_token
            self.merges[best_pair] = new_token

            # Merge pair in word frequencies
            word_freq = self._merge_pair(best_pair, word_freq)

            if (i + 1) % 1000 == 0:
                print(f"  Merge {i+1}/{num_merges} — vocab size: {len(self.vocab)}")

        print(f"Training complete. Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token ids

        Args:
            text: Input text in any language

        Returns:
            List of integer token ids
        """
        if not self.merges and len(self.vocab) <= len(self.special_tokens) + 256:
            raise RuntimeError(
                "Tokenizer not trained yet. Call train() first."
            )

        tokens = []

        # Add BOS token
        tokens.append(self.special_tokens["<BOS>"])

        # Process each word
        words = text.split()
        for word in words:
            # Convert to byte tokens
            word_tokens = list(self._text_to_bytes(word)) + ["</w>"]

            # Apply BPE merges
            word_tokens = self._apply_merges(word_tokens)

            # Convert to ids
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.special_tokens["<UNK>"])

        # Add EOS token
        tokens.append(self.special_tokens["<EOS>"])

        return tokens

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply learned BPE merges to a token sequence"""
        while len(tokens) > 1:
            # Find the highest priority merge
            best_pair = None
            best_idx = None

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    best_pair = pair
                    best_idx = i
                    break  # Apply leftmost merge first

            if best_pair is None:
                break

            # Apply the merge
            merged = self.merges[best_pair]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2:]

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token ids back to text

        Args:
            token_ids: List of integer token ids

        Returns:
            Decoded text string
        """
        tokens = []

        for idx in token_ids:
            # Skip special tokens in output
            if idx in self.special_tokens.values():
                continue

            if idx in self.reverse_vocab:
                token = self.reverse_vocab[idx]
                tokens.append(self._token_to_bytes(token))

        # Join all bytes and decode
        try:
            return b"".join(tokens).decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    def _token_to_bytes(self, token: str) -> bytes:
        """
        Convert a token string back to bytes.
        Handles merged tokens that contain multiple byte references.
        """
        if token == "</w>":
            return b" "

        # Pure single byte token
        if token.startswith("<byte_") and token.endswith(">") and token.count("<byte_") == 1:
            try:
                byte_val = int(token[6:-1])
                return bytes([byte_val])
            except ValueError:
                pass

        # Merged token — extract all byte segments
        result = bytearray()
        i = 0
        while i < len(token):
            if token[i:].startswith("<byte_"):
                # Find closing >
                end = token.find(">", i)
                if end != -1:
                    byte_str = token[i+6:end]
                    try:
                        byte_val = int(byte_str)
                        result.append(byte_val)
                        i = end + 1
                        continue
                    except ValueError:
                        pass
            if token[i:] == "</w>" or token[i:].startswith("</w>"):
                result.extend(b" ")
                i += 4
                continue
            # Raw character fallback
            result.extend(token[i].encode("utf-8"))
            i += 1

        return bytes(result)

    def save(self, path: str):
        """Save tokenizer to disk"""
        os.makedirs(path, exist_ok=True)

        # Save vocabulary
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # Save merges
        merges_list = [
            {"pair": list(pair), "merged": merged}
            for pair, merged in self.merges.items()
        ]
        with open(os.path.join(path, "merges.json"), "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "version": "0.1.0",
            "project": "Anad",
            "description": "Public AI for humanity",
            "license": "Public Domain",
            "special_tokens": self.special_tokens,
            "supported_scripts": list(SCRIPT_RANGES.keys()),
        }
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AnadTokenizer":
        """Load tokenizer from disk"""
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)

        tokenizer = cls(vocab_size=config["vocab_size"])

        with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)
            tokenizer.reverse_vocab = {
                int(v): k for k, v in tokenizer.vocab.items()
            }

        with open(os.path.join(path, "merges.json"), "r", encoding="utf-8") as f:
            merges_list = json.load(f)
            tokenizer.merges = {
                tuple(item["pair"]): item["merged"]
                for item in merges_list
            }

        print(f"Tokenizer loaded from {path}")
        print(f"Vocab size: {len(tokenizer.vocab)}")
        return tokenizer

    def get_stats(self) -> Dict:
        """Return tokenizer statistics"""
        return {
            "vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "special_tokens": len(self.special_tokens),
            "supported_scripts": list(SCRIPT_RANGES.keys()),
            "version": "0.1.0",
        }

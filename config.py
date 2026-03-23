"""
Anad Model Configuration
========================
All model hyperparameters in one place.
Transparent, documented, auditable.

Three size tiers:
  nano  → runs on mobile / CPU only
  small → runs on laptop GPU
  base  → runs on desktop GPU (V1 target)

Author: Anad Community
License: Public Domain
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnadConfig:
    """
    Configuration for Anad transformer model.

    Every parameter is named clearly.
    No magic numbers buried in code.
    """

    # ── Vocabulary ─────────────────────────────
    vocab_size: int = 32000         # tokenizer vocabulary size
    pad_token_id: int = 0           # <PAD>
    bos_token_id: int = 2           # <BOS> beginning of sequence
    eos_token_id: int = 3           # <EOS> end of sequence

    # ── Model Dimensions ───────────────────────
    dim: int = 512                  # embedding dimension
    n_layers: int = 8               # number of transformer layers
    n_heads: int = 8                # number of attention heads
    n_kv_heads: int = 4             # key/value heads (grouped query attention)
    hidden_dim: int = 1024          # feed forward hidden dimension

    # ── Context ────────────────────────────────
    max_seq_len: int = 2048         # maximum sequence length

    # ── Regularization ─────────────────────────
    dropout: float = 0.1            # dropout rate during training
    attention_dropout: float = 0.1  # dropout on attention weights

    # ── Normalization ──────────────────────────
    norm_eps: float = 1e-6          # epsilon for RMS normalization

    # ── Positional Encoding ────────────────────
    rope_theta: float = 10000.0     # RoPE base frequency

    # ── Training ───────────────────────────────
    learning_rate: float = 3e-4     # initial learning rate
    weight_decay: float = 0.1       # L2 regularization
    grad_clip: float = 1.0          # gradient clipping threshold

    # ── Version ────────────────────────────────
    version: str = "0.1.0"
    model_name: str = "anad-nano"

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, \
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head"""
        return self.dim // self.n_heads

    @property
    def param_count_approx(self) -> str:
        """Rough parameter count estimate"""
        # Embedding
        embed = self.vocab_size * self.dim
        # Each transformer layer
        attn = 4 * self.dim * self.dim
        ffn = 3 * self.dim * self.hidden_dim
        layer = attn + ffn
        # Total
        total = embed + (self.n_layers * layer)
        if total > 1e9:
            return f"~{total/1e9:.1f}B parameters"
        return f"~{total/1e6:.0f}M parameters"


# ── Preset Configurations ──────────────────────────────────────────────────
# Three sizes. Same architecture. Different scale.
# Start with nano. Grow as the network grows.

ANAD_NANO = AnadConfig(
    dim=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    hidden_dim=512,
    max_seq_len=512,
    model_name="anad-nano",
    # Target: ~15M parameters
    # Runs on: any device including mobile
    # Purpose: prove the concept, early testing
)

ANAD_SMALL = AnadConfig(
    dim=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    hidden_dim=1024,
    max_seq_len=2048,
    model_name="anad-small",
    # Target: ~120M parameters
    # Runs on: laptop with 8GB RAM
    # Purpose: V0 public release
)

ANAD_BASE = AnadConfig(
    dim=2048,
    n_layers=24,
    n_heads=16,
    n_kv_heads=8,
    hidden_dim=5632,
    max_seq_len=4096,
    model_name="anad-base",
    # Target: ~1B parameters
    # Runs on: desktop GPU 8GB+
    # Purpose: V1 — matches small commercial models
)

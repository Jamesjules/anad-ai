"""
Anad Transformer Model
======================
Built from scratch. Every component documented.
No black boxes. No corporate dependencies.

Architecture: Decoder-only Transformer
Techniques used:
  - RoPE positional encoding (better than learned positions)
  - Grouped Query Attention (faster, less memory)
  - RMSNorm (simpler and faster than LayerNorm)
  - SwiGLU activation (better than ReLU in feed forward)
  - Pre-normalization (more stable training)

This file is intentionally educational.
Every function explains what it does and why.

Author: Anad Community
License: Public Domain
"""

import numpy as np
import math
import json
import os
from typing import Optional, Tuple
from model.config import AnadConfig, ANAD_NANO


# ══════════════════════════════════════════════════════════════════
# BUILDING BLOCK 1 — RMS NORMALIZATION
# ══════════════════════════════════════════════════════════════════

class RMSNorm:
    """
    Root Mean Square Normalization.

    Simpler and faster than LayerNorm.
    Used in LLaMA, Mistral, and most modern models.

    Why: Stabilizes training by normalizing the scale
         of activations without centering them.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)  # learnable scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Compute RMS across last dimension
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight

    def parameters(self):
        return {"weight": self.weight}

    def load(self, params: dict):
        self.weight = params["weight"]


# ══════════════════════════════════════════════════════════════════
# BUILDING BLOCK 2 — ROTARY POSITIONAL ENCODING (RoPE)
# ══════════════════════════════════════════════════════════════════

def build_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> np.ndarray:
    """
    Build RoPE frequency table.

    RoPE encodes position by rotating query and key vectors.
    This is better than adding position embeddings because:
    - Naturally handles sequences longer than training length
    - Relative positions are preserved in dot products
    - No extra parameters needed

    Returns: [max_seq_len, head_dim/2] complex frequency matrix
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    # Position indices
    positions = np.arange(max_seq_len, dtype=np.float32)
    # Outer product: each position gets all frequencies
    freqs_matrix = np.outer(positions, freqs)
    # Convert to complex for rotation
    return np.exp(1j * freqs_matrix).astype(np.complex64)


def apply_rope(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Apply rotary position embedding to query or key tensor.

    Rotates pairs of dimensions by the position-dependent angle.
    This encodes position information into the attention computation.

    x shape: [batch, seq_len, n_heads, head_dim]
    freqs shape: [seq_len, head_dim/2]
    """
    seq_len = x.shape[1]
    # View as complex numbers (pairs of floats become one complex)
    x_complex = x.astype(np.float32).view(np.complex64)
    # Get frequencies for this sequence length
    freqs_seq = freqs[:seq_len]
    # Reshape for broadcasting: [1, seq_len, 1, head_dim/2]
    freqs_seq = freqs_seq[np.newaxis, :, np.newaxis, :]
    # Rotate
    x_rotated = x_complex * freqs_seq
    # Convert back to real
    return x_rotated.view(np.float32)


# ══════════════════════════════════════════════════════════════════
# BUILDING BLOCK 3 — GROUPED QUERY ATTENTION
# ══════════════════════════════════════════════════════════════════

class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA).

    Instead of one key/value head per query head,
    multiple query heads share one key/value head.

    Why: Reduces memory usage significantly.
         Faster inference. Used in LLaMA 2, Mistral.

    Example with n_heads=8, n_kv_heads=4:
      8 query heads, but only 4 key/value heads
      Every 2 query heads share 1 key/value head
    """

    def __init__(self, config: AnadConfig):
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.dim = config.dim
        self.dropout = config.attention_dropout

        # Groups: how many query heads share one kv head
        self.n_groups = self.n_heads // self.n_kv_heads

        # Query projection: full heads
        self.wq = np.random.randn(
            config.dim, config.n_heads * config.head_dim
        ).astype(np.float32) * 0.02

        # Key projection: fewer heads
        self.wk = np.random.randn(
            config.dim, config.n_kv_heads * config.head_dim
        ).astype(np.float32) * 0.02

        # Value projection: fewer heads
        self.wv = np.random.randn(
            config.dim, config.n_kv_heads * config.head_dim
        ).astype(np.float32) * 0.02

        # Output projection
        self.wo = np.random.randn(
            config.n_heads * config.head_dim, config.dim
        ).astype(np.float32) * 0.02

        # RoPE frequencies
        self.rope_freqs = build_rope_freqs(
            config.head_dim, config.max_seq_len, config.rope_theta
        )

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> np.ndarray:
        """
        Forward pass of attention.

        x shape: [batch, seq_len, dim]
        mask shape: [seq_len, seq_len] — causal mask
        """
        batch, seq_len, _ = x.shape

        # ── Project to queries, keys, values ──
        q = x @ self.wq  # [batch, seq_len, n_heads * head_dim]
        k = x @ self.wk  # [batch, seq_len, n_kv_heads * head_dim]
        v = x @ self.wv  # [batch, seq_len, n_kv_heads * head_dim]

        # ── Reshape to [batch, seq_len, n_heads, head_dim] ──
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # ── Apply RoPE to queries and keys ──
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # ── Transpose for attention: [batch, n_heads, seq_len, head_dim] ──
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # ── Expand kv heads to match query heads ──
        # Each kv head is repeated n_groups times
        k = np.repeat(k, self.n_groups, axis=1)  # [batch, n_heads, seq_len, head_dim]
        v = np.repeat(v, self.n_groups, axis=1)

        # ── Scaled dot product attention ──
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [batch, n_heads, seq_len, seq_len]

        # ── Apply causal mask ──
        # Prevents attending to future tokens
        if mask is not None:
            scores = scores + mask[np.newaxis, np.newaxis, :, :]

        # ── Softmax ──
        scores = scores - scores.max(axis=-1, keepdims=True)  # numerical stability
        attn_weights = np.exp(scores)
        attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-9)

        # ── Dropout during training ──
        if training and self.dropout > 0:
            mask_drop = np.random.binomial(1, 1 - self.dropout, attn_weights.shape)
            attn_weights = attn_weights * mask_drop / (1 - self.dropout)

        # ── Weighted sum of values ──
        out = attn_weights @ v  # [batch, n_heads, seq_len, head_dim]

        # ── Reshape and project output ──
        out = out.transpose(0, 2, 1, 3)  # [batch, seq_len, n_heads, head_dim]
        out = out.reshape(batch, seq_len, self.n_heads * self.head_dim)
        out = out @ self.wo  # [batch, seq_len, dim]

        return out

    def parameters(self) -> dict:
        return {"wq": self.wq, "wk": self.wk, "wv": self.wv, "wo": self.wo}

    def load(self, params: dict):
        self.wq = params["wq"]
        self.wk = params["wk"]
        self.wv = params["wv"]
        self.wo = params["wo"]


# ══════════════════════════════════════════════════════════════════
# BUILDING BLOCK 4 — FEED FORWARD NETWORK WITH SwiGLU
# ══════════════════════════════════════════════════════════════════

class FeedForward:
    """
    Feed Forward Network with SwiGLU activation.

    SwiGLU: better than ReLU, used in PaLM, LLaMA, Mistral.
    Uses a gating mechanism — one branch controls the other.

    Why 3 projections instead of 2:
      w1: gate projection
      w2: output projection
      w3: up projection
    SwiGLU(x) = swish(w1·x) ⊗ (w3·x)
    output = w2 · SwiGLU(x)
    """

    def __init__(self, config: AnadConfig):
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim

        # Three projection matrices
        self.w1 = np.random.randn(config.dim, config.hidden_dim).astype(np.float32) * 0.02
        self.w2 = np.random.randn(config.hidden_dim, config.dim).astype(np.float32) * 0.02
        self.w3 = np.random.randn(config.dim, config.hidden_dim).astype(np.float32) * 0.02

    def _swish(self, x: np.ndarray) -> np.ndarray:
        """Swish activation: x * sigmoid(x)"""
        return x * (1.0 / (1.0 + np.exp(-x)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: [batch, seq_len, dim]
        """
        # Gate: swish(x @ w1)
        gate = self._swish(x @ self.w1)
        # Up: x @ w3
        up = x @ self.w3
        # SwiGLU: gate controls up
        hidden = gate * up
        # Project back to dim
        return hidden @ self.w2

    def parameters(self) -> dict:
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3}

    def load(self, params: dict):
        self.w1 = params["w1"]
        self.w2 = params["w2"]
        self.w3 = params["w3"]


# ══════════════════════════════════════════════════════════════════
# BUILDING BLOCK 5 — TRANSFORMER LAYER
# ══════════════════════════════════════════════════════════════════

class TransformerLayer:
    """
    One complete transformer layer.

    Order (pre-normalization):
      1. RMSNorm → Attention → residual add
      2. RMSNorm → FeedForward → residual add

    Pre-norm (normalize before sublayer) is more stable
    than post-norm (normalize after). Used in all modern LLMs.
    """

    def __init__(self, config: AnadConfig, layer_idx: int):
        self.layer_idx = layer_idx

        # Attention sublayer
        self.attention = GroupedQueryAttention(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

        # Feed forward sublayer
        self.ffn = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def __call__(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        training: bool = False,
    ) -> np.ndarray:
        """
        x shape: [batch, seq_len, dim]
        """
        # ── Attention with residual ──
        # Normalize first, then attend, then add back original
        h = x + self.attention(self.attention_norm(x), mask, training)

        # ── Feed forward with residual ──
        out = h + self.ffn(self.ffn_norm(h))

        return out

    def parameters(self) -> dict:
        return {
            "attention": self.attention.parameters(),
            "attention_norm": self.attention_norm.parameters(),
            "ffn": self.ffn.parameters(),
            "ffn_norm": self.ffn_norm.parameters(),
        }

    def load(self, params: dict):
        self.attention.load(params["attention"])
        self.attention_norm.load(params["attention_norm"])
        self.ffn.load(params["ffn"])
        self.ffn_norm.load(params["ffn_norm"])


# ══════════════════════════════════════════════════════════════════
# THE FULL MODEL
# ══════════════════════════════════════════════════════════════════

class AnadModel:
    """
    Anad Language Model.

    Decoder-only transformer.
    Same architecture as GPT, LLaMA, Mistral.
    Built from scratch. No corporate code inside.

    Flow:
      tokens → embeddings → N transformer layers → norm → logits
    """

    def __init__(self, config: AnadConfig):
        self.config = config

        # ── Token Embeddings ──
        # Maps token ids to vectors
        self.embedding = np.random.randn(
            config.vocab_size, config.dim
        ).astype(np.float32) * 0.02

        # ── Transformer Layers ──
        self.layers = [
            TransformerLayer(config, i)
            for i in range(config.n_layers)
        ]

        # ── Final Normalization ──
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # ── Output Projection ──
        # Maps dim back to vocabulary (logits)
        # Tied with embedding weights (saves parameters)
        # Same matrix as embedding, transposed
        self.output = self.embedding  # weight tying

        # ── Causal Mask ──
        # Upper triangle is -inf → can't look at future tokens
        self._causal_mask = self._build_causal_mask(config.max_seq_len)

        print(f"Anad model initialized")
        print(f"Config: {config.model_name}")
        print(f"Size: {config.param_count_approx}")
        print(f"Layers: {config.n_layers}")
        print(f"Heads: {config.n_heads} (kv: {config.n_kv_heads})")
        print(f"Context: {config.max_seq_len} tokens")

    def _build_causal_mask(self, seq_len: int) -> np.ndarray:
        """
        Build causal attention mask.

        Upper triangle = -1e9 (effectively -inf after softmax)
        Lower triangle = 0 (attend freely to past)

        This prevents the model from "cheating" by looking
        at future tokens during training.
        """
        mask = np.full((seq_len, seq_len), -1e9, dtype=np.float32)
        mask = np.tril(np.zeros_like(mask), k=0) + np.triu(mask, k=1)
        return mask

    def __call__(
        self,
        token_ids: np.ndarray,
        training: bool = False,
    ) -> np.ndarray:
        """
        Forward pass.

        token_ids: [batch, seq_len] integer token ids
        returns:   [batch, seq_len, vocab_size] logits
        """
        batch, seq_len = token_ids.shape

        # ── Embed tokens ──
        x = self.embedding[token_ids]  # [batch, seq_len, dim]

        # ── Get causal mask for this sequence length ──
        mask = self._causal_mask[:seq_len, :seq_len]

        # ── Pass through transformer layers ──
        for layer in self.layers:
            x = layer(x, mask, training)

        # ── Final normalization ──
        x = self.norm(x)

        # ── Project to vocabulary ──
        logits = x @ self.output.T  # [batch, seq_len, vocab_size]

        return logits

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Each step: predict next token → append → repeat

        temperature: controls randomness
          0.0 = always pick most likely (deterministic)
          1.0 = sample from full distribution
          >1.0 = more random

        top_p: nucleus sampling
          Only sample from tokens whose cumulative
          probability exceeds top_p. Improves quality.
        """
        generated = prompt_ids.copy().tolist()[0]

        for _ in range(max_new_tokens):
            # Get current sequence
            current = np.array([generated], dtype=np.int32)

            # Trim to max context if needed
            if current.shape[1] > self.config.max_seq_len:
                current = current[:, -self.config.max_seq_len:]

            # Forward pass
            logits = self(current, training=False)

            # Get logits for last token only
            next_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            
            # Softmax to probabilities
            next_logits = next_logits - next_logits.max()
            probs = np.exp(next_logits)
            probs = probs / probs.sum()

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                cumulative = np.cumsum(sorted_probs)
                # Remove tokens beyond top_p
                cutoff = np.searchsorted(cumulative, top_p) + 1
                sorted_probs[cutoff:] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                probs = np.zeros_like(probs)
                probs[sorted_indices] = sorted_probs

            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(int(next_token))

            # Stop at EOS
            if next_token == self.config.eos_token_id:
                break

        return np.array(generated)

    def save(self, path: str):
        """Save model weights to disk"""
        os.makedirs(path, exist_ok=True)

        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save embeddings
        np.save(os.path.join(path, "embedding.npy"), self.embedding)
        np.save(os.path.join(path, "norm_weight.npy"), self.norm.weight)

        # Save each layer
        for i, layer in enumerate(self.layers):
            layer_path = os.path.join(path, f"layer_{i:02d}")
            os.makedirs(layer_path, exist_ok=True)
            params = layer.parameters()

            np.save(f"{layer_path}/attn_wq.npy", params["attention"]["wq"])
            np.save(f"{layer_path}/attn_wk.npy", params["attention"]["wk"])
            np.save(f"{layer_path}/attn_wv.npy", params["attention"]["wv"])
            np.save(f"{layer_path}/attn_wo.npy", params["attention"]["wo"])
            np.save(f"{layer_path}/attn_norm.npy", params["attention_norm"]["weight"])
            np.save(f"{layer_path}/ffn_w1.npy", params["ffn"]["w1"])
            np.save(f"{layer_path}/ffn_w2.npy", params["ffn"]["w2"])
            np.save(f"{layer_path}/ffn_w3.npy", params["ffn"]["w3"])
            np.save(f"{layer_path}/ffn_norm.npy", params["ffn_norm"]["weight"])

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AnadModel":
        """Load model from disk"""
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)

        config = AnadConfig(**config_dict)
        model = cls(config)

        model.embedding = np.load(os.path.join(path, "embedding.npy"))
        model.output = model.embedding  # tied weights
        model.norm.weight = np.load(os.path.join(path, "norm_weight.npy"))

        for i, layer in enumerate(model.layers):
            layer_path = os.path.join(path, f"layer_{i:02d}")
            layer.attention.wq = np.load(f"{layer_path}/attn_wq.npy")
            layer.attention.wk = np.load(f"{layer_path}/attn_wk.npy")
            layer.attention.wv = np.load(f"{layer_path}/attn_wv.npy")
            layer.attention.wo = np.load(f"{layer_path}/attn_wo.npy")
            layer.attention_norm.weight = np.load(f"{layer_path}/attn_norm.npy")
            layer.ffn.w1 = np.load(f"{layer_path}/ffn_w1.npy")
            layer.ffn.w2 = np.load(f"{layer_path}/ffn_w2.npy")
            layer.ffn.w3 = np.load(f"{layer_path}/ffn_w3.npy")
            layer.ffn_norm.weight = np.load(f"{layer_path}/ffn_norm.npy")

        print(f"Model loaded from {path}")
        return model

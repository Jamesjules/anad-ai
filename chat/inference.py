"""
Anad Inference Engine
=====================
Loads trained PyTorch weights and generates responses.
This is what makes Anad actually talk.

Author: Anad Community
License: Public Domain
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ANAD_NANO, ANAD_SMALL, AnadConfig
from tokenizer.tokenizer import AnadTokenizer


class AnadInference:
    """
    Loads Anad trained weights and generates text.
    Finds the latest checkpoint automatically.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        device: str = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None
        self._loaded = False

    def load(self) -> bool:
        """Load latest checkpoint. Returns True if successful."""
        # Find latest checkpoint
        ckpt_path = self._find_latest_checkpoint()
        if not ckpt_path:
            print("  No checkpoint found. Run python train.py first.")
            return False

        # Load tokenizer
        tok_path = os.path.join(self.checkpoint_dir, "tokenizer")
        if not os.path.exists(os.path.join(tok_path, "vocab.json")):
            print("  No tokenizer found. Run python train.py first.")
            return False

        print(f"  Loading tokenizer...")
        self.tokenizer = AnadTokenizer.load(tok_path)

        # Detect config from checkpoint
        self.config = ANAD_NANO  # default

        # Build and load model
        print(f"  Loading model from {os.path.basename(ckpt_path)}...")
        self.model = self._build_model(self.config)
        ckpt = torch.load(
            os.path.join(ckpt_path, "model.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        step = ckpt.get("step", 0)
        n = sum(p.numel() for p in self.model.parameters())
        print(f"  Model ready — step {step}, {n/1e6:.1f}M params on {self.device.upper()}")
        self._loaded = True
        return True

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """
        Generate a response to a prompt.

        temperature: 0.1 = focused, 1.0 = creative
        top_p:       nucleus sampling threshold
        top_k:       only sample from top K tokens
        """
        if not self._loaded:
            return "Model not loaded. Run python train.py first."

        # Encode prompt
        try:
            prompt_ids = self.tokenizer.encode(prompt)
        except Exception:
            prompt_ids = [self.tokenizer.special_tokens["<BOS>"]]

        tokens = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)

        generated = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Trim context if too long
                if tokens.shape[1] > self.config.max_seq_len:
                    tokens = tokens[:, -self.config.max_seq_len:]

                logits = self.model(tokens)
                next_logits = logits[0, -1, :]  # last token logits

                # Temperature
                if temperature > 0:
                    next_logits = next_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < values[-1]] = float('-inf')

                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float('-inf')
                    next_logits = torch.zeros_like(next_logits).scatter_(
                        0, sorted_idx, sorted_logits
                    )

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Stop at EOS
                if next_token.item() == self.tokenizer.special_tokens.get("<EOS>", 3):
                    break

                generated.append(next_token.item())
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        # Decode
        try:
            return self.tokenizer.decode(generated).strip()
        except Exception:
            return "[could not decode response]"

    def _find_latest_checkpoint(self) -> str:
        """Find the most recent checkpoint directory"""
        if not os.path.exists(self.checkpoint_dir):
            return None

        checkpoints = sorted([
            d for d in os.listdir(self.checkpoint_dir)
            if d.startswith("checkpoint_step_")
            and os.path.exists(os.path.join(self.checkpoint_dir, d, "model.pt"))
        ])

        if not checkpoints:
            return None

        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def _build_model(self, cfg):
        """Build the PyTorch model architecture"""
        import torch.nn as nn

        class N(nn.Module):
            def __init__(self, d, e=1e-6):
                super().__init__()
                self.w = nn.Parameter(torch.ones(d))
                self.e = e
            def forward(self, x):
                return x / x.pow(2).mean(-1, keepdim=True).add(self.e).sqrt() * self.w

        class A(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.nh, self.nkv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
                self.g = c.n_heads // c.n_kv_heads
                self.wq = nn.Linear(c.dim, c.n_heads * c.head_dim, bias=False)
                self.wk = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
                self.wv = nn.Linear(c.dim, c.n_kv_heads * c.head_dim, bias=False)
                self.wo = nn.Linear(c.n_heads * c.head_dim, c.dim, bias=False)
            def forward(self, x, mask):
                B, T, _ = x.shape
                q = self.wq(x).view(B, T, self.nh,  self.hd).transpose(1, 2)
                k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2).repeat_interleave(self.g, 1)
                v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2).repeat_interleave(self.g, 1)
                a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd) + mask[:T, :T]
                return self.wo((F.softmax(a, -1) @ v).transpose(1, 2).reshape(B, T, -1))

        class FF(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.w1 = nn.Linear(c.dim, c.hidden_dim, bias=False)
                self.w2 = nn.Linear(c.hidden_dim, c.dim, bias=False)
                self.w3 = nn.Linear(c.dim, c.hidden_dim, bias=False)
            def forward(self, x):
                return self.w2(F.silu(self.w1(x)) * self.w3(x))

        class Block(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.a, self.f = A(c), FF(c)
                self.n1, self.n2 = N(c.dim), N(c.dim)
            def forward(self, x, mask):
                x = x + self.a(self.n1(x), mask)
                return x + self.f(self.n2(x))

        class Model(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.emb    = nn.Embedding(c.vocab_size, c.dim)
                self.layers = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
                self.norm   = N(c.dim)
                self.head   = nn.Linear(c.dim, c.vocab_size, bias=False)
                self.head.weight = self.emb.weight
                m = torch.full((c.max_seq_len, c.max_seq_len), float('-inf'))
                self.register_buffer('mask', torch.triu(m, 1))
            def forward(self, x):
                h = self.emb(x)
                for l in self.layers:
                    h = l(h, self.mask)
                return self.head(self.norm(h))

        return Model(cfg).to(self.device)

"""
Anad Model Tests
================
Tests every building block of the transformer.
Run this to verify the model architecture is correct.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import AnadConfig, ANAD_NANO
from model.model import (
    RMSNorm,
    build_rope_freqs,
    apply_rope,
    GroupedQueryAttention,
    FeedForward,
    TransformerLayer,
    AnadModel,
)


def print_section(title):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


def test_config():
    print_section("TEST 1 — Config & Model Sizes")

    from model.config import ANAD_NANO, ANAD_SMALL, ANAD_BASE

    for cfg in [ANAD_NANO, ANAD_SMALL, ANAD_BASE]:
        print(f"  {cfg.model_name:15} → {cfg.param_count_approx}")

    assert ANAD_NANO.head_dim == ANAD_NANO.dim // ANAD_NANO.n_heads
    print("✓ Config passed")


def test_rmsnorm():
    print_section("TEST 2 — RMS Normalization")

    norm = RMSNorm(dim=256)
    x = np.random.randn(2, 10, 256).astype(np.float32)
    out = norm(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output mean:  {out.mean():.4f} (should be ~0)")
    print("✓ RMSNorm passed")


def test_rope():
    print_section("TEST 3 — RoPE Positional Encoding")

    head_dim = 32
    max_seq = 128
    freqs = build_rope_freqs(head_dim, max_seq)

    print(f"  Frequency table shape: {freqs.shape}")
    assert freqs.shape == (max_seq, head_dim // 2)

    x = np.random.randn(1, 10, 4, head_dim).astype(np.float32)
    rotated = apply_rope(x, freqs)

    assert rotated.shape == x.shape
    print(f"  Input shape:   {x.shape}")
    print(f"  Rotated shape: {rotated.shape}")
    print("✓ RoPE passed")


def test_attention():
    print_section("TEST 4 — Grouped Query Attention")

    config = ANAD_NANO
    attn = GroupedQueryAttention(config)

    batch, seq_len = 2, 16
    x = np.random.randn(batch, seq_len, config.dim).astype(np.float32)
    out = attn(x, training=False)

    assert out.shape == (batch, seq_len, config.dim), \
        f"Wrong shape: {out.shape}"

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Query heads: {config.n_heads}")
    print(f"  KV heads:    {config.n_kv_heads}")
    print(f"  Groups:      {config.n_heads // config.n_kv_heads}")
    print("✓ Grouped Query Attention passed")


def test_feedforward():
    print_section("TEST 5 — Feed Forward (SwiGLU)")

    config = ANAD_NANO
    ffn = FeedForward(config)

    x = np.random.randn(2, 10, config.dim).astype(np.float32)
    out = ffn(x)

    assert out.shape == x.shape, f"Wrong shape: {out.shape}"
    print(f"  Input:      {x.shape}")
    print(f"  Output:     {out.shape}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Activation: SwiGLU")
    print("✓ FeedForward passed")


def test_transformer_layer():
    print_section("TEST 6 — Transformer Layer")

    config = ANAD_NANO
    layer = TransformerLayer(config, layer_idx=0)

    x = np.random.randn(1, 8, config.dim).astype(np.float32)
    out = layer(x, training=False)

    assert out.shape == x.shape
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Components: Attention + FFN + 2x RMSNorm + 2x Residual")
    print("✓ TransformerLayer passed")


def test_full_model():
    print_section("TEST 7 — Full Model Forward Pass")

    config = ANAD_NANO
    model = AnadModel(config)

    batch, seq_len = 2, 12
    token_ids = np.random.randint(0, config.vocab_size, (batch, seq_len))

    logits = model(token_ids, training=False)

    assert logits.shape == (batch, seq_len, config.vocab_size), \
        f"Wrong shape: {logits.shape}"

    print(f"\n  Input tokens: {token_ids.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    print("✓ Full model forward pass passed")


def test_generation():
    print_section("TEST 8 — Token Generation")

    config = ANAD_NANO
    model = AnadModel(config)

    prompt = np.array([[2, 100, 200, 300]])  # BOS + 3 tokens
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9,
    )

    assert len(generated) >= len(prompt[0])
    print(f"  Prompt tokens:    {len(prompt[0])}")
    print(f"  Generated tokens: {len(generated)}")
    print(f"  New tokens:       {len(generated) - len(prompt[0])}")
    print(f"  Token ids: {generated[:10]}...")
    print("✓ Generation passed")


def test_save_load():
    print_section("TEST 9 — Save and Load")

    config = ANAD_NANO
    model = AnadModel(config)

    save_path = "/tmp/anad_model_test"
    model.save(save_path)

    loaded = AnadModel.load(save_path)

    # Verify weights match
    token_ids = np.array([[2, 10, 20, 30]])
    out1 = model(token_ids)
    out2 = loaded(token_ids)

    assert np.allclose(out1, out2, atol=1e-5), "Outputs differ after load"
    print(f"  Saved and loaded successfully")
    print(f"  Output match: ✓")
    print("✓ Save/load passed")


def test_causal_mask():
    print_section("TEST 10 — Causal Mask")

    config = ANAD_NANO
    model = AnadModel(config)

    seq_len = 5
    mask = model._causal_mask[:seq_len, :seq_len]

    print(f"  Causal mask ({seq_len}x{seq_len}):")
    for row in mask:
        line = ""
        for val in row:
            line += "  0  " if val == 0 else " -inf"
        print(f"    {line}")

    # Lower triangle should be 0, upper -inf
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i, j] == 0, f"Expected 0 at [{i},{j}]"
            else:
                assert mask[i, j] < -100, f"Expected -inf at [{i},{j}]"

    print("✓ Causal mask passed")


def run_all_tests():
    print("\n" + "█" * 55)
    print("  ANAD MODEL TEST SUITE")
    print("  Transformer architecture from scratch")
    print("  No corporate code inside")
    print("█" * 55)

    try:
        test_config()
        test_rmsnorm()
        test_rope()
        test_attention()
        test_feedforward()
        test_transformer_layer()
        test_full_model()
        test_generation()
        test_save_load()
        test_causal_mask()

        print("\n" + "█" * 55)
        print("  ALL 10 TESTS PASSED ✓")
        print("  Anad model architecture is functional")
        print("█" * 55 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

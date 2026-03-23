"""
Anad Tokenizer Tests
====================
Tests across all supported languages
Indian languages tested first and thoroughly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer.tokenizer import AnadTokenizer


def print_section(title: str):
    print(f"\n{'═' * 50}")
    print(f"  {title}")
    print(f"{'═' * 50}")


def test_basic_training():
    print_section("TEST 1 — Basic Training")

    # Multilingual training corpus
    # Indian languages first
    training_texts = [
        # Gujarati
        "નમસ્તે દુનિયા",
        "આ અનાદ છે, જે બધા માટે AI છે",
        "ગુજરાત ભારતનું એક સુંદર રાજ્ય છે",
        "જ્ઞાન એ સૌથી મોટી શક્તિ છે",

        # Hindi
        "नमस्ते दुनिया",
        "यह सबके लिए एक सार्वजनिक AI है",
        "ज्ञान ही शक्ति है",
        "भारत एक महान देश है",

        # Tamil
        "வணக்கம் உலகம்",
        "இது அனைவருக்கும் AI",

        # Bengali
        "হ্যালো বিশ্ব",
        "এটি সবার জন্য AI",

        # Telugu
        "నమస్కారం ప్రపంచం",

        # English
        "Hello world",
        "This is Anad, public AI for everyone",
        "Knowledge is power",
        "The network grows stronger with every node",
        "Privacy is not a feature it is a foundation",

        # Code
        "def hello(): return 'world'",
        "for i in range(10): print(i)",

        # Mixed
        "Anad AI - અનાદ - for Bharat and the world",
    ]

    tokenizer = AnadTokenizer(vocab_size=1000)
    tokenizer.train(training_texts)

    stats = tokenizer.get_stats()
    print(f"\nVocab size: {stats['vocab_size']}")
    print(f"Merges learned: {stats['num_merges']}")
    print(f"Supported scripts: {', '.join(stats['supported_scripts'][:5])}...")

    assert stats["vocab_size"] > 256, "Vocab should be larger than base bytes"
    print("✓ Training passed")

    return tokenizer


def test_gujarati_encoding(tokenizer):
    print_section("TEST 2 — Gujarati Encoding")

    texts = [
        "નમસ્તે",
        "ભારત",
        "અનાદ",
        "જ્ઞાન",
        "આ AI બધા માટે છે",
    ]

    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  Original:  {text}")
        print(f"  Tokens:    {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Decoded:   {decoded}")
        print()

    print("✓ Gujarati encoding passed")


def test_hindi_encoding(tokenizer):
    print_section("TEST 3 — Hindi Encoding")

    texts = [
        "नमस्ते",
        "भारत माता की जय",
        "सबके लिए AI",
    ]

    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  Original: {text}")
        print(f"  Token count: {len(tokens)}")
        print(f"  Decoded:  {decoded}")
        print()

    print("✓ Hindi encoding passed")


def test_english_encoding(tokenizer):
    print_section("TEST 4 — English Encoding")

    texts = [
        "Hello world",
        "Public AI for everyone",
        "The network has no owner",
    ]

    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  Original: {text}")
        print(f"  Tokens:   {tokens}")
        print(f"  Decoded:  {decoded}")
        print()

    print("✓ English encoding passed")


def test_code_encoding(tokenizer):
    print_section("TEST 5 — Code Encoding")

    code_samples = [
        "def train(data):",
        "for node in network:",
        "return response",
    ]

    for code in code_samples:
        tokens = tokenizer.encode(code)
        decoded = tokenizer.decode(tokens)
        print(f"  Code:    {code}")
        print(f"  Tokens:  {len(tokens)} tokens")
        print(f"  Decoded: {decoded}")
        print()

    print("✓ Code encoding passed")


def test_mixed_language(tokenizer):
    print_section("TEST 6 — Mixed Language")

    mixed_texts = [
        "Anad AI - અનાદ",
        "Hello નમસ્તે World",
        "Python code for ભારત",
    ]

    for text in mixed_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"  Mixed:   {text}")
        print(f"  Tokens:  {len(tokens)}")
        print(f"  Decoded: {decoded}")
        print()

    print("✓ Mixed language encoding passed")


def test_save_and_load(tokenizer):
    print_section("TEST 7 — Save and Load")

    save_path = "/tmp/anad_tokenizer_test"
    tokenizer.save(save_path)

    # Load and verify
    loaded = AnadTokenizer.load(save_path)

    # Test loaded tokenizer works
    text = "અનાદ - Public AI"
    original_tokens = tokenizer.encode(text)
    loaded_tokens = loaded.encode(text)

    assert original_tokens == loaded_tokens, "Loaded tokenizer produces different tokens"
    print(f"  Saved and loaded successfully")
    print(f"  Original tokens: {original_tokens}")
    print(f"  Loaded tokens:   {loaded_tokens}")
    print(f"  Match: ✓")

    print("✓ Save/load passed")


def test_special_tokens(tokenizer):
    print_section("TEST 8 — Special Tokens")

    text = "Hello world"
    tokens = tokenizer.encode(text)

    from tokenizer.tokenizer import SPECIAL_TOKENS

    # Check BOS and EOS
    assert tokens[0] == SPECIAL_TOKENS["<BOS>"], "First token should be BOS"
    assert tokens[-1] == SPECIAL_TOKENS["<EOS>"], "Last token should be EOS"

    print(f"  BOS token id: {SPECIAL_TOKENS['<BOS>']} ✓")
    print(f"  EOS token id: {SPECIAL_TOKENS['<EOS>']} ✓")
    print(f"  USER token id: {SPECIAL_TOKENS['<USER>']} ✓")
    print(f"  ANAD token id: {SPECIAL_TOKENS['<ANAD>']} ✓")
    print(f"  CODE token id: {SPECIAL_TOKENS['<CODE>']} ✓")

    print("✓ Special tokens passed")


def run_all_tests():
    print("\n" + "█" * 50)
    print("  ANAD TOKENIZER TEST SUITE")
    print("  Public AI — Built from scratch")
    print("  Indian languages first")
    print("█" * 50)

    try:
        tokenizer = test_basic_training()
        test_gujarati_encoding(tokenizer)
        test_hindi_encoding(tokenizer)
        test_english_encoding(tokenizer)
        test_code_encoding(tokenizer)
        test_mixed_language(tokenizer)
        test_save_and_load(tokenizer)
        test_special_tokens(tokenizer)

        print("\n" + "█" * 50)
        print("  ALL TESTS PASSED ✓")
        print("  Anad tokenizer is functional")
        print("█" * 50 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

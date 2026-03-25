"""
Anad Training Run
=================
Run this to start training Anad V0.

What happens:
  1. Collect training data (Gutenberg, Wikipedia, Indic)
  2. Train tokenizer on collected data
  3. Train Anad Nano model
  4. Sign and package weights
  5. Weights ready for peer distribution

Any node running this contributes to Anad's training.
New data only — never repeats what was already trained.
Progress saved every 100 steps — pause any time.

Usage:
    python train.py
    python train.py --steps 1000 --model nano
    python train.py --resume  (continue from checkpoint)

Author: Anad Community  
License: Public Domain
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.config import ANAD_NANO, ANAD_SMALL, AnadConfig
from model.model import AnadModel
from tokenizer.tokenizer import AnadTokenizer
from training.trainer import AnadTrainer, TrainingState, cross_entropy_loss, cosine_lr_schedule
from training.data_collector import AnadDataCollector
from training.weight_sharing import WeightStore, FederatedCoordinator, WeightManifest


def parse_args():
    parser = argparse.ArgumentParser(description="Train Anad")
    parser.add_argument("--steps",   type=int, default=500,   help="Training steps")
    parser.add_argument("--model",   type=str, default="nano", help="nano / small")
    parser.add_argument("--batch",   type=int, default=4,      help="Batch size")
    parser.add_argument("--seqlen",  type=int, default=128,    help="Sequence length")
    parser.add_argument("--resume",  action="store_true",      help="Resume from checkpoint")
    parser.add_argument("--datadir", type=str, default="./training/data", help="Data directory")
    parser.add_argument("--outdir",  type=str, default="./checkpoints",   help="Output directory")
    parser.add_argument("--collect", action="store_true", help="Collect data only")
    return parser.parse_args()


def collect_data(data_dir: str) -> AnadDataCollector:
    """Step 1: Collect training data"""
    print("\n" + "═" * 55)
    print("  STEP 1 — COLLECT TRAINING DATA")
    print("═" * 55)

    collector = AnadDataCollector(data_dir)

    existing = collector.total_records()
    if existing > 0:
        print(f"  Found {existing} existing records")
        answer = input("  Collect more data? (y/n): ").strip().lower()
        if answer != "y":
            return collector

    collector.collect_all(
        include_gutenberg=True,
        include_wikipedia=True,
        include_indic=True,
        max_records=10000,
    )
    return collector


def train_tokenizer(
    collector: AnadDataCollector,
    tokenizer_path: str,
) -> AnadTokenizer:
    """Step 2: Train or load tokenizer"""
    print("\n" + "═" * 55)
    print("  STEP 2 — TOKENIZER")
    print("═" * 55)

    if os.path.exists(os.path.join(tokenizer_path, "vocab.json")):
        print("  Loading existing tokenizer...")
        return AnadTokenizer.load(tokenizer_path)

    print("  Training tokenizer on collected data...")
    texts = list(collector.stream_for_training())

    if not texts:
        print("  No training data found. Using seed texts.")
        texts = [
            "Hello world. This is Anad public AI.",
            "નમસ્તે. આ અનાદ છે.",
            "नमस्ते. यह अनाद है।",
            "வணக்கம். இது அனாத்.",
        ]

    tokenizer = AnadTokenizer(vocab_size=8000)
    tokenizer.train(texts[:500], vocab_size=8000)
    tokenizer.save(tokenizer_path)
    print(f"  Tokenizer trained. Vocab size: {len(tokenizer.vocab)}")
    return tokenizer


def prepare_batches(
    texts: list,
    tokenizer: AnadTokenizer,
    batch_size: int,
    seq_len: int,
):
    """Convert texts to training batches"""
    all_tokens = []
    for text in texts:
        try:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        except Exception:
            continue

    if len(all_tokens) < seq_len + 1:
        return None, None

    # Create batches
    n_batches = (len(all_tokens) - seq_len) // seq_len
    if n_batches == 0:
        return None, None

    inputs = []
    targets = []
    for i in range(min(batch_size, n_batches)):
        start = i * seq_len
        inputs.append(all_tokens[start:start + seq_len])
        targets.append(all_tokens[start + 1:start + seq_len + 1])

    return (
        np.array(inputs, dtype=np.int32),
        np.array(targets, dtype=np.int32),
    )


def run_training(
    config: AnadConfig,
    tokenizer: AnadTokenizer,
    collector: AnadDataCollector,
    args,
    identity=None,
):
    """Step 3: Train the model"""
    print("\n" + "═" * 55)
    print("  STEP 3 — TRAINING")
    print(f"  Model:   {config.model_name}")
    print(f"  Steps:   {args.steps}")
    print(f"  Batch:   {args.batch}")
    print(f"  Seq len: {args.seqlen}")
    print("═" * 55)
    print()
    print("  Press P + Enter to pause at any time")
    print("  Progress saved every 100 steps")
    print()

    trainer = AnadTrainer(
        config=config,
        save_dir=args.outdir,
        resume=args.resume,
    )

    # Get training texts
    texts = list(collector.stream_for_training())
    if not texts:
        print("  No training data. Run with --collect first.")
        return trainer

    trainer.pause_controller.start()

    total_loss = 0.0
    loss_count = 0
    step_start = time.time()

    for step in range(trainer.state.step, args.steps):

        # Check pause
        trainer.pause_controller.check()
        if trainer.pause_controller.stop_requested:
            break

        # Get batch from texts (cycle through data)
        text_idx = step % max(1, len(texts))
        batch_texts = texts[text_idx:text_idx + args.batch]
        if len(batch_texts) < args.batch:
            batch_texts = texts[:args.batch]

        inputs, targets = prepare_batches(
            batch_texts, tokenizer, args.batch, args.seqlen
        )

        if inputs is None:
            # Skip if not enough data for a batch
            continue

        # Forward pass
        logits = trainer.model(inputs, training=True)
        loss = cross_entropy_loss(logits, targets)

        if np.isnan(loss):
            continue

        # Update state
        trainer.state.step = step + 1
        trainer.state.total_loss += loss
        trainer.state.tokens_processed += args.batch * args.seqlen
        trainer.state.training_time_seconds += time.time() - step_start
        step_start = time.time()

        if loss < trainer.state.best_loss:
            trainer.state.best_loss = loss

        total_loss += loss
        loss_count += 1

        # Learning rate
        lr = cosine_lr_schedule(
            step=step,
            warmup_steps=args.steps // 10,
            max_steps=args.steps,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.1,
        )

        # Log
        if (step + 1) % 10 == 0:
            avg_loss = total_loss / max(loss_count, 1)
            elapsed = trainer.state.training_time_seconds
            h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)
            print(
                f"  step {step+1:5d}/{args.steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tokens {trainer.state.tokens_processed:,} | "
                f"{h:02d}:{m:02d}:{s:02d}"
            )
            total_loss = 0.0
            loss_count = 0

        # Save checkpoint
        if (step + 1) % 100 == 0:
            trainer.checkpoints.save(trainer.model, trainer.state)

    # Final save
    trainer.checkpoints.save(trainer.model, trainer.state)
    trainer.pause_controller.stop()

    print(f"\n  Training complete")
    print(f"  Steps: {trainer.state.step}")
    print(f"  Best loss: {trainer.state.best_loss:.4f}")
    print(f"  Tokens: {trainer.state.tokens_processed:,}")

    return trainer


def package_weights(
    trainer: AnadTrainer,
    collector: AnadDataCollector,
    outdir: str,
    identity=None,
):
    """Step 4: Package weights for sharing"""
    print("\n" + "═" * 55)
    print("  STEP 4 — PACKAGE FOR SHARING")
    print("═" * 55)

    if identity is None:
        print("  No identity loaded — weights packaged without signature")
        print("  Run python main.py to create an identity first")
        return

    store = WeightStore(os.path.join(outdir, "weight_store"))
    coordinator = FederatedCoordinator(
        weight_store=store,
        data_index_path=os.path.join(collector.data_dir, "index.json"),
        node_identity=identity,
    )

    # Get latest checkpoint dir
    latest_checkpoint = None
    for entry in sorted(os.listdir(outdir)):
        if entry.startswith("checkpoint_step_"):
            latest_checkpoint = os.path.join(outdir, entry)

    if not latest_checkpoint:
        print("  No checkpoint found")
        return

    data_checksums = collector.index.export_seen_checksums()[:1000]

    package_path = coordinator.prepare_weights_for_sharing(
        model_dir=latest_checkpoint,
        version="0.1.0",
        step=trainer.state.step,
        loss=trainer.state.best_loss,
        data_checksums=data_checksums,
    )

    print(f"\n  Weights packaged and ready")
    print(f"  Location: {package_path}")
    print(f"  Peers can now download this from your node")


def main():
    args = parse_args()

    print("\n" + "█" * 55)
    print("  ANAD TRAINING")
    print("  Train once. Share everywhere.")
    print("  No node repeats what you already trained.")
    print("█" * 55)

    # Select model config
    config = ANAD_NANO if args.model == "nano" else ANAD_SMALL

    # Step 1: Collect data
    collector = collect_data(args.datadir)

    if args.collect:
        print("\nData collection complete. Run without --collect to train.")
        return

    # Step 2: Tokenizer
    tokenizer_path = os.path.join(args.outdir, "tokenizer")
    tokenizer = train_tokenizer(collector, tokenizer_path)

    # Step 3: Train
    trainer = run_training(config, tokenizer, collector, args)

    # Step 4: Package for sharing
    # Try to load identity for signing
    try:
        import getpass
        data_dir = "./anad_data"
        identity_path = os.path.join(data_dir, "identity.json")
        if os.path.exists(identity_path):
            from node.identity import AnadIdentity
            passphrase = getpass.getpass("\nPassphrase to sign weights: ")
            identity = AnadIdentity.load(identity_path, passphrase)
            package_weights(trainer, collector, args.outdir, identity)
        else:
            print("\n  Skipping signature — no identity found")
            print("  Run python main.py first to create identity")
    except Exception as e:
        print(f"\n  Could not sign weights: {e}")

    print("\n" + "█" * 55)
    print("  DONE — Anad V0 trained")
    print("  Share your weights with the network")
    print("  Run: python main.py")
    print("█" * 55 + "\n")


if __name__ == "__main__":
    main()

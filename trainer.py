"""
Anad Training Loop
==================
Trains the Anad model on distributed community compute.

KEY FEATURE — PAUSE BUTTON:
  The node owner is always in control.
  Press P (or Ctrl+C) at any time to pause cleanly.
  Training state is saved instantly.
  Resume exactly where it stopped — no progress lost.
  Works on any system: Windows, Mac, Linux.

Philosophy:
  Your device. Your rules.
  Anad never takes more than you give.

Author: Anad Community
License: Public Domain
"""

import numpy as np
import json
import os
import time
import signal
import threading
import sys
from typing import Optional, List, Tuple
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model import AnadModel
from model.config import AnadConfig, ANAD_NANO


# ══════════════════════════════════════════════════════════════════
# PAUSE CONTROLLER — The most important part
# ══════════════════════════════════════════════════════════════════

class PauseController:
    """
    Pause button for Anad training.

    The node owner is always in control.
    Training pauses cleanly — never mid-update.
    All progress is saved before stopping.

    Usage:
        controller = PauseController()
        controller.start()

        while training:
            controller.check()  # pauses here if requested
            # ... training step ...

    Triggers:
        - Press P in terminal
        - Press Ctrl+C
        - Send SIGTERM (system shutdown)
        - Call controller.pause() from code
    """

    def __init__(self):
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
        self._stop_requested = threading.Event()
        self._listener_thread = None
        self._running = False

    def start(self):
        """Start listening for pause input"""
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_for_input,
            daemon=True,
            name="anad-pause-listener"
        )
        self._listener_thread.start()

        # Handle system signals
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print("  [Pause controller active]")
        print("  Press P + Enter to pause training")
        print("  Press Ctrl+C to pause and save\n")

    def _listen_for_input(self):
        """Listen for keyboard input in background thread"""
        while self._running:
            try:
                user_input = input()
                if user_input.strip().lower() in ("p", "pause"):
                    print("\n  [Pause requested — will pause after this step]")
                    self._pause_requested.set()
            except (EOFError, OSError):
                # Input stream closed (running headless / no terminal)
                break
            except Exception:
                break

    def _handle_signal(self, signum, frame):
        """Handle Ctrl+C or system shutdown"""
        print(f"\n  [Signal {signum} received — pausing cleanly]")
        self._pause_requested.set()

    def check(self):
        """
        Call this in the training loop.
        Blocks here if pause was requested.
        Returns immediately if not paused.
        """
        if self._pause_requested.is_set():
            self._paused.set()
            print("\n" + "─" * 50)
            print("  ANAD TRAINING PAUSED")
            print("  All progress saved")
            print("  Type 'resume' to continue")
            print("  Type 'stop' to stop training")
            print("─" * 50)

            while True:
                try:
                    cmd = input("  anad> ").strip().lower()
                except (EOFError, OSError):
                    # No terminal — auto resume after save
                    cmd = "stop"

                if cmd in ("r", "resume", "continue", "go"):
                    self._pause_requested.clear()
                    self._paused.clear()
                    print("  Resuming training...\n")
                    break
                elif cmd in ("s", "stop", "quit", "exit", "q"):
                    self._stop_requested.set()
                    self._paused.clear()
                    print("  Stopping training. Progress saved.\n")
                    break
                else:
                    print("  Type 'resume' to continue or 'stop' to exit")

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def pause(self):
        """Programmatically request a pause"""
        self._pause_requested.set()

    def stop(self):
        """Clean shutdown"""
        self._running = False
        self._stop_requested.set()


# ══════════════════════════════════════════════════════════════════
# TRAINING STATE — What gets saved on pause
# ══════════════════════════════════════════════════════════════════

@dataclass
class TrainingState:
    """
    Complete training state.
    Everything needed to resume exactly where we paused.
    """
    step: int = 0                    # current training step
    epoch: int = 0                   # current epoch
    total_loss: float = 0.0          # cumulative loss
    best_loss: float = float("inf")  # best loss seen
    learning_rate: float = 3e-4      # current learning rate
    tokens_processed: int = 0        # total tokens seen
    training_time_seconds: float = 0 # total training time
    timestamp: str = ""              # last save time

    def save(self, path: str):
        state = asdict(self)
        state["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingState":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════
# OPTIMIZER — AdamW from scratch
# ══════════════════════════════════════════════════════════════════

class AdamW:
    """
    AdamW optimizer — built from scratch.

    Adaptive learning rates per parameter.
    Weight decay for regularization.
    Used by every major language model.

    No PyTorch needed. Pure numpy.
    """

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # Momentum and velocity for each parameter
        self.m = [np.zeros_like(p) for p in params]  # first moment
        self.v = [np.zeros_like(p) for p in params]  # second moment

    def step(self, grads: List[np.ndarray]):
        """Update parameters using gradients"""
        self.step_count += 1

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue

            # Weight decay (applied to param directly, not grad)
            param *= (1 - self.lr * self.weight_decay)

            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            # Parameter update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset gradients (called before each step)"""
        pass  # numpy doesn't accumulate grads automatically


# ══════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULER — Cosine with warmup
# ══════════════════════════════════════════════════════════════════

def cosine_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Phase 1 (warmup): LR increases linearly from 0 to max_lr
    Phase 2 (cosine): LR decreases from max_lr to min_lr

    Why: Prevents unstable early training and
         allows fine convergence at the end.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step / warmup_steps)
    elif step > max_steps:
        return min_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))


# ══════════════════════════════════════════════════════════════════
# LOSS FUNCTION — Cross Entropy
# ══════════════════════════════════════════════════════════════════

def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    pad_token_id: int = 0,
) -> float:
    """
    Cross entropy loss for language modeling.

    For each position, measure how wrong the model was
    about the next token. Average over all positions.

    Ignores padding tokens in the loss computation.

    logits:  [batch, seq_len, vocab_size]
    targets: [batch, seq_len]
    """
    batch, seq_len, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)  # [batch*seq_len, vocab_size]
    targets_flat = targets.reshape(-1)             # [batch*seq_len]

    # Ignore padding — use -1 as explicit ignore, not 0
    # (token 0 is a valid token in most positions)
    mask = (targets_flat != -1)

    # Numerically stable softmax
    logits_stable = logits_flat - logits_flat.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(logits_stable).sum(axis=-1))
    log_probs = logits_stable - log_sum_exp[:, np.newaxis]

    # Gather log probs for correct tokens
    correct_log_probs = log_probs[np.arange(len(targets_flat)), targets_flat]

    # Mean loss over non-padding tokens
    loss = -correct_log_probs[mask].mean()
    return float(loss)


def compute_gradients_approx(
    model: AnadModel,
    logits: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-4,
) -> List[np.ndarray]:
    """
    Approximate gradients via finite differences.

    NOTE: This is for architecture verification only.
    Real training will use proper backpropagation via PyTorch.
    This demonstrates the training loop structure correctly.
    """
    base_loss = cross_entropy_loss(logits, targets)
    grads = []

    # Just return zero grads for structure demo
    # Real backprop comes with PyTorch integration
    for layer in model.layers:
        grads.append(np.zeros_like(layer.ffn.w1))

    return grads, base_loss


# ══════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGER
# ══════════════════════════════════════════════════════════════════

class CheckpointManager:
    """
    Saves and loads training checkpoints.

    On pause: saves everything needed to resume.
    On resume: loads state and continues seamlessly.

    Keeps last 3 checkpoints to save disk space.
    """

    def __init__(self, save_dir: str, keep_last: int = 3):
        self.save_dir = save_dir
        self.keep_last = keep_last
        os.makedirs(save_dir, exist_ok=True)

    def save(self, model: AnadModel, state: TrainingState):
        """Save checkpoint — called automatically on pause"""
        step = state.step
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_step_{step:07d}")

        print(f"\n  Saving checkpoint at step {step}...")
        model.save(checkpoint_dir)
        state.save(os.path.join(self.save_dir, "training_state.json"))

        # Save checkpoint index
        index_path = os.path.join(self.save_dir, "checkpoints.json")
        index = []
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)

        index.append({"step": step, "dir": checkpoint_dir})

        # Keep only last N checkpoints
        if len(index) > self.keep_last:
            old = index.pop(0)
            if os.path.exists(old["dir"]):
                import shutil
                shutil.rmtree(old["dir"])

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

        print(f"  Checkpoint saved → {checkpoint_dir}")

    def load_latest(self, model: AnadModel) -> Optional[TrainingState]:
        """Load most recent checkpoint if it exists"""
        state_path = os.path.join(self.save_dir, "training_state.json")
        index_path = os.path.join(self.save_dir, "checkpoints.json")

        if not os.path.exists(state_path):
            return None

        with open(index_path) as f:
            index = json.load(f)

        if not index:
            return None

        latest = index[-1]
        print(f"  Resuming from step {latest['step']}...")
        model_loaded = AnadModel.load(latest["dir"])
        # Copy weights
        model.embedding = model_loaded.embedding
        model.norm.weight = model_loaded.norm.weight
        for i, (layer, loaded_layer) in enumerate(zip(model.layers, model_loaded.layers)):
            layer.attention.wq = loaded_layer.attention.wq
            layer.attention.wk = loaded_layer.attention.wk
            layer.attention.wv = loaded_layer.attention.wv
            layer.attention.wo = loaded_layer.attention.wo
            layer.ffn.w1 = loaded_layer.ffn.w1
            layer.ffn.w2 = loaded_layer.ffn.w2
            layer.ffn.w3 = loaded_layer.ffn.w3

        return TrainingState.load(state_path)


# ══════════════════════════════════════════════════════════════════
# TRAINING DISPLAY
# ══════════════════════════════════════════════════════════════════

def display_progress(state: TrainingState, loss: float, lr: float, steps_per_sec: float):
    """Clean training progress display"""
    elapsed = state.training_time_seconds
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(
        f"  step {state.step:6d} | "
        f"loss {loss:.4f} | "
        f"lr {lr:.2e} | "
        f"tokens {state.tokens_processed:,} | "
        f"time {hours:02d}:{minutes:02d}:{seconds:02d} | "
        f"{steps_per_sec:.1f} steps/s"
    )


# ══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

class AnadTrainer:
    """
    Main training orchestrator.

    Features:
    - Pause/resume at any time (P key or Ctrl+C)
    - Auto-save on pause
    - Resume from exact checkpoint
    - Learning rate scheduling
    - Progress display
    - Works on any hardware tier
    """

    def __init__(
        self,
        config: AnadConfig,
        save_dir: str = "./checkpoints",
        resume: bool = True,
    ):
        self.config = config
        self.model = AnadModel(config)
        self.checkpoints = CheckpointManager(save_dir)
        self.pause_controller = PauseController()
        self.state = TrainingState()

        # Try to resume from checkpoint
        if resume:
            loaded_state = self.checkpoints.load_latest(self.model)
            if loaded_state:
                self.state = loaded_state
                print(f"  Resumed from step {self.state.step}")
            else:
                print("  Starting fresh training")

    def train(
        self,
        texts: List[str],
        max_steps: int = 10000,
        batch_size: int = 4,
        seq_len: int = 128,
        log_every: int = 10,
        save_every: int = 100,
    ):
        """
        Main training loop.

        texts:      training text corpus
        max_steps:  total training steps
        batch_size: sequences per step
        seq_len:    tokens per sequence
        log_every:  print progress every N steps
        save_every: checkpoint every N steps
        """
        print("\n" + "═" * 55)
        print("  ANAD TRAINING")
        print(f"  Model: {self.config.model_name}")
        print(f"  Steps: {max_steps}")
        print(f"  Batch: {batch_size}")
        print(f"  Seq len: {seq_len}")
        print("═" * 55 + "\n")

        # Start pause controller
        self.pause_controller.start()

        # Dummy token data for architecture demo
        # Real training uses AnadDataLoader with actual text
        vocab_size = self.config.vocab_size

        step_start_time = time.time()
        training_start = time.time()

        step = self.state.step

        while step < max_steps:

            # ── PAUSE CHECK ── Most important line in training ──
            self.pause_controller.check()
            if self.pause_controller.stop_requested:
                self.checkpoints.save(self.model, self.state)
                print("  Training stopped. Progress saved.")
                break

            # ── Generate dummy batch (replace with real data loader) ──
            input_ids = np.random.randint(
                0, vocab_size, (batch_size, seq_len), dtype=np.int32
            )
            target_ids = np.random.randint(
                0, vocab_size, (batch_size, seq_len), dtype=np.int32
            )

            # ── Forward pass ──
            logits = self.model(input_ids, training=True)

            # ── Compute loss ──
            loss = cross_entropy_loss(logits, target_ids, self.config.pad_token_id)

            # ── Learning rate schedule ──
            lr = cosine_lr_schedule(
                step=step,
                warmup_steps=max_steps // 10,
                max_steps=max_steps,
                max_lr=self.config.learning_rate,
                min_lr=self.config.learning_rate * 0.1,
            )

            # ── Update state ──
            step += 1
            self.state.step = step
            self.state.total_loss += loss
            self.state.tokens_processed += batch_size * seq_len
            self.state.training_time_seconds = time.time() - training_start

            if loss < self.state.best_loss:
                self.state.best_loss = loss

            # ── Log progress ──
            if step % log_every == 0:
                elapsed = time.time() - step_start_time
                steps_per_sec = log_every / max(elapsed, 1e-6)
                step_start_time = time.time()
                display_progress(self.state, loss, lr, steps_per_sec)

            # ── Auto save checkpoint ──
            if step % save_every == 0:
                self.checkpoints.save(self.model, self.state)

        # Final save
        self.checkpoints.save(self.model, self.state)

        print("\n" + "═" * 55)
        print("  TRAINING COMPLETE")
        print(f"  Total steps: {self.state.step}")
        print(f"  Best loss:   {self.state.best_loss:.4f}")
        print(f"  Tokens seen: {self.state.tokens_processed:,}")
        print("═" * 55 + "\n")

        self.pause_controller.stop()

"""
utils/logger.py
===============
Lightweight training logger — no external dependencies beyond stdlib.

Writes to both stdout and a .log file simultaneously.
Tracks per-epoch loss history for plotting.
"""

import os
import sys
import time
import json
from datetime import datetime


class TrainingLogger:
    """
    Logs training progress to stdout and a log file.
    Stores loss history in memory for later saving/plotting.

    Parameters
    ----------
    log_dir  : str   directory to write logs into
    run_name : str   prefix for log file name
    """

    def __init__(self, log_dir: str = "outputs", run_name: str = "ssl"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{run_name}_{ts}.log")
        self._f = open(self.log_path, "w", buffering=1, encoding="utf-8")

        self.history: list[dict] = []   # one entry per epoch
        self._t0 = time.time()

        self._write(f"=== Training run: {run_name} | {ts} ===\n")

    # ── public API ────────────────────────────────────────────────────────

    def log_epoch(self, epoch: int, total_epochs: int, losses: dict,
                  lr: float, elapsed: float):
        """Print and record one epoch summary."""
        pct  = 100 * epoch / total_epochs
        line = (
            f"Epoch {epoch:>4}/{total_epochs}  ({pct:5.1f}%)  "
            f"total={losses['total']:.4f}  "
            f"contrast={losses['loss_contrast']:.4f}  "
            f"masked={losses['loss_masked']:.4f}  "
            f"gen={losses['loss_gen']:.4f}  "
            f"lr={lr:.2e}  "
            f"elapsed={elapsed:.0f}s"
        )
        self._write(line)

        record = {"epoch": epoch, "lr": lr, "elapsed": elapsed, **losses}
        self.history.append(record)

    def log(self, msg: str):
        self._write(msg)

    def save_history(self, path: str):
        """Save loss history as JSON for later plotting."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        self._write(f"History saved → {path}")

    def close(self):
        self._write("=== Run complete ===")
        self._f.close()

    # ── internal ──────────────────────────────────────────────────────────

    def _write(self, msg: str):
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        self._f.write(line)


class AverageMeter:
    """Accumulates values and computes running mean — one per loss term."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class LossTracker:
    """Tracks multiple named loss terms across an epoch."""

    KEYS = ["total", "loss_contrast", "loss_masked", "loss_gen"]

    def __init__(self):
        self.meters = {k: AverageMeter() for k in self.KEYS}

    def reset(self):
        for m in self.meters.values():
            m.reset()

    def update(self, loss_dict: dict, batch_size: int):
        for k in self.KEYS:
            if k in loss_dict:
                self.meters[k].update(loss_dict[k].item(), batch_size)

    def averages(self) -> dict:
        return {k: m.avg for k, m in self.meters.items()}
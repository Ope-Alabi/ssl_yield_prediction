"""
ssl/ssl_trainer.py
==================
SSL Pretraining — Step 3

Trains the SSLModel (encoder + 3 heads) using the combined loss:
  L = w_contrast * NT-Xent  +  w_masked * MSE_masked  +  w_gen * MSE_gen

Training features
-----------------
- Cosine annealing LR schedule with linear warm-up
- Best-model checkpointing (by total loss)
- Last-epoch checkpointing (for resuming)
- Per-epoch loss logging to stdout + .log file
- Loss history saved as JSON at the end

Usage
-----
    python src/ssl/ssl_trainer.py
    python src/ssl/ssl_trainer.py --resume          # resume from last.pt
    python src/ssl/ssl_trainer.py --epochs 100
"""

import os
import sys
import time
import argparse
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_split
from src.ssl.model_factory import build_ssl_model
from src.utils.logger import TrainingLogger, LossTracker
from src.utils.checkpoint import save_checkpoint, load_checkpoint


# ── Default training config ───────────────────────────────────────────────

TRAIN_CFG = {
    "epochs":        50,
    "batch_size":    256,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "warmup_epochs": 5,       # linear LR warm-up before cosine decay
    "num_workers":   0,       # set >0 if on Linux with multiple CPU cores
    "splits_dir":    "data/splits",
    "ckpt_dir":      "outputs/checkpoints",
    "log_dir":       "outputs",
    "seed":          42,
}

SSL_CFG = {
    "encoder": {"n_features": 5, "d_model": 64, "n_heads": 4,
                "n_layers": 3, "d_ff": 128, "dropout": 0.1,
                "max_len": 24, "pool": "mean"},
    "heads":   {"proj_dim": 32, "mask_ratio": 0.25,
                "n_gen_steps": 12, "gen_hidden": 64},
    "loss":    {"w_contrast": 1.0, "w_masked": 1.0,
                "w_gen": 0.5, "temperature": 0.07},
}


# ── LR schedule: linear warm-up + cosine decay ────────────────────────────

def get_lr(epoch: int, cfg: dict, base_lr: float) -> float:
    """Returns the learning rate for the given epoch."""
    warmup = cfg["warmup_epochs"]
    total  = cfg["epochs"]

    if epoch < warmup:
        # Linear ramp from base_lr/10 → base_lr
        return base_lr * (0.1 + 0.9 * epoch / warmup)
    else:
        # Cosine decay from base_lr → base_lr * 0.01
        progress = (epoch - warmup) / max(1, total - warmup)
        return base_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    tracker = LossTracker()

    for anchor, positive in loader:
        anchor   = anchor.to(device)
        positive = positive.to(device)

        optimizer.zero_grad()
        outputs = model(anchor, positive)
        losses  = criterion(outputs)

        losses["total"].backward()

        # Gradient clipping — prevents occasional spikes from contrastive loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        tracker.update(losses, batch_size=anchor.size(0))

    return tracker.averages()


def pretrain(cfg: dict = TRAIN_CFG, ssl_cfg: dict = SSL_CFG,
             resume: bool = False):

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = TrainingLogger(cfg["log_dir"], run_name="ssl_pretrain")
    logger.log(f"Device: {device}")
    logger.log(f"Training config: {cfg}")

    # ── Data ────────────────────────────────────────────────────────────
    dataset = load_split(cfg["splits_dir"], split="pretrain", mode="pretrain")
    loader  = DataLoader(
        dataset,
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg["num_workers"],
        pin_memory  = device.type == "cuda",
        drop_last   = True,   # keeps batch size uniform for NT-Xent
    )
    logger.log(f"Pretrain windows : {len(dataset):,}")
    logger.log(f"Batches per epoch: {len(loader)}")

    # ── Model ────────────────────────────────────────────────────────────
    model, criterion = build_ssl_model(ssl_cfg)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Model parameters : {total_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        betas        = (0.9, 0.999),
    )

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss   = float("inf")
    last_ckpt   = os.path.join(cfg["ckpt_dir"], "last.pt")

    if resume and os.path.exists(last_ckpt):
        start_epoch, best_loss = load_checkpoint(
            last_ckpt, model, optimizer
        )
        start_epoch += 1
        logger.log(f"Resumed from epoch {start_epoch-1} | best loss {best_loss:.4f}")

    # ── Training loop ────────────────────────────────────────────────────
    logger.log(f"\nStarting pretraining for {cfg['epochs']} epochs …\n")
    t_start = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # Manually set LR (cleaner than a scheduler for warm-up + cosine)
        current_lr = get_lr(epoch, cfg, cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        t_epoch = time.time()
        losses  = train_one_epoch(model, loader, optimizer, criterion, device)
        elapsed = time.time() - t_start

        logger.log_epoch(epoch, cfg["epochs"], losses, current_lr, elapsed)

        # ── Checkpointing ─────────────────────────────────────────────
        # Always save last
        save_checkpoint(
            os.path.join(cfg["ckpt_dir"], "last.pt"),
            model, optimizer, None, epoch, losses["total"], ssl_cfg,
        )

        # Save best
        if losses["total"] < best_loss:
            best_loss = losses["total"]
            save_checkpoint(
                os.path.join(cfg["ckpt_dir"], "best.pt"),
                model, optimizer, None, epoch, best_loss, ssl_cfg,
            )
            logger.log(f"  ★ New best loss: {best_loss:.4f}  → best.pt saved")

    # ── Save loss history ─────────────────────────────────────────────────
    history_path = os.path.join(cfg["log_dir"], "ssl_loss_history.json")
    logger.save_history(history_path)
    logger.log(f"\n✅ Pretraining complete. Best loss: {best_loss:.4f}")
    logger.log(f"   Encoder weights → {cfg['ckpt_dir']}/best.pt")
    logger.close()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Pretraining")
    parser.add_argument("--epochs",     type=int,   default=TRAIN_CFG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=TRAIN_CFG["batch_size"])
    parser.add_argument("--lr",         type=float, default=TRAIN_CFG["lr"])
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    cfg = {**TRAIN_CFG, "epochs": args.epochs,
           "batch_size": args.batch_size, "lr": args.lr}

    pretrain(cfg=cfg, ssl_cfg=SSL_CFG, resume=args.resume)
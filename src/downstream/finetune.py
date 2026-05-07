"""
downstream/finetune.py
======================
Step 4 — Downstream Fine-tuning for Yield Prediction

Loads the pretrained encoder from best.pt, attaches a regression
head, and trains on labeled yield data.

Two-phase training strategy
---------------------------
Phase 1 (epochs 1 → freeze_epochs):
  Encoder frozen  — only the regression head trains.
  Higher LR (head_lr). Quickly learns to map representations → yield.

Phase 2 (epochs freeze_epochs+1 → total_epochs):
  Encoder unfrozen — full end-to-end fine-tuning.
  Lower LR for encoder (enc_lr), higher for head (head_lr).
  Differential learning rates via param groups.

Metrics reported each epoch
---------------------------
  MAE   — mean absolute error in raw yield units (g or kg)
  RMSE  — root mean squared error in raw yield units
  R²    — coefficient of determination

Usage
-----
    python src/downstream/finetune.py
    python src/downstream/finetune.py --freeze_only   # Phase 1 only
    python src/downstream/finetune.py --no_side       # no tabular features
"""

import os
import sys
import time
import argparse
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.encoder import ClimateEncoder
from src.downstream.yield_model import YieldPredictor
from src.downstream.finetune_dataset import load_finetune_datasets
from src.utils.logger import TrainingLogger, AverageMeter
from src.utils.checkpoint import load_checkpoint, save_checkpoint


# ── Config ────────────────────────────────────────────────────────────────

FINETUNE_CFG = {
    # paths
    "pretrained_ckpt": "outputs/checkpoints/best.pt",
    "splits_dir":      "data/splits",
    "metadata_path":   "data/processed/metadata.csv",
    "ckpt_dir":        "outputs/checkpoints",
    "log_dir":         "outputs",

    # data
    "use_side":        True,    # include zone/recipe/age side features
    "normalize_y":     True,

    # training phases
    "total_epochs":    40,
    "freeze_epochs":   10,      # Phase 1: head-only training
    "batch_size":      128,
    "num_workers":     0,

    # learning rates
    "head_lr":         1e-3,    # regression head LR (both phases)
    "enc_lr":          1e-5,    # encoder LR (Phase 2 only, much lower)
    "weight_decay":    1e-4,

    # head architecture
    "head_hidden":     64,
    "head_dropout":    0.2,

    # encoder config (must match pretraining)
    "d_model":         64,
    "seed":            42,
}

SSL_ENCODER_CFG = {
    "n_features": 5, "d_model": 64, "n_heads": 4,
    "n_layers": 3,   "d_ff": 128,   "dropout": 0.1,
    "max_len": 24,   "pool": "mean",
}


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    y_std: float, y_mean: float) -> dict:
    """All metrics in raw yield units."""
    p = preds  * y_std + y_mean
    t = targets * y_std + y_mean

    mae  = (p - t).abs().mean().item()
    rmse = ((p - t) ** 2).mean().sqrt().item()
    ss_res = ((t - p) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item()
    r2   = 1 - ss_res / (ss_tot + 1e-8)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ── Train / eval loops ────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion,
              device, y_std, y_mean, train: bool):
    model.train() if train else model.eval()

    loss_meter = AverageMeter()
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if len(batch) == 3:
                x, side, y = batch
                side = side.to(device)
            else:
                x, y = batch
                side = None

            x, y = x.to(device), y.to(device)

            pred = model(x, side)
            loss = criterion(pred, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            all_preds.append(pred.detach().cpu())
            all_targets.append(y.detach().cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(preds, targets, y_std, y_mean)
    metrics["loss"] = loss_meter.avg
    return metrics


# ── Main fine-tuning function ─────────────────────────────────────────────

def finetune(cfg: dict = FINETUNE_CFG, freeze_only: bool = False):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = TrainingLogger(cfg["log_dir"], run_name="finetune")
    logger.log(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    ds_train, ds_test, y_mean, y_std, n_side = load_finetune_datasets(
        cfg["splits_dir"],
        cfg["metadata_path"],
        use_side    = cfg["use_side"],
        normalize_y = cfg["normalize_y"],
    )

    loader_train = DataLoader(ds_train, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=cfg["num_workers"],
                              drop_last=False)
    loader_test  = DataLoader(ds_test,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])

    logger.log(f"Train windows : {len(ds_train):,}")
    logger.log(f"Test  windows : {len(ds_test):,}")
    logger.log(f"Side features : {n_side}  (y_mean={y_mean:.1f}, y_std={y_std:.1f})")

    # ── Load pretrained encoder ──────────────────────────────────────────
    encoder = ClimateEncoder(**SSL_ENCODER_CFG)

    if os.path.exists(cfg["pretrained_ckpt"]):
        ckpt    = torch.load(cfg["pretrained_ckpt"], map_location="cpu")
        enc_state = {k.replace("encoder.", "", 1): v
                     for k, v in ckpt["model"].items()
                     if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state, strict=True)
        logger.log(f"Pretrained encoder loaded from '{cfg['pretrained_ckpt']}'")
    else:
        logger.log("WARNING: pretrained checkpoint not found — training from scratch")

    # ── Build downstream model ───────────────────────────────────────────
    model = YieldPredictor(
        encoder        = encoder,
        d_model        = cfg["d_model"],
        n_side         = n_side,
        head_hidden    = cfg["head_hidden"],
        head_dropout   = cfg["head_dropout"],
        freeze_encoder = True,          # always start frozen
    ).to(device)

    criterion = nn.MSELoss()

    # ── Phase 1: frozen encoder, head only ───────────────────────────────
    logger.log("\n--- Phase 1: Frozen encoder (head only) ---")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["head_lr"], weight_decay=cfg["weight_decay"],
    )

    best_mae  = float("inf")
    t_start   = time.time()
    phase1_epochs = cfg["freeze_epochs"] if not freeze_only else cfg["total_epochs"]

    for epoch in range(1, phase1_epochs + 1):
        tr = run_epoch(model, loader_train, optimizer, criterion,
                       device, y_std, y_mean, train=True)
        te = run_epoch(model, loader_test,  optimizer, criterion,
                       device, y_std, y_mean, train=False)

        elapsed = time.time() - t_start
        logger.log(
            f"Epoch {epoch:>3}/{phase1_epochs} [frozen]  "
            f"train_loss={tr['loss']:.4f}  "
            f"MAE={te['mae']:.1f}  RMSE={te['rmse']:.1f}  R2={te['r2']:.4f}  "
            f"elapsed={elapsed:.0f}s"
        )

        if te["mae"] < best_mae:
            best_mae = te["mae"]
            save_checkpoint(
                os.path.join(cfg["ckpt_dir"], "finetune_best.pt"),
                model, optimizer, None, epoch, best_mae, cfg,
            )
            logger.log(f"  ** Best MAE: {best_mae:.2f} -> finetune_best.pt saved")

    if freeze_only:
        logger.log(f"\nPhase 1 only. Best test MAE: {best_mae:.2f}")
        logger.close()
        return

    # ── Phase 2: full fine-tuning with differential LR ───────────────────
    logger.log("\n--- Phase 2: Full fine-tuning (encoder unfrozen) ---")
    model.unfreeze_encoder()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg["enc_lr"]},
            {"params": model.head.parameters(),    "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )

    # Cosine decay for Phase 2
    phase2_epochs = cfg["total_epochs"] - cfg["freeze_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase2_epochs, eta_min=cfg["enc_lr"] * 0.01
    )

    for epoch in range(1, phase2_epochs + 1):
        tr = run_epoch(model, loader_train, optimizer, criterion,
                       device, y_std, y_mean, train=True)
        te = run_epoch(model, loader_test,  optimizer, criterion,
                       device, y_std, y_mean, train=False)
        scheduler.step()

        enc_lr_now  = optimizer.param_groups[0]["lr"]
        head_lr_now = optimizer.param_groups[1]["lr"]
        elapsed     = time.time() - t_start

        logger.log(
            f"Epoch {epoch:>3}/{phase2_epochs} [full]    "
            f"train_loss={tr['loss']:.4f}  "
            f"MAE={te['mae']:.1f}  RMSE={te['rmse']:.1f}  R2={te['r2']:.4f}  "
            f"enc_lr={enc_lr_now:.1e}  head_lr={head_lr_now:.1e}  "
            f"elapsed={elapsed:.0f}s"
        )

        if te["mae"] < best_mae:
            best_mae = te["mae"]
            save_checkpoint(
                os.path.join(cfg["ckpt_dir"], "finetune_best.pt"),
                model, optimizer, scheduler, epoch, best_mae, cfg,
            )
            logger.log(f"  ** Best MAE: {best_mae:.2f} -> finetune_best.pt saved")

    logger.log(f"\nFine-tuning complete. Best test MAE: {best_mae:.2f}")
    logger.close()


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yield Prediction Fine-tuning")
    parser.add_argument("--freeze_only", action="store_true",
                        help="Run Phase 1 (frozen encoder) only")
    parser.add_argument("--no_side",     action="store_true",
                        help="Disable tabular side features")
    parser.add_argument("--total_epochs", type=int,
                        default=FINETUNE_CFG["total_epochs"])
    parser.add_argument("--freeze_epochs", type=int,
                        default=FINETUNE_CFG["freeze_epochs"])
    args = parser.parse_args()

    cfg = {**FINETUNE_CFG,
           "use_side":      not args.no_side,
           "total_epochs":  args.total_epochs,
           "freeze_epochs": args.freeze_epochs}

    finetune(cfg=cfg, freeze_only=args.freeze_only)
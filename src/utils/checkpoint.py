"""
utils/checkpoint.py
===================
Save and load model checkpoints during SSL pretraining.

Saves
-----
  - model state dict
  - optimizer state dict
  - scheduler state dict
  - epoch number
  - best loss so far
  - full config dict

Two checkpoint files are maintained:
  last.pt   — overwritten every epoch (resume training)
  best.pt   — overwritten only when total loss improves
"""

import os
import torch


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    cfg: dict,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch":          epoch,
            "loss":           loss,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict() if scheduler else None,
            "cfg":            cfg,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    """
    Load checkpoint. Returns (epoch, best_loss).
    optimizer / scheduler state are restored in-place if provided.
    """
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt["epoch"], ckpt["loss"]


def load_encoder_weights(encoder_path: str, model):
    """
    Load only the encoder weights from a saved SSL checkpoint.
    Used when transferring to the downstream fine-tuning model.
    """
    ckpt = torch.load(encoder_path, map_location="cpu")
    state = ckpt["model"]

    # Filter: keep only keys that belong to the encoder sub-module
    enc_state = {
        k.replace("encoder.", "", 1): v
        for k, v in state.items()
        if k.startswith("encoder.")
    }
    model.load_state_dict(enc_state, strict=True)
    print(f"✅ Encoder weights loaded from '{encoder_path}'")
    return model
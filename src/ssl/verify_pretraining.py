"""
verify_pretraining.py
=====================
Runs 3 mini-batches through the full training pipeline to confirm
everything works before committing to a full training run.

Run from the project root:
    python src/ssl/verify_pretraining.py

Checks
------
1. DataLoader returns correct shapes and dtype
2. Full forward pass through SSLModel
3. Loss computation for all three heads
4. Backward pass — gradients computed
5. Optimizer step — weights actually change
6. LR schedule — warm-up and cosine values are correct
7. Checkpoint save/load round-trip
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import copy
import math
import torch
from torch.utils.data import DataLoader

from src.data.dataset import load_split
from src.ssl.model_factory import build_ssl_model
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.ssl.ssl_trainer import get_lr, TRAIN_CFG, SSL_CFG


SEP = lambda t="": print(f"\n── {t} {'─'*(44-len(t))}")


def main():
    print("=" * 50)
    print("  Step 3 — Pretraining Setup Verification")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── 1. DataLoader ────────────────────────────────────────────────────
    SEP("1. DataLoader")
    dataset = load_split("data/splits", split="pretrain", mode="pretrain")
    loader  = DataLoader(dataset, batch_size=32, shuffle=True,
                         num_workers=0, drop_last=True)

    anchor, positive = next(iter(loader))
    print(f"  anchor shape   : {list(anchor.shape)}")    # [32, 24, 5]
    print(f"  positive shape : {list(positive.shape)}")  # [32, 24, 5]
    assert anchor.shape == positive.shape == torch.Size([32, 24, 5])
    assert anchor.dtype == torch.float32
    print(f"  ✅  DataLoader OK  ({len(dataset):,} samples, "
          f"{len(loader)} batches @ bs=32)")

    # ── 2. Forward pass ──────────────────────────────────────────────────
    SEP("2. Forward pass")
    model, criterion = build_ssl_model(SSL_CFG)
    model = model.to(device)
    anchor, positive = anchor.to(device), positive.to(device)

    outputs = model(anchor, positive)
    print(f"  z_anchor       : {list(outputs['z_anchor'].shape)}")
    print(f"  z_pos          : {list(outputs['z_pos'].shape)}")
    print(f"  masked_preds   : {list(outputs['masked_preds'].shape)}")
    print(f"  gen_preds      : {list(outputs['gen_preds'].shape)}")
    print("  ✅  Forward pass OK")

    # ── 3. Loss ──────────────────────────────────────────────────────────
    SEP("3. Loss computation")
    losses = criterion(outputs)
    for k, v in losses.items():
        print(f"  {k:<20} {v.item():.6f}")
    assert all(v.item() > 0 for v in losses.values()), "All losses must be > 0"
    print("  ✅  All losses positive")

    # ── 4. Backward ──────────────────────────────────────────────────────
    SEP("4. Backward pass")
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    none_grads = sum(1 for g in grads if g is None)
    print(f"  Params with grad : {len(grads) - none_grads}/{len(grads)}")
    assert none_grads == 0, f"{none_grads} parameters have no gradient"
    max_grad = max(g.abs().max().item() for g in grads)
    print(f"  Max gradient     : {max_grad:.4f}")
    print("  ✅  Gradients OK")

    # ── 5. Optimizer step ────────────────────────────────────────────────
    SEP("5. Optimizer step")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    params_before = copy.deepcopy(
        [p.data.clone() for p in model.parameters()]
    )
    # Re-run forward/backward (grads were zeroed by deepcopy side-effect)
    model.zero_grad()
    outputs = model(anchor, positive)
    losses  = criterion(outputs)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    changed = sum(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(params_before, model.parameters())
    )
    print(f"  Parameter tensors updated : {changed}/{len(params_before)}")
    assert changed > 0, "Optimizer step did not change any weights"
    print("  ✅  Optimizer step OK")

    # ── 6. LR schedule ───────────────────────────────────────────────────
    SEP("6. LR schedule")
    base_lr = TRAIN_CFG["lr"]
    cfg     = TRAIN_CFG
    lrs     = [(e, get_lr(e, cfg, base_lr)) for e in
               [1, cfg["warmup_epochs"], cfg["warmup_epochs"]+1,
                cfg["epochs"]//2, cfg["epochs"]]]
    for epoch, lr in lrs:
        print(f"  epoch {epoch:>4}  →  lr = {lr:.2e}")

    # Warm-up: epoch 1 should be < base_lr
    assert get_lr(1, cfg, base_lr) < base_lr, "Warm-up should start below base_lr"
    # After warm-up: should be at or near base_lr
    assert get_lr(cfg["warmup_epochs"], cfg, base_lr) <= base_lr
    # End: should be much lower than base_lr
    assert get_lr(cfg["epochs"], cfg, base_lr) < base_lr * 0.1
    print("  ✅  LR schedule OK")

    # ── 7. Checkpoint round-trip ─────────────────────────────────────────
    SEP("7. Checkpoint save/load")
    ckpt_path = "outputs/checkpoints/_verify_test.pt"
    save_checkpoint(ckpt_path, model, optimizer, None,
                    epoch=1, loss=losses["total"].item(), cfg=SSL_CFG)
    assert os.path.exists(ckpt_path), "Checkpoint file not created"
    print(f"  Saved  → {ckpt_path}  ({os.path.getsize(ckpt_path)//1024} KB)")

    model2, _ = build_ssl_model(SSL_CFG)
    loaded_epoch, loaded_loss = load_checkpoint(ckpt_path, model2)
    assert loaded_epoch == 1
    print(f"  Loaded → epoch={loaded_epoch}  loss={loaded_loss:.6f}")

    # Weights match?
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert torch.allclose(p1.cpu(), p2.cpu()), f"Mismatch at {n1}"
    print("  ✅  Checkpoint weights match")

    # Clean up test checkpoint
    os.remove(ckpt_path)

    # ── Summary ──────────────────────────────────────────────────────────
    SEP("Summary")
    print("  All checks passed ✅")
    print("\n  To start pretraining, run:")
    print("    python src/ssl/ssl_trainer.py")
    print("    python src/ssl/ssl_trainer.py --epochs 100  --batch_size 512")
    print("    python src/ssl/ssl_trainer.py --resume      # continue from last.pt")
    print("\n  After training, plot loss curves:")
    print("    python src/utils/plot_losses.py")


if __name__ == "__main__":
    main()
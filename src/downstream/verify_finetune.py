"""
downstream/verify_finetune.py
==============================
Pre-flight checks for Step 4 fine-tuning.
Run from the project root:
    python src/downstream/verify_finetune.py

Checks
------
1. Finetune / test datasets load correctly
2. Side features encode to correct shape and valid ranges
3. Pretrained encoder loads from best.pt
4. YieldPredictor forward pass (frozen + unfrozen)
5. Loss + backward pass
6. Optimizer step changes head weights but NOT encoder (Phase 1)
7. Phase 2: encoder weights also change after unfreeze
8. Metric computation returns sensible values
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader

from src.models.encoder import ClimateEncoder
from src.downstream.yield_model import YieldPredictor
from src.downstream.finetune_dataset import load_finetune_datasets
from src.downstream.finetune import compute_metrics, SSL_ENCODER_CFG, FINETUNE_CFG
import copy

SEP = lambda t="": print(f"\n-- {t} {'-'*(44-len(t))}")


def main():
    print("=" * 50)
    print("  Step 4 -- Fine-tuning Setup Verification")
    print("=" * 50)

    cfg    = FINETUNE_CFG
    device = torch.device("cpu")

    # ── 1. Dataset ───────────────────────────────────────────────────────
    SEP("1. Dataset loading")
    ds_train, ds_test, y_mean, y_std, n_side = load_finetune_datasets(
        cfg["splits_dir"], cfg["metadata_path"],
        use_side=True, normalize_y=True,
    )
    print(f"  Train : {len(ds_train):,} windows")
    print(f"  Test  : {len(ds_test):,} windows")
    print(f"  n_side: {n_side}   y_mean={y_mean:.1f}   y_std={y_std:.1f}")
    assert len(ds_train) > 0 and len(ds_test) > 0
    print("  OK  Dataset sizes valid")

    # ── 2. Side feature shape + range ────────────────────────────────────
    SEP("2. Side features")
    loader = DataLoader(ds_train, batch_size=16, shuffle=False)
    x, side, y = next(iter(loader))
    print(f"  x shape    : {list(x.shape)}")      # [16, 24, 5]
    print(f"  side shape : {list(side.shape)}")   # [16, 14]
    print(f"  y shape    : {list(y.shape)}")       # [16,]
    assert x.shape == torch.Size([16, 24, 5])
    assert side.shape == torch.Size([16, n_side])
    assert side.min() >= 0.0 and side.max() <= 1.0, "Side features out of [0,1]"
    print(f"  side range : [{side.min():.2f}, {side.max():.2f}]")
    print("  OK  Side features valid")

    # ── 3. Load pretrained encoder ────────────────────────────────────────
    SEP("3. Pretrained encoder")
    encoder = ClimateEncoder(**SSL_ENCODER_CFG)
    ckpt_path = cfg["pretrained_ckpt"]

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        enc_state = {k.replace("encoder.", "", 1): v
                     for k, v in ckpt["model"].items()
                     if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state, strict=True)
        print(f"  Loaded from : {ckpt_path}")
        print(f"  Saved epoch : {ckpt['epoch']}")
        print(f"  Saved loss  : {ckpt['loss']:.4f}")
        print("  OK  Encoder weights loaded")
    else:
        print(f"  WARNING: {ckpt_path} not found -- using random weights")
        print("  (Run ssl_trainer.py first for best results)")

    # ── 4. YieldPredictor forward pass ────────────────────────────────────
    SEP("4. YieldPredictor forward pass")
    model = YieldPredictor(
        encoder=encoder, d_model=64, n_side=n_side,
        head_hidden=64, head_dropout=0.2, freeze_encoder=True,
    )

    pred = model(x, side)
    print(f"  Output shape (frozen) : {list(pred.shape)}")  # [16,]
    assert pred.shape == torch.Size([16])
    print("  OK  Forward pass (frozen encoder)")

    model.unfreeze_encoder()
    pred2 = model(x, side)
    assert pred2.shape == torch.Size([16])
    print("  OK  Forward pass (unfrozen encoder)")
    model._freeze_encoder()   # re-freeze for next checks

    # ── 5. Loss + backward ────────────────────────────────────────────────
    SEP("5. Loss + backward")
    criterion = torch.nn.MSELoss()
    pred  = model(x, side)
    loss  = criterion(pred, y)
    loss.backward()
    print(f"  MSE loss : {loss.item():.6f}")
    assert loss.item() > 0
    print("  OK  Loss is positive")

    # ── 6. Phase 1: optimizer only updates head ───────────────────────────
    SEP("6. Phase 1 optimizer (head only)")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    enc_before  = copy.deepcopy([p.data.clone() for p in model.encoder.parameters()])
    head_before = copy.deepcopy([p.data.clone() for p in model.head.parameters()])

    model.zero_grad()
    loss = criterion(model(x, side), y)
    loss.backward()
    optimizer.step()

    enc_changed  = sum(not torch.equal(a, b)
                       for a, b in zip(enc_before,  model.encoder.parameters()))
    head_changed = sum(not torch.equal(a, b)
                       for a, b in zip(head_before, model.head.parameters()))
    print(f"  Encoder tensors changed : {enc_changed}  (should be 0)")
    print(f"  Head tensors changed    : {head_changed}  (should be >0)")
    assert enc_changed  == 0,  "Encoder should NOT change during Phase 1"
    assert head_changed  > 0,  "Head SHOULD change during Phase 1"
    print("  OK  Phase 1 gradient isolation correct")

    # ── 7. Phase 2: encoder also updates ─────────────────────────────────
    SEP("7. Phase 2 optimizer (full fine-tuning)")
    model.unfreeze_encoder()
    optimizer2 = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(),    "lr": 1e-3},
    ])
    enc_before2 = [p.data.clone() for p in model.encoder.parameters()]

    model.zero_grad()
    loss = criterion(model(x, side), y)
    loss.backward()
    optimizer2.step()

    enc_changed2 = sum(not torch.equal(a, b)
                       for a, b in zip(enc_before2, model.encoder.parameters()))
    print(f"  Encoder tensors changed : {enc_changed2}  (should be >0)")
    assert enc_changed2 > 0, "Encoder SHOULD change during Phase 2"
    print("  OK  Phase 2 differential LR correct")

    # ── 8. Metrics ────────────────────────────────────────────────────────
    SEP("8. Metric computation")
    dummy_preds   = torch.zeros(100)    # predicting mean → R2 near 0
    dummy_targets = torch.randn(100)
    m = compute_metrics(dummy_preds, dummy_targets, y_std=y_std, y_mean=y_mean)
    print(f"  MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  R2={m['r2']:.4f}")
    assert "mae" in m and "rmse" in m and "r2" in m
    print("  OK  Metrics compute without error")

    # ── Summary ───────────────────────────────────────────────────────────
    SEP("Summary")
    print("  All checks passed!")
    print("\n  To run fine-tuning:")
    print("    python src/downstream/finetune.py")
    print("    python src/downstream/finetune.py --freeze_only")
    print("    python src/downstream/finetune.py --no_side")


if __name__ == "__main__":
    main()
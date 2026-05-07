"""
verify_architecture.py
======================
Smoke-tests the full SSL architecture without any training.
Run from the project root:  python src/ssl/verify_architecture.py

Checks
------
1. ClimateEncoder forward pass (pooled + sequence modes)
2. Each SSL head forward pass independently
3. Full SSLModel forward pass
4. SSLLoss computation
5. Parameter counts
6. Backward pass (gradients flow to all components)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from src.models.encoder import ClimateEncoder, count_parameters
from src.ssl.heads import (
    ContrastiveHead, MaskedPredictionHead,
    GenerativeHead, SSLModel,
)
from src.ssl.losses import SSLLoss
from src.ssl.model_factory import build_ssl_model

# ── Config (matches ssl_config.yaml defaults) ─────────────────────────────
CFG = {
    "encoder": {"n_features": 5, "d_model": 64, "n_heads": 4,
                "n_layers": 3, "d_ff": 128, "dropout": 0.1,
                "max_len": 24, "pool": "mean"},
    "heads":   {"proj_dim": 32, "mask_ratio": 0.25,
                "n_gen_steps": 12, "gen_hidden": 64},
    "loss":    {"w_contrast": 1.0, "w_masked": 1.0,
                "w_gen": 0.5, "temperature": 0.07},
}

BATCH = 8
T, F  = 24, 5


def sep(title=""):
    print(f"\n── {title} {'─'*(44-len(title))}")


def check_shape(name, tensor, expected):
    ok = list(tensor.shape) == list(expected)
    status = "✅" if ok else f"❌ got {list(tensor.shape)}"
    print(f"  {status}  {name}: {list(tensor.shape)}")
    assert ok, f"Shape mismatch for {name}"


def main():
    print("=" * 50)
    print("  Step 2 — Architecture Verification")
    print("=" * 50)

    dummy_anchor   = torch.randn(BATCH, T, F)
    dummy_positive = torch.randn(BATCH, T, F)

    # ── 1. Encoder ───────────────────────────────────────────────────────
    sep("1. ClimateEncoder")
    enc = ClimateEncoder(**CFG["encoder"])

    pooled = enc(dummy_anchor)
    check_shape("pooled output", pooled, [BATCH, 64])

    seq = enc(dummy_anchor, return_sequence=True)
    check_shape("sequence output", seq, [BATCH, T, 64])
    print(f"  Encoder params: {count_parameters(enc):,}")

    # ── 2. Contrastive head ──────────────────────────────────────────────
    sep("2. ContrastiveHead")
    cont_head = ContrastiveHead(d_model=64, proj_dim=32)
    z = cont_head(pooled)
    check_shape("projection", z, [BATCH, 32])
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), \
        "Projections are not L2-normalised"
    print("  ✅  L2 normalisation: OK")
    print(f"  Head params: {count_parameters(cont_head):,}")

    # ── 3. Masked prediction head ────────────────────────────────────────
    sep("3. MaskedPredictionHead")
    mask_head = MaskedPredictionHead(d_model=64, n_features=F)
    mask = torch.zeros(BATCH, T, dtype=torch.bool)
    mask[:, :6] = True                     # pretend first 6 timesteps masked
    preds = mask_head(seq, mask)
    M = mask.sum().item()
    check_shape("masked preds", preds, [M, F])
    print(f"  Head params: {count_parameters(mask_head):,}")

    # ── 4. Generative head ───────────────────────────────────────────────
    sep("4. GenerativeHead")
    gen_head = GenerativeHead(d_model=64, n_features=F, n_steps=12, hidden_dim=64)
    z_ctx     = enc(dummy_anchor[:, :12, :])        # encode first 12 steps
    future    = dummy_anchor[:, 12:, :]             # target: last 12 steps
    gen_preds = gen_head(z_ctx, target=future)      # teacher forcing
    check_shape("gen preds (teacher forcing)", gen_preds, [BATCH, 12, F])

    gen_free = gen_head(z_ctx, target=None)         # free-run
    check_shape("gen preds (free-run)", gen_free, [BATCH, 12, F])
    print(f"  Head params: {count_parameters(gen_head):,}")

    # ── 5. Full SSLModel + SSLLoss ───────────────────────────────────────
    sep("5. Full SSLModel + SSLLoss")
    model, criterion = build_ssl_model(CFG)
    outputs = model(dummy_anchor, dummy_positive)

    check_shape("z_anchor",       outputs["z_anchor"],       [BATCH, 32])
    check_shape("z_pos",          outputs["z_pos"],          [BATCH, 32])
    check_shape("gen_preds",      outputs["gen_preds"],       [BATCH, 12, F])
    check_shape("gen_targets",    outputs["gen_targets"],     [BATCH, 12, F])
    M2 = outputs["masked_targets"].size(0)
    check_shape("masked_preds",   outputs["masked_preds"],   [M2, F])

    losses = criterion(outputs)
    print(f"\n  Loss breakdown:")
    for k, v in losses.items():
        print(f"    {k:<18} {v.item():.6f}")
    assert losses["total"].item() > 0, "Total loss should be > 0"
    print("  ✅  All losses are positive scalars")

    # ── 6. Backward pass ─────────────────────────────────────────────────
    sep("6. Backward pass (gradient check)")
    losses["total"].backward()
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    if no_grad:
        print(f"  ⚠️  {len(no_grad)} params have no gradient: {no_grad[:3]}…")
    else:
        print(f"  ✅  Gradients flow to all {count_parameters(model):,} parameters")

    # ── Summary ──────────────────────────────────────────────────────────
    sep("Summary")
    print(f"  Total SSL model params : {count_parameters(model):,}")
    print(f"  Encoder only           : {count_parameters(model.encoder):,}")
    print(f"  Contrastive head       : {count_parameters(model.contrastive_head):,}")
    print(f"  Masked head            : {count_parameters(model.masked_head):,}")
    print(f"  Generative head        : {count_parameters(model.generative_head):,}")
    print("\n✅  Step 2 complete. Ready for Step 3 (SSL Pretraining).")


if __name__ == "__main__":
    main()
"""
model_factory.py
================
Single entry point to build the full SSL model + loss from config.

Usage
-----
    from src.ssl.model_factory import build_ssl_model

    model, criterion = build_ssl_model(cfg)
"""

from src.models.encoder import ClimateEncoder
from src.ssl.heads import (
    ContrastiveHead,
    MaskedPredictionHead,
    GenerativeHead,
    SSLModel,
)
from src.ssl.losses import SSLLoss


def build_ssl_model(cfg: dict):
    """
    Build SSLModel and SSLLoss from a config dictionary.

    Expected cfg keys (all optional — defaults shown):
    ──────────────────────────────────────────────────
    encoder:
      n_features : 5
      d_model    : 64
      n_heads    : 4
      n_layers   : 3
      d_ff       : 128
      dropout    : 0.1
      max_len    : 24
      pool       : 'mean'

    heads:
      proj_dim   : 32     (contrastive projection dimension)
      mask_ratio : 0.25   (fraction of timesteps to mask)
      n_gen_steps: 12     (= T // 2, steps to predict)
      gen_hidden : 64     (GRU hidden size)

    loss:
      w_contrast  : 1.0
      w_masked    : 1.0
      w_gen       : 0.5
      temperature : 0.07

    Returns
    -------
    model     : SSLModel
    criterion : SSLLoss
    """
    enc_cfg  = cfg.get("encoder", {})
    head_cfg = cfg.get("heads",   {})
    loss_cfg = cfg.get("loss",    {})

    # ── Encoder ────────────────────────────────────────────────────────
    encoder = ClimateEncoder(
        n_features = enc_cfg.get("n_features", 5),
        d_model    = enc_cfg.get("d_model",    64),
        n_heads    = enc_cfg.get("n_heads",    4),
        n_layers   = enc_cfg.get("n_layers",   3),
        d_ff       = enc_cfg.get("d_ff",       128),
        dropout    = enc_cfg.get("dropout",    0.1),
        max_len    = enc_cfg.get("max_len",    24),
        pool       = enc_cfg.get("pool",       "mean"),
    )

    d_model    = enc_cfg.get("d_model",    64)
    n_features = enc_cfg.get("n_features", 5)

    # ── Heads ──────────────────────────────────────────────────────────
    contrastive_head = ContrastiveHead(
        d_model  = d_model,
        proj_dim = head_cfg.get("proj_dim", 32),
    )
    masked_head = MaskedPredictionHead(
        d_model    = d_model,
        n_features = n_features,
    )
    generative_head = GenerativeHead(
        d_model    = d_model,
        n_features = n_features,
        n_steps    = head_cfg.get("n_gen_steps", 12),
        hidden_dim = head_cfg.get("gen_hidden",  64),
    )

    # ── Combined model ─────────────────────────────────────────────────
    model = SSLModel(
        encoder          = encoder,
        contrastive_head = contrastive_head,
        masked_head      = masked_head,
        generative_head  = generative_head,
        mask_ratio       = head_cfg.get("mask_ratio", 0.25),
    )

    # ── Loss ────────────────────────────────────────────────────────────
    criterion = SSLLoss(
        w_contrast  = loss_cfg.get("w_contrast",  1.0),
        w_masked    = loss_cfg.get("w_masked",    1.0),
        w_gen       = loss_cfg.get("w_gen",       0.5),
        temperature = loss_cfg.get("temperature", 0.07),
    )

    return model, criterion
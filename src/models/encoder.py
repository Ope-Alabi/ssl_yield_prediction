"""
ClimateEncoder — Transformer backbone for SSL pretraining
==========================================================
Input  : (batch, T=24, F=5)   — normalized climate windows
Output : (batch, d_model)      — fixed-size latent representation

Architecture
------------
1. Linear input projection  : F → d_model
2. Learnable positional enc  : (T, d_model)
3. Transformer encoder       : n_layers × (MultiHeadAttention + FFN)
4. Pooling                   : mean over time → (batch, d_model)

Why Transformer over TCN?
  - Attention naturally captures long-range climate dependencies
    (e.g. morning humidity affecting afternoon VPD)
  - Compatible with the masked prediction SSL head (BERT-style masking)
  - Small enough (default ~200K params) to train on CPU if needed
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    Preferred over sinusoidal for short sequences (T=24) —
    the model can learn that hour-12 matters differently than hour-1.
    """

    def __init__(self, d_model: int, max_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Learnable: shape (1, max_len, d_model) — broadcast over batch
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ClimateEncoder(nn.Module):
    """
    Parameters
    ----------
    n_features  : int   number of climate input features (default 5)
    d_model     : int   hidden dimension throughout the encoder (default 64)
    n_heads     : int   attention heads — must divide d_model (default 4)
    n_layers    : int   transformer encoder layers (default 3)
    d_ff        : int   feedforward inner dimension (default 128)
    dropout     : float dropout rate (default 0.1)
    max_len     : int   maximum sequence length (default 24)
    pool        : str   'mean' | 'cls'
                        'mean'  → average all timestep outputs
                        'cls'   → prepend a [CLS] token and use its output
    """

    def __init__(
        self,
        n_features: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_len: int = 24,
        pool: str = "mean",
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert pool in ("mean", "cls"), "pool must be 'mean' or 'cls'"

        self.d_model = d_model
        self.pool = pool

        # ── 1. Input projection ──────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 2. Optional CLS token ────────────────────────────────────────
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            effective_len = max_len + 1
        else:
            effective_len = max_len

        # ── 3. Positional encoding ───────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, effective_len, dropout)

        # ── 4. Transformer layers ────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # input shape: (B, T, d_model)
            norm_first=True,    # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x               : (B, T, F)   input climate windows
        return_sequence : bool
            If True  → return full token sequence (B, T', d_model)
                        needed by the masked prediction head
            If False → return pooled representation  (B, d_model)
                        needed by the contrastive & generative heads

        Returns
        -------
        torch.Tensor
        """
        B, T, _ = x.shape

        # Project to d_model
        h = self.input_proj(x)          # (B, T, d_model)

        # Prepend CLS token if using cls pooling
        if self.pool == "cls":
            cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
            h = torch.cat([cls, h], dim=1)            # (B, T+1, d_model)

        # Add positional encoding
        h = self.pos_enc(h)             # (B, T', d_model)

        # Transformer
        h = self.transformer(h)         # (B, T', d_model)

        if return_sequence:
            return h                    # (B, T', d_model)

        # Pool to fixed-size vector
        if self.pool == "cls":
            return h[:, 0, :]           # (B, d_model)  — CLS token
        else:
            return h.mean(dim=1)        # (B, d_model)  — mean pooling

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() with return_sequence=False. Use at inference."""
        return self.forward(x, return_sequence=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick model summary ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ClimateEncoder()
    dummy = torch.randn(8, 24, 5)          # batch=8, T=24, F=5

    rep = model(dummy)                      # pooled  (8, 64)
    seq = model(dummy, return_sequence=True)  # sequence (8, 24, 64)

    print("=" * 45)
    print("  ClimateEncoder — model summary")
    print("=" * 45)
    print(f"  Input shape       : {list(dummy.shape)}")
    print(f"  Pooled output     : {list(rep.shape)}")
    print(f"  Sequence output   : {list(seq.shape)}")
    print(f"  Trainable params  : {count_parameters(model):,}")
    print("=" * 45)
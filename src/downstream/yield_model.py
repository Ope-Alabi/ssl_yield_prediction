"""
downstream/yield_model.py
=========================
Yield prediction model for Step 4 fine-tuning.

Architecture
------------
  PretrainedEncoder (frozen or unfrozen)
       ↓
  Pooled representation  (B, d_model)
       ↓
  Regression head MLP
       ↓
  Predicted yield  (B, 1)

Two fine-tuning strategies are supported:

  'frozen'   — encoder weights are locked; only the head trains.
               Fast, few labeled samples needed, lower ceiling.

  'full'     — encoder is unfrozen after a warm-up period and
               fine-tuned end-to-end with a lower LR.
               Slower, but higher accuracy when data allows.

The model also accepts optional tabular side-features
(zone, fert_recipe, age_of_crop) concatenated to the encoder
output before the regression head.
"""

import torch
import torch.nn as nn
from src.models.encoder import ClimateEncoder


class YieldRegressionHead(nn.Module):
    """
    MLP regression head.

    Parameters
    ----------
    in_dim   : int   input dimension (d_model + n_side_features)
    hidden   : int   hidden layer size
    dropout  : float dropout rate
    """

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


class YieldPredictor(nn.Module):
    """
    Full downstream model: encoder + optional side features + regression head.

    Parameters
    ----------
    encoder        : ClimateEncoder   pretrained encoder (loaded from best.pt)
    d_model        : int              encoder output dimension
    n_side         : int              number of tabular side features (0 = none)
    head_hidden    : int              regression head hidden dim
    head_dropout   : float
    freeze_encoder : bool             if True, encoder weights are frozen
    """

    def __init__(
        self,
        encoder: ClimateEncoder,
        d_model: int = 64,
        n_side: int = 0,
        head_hidden: int = 64,
        head_dropout: float = 0.2,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder       = encoder
        self.freeze_encoder = freeze_encoder
        self.n_side        = n_side

        if freeze_encoder:
            self._freeze_encoder()

        self.head = YieldRegressionHead(
            in_dim  = d_model + n_side,
            hidden  = head_hidden,
            dropout = head_dropout,
        )

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        """Call this to switch from frozen → full fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.freeze_encoder = False

    def forward(
        self,
        x: torch.Tensor,
        side: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, T, F)    climate window
        side : (B, n_side)  optional tabular features

        Returns
        -------
        pred : (B,)         predicted yield
        """
        z = self.encoder(x)              # (B, d_model)

        if self.n_side > 0 and side is not None:
            z = torch.cat([z, side], dim=-1)   # (B, d_model + n_side)

        return self.head(z)              # (B,)
"""
SSL Loss Functions
==================
Three losses combined into one weighted total.

Loss 1 — NT-Xent (contrastive)
    Pulls anchor and positive together, pushes all other pairs apart.
    Temperature-scaled cosine similarity on the unit sphere.

Loss 2 — MSE (masked prediction)
    Reconstruction error at masked timestep positions only.

Loss 3 — MSE (generative)
    Step-wise prediction error over the second half of each window.

Total loss
----------
    L = w_contrast * L_contrast
      + w_masked   * L_masked
      + w_gen      * L_gen

Default weights: 1.0 / 1.0 / 0.5
  The generative head is weighted lower because teacher forcing during
  training makes it an easier task than contrastive or masked prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    For a batch of B pairs (z_i, z_j):
      - 2B vectors total
      - For each z_i, its positive is z_j (the matching augmented view)
      - All other 2B-2 vectors are negatives

    Parameters
    ----------
    temperature : float   sharpness of the distribution (default 0.07)
                          lower → harder negatives weighted more
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j : (B, proj_dim)  L2-normalized projections
        """
        B    = z_i.size(0)
        device = z_i.device

        # Concatenate both views: (2B, proj_dim)
        z    = torch.cat([z_i, z_j], dim=0)

        # Cosine similarity matrix: (2B, 2B)
        sim  = torch.mm(z, z.T) / self.temperature

        # Mask self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=device)
        sim  = sim.masked_fill(mask, -1e9)

        # Positive pairs:  (i, i+B) and (i+B, i)
        pos_idx = torch.arange(B, device=device)
        labels  = torch.cat([pos_idx + B, pos_idx])   # (2B,)

        loss = F.cross_entropy(sim, labels)
        return loss


class MaskedPredictionLoss(nn.Module):
    """MSE between predicted and true feature values at masked positions."""

    def forward(
        self,
        preds: torch.Tensor,    # (M, F)
        targets: torch.Tensor,  # (M, F)
    ) -> torch.Tensor:
        return F.mse_loss(preds, targets)


class GenerativeLoss(nn.Module):
    """MSE between predicted and true future timesteps."""

    def forward(
        self,
        preds: torch.Tensor,    # (B, n_steps, F)
        targets: torch.Tensor,  # (B, n_steps, F)
    ) -> torch.Tensor:
        return F.mse_loss(preds, targets)


class SSLLoss(nn.Module):
    """
    Weighted combination of all three SSL losses.

    Parameters
    ----------
    w_contrast : float   weight for contrastive loss   (default 1.0)
    w_masked   : float   weight for masked prediction  (default 1.0)
    w_gen      : float   weight for generative loss    (default 0.5)
    temperature: float   NT-Xent temperature           (default 0.07)
    """

    def __init__(
        self,
        w_contrast: float = 1.0,
        w_masked:   float = 1.0,
        w_gen:      float = 0.5,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.w_contrast = w_contrast
        self.w_masked   = w_masked
        self.w_gen      = w_gen

        self.contrast_loss = NTXentLoss(temperature)
        self.masked_loss   = MaskedPredictionLoss()
        self.gen_loss      = GenerativeLoss()

    def forward(self, outputs: dict) -> dict:
        """
        Parameters
        ----------
        outputs : dict from SSLModel.forward()
            Keys: z_anchor, z_pos, masked_preds, masked_targets,
                  gen_preds, gen_targets

        Returns
        -------
        dict with keys:
            total, loss_contrast, loss_masked, loss_gen
        """
        l_contrast = self.contrast_loss(
            outputs["z_anchor"], outputs["z_pos"]
        )
        l_masked = self.masked_loss(
            outputs["masked_preds"], outputs["masked_targets"]
        )
        l_gen = self.gen_loss(
            outputs["gen_preds"], outputs["gen_targets"]
        )

        total = (
            self.w_contrast * l_contrast
            + self.w_masked * l_masked
            + self.w_gen    * l_gen
        )

        return {
            "total":         total,
            "loss_contrast": l_contrast,
            "loss_masked":   l_masked,
            "loss_gen":      l_gen,
        }
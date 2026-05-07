"""
SSL Heads
=========
Three task-specific heads that attach to ClimateEncoder.

Head 1 — Contrastive (SimCLR-style)
    Projects the pooled representation to a smaller space
    where the NT-Xent loss is computed between augmented pairs.

Head 2 — Masked Prediction (BERT-style)
    Receives the full token sequence from the encoder.
    Predicts the original feature values at masked positions.

Head 3 — Generative (autoregressive)
    Receives the pooled representation of the first half of a window.
    Predicts the raw feature values of the second half step-by-step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Head 1 — Contrastive projection head
# ─────────────────────────────────────────────────────────────────────────────

class ContrastiveHead(nn.Module):
    """
    Two-layer MLP projector.
    Maps d_model → proj_dim for NT-Xent loss computation.

    Following SimCLR: the projector is discarded after pretraining —
    only the encoder is kept for fine-tuning.

    Parameters
    ----------
    d_model  : int   encoder output dimension
    proj_dim : int   projection space dimension (typically d_model / 2)
    """

    def __init__(self, d_model: int = 64, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, d_model)  pooled encoder output
        → (B, proj_dim)   L2-normalized projection
        """
        out = self.net(z)
        return F.normalize(out, dim=-1)   # unit sphere → stable cosine sim


# ─────────────────────────────────────────────────────────────────────────────
# Head 2 — Masked prediction head
# ─────────────────────────────────────────────────────────────────────────────

class MaskedPredictionHead(nn.Module):
    """
    Predicts original feature values at masked timestep positions.

    The encoder is called with return_sequence=True, giving a token
    per timestep. This head projects each token back to F features,
    and the loss is computed only at masked positions.

    Parameters
    ----------
    d_model    : int   encoder hidden dimension
    n_features : int   number of climate features to reconstruct
    """

    def __init__(self, d_model: int = 64, n_features: int = 5):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sequence : (B, T, d_model)   encoder token sequence
        mask     : (B, T)  bool       True = position was masked

        Returns
        -------
        preds  : (M, n_features)   predictions at masked positions
        """
        preds_all = self.decoder(sequence)   # (B, T, n_features)
        preds     = preds_all[mask]          # (M, n_features)
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Head 3 — Generative head
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeHead(nn.Module):
    """
    Predicts future timesteps from a context representation.

    The input window is split in half:
      context  = first  T//2 steps  → encoded → pooled → z_context
      target   = second T//2 steps  → what the head must predict

    The head uses a small GRU decoder that autoregressively
    generates T//2 future feature vectors from z_context.

    Parameters
    ----------
    d_model    : int   encoder output (context vector) size
    n_features : int   number of climate features to generate
    n_steps    : int   number of future steps to predict (= T // 2)
    hidden_dim : int   GRU hidden state size
    """

    def __init__(
        self,
        d_model: int = 64,
        n_features: int = 5,
        n_steps: int = 12,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_steps    = n_steps
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Project context vector to GRU initial hidden state
        self.context_proj = nn.Linear(d_model, hidden_dim)

        # GRU: input is previous predicted step (n_features),
        #       output is next hidden state → next predicted step
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Project hidden state → feature prediction
        self.out_proj = nn.Linear(hidden_dim, n_features)

    def forward(
        self,
        z_context: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z_context : (B, d_model)         pooled encoding of first T//2 steps
        target    : (B, n_steps, n_features)  ground truth future steps
                    If provided → teacher forcing (training mode)
                    If None     → free-run autoregressive (eval mode)

        Returns
        -------
        preds : (B, n_steps, n_features)
        """
        B = z_context.size(0)

        # Initialise GRU hidden state from context
        h = self.context_proj(z_context)         # (B, hidden_dim)
        h = torch.tanh(h).unsqueeze(0)           # (1, B, hidden_dim)

        # Seed: a zero vector representing "start of future"
        inp = torch.zeros(B, 1, self.n_features,
                          device=z_context.device)

        preds = []
        for t in range(self.n_steps):
            out, h = self.gru(inp, h)            # out: (B, 1, hidden_dim)
            pred = self.out_proj(out)             # (B, 1, n_features)
            preds.append(pred)

            if target is not None:
                # Teacher forcing: feed real next step
                inp = target[:, t: t + 1, :]
            else:
                # Free-run: feed own prediction
                inp = pred

        return torch.cat(preds, dim=1)           # (B, n_steps, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# Combined SSL model
# ─────────────────────────────────────────────────────────────────────────────

class SSLModel(nn.Module):
    """
    Wraps ClimateEncoder + all three SSL heads.

    Parameters
    ----------
    encoder             : ClimateEncoder
    contrastive_head    : ContrastiveHead
    masked_head         : MaskedPredictionHead
    generative_head     : GenerativeHead
    mask_ratio          : float   fraction of timesteps to mask (default 0.25)
    """

    def __init__(
        self,
        encoder,
        contrastive_head: ContrastiveHead,
        masked_head: MaskedPredictionHead,
        generative_head: GenerativeHead,
        mask_ratio: float = 0.25,
    ):
        super().__init__()
        self.encoder          = encoder
        self.contrastive_head = contrastive_head
        self.masked_head      = masked_head
        self.generative_head  = generative_head
        self.mask_ratio       = mask_ratio

    def _apply_mask(self, x: torch.Tensor):
        """
        Randomly zero out mask_ratio fraction of timesteps.
        Returns masked input and boolean mask.
        """
        B, T, F = x.shape
        n_mask  = max(1, int(T * self.mask_ratio))

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        for i in range(B):
            idx = torch.randperm(T, device=x.device)[:n_mask]
            mask[i, idx] = True

        x_masked = x.clone()
        x_masked[mask] = 0.0
        return x_masked, mask

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor):
        """
        Parameters
        ----------
        anchor   : (B, T, F)   original (or lightly augmented) window
        positive : (B, T, F)   differently augmented view (for contrastive)

        Returns
        -------
        dict with keys:
          z_anchor, z_pos          — contrastive projections
          masked_preds, targets    — masked prediction outputs + ground truth
          gen_preds, gen_targets   — generative outputs + ground truth
        """
        B, T, F = anchor.shape

        # ── Contrastive ─────────────────────────────────────────────────
        z_anchor = self.contrastive_head(self.encoder(anchor))
        z_pos    = self.contrastive_head(self.encoder(positive))

        # ── Masked prediction ────────────────────────────────────────────
        x_masked, mask = self._apply_mask(anchor)
        seq = self.encoder(x_masked, return_sequence=True)
        # If encoder uses CLS token, strip it before masked prediction
        if seq.size(1) == T + 1:
            seq = seq[:, 1:, :]         # drop CLS → (B, T, d_model)
        masked_preds   = self.masked_head(seq, mask)   # (M, F)
        masked_targets = anchor[mask]                   # (M, F)

        # ── Generative ───────────────────────────────────────────────────
        half   = T // 2
        ctx    = anchor[:, :half, :]                    # first half
        future = anchor[:, half:, :]                    # second half (target)
        z_ctx  = self.encoder(ctx)                      # (B, d_model)
        gen_preds = self.generative_head(z_ctx, target=future)  # (B, half, F)

        return {
            "z_anchor":      z_anchor,       # (B, proj_dim)
            "z_pos":         z_pos,           # (B, proj_dim)
            "masked_preds":  masked_preds,    # (M, F)
            "masked_targets":masked_targets,  # (M, F)
            "gen_preds":     gen_preds,        # (B, T//2, F)
            "gen_targets":   future,           # (B, T//2, F)
        }
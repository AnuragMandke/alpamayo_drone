"""
flow_matching.py — FlowMatchingDecoder

A Transformer-based continuous normalizing flow decoder that maps
(language/visual context embedding, noisy action) → denoised action.

Action space for UAV: [vx, vy, vz, yaw_rate]  (4-DoF velocity commands)

Training: Conditional Flow Matching (CFM) loss
Inference: DDIM-style deterministic sampling over `num_diffusion_steps` steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """Encodes a scalar diffusion timestep t ∈ [0,1] into a vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None]           # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


class FlowMatchingDecoder(nn.Module):
    """
    Transformer denoiser for action generation via Conditional Flow Matching.

    Args:
        context_dim:       Dimensionality of backbone output embedding
        action_dim:        Action dimensionality (4 for UAV)
        hidden_dim:        Internal transformer width
        num_layers:        Number of transformer decoder layers
        num_heads:         Attention heads
        num_diffusion_steps:  DDIM steps used at inference
        num_train_steps:   Number of discrete noise steps for training schedule
        sigma_min:         Minimum noise level (prevents collapse)
    """

    def __init__(
        self,
        context_dim: int = 4096,
        action_dim: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_diffusion_steps: int = 8,
        num_train_steps: int = 100,
        sigma_min: float = 1e-3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_train_steps = num_train_steps
        self.sigma_min = sigma_min

        # Project context embedding into hidden_dim
        self.context_proj = nn.Linear(context_dim, hidden_dim)

        # Project noisy action into hidden_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Timestep embedding
        self.time_emb = SinusoidalTimestepEmbedding(hidden_dim)

        # Transformer decoder: action tokens attend to context
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,         # Pre-norm for stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection → predicted velocity field (same dim as action)
        self.out_proj = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core denoiser
    # ------------------------------------------------------------------

    def forward(
        self,
        noisy_action: torch.Tensor,   # (B, action_horizon, action_dim)
        context: torch.Tensor,        # (B, seq_len, context_dim)
        t: torch.Tensor,              # (B,) timestep in [0, 1]
    ) -> torch.Tensor:
        """
        Predict the velocity field v(x_t, t, c) for flow matching.
        Returns tensor of shape (B, action_horizon, action_dim).
        """
        B, H, _ = noisy_action.shape

        # Project context: (B, seq_len, hidden_dim)
        ctx = self.context_proj(context)

        # Project noisy action: (B, H, hidden_dim)
        act = self.action_proj(noisy_action)

        # Add timestep embedding (broadcast over action horizon)
        t_emb = self.time_emb(t).unsqueeze(1)   # (B, 1, hidden_dim)
        act = act + t_emb

        # Transformer decode
        out = self.transformer(tgt=act, memory=ctx)   # (B, H, hidden_dim)

        # Project to velocity field
        velocity = self.out_proj(out)                 # (B, H, action_dim)
        return velocity

    # ------------------------------------------------------------------
    # Training loss (Conditional Flow Matching)
    # ------------------------------------------------------------------

    def cfm_loss(
        self,
        action_gt: torch.Tensor,   # (B, H, action_dim)  ground-truth
        context: torch.Tensor,     # (B, seq_len, context_dim)
    ) -> torch.Tensor:
        """
        CFM training loss.

        Straight-path interpolation:
            x_t = (1 - t) * noise + t * action_gt
            target_velocity = action_gt - noise

        We predict v(x_t, t, c) and minimise MSE against target velocity.
        """
        B = action_gt.shape[0]
        device = action_gt.device

        # Sample random t ~ U[0, 1]
        t = torch.rand(B, device=device)

        # Sample noise from N(0, I)
        noise = torch.randn_like(action_gt)

        # Interpolate
        t_bc = t[:, None, None]                        # broadcast
        x_t = (1 - t_bc) * noise + t_bc * action_gt
        target_v = action_gt - noise

        # Predicted velocity
        pred_v = self.forward(x_t, context, t)

        return F.mse_loss(pred_v, target_v)

    # ------------------------------------------------------------------
    # Inference (DDIM-style deterministic sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,        # (B, seq_len, context_dim)
        action_horizon: int = 4,
    ) -> torch.Tensor:
        """
        Generate actions by integrating the learned velocity field
        from t=0 (noise) to t=1 (action).

        Returns (B, action_horizon, action_dim).
        """
        B = context.shape[0]
        device = context.device

        # Start from pure noise
        x = torch.randn(B, action_horizon, self.action_dim, device=device)

        # Euler integration over num_diffusion_steps steps
        dt = 1.0 / self.num_diffusion_steps
        for i in range(self.num_diffusion_steps):
            t_val = i / self.num_diffusion_steps
            t = torch.full((B,), t_val, device=device)
            v = self.forward(x, context, t)
            x = x + dt * v

        return x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Semantic Injection module.

    Paper:
        Q_f = W_Q Norm(v_sem)
        K_f / V_f = W_K / W_V E_clip(v_final)
        Z_fused = Softmax(QK^T / sqrt(d_k)) V

    In this engineering version:
        v_sem and v_final are both vector embeddings.
        We project them into token sequences and apply multi-head attention.
    """

    def __init__(
        self,
        clip_dim: int = 768,
        hidden_dim: int = 1024,
        num_tokens: int = 77,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        self.query_proj = nn.Linear(clip_dim, hidden_dim)
        self.key_proj = nn.Linear(clip_dim, hidden_dim)
        self.value_proj = nn.Linear(clip_dim, hidden_dim)

        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
        self.final_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )

        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, clip_dim),
        )

    def vector_to_tokens(self, v, base_tokens, proj):
        b = v.shape[0]
        x = proj(F.normalize(v, dim=-1))
        x = x.unsqueeze(1)
        tokens = base_tokens.expand(b, -1, -1)
        return tokens + x

    def forward(self, v_sem, v_final):
        q = self.vector_to_tokens(v_sem, self.query_tokens, self.query_proj)
        k = self.vector_to_tokens(v_final, self.final_tokens, self.key_proj)
        v = self.vector_to_tokens(v_final, self.final_tokens, self.value_proj)

        fused, attn_weights = self.attn(q, k, v)
        fused = self.self_attn(fused)
        c_inj = self.projector(fused)

        return {
            "C_inj": c_inj,
            "attn_weights": attn_weights,
        }


class ZeroConvAdapter(nn.Module):
    """
    Zero-convolution adapter.

    ControlNet-style zero-conv starts from zero so that the control branch
    initially does not disturb the frozen diffusion backbone.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class LightweightControlAdapter(nn.Module):
    """
    A lightweight stand-in for ControlNet encoder.

    Input:
        z_t: noisy latent tensor
        m_struct: structural map
        c_inj: semantic conditioning tokens

    Output:
        residual feature map that can be injected into a diffusion UNet.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        struct_channels: int = 1,
        base_channels: int = 128,
        cond_dim: int = 768,
    ):
        super().__init__()

        self.struct_encoder = nn.Sequential(
            nn.Conv2d(struct_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.latent_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, base_channels),
            nn.SiLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            ZeroConvAdapter(base_channels),
        )

    def forward(self, z_t, m_struct, c_inj):
        if m_struct.shape[-2:] != z_t.shape[-2:]:
            m_struct = F.interpolate(
                m_struct,
                size=z_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fs = self.struct_encoder(m_struct)
        fz = self.latent_encoder(z_t)

        cond = c_inj.mean(dim=1)
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)

        x = fs + fz + cond
        return self.out(x)


class DiffusionGeneratorStub(nn.Module):
    """
    Engineering stub for Stage-2 generative tuning.

    This module imitates the training interface of a latent diffusion model:

        L_ldm = || epsilon - epsilon_theta(z_t, t, c) ||_2^2

    For a true reproduction:
    - replace this with SDXL UNet
    - use VAE encoder for z_0
    - use scheduler.add_noise
    - inject C_inj as prompt embeddings
    - inject m_struct through ControlNet residuals
    """

    def __init__(
        self,
        clip_dim: int = 768,
        latent_channels: int = 4,
        base_channels: int = 128,
    ):
        super().__init__()

        self.semantic_fusion = CrossAttentionFusion(
            clip_dim=clip_dim,
            hidden_dim=1024,
            num_tokens=77,
            num_heads=8,
        )

        self.control_adapter = LightweightControlAdapter(
            latent_channels=latent_channels,
            struct_channels=1,
            base_channels=base_channels,
            cond_dim=clip_dim,
        )

        self.noise_predictor = nn.Sequential(
            nn.Conv2d(latent_channels + base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, latent_channels, 3, padding=1),
        )

    def forward(self, z_t, t, v_sem, v_final, m_struct):
        fusion = self.semantic_fusion(v_sem, v_final)
        c_inj = fusion["C_inj"]

        ctrl = self.control_adapter(z_t, m_struct, c_inj)

        x = torch.cat([z_t, ctrl], dim=1)
        eps_pred = self.noise_predictor(x)

        return {
            "eps_pred": eps_pred,
            "C_inj": c_inj,
            "control": ctrl,
            **fusion,
        }

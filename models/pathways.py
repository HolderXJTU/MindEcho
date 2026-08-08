import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticPathway(nn.Module):
    """
    3-layer MLP semantic pathway.

    It maps the perturbed or clean latent map into a global CLIP-aligned vector.

    Paper:
    Semantic Pathway maps Z_tilde into v_sem aligned with CLIP space.
    """

    def __init__(
        self,
        in_channels: int = 256,
        latent_hw: int = 4,
        hidden_dim: int = 2048,
        clip_dim: int = 768,
        dropout: float = 0.1,
        normalize_output: bool = True,
    ):
        super().__init__()

        self.in_dim = in_channels * latent_hw * latent_hw
        self.clip_dim = clip_dim
        self.normalize_output = normalize_output

        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(self, z):
        b = z.shape[0]
        x = z.reshape(b, -1)
        v = self.net(x)

        if self.normalize_output:
            v = F.normalize(v, dim=-1)

        return v


class UpsampleBlock(nn.Module):
    """
    Lightweight upsampling block used by the structural pathway.
    """

    def __init__(self, in_ch, out_ch, norm_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(norm_groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(norm_groups, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class StructuralPathway(nn.Module):
    """
    Lightweight convolutional decoder with four upsampling blocks.

    Paper:
    Structural Pathway translates Z_tilde into a pixel-aligned condition map
    m_struct that preserves edges, shapes, and layout.

    If latent_hw = 4 and four upsampling blocks are used:
        4 -> 8 -> 16 -> 32 -> 64
    """

    def __init__(
        self,
        in_channels: int = 256,
        base_channels: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 8, kernel_size=3, padding=1),
            nn.GroupNorm(32, base_channels * 8),
            nn.SiLU(),
        )

        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.up3 = UpsampleBlock(base_channels * 2, base_channels)
        self.up4 = UpsampleBlock(base_channels, base_channels)

        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.stem(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        m = self.head(x)
        return m

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """
    A residual MLP block for voxel-to-latent mapping.

    This block is useful because voxel vectors are high-dimensional and noisy.
    LayerNorm stabilizes subject-specific distributions, while the residual
    connection prevents representation collapse.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class SubjectSpecificEncoder(nn.Module):
    """
    Subject-specific voxel encoder.

    The paper follows MindEye2-style subject-specific encoders:
    every subject has a private input projection, but all projections map
    into a shared latent space.

    Input:
        fmri: Tensor [B, Q_i]
        subject_ids: Tensor/List [B]

    Output:
        shared_vector: Tensor [B, shared_dim]
    """

    def __init__(
        self,
        subject_voxel_dims: Dict[int, int],
        shared_dim: int = 4096,
        hidden_dim: int = 4096,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.subject_voxel_dims = {int(k): int(v) for k, v in subject_voxel_dims.items()}
        self.shared_dim = shared_dim

        self.input_proj = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        self.output_norm = nn.LayerNorm(shared_dim)

        for sid, qdim in self.subject_voxel_dims.items():
            sid_str = str(sid)

            self.input_proj[sid_str] = nn.Sequential(
                nn.LayerNorm(qdim),
                nn.Linear(qdim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, shared_dim),
            )

            self.blocks[sid_str] = nn.Sequential(
                *[
                    ResidualMLPBlock(
                        dim=shared_dim,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )

    def encode_one_subject(self, fmri, subject_id: int):
        sid = str(int(subject_id))

        if sid not in self.input_proj:
            raise KeyError(f"Unknown subject id {sid}. Known: {list(self.input_proj.keys())}")

        x = self.input_proj[sid](fmri)
        x = self.blocks[sid](x)
        x = self.output_norm(x)
        return x

    def forward(self, fmri, subject_ids):
        if isinstance(subject_ids, list):
            subject_ids = torch.tensor(subject_ids, device=fmri.device)

        subject_ids = subject_ids.long()
        outputs = torch.zeros(
            fmri.shape[0],
            self.shared_dim,
            device=fmri.device,
            dtype=fmri.dtype,
        )

        unique_subjects = torch.unique(subject_ids)

        for sid in unique_subjects:
            sid_int = int(sid.item())
            mask = subject_ids == sid
            x_sub = fmri[mask]
            outputs[mask] = self.encode_one_subject(x_sub, sid_int)

        return outputs


class LatentMapProjector(nn.Module):
    """
    Project shared vector into a spatial latent map.

    Paper:
    one-dimensional beta voxel responses are first projected into a shared
    latent representation and reshaped into a learned spatial feature map:

    Z in R^{C x H' x W'}
    """

    def __init__(
        self,
        shared_dim: int,
        latent_channels: int = 256,
        latent_hw: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_hw = latent_hw
        self.out_dim = latent_channels * latent_hw * latent_hw

        self.net = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, self.out_dim),
        )

        self.post = nn.Sequential(
            nn.GroupNorm(32, latent_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        b = x.shape[0]
        z = self.net(x)
        z = z.view(b, self.latent_channels, self.latent_hw, self.latent_hw)
        z = self.post(z)
        return z

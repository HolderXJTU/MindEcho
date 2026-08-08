from typing import Dict

import torch
import torch.nn as nn

from .subject_encoder import SubjectSpecificEncoder, LatentMapProjector
from .cpd import CausalPhaseDisentangler
from .pathways import SemanticPathway, StructuralPathway


class MindEchoEncoder(nn.Module):
    """
    MindEcho representation module.

    Includes:
    - subject-specific voxel alignment
    - latent map projector
    - causal-phase disentangler
    - semantic pathway
    - structural pathway

    Stage 1 trains this module with:
    - image/text contrastive alignment
    - structural edge supervision
    - causal consistency
    """

    def __init__(
        self,
        subject_voxel_dims: Dict[int, int],
        shared_dim: int = 4096,
        latent_channels: int = 256,
        latent_hw: int = 4,
        clip_dim: int = 768,
        semantic_hidden: int = 2048,
        structural_channels: int = 64,
        structural_out_channels: int = 1,
        dropout: float = 0.1,
        cpd_kwargs: Dict = None,
    ):
        super().__init__()

        cpd_kwargs = cpd_kwargs or {}

        self.subject_encoder = SubjectSpecificEncoder(
            subject_voxel_dims=subject_voxel_dims,
            shared_dim=shared_dim,
            hidden_dim=shared_dim,
            num_blocks=2,
            dropout=dropout,
        )

        self.map_projector = LatentMapProjector(
            shared_dim=shared_dim,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            dropout=dropout,
        )

        self.cpd = CausalPhaseDisentangler(**cpd_kwargs)

        self.semantic_pathway = SemanticPathway(
            in_channels=latent_channels,
            latent_hw=latent_hw,
            hidden_dim=semantic_hidden,
            clip_dim=clip_dim,
            dropout=dropout,
            normalize_output=True,
        )

        self.structural_pathway = StructuralPathway(
            in_channels=latent_channels,
            base_channels=structural_channels,
            out_channels=structural_out_channels,
        )

    def encode_latent(self, fmri, subject_ids):
        shared = self.subject_encoder(fmri, subject_ids)
        z = self.map_projector(shared)
        return z

    def forward(self, fmri, subject_ids, force_perturb: bool = False):
        z = self.encode_latent(fmri, subject_ids)

        z_clean, z_tilde, cpd_info = self.cpd(
            z,
            force_perturb=force_perturb,
        )

        v_sem_clean = self.semantic_pathway(z_clean)
        v_sem_perturbed = self.semantic_pathway(z_tilde)

        m_struct = self.structural_pathway(z_tilde)

        return {
            "z_clean": z_clean,
            "z_tilde": z_tilde,
            "v_sem": v_sem_perturbed,
            "v_sem_clean": v_sem_clean,
            "v_sem_perturbed": v_sem_perturbed,
            "m_struct": m_struct,
            "cpd_info": cpd_info,
        }

    @torch.no_grad()
    def infer(self, fmri, subject_ids):
        self.eval()
        z = self.encode_latent(fmri, subject_ids)
        v_sem = self.semantic_pathway(z)
        m_struct = self.structural_pathway(z)
        return {
            "z": z,
            "v_sem": v_sem,
            "m_struct": m_struct,
        }

"""
Model modules for MindEcho.

This subpackage includes:

- subject-specific fMRI encoders
- latent map projectors
- causal-phase disentangler
- semantic and structural pathways
- generator interface for SDXL / ControlNet-style training
"""

from .subject_encoder import (
    ResidualMLPBlock,
    SubjectSpecificEncoder,
    LatentMapProjector,
)

from .cpd import (
    CausalPhaseDisentangler,
    amplitude_phase_swap,
)

from .pathways import (
    SemanticPathway,
    StructuralPathway,
    UpsampleBlock,
)

from .mindecho import MindEchoEncoder

try:
    from .generator import (
        CrossAttentionFusion,
        ZeroConvAdapter,
        LightweightControlAdapter,
        DiffusionGeneratorStub,
    )
except Exception:
    CrossAttentionFusion = None
    ZeroConvAdapter = None
    LightweightControlAdapter = None
    DiffusionGeneratorStub = None

__all__ = [
    "ResidualMLPBlock",
    "SubjectSpecificEncoder",
    "LatentMapProjector",
    "CausalPhaseDisentangler",
    "amplitude_phase_swap",
    "SemanticPathway",
    "StructuralPathway",
    "UpsampleBlock",
    "MindEchoEncoder",
    "CrossAttentionFusion",
    "ZeroConvAdapter",
    "LightweightControlAdapter",
    "DiffusionGeneratorStub",
]

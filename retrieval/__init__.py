"""
Retrieval modules for MindEcho.

This subpackage implements Confidence-Aware Retrieval:

- external memory loading
- top-K visual-textual retrieval
- neighborhood-aware prototype construction
- confidence-gated SLERP fusion
"""

from .car import (
    ConfidenceAwareRetrieval,
    safe_normalize,
    slerp,
)

from .memory_bank import MemoryBank

__all__ = [
    "ConfidenceAwareRetrieval",
    "safe_normalize",
    "slerp",
    "MemoryBank",
]

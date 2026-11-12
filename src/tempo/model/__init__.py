"""Embedding model and metric-learning loss (requires PyTorch)."""

from tempo.model.encoder import EncoderConfig, RhythmEncoder
from tempo.model.losses import (
    TripletStats,
    batch_hard_triplet_loss,
    pairwise_distances,
    supervised_contrastive_loss,
)

__all__ = [
    "EncoderConfig",
    "RhythmEncoder",
    "TripletStats",
    "batch_hard_triplet_loss",
    "pairwise_distances",
    "supervised_contrastive_loss",
]

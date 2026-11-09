"""Embedding model and metric-learning loss (requires PyTorch)."""

from keyauth.model.encoder import EncoderConfig, RhythmEncoder
from keyauth.model.losses import TripletStats, batch_hard_triplet_loss, pairwise_distances

__all__ = [
    "EncoderConfig",
    "RhythmEncoder",
    "TripletStats",
    "batch_hard_triplet_loss",
    "pairwise_distances",
]

"""The real embedder: a trained checkpoint wrapped as timings -> embedding (needs PyTorch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from keyauth.data import KeystrokeData, Normalizer
from keyauth.thresholds import PerUserThresholdParams
from keyauth.train import embed, load_checkpoint


class TorchEmbedder:
    def __init__(self, model, normalizer: Normalizer):
        self.model = model.eval()
        self.normalizer = normalizer

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple["TorchEmbedder", float]:
        """Returns (embedder, threshold chosen during training)."""
        model, normalizer, meta = load_checkpoint(path, device=device)
        return cls(model, normalizer), float(meta["threshold"])

    @staticmethod
    def per_user_params(path: str | Path, device: str = "cpu") -> PerUserThresholdParams | None:
        """Per-user threshold rule stored in the checkpoint, or None for checkpoints written before C6b."""
        _, _, meta = load_checkpoint(path, device=device)
        return PerUserThresholdParams.from_meta(meta)

    def __call__(self, timings: np.ndarray) -> np.ndarray:
        data = KeystrokeData.from_arrays([timings], ["_"])
        return embed(self.model, self.normalizer, data)[0]

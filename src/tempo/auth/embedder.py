"""The real embedder: a trained checkpoint wrapped as timings -> embedding (needs PyTorch).

Attempts reach the engine with the submit key (Enter) as their last keystroke, as the web page and the CMU data
send them. Models trained on Aalto never saw that key (the Aalto log has no Enter); their checkpoint says
final_enter=False, and the embedder then drops the last keystroke before embedding, so the model gets the same
kind of input it was trained on. The engine still sees, stores and length-checks the full attempt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tempo.auth.validation import InvalidTimings
from tempo.data import KeystrokeData, Normalizer
from tempo.thresholds import PerUserThresholdParams
from tempo.train import embed, load_checkpoint


class TorchEmbedder:
    def __init__(self, model, normalizer: Normalizer, final_enter: bool = True):
        if model.config.key_vocab > 0:
            raise ValueError("this model needs key identity, which the engine never receives (anonymized events)")
        self.model = model.eval()
        self.normalizer = normalizer
        self.final_enter = final_enter  # False: the model expects attempts without the final Enter

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> tuple["TorchEmbedder", float]:
        """Returns (embedder, threshold chosen during training)."""
        model, normalizer, meta = load_checkpoint(path, device=device)
        return cls(model, normalizer, final_enter=meta["final_enter"]), float(meta["threshold"])

    @staticmethod
    def per_user_params(path: str | Path, device: str = "cpu") -> PerUserThresholdParams | None:
        """Per-user threshold rule stored in the checkpoint, or None for checkpoints written before C6b."""
        _, _, meta = load_checkpoint(path, device=device)
        return PerUserThresholdParams.from_meta(meta)

    def __call__(self, timings: np.ndarray) -> np.ndarray:
        data = KeystrokeData.from_arrays([timings], ["_"])
        if not self.final_enter and data.lengths[0] < 3:
            raise InvalidTimings("this model needs at least 2 keystrokes before the final Enter")
        if not self.final_enter:
            data = data.drop_final_key()
        return embed(self.model, self.normalizer, data)[0]

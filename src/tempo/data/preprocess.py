"""Turn raw timings (seconds) into well-scaled model inputs.

1. Transform. Timings are right-skewed (a few slow pauses), so:
     hold, dd -> log(x)                          (always > 0)
     ud       -> sign(x) * log1p(|x| / UD_SCALE)  (can be negative under rollover)
2. Standardize each channel with mean/std fitted on TRAINING users only.

Statistics are per channel (not per key position), so the same normalizer
works for any phrase length, which the API needs since every user has their
own password.

Invalid entries (padding, and dd/ud of the last keystroke) are set to 0 and
excluded from the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempo.config import N_CHANNELS
from tempo.data.container import KeystrokeData

EPS_S = 1e-3  # floor for log, 1 ms
UD_SCALE_S = 0.01


def valid_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """(N, L, 3) bool: hold valid for j < len, dd/ud valid for j < len - 1."""
    pos = np.arange(max_len)[None, :]
    lengths = np.asarray(lengths)[:, None]
    mask = np.zeros((lengths.shape[0], max_len, N_CHANNELS), dtype=bool)
    mask[:, :, 0] = pos < lengths
    mask[:, :, 1] = pos < lengths - 1
    mask[:, :, 2] = pos < lengths - 1
    return mask


def transform(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    out[..., 0] = np.log(np.maximum(values[..., 0], EPS_S))
    out[..., 1] = np.log(np.maximum(values[..., 1], EPS_S))
    out[..., 2] = np.sign(values[..., 2]) * np.log1p(np.abs(values[..., 2]) / UD_SCALE_S)
    out[~valid_mask(lengths, values.shape[1])] = 0.0
    return out


@dataclass
class Normalizer:
    mean: np.ndarray  # (3,)
    std: np.ndarray  # (3,)

    @classmethod
    def fit(cls, data: KeystrokeData) -> "Normalizer":
        t = transform(data.values, data.lengths)
        mask = valid_mask(data.lengths, data.max_len)
        mean = np.zeros(N_CHANNELS)
        std = np.ones(N_CHANNELS)
        for c in range(N_CHANNELS):
            vals = t[..., c][mask[..., c]]
            if vals.size < 2:
                raise ValueError("not enough data to fit the normalizer")
            mean[c] = vals.mean()
            std[c] = max(vals.std(), 1e-6)
        return cls(mean=mean, std=std)

    def apply(self, values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """(N, L, 3) seconds -> (N, L, 3) float32 standardized model input."""
        t = (transform(values, lengths) - self.mean) / self.std
        t[~valid_mask(lengths, t.shape[1])] = 0.0
        return t.astype(np.float32)

    def __call__(self, data: KeystrokeData) -> np.ndarray:
        return self.apply(data.values, data.lengths)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(mean=np.asarray(d["mean"], dtype=np.float64), std=np.asarray(d["std"], dtype=np.float64))

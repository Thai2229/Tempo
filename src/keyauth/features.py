"""Keystrokes -> timing time series.

For keystrokes k_0 .. k_{n-1} with press time d_i and release time u_i:

    hold_i = u_i - d_i              how long key i is held (dwell)
    dd_i   = d_{i+1} - d_i          press-to-press latency
    ud_i   = d_{i+1} - u_i          release-to-press latency (flight);
                                    negative when the next key is pressed
                                    before this one is released (rollover)

The result is a (n, 3) array, one row per keystroke, in seconds. The last
keystroke has no successor, so its dd/ud are 0 (models ignore them via the
sequence length; see TimingSequence.n_transitions).

The flat vector form (to_vector / from_vector) uses the CMU keystroke
benchmark column order: H.k0, DD.k0.k1, UD.k0.k1, H.k1, ..., H.k{n-1}
(length 3n - 2), so CMU rows and our own recordings share one format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from keyauth.config import MAX_HOLD_S, MAX_LATENCY_S, MS_PER_SECOND, N_CHANNELS
from keyauth.events import KeyEvent


@dataclass(frozen=True)
class TimingSequence:
    keys: tuple[str, ...]
    values: np.ndarray  # shape (n_keys, 3): hold, dd, ud in seconds

    def __post_init__(self):
        if self.values.ndim != 2 or self.values.shape[1] != N_CHANNELS:
            raise ValueError(f"values must have shape (n, {N_CHANNELS}), got {self.values.shape}")
        if self.values.shape[0] != len(self.keys):
            raise ValueError("values and keys length mismatch")

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def n_transitions(self) -> int:
        return self.n_keys - 1

    @property
    def hold(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def dd(self) -> np.ndarray:
        return self.values[:-1, 1]

    @property
    def ud(self) -> np.ndarray:
        return self.values[:-1, 2]

    def to_vector(self) -> np.ndarray:
        """Flatten to CMU column order: H, DD, UD per key, then the final H."""
        parts = []
        for i in range(self.n_keys - 1):
            parts.extend(self.values[i, :])
        parts.append(self.values[-1, 0])
        return np.asarray(parts, dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: Sequence[float], keys: Sequence[str]) -> "TimingSequence":
        """Inverse of to_vector."""
        vec = np.asarray(vec, dtype=np.float32)
        n = len(keys)
        if vec.shape != (3 * n - 2,):
            raise ValueError(f"expected vector of length {3 * n - 2} for {n} keys, got {vec.shape}")
        values = np.zeros((n, N_CHANNELS), dtype=np.float32)
        for i in range(n - 1):
            values[i, :] = vec[3 * i : 3 * i + 3]
        values[-1, 0] = vec[-1]
        return cls(keys=tuple(keys), values=values)


def extract_features(events: Sequence[KeyEvent], check_bounds: bool = True) -> TimingSequence:
    """Compute hold / down-down / up-down timings for a typed attempt."""
    if len(events) < 2:
        raise ValueError("need at least 2 keystrokes to compute timings")

    down = np.array([e.down for e in events], dtype=np.float64) / MS_PER_SECOND
    up = np.array([e.up for e in events], dtype=np.float64) / MS_PER_SECOND

    values = np.zeros((len(events), N_CHANNELS), dtype=np.float64)
    values[:, 0] = up - down
    values[:-1, 1] = down[1:] - down[:-1]
    values[:-1, 2] = down[1:] - up[:-1]

    if check_bounds:
        hold = values[:, 0]
        dd = values[:-1, 1]
        if np.any(hold < 0) or np.any(hold > MAX_HOLD_S):
            raise ValueError("implausible hold time; capture is probably broken")
        if np.any(dd < 0) or np.any(dd > MAX_LATENCY_S):
            raise ValueError("implausible key-to-key latency; capture is probably broken")

    return TimingSequence(keys=tuple(e.key for e in events), values=values.astype(np.float32))

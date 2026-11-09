"""KeystrokeData: a batch of typing attempts from many users, as padded arrays.

Every loader (CMU benchmark, our own recordings, synthetic typists) returns
this one type, so preprocessing, splitting and the PyTorch Dataset only have
to understand one format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from keyauth.config import N_CHANNELS
from keyauth.features import TimingSequence


@dataclass(frozen=True)
class KeystrokeData:
    values: np.ndarray  # (N, L, 3) float32 seconds; rows past lengths[i] are 0
    lengths: np.ndarray  # (N,) int64, keystrokes in each attempt
    user_ids: np.ndarray  # (N,) str
    sessions: np.ndarray  # (N,) int64, capture session (-1 = unknown)

    def __post_init__(self):
        n = self.values.shape[0]
        if self.values.ndim != 3 or self.values.shape[2] != N_CHANNELS:
            raise ValueError(f"values must have shape (N, L, {N_CHANNELS}), got {self.values.shape}")
        if self.lengths.shape != (n,) or self.user_ids.shape != (n,) or self.sessions.shape != (n,):
            raise ValueError("lengths, user_ids and sessions must all have shape (N,)")
        if n > 0 and (self.lengths.min() < 2 or self.lengths.max() > self.values.shape[1]):
            raise ValueError("every attempt needs 2..L keystrokes")

    def __len__(self) -> int:
        return self.values.shape[0]

    @property
    def max_len(self) -> int:
        return self.values.shape[1]

    @property
    def users(self) -> list[str]:
        return sorted(set(self.user_ids.tolist()))

    def subset(self, index) -> "KeystrokeData":
        """Rows selected by an index array or boolean mask."""
        return KeystrokeData(
            values=self.values[index],
            lengths=self.lengths[index],
            user_ids=self.user_ids[index],
            sessions=self.sessions[index],
        )

    def for_users(self, users: Sequence[str]) -> "KeystrokeData":
        return self.subset(np.isin(self.user_ids, list(users)))

    def sequence(self, i: int, keys: Sequence[str] | None = None) -> TimingSequence:
        """Row i as a TimingSequence (keys default to "?" since loaders don't keep them)."""
        n = int(self.lengths[i])
        if keys is None:
            keys = ["?"] * n
        return TimingSequence(keys=tuple(keys), values=self.values[i, :n].copy())

    @classmethod
    def from_arrays(
        cls,
        sequences: Sequence[np.ndarray],
        user_ids: Sequence[str],
        sessions: Sequence[int] | None = None,
    ) -> "KeystrokeData":
        """Build from a list of (n_i, 3) timing arrays of possibly different lengths."""
        if len(sequences) != len(user_ids):
            raise ValueError("sequences and user_ids length mismatch")
        if sessions is None:
            sessions = [-1] * len(sequences)
        lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        max_len = 0
        if len(sequences) > 0:
            max_len = int(lengths.max())
        values = np.zeros((len(sequences), max_len, N_CHANNELS), dtype=np.float32)
        for i, s in enumerate(sequences):
            values[i, : len(s)] = np.asarray(s, dtype=np.float32)
        return cls(
            values=values,
            lengths=lengths,
            user_ids=np.array([str(u) for u in user_ids], dtype=object),
            sessions=np.asarray(sessions, dtype=np.int64),
        )

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[TimingSequence],
        user_ids: Sequence[str],
        sessions: Sequence[int] | None = None,
    ) -> "KeystrokeData":
        return cls.from_arrays([s.values for s in sequences], user_ids, sessions)

    @staticmethod
    def concat(parts: Sequence["KeystrokeData"]) -> "KeystrokeData":
        seqs, users, sessions = [], [], []
        for p in parts:
            for i in range(len(p)):
                seqs.append(p.values[i, : p.lengths[i]])
            users.extend(p.user_ids.tolist())
            sessions.extend(p.sessions.tolist())
        return KeystrokeData.from_arrays(seqs, users, sessions)

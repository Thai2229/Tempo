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
    # Optional, for free-text data (Aalto): which sentence was typed (an id or hash, never the text),
    # the key identity of each keystroke (e.g. JavaScript keyCode), 0 past lengths[i],
    # and whether the attempt contains a correction (Backspace, arrows, ...).
    sentence_ids: np.ndarray | None = None  # (N,) int64
    key_codes: np.ndarray | None = None  # (N, L) int64
    has_correction: np.ndarray | None = None  # (N,) bool

    def __post_init__(self):
        n = self.values.shape[0]
        if self.values.ndim != 3 or self.values.shape[2] != N_CHANNELS:
            raise ValueError(f"values must have shape (N, L, {N_CHANNELS}), got {self.values.shape}")
        if self.lengths.shape != (n,) or self.user_ids.shape != (n,) or self.sessions.shape != (n,):
            raise ValueError("lengths, user_ids and sessions must all have shape (N,)")
        if n > 0 and (self.lengths.min() < 2 or self.lengths.max() > self.values.shape[1]):
            raise ValueError("every attempt needs 2..L keystrokes")
        if (
            (self.sentence_ids is not None and self.sentence_ids.shape != (n,))
            or (self.key_codes is not None and self.key_codes.shape != self.values.shape[:2])
            or (self.has_correction is not None and self.has_correction.shape != (n,))
        ):
            raise ValueError("sentence_ids and has_correction must have shape (N,) and key_codes shape (N, L)")

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
        sentence_ids = None
        if self.sentence_ids is not None:
            sentence_ids = self.sentence_ids[index]
        key_codes = None
        if self.key_codes is not None:
            key_codes = self.key_codes[index]
        has_correction = None
        if self.has_correction is not None:
            has_correction = self.has_correction[index]
        return KeystrokeData(
            values=self.values[index],
            lengths=self.lengths[index],
            user_ids=self.user_ids[index],
            sessions=self.sessions[index],
            sentence_ids=sentence_ids,
            key_codes=key_codes,
            has_correction=has_correction,
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
        sentence_ids: Sequence[int] | None = None,
        key_codes: Sequence[Sequence[int]] | None = None,
        has_correction: Sequence[bool] | None = None,
    ) -> "KeystrokeData":
        """Build from a list of (n_i, 3) timing arrays of possibly different lengths.

        key_codes, if given, holds one code per keystroke for each sequence (same lengths).
        """
        if (
            len(sequences) != len(user_ids)
            or (sentence_ids is not None and len(sentence_ids) != len(sequences))
            or (key_codes is not None and [len(k) for k in key_codes] != [len(s) for s in sequences])
            or (has_correction is not None and len(has_correction) != len(sequences))
        ):
            raise ValueError("sequences, user_ids, sentence_ids, key_codes and has_correction length mismatch")
        if sessions is None:
            sessions = [-1] * len(sequences)
        lengths = np.array([len(s) for s in sequences], dtype=np.int64)
        max_len = 0
        if len(sequences) > 0:
            max_len = int(lengths.max())
        values = np.zeros((len(sequences), max_len, N_CHANNELS), dtype=np.float32)
        for i, s in enumerate(sequences):
            values[i, : len(s)] = np.asarray(s, dtype=np.float32)
        sentence_array = None
        if sentence_ids is not None:
            sentence_array = np.asarray(sentence_ids, dtype=np.int64).reshape(len(sequences))
        code_array = None
        if key_codes is not None:
            code_array = np.zeros((len(sequences), max_len), dtype=np.int64)
            for i, k in enumerate(key_codes):
                code_array[i, : len(k)] = np.asarray(k, dtype=np.int64)
        correction_array = None
        if has_correction is not None:
            correction_array = np.asarray(has_correction, dtype=bool).reshape(len(sequences))
        return cls(
            values=values,
            lengths=lengths,
            user_ids=np.array([str(u) for u in user_ids], dtype=object),
            sessions=np.asarray(sessions, dtype=np.int64),
            sentence_ids=sentence_array,
            key_codes=code_array,
            has_correction=correction_array,
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
        """Optional fields are kept only if every part has them."""
        seqs, users, sessions, sentence_ids, key_codes, has_correction = [], [], [], [], [], []
        for p in parts:
            for i in range(len(p)):
                seqs.append(p.values[i, : p.lengths[i]])
                if p.key_codes is not None:
                    key_codes.append(p.key_codes[i, : p.lengths[i]])
            users.extend(p.user_ids.tolist())
            sessions.extend(p.sessions.tolist())
            if p.sentence_ids is not None:
                sentence_ids.extend(p.sentence_ids.tolist())
            if p.has_correction is not None:
                has_correction.extend(p.has_correction.tolist())
        if len(sentence_ids) != len(seqs):
            sentence_ids = None
        if len(key_codes) != len(seqs):
            key_codes = None
        if len(has_correction) != len(seqs):
            has_correction = None
        return KeystrokeData.from_arrays(seqs, users, sessions, sentence_ids, key_codes, has_correction)

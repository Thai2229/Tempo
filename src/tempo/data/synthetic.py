"""Synthetic KeystrokeData, for tests and for exercising training without real data."""

from __future__ import annotations

import numpy as np

from tempo.data.container import KeystrokeData
from tempo.features import extract_features
from tempo.synthetic import make_typist, simulate_attempt


def make_synthetic_dataset(
    n_users: int,
    samples_per_user: int,
    phrase: str = ".tie5Roanl",
    n_sessions: int = 1,
    seed: int = 0,
) -> KeystrokeData:
    """n_users fake typists, each typing `phrase` samples_per_user times.

    Samples are spread evenly over n_sessions (session = sample index * n_sessions // samples_per_user).
    """
    rng = np.random.default_rng(seed)
    seqs, users, sessions = [], [], []
    for u in range(n_users):
        typist = make_typist(phrase, rng)
        for s in range(samples_per_user):
            seqs.append(extract_features(simulate_attempt(typist, rng)).values)
            users.append(f"s{u:03d}")
            sessions.append(s * n_sessions // samples_per_user)
    return KeystrokeData.from_arrays(seqs, users, sessions)

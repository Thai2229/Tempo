"""Train/val/test splits.

Splits are by USER, not by sample: the model must produce good embeddings for
people it has never seen (every new API user is one), so test users never
appear in training.
"""

from __future__ import annotations

import numpy as np

from keyauth.data.container import KeystrokeData


def split_users(
    data: KeystrokeData,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> tuple[KeystrokeData, KeystrokeData, KeystrokeData]:
    users = np.array(data.users, dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n = len(users)
    n_val = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    if val_frac < 0 or test_frac < 0 or n - n_val - n_test < 2:
        raise ValueError(f"cannot split {n} users into val={val_frac}, test={test_frac} with >=2 training users")

    val_users = users[:n_val]
    test_users = users[n_val : n_val + n_test]
    train_users = users[n_val + n_test :]
    return data.for_users(train_users), data.for_users(val_users), data.for_users(test_users)


def split_sessions(data: KeystrokeData, enroll_sessions: int) -> tuple[KeystrokeData, KeystrokeData]:
    """Per user: attempts from the first `enroll_sessions` sessions vs the rest.

    Used in evaluation to enroll on early typing and verify on later typing,
    which is more realistic than random samples from the same sitting.
    """
    if np.any(data.sessions < 0):
        raise ValueError("session info is missing for some attempts")
    early = np.zeros(len(data), dtype=bool)
    for u in data.users:
        rows = data.user_ids == u
        first = data.sessions[rows].min()
        early |= rows & (data.sessions < first + enroll_sessions)
    return data.subset(early), data.subset(~early)

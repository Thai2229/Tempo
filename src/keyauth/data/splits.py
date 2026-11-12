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


def split_sentences(
    data: KeystrokeData,
    holdout_frac: float = 0.3,
    seed: int = 0,
) -> tuple[KeystrokeData, KeystrokeData]:
    """(seen, unseen): hold out a random fraction of sentence ids, for "unseen sentence" evaluation.

    Every attempt of a held-out sentence goes to `unseen`, whoever typed it. Combine with
    split_users to test on unseen people typing unseen sentences, e.g. train on
    train_users & seen, evaluate on test_users & unseen.
    """
    sentences = np.array([], dtype=np.int64)
    if data.sentence_ids is not None:
        sentences = np.unique(data.sentence_ids)
    n_out = int(round(holdout_frac * len(sentences)))
    if data.sentence_ids is None or n_out < 1 or n_out >= len(sentences):
        raise ValueError(
            f"cannot hold out {holdout_frac} of {len(sentences)} sentence ids (data needs sentence_ids, "
            "and both sides must be non-empty)"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(sentences)
    unseen = np.isin(data.sentence_ids, sentences[:n_out])
    return data.subset(~unseen), data.subset(unseen)


def split_phrase_independent(
    data: KeystrokeData,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    holdout_frac: float = 0.5,
    min_eval_attempts: int = 6,
    seed: int = 0,
) -> tuple[KeystrokeData, KeystrokeData, KeystrokeData]:
    """(train, val, test) for free-text data (Aalto): unseen people typing unseen sentences.

    Users are split with split_users and sentence ids with split_sentences (same seed). Train keeps
    train users' seen sentences; val and test keep their users' held-out sentences only, and only users
    with >= min_eval_attempts of them (enrollment needs n_enroll plus at least one genuine attempt).
    """
    train_users, val_users, test_users = split_users(data, val_frac, test_frac, seed)
    seen, unseen = split_sentences(data, holdout_frac, seed)
    train = seen.for_users(train_users.users)

    def eval_part(users: KeystrokeData) -> KeystrokeData:
        part = unseen.for_users(users.users)
        users, counts = np.unique(part.user_ids.astype(str), return_counts=True)
        return part.for_users(users[counts >= min_eval_attempts].tolist())

    val, test = eval_part(val_users), eval_part(test_users)
    if len(val.users) < 2 or len(test.users) < 2:
        raise ValueError(f"fewer than 2 val or test users have >= {min_eval_attempts} held-out attempts")
    return train, val, test

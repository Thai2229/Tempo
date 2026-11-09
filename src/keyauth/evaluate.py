"""Verification metrics (NumPy only, no PyTorch needed).

Protocol, mirroring how the API will be used:
  For every user:
    enroll   = their first n_enroll attempts -> template = normalized mean embedding
    genuine  = their remaining attempts, scored against their own template
    impostor = the first impostors_per_user attempts of every OTHER user,
               scored against this user's template (the CMU benchmark uses 5)
  score = Euclidean distance to the template (lower = more likely genuine)

Metrics:
  FAR (false accept rate) = impostor attempts with distance <= threshold
  FRR (false reject rate) = genuine attempts with distance > threshold
  EER (equal error rate)  = the error where FAR == FRR

"global" metrics use one threshold for everyone, which is what the API does.
"per_user" EER averages each user's own EER (the CMU paper's convention,
optimistic because it picks a different threshold per user).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Scores:
    genuine: np.ndarray  # distances of real users to their own template
    impostor: np.ndarray  # distances of other users to the template
    per_user: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class EERResult:
    eer: float
    threshold: float


def far_frr(genuine: np.ndarray, impostor: np.ndarray, threshold: float) -> tuple[float, float]:
    far = float(np.mean(np.asarray(impostor) <= threshold))
    frr = float(np.mean(np.asarray(genuine) > threshold))
    return far, frr


def roc_curve(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, far, frr) evaluated at every observed distance, ascending."""
    genuine = np.sort(np.asarray(genuine, dtype=np.float64))
    impostor = np.sort(np.asarray(impostor, dtype=np.float64))
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("need at least one genuine and one impostor score")
    thresholds = np.unique(np.concatenate([[-np.inf], genuine, impostor]))
    far = np.searchsorted(impostor, thresholds, side="right") / impostor.size
    frr = 1.0 - np.searchsorted(genuine, thresholds, side="right") / genuine.size
    return thresholds, far, frr


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> EERResult:
    thresholds, far, frr = roc_curve(genuine, impostor)
    i = int(np.argmin(np.abs(far - frr)))
    threshold = float(thresholds[i])
    if not np.isfinite(threshold):
        threshold = 0.0
    return EERResult(eer=float((far[i] + frr[i]) / 2), threshold=threshold)


def make_template(embeddings: np.ndarray) -> np.ndarray:
    """Mean of enrollment embeddings, projected back onto the unit sphere."""
    t = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(t)
    if norm < 1e-12:
        raise ValueError("enrollment embeddings cancel out")
    return t / norm


def verification_scores(
    embeddings: np.ndarray,
    user_ids: np.ndarray,
    n_enroll: int = 10,
    impostors_per_user: int = 5,
) -> Scores:
    """Score every user's template against genuine and impostor attempts.

    Rows are assumed to be in capture order within each user (true for CMU
    and for recordings), so "first n" means "earliest n".
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)

    def score_user(enroll: np.ndarray, genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        template = make_template(enroll)
        return np.linalg.norm(genuine - template, axis=1), np.linalg.norm(impostor - template, axis=1)

    return score_protocol(embeddings, user_ids, score_user, n_enroll, impostors_per_user)


def score_protocol(
    features: np.ndarray,
    user_ids: np.ndarray,
    score_user: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    n_enroll: int = 10,
    impostors_per_user: int = 5,
) -> Scores:
    """Run the enrollment protocol from this module's docstring over any scorer.

    score_user(enroll_rows, genuine_rows, impostor_rows) -> (genuine, impostor)
    distances. Every model is compared through this one function, so they all
    see the same enrollment attempts and the same impostors.
    """
    user_ids = np.asarray(user_ids)
    users = sorted(set(user_ids.tolist()))
    rows = {u: np.flatnonzero(user_ids == u) for u in users}
    if len(users) < 2 or n_enroll < 1:
        raise ValueError("need >= 2 users and n_enroll >= 1")

    scores = Scores(genuine=np.empty(0), impostor=np.empty(0))
    all_gen, all_imp = [], []
    for u in users:
        idx = rows[u]
        if len(idx) <= n_enroll:
            raise ValueError(f"user {u} has {len(idx)} attempts; need more than n_enroll={n_enroll}")
        imp_idx = np.concatenate([rows[v][:impostors_per_user] for v in users if v != u])
        gen, imp = score_user(features[idx[:n_enroll]], features[idx[n_enroll:]], features[imp_idx])
        scores.per_user[u] = (gen, imp)
        all_gen.append(gen)
        all_imp.append(imp)
    scores.genuine = np.concatenate(all_gen)
    scores.impostor = np.concatenate(all_imp)
    return scores


def summarize(scores: Scores, threshold: float | None = None) -> dict:
    """Global EER, mean per-user EER, and FAR/FRR at `threshold` (default: the global EER threshold)."""
    glob = compute_eer(scores.genuine, scores.impostor)
    per_user = [compute_eer(g, i).eer for g, i in scores.per_user.values()]
    if threshold is None:
        threshold = glob.threshold
    far, frr = far_frr(scores.genuine, scores.impostor, threshold)
    return {
        "eer": glob.eer,
        "eer_threshold": glob.threshold,
        "per_user_eer": float(np.mean(per_user)),
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
    }

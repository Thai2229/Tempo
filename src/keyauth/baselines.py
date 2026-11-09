"""Classic keystroke-dynamics detectors on raw timing vectors (NumPy only).

Scaled Manhattan and Mahalanobis, after Killourhy & Maxion (2009). Both fit one
small model per enrolled user from that user's enrollment attempts, so they
need no training users at all.

A raw timing vector is the valid timings of one attempt, in seconds and in
CMU column order: hold, dd, ud for each key, then the last key's hold.

Enrollment here is short (10 attempts, against 31 features) and dd = hold + ud
exactly, so the sample covariance is always singular. Mahalanobis therefore
shrinks it toward its own diagonal and uses a pseudo-inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from keyauth.data.container import KeystrokeData
from keyauth.data.preprocess import valid_mask
from keyauth.evaluate import Scores, score_protocol

# Floor on a per-feature spread (seconds): 1 ms, the resolution browsers give.
MIN_SCALE_S = 1e-3


def timing_vectors(data: KeystrokeData) -> np.ndarray:
    """(N, 3L - 2) raw timings in seconds. Every attempt must have the same length."""
    if len(data) == 0 or np.any(data.lengths != data.lengths[0]):
        raise ValueError("timing vectors need at least one attempt and one fixed phrase length")
    n_keys = int(data.lengths[0])
    mask = valid_mask(data.lengths, data.max_len)
    return data.values[mask].reshape(len(data), 3 * n_keys - 2).astype(np.float64)


def _check_matrix(x: np.ndarray, name: str, min_rows: int, n_features: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if (
        arr.ndim != 2
        or arr.shape[0] < min_rows
        or arr.shape[1] < 1
        or not np.all(np.isfinite(arr))
        or (n_features is not None and arr.shape[1] != n_features)
    ):
        raise ValueError(
            f"{name} must be a finite (N, D) array with N >= {min_rows} and D = {n_features} "
            f"(None = any), got shape {arr.shape}"
        )
    return arr


class Detector(Protocol):
    def fit(self, enroll: np.ndarray) -> None:
        """Learn one user's model from (N, D) enrollment vectors."""

    def score(self, x: np.ndarray) -> np.ndarray:
        """(M, D) vectors -> (M,) anomaly scores, lower = more likely genuine."""


@dataclass
class ScaledManhattan:
    """score = sum_i |x_i - mean_i| / mad_i, mad = mean absolute deviation."""

    min_scale: float = MIN_SCALE_S
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, enroll: np.ndarray) -> None:
        arr = _check_matrix(enroll, "enrollment", min_rows=2)
        self.mean = arr.mean(axis=0)
        self.scale = np.maximum(np.abs(arr - self.mean).mean(axis=0), self.min_scale)

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError("detector is not fitted")
        arr = _check_matrix(x, "attempts", min_rows=1, n_features=self.mean.shape[0])
        return np.sum(np.abs(arr - self.mean) / self.scale, axis=1)


@dataclass
class Mahalanobis:
    """score = sqrt((x - mean)' C^+ (x - mean)), C = shrunk sample covariance.

    C = (1 - shrinkage) * S + shrinkage * diag(S) + min_scale^2 * I. The ridge
    keeps C usable when a feature never varies. shrinkage = 0 is the textbook
    detector with a pseudo-inverse; 1 is a scaled Euclidean distance.
    """

    shrinkage: float = 0.5
    min_scale: float = MIN_SCALE_S
    mean: np.ndarray | None = None
    precision: np.ndarray | None = None

    def fit(self, enroll: np.ndarray) -> None:
        if self.shrinkage < 0.0 or self.shrinkage > 1.0:
            raise ValueError(f"shrinkage must be in [0, 1], got {self.shrinkage}")
        arr = _check_matrix(enroll, "enrollment", min_rows=2)
        self.mean = arr.mean(axis=0)
        sample = np.atleast_2d(np.cov(arr, rowvar=False, ddof=1))
        shrunk = (1.0 - self.shrinkage) * sample + self.shrinkage * np.diag(np.diag(sample))
        shrunk = shrunk + self.min_scale**2 * np.eye(arr.shape[1])
        self.precision = np.linalg.pinv(shrunk, hermitian=True)

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            raise ValueError("detector is not fitted")
        arr = _check_matrix(x, "attempts", min_rows=1, n_features=self.mean.shape[0])
        delta = arr - self.mean
        # the clip only removes round-off: the precision matrix is positive semi-definite
        return np.sqrt(np.clip(np.sum((delta @ self.precision) * delta, axis=1), 0.0, None))


def detector_scores(
    detector: Detector,
    data: KeystrokeData,
    n_enroll: int = 10,
    impostors_per_user: int = 5,
) -> Scores:
    """Score `data` with the same enrollment and impostors as verification_scores."""

    def score_user(enroll: np.ndarray, genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        detector.fit(enroll)
        return detector.score(genuine), detector.score(impostor)

    return score_protocol(timing_vectors(data), data.user_ids, score_user, n_enroll, impostors_per_user)

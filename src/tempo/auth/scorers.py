"""Classic detectors as per-user scorers the engine can persist (NumPy only).

The encoder maps an attempt to an embedding with no knowledge of the user, so
the engine only stores a template. Scaled Manhattan and Mahalanobis instead
need per-user statistics (mean, spread or precision matrix) at scoring time.
This module splits them into:

    state = scorer.fit(enroll_vectors)        at enrollment, (N, D) -> ScorerState
    score = scorer.score(state, vector)       at login, (D,) -> float, lower = more genuine
    state.to_json() / ScorerState.from_json() plain JSON the store can keep in a TEXT column

Vectors are raw timing vectors in seconds (baselines.timing_vectors layout);
`timing_vector` builds one from a validated (n, 3) timings array, so
D = 3n - 2. The numbers are exactly those of tempo.baselines, which
docs/RESULTS.md evaluates.

Per-user thresholds for these scores: `scorer_threshold(scorer, enroll, params)`
with params from tempo.thresholds, fitted on validation users for THIS
scorer (score scales differ between scorers and grow with phrase length).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tempo.baselines import MIN_SCALE_S, Mahalanobis, ScaledManhattan
from tempo.config import N_CHANNELS
from tempo.thresholds import PerUserThresholdParams, leave_one_out_scores, user_threshold

# Arrays each kind of state must hold; "vector" = shape (D,), "matrix" = shape (D, D).
_STATE_LAYOUT: dict[str, dict[str, str]] = {
    "scaled_manhattan": {"mean": "vector", "scale": "vector"},
    "mahalanobis": {"mean": "vector", "precision": "matrix"},
}


def timing_vector(timings: np.ndarray) -> np.ndarray:
    """One (n, 3) timings array in seconds -> (3n - 2,) float64 raw timing vector.

    Layout: hold, dd, ud for keys 0..n-2, then the last key's hold (the CMU column
    order, identical to one row of baselines.timing_vectors). The last row's dd/ud
    are undefined and are dropped. Pass timings that went through
    auth.validation.validate_timings; this only checks shape and finiteness.
    """
    arr = np.asarray(timings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != N_CHANNELS or arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
        raise ValueError(f"timings must be a finite (n, {N_CHANNELS}) array with n >= 2, got shape {arr.shape}")
    return np.concatenate([arr[:-1].reshape(-1), arr[-1:, 0]])


@dataclass(frozen=True)
class ScorerState:
    """One user's fitted statistics. Plain float64 arrays, so they serialize losslessly to JSON."""

    kind: str
    arrays: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        layout = _STATE_LAYOUT.get(self.kind)
        if layout is None or set(self.arrays) != set(layout):
            raise ValueError(f"state of kind {self.kind!r} must have arrays {layout}, got {sorted(self.arrays)}")
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.arrays.items()}
        d = -1
        if arrays["mean"].ndim == 1:
            d = arrays["mean"].shape[0]
        for name, shape_kind in layout.items():
            want: tuple[int, ...] = (d,)
            if shape_kind == "matrix":
                want = (d, d)
            if d < 1 or arrays[name].shape != want or not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"state array {name!r} must be finite with shape {want}, got {arrays[name].shape}")
        object.__setattr__(self, "arrays", arrays)

    @property
    def n_features(self) -> int:
        return int(self.arrays["mean"].shape[0])

    def to_dict(self) -> dict:
        return {"kind": self.kind, "arrays": {k: v.tolist() for k, v in self.arrays.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ScorerState":
        if not isinstance(d, dict) or not isinstance(d.get("kind"), str) or not isinstance(d.get("arrays"), dict):
            raise ValueError("scorer state must be {'kind': str, 'arrays': {name: list}}")
        return cls(d["kind"], {k: np.asarray(v, dtype=np.float64) for k, v in d["arrays"].items()})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "ScorerState":
        try:
            d = json.loads(s)
        except (TypeError, ValueError) as e:
            raise ValueError(f"scorer state is not valid JSON: {e}") from None
        return cls.from_dict(d)


class UserScorer(Protocol):
    kind: str

    def fit(self, enroll: np.ndarray) -> ScorerState:
        """(N, D) enrollment vectors -> one user's state."""

    def score(self, state: ScorerState, vector: np.ndarray) -> float:
        """(D,) vector -> anomaly score, lower = more likely genuine."""


def _check_vector(state: ScorerState, kind: str, vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if state.kind != kind or arr.shape != (state.n_features,) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"need a {kind} state and a finite vector of shape ({state.n_features},); "
            f"got state {state.kind!r} and vector shape {arr.shape}"
        )
    return arr[None, :]


@dataclass(frozen=True)
class ScaledManhattanScorer:
    min_scale: float = MIN_SCALE_S
    kind: str = "scaled_manhattan"

    def fit(self, enroll: np.ndarray) -> ScorerState:
        det = ScaledManhattan(min_scale=self.min_scale)
        det.fit(enroll)
        return ScorerState(self.kind, {"mean": det.mean, "scale": det.scale})

    def score(self, state: ScorerState, vector: np.ndarray) -> float:
        x = _check_vector(state, self.kind, vector)
        det = ScaledManhattan(min_scale=self.min_scale, mean=state.arrays["mean"], scale=state.arrays["scale"])
        return float(det.score(x)[0])


@dataclass(frozen=True)
class MahalanobisScorer:
    shrinkage: float = 0.5
    min_scale: float = MIN_SCALE_S
    kind: str = "mahalanobis"

    def fit(self, enroll: np.ndarray) -> ScorerState:
        det = Mahalanobis(shrinkage=self.shrinkage, min_scale=self.min_scale)
        det.fit(enroll)
        return ScorerState(self.kind, {"mean": det.mean, "precision": det.precision})

    def score(self, state: ScorerState, vector: np.ndarray) -> float:
        x = _check_vector(state, self.kind, vector)
        det = Mahalanobis(
            shrinkage=self.shrinkage, min_scale=self.min_scale, mean=state.arrays["mean"], precision=state.arrays["precision"]
        )
        return float(det.score(x)[0])


def enrollment_loo_scores(scorer: UserScorer, enroll: np.ndarray) -> np.ndarray:
    """(N,) leave-one-out scores: each enrollment vector against a state fitted on the other N - 1."""

    def score_fn(fit_rows: np.ndarray, test_rows: np.ndarray) -> np.ndarray:
        return np.array([scorer.score(scorer.fit(fit_rows), test_rows[0])])

    return leave_one_out_scores(enroll, score_fn)


def scorer_threshold(scorer: UserScorer, enroll: np.ndarray, params: PerUserThresholdParams) -> float:
    """Per-user accept threshold on this scorer's scale, from the user's (N, D) enrollment vectors."""
    return user_threshold(enrollment_loo_scores(scorer, enroll), params)

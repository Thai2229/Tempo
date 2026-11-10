"""Per-user accept thresholds set at enrollment (NumPy only).

One global threshold treats a steady typist and an erratic one the same, so
the erratic one is rejected often. Here every user gets their own threshold,
computed from ONLY their own enrollment attempts plus a few global numbers
that were fitted on validation users.

The rule, for any score where lower = more likely genuine (encoder distance,
scaled Manhattan, Mahalanobis):

    loo_i      = score of enrollment attempt i against a model fitted on the
                 other N - 1 enrollment attempts (leave-one-out)
    spread_u   = median_i loo_i                     the user's own consistency
    ref_u      = weight * spread_u + (1 - weight) * reference_spread
    ref_u      = clip(ref_u, min_ratio * reference_spread, max_ratio * reference_spread)
    threshold_u = scale * ref_u

weight = 0 is the old global threshold (scale * reference_spread for
everyone); weight = 1 makes the threshold proportional to the user's own
spread. reference_spread is the median spread of the validation users, and
scale and weight are chosen on validation users (scale at the EER of the
pooled scores divided by ref_u). The clip is a fixed safety bound: one
sloppy enrollment cannot buy a user an arbitrarily loose threshold.

Accepting `score <= scale * ref_u` is the same as accepting
`score / ref_u <= scale`, so the rule is evaluated by normalizing every
user's scores by ref_u and reusing the global metrics in keyauth.evaluate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from keyauth.evaluate import Scores, compute_eer, make_template

WEIGHT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_RATIO = 0.5
MAX_RATIO = 2.0
META_KEY = "per_user_threshold"  # key in the checkpoint meta and model.json
# serialized fields and their defaults (None = required)
_FIELDS: dict[str, float | None] = {
    "scale": None,
    "weight": None,
    "reference_spread": None,
    "min_ratio": MIN_RATIO,
    "max_ratio": MAX_RATIO,
}


@dataclass(frozen=True)
class PerUserThresholdParams:
    scale: float  # threshold = scale * ref_u
    weight: float  # 0 = global threshold, 1 = proportional to the user's own spread
    reference_spread: float  # median enrollment spread of the validation users (score units)
    min_ratio: float = MIN_RATIO  # ref_u is clipped to [min_ratio, max_ratio] * reference_spread
    max_ratio: float = MAX_RATIO

    def __post_init__(self) -> None:
        values = (self.scale, self.weight, self.reference_spread, self.min_ratio, self.max_ratio)
        if (
            not all(np.isfinite(v) for v in values)
            or self.scale <= 0
            or self.weight < 0
            or self.weight > 1
            or self.reference_spread <= 0
            or self.min_ratio <= 0
            or self.max_ratio < self.min_ratio
        ):
            raise ValueError(f"invalid per-user threshold parameters: {self}")

    @property
    def global_threshold(self) -> float:
        """The threshold of a user whose spread equals the reference (and of everyone when weight = 0)."""
        return float(self.scale * self.reference_spread)

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "PerUserThresholdParams":
        """Inverse of to_dict. min_ratio / max_ratio default to the module constants if absent."""
        if not isinstance(d, Mapping) or any(k not in d for k in ("scale", "weight", "reference_spread")):
            raise ValueError(f"per-user threshold params need scale, weight and reference_spread, got {d!r}")
        try:
            values = {k: float(d.get(k, default)) for k, default in _FIELDS.items()}  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise ValueError(f"per-user threshold params must be numbers: {e}") from None
        return cls(**values)

    @classmethod
    def from_meta(cls, meta: Mapping[str, object]) -> "PerUserThresholdParams | None":
        """Params stored in checkpoint meta, or None for checkpoints written before C6b."""
        d = meta.get(META_KEY)
        if d is None:
            return None
        return cls.from_dict(d)  # type: ignore[arg-type]


def leave_one_out_scores(enroll: np.ndarray, score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """(N,) score of each enrollment row against a model fitted on the other N - 1 rows.

    score_fn(fit_rows (N-1, D), test_rows (1, D)) -> (1,) scores; it fits and scores
    in one call, so the same helper works for the encoder and the classic detectors.
    """
    arr = np.asarray(enroll, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"enrollment must be a finite (N, D) array with N >= 3, got shape {arr.shape}")
    out = np.empty(arr.shape[0])
    for i in range(arr.shape[0]):
        rest = np.delete(arr, i, axis=0)
        out[i] = float(np.asarray(score_fn(rest, arr[i : i + 1])).reshape(-1)[0])
    return out


def embedding_loo_scores(enroll_embeddings: np.ndarray) -> np.ndarray:
    """Leave-one-out distances for the encoder: held-out embedding to the template of the others."""

    def score_fn(fit_rows: np.ndarray, test_rows: np.ndarray) -> np.ndarray:
        return np.linalg.norm(test_rows - make_template(fit_rows), axis=1)

    return leave_one_out_scores(enroll_embeddings, score_fn)


def enrollment_spread(loo_scores: np.ndarray) -> float:
    """spread_u: median leave-one-out score (robust to one sloppy enrollment attempt)."""
    arr = np.asarray(loo_scores, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 1 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"leave-one-out scores must be a non-empty finite non-negative 1-D array, got {arr!r}")
    return float(np.median(arr))


def reference_for(spread: float, params: PerUserThresholdParams) -> float:
    """ref_u from the module docstring; threshold_u = params.scale * ref_u."""
    if not np.isfinite(spread) or spread < 0:
        raise ValueError(f"spread must be finite and >= 0, got {spread}")
    ref = params.weight * spread + (1.0 - params.weight) * params.reference_spread
    return float(np.clip(ref, params.min_ratio * params.reference_spread, params.max_ratio * params.reference_spread))


def user_threshold(loo_scores: np.ndarray, params: PerUserThresholdParams) -> float:
    """Accept threshold for one user from their (N,) leave-one-out enrollment scores."""
    return float(params.scale * reference_for(enrollment_spread(loo_scores), params))


def embedding_threshold(enroll_embeddings: np.ndarray, params: PerUserThresholdParams) -> float:
    """Encoder shortcut: (N, D) enrollment embeddings -> accept threshold on the L2 distance."""
    return user_threshold(embedding_loo_scores(enroll_embeddings), params)


# ---- fitting and evaluation (validation / test users) --------------------


def normalize_scores(scores: Scores, references: Mapping[str, float]) -> Scores:
    """Divide each user's genuine and impostor scores by their ref_u (a positive number per user)."""
    if set(references) != set(scores.per_user) or not all(np.isfinite(r) and r > 0 for r in references.values()):
        raise ValueError("need one finite positive reference per user in scores.per_user")
    out = Scores(genuine=np.empty(0), impostor=np.empty(0))
    for u, (gen, imp) in scores.per_user.items():
        out.per_user[u] = (np.asarray(gen) / references[u], np.asarray(imp) / references[u])
    out.genuine = np.concatenate([g for g, _ in out.per_user.values()])
    out.impostor = np.concatenate([i for _, i in out.per_user.values()])
    return out


def fit_params(
    scores: Scores,
    spreads: Mapping[str, float],
    weights: Sequence[float] = WEIGHT_GRID,
    min_ratio: float = MIN_RATIO,
    max_ratio: float = MAX_RATIO,
) -> tuple[PerUserThresholdParams, dict[float, float]]:
    """Fit the rule on VALIDATION users only.

    scores: val verification scores; spreads: each val user's enrollment_spread.
    reference_spread = median spread; for each weight, scale = EER threshold of the
    normalized scores; the weight with the lowest val EER wins (ties go to the
    smaller weight, i.e. closer to the global threshold).
    Returns (params, {weight: val EER}).
    """
    if set(spreads) != set(scores.per_user) or len(weights) < 1:
        raise ValueError("need one spread per user in scores.per_user and at least one weight")
    reference_spread = float(np.median(list(spreads.values())))
    if not np.isfinite(reference_spread) or reference_spread <= 0:
        raise ValueError(f"median enrollment spread must be positive, got {reference_spread}")
    best: PerUserThresholdParams | None = None
    best_eer = float("inf")
    eers: dict[float, float] = {}
    for w in sorted(weights):
        probe = PerUserThresholdParams(1.0, float(w), reference_spread, min_ratio, max_ratio)
        refs = {u: reference_for(spreads[u], probe) for u in scores.per_user}
        normalized = normalize_scores(scores, refs)
        res = compute_eer(normalized.genuine, normalized.impostor)
        eers[float(w)] = res.eer
        if res.eer < best_eer and res.threshold > 0:
            best_eer = res.eer
            best = PerUserThresholdParams(res.threshold, float(w), reference_spread, min_ratio, max_ratio)
    if best is None:
        raise ValueError("no weight gave a positive threshold")
    return best, eers


def references(spreads: Mapping[str, float], params: PerUserThresholdParams) -> dict[str, float]:
    return {u: reference_for(s, params) for u, s in spreads.items()}

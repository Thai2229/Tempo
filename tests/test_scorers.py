import json

import numpy as np
import pytest

from keyauth.auth.scorers import (
    MahalanobisScorer,
    ScaledManhattanScorer,
    ScorerState,
    enrollment_loo_scores,
    scorer_threshold,
    timing_vector,
)
from keyauth.auth.validation import validate_timings
from keyauth.baselines import Mahalanobis, ScaledManhattan, timing_vectors
from keyauth.data import KeystrokeData
from keyauth.thresholds import PerUserThresholdParams, leave_one_out_scores, user_threshold

SCORERS = [ScaledManhattanScorer(), MahalanobisScorer(shrinkage=0.5)]


def attempts(rng: np.random.Generator, n: int, n_keys: int = 5, offset: float = 0.0) -> list[np.ndarray]:
    """Validated (n_keys, 3) timings in seconds with dd = hold + ud."""
    out = []
    for _ in range(n):
        hold = 0.08 + offset + 0.01 * rng.random(n_keys)
        ud = 0.12 + offset + 0.02 * rng.random(n_keys - 1)
        dd = hold[:-1] + ud
        out.append(validate_timings(np.column_stack([hold, np.append(dd, 0.0), np.append(ud, 0.0)])))
    return out


# ---- timing_vector -------------------------------------------------------


def test_timing_vector_hand_example():
    t = np.array([[0.1, 0.3, 0.2], [0.4, 0.9, 0.5], [0.6, 0.0, 0.0]])
    np.testing.assert_allclose(timing_vector(t), [0.1, 0.3, 0.2, 0.4, 0.9, 0.5, 0.6])


def test_timing_vector_matches_baselines_layout():
    rng = np.random.default_rng(0)
    seqs = attempts(rng, 4, n_keys=11)
    batch = timing_vectors(KeystrokeData.from_arrays(seqs, ["u"] * 4))
    single = np.stack([timing_vector(t) for t in seqs])
    assert single.shape == (4, 31) and single.dtype == np.float64
    np.testing.assert_array_equal(single, batch)


def test_timing_vector_ignores_last_row_latencies():
    t = np.array([[0.1, 0.3, 0.2], [0.4, 9.0, 9.0]])
    np.testing.assert_allclose(timing_vector(t), [0.1, 0.3, 0.2, 0.4])


@pytest.mark.parametrize(
    "bad",
    [np.zeros((1, 3)), np.zeros((4, 2)), np.zeros(6), np.zeros((2, 3, 1)), np.array([[0.1, 0.3, 0.2], [np.nan, 0.0, 0.0]])],
)
def test_timing_vector_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        timing_vector(bad)


# ---- fit / score ---------------------------------------------------------


def test_scores_match_the_baseline_detectors():
    rng = np.random.default_rng(1)
    enroll = np.stack([timing_vector(t) for t in attempts(rng, 10)])
    probes = np.stack([timing_vector(t) for t in attempts(rng, 3, offset=0.01)])
    for scorer, det in ((ScaledManhattanScorer(), ScaledManhattan()), (MahalanobisScorer(shrinkage=0.3), Mahalanobis(shrinkage=0.3))):
        det.fit(enroll)
        state = scorer.fit(enroll)
        np.testing.assert_allclose([scorer.score(state, p) for p in probes], det.score(probes))


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s.kind)
def test_genuine_scores_lower_than_impostor(scorer):
    rng = np.random.default_rng(2)
    enroll = np.stack([timing_vector(t) for t in attempts(rng, 10)])
    state = scorer.fit(enroll)
    genuine = scorer.score(state, timing_vector(attempts(rng, 1)[0]))
    impostor = scorer.score(state, timing_vector(attempts(rng, 1, offset=0.05)[0]))
    assert 0.0 <= genuine < impostor


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s.kind)
def test_state_roundtrips_through_json(scorer):
    rng = np.random.default_rng(3)
    enroll = np.stack([timing_vector(t) for t in attempts(rng, 10)])
    probe = timing_vector(attempts(rng, 1)[0])
    state = scorer.fit(enroll)
    restored = ScorerState.from_json(state.to_json())
    assert restored.kind == state.kind and restored.n_features == 13
    assert scorer.score(restored, probe) == scorer.score(state, probe)  # float64 JSON is lossless


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s.kind)
@pytest.mark.parametrize("bad", [np.zeros((1, 13)), np.zeros(13), np.full((10, 13), np.nan), np.zeros((10, 0))])
def test_fit_rejects_bad_enrollment(scorer, bad):
    with pytest.raises(ValueError):
        scorer.fit(bad)


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s.kind)
@pytest.mark.parametrize("bad", [np.zeros(12), np.zeros(14), np.zeros((1, 13)), np.array([np.nan] + [0.1] * 12)])
def test_score_rejects_wrong_length_or_nan(scorer, bad):
    rng = np.random.default_rng(4)
    state = scorer.fit(np.stack([timing_vector(t) for t in attempts(rng, 10)]))
    with pytest.raises(ValueError):
        scorer.score(state, bad)


def test_score_rejects_a_state_of_another_kind():
    rng = np.random.default_rng(5)
    enroll = np.stack([timing_vector(t) for t in attempts(rng, 10)])
    with pytest.raises(ValueError):
        MahalanobisScorer().score(ScaledManhattanScorer().fit(enroll), enroll[0])


def test_mahalanobis_rejects_bad_shrinkage():
    with pytest.raises(ValueError):
        MahalanobisScorer(shrinkage=1.5).fit(np.random.default_rng(0).random((10, 4)))


# ---- state validation (unfitted / corrupted states) ------------------------


@pytest.mark.parametrize(
    "kind, arrays",
    [
        ("nope", {"mean": [0.1], "scale": [0.1]}),
        ("scaled_manhattan", {}),  # unfitted: no statistics
        ("scaled_manhattan", {"mean": [0.1, 0.2]}),
        ("scaled_manhattan", {"mean": [0.1, 0.2], "scale": [0.1]}),
        ("scaled_manhattan", {"mean": [0.1, 0.2], "scale": [0.1, float("nan")]}),
        ("scaled_manhattan", {"mean": [], "scale": []}),
        ("scaled_manhattan", {"mean": [[0.1]], "scale": [[0.1]]}),
        ("mahalanobis", {"mean": [0.1, 0.2], "precision": [1.0, 2.0]}),
        ("mahalanobis", {"mean": [0.1, 0.2], "precision": [[1.0, 0.0], [0.0, float("inf")]]}),
        ("mahalanobis", {"mean": [0.1, 0.2], "scale": [0.1, 0.1]}),
    ],
)
def test_state_rejects_bad_contents(kind, arrays):
    with pytest.raises(ValueError):
        ScorerState(kind, {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})
    with pytest.raises(ValueError):
        ScorerState.from_json(json.dumps({"kind": kind, "arrays": arrays}))


@pytest.mark.parametrize("bad", ["not json", "[]", '{"kind": 3, "arrays": {}}', '{"kind": "mahalanobis"}', "null"])
def test_from_json_rejects_malformed_payloads(bad):
    with pytest.raises(ValueError):
        ScorerState.from_json(bad)


# ---- per-user thresholds on detector scores ---------------------------------


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s.kind)
def test_enrollment_loo_scores_refit_without_the_held_out_row(scorer):
    rng = np.random.default_rng(6)
    enroll = np.stack([timing_vector(t) for t in attempts(rng, 6)])
    loo = enrollment_loo_scores(scorer, enroll)
    for i in range(6):
        rest = np.delete(enroll, i, axis=0)
        assert loo[i] == pytest.approx(scorer.score(scorer.fit(rest), enroll[i]))
    params = PerUserThresholdParams(scale=1.7, weight=0.5, reference_spread=float(np.median(loo)))
    assert scorer_threshold(scorer, enroll, params) == pytest.approx(user_threshold(loo, params))


def test_enrollment_loo_needs_three_attempts():
    with pytest.raises(ValueError):
        enrollment_loo_scores(ScaledManhattanScorer(), np.random.default_rng(0).random((2, 4)))
    with pytest.raises(ValueError):
        leave_one_out_scores(np.random.default_rng(0).random((2, 4)), lambda a, b: np.zeros(1))

import json

import numpy as np
import pytest

from keyauth.evaluate import Scores, make_template, summarize
from keyauth.thresholds import (
    PerUserThresholdParams,
    embedding_loo_scores,
    embedding_threshold,
    enrollment_spread,
    fit_params,
    leave_one_out_scores,
    normalize_scores,
    reference_for,
    references,
    user_threshold,
)

PARAMS = PerUserThresholdParams(scale=2.0, weight=0.5, reference_spread=1.0, min_ratio=0.5, max_ratio=2.0)


# ---- params --------------------------------------------------------------


def test_params_roundtrip_through_json():
    d = json.loads(json.dumps(PARAMS.to_dict()))
    assert PerUserThresholdParams.from_dict(d) == PARAMS
    assert PARAMS.global_threshold == pytest.approx(2.0)


def test_from_dict_defaults_the_clip_ratios():
    p = PerUserThresholdParams.from_dict({"scale": 1.0, "weight": 0.0, "reference_spread": 0.3})
    assert (p.min_ratio, p.max_ratio) == (0.5, 2.0)


def test_from_meta_returns_none_for_old_checkpoints():
    assert PerUserThresholdParams.from_meta({"threshold": 0.7, "metrics": {}}) is None
    assert PerUserThresholdParams.from_meta({"threshold": 0.7, "per_user_threshold": None}) is None
    assert PerUserThresholdParams.from_meta({"per_user_threshold": PARAMS.to_dict()}) == PARAMS


@pytest.mark.parametrize(
    "bad",
    [
        {"scale": 0.0},
        {"scale": -1.0},
        {"scale": float("nan")},
        {"weight": -0.1},
        {"weight": 1.1},
        {"reference_spread": 0.0},
        {"reference_spread": float("inf")},
        {"min_ratio": 0.0},
        {"min_ratio": 3.0, "max_ratio": 2.0},
    ],
)
def test_params_reject_bad_values(bad):
    kwargs = {**PARAMS.to_dict(), **bad}
    with pytest.raises(ValueError):
        PerUserThresholdParams(**kwargs)


@pytest.mark.parametrize(
    "bad",
    [
        {"scale": 1.0},
        {"weight": 0.0, "reference_spread": 1.0},
        None,
        [1, 2, 3],
        {"scale": None, "weight": 0.0, "reference_spread": 1.0},
        {"scale": "x", "weight": 0.0, "reference_spread": 1.0},
    ],
)
def test_from_dict_rejects_missing_keys(bad):
    with pytest.raises(ValueError):
        PerUserThresholdParams.from_dict(bad)


# ---- the rule ------------------------------------------------------------


def test_reference_blends_and_clips():
    assert reference_for(1.0, PARAMS) == pytest.approx(1.0)
    assert reference_for(1.6, PARAMS) == pytest.approx(1.3)  # 0.5 * 1.6 + 0.5 * 1.0
    assert reference_for(100.0, PARAMS) == pytest.approx(2.0)  # clipped to max_ratio * reference_spread
    assert reference_for(0.0, PARAMS) == pytest.approx(0.5)  # 0.5 * 0 + 0.5 = 0.5, at the floor
    glob = PerUserThresholdParams(scale=2.0, weight=0.0, reference_spread=1.0)
    assert reference_for(0.1, glob) == reference_for(1.9, glob) == pytest.approx(1.0)  # weight 0 = global threshold


@pytest.mark.parametrize("spread", [float("nan"), float("inf"), -0.1])
def test_reference_rejects_bad_spread(spread):
    with pytest.raises(ValueError):
        reference_for(spread, PARAMS)


def test_user_threshold_uses_median_loo_score():
    # median of [1, 1.6, 9] = 1.6 -> ref 1.3 -> threshold 2.6; the outlier 9 does not matter
    assert user_threshold(np.array([1.0, 9.0, 1.6]), PARAMS) == pytest.approx(2.6)


@pytest.mark.parametrize("bad", [np.array([]), np.array([[1.0, 2.0]]), np.array([1.0, np.nan]), np.array([1.0, -1.0])])
def test_enrollment_spread_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        enrollment_spread(bad)


def test_leave_one_out_scores_hold_out_each_row():
    enroll = np.array([[0.0], [1.0], [2.0], [6.0]])

    def score_fn(fit_rows, test_rows):
        return np.abs(test_rows[:, 0] - fit_rows[:, 0].mean())

    # held out 0: mean(1,2,6)=3 -> 3; 1: mean(0,2,6)=8/3; 2: mean(0,1,6)=7/3 -> 1/3; 6: mean(0,1,2)=1 -> 5
    np.testing.assert_allclose(leave_one_out_scores(enroll, score_fn), [3.0, 5 / 3, 1 / 3, 5.0])


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 3)), np.zeros(5), np.zeros((4, 0)), np.array([[0.0, 1.0], [np.nan, 1.0], [0.0, 2.0]]), np.zeros((3, 2, 1))],
)
def test_leave_one_out_rejects_bad_enrollment(bad):
    with pytest.raises(ValueError):
        leave_one_out_scores(bad, lambda a, b: np.zeros(1))


def test_embedding_loo_matches_manual_template():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 4))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    loo = embedding_loo_scores(z)
    expected = [np.linalg.norm(z[i] - make_template(np.delete(z, i, axis=0))) for i in range(5)]
    np.testing.assert_allclose(loo, expected)
    assert embedding_threshold(z, PARAMS) == pytest.approx(user_threshold(loo, PARAMS))


def test_tighter_enrollment_gets_a_tighter_threshold():
    rng = np.random.default_rng(1)
    center = np.ones(8) / np.sqrt(8)
    tight = center + 0.02 * rng.normal(size=(10, 8))
    loose = center + 0.20 * rng.normal(size=(10, 8))
    p = PerUserThresholdParams(scale=1.5, weight=1.0, reference_spread=0.3, min_ratio=0.01, max_ratio=100.0)
    assert embedding_threshold(tight, p) < embedding_threshold(loose, p)


# ---- fitting -------------------------------------------------------------


def _scores(per_user: dict) -> Scores:
    s = Scores(genuine=np.empty(0), impostor=np.empty(0), per_user=per_user)
    s.genuine = np.concatenate([g for g, _ in per_user.values()])
    s.impostor = np.concatenate([i for _, i in per_user.values()])
    return s


def test_normalize_scores_divides_per_user():
    s = _scores({"a": (np.array([1.0, 2.0]), np.array([4.0])), "b": (np.array([3.0]), np.array([6.0, 9.0]))})
    n = normalize_scores(s, {"a": 2.0, "b": 3.0})
    np.testing.assert_allclose(n.per_user["a"][0], [0.5, 1.0])
    np.testing.assert_allclose(n.impostor, [2.0, 2.0, 3.0])


@pytest.mark.parametrize("refs", [{"a": 1.0}, {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": float("nan")}, {"a": 1.0, "b": 1.0, "c": 1.0}])
def test_normalize_scores_rejects_bad_references(refs):
    s = _scores({"a": (np.array([1.0]), np.array([2.0])), "b": (np.array([1.0]), np.array([2.0]))})
    with pytest.raises(ValueError):
        normalize_scores(s, refs)


def test_fit_params_prefers_per_user_when_users_differ_in_scale():
    # user "b" types twice as loosely as "a": same shape, every score doubled.
    # One global threshold cannot separate both; thresholds proportional to spread can.
    rng = np.random.default_rng(0)
    per_user, spreads = {}, {}
    for u, k in (("a", 1.0), ("b", 2.0), ("c", 1.0), ("d", 2.0)):
        per_user[u] = (k * rng.uniform(0.5, 1.0, 50), k * rng.uniform(1.1, 2.0, 50))
        spreads[u] = 0.7 * k
    params, eers = fit_params(_scores(per_user), spreads)
    assert params.weight == 1.0
    assert eers[1.0] == 0.0 and eers[0.0] > 0.0
    assert params.reference_spread == pytest.approx(np.median(list(spreads.values())))
    refs = references(spreads, params)
    assert summarize(normalize_scores(_scores(per_user), refs), threshold=params.scale)["far"] == 0.0


def test_fit_params_ties_go_to_the_global_threshold():
    per_user = {"a": (np.array([0.1, 0.2]), np.array([0.9])), "b": (np.array([0.1]), np.array([0.8, 0.9]))}
    params, _ = fit_params(_scores(per_user), {"a": 0.3, "b": 0.6})
    assert params.weight == 0.0


def test_fit_params_rejects_mismatched_users_and_bad_spreads():
    s = _scores({"a": (np.array([0.1]), np.array([0.9])), "b": (np.array([0.1]), np.array([0.9]))})
    with pytest.raises(ValueError):
        fit_params(s, {"a": 0.3})
    with pytest.raises(ValueError):
        fit_params(s, {"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError):
        fit_params(s, {"a": 0.3, "b": 0.3}, weights=[])

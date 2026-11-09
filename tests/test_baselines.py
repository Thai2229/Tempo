import numpy as np
import pytest

from keyauth.baselines import Mahalanobis, ScaledManhattan, detector_scores, timing_vectors
from keyauth.data import KeystrokeData
from keyauth.data.cmu import cmu_timing_columns
from keyauth.evaluate import summarize, verification_scores


def make_data(rng: np.random.Generator, n_users: int = 3, per_user: int = 30, noise: float = 0.002) -> KeystrokeData:
    """Users with well-separated hold and up-down times; dd = hold + ud, as in real captures."""
    seqs, users = [], []
    for u in range(n_users):
        for _ in range(per_user):
            hold = 0.08 + 0.03 * u + noise * rng.normal(size=4)
            ud = 0.10 + 0.05 * u + noise * rng.normal(size=3)
            dd = hold[:-1] + ud
            seqs.append(np.column_stack([hold, np.append(dd, 0.0), np.append(ud, 0.0)]))
            users.append(f"u{u}")
    return KeystrokeData.from_arrays(seqs, users)


# ---- timing_vectors -----------------------------------------------------


def test_timing_vectors_follow_cmu_column_order():
    t = np.array([[0.1, 0.3, 0.2], [0.4, 0.9, 0.5], [0.6, 0.0, 0.0]])
    data = KeystrokeData.from_arrays([t, t], ["a", "b"])
    v = timing_vectors(data)
    assert v.shape == (2, 7)
    np.testing.assert_allclose(v[0], [0.1, 0.3, 0.2, 0.4, 0.9, 0.5, 0.6], atol=1e-6)
    assert len(cmu_timing_columns()) == 3 * 11 - 2  # the same formula gives CMU's 31 columns


def test_timing_vectors_reject_mixed_lengths():
    short = np.array([[0.1, 0.3, 0.2], [0.1, 0.0, 0.0]])
    long = np.array([[0.1, 0.3, 0.2], [0.1, 0.3, 0.2], [0.1, 0.0, 0.0]])
    with pytest.raises(ValueError):
        timing_vectors(KeystrokeData.from_arrays([short, long], ["a", "b"]))


def test_timing_vectors_reject_empty():
    empty = KeystrokeData.from_arrays([], [])
    with pytest.raises(ValueError):
        timing_vectors(empty)


# ---- scaled Manhattan ---------------------------------------------------


def test_scaled_manhattan_hand_example():
    det = ScaledManhattan()
    det.fit(np.array([[0.1, 1.0], [0.3, 3.0]]))  # means (0.2, 2), mean abs deviations (0.1, 1)
    np.testing.assert_allclose(det.score(np.array([[0.2, 2.0], [0.4, 4.0]])), [0.0, 4.0])


def test_scaled_manhattan_constant_feature_stays_finite():
    det = ScaledManhattan()
    det.fit(np.array([[0.1, 0.5], [0.1, 0.7], [0.1, 0.6]]))  # the first feature never varies
    scores = det.score(np.array([[0.1, 0.6], [0.2, 0.6]]))
    assert np.all(np.isfinite(scores))
    assert scores[1] > scores[0]


@pytest.mark.parametrize(
    "enroll",
    [
        np.zeros(5),  # 1-D
        np.zeros((1, 3)),  # one attempt has no spread
        np.zeros((3, 0)),  # no features
        np.zeros((3, 2, 2)),  # 3-D
        np.array([[0.1, np.nan], [0.2, 0.3]]),
        np.array([[0.1, np.inf], [0.2, 0.3]]),
    ],
)
@pytest.mark.parametrize("make", [ScaledManhattan, Mahalanobis])
def test_fit_rejects_bad_enrollment(make, enroll):
    with pytest.raises(ValueError):
        make().fit(enroll)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(2),  # 1-D
        np.zeros((2, 3)),  # wrong number of features
        np.zeros((0, 2)),  # no attempts
        np.array([[np.nan, 0.1]]),
    ],
)
@pytest.mark.parametrize("make", [ScaledManhattan, Mahalanobis])
def test_score_rejects_bad_attempts(make, x):
    det = make()
    det.fit(np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.3]]))
    with pytest.raises(ValueError):
        det.score(x)


@pytest.mark.parametrize("make", [ScaledManhattan, Mahalanobis])
def test_score_before_fit_is_an_error(make):
    with pytest.raises(ValueError):
        make().score(np.zeros((1, 2)))


# ---- Mahalanobis --------------------------------------------------------


def test_mahalanobis_matches_textbook_when_covariance_is_full_rank():
    rng = np.random.default_rng(0)
    enroll = rng.normal(size=(200, 3)) * [0.01, 0.02, 0.03]
    x = rng.normal(size=(5, 3)) * 0.02
    det = Mahalanobis(shrinkage=0.0, min_scale=0.0)
    det.fit(enroll)
    delta = x - enroll.mean(axis=0)
    expected = np.sqrt(np.sum((delta @ np.linalg.inv(np.cov(enroll, rowvar=False))) * delta, axis=1))
    np.testing.assert_allclose(det.score(x), expected)


def test_mahalanobis_full_shrinkage_is_scaled_euclidean():
    enroll = np.array([[0.1, 1.0], [0.3, 3.0], [0.2, 2.5]])
    det = Mahalanobis(shrinkage=1.0, min_scale=0.0)
    det.fit(enroll)
    x = np.array([[0.5, 1.0]])
    expected = np.sqrt(np.sum((x - enroll.mean(axis=0)) ** 2 / enroll.var(axis=0, ddof=1), axis=1))
    np.testing.assert_allclose(det.score(x), expected)


def test_mahalanobis_singular_when_fewer_attempts_than_features():
    rng = np.random.default_rng(1)
    enroll = rng.normal(0.1, 0.01, size=(10, 31))
    assert np.linalg.matrix_rank(np.cov(enroll, rowvar=False)) < 31
    det = Mahalanobis(shrinkage=0.0)
    det.fit(enroll)
    assert np.all(np.isfinite(det.score(rng.normal(0.1, 0.01, size=(20, 31)))))


def test_mahalanobis_singular_from_dd_equals_hold_plus_ud():
    rng = np.random.default_rng(2)
    hold, ud = rng.normal(0.1, 0.01, size=(100, 1)), rng.normal(0.1, 0.01, size=(100, 1))
    enroll = np.hstack([hold, hold + ud, ud])  # exact linear dependence, however many attempts
    assert np.linalg.matrix_rank(np.cov(enroll, rowvar=False), tol=1e-10) == 2
    for shrinkage in (0.0, 0.5, 1.0):
        det = Mahalanobis(shrinkage=shrinkage)
        det.fit(enroll)
        assert np.all(np.isfinite(det.score(enroll[:5] + 0.01)))


def test_mahalanobis_identical_enrollment_is_finite_and_ordered():
    enroll = np.full((10, 4), 0.1)  # zero covariance: only the ridge is left
    det = Mahalanobis()
    det.fit(enroll)
    scores = det.score(np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.1], [0.3, 0.1, 0.1, 0.1]]))
    assert np.all(np.isfinite(scores))
    assert scores[0] == pytest.approx(0.0)
    assert scores[0] < scores[1] < scores[2]


@pytest.mark.parametrize("shrinkage", [-0.1, 1.5])
def test_mahalanobis_rejects_bad_shrinkage(shrinkage):
    with pytest.raises(ValueError):
        Mahalanobis(shrinkage=shrinkage).fit(np.array([[0.1, 0.2], [0.2, 0.3]]))


# ---- protocol -----------------------------------------------------------


@pytest.mark.parametrize("make", [ScaledManhattan, Mahalanobis])
def test_detector_scores_use_the_encoder_protocol(make):
    data = make_data(np.random.default_rng(3))
    vectors = timing_vectors(data)
    embedding_scores = verification_scores(vectors, data.user_ids, n_enroll=10, impostors_per_user=5)
    scores = detector_scores(make(), data, n_enroll=10, impostors_per_user=5)
    assert scores.genuine.shape == embedding_scores.genuine.shape
    assert scores.impostor.shape == embedding_scores.impostor.shape
    assert set(scores.per_user) == set(embedding_scores.per_user)
    assert scores.genuine.size == 3 * 20 and scores.impostor.size == 3 * 2 * 5


@pytest.mark.parametrize("make", [ScaledManhattan, Mahalanobis])
def test_detector_scores_separate_distinct_typists(make):
    summary = summarize(detector_scores(make(), make_data(np.random.default_rng(4))))
    assert summary["eer"] < 0.05


@pytest.mark.parametrize("make", [ScaledManhattan, Mahalanobis])
def test_detector_scores_cannot_tell_identical_typists_apart(make):
    rng = np.random.default_rng(5)
    seqs, users = [], []
    for u in range(4):
        for _ in range(60):
            hold = 0.1 + 0.01 * rng.normal(size=4)
            ud = 0.1 + 0.01 * rng.normal(size=3)
            seqs.append(np.column_stack([hold, np.append(hold[:-1] + ud, 0.0), np.append(ud, 0.0)]))
            users.append(f"u{u}")
    summary = summarize(detector_scores(make(), KeystrokeData.from_arrays(seqs, users)))
    assert summary["eer"] > 0.3


@pytest.mark.parametrize("n_enroll", [0, 30, 1])
def test_detector_scores_reject_unusable_enrollment_counts(n_enroll):
    data = make_data(np.random.default_rng(6))
    with pytest.raises(ValueError):
        detector_scores(ScaledManhattan(), data, n_enroll=n_enroll)

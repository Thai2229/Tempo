import numpy as np
import pytest

from keyauth.evaluate import (
    compute_eer,
    far_frr,
    make_template,
    roc_curve,
    summarize,
    verification_scores,
)


def test_eer_perfect_separation():
    r = compute_eer(np.array([0.1, 0.2]), np.array([0.5, 0.6]))
    assert r.eer == 0.0
    assert 0.2 <= r.threshold < 0.5


def test_eer_identical_distributions():
    s = np.arange(100, dtype=float)
    assert compute_eer(s, s).eer == pytest.approx(0.5)


def test_eer_hand_example():
    # at threshold 3.5: FRR = 1/4 (the 4), FAR = 1/4 (the 3.5)
    r = compute_eer(np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.5, 5.0, 6.0, 7.0]))
    assert r.eer == pytest.approx(0.25)
    assert r.threshold == pytest.approx(3.5)


def test_far_frr_and_roc():
    gen = np.array([1.0, 2.0, 3.0, 4.0])
    imp = np.array([3.5, 5.0, 6.0, 7.0])
    assert far_frr(gen, imp, 4.0) == (0.25, 0.0)
    thresholds, far, frr = roc_curve(gen, imp)
    assert np.all(np.diff(thresholds) > 0)
    assert np.all(np.diff(far) >= 0) and np.all(np.diff(frr) <= 0)
    assert far[0] == 0.0 and frr[0] == 1.0 and far[-1] == 1.0 and frr[-1] == 0.0


def test_make_template_is_unit_length():
    t = make_template(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.linalg.norm(t) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        make_template(np.array([[1.0, 0.0], [-1.0, 0.0]]))


def _clustered_embeddings(rng, centers, per_user, noise):
    emb, users = [], []
    for name, c in centers.items():
        pts = np.asarray(c) + noise * rng.normal(size=(per_user, len(c)))
        emb.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
        users += [name] * per_user
    return np.concatenate(emb), np.array(users, dtype=object)


def test_verification_scores_counts_and_separation():
    rng = np.random.default_rng(0)
    emb, users = _clustered_embeddings(rng, {"a": [1, 0], "b": [0, 1], "c": [-1, 0]}, per_user=15, noise=0.05)
    scores = verification_scores(emb, users, n_enroll=5, impostors_per_user=5)
    assert scores.genuine.size == 3 * 10
    assert scores.impostor.size == 3 * 2 * 5
    assert set(scores.per_user) == {"a", "b", "c"}
    summary = summarize(scores)
    assert summary["eer"] == 0.0 and summary["per_user_eer"] == 0.0
    assert summary["far"] == 0.0 and summary["frr"] == 0.0


def test_verification_scores_overlapping_users_has_errors():
    rng = np.random.default_rng(0)
    emb, users = _clustered_embeddings(rng, {"a": [1, 0], "b": [1, 0.05]}, per_user=30, noise=0.3)
    assert summarize(verification_scores(emb, users, n_enroll=5))["eer"] > 0.1


def test_verification_scores_validation():
    emb = np.eye(2)
    with pytest.raises(ValueError):
        verification_scores(emb, np.array(["a", "a"]), n_enroll=1)  # one user
    with pytest.raises(ValueError):
        verification_scores(np.eye(4), np.array(["a", "a", "b", "b"]), n_enroll=2)  # nothing left to test

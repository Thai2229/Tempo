import numpy as np

from keyauth.events import matches_phrase, pair_events
from keyauth.features import extract_features
from keyauth.synthetic import make_typist, simulate_attempt, to_raw_events

PHRASE = ".tie5Roanl"


def test_raw_roundtrip_through_pairing():
    rng = np.random.default_rng(0)
    typist = make_typist(PHRASE, rng)
    attempt = simulate_attempt(typist, rng, t0_ms=1000.0)
    paired = pair_events(to_raw_events(attempt))
    assert matches_phrase(paired, PHRASE)
    np.testing.assert_allclose(
        extract_features(paired).values, extract_features(attempt).values, atol=1e-5
    )


def test_same_typist_closer_than_different_typists():
    """The signal the model will learn: within-person distance < between-person distance."""
    rng = np.random.default_rng(1)
    typists = [make_typist(PHRASE, rng) for _ in range(20)]
    samples = {
        i: np.stack([np.log(extract_features(simulate_attempt(t, rng)).to_vector()[::3]) for _ in range(30)])
        for i, t in enumerate(typists)
    }
    within = np.mean([np.linalg.norm(s - s.mean(0), axis=1).mean() for s in samples.values()])
    means = np.stack([s.mean(0) for s in samples.values()])
    between = np.mean(
        [np.linalg.norm(means[i] - means[j]) for i in range(len(means)) for j in range(i + 1, len(means))]
    )
    assert within < between

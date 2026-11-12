import numpy as np
import pytest

from tempo.events import KeyEvent
from tempo.features import TimingSequence, extract_features


def test_timings_hand_computed():
    ks = [KeyEvent("a", 0, 100), KeyEvent("b", 250, 330), KeyEvent("c", 300, 400)]
    seq = extract_features(ks)
    expected = np.array(
        [
            [0.100, 0.250, 0.150],
            [0.080, 0.050, -0.030],  # rollover -> negative flight
            [0.100, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(seq.values, expected, atol=1e-6)
    assert seq.keys == ("a", "b", "c")
    assert seq.n_transitions == 2


def test_dd_equals_hold_plus_ud():
    ks = [KeyEvent("a", 0, 90), KeyEvent("b", 210, 280), KeyEvent("c", 400, 470)]
    seq = extract_features(ks)
    np.testing.assert_allclose(seq.dd, seq.hold[:-1] + seq.ud, atol=1e-6)


def test_vector_roundtrip_cmu_order():
    ks = [KeyEvent("a", 0, 100), KeyEvent("b", 250, 330), KeyEvent("c", 500, 600)]
    seq = extract_features(ks)
    vec = seq.to_vector()
    # H.a, DD.a.b, UD.a.b, H.b, DD.b.c, UD.b.c, H.c
    np.testing.assert_allclose(vec, [0.1, 0.25, 0.15, 0.08, 0.25, 0.17, 0.1], atol=1e-6)
    back = TimingSequence.from_vector(vec, seq.keys)
    np.testing.assert_allclose(back.values, seq.values, atol=1e-6)


def test_rejects_too_short_and_implausible():
    with pytest.raises(ValueError):
        extract_features([KeyEvent("a", 0, 100)])
    with pytest.raises(ValueError, match="latency"):
        extract_features([KeyEvent("a", 0, 100), KeyEvent("b", 60_000, 60_100)])
    # bounds check can be turned off for offline analysis
    extract_features([KeyEvent("a", 0, 100), KeyEvent("b", 60_000, 60_100)], check_bounds=False)


def test_from_vector_length_check():
    with pytest.raises(ValueError):
        TimingSequence.from_vector([0.1, 0.2], ("a", "b"))

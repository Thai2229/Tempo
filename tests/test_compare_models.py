"""The pure helpers of scripts/compare_models.py. The experiments themselves need the CMU file."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_models.py"


@pytest.fixture(scope="module")
def compare():
    spec = importlib.util.spec_from_file_location("compare_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_epoch_keeps_the_minimum(compare):
    assert compare.select_epoch([0.5, 0.2, 0.3, 0.4], patience=10) == 1


def test_select_epoch_stops_after_patience(compare):
    # 0.1 at index 4 is never reached: patience runs out after index 2
    assert compare.select_epoch([0.5, 0.6, 0.7, 0.8, 0.1], patience=2) == 0


def test_ema_smooths_a_spike(compare):
    out = compare.ema([1.0, 0.0, 1.0], alpha=0.5)
    assert out == pytest.approx([1.0, 0.5, 0.75])


def test_mean_std_uses_sample_std(compare):
    assert compare.mean_std([1.0, 3.0]) == pytest.approx((2.0, 2.0**0.5))
    assert compare.mean_std([4.0]) == (4.0, 0.0)


def test_enroll_rows_are_each_users_first_attempts(compare):
    import numpy as np

    rows = compare.enroll_rows(np.array(["b", "a", "b", "a", "b", "a"], dtype=object), n_enroll=2)
    assert {u: r.tolist() for u, r in rows.items()} == {"a": [1, 3], "b": [0, 2]}

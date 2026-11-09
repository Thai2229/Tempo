import numpy as np
import pandas as pd
import pytest

from keyauth.data import (
    CMU_KEYS,
    KeystrokeData,
    Normalizer,
    append_recording,
    load_cmu,
    load_recordings,
    make_synthetic_dataset,
    save_recordings,
    split_sessions,
    split_users,
)
from keyauth.data.cmu import cmu_timing_columns
from keyauth.data.preprocess import transform, valid_mask
from keyauth.features import TimingSequence


@pytest.fixture
def synth():
    return make_synthetic_dataset(n_users=10, samples_per_user=20, n_sessions=4, seed=0)


# ---- container ----------------------------------------------------------


def test_from_arrays_pads_variable_lengths():
    a = np.full((3, 3), 0.1)
    b = np.full((5, 3), 0.2)
    d = KeystrokeData.from_arrays([a, b], ["u1", "u2"])
    assert d.values.shape == (2, 5, 3)
    assert d.lengths.tolist() == [3, 5]
    assert np.all(d.values[0, 3:] == 0)
    assert d.sessions.tolist() == [-1, -1]
    assert d.users == ["u1", "u2"]


def test_subset_and_concat(synth):
    s0 = synth.for_users(["s000"])
    assert len(s0) == 20 and s0.users == ["s000"]
    both = KeystrokeData.concat([s0, synth.for_users(["s001"])])
    assert len(both) == 40 and both.users == ["s000", "s001"]


def test_rejects_bad_lengths():
    with pytest.raises(ValueError):
        KeystrokeData.from_arrays([np.zeros((1, 3))], ["u"])


# ---- CMU loader ---------------------------------------------------------


def test_cmu_columns_shape():
    cols = cmu_timing_columns()
    assert len(cols) == 3 * len(CMU_KEYS) - 2 == 31
    assert cols[:3] == ["H.period", "DD.period.t", "UD.period.t"]
    assert cols[-1] == "H.Return"


def test_load_cmu_roundtrip(tmp_path, synth):
    # write synthetic data in the CMU file format, read it back
    rows = []
    for i in range(len(synth)):
        vec = synth.sequence(i, CMU_KEYS).to_vector()
        rows.append([synth.user_ids[i], int(synth.sessions[i]) + 1, i % 50 + 1, *vec.tolist()])
    df = pd.DataFrame(rows, columns=["subject", "sessionIndex", "rep", *cmu_timing_columns()])
    path = tmp_path / "DSL-StrongPasswordData.csv"
    df.to_csv(path, index=False)

    loaded = load_cmu(path)
    assert len(loaded) == len(synth)
    np.testing.assert_allclose(loaded.values, synth.values, atol=1e-6)
    assert loaded.user_ids.tolist() == synth.user_ids.tolist()
    assert loaded.sessions.tolist() == (synth.sessions + 1).tolist()


def test_load_cmu_rejects_other_csv(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="CMU"):
        load_cmu(path)


# ---- recordings ---------------------------------------------------------


def test_recordings_roundtrip(tmp_path, synth):
    path = tmp_path / "rec.jsonl"
    save_recordings(path, synth)
    back = load_recordings(path)
    np.testing.assert_allclose(back.values, synth.values, atol=1e-6)
    assert back.user_ids.tolist() == synth.user_ids.tolist()
    assert back.sessions.tolist() == synth.sessions.tolist()


def test_append_recording_mixed_lengths(tmp_path):
    path = tmp_path / "rec.jsonl"
    append_recording(path, "alice", TimingSequence(("a", "b"), np.array([[0.1, 0.2, 0.1], [0.1, 0, 0]], np.float32)))
    append_recording(path, "bob", TimingSequence(("a", "b", "c"), np.full((3, 3), 0.1, np.float32)), session=2)
    d = load_recordings(path)
    assert d.lengths.tolist() == [2, 3]
    assert d.sessions.tolist() == [0, 2]
    # no typed characters are stored
    assert '"a"' not in path.read_text()


# ---- preprocessing ------------------------------------------------------


def test_valid_mask():
    m = valid_mask(np.array([2, 3]), 3)
    assert m[0, :, 0].tolist() == [True, True, False]
    assert m[0, :, 1].tolist() == [True, False, False]
    assert m[1, :, 2].tolist() == [True, True, False]


def test_transform_handles_negative_flight():
    values = np.array([[[0.1, 0.2, -0.05], [0.1, 0.0, 0.0]]])
    t = transform(values, np.array([2]))
    assert t[0, 0, 2] < 0
    assert np.all(np.isfinite(t))
    assert np.all(t[0, 1, 1:] == 0)  # last key's dd/ud masked


def test_normalizer_standardizes_and_masks(synth):
    norm = Normalizer.fit(synth)
    x = norm(synth)
    mask = valid_mask(synth.lengths, synth.max_len)
    for c in range(3):
        vals = x[..., c][mask[..., c]]
        assert abs(vals.mean()) < 1e-4
        assert abs(vals.std() - 1) < 1e-3
    assert np.all(x[~mask] == 0)
    again = Normalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(again(synth), x)


# ---- splits -------------------------------------------------------------


def test_split_users_disjoint(synth):
    train, val, test = split_users(synth, val_frac=0.2, test_frac=0.2, seed=1)
    assert len(train) + len(val) + len(test) == len(synth)
    assert not set(train.users) & set(val.users)
    assert not set(train.users) & set(test.users)
    assert not set(val.users) & set(test.users)
    assert len(val.users) == 2 and len(test.users) == 2


def test_split_users_too_few():
    d = make_synthetic_dataset(n_users=3, samples_per_user=5)
    with pytest.raises(ValueError):
        split_users(d, val_frac=0.4, test_frac=0.4)


def test_split_sessions(synth):
    early, late = split_sessions(synth, enroll_sessions=1)
    assert set(early.sessions.tolist()) == {0}
    assert set(late.sessions.tolist()) == {1, 2, 3}
    assert len(early) == 10 * 5

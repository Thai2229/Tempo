"""The engine driven by the real TorchEmbedder, loaded from a checkpoint the way the app will."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from keyauth.auth import EngineConfig, InvalidTimings, KeyAuthEngine, TemplateStore  # noqa: E402
from keyauth.auth.embedder import TorchEmbedder  # noqa: E402
from keyauth.data import make_synthetic_dataset, split_users  # noqa: E402
from keyauth.thresholds import PerUserThresholdParams  # noqa: E402
from keyauth.train import TrainConfig, load_checkpoint, save_checkpoint, train  # noqa: E402

MATCH_ANYTHING = 2.0  # the largest possible distance between unit vectors


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    data = make_synthetic_dataset(n_users=16, samples_per_user=30, seed=0)
    train_data, val_data, test_data = split_users(data, val_frac=0.25, test_frac=0.25, seed=0)
    cfg = TrainConfig(epochs=2, batches_per_epoch=10, p=4, k=4, n_enroll=5)
    model, normalizer, _ = train(train_data, val_data, cfg, log=lambda s: None)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    save_checkpoint(path, model, normalizer, threshold=MATCH_ANYTHING, metrics={})
    return path, test_data


def attempts(data, user: str, n: int) -> list[np.ndarray]:
    rows = np.flatnonzero(data.user_ids == user)[:n]
    return [data.values[i, : data.lengths[i]].copy() for i in rows]


def make_engine(checkpoint, **config) -> KeyAuthEngine:
    embedder, threshold = TorchEmbedder.from_checkpoint(checkpoint[0])
    return KeyAuthEngine(embedder, TemplateStore(), threshold, EngineConfig(enroll_samples=5, **config))


def enrolled(checkpoint, **config) -> tuple[KeyAuthEngine, list[np.ndarray]]:
    _, data = checkpoint
    engine = make_engine(checkpoint, **config)
    user = data.users[0]
    samples = attempts(data, user, 12)
    results = [engine.enroll("app", "alice", a) for a in samples[:5]]
    assert results[-1].status == "complete"
    return engine, samples


def test_embedder_returns_a_unit_vector(checkpoint):
    embedder, _ = TorchEmbedder.from_checkpoint(checkpoint[0])
    z = embedder(attempts(checkpoint[1], checkpoint[1].users[0], 1)[0])
    assert z.ndim == 1
    assert np.linalg.norm(z) == pytest.approx(1.0, abs=1e-5)


def test_genuine_attempt_is_accepted(checkpoint):
    engine, samples = enrolled(checkpoint)
    r = engine.verify("app", "alice", samples[6])
    assert r.accepted and r.reason == "match" and 0.0 <= r.score <= MATCH_ANYTHING


def test_replayed_enrollment_attempt_is_rejected(checkpoint):
    engine, samples = enrolled(checkpoint)
    r = engine.verify("app", "alice", samples[2])
    assert not r.accepted and r.reason == "replay"


def test_replayed_login_is_rejected(checkpoint):
    engine, samples = enrolled(checkpoint)
    assert engine.verify("app", "alice", samples[6]).accepted
    assert engine.verify("app", "alice", samples[6].copy()).reason == "replay"


def test_wrong_length_is_rejected(checkpoint):
    engine, samples = enrolled(checkpoint)
    r = engine.verify("app", "alice", samples[6][:-1])
    assert not r.accepted and r.reason == "length_mismatch"


def test_score_above_threshold_is_rejected_then_locks_out(checkpoint):
    engine, samples = enrolled(checkpoint, max_failures=2, lockout_s=60.0)
    engine.store.set_threshold("app", "alice", 1e-9)
    assert [engine.verify("app", "alice", a).reason for a in samples[6:9]] == ["no_match", "no_match", "locked"]


def test_malformed_timings_are_rejected_before_the_model(checkpoint):
    engine, _ = enrolled(checkpoint)
    with pytest.raises(InvalidTimings):
        engine.verify("app", "alice", [[0.1, float("nan"), 0.1], [0.1, 0.0, 0.0]])


def test_inconsistent_enrollment_restarts(checkpoint):
    _, data = checkpoint
    embedder, _ = TorchEmbedder.from_checkpoint(checkpoint[0])
    engine = KeyAuthEngine(embedder, TemplateStore(), threshold=1e-9, config=EngineConfig(enroll_samples=5))
    results = [engine.enroll("app", "alice", a) for a in attempts(data, data.users[0], 5)]
    assert results[-1].status == "rejected" and results[-1].reason == "inconsistent"


def test_per_user_params_from_checkpoint(checkpoint, tmp_path):
    """Old checkpoints (no params) give None, new ones give the saved rule; from_checkpoint is unchanged."""
    assert TorchEmbedder.per_user_params(checkpoint[0]) is None
    model, normalizer, _ = load_checkpoint(checkpoint[0])
    params = PerUserThresholdParams(scale=1.5, weight=0.5, reference_spread=0.2)
    path = tmp_path / "with_params.pt"
    save_checkpoint(path, model, normalizer, threshold=0.3, metrics={}, per_user_threshold=params)
    assert TorchEmbedder.per_user_params(path) == params
    embedder, threshold = TorchEmbedder.from_checkpoint(path)
    assert threshold == pytest.approx(0.3) and callable(embedder)


def test_engine_with_checkpoint_params_stores_a_user_threshold(checkpoint):
    path, data = checkpoint
    embedder, threshold = TorchEmbedder.from_checkpoint(path)
    params = PerUserThresholdParams(scale=1.0, weight=1.0, reference_spread=0.1)
    engine = KeyAuthEngine(embedder, TemplateStore(), threshold, EngineConfig(enroll_samples=5), per_user=params)
    results = [engine.enroll("app", "alice", a) for a in attempts(data, data.users[0], 5)]
    assert results[-1].status == "complete"
    stored = engine.store.get_user("app", "alice").threshold
    assert stored is not None and 0.05 <= stored <= 0.2  # clipped to [0.5, 2] * reference_spread


def test_checkpoint_that_is_not_keyauth_is_rejected(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError):
        TorchEmbedder.from_checkpoint(path)

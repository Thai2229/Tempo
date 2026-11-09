import numpy as np
import pytest

from keyauth.auth import EngineConfig, InvalidTimings, KeyAuthEngine, TemplateStore, validate_timings
from keyauth.evaluate import compute_eer, verification_scores
from keyauth.features import extract_features
from keyauth.synthetic import make_typist, simulate_attempt

PHRASE = "hunter2!"


def fake_embedder(timings: np.ndarray) -> np.ndarray:
    """Stand-in for the neural model: log hold + log dd, speed-normalized."""
    t = np.asarray(timings, dtype=np.float64)
    v = np.concatenate([np.log(np.maximum(t[:, 0], 1e-3)), np.log(np.maximum(t[:-1, 1], 1e-3))])
    v = v - v.mean()
    return v / np.linalg.norm(v)


class RandomEmbedder:
    """Ignores the input: every attempt looks like a different person."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, timings):
        v = self.rng.normal(size=32)
        return v / np.linalg.norm(v)


def attempts(typist, rng, n):
    return [extract_features(simulate_attempt(typist, rng)).values for _ in range(n)]


@pytest.fixture(scope="module")
def threshold():
    """Calibrate the fake embedder's threshold the way train.py does: EER on separate users."""
    rng = np.random.default_rng(100)
    emb, users = [], []
    for u in range(10):
        typist = make_typist(PHRASE, rng)
        for a in attempts(typist, rng, 30):
            emb.append(fake_embedder(a))
            users.append(f"cal{u}")
    scores = verification_scores(np.stack(emb), np.array(users), n_enroll=10)
    return compute_eer(scores.genuine, scores.impostor).threshold


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def engine(threshold, clock):
    config = EngineConfig(enroll_samples=10, max_failures=1000)
    return KeyAuthEngine(fake_embedder, TemplateStore(), threshold, config, clock=clock)


@pytest.fixture
def typists():
    rng = np.random.default_rng(7)
    return make_typist(PHRASE, rng), make_typist(PHRASE, rng), rng


def enroll_fully(engine, app, user, typist, rng):
    result = None
    for a in attempts(typist, rng, engine.config.enroll_samples):
        result = engine.enroll(app, user, a)
    return result


# ---- validation ---------------------------------------------------------


def test_validate_accepts_real_attempt(typists):
    alice, _, rng = typists
    a = attempts(alice, rng, 1)[0]
    out = validate_timings(a.tolist())
    assert out.shape == a.shape and out.dtype == np.float32


@pytest.mark.parametrize(
    "bad",
    [
        "nope",
        [[0.1, 0.2]],  # wrong width
        [[0.1, 0.2, 0.1]],  # one key
        [[0.1, 0.2, 0.1]] * 200,  # too many keys
        [[0.1, float("nan"), 0.1], [0.1, 0, 0]],
        [[-0.1, 0.2, 0.3], [0.1, 0, 0]],  # negative hold
        [[100.0, 200.0, 100.0], [0.1, 0, 0]],  # milliseconds, not seconds
        [[0.1, 0.5, 0.1], [0.1, 0, 0]],  # dd != hold + ud
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(InvalidTimings):
        validate_timings(bad)


def test_validate_zeroes_last_row():
    out = validate_timings([[0.1, 0.3, 0.2], [0.1, 9.0, 9.0]])
    assert out[-1, 1] == 0 and out[-1, 2] == 0


# ---- enrollment ---------------------------------------------------------


def test_enrollment_progress_then_complete(engine, typists):
    alice, _, rng = typists
    results = [engine.enroll("app", "alice", a) for a in attempts(alice, rng, 10)]
    assert [r.status for r in results[:-1]] == ["in_progress"] * 9
    assert [r.samples for r in results] == list(range(1, 11))
    assert results[-1].status == "complete"
    assert engine.status("app", "alice")["status"] == "active"


def test_enrollment_rejects_length_change(engine, typists):
    alice, _, rng = typists
    engine.enroll("app", "alice", attempts(alice, rng, 1)[0])
    other = extract_features(simulate_attempt(make_typist("short", rng), rng)).values
    r = engine.enroll("app", "alice", other)
    assert r.status == "rejected" and r.reason == "length_mismatch" and r.samples == 1


def test_enrollment_rejects_replay(engine, typists):
    alice, _, rng = typists
    a = attempts(alice, rng, 1)[0]
    assert engine.enroll("app", "alice", a).status == "in_progress"
    r = engine.enroll("app", "alice", a.copy())
    assert r.reason == "replay" and r.samples == 1


def test_enrollment_restarts_when_inconsistent(typists):
    alice, _, rng = typists
    eng = KeyAuthEngine(RandomEmbedder(), TemplateStore(), threshold=0.5)
    results = [eng.enroll("app", "alice", a) for a in attempts(alice, rng, 10)]
    assert results[-1].status == "rejected" and results[-1].reason == "inconsistent"
    assert eng.status("app", "alice") == {
        "user_id": "alice",
        "status": "enrolling",
        "enroll_samples": 0,
        "enroll_required": 10,
        "locked": False,
    }


def test_cannot_enroll_twice(engine, typists):
    alice, _, rng = typists
    assert enroll_fully(engine, "app", "alice", alice, rng).status == "complete"
    assert engine.enroll("app", "alice", attempts(alice, rng, 1)[0]).reason == "already_enrolled"


# ---- verification -------------------------------------------------------


def test_verify_unknown_user(engine, typists):
    alice, _, rng = typists
    r = engine.verify("app", "nobody", attempts(alice, rng, 1)[0])
    assert not r.accepted and r.reason == "not_enrolled"


def test_genuine_accepted_impostor_rejected(engine, typists):
    alice, mallory, rng = typists
    enroll_fully(engine, "app", "alice", alice, rng)
    genuine = [engine.verify("app", "alice", a) for a in attempts(alice, rng, 30)]
    impostor = [engine.verify("app", "alice", a) for a in attempts(mallory, rng, 30)]
    assert np.mean([r.accepted for r in genuine]) >= 0.7
    assert np.mean([not r.accepted for r in impostor]) >= 0.7
    assert all(r.score is not None for r in genuine + impostor)


def test_verify_rejects_replayed_login(engine, typists):
    alice, _, rng = typists
    enroll_fully(engine, "app", "alice", alice, rng)
    for a in attempts(alice, rng, 20):
        if engine.verify("app", "alice", a).accepted:
            again = engine.verify("app", "alice", a.copy())
            assert not again.accepted and again.reason == "replay"
            return
    pytest.fail("no genuine attempt was accepted")


def test_verify_rejects_enrollment_replay(engine, typists):
    alice, _, rng = typists
    enroll_attempts = attempts(alice, rng, 10)
    for a in enroll_attempts:
        engine.enroll("app", "alice", a)
    assert engine.verify("app", "alice", enroll_attempts[3]).reason == "replay"


def test_verify_length_mismatch(engine, typists):
    alice, _, rng = typists
    enroll_fully(engine, "app", "alice", alice, rng)
    r = engine.verify("app", "alice", [[0.1, 0.3, 0.2], [0.1, 0, 0]])
    assert not r.accepted and r.reason == "length_mismatch"


def test_lockout_and_recovery(threshold, clock, typists):
    alice, _, rng = typists
    eng = KeyAuthEngine(
        fake_embedder, TemplateStore(), threshold, EngineConfig(max_failures=3, lockout_s=60), clock=clock
    )
    enroll_fully(eng, "app", "alice", alice, rng)
    eng.store.set_threshold("app", "alice", 1e-6)  # nothing can match
    reasons = [eng.verify("app", "alice", a).reason for a in attempts(alice, rng, 4)]
    assert reasons == ["no_match", "no_match", "no_match", "locked"]
    assert eng.status("app", "alice")["locked"]

    clock.t += 61
    eng.store.set_threshold("app", "alice", None)  # back to the engine default
    results = [eng.verify("app", "alice", a) for a in attempts(alice, rng, 10)]
    assert any(r.accepted for r in results)
    assert not eng.status("app", "alice")["locked"]


def test_adaptation_moves_template(threshold, typists):
    alice, _, rng = typists
    eng = KeyAuthEngine(fake_embedder, TemplateStore(), threshold, EngineConfig(adapt_max_ratio=1.0, max_failures=1000))
    enroll_fully(eng, "app", "alice", alice, rng)
    before = eng.store.get_user("app", "alice").template
    assert any(eng.verify("app", "alice", a).accepted for a in attempts(alice, rng, 10))
    after = eng.store.get_user("app", "alice").template
    assert not np.allclose(before, after)
    assert np.linalg.norm(after) == pytest.approx(1.0, abs=1e-5)


def test_adaptation_can_be_disabled(threshold, typists):
    alice, _, rng = typists
    eng = KeyAuthEngine(fake_embedder, TemplateStore(), threshold, EngineConfig(adapt=False, max_failures=1000))
    enroll_fully(eng, "app", "alice", alice, rng)
    before = eng.store.get_user("app", "alice").template
    for a in attempts(alice, rng, 10):
        eng.verify("app", "alice", a)
    np.testing.assert_array_equal(before, eng.store.get_user("app", "alice").template)


def test_history_is_trimmed(threshold, typists):
    alice, _, rng = typists
    eng = KeyAuthEngine(fake_embedder, TemplateStore(), threshold, EngineConfig(history_size=3, max_failures=1000))
    enroll_fully(eng, "app", "alice", alice, rng)
    accepted = sum(eng.verify("app", "alice", a).accepted for a in attempts(alice, rng, 20))
    assert accepted > 3
    assert eng.store.count_samples("app", "alice", "verify") == 3


# ---- tenancy, deletion, persistence -------------------------------------


def test_apps_are_isolated(engine, typists):
    alice, mallory, rng = typists
    enroll_fully(engine, "app1", "alice", alice, rng)
    assert engine.status("app2", "alice") is None
    assert engine.verify("app2", "alice", attempts(alice, rng, 1)[0]).reason == "not_enrolled"
    enroll_fully(engine, "app2", "alice", mallory, rng)  # a different "alice"
    assert engine.status("app1", "alice")["status"] == "active"


def test_delete_user(engine, typists):
    alice, _, rng = typists
    enroll_fully(engine, "app", "alice", alice, rng)
    assert engine.delete_user("app", "alice")
    assert engine.status("app", "alice") is None
    assert engine.store.count_samples("app", "alice", "enroll") == 0
    assert not engine.delete_user("app", "alice")
    assert enroll_fully(engine, "app", "alice", alice, rng).status == "complete"


def test_store_persists_to_disk(tmp_path, threshold, typists):
    alice, _, rng = typists
    path = tmp_path / "keyauth.db"
    store = TemplateStore(path)
    eng = KeyAuthEngine(fake_embedder, store, threshold)
    enroll_fully(eng, "app", "alice", alice, rng)
    template = store.get_user("app", "alice").template
    store.close()

    reopened = TemplateStore(path)
    np.testing.assert_array_equal(reopened.get_user("app", "alice").template, template)
    assert reopened.count_samples("app", "alice", "enroll") == 10

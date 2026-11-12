"""C6b engine wiring: per-user thresholds and the classic scorers (scaled Manhattan / Mahalanobis).

Uses the synthetic typist generators; every number that decides accept/reject is calibrated
on separate calibration users inside the fixtures, not hard-coded.
"""

import sqlite3

import numpy as np
import pytest

from tempo.auth import EngineConfig, TempoEngine, TemplateStore
from tempo.auth.scorers import MahalanobisScorer, ScaledManhattanScorer, ScorerState, enrollment_loo_scores, timing_vector
from tempo.evaluate import compute_eer, verification_scores
from tempo.features import extract_features
from tempo.synthetic import make_typist, simulate_attempt
from tempo.thresholds import PerUserThresholdParams, embedding_threshold, enrollment_spread, embedding_loo_scores

PHRASE = "hunter2!"
N_ENROLL = 10


def fake_embedder(timings: np.ndarray) -> np.ndarray:
    t = np.asarray(timings, dtype=np.float64)
    v = np.concatenate([np.log(np.maximum(t[:, 0], 1e-3)), np.log(np.maximum(t[:-1, 1], 1e-3))])
    v = v - v.mean()
    return v / np.linalg.norm(v)


class RandomEmbedder:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, timings):
        v = self.rng.normal(size=32)
        return v / np.linalg.norm(v)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def attempts(typist, rng, n):
    return [extract_features(simulate_attempt(typist, rng)).values for _ in range(n)]


def enroll_fully(engine, typist, rng, user="alice"):
    result = None
    for a in attempts(typist, rng, engine.config.enroll_samples):
        result = engine.enroll("app", user, a)
    return result


@pytest.fixture
def typists():
    rng = np.random.default_rng(7)
    return make_typist(PHRASE, rng), make_typist(PHRASE, rng), rng


@pytest.fixture
def clock():
    return Clock()


# ---- calibration (separate users) ---------------------------------------


@pytest.fixture(scope="module")
def embedder_calibration():
    """(global threshold at the EER, per-user params whose global_threshold equals it)."""
    rng = np.random.default_rng(100)
    emb, users, spreads = [], [], []
    for u in range(10):
        typist = make_typist(PHRASE, rng)
        z = np.stack([fake_embedder(a) for a in attempts(typist, rng, 30)])
        emb.extend(z)
        users.extend([f"cal{u}"] * len(z))
        spreads.append(enrollment_spread(embedding_loo_scores(z[:N_ENROLL])))
    scores = verification_scores(np.stack(emb), np.array(users), n_enroll=N_ENROLL)
    threshold = compute_eer(scores.genuine, scores.impostor).threshold
    reference = float(np.median(spreads))
    return threshold, PerUserThresholdParams(scale=threshold / reference, weight=1.0, reference_spread=reference)


@pytest.fixture(scope="module", params=[ScaledManhattanScorer(), MahalanobisScorer()], ids=lambda s: s.kind)
def scorer_calibration(request):
    """(scorer, global threshold at the EER of its own scores, per-user params consistent with it)."""
    scorer = request.param
    rng = np.random.default_rng(200)
    typists = [make_typist(PHRASE, rng) for _ in range(8)]
    vectors = [np.stack([timing_vector(a) for a in attempts(t, rng, 40)]) for t in typists]
    genuine, impostor, spreads = [], [], []
    for i, v in enumerate(vectors):
        state = scorer.fit(v[:N_ENROLL])
        spreads.append(float(np.median(enrollment_loo_scores(scorer, v[:N_ENROLL]))))
        genuine.extend(scorer.score(state, x) for x in v[N_ENROLL:])
        for j, other in enumerate(vectors):
            if j != i:
                impostor.extend(scorer.score(state, x) for x in other[N_ENROLL : N_ENROLL + 5])
    threshold = compute_eer(np.array(genuine), np.array(impostor)).threshold
    reference = float(np.median(spreads))
    return scorer, threshold, PerUserThresholdParams(scale=threshold / reference, weight=1.0, reference_spread=reference)


# ---- per-user thresholds: embedder mode ---------------------------------


def make_embedder_engine(calibration, clock, per_user=True, **config):
    threshold, params = calibration
    if not per_user:
        params = None
    config.setdefault("max_failures", 1000)
    config.setdefault("enroll_samples", N_ENROLL)
    return TempoEngine(
        fake_embedder,
        TemplateStore(),
        threshold,
        EngineConfig(**config),
        clock=clock,
        per_user=params,
    )


def test_per_user_threshold_is_stored_at_enrollment_and_used_at_verify(embedder_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_embedder_engine(embedder_calibration, clock)
    assert enroll_fully(engine, alice, rng).status == "complete"
    stored = engine.store.get_user("app", "alice").threshold
    expected = embedding_threshold(engine.store.get_samples("app", "alice", "enroll"), embedder_calibration[1])
    assert stored == pytest.approx(expected, rel=1e-6)
    assert stored != pytest.approx(embedder_calibration[0])
    r = engine.verify("app", "alice", attempts(alice, rng, 1)[0])
    assert r.threshold == pytest.approx(expected, rel=1e-6)


def test_without_params_the_global_threshold_is_used(embedder_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_embedder_engine(embedder_calibration, clock, per_user=False)
    assert enroll_fully(engine, alice, rng).status == "complete"
    assert engine.store.get_user("app", "alice").threshold is None
    assert engine.verify("app", "alice", attempts(alice, rng, 1)[0]).threshold == embedder_calibration[0]


def test_genuine_accepted_and_impostor_rejected_with_per_user_threshold(embedder_calibration, clock, typists):
    alice, mallory, rng = typists
    engine = make_embedder_engine(embedder_calibration, clock)
    enroll_fully(engine, alice, rng)
    genuine = [engine.verify("app", "alice", a) for a in attempts(alice, rng, 30)]
    impostor = [engine.verify("app", "alice", a) for a in attempts(mallory, rng, 30)]
    assert np.mean([r.accepted for r in genuine]) >= 0.7
    assert np.mean([not r.accepted for r in impostor]) >= 0.7


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_or_non_positive_user_threshold_falls_back_to_global(
    embedder_calibration, clock, typists, monkeypatch, bad
):
    alice, _, rng = typists
    monkeypatch.setattr("tempo.auth.engine.embedding_threshold", lambda samples, params: bad)
    engine = make_embedder_engine(embedder_calibration, clock)
    assert enroll_fully(engine, alice, rng).status == "complete"
    assert engine.store.get_user("app", "alice").threshold is None
    assert engine.verify("app", "alice", attempts(alice, rng, 1)[0]).threshold == embedder_calibration[0]


def test_failing_computation_falls_back_to_global_without_crashing(embedder_calibration, clock, typists, monkeypatch):
    alice, _, rng = typists

    def boom(samples, params):
        raise ValueError("cannot compute")

    monkeypatch.setattr("tempo.auth.engine.embedding_threshold", boom)
    engine = make_embedder_engine(embedder_calibration, clock)
    assert enroll_fully(engine, alice, rng).status == "complete"
    assert engine.store.get_user("app", "alice").threshold is None


def test_too_few_enrollment_attempts_fall_back_to_global(embedder_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_embedder_engine(embedder_calibration, clock, enroll_samples=2)
    assert enroll_fully(engine, alice, rng).status == "complete"  # leave-one-out needs 3, enrollment must not crash
    assert engine.store.get_user("app", "alice").threshold is None


def test_inconsistent_enrollment_check_stays_on_the_global_threshold(typists):
    alice, _, rng = typists
    loose = PerUserThresholdParams(scale=100.0, weight=0.0, reference_spread=1.0)  # per-user threshold would be huge
    engine = TempoEngine(RandomEmbedder(), TemplateStore(), threshold=0.5, per_user=loose)
    results = [engine.enroll("app", "alice", a) for a in attempts(alice, rng, 10)]
    assert results[-1].status == "rejected" and results[-1].reason == "inconsistent"
    assert engine.store.get_user("app", "alice").threshold is None


def test_adaptation_uses_the_per_user_threshold(embedder_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_embedder_engine(embedder_calibration, clock, adapt_max_ratio=0.5)
    engine.threshold = 2.0  # the global threshold would adapt on any plausible genuine attempt
    enroll_fully(engine, alice, rng)
    template = engine.store.get_user("app", "alice").template
    probe, nxt = attempts(alice, rng, 2)
    score = float(np.linalg.norm(fake_embedder(probe) - template))
    assert 0.0 < score < 1.0

    engine.store.set_threshold("app", "alice", 1.2 * score)  # accepted, but score > 0.5 * threshold
    assert engine.verify("app", "alice", probe).accepted
    np.testing.assert_array_equal(engine.store.get_user("app", "alice").template, template)

    # a looser personal threshold makes the same kind of attempt confident enough to adapt
    engine.store.set_threshold("app", "alice", 1e6)
    engine.store.set_lockout("app", "alice", 0, 0.0)
    assert engine.verify("app", "alice", nxt).accepted
    assert not np.allclose(engine.store.get_user("app", "alice").template, template)


def test_constructor_needs_exactly_one_of_embedder_and_scorer():
    with pytest.raises(ValueError):
        TempoEngine(None, TemplateStore(), 1.0)
    with pytest.raises(ValueError):
        TempoEngine(fake_embedder, TemplateStore(), 1.0, scorer=ScaledManhattanScorer())
    with pytest.raises(ValueError):
        TempoEngine(None, TemplateStore(), 1.0, EngineConfig(enroll_samples=2), scorer=ScaledManhattanScorer())


# ---- scorer mode ----------------------------------------------------------


def make_scorer_engine(calibration, clock, per_user=False, **config):
    scorer, threshold, params = calibration
    if not per_user:
        params = None
    config.setdefault("max_failures", 1000)
    return TempoEngine(
        None,
        TemplateStore(),
        threshold,
        EngineConfig(enroll_samples=N_ENROLL, **config),
        clock=clock,
        per_user=params,
        scorer=scorer,
    )


def test_scorer_enrollment_stores_state_and_timing_vectors(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    enrolled = attempts(alice, rng, N_ENROLL)
    results = [engine.enroll("app", "alice", a) for a in enrolled]
    assert [r.status for r in results] == ["in_progress"] * (N_ENROLL - 1) + ["complete"]
    user = engine.store.get_user("app", "alice")
    assert user.status == "active" and user.template is None and user.threshold is None
    state = ScorerState.from_json(user.scorer_state)
    assert state.kind == scorer_calibration[0].kind
    samples = engine.store.get_samples("app", "alice", "enroll")
    assert samples.shape == (N_ENROLL, 3 * len(enrolled[0]) - 2)
    np.testing.assert_allclose(samples[0], timing_vector(enrolled[0]), atol=1e-7)
    assert state.to_json() == user.scorer_state  # lossless JSON round trip


def test_scorer_state_survives_reopening_the_database(scorer_calibration, clock, typists, tmp_path):
    alice, _, rng = typists
    scorer, threshold, _ = scorer_calibration
    path = tmp_path / "tempo.db"
    engine = TempoEngine(None, TemplateStore(path), threshold, EngineConfig(enroll_samples=N_ENROLL), clock=clock, scorer=scorer)
    enroll_fully(engine, alice, rng)
    saved = engine.store.get_user("app", "alice").scorer_state
    engine.store.close()
    reopened = TempoEngine(None, TemplateStore(path), threshold, EngineConfig(enroll_samples=N_ENROLL), clock=clock, scorer=scorer)
    assert reopened.store.get_user("app", "alice").scorer_state == saved
    assert reopened.verify("app", "alice", attempts(alice, rng, 1)[0]).reason in ("match", "no_match")


def test_scorer_genuine_accepted_impostor_rejected(scorer_calibration, clock, typists):
    alice, mallory, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    enroll_fully(engine, alice, rng)
    genuine = [engine.verify("app", "alice", a) for a in attempts(alice, rng, 30)]
    impostor = [engine.verify("app", "alice", a) for a in attempts(mallory, rng, 30)]
    assert np.mean([r.accepted for r in genuine]) >= 0.7
    assert np.mean([not r.accepted for r in impostor]) >= 0.7
    assert all(r.score is not None for r in genuine + impostor)


def test_scorer_per_user_threshold_is_stored_and_used(scorer_calibration, clock, typists):
    alice, mallory, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock, per_user=True)
    enroll_fully(engine, alice, rng)
    stored = engine.store.get_user("app", "alice").threshold
    assert stored is not None and np.isfinite(stored) and stored > 0
    assert stored != pytest.approx(scorer_calibration[1])
    assert engine.verify("app", "alice", attempts(alice, rng, 1)[0]).threshold == pytest.approx(stored)
    impostor = [engine.verify("app", "alice", a) for a in attempts(mallory, rng, 30)]
    assert np.mean([not r.accepted for r in impostor]) >= 0.7


def test_scorer_per_user_fallback_when_threshold_is_not_finite(scorer_calibration, clock, typists, monkeypatch):
    alice, _, rng = typists
    monkeypatch.setattr("tempo.auth.engine.user_threshold", lambda loo, params: float("nan"))
    engine = make_scorer_engine(scorer_calibration, clock, per_user=True)
    assert enroll_fully(engine, alice, rng).status == "complete"
    assert engine.store.get_user("app", "alice").threshold is None
    assert engine.verify("app", "alice", attempts(alice, rng, 1)[0]).threshold == scorer_calibration[1]


def test_scorer_inconsistent_enrollment_restarts(scorer_calibration, clock, typists):
    alice, _, rng = typists
    scorer, _, params = scorer_calibration
    engine = TempoEngine(
        None, TemplateStore(), 1e-9, EngineConfig(enroll_samples=N_ENROLL), per_user=params, scorer=scorer
    )
    result = enroll_fully(engine, alice, rng)
    assert result.status == "rejected" and result.reason == "inconsistent"
    assert engine.status("app", "alice")["enroll_samples"] == 0
    assert engine.store.get_user("app", "alice").scorer_state is None


def test_scorer_not_enrolled(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    assert engine.verify("app", "nobody", attempts(alice, rng, 1)[0]).reason == "not_enrolled"
    engine.enroll("app", "alice", attempts(alice, rng, 1)[0])  # still enrolling
    assert engine.verify("app", "alice", attempts(alice, rng, 1)[0]).reason == "not_enrolled"


def test_scorer_length_mismatch(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    enroll_fully(engine, alice, rng)
    r = engine.verify("app", "alice", [[0.1, 0.3, 0.2], [0.1, 0, 0]])
    assert not r.accepted and r.reason == "length_mismatch"


def test_scorer_replay_of_identical_timings_is_rejected(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    enrolled = attempts(alice, rng, N_ENROLL)
    for a in enrolled:
        engine.enroll("app", "alice", a)
    assert engine.verify("app", "alice", enrolled[3].copy()).reason == "replay"  # replayed enrollment attempt
    for a in attempts(alice, rng, 20):
        if engine.verify("app", "alice", a).accepted:
            again = engine.verify("app", "alice", a.copy())
            assert not again.accepted and again.reason == "replay"
            return
    pytest.fail("no genuine attempt was accepted")


def test_scorer_replay_with_sub_epsilon_jitter_is_rejected_but_real_variation_is_not(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    enrolled = attempts(alice, rng, N_ENROLL)
    for a in enrolled:
        engine.enroll("app", "alice", a)
    jittered = enrolled[0].astype(np.float64).copy()
    jittered[0, [0, 1]] += 2e-5  # 0.02 ms on one hold time (dd = hold + ud stays valid)
    assert engine.verify("app", "alice", jittered).reason == "replay"
    shifted = enrolled[0].astype(np.float64).copy()
    shifted[0, [0, 1]] += 5e-3  # 5 ms: a different attempt
    assert engine.verify("app", "alice", shifted).reason != "replay"


def test_scorer_lockout_after_max_failures_and_recovery(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock, max_failures=3, lockout_s=60)
    enroll_fully(engine, alice, rng)
    engine.store.set_threshold("app", "alice", 1e-12)  # nothing can match
    reasons = [engine.verify("app", "alice", a).reason for a in attempts(alice, rng, 4)]
    assert reasons == ["no_match", "no_match", "no_match", "locked"]
    assert engine.status("app", "alice")["locked"]
    clock.t += 61
    engine.store.set_threshold("app", "alice", None)
    assert any(engine.verify("app", "alice", a).accepted for a in attempts(alice, rng, 10))
    assert not engine.status("app", "alice")["locked"]


def test_scorer_mode_does_not_adapt_but_keeps_login_history(scorer_calibration, clock, typists):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock, history_size=3, adapt_max_ratio=1.0)
    enroll_fully(engine, alice, rng)
    before = engine.store.get_user("app", "alice").scorer_state
    accepted = sum(engine.verify("app", "alice", a).accepted for a in attempts(alice, rng, 20))
    assert accepted > 3
    assert engine.store.get_user("app", "alice").scorer_state == before
    assert engine.store.count_samples("app", "alice", "verify") == 3


@pytest.mark.parametrize(
    "corrupt",
    [
        None,
        "not json",
        "{}",
        '{"kind": "mahalanobis", "arrays": {"mean": [1.0], "precision": [[1.0]]}}',  # wrong dimension
        '{"kind": "scaled_manhattan", "arrays": {"mean": [1.0], "scale": [NaN]}}',
    ],
)
def test_scorer_corrupt_state_fails_closed_with_a_reason(scorer_calibration, clock, typists, corrupt):
    alice, _, rng = typists
    engine = make_scorer_engine(scorer_calibration, clock)
    enroll_fully(engine, alice, rng)
    engine.store._conn.execute("UPDATE users SET scorer_state = ?", (corrupt,))
    r = engine.verify("app", "alice", attempts(alice, rng, 1)[0])
    assert not r.accepted and r.reason == "scorer_error" and r.score is None
    assert engine.store.get_user("app", "alice").failures == 0  # a server fault is not the user's failure


def test_scorer_state_of_another_kind_is_rejected(clock, typists):
    alice, _, rng = typists
    manhattan, mahalanobis = ScaledManhattanScorer(), MahalanobisScorer()
    engine = TempoEngine(None, TemplateStore(), 100.0, EngineConfig(enroll_samples=N_ENROLL), clock=clock, scorer=manhattan)
    enroll_fully(engine, alice, rng)
    vectors = engine.store.get_samples("app", "alice", "enroll")
    engine.store.set_scorer_state("app", "alice", mahalanobis.fit(vectors).to_json(), clock())
    r = engine.verify("app", "alice", attempts(alice, rng, 1)[0])
    assert not r.accepted and r.reason == "scorer_error"


# ---- store ----------------------------------------------------------------


def test_store_migrates_a_database_created_before_scorer_state(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (
            app_id TEXT NOT NULL, user_id TEXT NOT NULL, n_keys INTEGER NOT NULL, status TEXT NOT NULL,
            template BLOB, threshold REAL, failures INTEGER NOT NULL DEFAULT 0,
            locked_until REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (app_id, user_id)
        );
        INSERT INTO users (app_id, user_id, n_keys, status, threshold, created_at, updated_at)
        VALUES ('app', 'old', 5, 'enrolling', 0.25, 1.0, 1.0);
        """
    )
    conn.commit()
    conn.close()
    store = TemplateStore(path)
    user = store.get_user("app", "old")
    assert user.threshold == 0.25 and user.scorer_state is None
    store.set_scorer_state("app", "old", '{"x": 1}', 2.0)
    assert store.get_user("app", "old").scorer_state == '{"x": 1}'
    assert store.get_user("app", "old").status == "active"
    store.close()
    TemplateStore(path).close()  # reopening a migrated database is a no-op

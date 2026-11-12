import logging
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tempo.api import Settings, create_app  # noqa: E402
from tempo.api.ratelimit import SlidingWindowLimiter  # noqa: E402
from tempo.api.settings import parse_rate_limit  # noqa: E402
from tempo.auth import EngineConfig, TempoEngine, TemplateStore  # noqa: E402
from tempo.config import MAX_KEYS  # noqa: E402
from tempo.evaluate import compute_eer, verification_scores  # noqa: E402
from tempo.features import extract_features  # noqa: E402
from tempo.synthetic import make_typist, simulate_attempt, to_raw_events  # noqa: E402

PHRASE = "hunter2!"
REQUIRED = 10


def fake_embedder(timings: np.ndarray) -> np.ndarray:
    t = np.asarray(timings, dtype=np.float64)
    v = np.concatenate([np.log(np.maximum(t[:, 0], 1e-3)), np.log(np.maximum(t[:-1, 1], 1e-3))])
    v = v - v.mean()
    return v / np.linalg.norm(v)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def timings_of(typist, rng) -> list:
    return extract_features(simulate_attempt(typist, rng)).values.tolist()


def events_of(typist, rng) -> list:
    raw = to_raw_events(simulate_attempt(typist, rng))
    return [{"key": e.key, "type": e.type, "t": e.t, "code": e.code} for e in raw]


@pytest.fixture(scope="module")
def threshold():
    rng = np.random.default_rng(100)
    emb, users = [], []
    for u in range(10):
        typist = make_typist(PHRASE, rng)
        for _ in range(30):
            emb.append(fake_embedder(extract_features(simulate_attempt(typist, rng)).values))
            users.append(f"cal{u}")
    scores = verification_scores(np.stack(emb), np.array(users), n_enroll=10)
    return compute_eer(scores.genuine, scores.impostor).threshold


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def typists():
    rng = np.random.default_rng(7)
    return make_typist(PHRASE, rng), make_typist(PHRASE, rng), rng


def build(threshold, clock, max_failures=1000, **settings):
    config = EngineConfig(enroll_samples=REQUIRED, max_failures=max_failures, lockout_s=300.0)
    engine = TempoEngine(fake_embedder, TemplateStore(), threshold, config, clock=clock)
    values = {"rate_limit": 10000}
    values.update(settings)
    app = create_app(engine, Settings(**values))
    app.state.limiter.clock = clock
    return TestClient(app), engine


@pytest.fixture
def client(threshold, clock):
    return build(threshold, clock)[0]


def enroll_all(client, user, typist, rng):
    r = None
    for _ in range(REQUIRED):
        r = client.post("/enroll", json={"user_id": user, "timings": timings_of(typist, rng)})
        assert r.status_code == 200, r.text
    return r


def assert_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert isinstance(body["error"]["message"], str)


# ---- health / availability ------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_model_unavailable(tmp_path):
    settings = Settings(checkpoint=str(tmp_path / "missing.pt"), db_path=str(tmp_path / "k.db"))
    c = TestClient(create_app(settings=settings))
    r = c.get("/health")
    assert r.json() == {"status": "ok", "model_loaded": False}
    assert "missing" not in r.text
    body = {"user_id": "alice", "timings": [[0.1, 0.2, 0.1], [0.1, 0, 0]]}
    assert_error(c.post("/enroll", json=body), 503, "model_unavailable")
    assert_error(c.post("/verify", json=body), 503, "model_unavailable")
    assert_error(c.get("/status", params={"user_id": "alice"}), 503, "model_unavailable")


def test_real_checkpoint_loads(tmp_path):
    path = Path(__file__).resolve().parents[1] / "checkpoints" / "model.pt"
    if not path.exists():
        pytest.skip("no trained checkpoint")
    c = TestClient(create_app(settings=Settings(checkpoint=str(path), db_path=str(tmp_path / "k.db"))))
    assert c.get("/health").json()["model_loaded"] is True


def test_unknown_route_uses_error_shape(client):
    assert_error(client.get("/nope"), 404, "not_found")


# ---- enroll / verify ------------------------------------------------------


def test_enroll_progress_to_complete(client, typists):
    alice, _, rng = typists
    for i in range(1, REQUIRED):
        r = client.post("/enroll", json={"user_id": "alice", "timings": timings_of(alice, rng)})
        assert r.status_code == 200
        assert r.json() == {"status": "in_progress", "samples": i, "required": REQUIRED}
    r = client.post("/enroll", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert r.json() == {"status": "complete", "samples": REQUIRED, "required": REQUIRED}
    s = client.get("/status", params={"user_id": "alice"})
    assert s.status_code == 200 and s.json()["status"] == "active" and s.json()["locked"] is False


def test_enroll_after_complete_is_conflict(client, typists):
    alice, _, rng = typists
    enroll_all(client, "alice", alice, rng)
    r = client.post("/enroll", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert_error(r, 409, "already_enrolled")


def test_enroll_replay_and_length_mismatch(client, typists):
    alice, _, rng = typists
    t = timings_of(alice, rng)
    assert client.post("/enroll", json={"user_id": "alice", "timings": t}).status_code == 200
    assert_error(client.post("/enroll", json={"user_id": "alice", "timings": t}), 409, "replay")
    short = timings_of(make_typist("short", rng), rng)
    assert_error(client.post("/enroll", json={"user_id": "alice", "timings": short}), 422, "length_mismatch")


def test_verify_accept_and_reject(client, typists):
    alice, bob, rng = typists
    enroll_all(client, "alice", alice, rng)
    r = client.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert r.status_code == 200 and r.json() == {"accepted": True, "reason": "match"}
    r = client.post("/verify", json={"user_id": "alice", "timings": timings_of(bob, rng)})
    assert r.status_code == 200 and r.json() == {"accepted": False, "reason": "no_match"}


def test_verify_not_enrolled(client, typists):
    alice, _, rng = typists
    r = client.post("/verify", json={"user_id": "ghost", "timings": timings_of(alice, rng)})
    assert_error(r, 404, "not_enrolled")
    client.post("/enroll", json={"user_id": "half", "timings": timings_of(alice, rng)})
    r = client.post("/verify", json={"user_id": "half", "timings": timings_of(alice, rng)})
    assert_error(r, 404, "not_enrolled")


def test_verify_replay(client, typists):
    alice, _, rng = typists
    enroll_all(client, "alice", alice, rng)
    t = timings_of(alice, rng)
    assert client.post("/verify", json={"user_id": "alice", "timings": t}).json()["accepted"] is True
    assert_error(client.post("/verify", json={"user_id": "alice", "timings": t}), 409, "replay")


def test_verify_length_mismatch(client, typists):
    alice, _, rng = typists
    enroll_all(client, "alice", alice, rng)
    short = timings_of(make_typist("short", rng), rng)
    assert_error(client.post("/verify", json={"user_id": "alice", "timings": short}), 422, "length_mismatch")


def test_lockout_with_retry_after(threshold, clock, typists):
    alice, bob, rng = typists
    client, _ = build(threshold, clock, max_failures=3)
    enroll_all(client, "alice", alice, rng)
    last = None
    for _ in range(3):
        last = client.post("/verify", json={"user_id": "alice", "timings": timings_of(bob, rng)})
        assert last.status_code == 200 and last.json()["accepted"] is False
    assert last.json()["retry_after_s"] == 300
    clock.t += 100
    r = client.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert_error(r, 423, "locked")
    assert r.headers["Retry-After"] == "200"
    assert client.get("/status", params={"user_id": "alice"}).json()["locked"] is True
    clock.t += 201
    r = client.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert r.status_code == 200 and r.json()["accepted"] is True


def test_scorer_error_fails_closed(clock, typists):
    from tempo.auth.scorers import ScaledManhattanScorer

    alice, _, rng = typists
    config = EngineConfig(enroll_samples=REQUIRED)
    engine = TempoEngine(None, TemplateStore(), 100.0, config, clock=clock, scorer=ScaledManhattanScorer())
    client = TestClient(create_app(engine, Settings(rate_limit=10000)))
    enroll_all(client, "alice", alice, rng)
    engine.store._conn.execute("UPDATE users SET scorer_state = ?", ("not json",))
    r = client.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert_error(r, 503, "scorer_error")
    assert engine.store.get_user("default", "alice").failures == 0


# ---- status / delete ------------------------------------------------------


def test_status_404_then_200(client, typists):
    alice, _, rng = typists
    assert_error(client.get("/status", params={"user_id": "alice"}), 404, "not_enrolled")
    client.post("/enroll", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    r = client.get("/status", params={"user_id": "alice"})
    assert r.status_code == 200
    assert r.json() == {"user_id": "alice", "status": "enrolling", "enroll_samples": 1, "enroll_required": REQUIRED, "locked": False}


ADMIN = "s3cret-token-0123456789"
GOOD = {"Authorization": f"Bearer {ADMIN}"}


def test_delete_then_not_enrolled(threshold, clock, typists):
    alice, _, rng = typists
    client, _ = build(threshold, clock, admin_token=ADMIN)
    enroll_all(client, "alice", alice, rng)
    assert client.delete("/user", params={"user_id": "alice"}, headers=GOOD).json() == {"deleted": True}
    assert_error(client.get("/status", params={"user_id": "alice"}), 404, "not_enrolled")
    r = client.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert_error(r, 404, "not_enrolled")
    assert_error(client.delete("/user", params={"user_id": "alice"}, headers=GOOD), 404, "not_enrolled")


@pytest.mark.parametrize("token", [None, ""])
@pytest.mark.parametrize("header", [None, "Bearer ", "Bearer", "Bearer None", "Bearer anything", "Basic abc"])
def test_delete_refused_when_no_admin_token_configured(threshold, clock, typists, token, header):
    alice, _, rng = typists
    client, _ = build(threshold, clock, admin_token=token)
    enroll_all(client, "alice", alice, rng)
    headers = {}
    if header is not None:
        headers["Authorization"] = header
    r = client.delete("/user", params={"user_id": "alice"}, headers=headers)
    assert_error(r, 401, "unauthorized")
    assert "TEMPO_ADMIN_TOKEN" in r.json()["error"]["message"]
    assert client.get("/status", params={"user_id": "alice"}).status_code == 200


@pytest.mark.parametrize("header", [None, "Bearer nope", "Bearer ", f"Basic {ADMIN}", ADMIN, f"bearer {ADMIN}"])
def test_delete_with_token_configured_rejects_bad_credentials(threshold, clock, typists, header):
    alice, _, rng = typists
    client, _ = build(threshold, clock, admin_token=ADMIN)
    enroll_all(client, "alice", alice, rng)
    headers = {}
    if header is not None:
        headers["Authorization"] = header
    assert_error(client.delete("/user", params={"user_id": "alice"}, headers=headers), 401, "unauthorized")
    assert client.get("/status", params={"user_id": "alice"}).status_code == 200


def test_short_admin_token_warns_but_works(threshold, clock, typists, caplog):
    alice, _, rng = typists
    with caplog.at_level(logging.WARNING, logger="tempo.api"):
        client, _ = build(threshold, clock, admin_token="short")
    assert "TEMPO_ADMIN_TOKEN" in caplog.text
    assert "short" not in caplog.text.replace("shorter", "")
    enroll_all(client, "alice", alice, rng)
    assert client.delete("/user", params={"user_id": "alice"}, headers={"Authorization": "Bearer short"}).status_code == 200


# ---- per-user thresholds setting -------------------------------------------


def test_per_user_is_on_by_default_and_env_can_disable():
    assert Settings().per_user is True
    assert Settings.from_env({}).per_user is True
    assert Settings.from_env({"TEMPO_PER_USER": "1"}).per_user is True
    for off in ["0", "false", "no", "off", ""]:
        assert Settings.from_env({"TEMPO_PER_USER": off}).per_user is False


def test_build_engine_falls_back_to_global_threshold_without_params(tmp_path, monkeypatch, caplog):
    pytest.importorskip("torch")
    from tempo.api.app import build_engine
    from tempo.auth.embedder import TorchEmbedder

    monkeypatch.setattr(TorchEmbedder, "from_checkpoint", staticmethod(lambda path: (fake_embedder, 0.5)))
    monkeypatch.setattr(TorchEmbedder, "per_user_params", staticmethod(lambda path: None))
    settings = Settings(checkpoint=str(tmp_path / "old.pt"), db_path=str(tmp_path / "k.db"))
    with caplog.at_level(logging.WARNING, logger="tempo.api"):
        engine = build_engine(settings)
    assert engine.per_user is None
    assert engine.threshold == 0.5
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "old.pt" not in warnings[0].getMessage()


# ---- validation -----------------------------------------------------------


def test_invalid_timings_shape(client):
    bad = {"user_id": "alice", "timings": [[0.1, 0.9, 0.1], [0.1, 0, 0]]}  # dd != hold + ud
    assert_error(client.post("/verify", json=bad), 422, "invalid_timings")
    bad = {"user_id": "alice", "timings": [[100.0, 200.0, 100.0], [0.1, 0, 0]]}  # ms, not s
    assert_error(client.post("/enroll", json=bad), 422, "invalid_timings")
    bad = {"user_id": "alice", "timings": [[0.1, 0.2]] * 3}
    assert_error(client.post("/enroll", json=bad), 422, "invalid_timings")


@pytest.mark.parametrize("which", ["both", "neither"])
def test_exactly_one_of_events_or_timings(client, typists, which):
    alice, _, rng = typists
    body = {"user_id": "alice"}
    if which == "both":
        body["timings"] = timings_of(alice, rng)
        body["events"] = events_of(alice, rng)
    r = client.post("/enroll", json=body)
    assert_error(r, 422, "validation_error")
    assert "exactly one" in r.text


def test_validation_error_does_not_echo_input(client):
    r = client.post("/enroll", json={"user_id": "alice", "timings": [[0.123456, "x", 0.1]]})
    assert_error(r, 422, "validation_error")
    assert "0.123456" not in r.text


def test_events_must_pair(client):
    events = [{"key": "a", "type": "down", "t": 10.0}, {"key": "b", "type": "down", "t": 50.0}]
    assert_error(client.post("/enroll", json={"user_id": "alice", "events": events}), 422, "invalid_timings")
    bad_type = [{"key": "a", "type": "press", "t": 10.0}]
    assert_error(client.post("/enroll", json={"user_id": "alice", "events": bad_type}), 422, "validation_error")


def test_oversize_payloads(client):
    ev = {"key": "a", "type": "down", "t": 1.0}
    r = client.post("/enroll", json={"user_id": "alice", "events": [ev] * (4 * MAX_KEYS + 1)})
    assert_error(r, 422, "validation_error")
    r = client.post("/enroll", json={"user_id": "alice", "timings": [[0.1, 0.2, 0.1]] * (MAX_KEYS + 1)})
    assert_error(r, 422, "validation_error")
    huge = '{"user_id": "alice", "timings": [' + "[0.1,0.2,0.1]," * 20000 + "[0.1,0,0]]}"
    r = client.post("/enroll", content=huge, headers={"Content-Type": "application/json"})
    assert_error(r, 413, "payload_too_large")


@pytest.mark.parametrize("user_id", ["", "a" * 65, "bad user", "x;DROP", "../etc", "é", 5, None])
def test_bad_user_id(client, user_id):
    body = {"user_id": user_id, "timings": [[0.1, 0.2, 0.1], [0.1, 0, 0]]}
    assert_error(client.post("/verify", json=body), 422, "validation_error")
    if isinstance(user_id, str):
        assert_error(client.get("/status", params={"user_id": user_id}), 422, "validation_error")
        assert_error(client.delete("/user", params={"user_id": user_id}), 422, "validation_error")


def test_missing_user_id_query(client):
    assert_error(client.get("/status"), 422, "validation_error")


# ---- raw events -----------------------------------------------------------


def test_raw_events_end_to_end(client, typists):
    alice, bob, rng = typists
    for i in range(REQUIRED):
        r = client.post("/enroll", json={"user_id": "alice", "events": events_of(alice, rng)})
        assert r.status_code == 200, r.text
    assert r.json()["status"] == "complete"
    r = client.post("/verify", json={"user_id": "alice", "events": events_of(alice, rng)})
    assert r.json() == {"accepted": True, "reason": "match"}
    r = client.post("/verify", json={"user_id": "alice", "events": events_of(bob, rng)})
    assert r.json()["accepted"] is False


# ---- rate limiting --------------------------------------------------------


def test_rate_limit_per_user_and_recovery(threshold, clock, typists):
    alice, _, rng = typists
    client, _ = build(threshold, clock, rate_limit=3, rate_window_s=60.0)
    body = {"user_id": "alice", "timings": timings_of(alice, rng)}
    for _ in range(3):
        assert client.post("/verify", json=body).status_code == 404
    r = client.post("/verify", json=body)
    assert_error(r, 429, "rate_limited")
    assert r.headers["Retry-After"] == "60"
    clock.t += 30
    assert client.post("/enroll", json=body).headers["Retry-After"] == "30"
    clock.t += 31
    assert client.post("/verify", json=body).status_code == 404


def test_rate_limit_is_per_user_not_global(threshold, clock, typists):
    alice, _, rng = typists
    # one client IP in tests, so the per-IP bucket is shared: limit applies to the IP here
    client, _ = build(threshold, clock, rate_limit=2)
    t = timings_of(alice, rng)
    assert client.post("/verify", json={"user_id": "u1", "timings": t}).status_code == 404
    assert client.post("/verify", json={"user_id": "u2", "timings": t}).status_code == 404
    assert_error(client.post("/verify", json={"user_id": "u3", "timings": t}), 429, "rate_limited")
    assert client.get("/health").status_code == 200  # other endpoints are not limited


def test_limiter_unit():
    now = [0.0]
    lim = SlidingWindowLimiter(2, 10.0, clock=lambda: now[0])
    assert lim.check(["a"]) is None
    assert lim.check(["a", "b"]) is None
    assert lim.check(["a"]) == 10
    assert lim.check(["b"]) is None  # refused request above was not recorded for b
    now[0] = 10.0
    assert lim.check(["a"]) is None


def test_parse_rate_limit():
    assert parse_rate_limit("30/60s") == (30, 60.0)
    assert parse_rate_limit("5/1") == (5, 1.0)
    for bad in ["", "abc", "0/60s", "5/0s", "-1/5s"]:
        with pytest.raises(ValueError):
            parse_rate_limit(bad)


# ---- debug scores ---------------------------------------------------------


def test_scores_hidden_by_default_and_shown_in_debug(threshold, clock, typists):
    alice, _, rng = typists
    off, _ = build(threshold, clock)
    enroll_all(off, "alice", alice, rng)
    r = off.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert "score" not in r.json() and "threshold" not in r.json()

    on, _ = build(threshold, clock, debug_scores=True)
    enroll_all(on, "alice", alice, rng)
    r = on.post("/verify", json={"user_id": "alice", "timings": timings_of(alice, rng)})
    assert isinstance(r.json()["score"], float) and r.json()["threshold"] == pytest.approx(threshold)


# ---- logging --------------------------------------------------------------


def test_logs_contain_no_timings_or_scores(threshold, clock, typists, caplog):
    alice, bob, rng = typists
    client, _ = build(threshold, clock, debug_scores=True)
    caplog.set_level(logging.DEBUG)
    sent = []
    responses = []
    for _ in range(REQUIRED):
        t = timings_of(alice, rng)
        sent.append(t)
        client.post("/enroll", json={"user_id": "alice", "timings": t})
    for typist in (alice, bob):
        t = timings_of(typist, rng)
        sent.append(t)
        responses.append(client.post("/verify", json={"user_id": "alice", "timings": t}))
    ev = events_of(alice, rng)
    client.post("/verify", json={"user_id": "alice", "events": ev})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "endpoint=enroll" in text and "endpoint=verify" in text
    assert "alice" not in text  # only a hash of the user_id is logged
    for t in sent:
        for row in t[:3]:
            for value in row[:1]:
                assert f"{value:.3f}" not in text and str(value) not in text
    for r in responses:
        assert f"{r.json()['score']:.3f}" not in text and str(r.json()["score"]) not in text
    for e in ev[:3]:
        assert str(e["t"]) not in text

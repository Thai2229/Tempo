"""Multi-site service (/v1): happy path and failure paths. See docs/SERVICE.md for what each test defends."""

import base64
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tempo.api import Settings, create_app  # noqa: E402
from tempo.api.tokens import TOKEN_TTL_S, TokenClaims, sign_token, verify_token  # noqa: E402
from tempo.api.v1 import TICKET_TTL_S  # noqa: E402
from tempo.auth import EngineConfig, TempoEngine, TemplateStore  # noqa: E402
from tempo.sites import create_site, main as sites_main, rotate_secret  # noqa: E402
from tempo.synthetic import make_typist  # noqa: E402

from test_api import PHRASE, REQUIRED, Clock, assert_error, fake_embedder, threshold, timings_of  # noqa: E402,F401

KEY = "k" * 48
ORIGIN_A = "https://a.example"
ORIGIN_B = "https://b.example"


class Env:
    def __init__(self, threshold, clock, **settings):
        config = EngineConfig(enroll_samples=REQUIRED, max_failures=1000, lockout_s=300.0)
        self.store = TemplateStore()
        self.engine = TempoEngine(fake_embedder, self.store, threshold, config, clock=clock)
        values = {"rate_limit": 10000, "enable_v1": True, "token_key": KEY}
        values.update(settings)
        self.app = create_app(self.engine, Settings(**values))
        self.app.state.limiter.clock = clock
        self.app.state.server_limiter.clock = clock
        self.client = TestClient(self.app)
        self.clock = clock
        self.site_a, self.secret_a = create_site(self.store, "A", [ORIGIN_A])
        self.site_b, self.secret_b = create_site(self.store, "B", [ORIGIN_B])
        rng = np.random.default_rng(7)
        self.typist = make_typist(PHRASE, rng)
        self.other = make_typist(PHRASE, rng)
        self.rng = rng

    def ticket(self, user="u1", secret=None, reset=False):
        r = self.client.post("/v1/tickets", json={"secret": secret or self.secret_a, "user_id": user, "reset": reset})
        assert r.status_code == 200, r.text
        return r.json()["ticket"]

    def enroll(self, user="u1", site=None, origin=ORIGIN_A, ticket=None, typist=None):
        site = site or self.site_a
        ticket = ticket or self.ticket(user)
        r = None
        for _ in range(REQUIRED):
            body = {"sitekey": site.site_key, "user_id": user, "ticket": ticket, "timings": timings_of(typist or self.typist, self.rng)}
            r = self.client.post("/v1/enroll", json=body, headers={"Origin": origin})
            assert r.status_code == 200, r.text
        return r

    def verify(self, user="u1", site=None, origin=ORIGIN_A, typist=None):
        site = site or self.site_a
        body = {"sitekey": site.site_key, "user_id": user, "timings": timings_of(typist or self.typist, self.rng)}
        return self.client.post("/v1/verify", json=body, headers={"Origin": origin})

    def siteverify(self, token, secret=None):
        return self.client.post("/v1/siteverify", json={"secret": secret or self.secret_a, "token": token})


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def env(threshold, clock):
    return Env(threshold, clock)


def claims(clock, **over):
    values = dict(site_id="s1", user_id="u1", rhythm_match=True, risk="low", issued_at=clock.t, expires_at=clock.t + TOKEN_TTL_S, jti="a" * 32)
    values.update(over)
    return TokenClaims(**values)


# ---- happy path ---------------------------------------------------------------


def test_end_to_end(env):
    r = env.enroll()
    assert r.json()["status"] == "complete"
    r = env.verify()
    assert r.status_code == 200 and set(r.json()) == {"token"}
    assert r.headers["access-control-allow-origin"] == ORIGIN_A and r.headers["vary"] == "Origin"
    r = env.siteverify(r.json()["token"])
    body = r.json()
    assert r.status_code == 200 and body["success"] is True and body["user_id"] == "u1"
    assert body["rhythm_match"] is True and body["risk"] in {"low", "medium"}


def test_impostor_gets_a_token_too_but_no_match(env):
    env.enroll()
    r = env.verify(typist=env.other)
    assert r.status_code == 200 and set(r.json()) == {"token"}
    body = env.siteverify(r.json()["token"]).json()
    assert body["success"] is True and body["rhythm_match"] is False and body["risk"] == "high"


def test_browser_response_is_uniform(env):
    """Match, mismatch, unknown user, replay and locked user all look the same to the browser."""
    env.enroll()
    shapes = set()
    shapes.add((env.verify().status_code, tuple(env.verify().json())))
    shapes.add((env.verify(typist=env.other).status_code, tuple(env.verify(typist=env.other).json())))
    shapes.add((env.verify(user="ghost").status_code, tuple(env.verify(user="ghost").json())))
    body = {"sitekey": env.site_a.site_key, "user_id": "u1", "timings": timings_of(env.typist, env.rng)}
    for _ in range(2):  # second one is a replay
        r = env.client.post("/v1/verify", json=body, headers={"Origin": ORIGIN_A})
        shapes.add((r.status_code, tuple(r.json())))
    assert shapes == {(200, ("token",))}
    assert "accepted" not in env.verify().text and "score" not in env.verify().text


def test_lockout_is_a_token_not_423(threshold, clock):
    e = Env(threshold, clock)
    e.engine.config.max_failures = 2
    e.enroll()
    for _ in range(3):
        e.verify(typist=e.other)
    r = e.verify()  # right typist, but locked
    assert r.status_code == 200
    assert e.siteverify(r.json()["token"]).json()["rhythm_match"] is False


def test_rate_limit_applies_to_browser_endpoints(threshold, clock):
    e = Env(threshold, clock, rate_limit=3)
    codes = [e.verify().status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    assert_error(e.verify(), 429, "rate_limited")


# ---- tokens: forged, altered, expired, reused --------------------------------


def test_token_roundtrip_and_bad_signature(clock):
    key = KEY.encode()
    token = sign_token(key, claims(clock))
    assert verify_token(key, token, clock.t) is not None
    assert verify_token(b"x" * 48, token, clock.t) is None
    payload, sig = token.split(".")
    flipped = sig[:-2] + "AA"
    if sig.endswith("AA"):
        flipped = sig[:-2] + "BB"
    assert verify_token(key, payload + "." + flipped, clock.t) is None


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def test_altered_payload_keeps_old_signature(clock):
    key = KEY.encode()
    token = sign_token(key, claims(clock, rhythm_match=False))
    payload, sig = token.split(".")
    data = json.loads(base64.urlsafe_b64decode(payload + "=="))
    for field, value in (("m", True), ("uid", "victim"), ("sid", "s2"), ("exp", clock.t + 9999), ("r", "high")):
        forged = dict(data)
        forged[field] = value
        text = b64(json.dumps(forged, sort_keys=True, separators=(",", ":")).encode())
        assert verify_token(key, text + "." + sig, clock.t) is None, field


def test_alg_confusion_style_tokens(clock):
    key = KEY.encode()
    good = json.loads(base64.urlsafe_b64decode(sign_token(key, claims(clock)).split(".")[0] + "=="))
    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    body = b64(json.dumps(good).encode())
    forged = [
        header + "." + body + ".",  # JWT-shaped, alg none, empty signature
        body + ".",  # empty signature
        body + "." + b64(b"\x00" * 32),  # all-zero signature
        body,  # no signature at all
        "." + b64(b"x"),
        body + "." + body + "." + body,
        "",
    ]
    for token in forged:
        assert verify_token(key, token, clock.t) is None, token
    # signed with an empty key / a key derived from public data
    import hashlib, hmac
    for k in (b"", b"tempo", ORIGIN_A.encode()):
        sig = b64(hmac.new(k, b"tempo-token-v1." + body.encode(), hashlib.sha256).digest())
        assert verify_token(key, body + "." + sig, clock.t) is None
    # a MAC without the domain prefix is also not accepted
    sig = b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    assert verify_token(key, body + "." + sig, clock.t) is None


def test_malformed_claims_are_rejected_even_when_signed(clock):
    import hashlib, hmac
    key = KEY.encode()

    def signed(payload):
        text = b64(json.dumps(payload).encode())
        return text + "." + b64(hmac.new(key, b"tempo-token-v1." + text.encode(), hashlib.sha256).digest())

    ok = dict(v=1, sid="s", uid="u", m=True, r="low", iat=clock.t, exp=clock.t + 30, jti="a" * 32)
    assert verify_token(key, signed(ok), clock.t) is not None
    for bad in (dict(ok, v=2), dict(ok, m=1), dict(ok, r="none"), dict(ok, exp=clock.t + 3600), dict(ok, jti="short"), dict(ok, iat=clock.t + 500)):
        assert verify_token(key, signed(bad), clock.t) is None, bad
    assert verify_token(key, signed([1, 2]), clock.t) is None


def test_forged_token_over_http(env):
    for token in ("garbage", "a.b", sign_token(b"y" * 48, claims(env.clock, site_id=env.site_a.site_id))):
        r = env.siteverify(token)
        assert r.status_code == 200 and r.json() == {"success": False, "error": "invalid_token"}


def test_expired_token(env):
    env.enroll()
    token = env.verify().json()["token"]
    env.clock.t += TOKEN_TTL_S + 1
    assert env.siteverify(token).json() == {"success": False, "error": "invalid_token"}


def test_token_valid_until_expiry(env):
    env.enroll()
    token = env.verify().json()["token"]
    env.clock.t += TOKEN_TTL_S - 1
    assert env.siteverify(token).json()["success"] is True


def test_reused_token(env):
    env.enroll()
    token = env.verify().json()["token"]
    assert env.siteverify(token).json()["success"] is True
    assert env.siteverify(token).json() == {"success": False, "error": "invalid_token"}


def test_failure_reasons_are_indistinguishable(env):
    env.enroll()
    token = env.verify().json()["token"]
    other_site = env.siteverify(token, secret=env.secret_b)
    forged = env.siteverify("x.y")
    env.siteverify(token)
    reused = env.siteverify(token)
    assert other_site.json() == forged.json() == reused.json()


def test_concurrent_double_consume(env):
    env.enroll()
    token = env.verify().json()["token"]
    barrier = threading.Barrier(8)

    def call(_):
        barrier.wait()
        return env.siteverify(token).json()["success"]

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(call, range(8)))
    assert results.count(True) == 1


def test_store_consume_is_atomic_and_cleans_up():
    store = TemplateStore()
    barrier = threading.Barrier(16)
    wins = []

    def call():
        barrier.wait()
        wins.append(store.consume_token("jti-1", 100.0, 10.0))

    threads = [threading.Thread(target=call) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert wins.count(True) == 1
    assert store.consume_token("jti-2", 50.0, 200.0) is True  # purges expired rows
    count = store._conn.execute("SELECT COUNT(*) FROM used_tokens").fetchone()[0]
    assert count == 1


# ---- secrets, sites -----------------------------------------------------------


def test_wrong_secret(env):
    env.enroll()
    token = env.verify().json()["token"]
    for secret in ("nope", "sk." + env.site_a.site_id + ".wrong", env.secret_a + "x", "sk.unknown.abc", "sk..", "." * 256):
        assert_error(env.siteverify(token, secret=secret), 401, "unauthorized")
        assert_error(env.client.post("/v1/tickets", json={"secret": secret, "user_id": "u1"}), 401, "unauthorized")
    assert env.siteverify(token).json()["success"] is True  # bad secrets did not burn the token


def test_token_for_site_a_with_site_b_secret(env):
    env.enroll()
    token = env.verify().json()["token"]
    assert env.siteverify(token, secret=env.secret_b).json() == {"success": False, "error": "invalid_token"}
    assert env.siteverify(token).json()["success"] is True  # B could not burn A's token


def test_templates_are_per_site(env):
    env.enroll(user="u1")
    r = env.verify(user="u1", site=env.site_b, origin=ORIGIN_B)  # u1 not enrolled at B
    assert env.siteverify(r.json()["token"], secret=env.secret_b).json()["rhythm_match"] is False
    assert env.store.get_user(env.site_a.site_id, "u1") is not None
    assert env.store.get_user(env.site_b.site_id, "u1") is None


def test_token_for_user_x_is_x(env):
    env.enroll(user="x")
    env.enroll(user="y", typist=env.other)
    token = env.verify(user="x").json()["token"]
    body = env.siteverify(token).json()
    assert body["user_id"] == "x" and body["user_id"] != "y"
    # y's rhythm presented as x does not match
    bad = env.siteverify(env.verify(user="x", typist=env.other).json()["token"]).json()
    assert bad["user_id"] == "x" and bad["rhythm_match"] is False


def test_unknown_and_disabled_site(env):
    env.enroll()
    body = {"sitekey": "pk_nope", "user_id": "u1", "timings": timings_of(env.typist, env.rng)}
    assert_error(env.client.post("/v1/verify", json=body, headers={"Origin": ORIGIN_A}), 403, "forbidden")
    token = env.verify().json()["token"]
    env.store.set_site_disabled(env.site_a.site_id, True)
    assert_error(env.verify(), 403, "forbidden")
    assert_error(env.client.post("/v1/tickets", json={"secret": env.secret_a, "user_id": "z"}), 401, "unauthorized")
    assert_error(env.siteverify(token), 401, "unauthorized")
    env.store.set_site_disabled(env.site_a.site_id, False)
    assert env.siteverify(token).json()["success"] is True


def test_rotate_secret_invalidates_old(env):
    new = rotate_secret(env.store, env.site_a.site_id)
    assert_error(env.client.post("/v1/tickets", json={"secret": env.secret_a, "user_id": "z"}), 401, "unauthorized")
    assert env.client.post("/v1/tickets", json={"secret": new, "user_id": "z"}).status_code == 200


def test_secret_never_stored_in_plaintext(env):
    ticket = env.ticket()
    env.enroll(ticket=ticket)
    token = env.verify().json()["token"]
    env.siteverify(token)
    dump = "\n".join(env.store._conn.iterdump())
    for secret in (env.secret_a, env.secret_b, env.secret_a.split(".")[2], env.secret_b.split(".")[2], ticket, token, KEY):
        assert secret not in dump
    row = env.store.get_site(env.site_a.site_id)
    assert row.secret_hash != env.secret_a and len(row.secret_salt) == 32


def test_secret_not_in_logs(env, caplog):
    caplog.set_level(logging.DEBUG)
    ticket = env.ticket()
    env.enroll(ticket=ticket)
    token = env.verify().json()["token"]
    env.siteverify(token)
    env.siteverify(token)
    env.client.post("/v1/siteverify", json={"secret": "sk.bad.bad", "token": token})
    for secret in (env.secret_a, env.secret_a.split(".")[2], ticket, token, KEY):
        assert secret not in caplog.text
    assert "u1" not in caplog.text  # only the hashed user tag


def test_siteverify_brute_force_is_rate_limited(threshold, clock):
    e = Env(threshold, clock, server_rate_limit=5)
    codes = [e.client.post("/v1/siteverify", json={"secret": f"sk.{e.site_a.site_id}.guess{i}", "token": "x"}).status_code for i in range(8)]
    assert codes[:5] == [401] * 5 and set(codes[5:]) == {429}
    assert_error(e.client.post("/v1/siteverify", json={"secret": e.secret_a, "token": "x"}), 429, "rate_limited")
    clock.t += 61
    assert e.client.post("/v1/siteverify", json={"secret": e.secret_a, "token": "x"}).status_code == 200


# ---- tickets ------------------------------------------------------------------


def enroll_once(env, ticket, user="u1", site=None, origin=ORIGIN_A):
    site = site or env.site_a
    body = {"sitekey": site.site_key, "user_id": user, "ticket": ticket, "timings": timings_of(env.typist, env.rng)}
    return env.client.post("/v1/enroll", json=body, headers={"Origin": origin})


def test_enroll_without_or_with_bad_ticket(env):
    body = {"sitekey": env.site_a.site_key, "user_id": "u1", "timings": timings_of(env.typist, env.rng)}
    r = env.client.post("/v1/enroll", json=body, headers={"Origin": ORIGIN_A})
    assert_error(r, 422, "validation_error")
    assert_error(enroll_once(env, "made-up-ticket"), 403, "invalid_ticket")
    assert env.store.get_user(env.site_a.site_id, "u1") is None


def test_ticket_for_x_cannot_enroll_y(env):
    ticket = env.ticket(user="x")
    assert_error(enroll_once(env, ticket, user="y"), 403, "invalid_ticket")
    assert env.store.get_user(env.site_a.site_id, "y") is None
    assert enroll_once(env, ticket, user="x").status_code == 200


def test_ticket_for_site_a_does_not_work_at_site_b(env):
    ticket = env.ticket(user="x")
    assert_error(enroll_once(env, ticket, user="x", site=env.site_b, origin=ORIGIN_B), 403, "invalid_ticket")


def test_expired_ticket(env):
    ticket = env.ticket()
    env.clock.t += TICKET_TTL_S + 1
    assert_error(enroll_once(env, ticket), 403, "invalid_ticket")


def test_ticket_is_used_up_by_one_enrollment(env):
    ticket = env.ticket()
    env.enroll(ticket=ticket)
    assert_error(enroll_once(env, ticket), 403, "invalid_ticket")


def test_victim_cannot_be_re_enrolled_from_the_browser(env):
    env.enroll()
    token_before = env.verify().json()["token"]
    assert_error(enroll_once(env, "guess"), 403, "invalid_ticket")
    assert env.siteverify(token_before).json()["rhythm_match"] is True
    # the customer's server must opt in to a reset
    r = env.client.post("/v1/tickets", json={"secret": env.secret_a, "user_id": "u1"})
    assert_error(r, 409, "already_enrolled")
    assert env.enroll(ticket=env.ticket(reset=True), typist=env.other).json()["status"] == "complete"
    matches = [env.siteverify(env.verify(typist=env.other).json()["token"]).json()["rhythm_match"] for _ in range(5)]
    assert any(matches)  # the template is now the other typist's


def test_new_ticket_replaces_old(env):
    first = env.ticket()
    second = env.ticket()
    assert_error(enroll_once(env, first), 403, "invalid_ticket")
    assert enroll_once(env, second).status_code == 200


# ---- CORS / origin ------------------------------------------------------------


def test_wrong_and_missing_origin(env):
    env.enroll()
    assert_error(env.verify(origin=ORIGIN_B), 403, "forbidden")
    assert_error(env.verify(origin="https://a.example.evil.com"), 403, "forbidden")
    assert_error(env.verify(origin="null"), 403, "forbidden")
    body = {"sitekey": env.site_a.site_key, "user_id": "u1", "timings": timings_of(env.typist, env.rng)}
    r = env.client.post("/v1/verify", json=body)
    assert_error(r, 403, "forbidden")
    assert "access-control-allow-origin" not in r.headers
    assert_error(enroll_once(env, env.ticket(user="u2"), user="u2", origin=ORIGIN_B), 403, "forbidden")


def test_preflight(env):
    r = env.client.options("/v1/verify", headers={"Origin": ORIGIN_A, "Access-Control-Request-Method": "POST"})
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == ORIGIN_A and r.headers["access-control-allow-origin"] != "*"
    assert r.headers["access-control-allow-methods"] == "POST" and r.headers["vary"] == "Origin"
    assert env.client.options("/v1/enroll", headers={"Origin": ORIGIN_B}).headers["access-control-allow-origin"] == ORIGIN_B
    for headers in ({"Origin": "https://evil.example"}, {}):
        r = env.client.options("/v1/verify", headers=headers)
        assert_error(r, 403, "forbidden")
        assert "access-control-allow-origin" not in r.headers


def test_error_responses_to_allowed_origin_carry_cors(env):
    r = enroll_once(env, "bad-ticket")
    assert r.status_code == 403 and r.headers["access-control-allow-origin"] == ORIGIN_A


def test_server_endpoints_are_not_cross_origin(env):
    token = "x"
    r = env.client.post("/v1/siteverify", json={"secret": env.secret_a, "token": token}, headers={"Origin": ORIGIN_A})
    assert_error(r, 403, "forbidden")
    assert "access-control-allow-origin" not in r.headers
    r = env.client.post("/v1/tickets", json={"secret": env.secret_a, "user_id": "u1"}, headers={"Origin": ORIGIN_A})
    assert_error(r, 403, "forbidden")
    r = env.client.options("/v1/siteverify", headers={"Origin": ORIGIN_A, "Access-Control-Request-Method": "POST"})
    assert r.status_code == 405 and "access-control-allow-origin" not in r.headers


def test_user_ids_are_opaque(env):
    for bad in ("alice@example.com", "a b", "", "x" * 65, "../x", "a/b"):
        r = env.client.post("/v1/tickets", json={"secret": env.secret_a, "user_id": bad})
        assert_error(r, 422, "validation_error")
        body = {"sitekey": env.site_a.site_key, "user_id": bad, "timings": [[0.1, 0.2, 0.1]] * 3}
        assert_error(env.client.post("/v1/verify", json=body, headers={"Origin": ORIGIN_A}), 422, "validation_error")


# ---- configuration, CLI, demo API untouched -------------------------------------


@pytest.mark.parametrize("key", [None, "", "short"])
def test_refuses_to_start_without_a_proper_key(threshold, clock, key):
    engine = TempoEngine(fake_embedder, TemplateStore(), threshold, clock=clock)
    with pytest.raises(ValueError) as e:
        create_app(engine, Settings(enable_v1=True, token_key=key))
    assert "TEMPO_TOKEN_KEY" in str(e.value)
    if key:
        assert key not in str(e.value)


def test_v1_off_by_default_and_demo_api_unchanged(threshold, clock):
    engine = TempoEngine(fake_embedder, TemplateStore(), threshold, clock=clock)
    client = TestClient(create_app(engine, Settings()))
    assert_error(client.post("/v1/siteverify", json={"secret": "x", "token": "y"}), 404, "not_found")
    assert client.get("/health").status_code == 200
    assert client.get("/status", params={"user_id": "alice@example.com"}).status_code == 404


def test_settings_from_env():
    s = Settings.from_env({"TEMPO_ENABLE_V1": "1", "TEMPO_TOKEN_KEY": KEY, "TEMPO_SERVER_RATE_LIMIT": "5/10s"})
    assert s.enable_v1 and s.token_key == KEY and s.server_rate_limit == 5 and s.server_rate_window_s == 10.0
    assert Settings.from_env({}).enable_v1 is False and Settings.from_env({}).token_key is None


def test_cli_prints_secret_once(tmp_path, capsys):
    db = str(tmp_path / "t.db")
    assert sites_main(["--db", db, "create", "--name", "Shop", "--origin", "https://Shop.example", "--origin", "http://localhost:3000"]) == 0
    out = capsys.readouterr().out
    secret = next(line.split()[-1] for line in out.splitlines() if line.startswith("secret:"))
    assert secret.startswith("sk.")
    assert sites_main(["--db", db, "list"]) == 0
    listing = capsys.readouterr().out
    assert "https://shop.example" in listing and "http://localhost:3000" in listing and secret not in listing
    assert secret.split(".")[2] not in open(db, "rb").read().decode("latin-1")
    site_id = secret.split(".")[1]
    assert sites_main(["--db", db, "disable", site_id]) == 0
    store = TemplateStore(db)
    assert store.get_site(site_id).disabled is True
    store.close()
    assert sites_main(["--db", db, "rotate-secret", site_id]) == 0
    assert secret not in capsys.readouterr().out
    assert sites_main(["--db", db, "disable", "nope"]) == 1


@pytest.mark.parametrize("origin", ["*", "example.com", "https://example.com/path", "http://example.com", "ftp://x.com", "https://u:p@x.com"])
def test_cli_rejects_bad_origins(tmp_path, origin):
    assert sites_main(["--db", str(tmp_path / "t.db"), "create", "--name", "x", "--origin", origin]) == 2

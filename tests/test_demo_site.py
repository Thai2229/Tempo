"""Server logic of demo_site (the independent customer site), against a stub keyauth. No model, no network."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from demo_site.app import COOKIE, create_app, decide_login  # noqa: E402
from demo_site.config import SiteConfig  # noqa: E402
from demo_site.keyauth_client import KeyauthClient  # noqa: E402
from demo_site.store import Store, check_password, hash_password  # noqa: E402

SECRET = "sk.site1.SUPER-SECRET-VALUE"
SITEKEY = "pk_public_key"
KEYAUTH = "http://127.0.0.1:8765"
PASSWORD = "correct horse"


class StubKeyauth:
    """Records calls; answers siteverify with `verdict` (None = unreachable)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.verdict: dict[str, Any] | None = None
        self.ticket_status = 200

    def create_ticket(self, user_id: str, reset: bool) -> tuple[int, dict[str, Any] | None]:
        self.calls.append(("ticket", (user_id, reset)))
        if self.ticket_status == 200:
            return 200, {"ticket": "tk_" + user_id, "expires_in": 600}
        return self.ticket_status, {"error": {"code": "x"}}

    def siteverify(self, token: str) -> dict[str, Any] | None:
        self.calls.append(("siteverify", token))
        return self.verdict

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


def make_config(tmp_path, **over: Any) -> SiteConfig:
    values = dict(keyauth_url=KEYAUTH, site_secret=SECRET, site_key=SITEKEY, db_path=str(tmp_path / "site.db"), pepper=b"p" * 32)
    values.update(over)
    return SiteConfig(**values)


@pytest.fixture
def stub() -> StubKeyauth:
    return StubKeyauth()


@pytest.fixture
def config(tmp_path) -> SiteConfig:
    return make_config(tmp_path)


@pytest.fixture
def client(config, stub) -> TestClient:
    return TestClient(create_app(config, client=stub), follow_redirects=False)


def register(c: TestClient, name: str = "alice", password: str = PASSWORD):
    return c.post("/register", data={"username": name, "password": password})


def opaque_id(config: SiteConfig, name: str = "alice") -> str:
    return Store(config.db_path).get_user(name).opaque_id


def fresh(config: SiteConfig, stub: StubKeyauth) -> TestClient:
    """A second browser (no cookies) against the same site."""
    return TestClient(create_app(config, client=stub), follow_redirects=False)


def login(c: TestClient, token: str = "", name: str = "alice", password: str = PASSWORD):
    return c.post("/login", data={"username": name, "password": password, "keyauth_token": token})


def logged_in(c: TestClient) -> bool:
    return c.get("/account").status_code == 200


# ---- passwords, sessions, storage ------------------------------------------------------------------------------


def test_scrypt_hash_salted_and_verified():
    a = hash_password("pw-12345678")
    b = hash_password("pw-12345678")
    assert a != b and a.startswith("scrypt$")
    assert check_password("pw-12345678", a) and not check_password("pw-12345679", a)
    assert not check_password("x", "garbage") and not check_password("x", "md5$1$2$3$AA==$AA==")


def test_no_plaintext_in_database_and_opaque_id_is_random(client, config):
    register(client, "alice@example.com", "my-secret-pw")
    raw = open(config.db_path, "rb").read()
    assert b"my-secret-pw" not in raw
    user = Store(config.db_path).get_user("alice@example.com")
    assert user.opaque_id.startswith("u_") and len(user.opaque_id) == 34
    assert "alice" not in user.opaque_id and "@" not in user.opaque_id
    with sqlite3.connect(config.db_path) as db:
        stored = db.execute("SELECT password_hash FROM users").fetchone()[0]
        sessions = db.execute("SELECT sid_hash FROM sessions").fetchall()
    assert stored.startswith("scrypt$")
    sid = client.cookies.get(COOKIE)
    assert sid and all(sid != row[0] for row in sessions)  # only a hash of the cookie value is stored


def test_session_cookie_flags(client):
    response = register(client)
    assert response.status_code == 303
    cookie = response.headers["set-cookie"].lower()
    assert COOKIE in cookie and "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    assert "secure" not in cookie


def test_secure_flag_when_configured(tmp_path, stub):
    c = TestClient(create_app(make_config(tmp_path, secure_cookie=True), client=stub), follow_redirects=False)
    assert "secure" in register(c).headers["set-cookie"].lower()


def test_registration_validation_and_duplicates(client):
    assert client.post("/register", data={"username": "al", "password": PASSWORD}).status_code == 422
    assert client.post("/register", data={"username": "alice", "password": "short"}).status_code == 422
    assert register(client).status_code == 303
    assert register(client, "ALICE").status_code == 409  # case-insensitive username


def test_logout_deletes_session_server_side(client):
    register(client)
    sid = client.cookies.get(COOKIE)
    client.post("/logout")
    client.cookies.set(COOKIE, sid)
    assert not logged_in(client)


def test_garbage_cookie_and_anonymous_get_login(client):
    client.cookies.set(COOKIE, "nope")
    assert client.get("/account").status_code == 303
    assert client.get("/").headers["location"] == "/login"


# ---- the login policy ---------------------------------------------------------------------------------------


def test_wrong_password_never_reaches_siteverify(client, config, stub):
    register(client)
    c2 = fresh(config, stub)
    response = login(c2, token="some.token", password="wrong password")
    assert response.status_code == 401 and "Invalid username or password" in response.text
    assert "siteverify" not in stub.names()
    assert not logged_in(c2)


def test_unknown_user_same_answer_as_wrong_password(client, config, stub):
    register(client)
    a = login(fresh(config, stub), "t", name="alice", password="nope nope nope")
    b = login(fresh(config, stub), "t", name="nobody", password="nope nope nope")
    assert (a.status_code, a.text) == (b.status_code, b.text)
    assert stub.names() == []


def test_rhythm_match_logs_in(client, config, stub):
    register(client)
    c2 = fresh(config, stub)
    stub.verdict = {"success": True, "user_id": opaque_id(config), "rhythm_match": True, "risk": "low"}
    response = login(c2, "tok")
    assert response.status_code == 303 and response.headers["location"] == "/account"
    assert stub.calls == [("siteverify", "tok")]
    assert logged_in(c2)


def test_user_id_mismatch_is_not_a_rhythm_login(client, config, stub):
    register(client)
    register(fresh(config, stub), "bob")
    c2 = fresh(config, stub)
    # a valid, matching token, but minted for bob, presented with alice's password
    stub.verdict = {"success": True, "user_id": opaque_id(config, "bob"), "rhythm_match": True, "risk": "low"}
    response = login(c2, "tok")
    assert response.headers["location"] == "/login/confirm"
    assert not logged_in(c2)


def test_rhythm_mismatch_goes_to_second_step_not_denial(client, config, stub, caplog):
    register(client)
    c2 = fresh(config, stub)
    stub.verdict = {"success": True, "user_id": opaque_id(config), "rhythm_match": False, "risk": "high"}
    with caplog.at_level(logging.INFO, logger="demo_site"):
        response = login(c2, "tok")
    assert response.headers["location"] == "/login/confirm"
    assert not logged_in(c2)
    assert c2.get("/login/confirm").status_code == 200
    code = [r.getMessage().rsplit("code=", 1)[1] for r in caplog.records if "SECOND-STEP CODE" in r.getMessage()][0]
    wrong = "000000"
    if code == wrong:
        wrong = "111111"
    assert c2.post("/login/confirm", data={"code": wrong}).status_code == 401
    assert not logged_in(c2)
    assert c2.post("/login/confirm", data={"code": code}).headers["location"] == "/account"
    assert logged_in(c2)


def test_second_step_code_expires_with_attempt_limit(client, config, stub):
    register(client)
    c2 = fresh(config, stub)
    stub.verdict = {"success": False, "error": "invalid_token"}
    login(c2, "tok")
    for _ in range(5):
        assert c2.post("/login/confirm", data={"code": "abc"}).status_code == 401
    assert c2.post("/login/confirm", data={"code": "123456"}).status_code == 303
    assert c2.get("/login/confirm").headers["location"] == "/login"  # the pending session is gone


def test_pending_session_is_not_logged_in_and_cannot_get_a_ticket(client, config, stub):
    register(client)
    c2 = fresh(config, stub)
    stub.verdict = {"success": False, "error": "invalid_token"}
    login(c2, "tok")
    assert c2.post("/account/ticket", json={}).status_code == 401
    assert "ticket" not in stub.names()


def test_siteverify_failure_goes_to_second_step(client, config, stub):
    register(client)
    c2 = fresh(config, stub)
    stub.verdict = {"success": False, "error": "invalid_token"}
    assert login(c2, "tampered").headers["location"] == "/login/confirm"
    assert not logged_in(c2)


def test_oversized_token_is_invalid_not_fail_open(client, config, stub):
    register(client)
    c2 = fresh(config, stub)
    assert login(c2, "a" * 5000).headers["location"] == "/login/confirm"
    assert stub.names() == []


def test_unreachable_keyauth_fails_open_and_is_flagged(client, config, stub, caplog):
    register(client)
    c2 = fresh(config, stub)
    stub.verdict = None
    with caplog.at_level(logging.INFO, logger="demo_site"):
        response = login(c2, "tok")
    assert response.headers["location"] == "/account" and logged_in(c2)
    flagged = [r for r in caplog.records if r.levelno >= logging.WARNING and "FLAG" in r.getMessage()]
    assert flagged and "keyauth_unreachable" in flagged[0].getMessage()


def test_empty_token_fails_open_and_is_flagged(client, config, stub, caplog):
    register(client)
    c2 = fresh(config, stub)
    with caplog.at_level(logging.INFO, logger="demo_site"):
        response = login(c2, "")
    assert response.headers["location"] == "/account"
    assert "siteverify" not in stub.names()
    assert any("FLAG" in r.getMessage() and "no_token" in r.getMessage() for r in caplog.records)


def test_fail_closed_mode(tmp_path, stub):
    c = TestClient(create_app(make_config(tmp_path, fail_mode="closed"), client=stub), follow_redirects=False)
    register(c)
    c2 = TestClient(create_app(make_config(tmp_path, fail_mode="closed"), client=stub), follow_redirects=False)
    assert login(c2, "").headers["location"] == "/login/confirm"
    stub.verdict = None
    assert login(c2, "tok").headers["location"] == "/login/confirm"


def test_decide_login_table():
    me = "u_me"
    ok = {"success": True, "user_id": me, "rhythm_match": True, "risk": "low"}
    assert decide_login("open", "t", ok, me).action == "allow"
    assert decide_login("open", "t", {**ok, "rhythm_match": False}, me).action == "second_step"
    assert decide_login("open", "t", {**ok, "user_id": "u_other"}, me).action == "second_step"
    assert decide_login("open", "t", {**ok, "success": "true"}, me).action == "second_step"
    assert decide_login("open", "t", {"success": False}, me).action == "second_step"
    assert decide_login("open", "t", None, me).action == "allow_flagged"
    assert decide_login("open", "", None, me).action == "allow_flagged"
    assert decide_login("closed", "", None, me).action == "second_step"


# ---- identify (username enumeration) ---------------------------------------------------------------------------


def test_identify_does_not_reveal_whether_a_username_exists(client):
    register(client)
    real = client.post("/login/identify", json={"username": "alice"})
    fake = client.post("/login/identify", json={"username": "nobody"})
    assert real.status_code == fake.status_code == 200
    r, f = real.json()["user_id"], fake.json()["user_id"]
    assert len(r) == len(f) == 34 and r.startswith("u_") and f.startswith("u_")
    assert client.post("/login/identify", json={"username": "nobody"}).json()["user_id"] == f  # stable
    assert client.post("/login/identify", json={"username": "other"}).json()["user_id"] != f
    assert f != "u_" + "nobody"
    assert client.post("/login/identify", content=b"x").status_code == 400


# ---- tickets ------------------------------------------------------------------------------------------------


def test_ticket_only_for_logged_in_session(client, config, stub):
    anon = fresh(config, stub)
    assert anon.post("/account/ticket", json={}).status_code == 401
    assert stub.calls == []
    register(client)
    response = client.post("/account/ticket", json={"user_id": "u_victim"})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == opaque_id(config) and body["sitekey"] == SITEKEY  # the id comes from the session
    assert stub.calls == [("ticket", (opaque_id(config), False))]


def test_register_does_not_issue_a_ticket_and_wrong_password_neither(client, config, stub):
    register(client)
    login(fresh(config, stub), "tok", password="wrong password")
    assert stub.calls == []


def test_ticket_reset_and_conflict(client, stub):
    register(client)
    client.post("/account/ticket", json={"reset": True})
    assert stub.calls[-1][1][1] is True
    stub.ticket_status = 409
    assert client.post("/account/ticket", json={}).status_code == 409
    stub.ticket_status = 401
    assert client.post("/account/ticket", json={}).status_code == 502


# ---- secrets and CSP -------------------------------------------------------------------------------------------


def test_pages_have_strict_csp_and_no_secret_or_inline_script(client, config):
    register(client)
    for path in ["/login", "/register", "/account", "/static/login.js", "/static/account.js", "/static/site.css"]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert SECRET not in response.text and "sk." not in response.text.replace("task.", "")
        csp = response.headers["content-security-policy"]
        assert f"script-src 'self' {KEYAUTH};" in csp and f"connect-src 'self' {KEYAUTH};" in csp
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
        if path in {"/login", "/account"}:
            assert f'<script src="{KEYAUTH}/v1/keyauth.js" defer>' in response.text
            assert f'data-sitekey="{SITEKEY}"' in response.text
        assert "<script>" not in response.text and " onclick=" not in response.text


def test_username_is_escaped(client):
    client.post("/register", data={"username": "<b>x</b>", "password": PASSWORD})
    assert client.get("/account").status_code == 303  # invalid name was refused, nothing rendered
    register(client, "a.b+c@d")
    assert "a.b+c@d" in client.get("/account").text


def test_config_requires_env_and_hides_secret():
    with pytest.raises(ValueError, match="KEYAUTH_SITE_SECRET"):
        SiteConfig.from_env({"KEYAUTH_URL": KEYAUTH, "KEYAUTH_SITE_KEY": "k"})
    cfg = SiteConfig.from_env({"KEYAUTH_URL": KEYAUTH + "/x/", "KEYAUTH_SITE_SECRET": SECRET, "KEYAUTH_SITE_KEY": "k"})
    assert cfg.keyauth_url == KEYAUTH and SECRET not in repr(cfg) and cfg.fail_mode == "open"
    with pytest.raises(ValueError):
        SiteConfig.from_env({"KEYAUTH_URL": KEYAUTH, "KEYAUTH_SITE_SECRET": "s", "KEYAUTH_SITE_KEY": "k", "DEMO_SITE_FAIL_MODE": "maybe"})


# ---- the real HTTP client against a tiny local stub ------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append({"path": self.path, "origin": self.headers.get("Origin"), "body": body})
        if self.path == "/v1/siteverify":
            payload, status = {"success": True, "user_id": "u_x", "rhythm_match": True, "risk": "low"}, 200
        else:
            payload, status = {"error": {"code": "unauthorized"}}, 401
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: Any) -> None:
        return


def test_real_client_sends_secret_in_body_without_origin():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        c = KeyauthClient(f"http://127.0.0.1:{server.server_port}", SECRET)
        assert c.siteverify("tok")["rhythm_match"] is True
        assert c.siteverify("tok") is not None
        status, _ = c.create_ticket("u_x", False)
        assert status == 401
        assert Handler.seen[0] == {"path": "/v1/siteverify", "origin": None, "body": {"secret": SECRET, "token": "tok"}}
    finally:
        server.shutdown()


def test_real_client_maps_unreachable_and_errors_to_none():
    c = KeyauthClient("http://127.0.0.1:9", SECRET, timeout_s=0.5)  # nothing listens on the discard port
    assert c.siteverify("tok") is None
    assert c.create_ticket("u_x", False) == (0, None)


def test_account_page_says_how_the_session_was_authenticated(client, config, stub):
    register(client)
    assert 'data-method="registered"' in client.get("/account").text
    c2 = fresh(config, stub)
    stub.verdict = {"success": True, "user_id": opaque_id(config), "rhythm_match": True, "risk": "medium"}
    login(c2, "tok")
    assert 'data-method="rhythm"' in c2.get("/account").text
    c3 = fresh(config, stub)
    login(c3, "")
    assert 'data-method="password_only"' in c3.get("/account").text

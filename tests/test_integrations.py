"""Customer-server helpers (integrations/python, integrations/node) against a local stub HTTP server. No model needed."""

import importlib.util
import json
import logging
import shutil
import socket
import sys
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SECRET = "sk.s0123456789ab.SUPER-SECRET-VALUE"

_spec = importlib.util.spec_from_file_location("tempo_verify", ROOT / "integrations" / "python" / "tempo_verify.py")
tv = importlib.util.module_from_spec(_spec)
sys.modules["tempo_verify"] = tv  # dataclasses look the module up here
_spec.loader.exec_module(tv)


class Stub:
    """A tiny HTTP server: `reply` is (status, body bytes, extra headers) or a callable returning that; requests are recorded."""

    def __init__(self) -> None:
        self.reply = (200, b"{}", {})
        self.delay = 0.0
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                stub.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
                if stub.delay:
                    time.sleep(stub.delay)
                status, payload, headers = stub.reply
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def json(self, status: int, data: object, headers: dict | None = None) -> None:
        self.reply = (status, json.dumps(data).encode(), headers or {})

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


OK_BODY = {"success": True, "user_id": "u_1", "rhythm_match": True, "risk": "low"}


def test_verify_success(stub):
    stub.json(200, OK_BODY)
    r = tv.verify(SECRET, "tok", stub.url)
    assert r == tv.VerifyResult(True, "u_1", True, "low", None)
    sent = stub.requests[0]
    assert sent["path"] == "/v1/siteverify"
    assert sent["body"] == {"secret": SECRET, "token": "tok"}
    assert "Origin" not in sent["headers"]


def test_verify_mismatch_is_success_with_rhythm_false(stub):
    stub.json(200, {"success": True, "user_id": "u_1", "rhythm_match": False, "risk": "high"})
    r = tv.verify(SECRET, "tok", stub.url + "/")
    assert r.success and not r.rhythm_match and r.risk == "high"


def test_verify_invalid_token(stub):
    stub.json(200, {"success": False, "error": "invalid_token"})
    r = tv.verify(SECRET, "tok", stub.url)
    assert (r.success, r.reason, r.user_id, r.rhythm_match) == (False, "invalid_token", None, False)


def test_verify_wrong_secret_401(stub):
    stub.json(401, {"error": {"code": "unauthorized", "message": "invalid secret"}})
    r = tv.verify(SECRET, "tok", stub.url)
    assert (r.success, r.reason) == (False, "unauthorized")


def test_verify_rate_limited(stub):
    stub.json(429, {"error": {"code": "rate_limited", "message": "x"}}, {"Retry-After": "5"})
    assert tv.verify(SECRET, "tok", stub.url).reason == "rate_limited"


@pytest.mark.parametrize("status", [200, 500, 502])
def test_verify_non_json(stub, status):
    stub.reply = (status, b"<html>Bad gateway</html>", {})
    r = tv.verify(SECRET, "tok", stub.url)
    assert (r.success, r.reason) == (False, "bad_response")


@pytest.mark.parametrize("body", [b"[]", b'"x"', b"null", b'{"success": true}', b'{"success": true, "user_id": 5, "rhythm_match": "yes", "risk": "low"}'])
def test_verify_wrong_json_shape(stub, body):
    stub.reply = (200, body, {})
    r = tv.verify(SECRET, "tok", stub.url)
    assert (r.success, r.reason) == (False, "bad_response")


def test_verify_success_false_without_error_text(stub):
    stub.json(200, {"success": False})
    assert tv.verify(SECRET, "tok", stub.url).reason == "invalid_token"


def test_verify_timeout(stub):
    stub.json(200, OK_BODY)
    stub.delay = 1.5
    started = time.monotonic()
    r = tv.verify(SECRET, "tok", stub.url, timeout=0.2)
    assert (r.success, r.reason) == (False, "unreachable")
    assert time.monotonic() - started < 1.2


def test_verify_unreachable():
    r = tv.verify(SECRET, "tok", closed_port_url(), timeout=1.0)
    assert (r.success, r.reason) == (False, "unreachable")


def test_verify_unresolvable_host():
    r = tv.verify(SECRET, "tok", "http://tempo.invalid", timeout=2.0)
    assert (r.success, r.reason) == (False, "unreachable")


def test_verify_empty_token_makes_no_request(stub):
    r = tv.verify(SECRET, "", stub.url)
    assert (r.success, r.reason) == (False, "no_token")
    assert stub.requests == []


@pytest.mark.parametrize("url", ["", "ftp://x", "file:///etc/passwd", "tempo.example.com"])
def test_bad_base_url_is_refused_without_a_request(url):
    assert tv.verify(SECRET, "tok", url).reason == "bad_config"
    assert tv.issue_ticket(SECRET, "u_1", url).reason == "bad_config"


def test_redirect_is_not_followed(stub):
    other = Stub()
    try:
        other.json(200, OK_BODY)
        stub.reply = (307, b"", {"Location": other.url + "/v1/siteverify"})
        r = tv.verify(SECRET, "tok", stub.url)
        assert (r.success, r.reason) == (False, "bad_response")
        assert other.requests == []  # the secret never reached the redirect target
    finally:
        other.close()


def test_secret_is_never_logged_or_in_results(stub, caplog):
    caplog.set_level(logging.DEBUG)
    for reply in [(401, b'{"error": {"code": "unauthorized"}}', {}), (200, b"nope", {})]:
        stub.reply = reply
        r = tv.verify(SECRET, "tok", stub.url)
        assert SECRET not in repr(r)
    tv.verify(SECRET, "tok", closed_port_url(), timeout=1.0)
    assert SECRET not in caplog.text


def test_issue_ticket(stub):
    stub.json(200, {"ticket": "t123", "expires_in": 600})
    r = tv.issue_ticket(SECRET, "u_1", stub.url)
    assert r == tv.TicketResult(True, "t123", 600, None)
    assert stub.requests[0]["path"] == "/v1/tickets"
    assert stub.requests[0]["body"] == {"secret": SECRET, "user_id": "u_1", "reset": False}
    assert "Origin" not in stub.requests[0]["headers"]
    tv.issue_ticket(SECRET, "u_1", stub.url, reset=True)
    assert stub.requests[1]["body"]["reset"] is True


def test_issue_ticket_failures(stub):
    stub.json(409, {"error": {"code": "already_enrolled", "message": "x"}})
    assert tv.issue_ticket(SECRET, "u_1", stub.url).reason == "already_enrolled"
    stub.json(401, {"error": {"code": "unauthorized", "message": "x"}})
    assert tv.issue_ticket(SECRET, "u_1", stub.url).reason == "unauthorized"
    stub.json(422, {"error": {"code": "validation_error", "message": "x"}})
    assert tv.issue_ticket(SECRET, "u@x", stub.url).reason == "bad_response"
    stub.reply = (200, b"not json", {})
    assert tv.issue_ticket(SECRET, "u_1", stub.url).reason == "bad_response"
    stub.json(200, {"ticket": 5})
    assert tv.issue_ticket(SECRET, "u_1", stub.url).reason == "bad_response"
    stub.json(200, {"ticket": "t", "expires_in": 600})
    stub.delay = 1.5
    assert tv.issue_ticket(SECRET, "u_1", stub.url, timeout=0.2).reason == "unreachable"
    assert tv.issue_ticket(SECRET, "u_1", closed_port_url(), timeout=1.0).reason == "unreachable"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_helper():
    result = subprocess.run(
        ["node", "--test", "integrations/node/tempo-verify.test.mjs"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr

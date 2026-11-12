"""tempo.js, the embeddable script served at /v1/tempo.js: sync with capture.js and the serving route.

The browser behaviour is tested in tests/web/tempo.test.mjs (run by test_web_capture.py).
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tempo.api import Settings, create_app  # noqa: E402
from tempo.auth import EngineConfig, TempoEngine, TemplateStore  # noqa: E402

from test_api import fake_embedder  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "src" / "tempo" / "web" / "static"
SYNC = ROOT / "scripts" / "sync_tempo.py"
KEY = "k" * 48


def make_client(**settings) -> TestClient:
    engine = TempoEngine(fake_embedder, TemplateStore(), 1.0, EngineConfig())
    return TestClient(create_app(engine, Settings(**settings)))


def load_sync():
    spec = importlib.util.spec_from_file_location("sync_tempo", SYNC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tempo_js_is_in_sync_with_capture_js():
    # tempo.js embeds capture.js's logic between markers; edit capture.js, then run scripts/sync_tempo.py
    assert load_sync().main(["--check"]) == 0


def test_sync_detects_drift_and_is_idempotent():
    sync = load_sync()
    current = (STATIC / "tempo.js").read_text(encoding="utf-8")
    assert sync.synced_text(current) == current
    drifted = current.replace("export", "exp0rt").replace("class Recorder", "class Recorder2")
    assert drifted != current
    assert sync.synced_text(drifted) == current  # regenerating repairs hand edits inside the markers
    assert "export " not in sync.generated_block()  # a classic script cannot contain ES module syntax


def test_route_serves_the_script_when_v1_is_enabled():
    client = make_client(enable_v1=True, token_key=KEY)
    r = client.get("/v1/tempo.js")
    assert r.status_code == 200
    assert r.headers["content-type"].split(";")[0] == "application/javascript"
    assert r.headers["cache-control"] == "public, max-age=300"
    assert r.headers["access-control-allow-origin"] == "*"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.content == (STATIC / "tempo.js").read_bytes()
    assert b"tempo" in r.content


def test_route_is_absent_when_v1_is_disabled():
    client = make_client()
    assert client.get("/v1/tempo.js").status_code == 404


def test_wildcard_cors_is_only_on_the_script():
    client = make_client(enable_v1=True, token_key=KEY)
    other = client.post("/v1/verify", json={"sitekey": "nope", "user_id": "u", "timings": [[0.1] * 31]}, headers={"Origin": "https://x.example"})
    assert other.status_code == 403
    assert other.headers.get("access-control-allow-origin") != "*"
    assert client.options("/v1/tempo.js").status_code in (404, 405)  # no preflight handling; it is a plain script load


def test_script_is_packaged_like_the_other_static_files():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"tempo.web" = ["static/*"]' in pyproject
    assert (STATIC / "tempo.js").is_file()

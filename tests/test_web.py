import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from keyauth.api import Settings, create_app  # noqa: E402
from keyauth.auth import EngineConfig, KeyAuthEngine, TemplateStore  # noqa: E402


def make_client(**settings) -> TestClient:
    engine = KeyAuthEngine(lambda t: t[:, 0], TemplateStore(), 0.5, EngineConfig())
    return TestClient(create_app(engine, Settings(**settings)))


def test_index_is_html_with_security_headers():
    r = make_client().get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert "unsafe-inline" not in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"
    assert "static/app.js" in r.text


def test_static_assets_content_types():
    c = make_client()
    assert c.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert c.get("/static/capture.js").headers["content-type"].startswith("text/javascript")
    assert c.get("/static/style.css").headers["content-type"].startswith("text/css")


def test_page_has_no_inline_script_or_style():
    html = make_client().get("/").text
    assert "<style" not in html
    assert " style=" not in html
    assert "<script>" not in html
    assert " onclick=" not in html


@pytest.mark.parametrize("path", ["/static/missing.js", "/static/..%2Fapp.py", "/static/%2e%2e%2fsettings.py", "/static/__init__.py", "/static/", "/nope", "/static/index.html/x"])
def test_unknown_paths_are_json_404(path):
    r = make_client().get(path)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_ui_disabled_gives_json_404():
    c = make_client(serve_ui=False)
    for path in ["/", "/static/app.js"]:
        r = c.get(path)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"
    assert c.get("/health").json()["status"] == "ok"


def test_api_routes_unaffected():
    c = make_client()
    assert c.get("/health").json()["model_loaded"] is True
    assert c.get("/status", params={"user_id": "nobody"}).json()["error"]["code"] == "not_enrolled"
    assert c.post("/enroll", json={}).status_code == 422
    assert c.post("/").status_code == 405
    assert c.post("/").json()["error"]["code"] == "method_not_allowed"


def test_serve_ui_env_setting():
    assert Settings.from_env({}).serve_ui is True
    assert Settings.from_env({"KEYAUTH_SERVE_UI": "0"}).serve_ui is False


def test_favicon_is_empty_204():
    assert make_client().get("/favicon.ico").status_code == 204

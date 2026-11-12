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


# ---- landing page (KEYAUTH_LANDING) ----------------------------------------


def test_landing_env_setting():
    assert Settings.from_env({}).landing is False
    assert Settings.from_env({"KEYAUTH_LANDING": "1"}).landing is True
    assert Settings.from_env({"KEYAUTH_LANDING": "0"}).landing is False


def test_landing_off_by_default_demo_stays_at_root():
    c = make_client()
    assert "static/app.js" in c.get("/").text
    assert "static/landing.js" not in c.get("/").text
    assert c.get("/demo").status_code == 404
    assert c.get("/demo").json()["error"]["code"] == "not_found"


def test_landing_on_serves_landing_at_root_and_demo_at_demo():
    c = make_client(landing=True)
    root = c.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert "static/landing.js" in root.text
    assert "static/app.js" not in root.text
    demo = c.get("/demo")
    assert demo.status_code == 200
    assert "static/app.js" in demo.text
    assert c.get("/static/landing.js").headers["content-type"].startswith("text/javascript")
    assert c.get("/static/landing.css").headers["content-type"].startswith("text/css")
    assert c.get("/static/app.js").status_code == 200  # assets of the demo page still served


def test_landing_security_headers_unchanged():
    c = make_client(landing=True)
    for path in ["/", "/demo"]:
        r = c.get(path)
        assert r.headers["content-security-policy"] == "default-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"


def test_landing_has_no_inline_script_style_or_handlers():
    import re

    html = make_client(landing=True).get("/").text
    assert "<style" not in html
    assert " style=" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert not re.search(r"\son[a-z]+\s*=", html)
    assert "javascript:" not in html


def test_landing_has_no_external_urls():
    import re

    c = make_client(landing=True)
    page = c.get("/").text
    # the embed snippet is escaped text inside <pre> (a placeholder host, never fetched); everything else is real markup
    html = re.sub(r"<pre.*?</pre>", "", page, flags=re.S)
    assert "<script src=\"https" not in page
    # references that the browser would fetch or navigate to
    for match in re.finditer(r"""(?:src|href|action)\s*=\s*["']([^"']*)["']""", html):
        assert not re.match(r"(?:[a-z]+:)?//", match.group(1)), match.group(1)
    assert "@import" not in c.get("/static/landing.css").text
    assert "url(" not in c.get("/static/landing.css").text
    js = c.get("/static/landing.js").text
    assert "http://" not in js and "https://" not in js


def test_landing_page_content():
    html = make_client(landing=True).get("/").text
    for needle in ["Log in by the rhythm of your typing", "How it works", "Honest numbers", "21.50", "22.14", "Use it on your site", "siteverify", "Delete my data", "never leave the browser", "aria-live"]:
        assert needle in html, needle


def test_landing_ignored_when_ui_disabled():
    c = make_client(landing=True, serve_ui=False)
    for path in ["/", "/demo"]:
        assert c.get(path).status_code == 404

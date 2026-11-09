"""scripts/get_cmu.py, with the network mocked. Nothing here touches data/."""

import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

from keyauth.data.cmu import cmu_timing_columns

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "get_cmu.py"


@pytest.fixture
def get_cmu(monkeypatch):
    spec = importlib.util.spec_from_file_location("get_cmu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "EXPECTED_ROWS", 4)
    monkeypatch.setattr(module, "EXPECTED_USERS", 2)
    return module


@pytest.fixture
def out(tmp_path) -> Path:
    return tmp_path / "nested" / "DSL-StrongPasswordData.csv"


def valid_csv(rows: int = 4) -> bytes:
    columns = cmu_timing_columns()
    lines = [",".join(["subject", "sessionIndex", "rep", *columns])]
    for i in range(rows):
        lines.append(",".join([f"s{i % 2}", "1", str(i), *["0.1"] * len(columns)]))
    return ("\n".join(lines) + "\n").encode()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BrokenResponse(FakeResponse):
    """Delivers some bytes, then the connection dies."""

    def __init__(self, body: bytes):
        super().__init__(body)
        self.calls = 0

    def read(self, size: int = -1) -> bytes:
        self.calls += 1
        if self.calls > 1:
            raise TimeoutError("connection reset mid-download")
        return super().read(size)


def serve(monkeypatch, module, result) -> list[str]:
    """Make urlopen return `result`, or raise it if it is an exception. Returns the requested URLs."""
    urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        urls.append(request.full_url)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    return urls


def leftovers(out: Path) -> list[Path]:
    if not out.parent.exists():
        return []
    return list(out.parent.iterdir())


def test_success_writes_validated_file(get_cmu, monkeypatch, out):
    body = valid_csv()
    urls = serve(monkeypatch, get_cmu, FakeResponse(body))
    assert get_cmu.main(["--out", str(out)]) == 0
    assert out.read_bytes() == body
    assert leftovers(out) == [out]
    assert urls == [get_cmu.CMU_URL]


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("name resolution failed"),
        urllib.error.HTTPError("http://x", 404, "Not Found", {}, io.BytesIO()),
        TimeoutError("timed out"),
    ],
)
def test_unreachable_fails_clearly(get_cmu, monkeypatch, out, capsys, failure):
    serve(monkeypatch, get_cmu, failure)
    assert get_cmu.main(["--out", str(out)]) != 0
    err = capsys.readouterr().err
    assert "error:" in err and get_cmu.PAGE_URL in err and "--data synthetic" in err
    assert leftovers(out) == []


def test_connection_dropped_mid_download_leaves_nothing(get_cmu, monkeypatch, out):
    serve(monkeypatch, get_cmu, BrokenResponse(valid_csv()))
    assert get_cmu.main(["--out", str(out)]) != 0
    assert leftovers(out) == []


@pytest.mark.parametrize(
    "body",
    [
        b"<html><body>Moved</body></html>",
        b"",
        valid_csv(rows=3),  # parses, but is incomplete
    ],
)
def test_wrong_content_is_rejected(get_cmu, monkeypatch, out, body):
    serve(monkeypatch, get_cmu, FakeResponse(body))
    assert get_cmu.main(["--out", str(out)]) != 0
    assert leftovers(out) == []


def test_oversized_response_is_rejected(get_cmu, monkeypatch, out):
    monkeypatch.setattr(get_cmu, "MAX_BYTES", 10)
    serve(monkeypatch, get_cmu, FakeResponse(valid_csv()))
    assert get_cmu.main(["--out", str(out)]) != 0
    assert leftovers(out) == []


def test_existing_file_is_kept_without_force(get_cmu, monkeypatch, out, capsys):
    out.parent.mkdir(parents=True)
    out.write_bytes(b"keep me")
    urls = serve(monkeypatch, get_cmu, FakeResponse(valid_csv()))
    assert get_cmu.main(["--out", str(out)]) == 0
    assert out.read_bytes() == b"keep me"
    assert urls == []
    assert "already exists" in capsys.readouterr().out


def test_failed_force_keeps_the_old_file(get_cmu, monkeypatch, out):
    out.parent.mkdir(parents=True)
    out.write_bytes(b"keep me")
    serve(monkeypatch, get_cmu, urllib.error.URLError("offline"))
    assert get_cmu.main(["--out", str(out), "--force"]) != 0
    assert out.read_bytes() == b"keep me"
    assert leftovers(out) == [out]


def test_force_replaces_on_success(get_cmu, monkeypatch, out):
    out.parent.mkdir(parents=True)
    out.write_bytes(b"old")
    body = valid_csv()
    serve(monkeypatch, get_cmu, FakeResponse(body))
    assert get_cmu.main(["--out", str(out), "--force"]) == 0
    assert out.read_bytes() == body

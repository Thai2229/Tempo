"""scripts/get_aalto.py against a small zip served by a local HTTP server. No internet, nothing in data/."""

import importlib.util
import io
import re
import socket
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "get_aalto.py"
HEADER = "PARTICIPANT_ID\tTEST_SECTION_ID\tSENTENCE\tUSER_INPUT\tKEYSTROKE_ID\tPRESS_TIME\tRELEASE_TIME\tLETTER\tKEYCODE"
README = "The 136M KEYSTROKES Dataset\n\nLICENSE AND ATTRIBUTION\n------\n\nNon-commercial use only (test copy).\n\n------\nCONTENT\n------\nfiles\n"
N_PARTICIPANTS = 6


def participant(pid: int) -> bytes:
    rows = [HEADER]
    for i in range(30):
        rows.append(f"{pid}\t1\tab\tab\t{i}\t{1000 + 100 * i}\t{1080 + 100 * i}\ta\t65")
    return ("\n".join(rows) + "\n").encode()


def make_zip(bad_member: int | None = None, metadata: bytes = b"PARTICIPANT_ID\tAGE\n1\t30\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Keystrokes/", b"")
        zf.writestr("Keystrokes/files/", b"")
        for pid in range(1, N_PARTICIPANTS + 1):
            body = participant(pid)
            if pid == bad_member:
                body = b"not\ta\ttable\nx\ty\tz\n"
            zf.writestr(f"Keystrokes/files/{pid}_keystrokes.txt", body, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("Keystrokes/files/metadata_participants.txt", metadata, compress_type=zipfile.ZIP_STORED)
        zf.writestr("Keystrokes/files/readme.txt", README, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


class Server:
    """Serves self.body at /Keystrokes.zip. mode: "range" (honours Range), "no-range" (always 200), "short" (cuts bodies)."""

    def __init__(self):
        self.body = make_zip()
        self.mode = "range"
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                outer.requests.append(self.headers.get("Range", ""))
                body = outer.body
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
                if outer.mode == "no-range" or match is None:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                start, end = int(match.group(1)), min(int(match.group(2)), len(body) - 1)
                chunk = body[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                if outer.mode == "short" and len(chunk) > 100:
                    chunk = chunk[: len(chunk) // 2]
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/Keystrokes.zip"
        self.thread = threading.Thread(target=self.httpd.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    s = Server()
    yield s
    s.close()


@pytest.fixture
def get_aalto(monkeypatch):
    spec = importlib.util.spec_from_file_location("get_aalto", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RETRY_WAIT_S", 0.0)
    return module


@pytest.fixture
def out(tmp_path) -> Path:
    return tmp_path / "data" / "aalto"


def files_under(out: Path) -> list[str]:
    if not out.exists():
        return []
    return sorted(str(p.relative_to(out)) for p in out.rglob("*"))


def participant_files(out: Path) -> list[str]:
    return sorted(p.name for p in (out / "files").glob("*_keystrokes.txt"))


def run(get_aalto, server, out, *extra: str) -> int:
    return get_aalto.main(["--out", str(out), "--url", server.url, *extra])


def test_fetches_subset_validated_and_prints_license(get_aalto, server, out, capsys):
    assert run(get_aalto, server, out, "--n-participants", "3") == 0
    assert files_under(out) == sorted(["files", "readme.txt", "metadata_participants.txt", *[f"files/{n}" for n in participant_files(out)]])
    assert len(participant_files(out)) == 3
    for name in participant_files(out):
        pid = int(name.split("_")[0])
        assert (out / "files" / name).read_bytes() == participant(pid)
    assert (out / "readme.txt").read_text() == README
    printed = capsys.readouterr().out
    assert "Non-commercial use only (test copy)." in printed and "CONTENT" not in printed
    assert all(r.startswith("bytes=") for r in server.requests)


def test_selection_is_deterministic_and_nested(get_aalto, server, out, tmp_path):
    assert run(get_aalto, server, out, "--n-participants", "2") == 0
    small = participant_files(out)
    other = tmp_path / "other"
    assert run(get_aalto, server, other, "--n-participants", "2") == 0
    assert participant_files(other) == small
    assert run(get_aalto, server, out, "--n-participants", "4") == 0
    assert set(small) < set(participant_files(out)) and len(participant_files(out)) == 4


def test_resume_skips_present_files_and_force_refetches(get_aalto, server, out, capsys):
    assert run(get_aalto, server, out, "--n-participants", "3") == 0
    capsys.readouterr()
    assert run(get_aalto, server, out, "--n-participants", "3") == 0
    assert "fetched 0 files, 5 already present" in capsys.readouterr().out
    victim = out / "files" / participant_files(out)[0]
    victim.write_bytes(b"stale")
    assert run(get_aalto, server, out, "--n-participants", "3", "--force") == 0
    assert "fetched 5 files" in capsys.readouterr().out
    assert victim.read_bytes() != b"stale"


def test_server_without_range_support_fails_clearly(get_aalto, server, out, capsys):
    server.mode = "no-range"
    assert run(get_aalto, server, out) == 1
    err = capsys.readouterr().err
    assert "does not support HTTP range requests" in err and "refusing to download the whole archive" in err
    assert len(server.requests) == 1  # gave up on the first answer, no retries, no fallback
    assert files_under(out) == []


def test_short_range_responses_fail(get_aalto, server, out):
    server.mode = "short"
    assert run(get_aalto, server, out) == 1
    assert files_under(out) == []


def test_corrupt_member_fails_without_partial_files(get_aalto, server, out, capsys):
    body = bytearray(server.body)
    with zipfile.ZipFile(io.BytesIO(bytes(body))) as zf:
        info = zf.getinfo("Keystrokes/files/3_keystrokes.txt")
    data_start = info.header_offset + 30 + len(info.filename) + len(info.extra)
    for i in range(data_start + 2, data_start + 12):
        body[i] ^= 0xFF  # damage the compressed bytes; the central directory still claims the old CRC
    server.body = bytes(body)
    assert run(get_aalto, server, out, "--n-participants", str(N_PARTICIPANTS), "--workers", "1") == 1
    assert "3_keystrokes.txt" in capsys.readouterr().err
    assert "3_keystrokes.txt" not in participant_files(out)
    assert not [p for p in out.rglob("*") if p.name.startswith(".part")]
    for name in participant_files(out):  # whatever was completed is whole and valid
        assert (out / "files" / name).read_bytes() == participant(int(name.split("_")[0]))


def test_member_with_wrong_columns_is_rejected(get_aalto, server, out, capsys):
    server.body = make_zip(bad_member=2)
    assert run(get_aalto, server, out, "--n-participants", str(N_PARTICIPANTS)) == 1
    assert "missing columns" in capsys.readouterr().err
    assert "2_keystrokes.txt" not in participant_files(out)


def test_empty_metadata_is_rejected(get_aalto, server, out):
    server.body = make_zip(metadata=b"")
    assert run(get_aalto, server, out) == 1
    assert not (out / "metadata_participants.txt").exists()


def test_not_a_zip_fails(get_aalto, server, out, capsys):
    server.body = b"<html>this is not a zip</html>" * 10
    assert run(get_aalto, server, out) == 1
    assert "not a readable zip" in capsys.readouterr().err
    assert files_under(out) == []


def test_too_many_participants_requested_fails(get_aalto, server, out, capsys):
    assert run(get_aalto, server, out, "--n-participants", str(N_PARTICIPANTS + 1)) == 1
    assert "the archive has 6" in capsys.readouterr().err
    assert files_under(out) == []


def test_unreachable_host_fails_clearly(get_aalto, out, capsys, monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert get_aalto.main(["--out", str(out), "--url", f"http://127.0.0.1:{port}/Keystrokes.zip"]) == 1
    err = capsys.readouterr().err
    assert "could not fetch" in err and get_aalto.PAGE_URL in err
    assert files_under(out) == []


def test_license_section_extraction(get_aalto):
    assert get_aalto.license_section(README).startswith("LICENSE AND ATTRIBUTION")
    assert get_aalto.license_section("no heading here") == "no heading here"

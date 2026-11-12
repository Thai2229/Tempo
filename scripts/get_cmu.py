"""Download the CMU Keystroke Dynamics Benchmark into data/.

    python scripts/get_cmu.py
    python scripts/get_cmu.py --out some/other/path.csv
    python scripts/get_cmu.py --force      # replace an existing file

Exits 0 if the file is present and valid afterwards, 1 otherwise. The download
goes to a temporary file next to the target and is only moved into place once
it parses as the benchmark, so a failure never leaves a partial file behind
and never damages a file that was already there.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from tempo.data import load_cmu

PAGE_URL = "https://www.cs.cmu.edu/~keystroke/"
CMU_URL = "https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "DSL-StrongPasswordData.csv"

EXPECTED_ROWS = 20400  # 51 subjects x 8 sessions x 50 repetitions
EXPECTED_USERS = 51
TIMEOUT_S = 60.0
CHUNK_BYTES = 1 << 16
MAX_BYTES = 50 * 1024 * 1024  # the real file is about 4.7 MB


class DownloadError(RuntimeError):
    pass


def _fetch(url: str, destination: Path) -> None:
    """Stream `url` into `destination`, or raise DownloadError."""
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "tempo-get-cmu"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, destination.open("wb") as f:
            total = 0
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    raise DownloadError(f"response is larger than {MAX_BYTES} bytes, which is not the benchmark file")
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DownloadError(f"could not fetch {url}: {e}") from e


def _validate(path: Path) -> None:
    """Raise DownloadError unless `path` is the complete benchmark."""
    try:
        data = load_cmu(path)
    except Exception as e:  # pandas raises many types on non-CSV input (HTML error pages, truncation)
        raise DownloadError(f"the downloaded file is not the CMU benchmark: {e}") from e
    if len(data) != EXPECTED_ROWS or len(data.users) != EXPECTED_USERS:
        raise DownloadError(
            f"the downloaded file is incomplete: {len(data)} rows and {len(data.users)} subjects, "
            f"expected {EXPECTED_ROWS} and {EXPECTED_USERS}"
        )


def download(out: Path, url: str = CMU_URL, force: bool = False) -> bool:
    """Returns True if a file was downloaded, False if a file was already there."""
    if out.exists() and not force:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.name}.", suffix=".part")
    os.close(handle)
    part = Path(name)
    try:
        _fetch(url, part)
        _validate(part)
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download DSL-StrongPasswordData.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--url", default=CMU_URL)
    ap.add_argument("--force", action="store_true", help="replace an existing file")
    args = ap.parse_args(argv)

    try:
        fetched = download(args.out, args.url, args.force)
    except (DownloadError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        print(
            f"No file was written to {args.out}. To proceed, download DSL-StrongPasswordData.csv by hand from "
            f"{PAGE_URL} and save it there, or train with --data synthetic and say so in the PR "
            "(synthetic numbers are not CMU numbers).",
            file=sys.stderr,
        )
        return 1
    if fetched:
        print(f"downloaded {args.out}")
    else:
        print(f"{args.out} already exists; use --force to replace it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

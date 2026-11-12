"""Download a subset of the Aalto "136M Keystrokes" dataset into data/aalto/.

    python scripts/get_aalto.py                        # 10000 participants, seed 0
    python scripts/get_aalto.py --n-participants 50
    python scripts/get_aalto.py --force                # re-download files that are already there

Dhakal, Feit, Kristensson, Oulasvirta. Observations on Typing from 136 Million
Keystrokes. CHI 2018. https://doi.org/10.1145/3173574.3174220
License (readme.txt): free for non-commercial use in research or projects, with
attribution to the authors. The license section is printed on every run.

The archive is 1.5 GB zipped (16 GB unzipped), so it is never downloaded whole.
HTTP range requests read the zip's central directory (about 20 MB), then fetch
only readme.txt, metadata_participants.txt and the chosen participants'
<id>_keystrokes.txt members. A server that ignores Range is an error, never a
silent full download.

Participants are chosen deterministically: member names sorted, shuffled with
--seed, first N taken. So a smaller N is a subset of a larger N with the same
seed, and re-running with a larger N only fetches the new files.

Every file is fetched into a temporary directory, checked (zip CRC and size,
expected columns, at least one data row) and only then renamed into place, so
data/aalto/ never holds a partial file. Files already present are skipped
(resume after a failure), unless --force. Exits 0 when all requested files are
present and valid afterwards, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAGE_URL = "https://userinterfaces.aalto.fi/136Mkeystrokes/"
AALTO_URL = "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "aalto"

README_NAME = "readme.txt"
METADATA_NAME = "metadata_participants.txt"
PARTICIPANT_SUFFIX = "_keystrokes.txt"
PARTICIPANT_DIR = "files"  # participant files go to <out>/files/, readme and metadata to <out>/

KEYSTROKE_COLUMNS = (
    "PARTICIPANT_ID",
    "TEST_SECTION_ID",
    "SENTENCE",
    "USER_INPUT",
    "KEYSTROKE_ID",
    "PRESS_TIME",
    "RELEASE_TIME",
    "LETTER",
    "KEYCODE",
)
METADATA_COLUMNS = ("PARTICIPANT_ID",)

DEFAULT_N = 10000
TIMEOUT_S = 60.0
RETRIES = 3  # attempts per range request (network errors only; a wrong response is not retried)
RETRY_WAIT_S = 2.0
WORKERS = 4
MAX_REQUEST_BYTES = 64 * 1024 * 1024  # largest single range we will ask for (central directory is ~20 MB)
MAX_MEMBER_BYTES = 64 * 1024 * 1024  # largest decompressed member we accept (metadata is ~21 MB)
LOCAL_HEADER_SLACK = 1024  # extra bytes fetched for the local header's name + extra field
LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"


class DownloadError(RuntimeError):
    pass


def _open(request: urllib.request.Request, timeout: float):
    """Indirection so tests can replace the network."""
    return urllib.request.urlopen(request, timeout=timeout)


class RangeReader:
    """Byte ranges of one remote file. Raises DownloadError if the server does not honour Range."""

    def __init__(self, url: str):
        self.url = url
        self.requests = 0
        self.bytes = 0
        self.size = self._probe_size()

    def _get(self, start: int, length: int) -> tuple[bytes, str]:
        end = start + length - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "tempo-get-aalto"}
        )
        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                with _open(request, TIMEOUT_S) as response:
                    status = getattr(response, "status", None)
                    content_range = response.headers.get("Content-Range", "")
                    if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                        # never read the body: a 200 here would be the whole 1.5 GB archive
                        raise DownloadError(
                            f"{self.url} does not support HTTP range requests (status {status}, "
                            f"Content-Range {content_range!r}); refusing to download the whole archive"
                        )
                    body = response.read(length + 1)
                self.requests += 1
                self.bytes += len(body)
                if len(body) != length:
                    raise DownloadError(f"short read from {self.url}: asked for {length} bytes at {start}, got {len(body)}")
                return body, content_range
            except DownloadError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e
                if attempt + 1 < RETRIES:
                    time.sleep(RETRY_WAIT_S * (attempt + 1))
        raise DownloadError(f"could not fetch {self.url}: {last_error}") from last_error

    def _probe_size(self) -> int:
        _, content_range = self._get(0, 1)
        total = content_range.rsplit("/", 1)[-1]
        if not total.isdigit():
            raise DownloadError(f"{self.url} did not report its size (Content-Range {content_range!r})")
        return int(total)

    def read(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if start < 0 or start + length > self.size or length > MAX_REQUEST_BYTES:
            raise DownloadError(f"bad range {start}+{length} for a {self.size}-byte file; the zip is probably corrupt")
        return self._get(start, length)[0]


class RangeFile:
    """Minimal seekable read-only file over a RangeReader, enough for zipfile to parse the central directory."""

    def __init__(self, reader: RangeReader):
        self.reader = reader
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.reader.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.reader.size - self.pos
        n = min(n, self.reader.size - self.pos)
        data = self.reader.read(self.pos, n)
        self.pos += len(data)
        return data


def list_members(reader: RangeReader) -> list[zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(RangeFile(reader)) as zf:
            return zf.infolist()
    except (zipfile.BadZipFile, ValueError, EOFError) as e:
        raise DownloadError(f"{reader.url} is not a readable zip archive: {e}") from e


def fetch_member(reader: RangeReader, info: zipfile.ZipInfo) -> bytes:
    """One member's decompressed bytes, checked against the central directory's size and CRC."""
    if info.flag_bits & 0x1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) or info.file_size > MAX_MEMBER_BYTES:
        raise DownloadError(f"{info.filename}: unsupported member (encrypted, method {info.compress_type}, or too large)")
    want = LOCAL_HEADER.size + len(info.filename.encode("utf-8")) + LOCAL_HEADER_SLACK + info.compress_size
    block = reader.read(info.header_offset, min(want, reader.size - info.header_offset))
    fields = LOCAL_HEADER.unpack_from(block)
    start = LOCAL_HEADER.size + fields[9] + fields[10]
    if fields[0] != LOCAL_HEADER_SIGNATURE:
        raise DownloadError(f"{info.filename}: bad local header; the zip is corrupt")
    if start + info.compress_size > len(block):
        block += reader.read(info.header_offset + len(block), start + info.compress_size - len(block))
    raw = block[start : start + info.compress_size]
    try:
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(raw, MAX_MEMBER_BYTES + 1)
        else:
            data = raw
    except zlib.error as e:
        raise DownloadError(f"{info.filename}: corrupt compressed data: {e}") from e
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise DownloadError(f"{info.filename}: size or CRC mismatch; the member is truncated or corrupt")
    return data


def validate_table(name: str, data: bytes, columns: tuple[str, ...]) -> None:
    """Raise DownloadError unless `data` is a tab-separated table with `columns` and at least one data row."""
    lines = data.decode("utf-8", errors="replace").splitlines()
    missing = []
    if lines:
        header = lines[0].split("\t")
        missing = [c for c in columns if c not in header]
    if len(lines) < 2 or missing:
        raise DownloadError(f"{name}: not the expected table (rows={max(len(lines) - 1, 0)}, missing columns {missing})")


def license_section(readme: str) -> str:
    """The LICENSE AND ATTRIBUTION section of readme.txt (the whole readme if the heading is not found)."""
    lines = readme.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "LICENSE" in line.upper():
            start = i
            break
    if start is None:
        return readme
    end = len(lines)
    for i in range(start + 2, len(lines)):
        if lines[i].strip().upper() == "CONTENT":
            end = i - 1
            break
    return "\n".join(lines[start:end]).strip()


def choose_participants(members: list[zipfile.ZipInfo], n: int, seed: int) -> list[zipfile.ZipInfo]:
    names = sorted(m.filename for m in members if m.filename.endswith(PARTICIPANT_SUFFIX))
    random.Random(seed).shuffle(names)
    chosen = set(names[:n])
    return [m for m in members if m.filename in chosen]


def target_path(out: Path, info: zipfile.ZipInfo) -> Path:
    name = info.filename.rsplit("/", 1)[-1]
    if name.endswith(PARTICIPANT_SUFFIX):
        return out / PARTICIPANT_DIR / name
    return out / name


def download(
    out: Path,
    n_participants: int = DEFAULT_N,
    seed: int = 0,
    url: str = AALTO_URL,
    force: bool = False,
    workers: int = WORKERS,
) -> dict:
    """Fetch readme, metadata and n participants into `out`. Returns run statistics."""
    if n_participants < 1 or workers < 1:
        raise DownloadError("--n-participants and --workers must be at least 1")
    started = time.monotonic()
    reader = RangeReader(url)
    members = list_members(reader)
    by_name = {m.filename.rsplit("/", 1)[-1]: m for m in members}
    if README_NAME not in by_name or METADATA_NAME not in by_name:
        raise DownloadError(f"the archive has no {README_NAME} / {METADATA_NAME}; it is not the 136M Keystrokes dataset")
    chosen = choose_participants(members, n_participants, seed)
    if len(chosen) < n_participants:
        raise DownloadError(f"asked for {n_participants} participants, the archive has {len(chosen)}")

    wanted = [(by_name[README_NAME], ()), (by_name[METADATA_NAME], METADATA_COLUMNS)]
    wanted += [(m, KEYSTROKE_COLUMNS) for m in sorted(chosen, key=lambda m: m.header_offset)]
    todo = []
    for info, columns in wanted:
        if force or not target_path(out, info).exists():
            todo.append((info, columns))

    (out / PARTICIPANT_DIR).mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out, prefix=".part-"))
    written = 0

    def fetch_one(item: tuple[zipfile.ZipInfo, tuple[str, ...]]) -> Path:
        info, columns = item
        data = fetch_member(reader, info)
        if columns:
            validate_table(info.filename, data, columns)
        elif not data.strip():
            raise DownloadError(f"{info.filename} is empty")
        part = tmp / target_path(out, info).name
        part.write_bytes(data)
        return part

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                for (info, _), part in zip(todo, pool.map(fetch_one, todo)):
                    os.replace(part, target_path(out, info))
                    written += 1
                    if written % 500 == 0:
                        print(f"  {written}/{len(todo)} files", file=sys.stderr)
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        "archive_bytes": reader.size,
        "members": len(members),
        "requested": len(wanted),
        "fetched": written,
        "skipped": len(wanted) - len(todo),
        "requests": reader.requests,
        "bytes_downloaded": reader.bytes,
        "seconds": time.monotonic() - started,
    }


def disk_bytes(out: Path) -> int:
    total = 0
    for p in out.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download a subset of the Aalto 136M Keystrokes dataset")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--url", default=AALTO_URL)
    ap.add_argument("--n-participants", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS, help="parallel range requests")
    ap.add_argument("--force", action="store_true", help="re-download files that are already present")
    args = ap.parse_args(argv)

    try:
        stats = download(args.out, args.n_participants, args.seed, args.url, args.force, args.workers)
    except (DownloadError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        print(
            f"No partial file was left in {args.out} (files completed before the failure were kept; re-run to "
            f"resume). The dataset page is {PAGE_URL}. If the host is unreachable, download Keystrokes.zip on "
            "another machine and copy the needed members into that folder.",
            file=sys.stderr,
        )
        return 1

    readme = (args.out / README_NAME).read_text(encoding="utf-8", errors="replace")
    print(license_section(readme))
    print()
    print(
        f"{args.out}: fetched {stats['fetched']} files, {stats['skipped']} already present; "
        f"{stats['requests']} range requests, {stats['bytes_downloaded']} bytes downloaded "
        f"in {stats['seconds']:.1f} s; {disk_bytes(args.out)} bytes on disk"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

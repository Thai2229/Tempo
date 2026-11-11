"""Loader for the Aalto "136M Keystrokes" dataset (subset fetched by scripts/get_aalto.py).

Dhakal, Feit, Kristensson, Oulasvirta. Observations on Typing from 136 Million
Keystrokes. CHI 2018. https://doi.org/10.1145/3173574.3174220
License: free for non-commercial use in research or projects, with attribution.

Layout under `root`: files/<participant>_keystrokes.txt, one per participant,
tab separated, UTF-8, one row per key press with its release already paired:

    PARTICIPANT_ID TEST_SECTION_ID SENTENCE USER_INPUT KEYSTROKE_ID PRESS_TIME RELEASE_TIME LETTER KEYCODE

PRESS_TIME / RELEASE_TIME are Unix epoch milliseconds. One TEST_SECTION_ID is
one presented sentence (15 per participant). LETTER is often empty or a name
("SHIFT", "BKSP", "CAPS_LOCK", "ARW_LEFT"), and once in a while a literal
newline that splits the row over two lines, so key identity comes from KEYCODE
(the JavaScript keyCode) and rows are re-joined until they have 9 fields. The
final Enter / Next is not logged.

Each sentence becomes one attempt of (hold, dd, ud) in seconds, through the
same code as live capture: rows -> RawEvent down/up pairs -> events.pair_events
(drops modifiers, orders by press time, rejects release-before-press) ->
features.extract_features (rejects hold > MAX_HOLD_S or dd > MAX_LATENCY_S).

Decisions (counts of every drop reason go into `report`):
- corrections="drop" (default): a sentence with any editing key (Backspace,
  Delete, arrows, Home/End, ...) is dropped, like the CMU benchmark and the web
  page, which reject corrected attempts. "keep" keeps it, with the editing keys
  as ordinary keystrokes. Uncorrected typos are kept either way: the timing is
  still real typing, and the sentence id is the presented sentence.
- Unpaired or malformed rows, timing outliers, or fewer than min_keys / more
  than max_keys keystrokes after dropping modifiers: the sentence is dropped.
- Participants with fewer than min_sentences clean sentences are dropped.
- sentence_ids are a 63-bit hash of the presented sentence; the text itself is
  not kept. key_codes (keyCode per keystroke) are only kept if with_keys=True.
- sessions are -1: all 15 sentences come from one sitting.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from keyauth.config import MAX_KEYS
from keyauth.data.container import KeystrokeData
from keyauth.events import RawEvent, pair_events
from keyauth.features import extract_features

AALTO_COLUMNS = (
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
PARTICIPANT_SUFFIX = "_keystrokes.txt"

# JavaScript keyCodes of modifiers, mapped to the names in config.MODIFIER_KEYS so pair_events drops them.
MODIFIER_KEYCODES = {16: "Shift", 17: "Control", 18: "Alt", 20: "CapsLock", 91: "Meta", 92: "Meta", 93: "Meta", 225: "AltGraph"}
# Editing / navigation keys (and "unidentified" 0 / 229): their presence means the sentence was corrected.
EDIT_KEYCODES = frozenset({0, 8, 33, 34, 35, 36, 37, 38, 39, 40, 45, 46, 144, 229})

DROP_REASONS = ("malformed", "correction", "unpaired", "length", "outlier", "few_sentences")


def sentence_id(text: str) -> int:
    """Stable 63-bit id for a sentence text (so splits can hold out sentences without storing them)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def read_participant(path: Path) -> list[tuple[str, list[list[str]] | None]]:
    """[(sentence text, rows)] per TEST_SECTION_ID in file order; rows is None if any row is malformed.

    Each row is [keystroke_id, press, release, keycode] as strings.
    Raises ValueError if the file is empty or lacks the expected columns.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    header = lines[0].rstrip("\r").split("\t")
    missing = [c for c in AALTO_COLUMNS if c not in header]
    if not lines[0].strip() or missing:
        raise ValueError(f"{path} is not an Aalto keystrokes file; missing columns {missing}")
    col = {c: header.index(c) for c in AALTO_COLUMNS}

    sections: OrderedDict[str, tuple[str, list[list[str]] | None]] = OrderedDict()
    i = 1
    while i < len(lines):
        line = lines[i].rstrip("\r")
        i += 1
        if not line:
            continue
        # a LETTER holding a newline splits one row over several lines: re-join them
        while len(line.split("\t")) < len(header) and i < len(lines) and lines[i]:
            line = line + "\n" + lines[i].rstrip("\r")
            i += 1
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        section = fields[col["TEST_SECTION_ID"]]
        text, rows = sections.get(section, ("", []))
        if len(fields) != len(header):
            sections[section] = (text, None)
            continue
        if rows is not None:
            rows.append([fields[col[c]] for c in ("KEYSTROKE_ID", "PRESS_TIME", "RELEASE_TIME", "KEYCODE")])
        sections[section] = (fields[col["SENTENCE"]], rows)
    if not sections:
        raise ValueError(f"{path} has no keystrokes")
    return list(sections.values())


def sentence_timings(
    rows: list[list[str]],
    corrections: str = "drop",
    min_keys: int = 10,
    max_keys: int = MAX_KEYS,
) -> tuple[np.ndarray | None, list[int], str]:
    """One sentence's rows -> ((n, 3) seconds, key codes, "") or (None, [], drop reason)."""
    try:
        keycodes = [int(r[3]) for r in rows]
        press = [float(r[1]) for r in rows]
        release = [float(r[2]) for r in rows]
    except ValueError:
        return None, [], "unpaired"  # missing or non-numeric press / release time
    if corrections == "drop" and any(k in EDIT_KEYCODES for k in keycodes):
        return None, [], "correction"

    t0 = min(press)
    raw = []
    for i, (k, d, u) in enumerate(zip(keycodes, press, release)):
        # key = the keyCode as text, so the paired KeyEvents carry it; code = row index, since rows are pre-paired
        key = MODIFIER_KEYCODES.get(k, str(k))
        raw.append(RawEvent(key=key, type="down", t=d - t0, code=f"row{i}"))
        raw.append(RawEvent(key=key, type="up", t=u - t0, code=f"row{i}"))
    try:
        events = pair_events(raw)
    except ValueError:
        return None, [], "unpaired"
    if len(events) < max(min_keys, 2) or len(events) > max_keys:
        return None, [], "length"
    try:
        seq = extract_features(events)
    except ValueError:
        return None, [], "outlier"
    return seq.values, [int(e.key) for e in events], ""


def load_aalto(
    root: str | Path,
    with_keys: bool = False,
    corrections: str = "drop",
    min_keys: int = 10,
    max_keys: int = MAX_KEYS,
    min_sentences: int = 5,
    max_participants: int | None = None,
    report: Counter | None = None,
) -> KeystrokeData:
    """All participants under root/files (sorted by file name; the first max_participants if given).

    `report`, if given, is filled with counts: participants_seen / _kept, sentences_seen / _kept,
    keystrokes_kept, and dropped_<reason> for each of DROP_REASONS (in sentences).
    """
    root = Path(root)
    paths = sorted((root / "files").glob("*" + PARTICIPANT_SUFFIX))
    if max_participants is not None:
        paths = paths[:max_participants]
    if corrections not in ("drop", "keep") or min_sentences < 1 or not paths:
        raise ValueError(
            f"need corrections in ('drop', 'keep'), min_sentences >= 1 and participant files in {root / 'files'}"
        )
    if report is None:
        report = Counter()

    seqs, users, sentences, codes = [], [], [], []
    for path in paths:
        participant = path.name.removesuffix(PARTICIPANT_SUFFIX)
        report["participants_seen"] += 1
        kept = []
        for text, rows in read_participant(path):
            report["sentences_seen"] += 1
            if rows is None:
                report["dropped_malformed"] += 1
                continue
            values, keys, reason = sentence_timings(rows, corrections, min_keys, max_keys)
            if values is None:
                report[f"dropped_{reason}"] += 1
                continue
            kept.append((values, sentence_id(text), keys))
        if len(kept) < min_sentences:
            report["dropped_few_sentences"] += len(kept)
            continue
        report["participants_kept"] += 1
        for values, sid, keys in kept:
            report["sentences_kept"] += 1
            report["keystrokes_kept"] += len(values)
            seqs.append(values)
            users.append(participant)
            sentences.append(sid)
            codes.append(keys)

    if not with_keys:
        codes = None
    return KeystrokeData.from_arrays(seqs, users, None, sentences, codes)


def main(argv: list[str] | None = None) -> int:
    """Print dataset statistics: python -m keyauth.data.aalto data/aalto"""
    ap = argparse.ArgumentParser(description="Statistics of a downloaded Aalto subset")
    ap.add_argument("root", type=Path)
    ap.add_argument("--corrections", choices=("drop", "keep"), default="drop")
    ap.add_argument("--min-sentences", type=int, default=5)
    ap.add_argument("--min-keys", type=int, default=10)
    ap.add_argument("--max-participants", type=int, default=None)
    args = ap.parse_args(argv)

    report: Counter = Counter()
    data = load_aalto(
        args.root,
        corrections=args.corrections,
        min_keys=args.min_keys,
        min_sentences=args.min_sentences,
        max_participants=args.max_participants,
        report=report,
    )
    per_user = np.array([int(np.sum(data.user_ids == u)) for u in data.users])
    print(f"participants: {report['participants_kept']} kept of {report['participants_seen']}")
    print(f"sentences: {report['sentences_kept']} kept of {report['sentences_seen']}")
    for reason in DROP_REASONS:
        n = report[f"dropped_{reason}"]
        print(f"  dropped {reason}: {n} ({100.0 * n / max(report['sentences_seen'], 1):.1f}%)")
    if len(data):
        print(f"sentences per participant: mean {per_user.mean():.2f}, median {np.median(per_user):.0f}, min {per_user.min()}, max {per_user.max()}")
        print(f"keystrokes per sentence: mean {data.lengths.mean():.1f}, median {np.median(data.lengths):.0f}, min {data.lengths.min()}, max {data.lengths.max()}")
        print(f"distinct sentences: {len(np.unique(data.sentence_ids))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

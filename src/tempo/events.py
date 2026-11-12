"""Raw keyboard events -> paired keystrokes.

A capture front end (browser, terminal, ...) emits a flat stream of key
"down"/"up" events. This module pairs them into KeyEvent(key, down, up)
records, one per physical keystroke, which is what feature extraction needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from tempo.config import MODIFIER_KEYS, SUBMIT_KEY


@dataclass(frozen=True)
class RawEvent:
    """One keydown or keyup as reported by the capture front end.

    key:  the produced character / key name (browser `KeyboardEvent.key`), e.g. "R", "Enter".
    type: "down" or "up".
    t:    timestamp in milliseconds.
    code: physical key id (browser `KeyboardEvent.code`), e.g. "KeyR". Used to
          pair down/up, because `key` can change between them (press "R" with
          Shift, release Shift first, the keyup reports "r").
    """

    key: str
    type: str
    t: float
    code: str = ""

    @property
    def pair_id(self) -> str:
        if self.code:
            return self.code
        return self.key.lower()

    @classmethod
    def from_dict(cls, d: Mapping) -> "RawEvent":
        return cls(key=str(d["key"]), type=str(d["type"]), t=float(d["t"]), code=str(d.get("code", "")))


@dataclass(frozen=True)
class KeyEvent:
    """A single completed keystroke. Times in milliseconds."""

    key: str
    down: float
    up: float

    @property
    def hold_ms(self) -> float:
        return self.up - self.down


def pair_events(raw: Iterable[RawEvent | Mapping]) -> list[KeyEvent]:
    """Pair down/up events into keystrokes, ordered by press time.

    - Modifier keys (Shift, Ctrl, ...) are dropped.
    - OS auto-repeat (a second "down" for a key that is still held) is ignored.
    - An "up" with no matching "down" is ignored (e.g. key held before capture began).
    - A "down" that is never released raises ValueError: the capture is incomplete.
    """
    events = []
    for e in raw:
        if not isinstance(e, RawEvent):
            e = RawEvent.from_dict(e)
        events.append(e)
    # stable sort keeps the front end's order for identical timestamps
    events.sort(key=lambda e: e.t)

    keys: list[str] = []
    downs: list[float] = []
    ups: list[float | None] = []
    open_presses: dict[str, int] = {}

    for e in events:
        if e.key in MODIFIER_KEYS:
            continue
        if e.type == "down":
            if e.pair_id in open_presses:
                continue  # auto-repeat
            open_presses[e.pair_id] = len(keys)
            keys.append(e.key)
            downs.append(e.t)
            ups.append(None)
        elif e.type == "up":
            idx = open_presses.pop(e.pair_id, None)
            if idx is None:
                continue  # orphan release
            ups[idx] = e.t
        else:
            raise ValueError(f"unknown event type {e.type!r}")

    if open_presses:
        missing = sorted(keys[i] for i in open_presses.values())
        raise ValueError(f"keys pressed but never released: {missing}")

    out = []
    for k, d, u in zip(keys, downs, ups):
        if u < d:
            raise ValueError(f"key {k!r} released before it was pressed ({u} < {d})")
        out.append(KeyEvent(key=k, down=d, up=u))
    return out


def typed_text(events: Sequence[KeyEvent]) -> str:
    """The text the keystrokes spell, ignoring the trailing submit key."""
    return "".join(e.key for e in events if e.key != SUBMIT_KEY)


def matches_phrase(events: Sequence[KeyEvent], phrase: str, require_submit: bool = True) -> bool:
    """True if the keystrokes are exactly `phrase` (no typos, no backspaces).

    Fixed-text keystroke dynamics only compares attempts of the same text, so
    any attempt containing corrections is rejected and the user retypes.
    """
    expected = list(phrase)
    if require_submit:
        expected.append(SUBMIT_KEY)
    return [e.key for e in events] == expected

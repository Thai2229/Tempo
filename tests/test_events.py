import pytest

from keyauth.events import KeyEvent, RawEvent, matches_phrase, pair_events, typed_text


def raw(key, type_, t, code=""):
    return RawEvent(key=key, type=type_, t=t, code=code)


def test_pairs_simple_sequence():
    stream = [raw("a", "down", 0), raw("a", "up", 80), raw("b", "down", 150), raw("b", "up", 240)]
    assert pair_events(stream) == [KeyEvent("a", 0, 80), KeyEvent("b", 150, 240)]


def test_rollover_overlapping_keys():
    # b pressed before a released
    stream = [raw("a", "down", 0), raw("b", "down", 50), raw("a", "up", 90), raw("b", "up", 140)]
    assert pair_events(stream) == [KeyEvent("a", 0, 90), KeyEvent("b", 50, 140)]


def test_drops_modifiers_and_pairs_by_code():
    # Shift released before R, so keyup reports lowercase "r"
    stream = [
        raw("Shift", "down", 0, "ShiftLeft"),
        raw("R", "down", 40, "KeyR"),
        raw("Shift", "up", 70, "ShiftLeft"),
        raw("r", "up", 120, "KeyR"),
    ]
    assert pair_events(stream) == [KeyEvent("R", 40, 120)]


def test_ignores_autorepeat_and_orphan_up():
    stream = [
        raw("x", "up", -10),  # released before capture began
        raw("a", "down", 0),
        raw("a", "down", 500),  # auto-repeat
        raw("a", "up", 600),
    ]
    assert pair_events(stream) == [KeyEvent("a", 0, 600)]


def test_unreleased_key_raises():
    with pytest.raises(ValueError, match="never released"):
        pair_events([raw("a", "down", 0)])


def test_accepts_dicts_and_sorts_by_time():
    stream = [
        {"key": "b", "type": "down", "t": 150},
        {"key": "a", "type": "down", "t": 0},
        {"key": "a", "type": "up", "t": 80},
        {"key": "b", "type": "up", "t": 240},
    ]
    assert [e.key for e in pair_events(stream)] == ["a", "b"]


def test_phrase_matching():
    ks = [KeyEvent(c, i * 100, i * 100 + 50) for i, c in enumerate(["h", "i", "Enter"])]
    assert typed_text(ks) == "hi"
    assert matches_phrase(ks, "hi")
    assert not matches_phrase(ks, "ho")
    assert not matches_phrase(ks[:-1], "hi")
    assert matches_phrase(ks[:-1], "hi", require_submit=False)
    with_backspace = ks[:1] + [KeyEvent("Backspace", 60, 90)] + ks[1:]
    assert not matches_phrase(with_backspace, "hi")

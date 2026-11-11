"""Aalto 136M Keystrokes loader and the sentence split.

tests/fixtures/aalto is hand-made and synthetic (made-up sentences and timings
in the real file format), not participant data.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from keyauth.data import KeystrokeData, load_aalto, split_sentences, split_users
from keyauth.data.aalto import AALTO_COLUMNS, main, read_participant, sentence_id, sentence_timings

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aalto"
HEADER = "\t".join(AALTO_COLUMNS)


def sentence_rows(pid: str, section: int, text: str, t0: int = 1_600_000_000_000, hold: int = 100, gap: int = 180) -> list[str]:
    """Rows of a clean typed sentence in the real format (lower-case letters and spaces)."""
    rows = []
    for i, ch in enumerate(text):
        code = 32
        if ch != " ":
            code = ord(ch.upper())
        d = t0 + i * gap
        rows.append(f"{pid}\t{section}\t{text}\t{text}\t{section * 1000 + i}\t{d}\t{d + hold}\t{ch}\t{code}")
    return rows


def write_participant(root: Path, pid: str, lines: list[str], header: str = HEADER) -> Path:
    path = root / "files" / f"{pid}_keystrokes.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    return path


SENTENCES = ["the cat sat on the mat", "a quick brown fox", "pack my box with jugs", "we few we happy few"]


@pytest.fixture
def root(tmp_path) -> Path:
    for p in range(6):
        lines = []
        for s, text in enumerate(SENTENCES):
            lines += sentence_rows(str(p), 100 * p + s, text, hold=80 + 10 * p)
        write_participant(tmp_path, str(p), lines)
    return tmp_path


# ---- the hand-made fixture ------------------------------------------------


def test_fixture_matches_feature_rules():
    report = Counter()
    data = load_aalto(FIXTURE, with_keys=True, min_keys=5, min_sentences=1, report=report)
    # section 1002 has a Backspace, so it is dropped; 1001 and 1003 are kept
    assert len(data) == 2
    assert report["dropped_correction"] == 1 and report["sentences_seen"] == 3
    assert data.users == ["900001"]
    assert data.sessions.tolist() == [-1, -1]

    # section 1001: Shift dropped, 9 keystrokes "Hi there."
    assert data.lengths[0] == 9
    first = data.values[0]
    np.testing.assert_allclose(first[0], [0.090, 0.150, 0.060], atol=1e-6)  # H: hold, dd, ud in seconds
    np.testing.assert_allclose(first[5, 2], -0.040, atol=1e-6)  # rollover e -> r gives negative ud
    np.testing.assert_allclose(first[8, 1:], [0.0, 0.0])  # last keystroke has no successor
    assert data.key_codes[0, :9].tolist() == [72, 73, 32, 84, 72, 69, 82, 69, 190]

    # section 1003: a newline LETTER split the last row over two lines; it is re-joined, keycode 13 kept
    assert data.lengths[1] == 9
    assert data.key_codes[1, 8] == 13
    assert data.sentence_ids.tolist() == [sentence_id("Hi there."), sentence_id("ok then.")]


def test_keep_corrections_keeps_backspace_as_a_keystroke():
    data = load_aalto(FIXTURE, with_keys=True, corrections="keep", min_keys=5, min_sentences=1)
    assert len(data) == 3
    assert 8 in data.key_codes[1].tolist()


def test_keys_are_not_kept_by_default():
    data = load_aalto(FIXTURE, min_keys=5, min_sentences=1)
    assert data.key_codes is None
    assert data.sentence_ids is not None


# ---- filters and drop reasons --------------------------------------------


def test_loads_all_clean_sentences(root):
    report = Counter()
    data = load_aalto(root, min_sentences=4, report=report)
    assert len(data.users) == 6 and len(data) == 24
    assert report["participants_kept"] == 6 and report["sentences_kept"] == 24
    assert report["keystrokes_kept"] == int(data.lengths.sum())
    assert len(np.unique(data.sentence_ids)) == len(SENTENCES)


def test_participants_with_too_few_sentences_are_dropped(root):
    report = Counter()
    data = load_aalto(root, min_sentences=5, report=report)
    assert len(data) == 0
    assert report["dropped_few_sentences"] == 24 and report["participants_kept"] == 0


def test_max_participants(root):
    data = load_aalto(root, min_sentences=1, max_participants=2)
    assert data.users == ["0", "1"]


def test_out_of_range_timings_drop_the_sentence(tmp_path):
    long_hold = sentence_rows("7", 1, "the cat sat on the mat", hold=2500)
    long_pause = sentence_rows("7", 2, "a quick brown fox", gap=6000)
    ok = sentence_rows("7", 3, "pack my box with jugs")
    write_participant(tmp_path, "7", long_hold + long_pause + ok)
    report = Counter()
    data = load_aalto(tmp_path, min_sentences=1, report=report)
    assert len(data) == 1 and report["dropped_outlier"] == 2


def test_unpaired_events_drop_the_sentence(tmp_path):
    released_early = sentence_rows("7", 1, "the cat sat on the mat")
    fields = released_early[3].split("\t")
    fields[6] = str(int(fields[5]) - 10)  # released before pressed
    released_early[3] = "\t".join(fields)
    never_released = sentence_rows("7", 2, "a quick brown fox")
    fields = never_released[2].split("\t")
    fields[6] = ""  # no release time
    never_released[2] = "\t".join(fields)
    ok = sentence_rows("7", 3, "pack my box with jugs")
    write_participant(tmp_path, "7", released_early + never_released + ok)
    report = Counter()
    data = load_aalto(tmp_path, min_sentences=1, report=report)
    assert len(data) == 1 and report["dropped_unpaired"] == 2


def test_malformed_rows_and_length_limits(tmp_path):
    broken = sentence_rows("7", 1, "the cat sat on the mat")
    broken[4] = broken[4] + "\textra"
    short = sentence_rows("7", 2, "hi there")  # 8 keys < min_keys 10
    long = sentence_rows("7", 3, "a" * 129, gap=100)  # > MAX_KEYS
    ok = sentence_rows("7", 4, "pack my box with jugs")
    write_participant(tmp_path, "7", broken + short + long + ok)
    report = Counter()
    data = load_aalto(tmp_path, min_sentences=1, report=report)
    assert len(data) == 1
    assert report["dropped_malformed"] == 1 and report["dropped_length"] == 2


def test_missing_columns_raise(tmp_path):
    header = "\t".join(c for c in AALTO_COLUMNS if c != "KEYCODE")
    write_participant(tmp_path, "7", ["7\t1\tx\tx\t1\t0\t100\tx"], header=header)
    with pytest.raises(ValueError, match="KEYCODE"):
        load_aalto(tmp_path, min_sentences=1)


def test_empty_files_raise(tmp_path):
    path = tmp_path / "files" / "7_keystrokes.txt"
    path.parent.mkdir(parents=True)
    path.write_text("")
    with pytest.raises(ValueError, match="not an Aalto keystrokes file"):
        read_participant(path)
    path.write_text(HEADER + "\n")
    with pytest.raises(ValueError, match="no keystrokes"):
        read_participant(path)


@pytest.mark.parametrize("kwargs", [{"corrections": "fix"}, {"min_sentences": 0}])
def test_bad_arguments_and_missing_folder_raise(root, tmp_path, kwargs):
    with pytest.raises(ValueError):
        load_aalto(root, **kwargs)
    with pytest.raises(ValueError):
        load_aalto(tmp_path / "nowhere")


def test_sentence_timings_reports_reason():
    rows = [["1", "0", "100", "8"], ["2", "200", "300", "65"]]
    assert sentence_timings(rows, min_keys=2)[2] == "correction"
    assert sentence_timings(rows, corrections="keep", min_keys=2)[2] == ""


def test_stats_entry_point(root, capsys):
    assert main([str(root), "--min-sentences", "4"]) == 0
    out = capsys.readouterr().out
    assert "participants: 6 kept of 6" in out and "dropped correction" in out


# ---- container and splits -------------------------------------------------


def test_optional_fields_survive_subset_and_concat(root):
    data = load_aalto(root, with_keys=True, min_sentences=1)
    part = data.subset(np.arange(3))
    assert part.sentence_ids.tolist() == data.sentence_ids[:3].tolist()
    assert part.key_codes.shape == (3, data.max_len)
    both = KeystrokeData.concat([part, data.subset(np.arange(3, 6))])
    assert both.sentence_ids.tolist() == data.sentence_ids[:6].tolist()
    np.testing.assert_array_equal(both.key_codes[:, : both.max_len], data.key_codes[:6, : both.max_len])
    plain = KeystrokeData.from_arrays([np.ones((3, 3))], ["u"])
    assert KeystrokeData.concat([part, plain]).sentence_ids is None


def test_container_rejects_bad_optional_shapes():
    with pytest.raises(ValueError):
        KeystrokeData.from_arrays([np.ones((3, 3))], ["u"], sentence_ids=[1, 2])
    with pytest.raises(ValueError):
        KeystrokeData.from_arrays([np.ones((3, 3))], ["u"], key_codes=[[1, 2]])
    with pytest.raises(ValueError):
        KeystrokeData(
            values=np.ones((1, 3, 3), dtype=np.float32),
            lengths=np.array([3]),
            user_ids=np.array(["u"], dtype=object),
            sessions=np.array([-1]),
            key_codes=np.zeros((1, 2), dtype=np.int64),
        )


def test_split_sentences_holds_out_whole_sentences(root):
    data = load_aalto(root, min_sentences=1)
    seen, unseen = split_sentences(data, holdout_frac=0.25, seed=0)
    assert len(seen) + len(unseen) == len(data)
    assert len(np.unique(unseen.sentence_ids)) == 1
    assert not set(seen.sentence_ids.tolist()) & set(unseen.sentence_ids.tolist())
    again, _ = split_sentences(data, holdout_frac=0.25, seed=0)
    assert again.sentence_ids.tolist() == seen.sentence_ids.tolist()
    # combined with user splits: the optional fields go along
    train, val, test = split_users(data, val_frac=0.2, test_frac=0.2, seed=0)
    assert test.sentence_ids is not None and not set(train.users) & set(test.users)


@pytest.mark.parametrize("frac", [0.0, 1.0, 0.05])
def test_split_sentences_rejects_empty_sides(root, frac):
    data = load_aalto(root, min_sentences=1)
    with pytest.raises(ValueError):
        split_sentences(data, holdout_frac=frac)


def test_split_sentences_needs_sentence_ids():
    data = KeystrokeData.from_arrays([np.ones((3, 3))] * 4, ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="sentence_ids"):
        split_sentences(data)

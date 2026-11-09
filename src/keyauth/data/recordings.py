"""Our own recorded attempts, stored as JSON Lines (one attempt per line).

    {"user_id": "alice", "session": 0, "timings": [[hold, dd, ud], ...]}

Only timings are stored, never the typed characters: the same privacy rule
the API follows (the phrase is usually a password).
"""

from __future__ import annotations

import json
from pathlib import Path

from keyauth.data.container import KeystrokeData
from keyauth.features import TimingSequence


def append_recording(path: str | Path, user_id: str, seq: TimingSequence, session: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"user_id": str(user_id), "session": int(session), "timings": seq.values.tolist()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def save_recordings(path: str | Path, data: KeystrokeData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(len(data)):
            line = {
                "user_id": str(data.user_ids[i]),
                "session": int(data.sessions[i]),
                "timings": data.values[i, : data.lengths[i]].tolist(),
            }
            f.write(json.dumps(line) + "\n")


def load_recordings(path: str | Path) -> KeystrokeData:
    seqs, users, sessions = [], [], []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            rec = json.loads(raw)
            timings = rec.get("timings")
            if "user_id" not in rec or not isinstance(timings, list) or len(timings) < 2:
                raise ValueError(f"{path}:{lineno}: need user_id and at least 2 timing rows")
            seqs.append(timings)
            users.append(rec["user_id"])
            sessions.append(rec.get("session", -1))
    return KeystrokeData.from_arrays(seqs, users, sessions)

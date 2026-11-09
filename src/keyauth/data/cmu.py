"""Loader for the CMU Keystroke Dynamics Benchmark (DSL-StrongPasswordData.csv).

Killourhy & Maxion (2009): 51 subjects each typed ".tie5Roanl" + Return 400
times, over 8 sessions of 50 repetitions. Columns:

    subject, sessionIndex, rep,
    H.period, DD.period.t, UD.period.t, H.t, DD.t.i, UD.t.i, ..., H.Return

i.e. exactly TimingSequence.to_vector() order, in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from keyauth.config import N_CHANNELS, SUBMIT_KEY
from keyauth.data.container import KeystrokeData

CMU_PHRASE = ".tie5Roanl"
CMU_KEYS = tuple(CMU_PHRASE) + (SUBMIT_KEY,)
# key names as they appear in the CSV header
CMU_COLUMN_KEYS = ("period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return")


def cmu_timing_columns() -> list[str]:
    """The 31 timing column names, in file order."""
    cols = []
    for a, b in zip(CMU_COLUMN_KEYS[:-1], CMU_COLUMN_KEYS[1:]):
        cols.extend([f"H.{a}", f"DD.{a}.{b}", f"UD.{a}.{b}"])
    cols.append(f"H.{CMU_COLUMN_KEYS[-1]}")
    return cols


def load_cmu(path: str | Path) -> KeystrokeData:
    df = pd.read_csv(path)
    expected = cmu_timing_columns()
    missing = [c for c in ["subject", "sessionIndex", *expected] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is not the CMU DSL-StrongPasswordData format; missing columns {missing[:5]}")

    vec = df[expected].to_numpy(dtype=np.float32)
    n_keys = len(CMU_KEYS)
    values = np.zeros((len(df), n_keys, N_CHANNELS), dtype=np.float32)
    for i in range(n_keys - 1):
        values[:, i, :] = vec[:, 3 * i : 3 * i + 3]
    values[:, -1, 0] = vec[:, -1]

    return KeystrokeData(
        values=values,
        lengths=np.full(len(df), n_keys, dtype=np.int64),
        user_ids=df["subject"].astype(str).to_numpy(dtype=object),
        sessions=df["sessionIndex"].to_numpy(dtype=np.int64),
    )

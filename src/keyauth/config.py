"""Project-wide constants.

Units: raw key events are timestamped in MILLISECONDS (what browsers give via
performance.now()). Extracted timing features are in SECONDS, matching the CMU
keystroke benchmark dataset.
"""

# Keys that only modify other keys. They are dropped before feature
# extraction, so "Shift + r" becomes a single "R" keystroke (same convention
# as the CMU dataset's "Shift.r" column).
MODIFIER_KEYS = frozenset({"Shift", "Control", "Alt", "Meta", "CapsLock", "AltGraph"})

# Key that submits the phrase. It is kept as the final keystroke, like the
# CMU dataset's "Return" column.
SUBMIT_KEY = "Enter"

# Number of timing channels per keystroke: hold, down-down, up-down.
N_CHANNELS = 3
CHANNEL_NAMES = ("hold", "dd", "ud")

MS_PER_SECOND = 1000.0

# Sanity bounds on a single timing value (seconds). Anything outside this is
# treated as a broken capture (e.g. user walked away mid-phrase).
MAX_HOLD_S = 2.0
MAX_LATENCY_S = 5.0

# Limits on timing arrays submitted to the engine / API.
MAX_KEYS = 128
# dd must equal hold + ud for each keystroke; allow this much rounding error (seconds).
DD_TOLERANCE_S = 0.002

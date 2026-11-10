// Pure keystroke-capture helpers for the keyauth demo page. No DOM access here.
// Timestamps are milliseconds (performance.now()), like the API expects.

export const PHRASE = ".tie5Roanl";
export const SUBMIT_KEY = "Enter";
// Same list as MODIFIER_KEYS in src/keyauth/config.py: the server drops these.
export const MODIFIER_KEYS = new Set(["Shift", "Control", "Alt", "Meta", "CapsLock", "AltGraph"]);
// Keys that mean "the user is correcting a typo": rhythm is not comparable, start over.
export const RESET_KEYS = new Set(["Backspace", "Delete"]);
export const USER_ID_PATTERN = /^[A-Za-z0-9._@-]{1,64}$/;

export function isValidUserId(value) {
  return typeof value === "string" && USER_ID_PATTERN.test(value);
}

// Text spelled by the non-modifier keydowns, without the trailing Enter.
// Events are the raw ones (real key names), before anonymize().
export function matchesPhrase(events, phrase = PHRASE) {
  const keys = events.filter((e) => e.type === "down" && !MODIFIER_KEYS.has(e.key)).map((e) => e.key);
  const expected = Array.from(phrase);
  expected.push(SUBMIT_KEY);
  if (keys.length !== expected.length) {
    return false;
  }
  return keys.every((k, i) => k === expected[i]);
}

// True when every keydown has a keyup (auto-repeat already filtered by the recorder).
export function isComplete(events) {
  const open = new Set();
  for (const e of events) {
    if (e.type === "down") {
      open.add(e.code);
    } else {
      open.delete(e.code);
    }
  }
  return open.size === 0;
}

// Remove everything that reveals what was typed. Keeps `key` only for modifiers and Enter;
// every other key becomes "x". `code` becomes an opaque per-press id ("k0", "k1", ...) assigned
// in order of first press, shared by the down and up of the same press, so the server (events.py
// pair_events) still pairs them. A repeated physical key gets a new id, so repeated letters do not
// show. Timestamps are rebased to the first event (only differences matter). Keyups without a
// matching keydown are dropped (the server would ignore them anyway).
export function anonymize(events) {
  const sorted = events.slice();
  const openIds = new Map(); // original code -> opaque id of the press that is still held
  let next = 0;
  let origin = null;
  const out = [];
  for (const e of sorted) {
    if (origin === null) {
      origin = e.t;
    }
    let key = "x";
    if (MODIFIER_KEYS.has(e.key) || e.key === SUBMIT_KEY) {
      key = e.key;
    }
    if (e.type === "down") {
      if (openIds.has(e.code)) {
        continue; // auto-repeat that slipped through
      }
      const id = "k" + next;
      next += 1;
      openIds.set(e.code, id);
      out.push({ key, type: "down", t: e.t - origin, code: id });
    } else {
      const id = openIds.get(e.code);
      if (id === undefined) {
        continue;
      }
      openIds.delete(e.code);
      out.push({ key, type: "up", t: e.t - origin, code: id });
    }
  }
  return out;
}

// Collects raw events for one attempt. Feed it keydown/keyup (with performance.now() times).
// Every method returns a result object: {status: "ok"} | {status: "reset", reason} |
// {status: "done", events}.  reason: "correction" | "incomplete" | "wrong_password".
// "done" events are already anonymized and safe to send.
export class Recorder {
  constructor(phrase = PHRASE) {
    this.phrase = phrase;
    this.reset();
  }

  reset() {
    this.events = [];
    this.held = new Set();
  }

  get length() {
    return this.events.length;
  }

  keyDown(key, code, t, repeat = false) {
    if (repeat) {
      return { status: "ok" }; // auto-repeat is not a new keystroke
    }
    if (RESET_KEYS.has(key)) {
      this.reset();
      return { status: "reset", reason: "correction" };
    }
    if (this.held.has(code)) {
      return { status: "ok" };
    }
    this.held.add(code);
    this.events.push({ key, type: "down", t, code });
    return { status: "ok" };
  }

  keyUp(key, code, t) {
    if (!this.held.has(code)) {
      return { status: "ok" }; // key was pressed before capture began
    }
    this.held.delete(code);
    this.events.push({ key, type: "up", t, code });
    if (key !== SUBMIT_KEY) {
      return { status: "ok" };
    }
    const raw = this.events;
    this.reset();
    if (!isComplete(raw)) {
      return { status: "reset", reason: "incomplete" };
    }
    if (!matchesPhrase(raw, this.phrase)) {
      return { status: "reset", reason: "wrong_password" };
    }
    return { status: "done", events: anonymize(raw) };
  }

  // paste, cut, drop, blur
  interrupt() {
    this.reset();
    return { status: "reset", reason: "correction" };
  }
}

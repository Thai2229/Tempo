import test from "node:test";
import assert from "node:assert/strict";

import { PHRASE, Recorder, anonymize, isComplete, isValidUserId, matchesPhrase } from "../../src/keyauth/web/static/capture.js";

const CODES = { ".": "Period", t: "KeyT", i: "KeyI", e: "KeyE", 5: "Digit5", R: "KeyR", o: "KeyO", a: "KeyA", n: "KeyN", l: "KeyL", Enter: "Enter" };

// Feed the phrase (R typed with Shift) into a recorder; returns the last result.
function typePhrase(rec, { phrase = PHRASE, start = 1000, enter = true } = {}) {
  let t = start;
  let last = { status: "ok" };
  for (const ch of phrase) {
    if (ch === "R") {
      rec.keyDown("Shift", "ShiftLeft", t, false);
      t += 20;
    }
    rec.keyDown(ch, CODES[ch], t, false);
    t += 90;
    if (ch === "R") {
      rec.keyUp("Shift", "ShiftLeft", t - 30);
    }
    rec.keyUp(ch, CODES[ch], t);
    t += 60;
  }
  if (enter) {
    rec.keyDown("Enter", "Enter", t, false);
    t += 80;
    last = rec.keyUp("Enter", "Enter", t);
  }
  return last;
}

test("a clean attempt is done and anonymized", () => {
  const rec = new Recorder();
  const res = typePhrase(rec);
  assert.equal(res.status, "done");
  const json = JSON.stringify(res.events);
  for (const ch of ".tie5Roanl") {
    // only "x" placeholders, modifier names and Enter may appear as keys
    assert.ok(!res.events.some((e) => e.key === ch), "leaked key " + ch);
  }
  for (const e of res.events) {
    assert.match(e.code, /^k\d+$/);
    assert.ok(["x", "Shift", "Enter"].includes(e.key));
  }
  assert.ok(!/Key[A-Z]|Digit|Period/.test(json));
  assert.equal(res.events.length, 2 * (PHRASE.length + 2)); // phrase + Shift + Enter, down and up
  assert.equal(res.events[0].t, 0); // rebased
  assert.equal(rec.length, 0);
});

test("anonymize keeps modifiers and Enter, pairs by opaque id", () => {
  const raw = [
    { key: "Shift", type: "down", t: 100, code: "ShiftLeft" },
    { key: "R", type: "down", t: 120, code: "KeyR" },
    { key: "Shift", type: "up", t: 150, code: "ShiftLeft" },
    { key: "r", type: "up", t: 200, code: "KeyR" }, // key changed between down and up
    { key: "Enter", type: "down", t: 250, code: "Enter" },
    { key: "Enter", type: "up", t: 300, code: "Enter" },
  ];
  const out = anonymize(raw);
  assert.deepEqual(out, [
    { key: "Shift", type: "down", t: 0, code: "k0" },
    { key: "x", type: "down", t: 20, code: "k1" },
    { key: "Shift", type: "up", t: 50, code: "k0" },
    { key: "x", type: "up", t: 100, code: "k1" },
    { key: "Enter", type: "down", t: 150, code: "k2" },
    { key: "Enter", type: "up", t: 200, code: "k2" },
  ]);
});

test("a repeated physical key gets a new id each press", () => {
  const raw = [
    { key: "l", type: "down", t: 0, code: "KeyL" },
    { key: "l", type: "up", t: 50, code: "KeyL" },
    { key: "l", type: "down", t: 100, code: "KeyL" },
    { key: "l", type: "up", t: 150, code: "KeyL" },
  ];
  const codes = anonymize(raw).map((e) => e.code);
  assert.deepEqual(codes, ["k0", "k0", "k1", "k1"]);
});

test("auto-repeat is ignored", () => {
  const rec = new Recorder();
  rec.keyDown(".", "Period", 0, false);
  rec.keyDown(".", "Period", 30, true);
  rec.keyDown(".", "Period", 60, true);
  assert.equal(rec.length, 1);
  // a repeat without the flag for a key that is still held is ignored too
  rec.keyDown(".", "Period", 90, false);
  assert.equal(rec.length, 1);
});

test("Backspace and Delete reset the attempt", () => {
  for (const key of ["Backspace", "Delete"]) {
    const rec = new Recorder();
    rec.keyDown(".", "Period", 0, false);
    rec.keyUp(".", "Period", 50);
    const res = rec.keyDown(key, key, 100, false);
    assert.deepEqual(res, { status: "reset", reason: "correction" });
    assert.equal(rec.length, 0);
  }
});

test("interrupt (paste/blur) resets", () => {
  const rec = new Recorder();
  rec.keyDown(".", "Period", 0, false);
  assert.equal(rec.interrupt().status, "reset");
  assert.equal(rec.length, 0);
});

test("a down without an up is incomplete and not sent", () => {
  const rec = new Recorder();
  rec.keyDown("z", "KeyZ", 0, false); // never released
  const res = typePhrase(rec, { start: 100 });
  assert.deepEqual(res, { status: "reset", reason: "incomplete" });
  assert.equal(isComplete([{ key: "a", type: "down", t: 0, code: "KeyA" }]), false);
  assert.equal(isComplete([]), true);
});

test("wrong text is rejected before sending", () => {
  const rec = new Recorder();
  const res = typePhrase(rec, { phrase: ".tie5Roanl".replace("o", "a").replace("R", "R") });
  // "a" has a code in CODES, so this types .tie5Raanl
  assert.deepEqual(res, { status: "reset", reason: "wrong_password" });
});

test("missing Enter never completes; extra keys fail the phrase check", () => {
  const rec = new Recorder();
  assert.equal(typePhrase(rec, { enter: false }).status, "ok");
  const rec2 = new Recorder();
  typePhrase(rec2, { enter: false });
  rec2.keyDown("l", "KeyL", 5000, false);
  rec2.keyUp("l", "KeyL", 5050);
  rec2.keyDown("Enter", "Enter", 5100, false);
  assert.equal(rec2.keyUp("Enter", "Enter", 5150).reason, "wrong_password");
});

test("keyup of a key pressed before capture is ignored", () => {
  const rec = new Recorder();
  assert.deepEqual(rec.keyUp("Enter", "Enter", 5), { status: "ok" });
  assert.equal(rec.length, 0);
});

test("matchesPhrase ignores modifiers", () => {
  const down = (key, t) => ({ key, type: "down", t, code: key });
  const events = [".", "t", "i", "e", "5"].map(down);
  events.push(down("Shift", 1), down("R", 2), down("o", 3), down("a", 4), down("n", 5), down("l", 6), down("Enter", 7));
  assert.equal(matchesPhrase(events), true);
  assert.equal(matchesPhrase(events.slice(0, -1)), false);
});

test("user id validation matches the API charset", () => {
  for (const ok of ["a", "Ann.Lee-1@x_y", "x".repeat(64)]) {
    assert.equal(isValidUserId(ok), true, ok);
  }
  for (const bad of ["", " ", "a b", "x".repeat(65), "né", "a/b", "a\n"]) {
    assert.equal(isValidUserId(bad), false, JSON.stringify(bad));
  }
});

test("phrase null (embedded login) accepts any password of allowed length ending with Enter", () => {
  const press = (rec, ch, t) => {
    rec.keyDown(ch, "C" + ch, t, false);
    rec.keyUp(ch, "C" + ch, t + 40);
  };
  const attempt = (text) => {
    const rec = new Recorder(null);
    let t = 0;
    for (const ch of text) {
      press(rec, ch, t);
      t += 100;
    }
    rec.keyDown("Enter", "Enter", t, false);
    return rec.keyUp("Enter", "Enter", t + 40);
  };
  assert.equal(attempt("my-own-pw").status, "done");
  assert.equal(attempt("abcd").status, "done");
  assert.deepEqual(attempt("abc"), { status: "reset", reason: "wrong_password" }); // too short
  assert.equal(attempt("a".repeat(100)).status, "done");
  assert.equal(attempt("b".repeat(101)).reason, "wrong_password"); // too long
  assert.equal(matchesPhrase([], null), false);
});

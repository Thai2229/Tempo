import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const SOURCE = fs.readFileSync(new URL("../../src/tempo/web/static/tempo.js", import.meta.url), "utf8");
const PASSWORD = "Zebra-9quilt!";
const SCRIPT_URL = "https://ka.example/v1/tempo.js";

// ---- a minimal fake DOM (no jsdom) --------------------------------------------

class Target {
  constructor() {
    this.listeners = [];
  }
  addEventListener(type, fn, capture = false) {
    this.listeners.push({ type, fn, capture: capture === true });
  }
  removeEventListener(type, fn, capture = false) {
    this.listeners = this.listeners.filter((l) => !(l.type === type && l.fn === fn && l.capture === (capture === true)));
  }
  count(type) {
    return this.listeners.filter((l) => l.type === type).length;
  }
  dispatchEvent(ev) {
    ev.defaultPrevented = false;
    ev.stopped = false;
    ev.preventDefault = () => {
      ev.defaultPrevented = true;
    };
    ev.stopImmediatePropagation = () => {
      ev.stopped = true;
    };
    for (const l of this.listeners.slice()) {
      if (l.type === ev.type && !ev.stopped) {
        l.fn(ev);
      }
    }
    return !ev.defaultPrevented;
  }
}

class FakeInput extends Target {
  constructor(name = "") {
    super();
    this.name = name;
    this.id = "";
    this.type = "text";
    this._value = "";
    this.reads = 0;
    this.form = null;
  }
  get value() {
    this.reads += 1;
    return this._value;
  }
  set value(v) {
    this._value = v;
  }
  remove() {
    if (this.parent) {
      this.parent.children = this.parent.children.filter((c) => c !== this);
      this.parent = null;
    }
  }
}

class FakeForm extends Target {
  constructor() {
    super();
    this.children = [];
    this.submissions = []; // token value of every submission that went through
  }
  appendChild(c) {
    c.parent = this;
    this.children.push(c);
    return c;
  }
  querySelector(sel) {
    const found = this.children.find((c) => "#" + c.id === sel);
    return found || null;
  }
  field(name) {
    return this.children.find((c) => c.name === name);
  }
  requestSubmit() {
    if (this.dispatchEvent({ type: "submit", submitter: null })) {
      const hidden = this.field("tempo_token");
      this.submissions.push(hidden ? hidden._value : undefined);
    }
  }
}

function setup({ fetchImpl, withForm = true } = {}) {
  const clock = { t: 5000 };
  const calls = [];
  const timeouts = [];
  const aborted = [];
  const form = new FakeForm();
  const password = new FakeInput("password");
  const user = new FakeInput("user");
  user.id = "username";
  user.value = "ann.lee-1";
  if (withForm) {
    password.form = form;
    form.appendChild(password);
    form.appendChild(user);
  }
  const document = {
    currentScript: { src: SCRIPT_URL },
    createElement: () => new FakeInput(),
    querySelector: (sel) => {
      if (sel === "#username") {
        return user;
      }
      return null;
    },
  };
  const defaultFetch = async () => ({ ok: true, status: 200, json: async () => ({ token: "tok.sig" }) });
  const impl = fetchImpl || defaultFetch;
  const sandbox = {
    document,
    performance: { now: () => clock.t },
    URL,
    JSON,
    console,
    AbortController: class {
      constructor() {
        this.signal = { aborted: false };
      }
      abort() {
        this.signal.aborted = true;
        aborted.push(true);
      }
    },
    setTimeout: (fn, ms) => {
      timeouts.push(ms);
      return setTimeout(fn, 5); // shrink the 5 s timeout for the tests
    },
    clearTimeout,
    fetch: (url, init) => {
      calls.push({ url, init, body: JSON.parse(init.body) });
      return impl(url, init);
    },
  };
  sandbox.globalThis = sandbox;
  vm.runInNewContext(SOURCE, sandbox);
  return { tempo: sandbox.tempo, clock, calls, timeouts, aborted, form, password, user, document, sandbox };
}

function key(el, type, k, code, extra = {}) {
  return el.dispatchEvent(Object.assign({ type, key: k, code, repeat: false }, extra));
}

const CODE = (ch) => {
  if (ch === "Enter" || ch === "Backspace") {
    return ch;
  }
  return "Key" + ch.toUpperCase().replace(/[^A-Z]/, "_");
};

// Type text like a person: keydown, hold, keyup. Returns the Enter keydown's "not prevented" flag.
function type(env, el, text, { enter = true } = {}) {
  let last = true;
  for (const ch of text) {
    let code = CODE(ch);
    if (ch === ch.toUpperCase() && /[A-Z]/.test(ch)) {
      code += "U";
    }
    key(el, "keydown", ch, code);
    env.clock.t += 80;
    key(el, "keyup", ch, code);
    env.clock.t += 60;
  }
  if (enter) {
    last = key(el, "keydown", "Enter", "Enter");
    env.clock.t += 70;
    key(el, "keyup", "Enter", "Enter");
  }
  return last;
}

const isTypeError = (e) => e.name === "TypeError"; // thrown inside the vm realm
const plain = (x) => JSON.parse(JSON.stringify(x));

const settle = async () => {
  for (let i = 0; i < 5; i++) {
    await new Promise((r) => setImmediate(r));
  }
};

// ---- login ---------------------------------------------------------------------

test("posts anonymized events to /v1/verify, never the password or key names", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk_demo", userIdField: "#username" });
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.calls.length, 1);
  const call = env.calls[0];
  assert.equal(call.url, "https://ka.example/v1/verify"); // default endpoint = the script's origin
  assert.equal(call.init.method, "POST");
  assert.equal(call.init.headers["Content-Type"], "application/json");
  assert.equal(call.init.credentials, "omit");
  assert.deepEqual(Object.keys(call.body).sort(), ["events", "sitekey", "user_id"]);
  assert.equal(call.body.sitekey, "pk_demo");
  assert.equal(call.body.user_id, "ann.lee-1");
  const raw = call.init.body;
  assert.equal(call.body.events.length, 2 * (PASSWORD.length + 1));
  for (const e of call.body.events) {
    assert.ok(["x", "Enter"].includes(e.key), "key " + e.key);
    assert.match(e.code, /^k\d+$/);
    assert.equal(typeof e.t, "number");
  }
  for (const piece of ["Zebra", "ebra", "quilt", "-9q", "Key", "Digit", "Minus", "Backspace", PASSWORD]) {
    assert.ok(!raw.includes(piece), "leaked " + piece);
  }
  assert.equal(call.body.events[0].t, 0);
  assert.equal(env.password.reads, 0, "the password field's value must never be read");
  assert.equal(env.form.submissions.length, 1);
});

test("the token lands in the hidden field and the submit continues with it", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  assert.equal(key(env.password, "keydown", "Enter", "Enter"), false, "Enter keydown is held back so the form waits for keyup");
  key(env.password, "keyup", "Enter", "Enter");
  const hidden = env.form.field("tempo_token");
  assert.equal(hidden.type, "hidden");
  assert.equal(hidden.value, "");

  const env2 = setup();
  env2.tempo.attach(env2.password, { sitekey: "pk", userIdField: "#username", tokenField: "ka_tok" });
  type(env2, env2.password, PASSWORD);
  assert.equal(env2.form.submissions.length, 0, "intercepted until tempo answers");
  await settle();
  assert.equal(env2.form.field("ka_tok").value, "tok.sig");
  assert.equal(env2.form.field("tempo_token"), undefined);
  assert.equal(env2.form.children.filter((c) => c.name === "ka_tok").length, 1);
});

test("server error, network failure and bad JSON: submit continues with an empty token", async () => {
  const cases = {
    "500": async () => ({ ok: false, status: 500, json: async () => ({ error: { code: "internal" } }) }),
    "403": async () => ({ ok: false, status: 403, json: async () => ({ error: { code: "forbidden" } }) }),
    reject: async () => {
      throw new TypeError("Failed to fetch");
    },
    throws: () => {
      throw new Error("sync failure");
    },
    "not json": async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError("x"); } }),
    "no token": async () => ({ ok: true, status: 200, json: async () => ({}) }),
  };
  for (const [name, fetchImpl] of Object.entries(cases)) {
    const env = setup({ fetchImpl });
    env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
    type(env, env.password, PASSWORD);
    await settle();
    assert.deepEqual(env.form.submissions, [""], name);
  }
});

test("a 5 second timeout aborts the request and the submit continues with an empty token", async () => {
  const env = setup({ fetchImpl: () => new Promise(() => {}) });
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  type(env, env.password, PASSWORD);
  await new Promise((r) => setTimeout(r, 40));
  await settle();
  assert.ok(env.timeouts.includes(5000), "timeout is 5000 ms");
  assert.equal(env.aborted.length, 1);
  assert.deepEqual(env.form.submissions, [""]);
});

test("Backspace and Delete reset the attempt", async () => {
  for (const reset of ["Backspace", "Delete"]) {
    const env = setup();
    env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
    type(env, env.password, "abcdef", { enter: false });
    key(env.password, "keydown", reset, reset);
    key(env.password, "keyup", reset, reset);
    type(env, env.password, "ghijk");
    await settle();
    assert.equal(env.calls.length, 1);
    assert.equal(env.calls[0].body.events.length, 2 * (5 + 1), "only the attempt after the correction is sent");
  }
});

test("Backspace then too short a password: no request, submit continues", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  type(env, env.password, "abcdef", { enter: false });
  key(env.password, "keydown", "Backspace", "Backspace");
  key(env.password, "keyup", "Backspace", "Backspace");
  type(env, env.password, "ab");
  await settle();
  assert.equal(env.calls.length, 0);
  assert.deepEqual(env.form.submissions, [""]);
});

test("a click submit without a typed attempt is not intercepted", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  env.form.requestSubmit();
  await settle();
  assert.equal(env.calls.length, 0);
  assert.deepEqual(env.form.submissions, [""]);
});

test("paste and blur reset the attempt", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  type(env, env.password, "abcdef", { enter: false });
  env.password.dispatchEvent({ type: "paste" });
  type(env, env.password, "xyz");
  await settle();
  assert.equal(env.calls.length, 0, "3 characters after the paste are below the minimum");
  type(env, env.password, "abcdef", { enter: false });
  env.password.dispatchEvent({ type: "blur" });
  type(env, env.password, "abcd");
  await settle();
  assert.equal(env.calls.length, 1);
  assert.equal(env.calls[0].body.events.length, 2 * 5);
});

test("missing or invalid user_id: no request and the submit continues", async () => {
  for (const value of ["", "   ", "a@b.example", "has space"]) {
    const env = setup();
    env.user.value = value;
    env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
    type(env, env.password, PASSWORD);
    await settle();
    assert.equal(env.calls.length, 0, JSON.stringify(value));
    assert.deepEqual(env.form.submissions, [""]);
  }
  const noOption = setup();
  noOption.tempo.attach(noOption.password, { sitekey: "pk" });
  type(noOption, noOption.password, PASSWORD);
  await settle();
  assert.equal(noOption.calls.length, 0);
  assert.equal(noOption.form.submissions.length, 1);
});

test("userIdField accepts a selector, an element or a function", async () => {
  for (const make of [(env) => "#username", (env) => env.user, () => () => "ann.lee-1"]) {
    const env = setup();
    env.tempo.attach(env.password, { sitekey: "pk", userIdField: make(env) });
    type(env, env.password, PASSWORD);
    await settle();
    assert.equal(env.calls[0].body.user_id, "ann.lee-1");
  }
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: () => { throw new Error("boom"); } });
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.calls.length, 0);
  assert.equal(env.form.submissions.length, 1);
});

test("an explicit endpoint is used, without a trailing slash; a bad one never sends", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username", endpoint: "https://eu.tempo.example/" });
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.calls[0].url, "https://eu.tempo.example/v1/verify");

  const bad = setup();
  bad.tempo.attach(bad.password, { sitekey: "pk", userIdField: "#username", endpoint: "javascript:alert(1)" });
  type(bad, bad.password, PASSWORD);
  await settle();
  assert.equal(bad.calls.length, 0);
  assert.equal(bad.form.submissions.length, 1);
});

test("double attach replaces the first instance; destroy twice is safe and removes everything", async () => {
  const env = setup();
  const first = env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  const second = env.tempo.attach(env.password, { sitekey: "pk2", userIdField: "#username" });
  assert.notEqual(first, second);
  assert.equal(env.password.count("keydown"), 1);
  assert.equal(env.form.count("submit"), 1);
  assert.equal(env.form.children.filter((c) => c.name === "tempo_token").length, 1);
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.calls.length, 1);
  assert.equal(env.calls[0].body.sitekey, "pk2");

  second.destroy();
  second.destroy();
  first.destroy();
  assert.equal(env.password.count("keydown"), 0);
  assert.equal(env.password.count("keyup"), 0);
  assert.equal(env.form.count("submit"), 0);
  assert.equal(env.form.field("tempo_token"), undefined);
  const before = env.calls.length;
  assert.equal(key(env.password, "keydown", "Enter", "Enter"), true, "Enter is no longer held back");
  key(env.password, "keyup", "Enter", "Enter");
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.calls.length, before);
  assert.deepEqual(await second.getToken(), "");
});

test("reset() drops the attempt and the token", async () => {
  const env = setup();
  const k = env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.form.field("tempo_token").value, "tok.sig");
  k.reset();
  assert.equal(env.form.field("tempo_token").value, "");
  assert.equal(await k.getToken(), "");
});

test("getToken works without a form and never rejects", async () => {
  const env = setup({ withForm: false });
  const k = env.tempo.attach(env.password, { sitekey: "pk", userIdField: () => "ann" });
  assert.equal(await k.getToken(), "", "no attempt yet");
  assert.equal(key(env.password, "keydown", "Enter", "Enter"), true, "without a form Enter is left alone");
  key(env.password, "keyup", "Enter", "Enter");
  type(env, env.password, PASSWORD);
  assert.equal(await k.getToken(), "tok.sig");
  assert.equal(env.calls.length, 1);

  const failing = setup({ withForm: false, fetchImpl: async () => { throw new Error("down"); } });
  const f = failing.tempo.attach(failing.password, { sitekey: "pk", userIdField: () => "ann" });
  type(failing, failing.password, PASSWORD);
  assert.equal(await f.getToken(), "");
});

test("typing again after a finished attempt clears the old token", async () => {
  const env = setup();
  env.tempo.attach(env.password, { sitekey: "pk", userIdField: "#username" });
  type(env, env.password, PASSWORD);
  await settle();
  assert.equal(env.form.field("tempo_token").value, "tok.sig");
  key(env.password, "keydown", "a", "KeyA");
  assert.equal(env.form.field("tempo_token").value, "");
});

test("attach validates its arguments", () => {
  const env = setup();
  assert.throws(() => env.tempo.attach(null, { sitekey: "pk" }), isTypeError);
  assert.throws(() => env.tempo.attach(env.password, {}), isTypeError);
  assert.throws(() => env.tempo.attach(env.password), isTypeError);
});

// ---- enroll --------------------------------------------------------------------

test("enroll posts one anonymized attempt with the ticket and resolves the progress", async () => {
  const env = setup({
    fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ status: "collecting", samples: 3, required: 10 }) }),
  });
  const pending = env.tempo.enroll(env.password, { sitekey: "pk", ticket: "tk123", user_id: "ann.lee-1" });
  type(env, env.password, "abcdef", { enter: false });
  key(env.password, "keydown", "Backspace", "Backspace"); // correction: keep waiting
  key(env.password, "keyup", "Backspace", "Backspace");
  assert.equal(env.calls.length, 0);
  assert.equal(type(env, env.password, PASSWORD), false, "Enter must not submit the customer's form while enrolling");
  const result = await pending;
  assert.deepEqual(plain(result), { samples: 3, required: 10, done: false });
  assert.equal(env.calls.length, 1);
  assert.equal(env.calls[0].url, "https://ka.example/v1/enroll");
  assert.deepEqual(Object.keys(env.calls[0].body).sort(), ["events", "sitekey", "ticket", "user_id"]);
  assert.equal(env.calls[0].body.ticket, "tk123");
  assert.ok(!env.calls[0].init.body.includes("Zebra"));
  assert.ok(!/Key[A-Z]|Digit/.test(env.calls[0].init.body));
  assert.equal(env.password.reads, 0);
  assert.equal(env.password.count("keydown"), 0, "listeners are removed after the sample");
  assert.equal(env.form.submissions.length, 0);
});

test("enroll reports done, and the reason when the server restarts enrollment", async () => {
  const env = setup({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ status: "complete", samples: 10, required: 10 }) }) });
  const p = env.tempo.enroll(env.password, { sitekey: "pk", ticket: "t", user_id: "ann" });
  type(env, env.password, PASSWORD);
  assert.deepEqual(plain(await p), { samples: 10, required: 10, done: true });

  const env2 = setup({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ status: "rejected", reason: "inconsistent", samples: 0, required: 10 }) }) });
  const p2 = env2.tempo.enroll(env2.password, { sitekey: "pk", ticket: "t", user_id: "ann" });
  type(env2, env2.password, PASSWORD);
  assert.deepEqual(plain(await p2), { samples: 0, required: 10, done: false, reason: "inconsistent" });
});

test("enroll rejects with the server's error code, and on network failure or timeout", async () => {
  const env = setup({ fetchImpl: async () => ({ ok: false, status: 403, json: async () => ({ error: { code: "invalid_ticket" } }) }) });
  const p = env.tempo.enroll(env.password, { sitekey: "pk", ticket: "bad", user_id: "ann" });
  type(env, env.password, PASSWORD);
  await assert.rejects(p, (e) => e.code === "invalid_ticket" && e.status === 403);

  const net = setup({ fetchImpl: async () => { throw new TypeError("down"); } });
  const p2 = net.tempo.enroll(net.password, { sitekey: "pk", ticket: "t", user_id: "ann" });
  type(net, net.password, PASSWORD);
  await assert.rejects(p2, (e) => e.code === "network");

  const slow = setup({ fetchImpl: () => new Promise(() => {}) });
  const p3 = slow.tempo.enroll(slow.password, { sitekey: "pk", ticket: "t", user_id: "ann" });
  type(slow, slow.password, PASSWORD);
  await assert.rejects(p3, (e) => e.code === "timeout");
});

test("enroll validates arguments and a second call cancels the first", async () => {
  const env = setup({ fetchImpl: async () => ({ ok: true, status: 200, json: async () => ({ status: "collecting", samples: 1, required: 10 }) }) });
  await assert.rejects(env.tempo.enroll(env.password, { sitekey: "pk", ticket: "", user_id: "ann" }), (e) => e.code === "bad_arguments");
  await assert.rejects(env.tempo.enroll(env.password, { sitekey: "pk", ticket: "t", user_id: "a@b" }), (e) => e.code === "bad_arguments");
  const first = env.tempo.enroll(env.password, { sitekey: "pk", ticket: "t", user_id: "ann" });
  const second = env.tempo.enroll(env.password, { sitekey: "pk", ticket: "t", user_id: "ann" });
  await assert.rejects(first, (e) => e.code === "cancelled");
  assert.equal(env.password.count("keydown"), 1);
  type(env, env.password, PASSWORD);
  await second;
  assert.equal(env.calls.length, 1);
});

// ---- the file itself -----------------------------------------------------------

test("script is CSP friendly: no eval, no dynamic code, no storage, no other hosts", () => {
  for (const banned of [/\beval\s*\(/, /new\s+Function/, /\bFunction\s*\(/, /localStorage/, /sessionStorage/, /document\.cookie/, /indexedDB/, /\.innerHTML/, /document\.write/, /setTimeout\s*\(\s*["'`]/, /\bon[a-z]+\s*=/, /console\./, /XMLHttpRequest/, /sendBeacon/, /importScripts/]) {
    assert.ok(!banned.test(SOURCE), String(banned));
  }
  assert.ok(!/https?:\/\/(?!\\\/)[a-z]/i.test(SOURCE.replace(/\/\*[\s\S]*?\*\//, "").replace(/\/\/.*$/gm, "")), "no hard-coded hosts");
});

test("exposes only attach and enroll", () => {
  const env = setup();
  assert.deepEqual(Object.keys(env.tempo).sort(), ["attach", "enroll"]);
  const k = env.tempo.attach(env.password, { sitekey: "pk" });
  assert.deepEqual(Object.keys(k).sort(), ["destroy", "getToken", "reset"]);
});

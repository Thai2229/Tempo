import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";

import { verify, issueTicket } from "./tempo-verify.mjs";

const SECRET = "sk.s0123456789ab.SUPER-SECRET-VALUE";
const OK = { success: true, user_id: "u_1", rhythm_match: true, risk: "low" };

// A stub server; `state.reply` = [status, body string, headers], `state.delay` in ms.
async function startStub() {
  const state = { reply: [200, "{}", {}], delay: 0, requests: [] };
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      state.requests.push({ path: req.url, headers: req.headers, body: JSON.parse(body) });
      setTimeout(() => {
        const [status, text, headers] = state.reply;
        res.writeHead(status, headers);
        res.end(text);
      }, state.delay);
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  state.url = `http://127.0.0.1:${server.address().port}`;
  state.close = () => {
    server.closeAllConnections();
    server.close();
  };
  return state;
}

async function closedPortUrl() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return `http://127.0.0.1:${port}`;
}

test("verify: success and request shape", async () => {
  const s = await startStub();
  try {
    s.reply = [200, JSON.stringify(OK), {}];
    const r = await verify(SECRET, "tok", s.url + "/");
    assert.deepEqual(r, { success: true, user_id: "u_1", rhythm_match: true, risk: "low", reason: null });
    assert.equal(s.requests[0].path, "/v1/siteverify");
    assert.deepEqual(s.requests[0].body, { secret: SECRET, token: "tok" });
    assert.equal(s.requests[0].headers.origin, undefined);
  } finally {
    s.close();
  }
});

test("verify: invalid_token, wrong secret, non-JSON, wrong shape", async () => {
  const s = await startStub();
  try {
    s.reply = [200, JSON.stringify({ success: false, error: "invalid_token" }), {}];
    assert.equal((await verify(SECRET, "tok", s.url)).reason, "invalid_token");
    s.reply = [401, JSON.stringify({ error: { code: "unauthorized", message: "x" } }), {}];
    const r = await verify(SECRET, "tok", s.url);
    assert.equal(r.success, false);
    assert.equal(r.reason, "unauthorized");
    s.reply = [429, "{}", {}];
    assert.equal((await verify(SECRET, "tok", s.url)).reason, "rate_limited");
    for (const status of [200, 502]) {
      s.reply = [status, "<html>Bad gateway</html>", {}];
      assert.equal((await verify(SECRET, "tok", s.url)).reason, "bad_response");
    }
    for (const body of ["[]", "null", '"x"', '{"success": true}']) {
      s.reply = [200, body, {}];
      assert.equal((await verify(SECRET, "tok", s.url)).reason, "bad_response");
    }
  } finally {
    s.close();
  }
});

test("verify: timeout, unreachable, no token, bad base url", async () => {
  const s = await startStub();
  try {
    s.reply = [200, JSON.stringify(OK), {}];
    s.delay = 1500;
    const started = Date.now();
    const r = await verify(SECRET, "tok", s.url, 0.2);
    assert.equal(r.success, false);
    assert.equal(r.reason, "unreachable");
    assert.ok(Date.now() - started < 1200);
    s.delay = 0;
    const before = s.requests.length;
    assert.equal((await verify(SECRET, "", s.url)).reason, "no_token");
    assert.equal(s.requests.length, before);
  } finally {
    s.close();
  }
  assert.equal((await verify(SECRET, "tok", await closedPortUrl(), 1)).reason, "unreachable");
  assert.equal((await verify(SECRET, "tok", "file:///etc/passwd")).reason, "bad_config");
});

test("verify: a redirect is not followed", async () => {
  const target = await startStub();
  const s = await startStub();
  try {
    target.reply = [200, JSON.stringify(OK), {}];
    s.reply = [307, "", { Location: target.url + "/v1/siteverify" }];
    const r = await verify(SECRET, "tok", s.url);
    assert.equal(r.success, false);
    assert.equal(r.reason, "bad_response");
    assert.equal(target.requests.length, 0);
  } finally {
    s.close();
    target.close();
  }
});

test("issueTicket: success, reset flag, failures", async () => {
  const s = await startStub();
  try {
    s.reply = [200, JSON.stringify({ ticket: "t123", expires_in: 600 }), {}];
    assert.deepEqual(await issueTicket(SECRET, "u_1", s.url), { success: true, ticket: "t123", expires_in: 600, reason: null });
    await issueTicket(SECRET, "u_1", s.url, true);
    assert.deepEqual(s.requests[0].body, { secret: SECRET, user_id: "u_1", reset: false });
    assert.equal(s.requests[1].body.reset, true);
    s.reply = [409, JSON.stringify({ error: { code: "already_enrolled", message: "x" } }), {}];
    assert.equal((await issueTicket(SECRET, "u_1", s.url)).reason, "already_enrolled");
    s.reply = [401, "{}", {}];
    assert.equal((await issueTicket(SECRET, "u_1", s.url)).reason, "unauthorized");
    s.reply = [200, "nope", {}];
    assert.equal((await issueTicket(SECRET, "u_1", s.url)).reason, "bad_response");
    s.reply = [200, JSON.stringify({ ticket: "t", expires_in: 600 }), {}];
    s.delay = 1500;
    assert.equal((await issueTicket(SECRET, "u_1", s.url, false, 0.2)).reason, "unreachable");
  } finally {
    s.close();
  }
  assert.equal((await issueTicket(SECRET, "u_1", await closedPortUrl(), false, 1)).reason, "unreachable");
});

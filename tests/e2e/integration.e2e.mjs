// Real-browser end-to-end check of the embedding (C13): the independent demo site (demo_site/) + tempo /v1.
// Not part of pytest. Start both servers (scripts/run_integration_e2e.sh does all of this for you), then:
//
//   1. tempo with the multi-site service on, a signing key, a throw-away database and the shipped model:
//        export TEMPO_ENABLE_V1=1 TEMPO_TOKEN_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
//        export TEMPO_DB=/tmp/ka.db TEMPO_PORT=8765
//        python -m tempo.sites create --name "Demo Shop" --origin http://127.0.0.1:9000   # prints site key + secret ONCE
//        python -m tempo.api &
//   2. the customer site, configured only through environment variables:
//        TEMPO_URL=http://127.0.0.1:8765 TEMPO_SITE_KEY=pk_... TEMPO_SITE_SECRET=sk.... \
//        DEMO_SITE_DB=/tmp/site.db DEMO_SITE_PORT=9000 python -m demo_site &
//   3. the browser test (the secret is needed here only to prove it never reaches the browser, and to call
//      /v1/siteverify once itself to see whether the site consumed a token):
//        TEMPO_SITE_SECRET=sk.... node tests/e2e/integration.e2e.mjs
//
// Env: SITE_URL (default http://127.0.0.1:9000), TEMPO_URL (default http://127.0.0.1:8765), TEMPO_SITE_SECRET (required),
// IMPOSTER_TRIES (default 3), IMPOSTER_MIN_FLAGGED (default 3: how many tries must land on the second step).
import { createRequire } from "node:module";
import assert from "node:assert/strict";
const { chromium } = createRequire(import.meta.url)("playwright");

const SITE = process.env.SITE_URL || "http://127.0.0.1:9000";
const TEMPO = process.env.TEMPO_URL || "http://127.0.0.1:8765";
const SECRET = process.env.TEMPO_SITE_SECRET || "";
const IMPOSTER_TRIES = Number(process.env.IMPOSTER_TRIES || "3");
const IMPOSTER_MIN_FLAGGED = Number(process.env.IMPOSTER_MIN_FLAGGED || String(IMPOSTER_TRIES));
assert.ok(SECRET.startsWith("sk."), "set TEMPO_SITE_SECRET (only used to check that it never reaches the browser)");
const SECRET_PART = SECRET.split(".")[2];

const PASSWORD = ".tie5Roanl";
const WRONG_PASSWORD = ".tie5Roanx";
const username = "e2e" + Math.random().toString(36).slice(2, 8);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rand = (a, b) => a + Math.random() * (b - a);

// a fixed per-key rhythm (ms) plus jitter: similar every time, never identical
const OWNER = [...PASSWORD, "Enter"].map((_, i) => ({ hold: 80 + ((i * 17) % 30), gap: 110 + ((i * 29) % 60) }));
// somebody else who knows the password: slow, with a very different pattern
const IMPOSTER = [...PASSWORD, "Enter"].map((_, i) => ({ hold: 190 + ((i * 53) % 90), gap: 330 + ((i * 71) % 220) }));
async function type(page, text, profile, jitter = 12) {
  const keys = [...text, "Enter"];
  for (let i = 0; i < keys.length; i++) {
    const b = profile[i % profile.length];
    await page.keyboard.down(keys[i]);
    await sleep(b.hold + rand(-jitter, jitter));
    await page.keyboard.up(keys[i]);
    if (i < keys.length - 1) {
      await sleep(b.gap + rand(-jitter, jitter));
    }
  }
}

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 900, height: 800 } });
const page = await ctx.newPage();
const problems = [];
page.on("console", (m) => {
  if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) problems.push("console: " + m.text());
});
page.on("pageerror", (e) => problems.push("pageerror: " + e.message));

const tempoPosts = []; // everything the browser sent to tempo
const loginPosts = []; // bodies of POST /login to the site
const bodies = []; // every response body the browser saw: {url, type, text}
page.on("request", (rq) => {
  if (rq.method() !== "POST") return;
  if (rq.url().startsWith(TEMPO)) tempoPosts.push({ url: rq.url(), body: rq.postData() || "" });
  if (rq.url() === SITE + "/login") loginPosts.push(rq.postData() || "");
});
// Bodies are read inside a route handler: reading them from a "response" event is lost when the page navigates.
async function record(route) {
  const res = await route.fetch({ maxRedirects: 0 });
  const type = res.headers()["content-type"] || "";
  if (/json|html|javascript|css|text/.test(type)) bodies.push({ url: res.url(), type, text: await res.text() });
  await route.fulfill({ response: res });
}
await ctx.route(TEMPO + "/**", record);
await ctx.route(SITE + "/**", record);

function tokenOf(formBody) {
  return new URLSearchParams(formBody).get("tempo_token");
}

async function siteverify(token) {
  const res = await fetch(TEMPO + "/v1/siteverify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ secret: SECRET, token }),
  });
  return res.json();
}

// Fill the login form, type the password with a rhythm, wait for the site's answer. Returns where we landed.
async function attemptLogin(name, text, profile, beforeSubmit, arg) {
  await page.goto(SITE + "/login");
  await page.fill("#username", name);
  await page.press("#username", "Tab"); // fires "change": the page asks the site for the opaque id
  await page.waitForFunction(() => /^u_[0-9a-f]{32}$/.test(document.querySelector("#tempo-uid").value), null, { timeout: 5000 });
  if (beforeSubmit) await page.evaluate(beforeSubmit, arg);
  const answered = page.waitForResponse((r) => r.url() === SITE + "/login" && r.request().method() === "POST");
  await page.focus("#password");
  await type(page, text, profile);
  const response = await answered;
  if (response.status() === 303) {
    await page.waitForURL((u) => u.pathname !== "/login"); // follow the redirect
  }
  await page.waitForLoadState("load");
  return { status: response.status(), path: new URL(page.url()).pathname, text: await page.textContent("body") };
}

// ---- register ---------------------------------------------------------------------------------------------
await page.goto(SITE + "/register");
await page.fill("#username", username);
await page.fill("#password", PASSWORD);
await Promise.all([page.waitForURL(SITE + "/account"), page.click("button")]);
assert.match(await page.textContent("body"), /Signed in as/);
console.log("registered:", username);

// ---- enroll: 10 typed attempts via tempo.enroll with a ticket the site issued to this session -------
const ticketReq = page.waitForRequest((rq) => rq.url() === SITE + "/account/ticket");
await page.click("#start-enroll");
await ticketReq;
await page.waitForSelector("#enroll-area:not([hidden])");
await page.focus("#enroll-pw");
for (let n = 1; n <= 10; n++) {
  await type(page, PASSWORD, OWNER);
  await page.waitForFunction((k) => document.querySelector("#enroll-status").textContent === k + " / 10", n, { timeout: 5000 });
}
await page.waitForFunction(() => /Enrollment complete/.test(document.querySelector("#enroll-message").textContent), null, { timeout: 5000 });
console.log("enrolled: 10 / 10");

// ---- log out, log in with the same rhythm -> rhythm_match true ------------------------------------------
await Promise.all([page.waitForURL(SITE + "/login"), page.click("#logout")]);

const ok = await attemptLogin(username, PASSWORD, OWNER);
assert.equal(ok.path, "/account", "genuine login should land on /account, got " + ok.path + " " + ok.text.slice(0, 200));
assert.equal(await page.getAttribute("#login-method", "data-method"), "rhythm", "logged in, but not through a rhythm match");
const goodToken = tokenOf(loginPosts[loginPosts.length - 1]);
assert.ok(goodToken && goodToken.includes("."), "the login form carried a tempo token");
console.log("login (same rhythm):", (await page.textContent("#login-method")).trim());
await Promise.all([page.waitForURL(SITE + "/login"), page.click("#logout")]);

// ---- wrong password: the site must reject it BEFORE calling siteverify ------------------------------------
const wrong = await attemptLogin(username, WRONG_PASSWORD, OWNER);
assert.equal(wrong.status, 401);
assert.equal(wrong.path, "/login");
assert.match(wrong.text, /Invalid username or password/);
const unusedToken = tokenOf(loginPosts[loginPosts.length - 1]);
assert.ok(unusedToken, "the wrong-password form still carried a token");
const probe = await siteverify(unusedToken); // if the site had called siteverify, this token would be spent
assert.equal(probe.success, true, "token was already consumed: the site called siteverify before checking the password");
console.log("wrong password: rejected (401); its token was still unspent, so siteverify was not called");

// ---- tampered token: not accepted through the rhythm path -> second step ---------------------------------
const tamper = () => {
  const form = document.getElementById("login-form");
  // runs after tempo.js put the real token into the hidden field and re-submitted the form
  form.addEventListener("submit", () => {
    const f = form.querySelector('input[name="tempo_token"]');
    const i = f.value.length - 10; // inside the signature, away from the padding bits of the last character
    const swap = { A: "B" };
    f.value = f.value.slice(0, i) + (swap[f.value[i]] || "A") + f.value.slice(i + 1);
  });
};
const tampered = await attemptLogin(username, PASSWORD, OWNER, tamper);
assert.equal(tampered.path, "/login/confirm", "tampered token must lead to the second step, got " + tampered.path);
assert.match(tampered.text, /Unusual typing/);
assert.notEqual(tokenOf(loginPosts[loginPosts.length - 1]), goodToken);
console.log("tampered token: second step");
await page.goto(SITE + "/account");
assert.equal(new URL(page.url()).pathname, "/login", "second step page must not be a login");

// ---- stale token (already used once): second step ------------------------------------------------------
const stale = (t) => {
  const form = document.getElementById("login-form");
  form.addEventListener("submit", () => {
    form.querySelector('input[name="tempo_token"]').value = t;
  });
};
const replayed = await attemptLogin(username, PASSWORD, OWNER, stale, goodToken);
assert.equal(replayed.path, "/login/confirm", "a used token must not log in, got " + replayed.path);
console.log("stale (already used) token: second step");

// ---- a different typist (right password, other rhythm): second step -------------------------------------
let flagged = 0;
for (let i = 1; i <= IMPOSTER_TRIES; i++) {
  const r = await attemptLogin(username, PASSWORD, IMPOSTER);
  if (r.path === "/login/confirm") flagged += 1;
  console.log("different typist try " + i + ":", r.path);
}
assert.ok(flagged >= IMPOSTER_MIN_FLAGGED, "different typist reached the second step " + flagged + "/" + IMPOSTER_TRIES + " times");
console.log("different typist: " + flagged + "/" + IMPOSTER_TRIES + " sent to the second step");

// ---- privacy and secrecy ----------------------------------------------------------------------------------
const verifies = tempoPosts.filter((p) => p.url === TEMPO + "/v1/verify");
const enrolls = tempoPosts.filter((p) => p.url === TEMPO + "/v1/enroll");
assert.ok(enrolls.length >= 10 && verifies.length >= 5, "posts to tempo: enroll " + enrolls.length + ", verify " + verifies.length);
for (const p of tempoPosts) {
  const j = JSON.parse(p.body);
  assert.ok(Object.keys(j).every((k) => ["sitekey", "user_id", "ticket", "events"].includes(k)), "unexpected field: " + Object.keys(j));
  assert.match(j.user_id, /^u_[0-9a-f]{32}$/);
  assert.ok(!j.user_id.includes(username));
  for (const e of j.events) {
    assert.ok(/^(x|Enter|Shift|Control|Alt|Meta)$/.test(e.key) && /^k\d+$/.test(e.code), "leaked key: " + JSON.stringify(e));
  }
  for (const piece of ["tie5", "Roanl", "Roanx", username]) assert.ok(!p.body.includes(piece), "typed text in a tempo request");
}
console.log("sample tempo request:", verifies[0].body.slice(0, 170));
const answers = bodies.filter((b) => b.url.startsWith(TEMPO) && /json/.test(b.type));
assert.ok(answers.length >= 15, "tempo JSON answers seen: " + answers.length + " (enroll " + enrolls.length + ", verify " + verifies.length + ")");
for (const a of answers) {
  assert.ok(!/"(accepted|score|threshold|rhythm_match|risk)"/.test(a.text), "browser saw a verdict field: " + a.text.slice(0, 120));
}
for (const a of answers.filter((b) => b.url.endsWith("/v1/verify"))) {
  assert.deepEqual(Object.keys(JSON.parse(a.text)), ["token"]);
}
for (const b of bodies) {
  assert.ok(!b.text.includes(SECRET) && !b.text.includes(SECRET_PART), "site secret appeared in " + b.url);
}
for (const p of [...tempoPosts.map((k) => k.body), ...loginPosts]) assert.ok(!p.includes(SECRET_PART), "site secret in a request body");
assert.ok(!(await page.evaluate(() => document.cookie)).includes(SECRET_PART));
const loginHtml = bodies.find((b) => b.url === SITE + "/login" && b.type.includes("html"));
assert.ok(loginHtml.text.includes('data-sitekey="pk_'), "page carries the public site key");
assert.ok(!/<script(?![^>]*\bsrc=)[^>]*>/i.test(loginHtml.text), "inline script in the login page");
console.log("privacy: tempo requests anonymized; browser never saw accepted/score; secret never in any page, script or request");

await ctx.close();
await browser.close();
assert.deepEqual(problems, [], problems.join("\n"));
console.log("INTEGRATION E2E OK");

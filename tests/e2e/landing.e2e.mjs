// Real-browser check of the landing page (TEMPO_LANDING=1). Not part of pytest. Run:
//   TEMPO_LANDING=1 TEMPO_DB=/tmp/landing.db TEMPO_PORT=8765 python -m tempo.api &
//   E2E_URL=http://127.0.0.1:8765 E2E_SHOT=/tmp/landing-shots node tests/e2e/landing.e2e.mjs
// E2E_SHOT is a directory; it gets landing-{light,dark}-{desktop,mobile}.png. Needs playwright (npm) and its Chromium.
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
const { chromium } = createRequire(import.meta.url)("playwright");

const URL_BASE = process.env.E2E_URL || "http://127.0.0.1:8765";
const SHOT_DIR = process.env.E2E_SHOT || "landing-shots";
const PHRASE = ".tie5Roanl";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rand = (a, b) => a + Math.random() * (b - a);
fs.mkdirSync(SHOT_DIR, { recursive: true });

// a fixed per-key rhythm (ms) plus jitter, so attempts are similar but never identical
const base = [...PHRASE, "Enter"].map((_, i) => ({ hold: 80 + ((i * 17) % 30), gap: 110 + ((i * 29) % 60) }));
async function typeKeys(page, keys, jitter = 12) {
  for (let i = 0; i < keys.length; i++) {
    const b = base[i % base.length];
    await page.keyboard.down(keys[i]);
    await sleep(b.hold + rand(-jitter, jitter));
    await page.keyboard.up(keys[i]);
    await sleep(b.gap + rand(-jitter, jitter));
  }
}
const phrase = () => [...PHRASE, "Enter"];

const browser = await chromium.launch();
const problems = [];
const posted = [];

function watch(page) {
  page.on("console", (m) => {
    // Chromium logs every 4xx as a console error; those are checked on the response below (CSP violations never match this)
    if ((m.type() === "error" && !m.text().startsWith("Failed to load resource")) || m.text().includes("Content Security Policy")) problems.push("console: " + m.text());
  });
  page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
  page.on("response", (r) => {
    // expected: GET /status 404 (new user) and DELETE /user 401 (no admin token on this server)
    const expected = (r.status() === 404 && r.url().includes("/status?user_id=")) || (r.status() === 401 && r.request().method() === "DELETE");
    if (r.status() >= 400 && !expected) problems.push("http " + r.status() + " " + r.url());
  });
  page.on("request", (rq) => {
    if (!rq.url().startsWith(URL_BASE)) problems.push("external request: " + rq.url());
    if (rq.method() === "POST") posted.push(rq.postData() || "");
  });
}

async function shot(scheme, name, viewport, fullPage) {
  const ctx = await browser.newContext({ viewport, colorScheme: scheme, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  watch(page);
  await page.goto(URL_BASE + "/");
  await page.waitForFunction(() => document.querySelector("#status-line").textContent.includes("not enrolled"));
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  assert.ok(overflow <= 0, name + ": horizontal overflow of " + overflow + "px");
  await page.screenshot({ path: path.join(SHOT_DIR, "landing-" + scheme + "-" + name + ".png"), fullPage });
  await ctx.close();
}

// ---- main flow: enroll 10, log in, by keyboard only, desktop light ------------
const ctx = await browser.newContext({ viewport: { width: 1100, height: 900 }, colorScheme: "light" });
const page = await ctx.newPage();
watch(page);
await page.goto(URL_BASE + "/");
assert.match(await page.title(), /Tempo/);
assert.equal(await page.textContent("h1"), "Log in by the rhythm of your typing");
await page.waitForFunction(() => document.querySelector("#status-line").textContent.includes("not enrolled"));
const user = await page.inputValue("#user");
assert.match(user, /^visitor-[0-9a-f]{8}$/);
console.log("status:", await page.textContent("#status-line"));

// keyboard only: Tab from the page start reaches the typing field (skip link, nav links, name field, then the field)
await page.keyboard.press("Tab");
assert.equal(await page.evaluate(() => document.activeElement.className), "skip");
await page.keyboard.press("Enter"); // skip link -> #try
await page.focus("#typing");
for (let n = 1; n <= 10; n++) {
  await typeKeys(page, phrase());
  await page.waitForFunction((k) => document.querySelector("#progress").value >= k || document.querySelector("#progress-wrap").hidden, n, { timeout: 5000 });
  await sleep(150);
}
await page.waitForFunction(() => /Enrollment complete/.test(document.querySelector("#result").textContent), null, { timeout: 5000 });
console.log("enroll:", await page.textContent("#result"));
assert.match(await page.textContent("#type-title"), /Log in/);
assert.equal(await page.getAttribute("#result", "aria-live"), "polite");

await page.focus("#typing");
await typeKeys(page, phrase());
await page.waitForFunction(() => /Accepted|Rejected/.test(document.querySelector("#result").textContent), null, { timeout: 5000 });
const login = await page.textContent("#result");
console.log("login:", login);
await page.screenshot({ path: path.join(SHOT_DIR, "landing-light-desktop-after-login.png") });

// delete: public server has no admin token -> an explanation, not a crash
page.once("dialog", (d) => d.accept());
await page.click("#delete");
await page.waitForFunction(() => /Deleting needs the operator|Your data was deleted/.test(document.querySelector("#result").textContent), null, { timeout: 5000 });
console.log("delete:", await page.textContent("#result"));

// privacy: posted JSON carries anonymized key names only
assert.ok(posted.length >= 11, "posted " + posted.length);
for (const body of posted) {
  const j = JSON.parse(body);
  assert.deepEqual(Object.keys(j).sort(), ["events", "user_id"]);
  for (const e of j.events) {
    assert.ok(/^(x|Enter|Shift|Control|Alt|Meta)$/.test(e.key) && /^k\d+$/.test(e.code), "leaked key: " + JSON.stringify(e));
  }
  assert.ok(!body.includes("tie5Roanl"));
}
await ctx.close();

// ---- screenshots + mobile overflow ---------------------------------------------
await shot("light", "desktop", { width: 1280, height: 900 }, true);
await shot("dark", "desktop", { width: 1280, height: 900 }, true);
await shot("light", "mobile", { width: 360, height: 740 }, true);
await shot("dark", "mobile", { width: 360, height: 740 }, true);

// the plain demo page is still at /demo
const dctx = await browser.newContext();
const dpage = await dctx.newPage();
watch(dpage);
await dpage.goto(URL_BASE + "/demo");
assert.match(await dpage.textContent("h1"), /demo/);
await dctx.close();

await browser.close();
assert.deepEqual(problems, [], problems.join("\n"));
let outcome = "rejected";
if (login.startsWith("Accepted")) {
  outcome = "accepted";
}
console.log("E2E OK (" + outcome + ")");

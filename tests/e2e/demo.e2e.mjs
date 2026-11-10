// Real-browser check of the demo page. Not part of pytest. Run (server already started):
//   KEYAUTH_DB=/tmp/e2e.db KEYAUTH_DEBUG_SCORES=1 KEYAUTH_PORT=8765 python -m keyauth.api &
//   E2E_URL=http://127.0.0.1:8765 E2E_SHOT=/tmp/demo.png node tests/e2e/demo.e2e.mjs
import { createRequire } from "node:module";
import assert from "node:assert/strict";
const { chromium } = createRequire(import.meta.url)("playwright");

const URL_BASE = process.env.E2E_URL || "http://127.0.0.1:8765";
const SHOT = process.env.E2E_SHOT || "demo.png";
const PHRASE = ".tie5Roanl";
const user = "e2e" + Math.random().toString(36).slice(2, 8);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rand = (a, b) => a + Math.random() * (b - a);

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
const ctx = await browser.newContext({ viewport: { width: 900, height: 900 }, deviceScaleFactor: 1 });
const page = await ctx.newPage();
const problems = [];
page.on("console", (m) => {
  if (m.type() === "error" && !m.text().startsWith("Failed to load resource")) problems.push("console: " + m.text());
});
page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
// Chromium logs every 4xx as a console error; only GET /status 404 (new user) is expected
page.on("response", (r) => {
  if (r.status() >= 400 && !(r.status() === 404 && r.url().includes("/status?user_id="))) {
    problems.push("http " + r.status() + " " + r.url());
  }
});
const posted = [];
page.on("request", (rq) => {
  if (rq.method() === "POST") posted.push(rq.postData() || "");
});

await page.goto(URL_BASE + "/");
await page.fill("#user", user);
await page.click("#check");
await page.waitForSelector("#type-section:not([hidden])");
await page.waitForFunction(() => document.querySelector("#status-line").textContent.includes("not enrolled"));
console.log("status:", await page.textContent("#status-line"));

// enroll
await page.focus("#typing");
for (let n = 1; n <= 10; n++) {
  await typeKeys(page, phrase());
  await page.waitForFunction((k) => document.querySelector("#progress").value >= k, n, { timeout: 5000 });
  await sleep(150);
}
assert.match(await page.textContent("#result"), /Enrollment complete/);
assert.equal(await page.textContent("#progress-text"), "10 / 10");
console.log("enroll:", await page.textContent("#result"));

// login
await page.click("#debug");
await page.click("#mode-login");
await page.focus("#typing");
await typeKeys(page, phrase());
await page.waitForFunction(() => /Accepted|Rejected/.test(document.querySelector("#result").textContent), null, { timeout: 5000 });
const login = await page.textContent("#result");
const dbg = await page.textContent("#debug-info");
assert.match(dbg, /score -?\d/, "debug score shown: " + dbg);
console.log("login:", login, "|", dbg);
await page.screenshot({ path: SHOT });

// wrong password (one wrong char, same length)
await page.focus("#typing");
await typeKeys(page, [...".tie5Roanx", "Enter"]);
await page.waitForFunction(() => /Wrong password/.test(document.querySelector("#result").textContent), null, { timeout: 5000 });
console.log("wrong:", await page.textContent("#result"));

// Backspace reset
await page.focus("#typing");
await typeKeys(page, [".", "t", "Backspace"]);
await page.waitForFunction(() => /Start over/.test(document.querySelector("#result").textContent), null, { timeout: 5000 });
console.log("backspace:", await page.textContent("#result"));

// privacy: posted JSON may only contain anonymized key names
assert.ok(posted.length >= 11, "posted " + posted.length);
for (const body of posted) {
  const j = JSON.parse(body);
  assert.deepEqual(Object.keys(j).sort(), ["events", "user_id"]);
  for (const e of j.events) {
    assert.ok(/^(x|Enter|Shift|Control|Alt|Meta)$/.test(e.key) && /^k\d+$/.test(e.code), "leaked key: " + JSON.stringify(e));
  }
  assert.ok(!body.includes("tie5Roanl"));
}
console.log("sample event:", posted[0].slice(0, 160));

await browser.close();
assert.deepEqual(problems, [], problems.join("\n"));
let outcome = "rejected";
if (login.startsWith("Accepted")) {
  outcome = "accepted";
}
console.log("E2E OK (" + outcome + ")");

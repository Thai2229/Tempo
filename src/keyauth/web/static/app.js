// keyauth demo page: username -> enroll (N samples) -> log in. Same-origin API calls only.
import { PHRASE, Recorder, isValidUserId } from "./capture.js";

const $ = (id) => document.getElementById(id);
const el = {
  user: $("user"),
  check: $("check"),
  userError: $("user-error"),
  statusLine: $("status-line"),
  modeEnroll: $("mode-enroll"),
  modeLogin: $("mode-login"),
  del: $("delete"),
  typeSection: $("type-section"),
  typeTitle: $("type-title"),
  phrase: $("phrase"),
  progressWrap: $("progress-wrap"),
  progress: $("progress"),
  progressText: $("progress-text"),
  typing: $("typing"),
  result: $("result"),
  debug: $("debug"),
  debugInfo: $("debug-info"),
};

const state = {
  userId: null, // validated username the page is working with
  mode: null, // "enroll" | "login"
  required: 10, // replaced by the API's `required` / `enroll_required`
  samples: 0,
  busy: false,
  countdown: null, // interval id
  lastDebug: null, // {score, threshold} from the last /verify, if the API sent them
  debugSeen: false, // true once a /verify response was handled
};

const recorder = new Recorder(PHRASE);
el.phrase.textContent = PHRASE;

// ---- API ---------------------------------------------------------------

async function api(method, path, body, headers) {
  const init = { method, headers: Object.assign({}, headers) };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers["Content-Type"] = "application/json";
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    return { ok: false, status: 0, data: null, code: "network", retryAfter: 0 };
  }
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    data = null;
  }
  let code = "";
  if (data && data.error && data.error.code) {
    code = data.error.code;
  }
  const retryAfter = parseInt(res.headers.get("Retry-After"), 10);
  if (Number.isNaN(retryAfter)) {
    return { ok: res.ok, status: res.status, data, code, retryAfter: 0 };
  }
  return { ok: res.ok, status: res.status, data, code, retryAfter };
}

// ---- rendering ---------------------------------------------------------

function show(kind, title, detail) {
  el.result.className = "result " + kind;
  el.result.textContent = "";
  const strong = document.createElement("strong");
  strong.textContent = title;
  el.result.appendChild(strong);
  if (detail) {
    el.result.appendChild(document.createTextNode(detail));
  }
}

function clearResult() {
  el.result.className = "result";
  el.result.textContent = "";
}

function renderDebug() {
  if (!el.debug.checked) {
    el.debugInfo.hidden = true;
    return;
  }
  el.debugInfo.hidden = false;
  if (state.lastDebug) {
    el.debugInfo.textContent = "score " + state.lastDebug.score.toFixed(4) + " / threshold " + state.lastDebug.threshold.toFixed(4) + " (lower score = more similar; accepted when score <= threshold)";
  } else if (state.debugSeen) {
    el.debugInfo.textContent = "The server did not return a score (it needs KEYAUTH_DEBUG_SCORES=1).";
  } else {
    el.debugInfo.textContent = "No login attempt yet.";
  }
}

function setProgress(samples, required) {
  state.samples = samples;
  state.required = required;
  el.progress.max = required;
  el.progress.value = samples;
  el.progressText.textContent = samples + " / " + required;
}

function setMode(mode) {
  state.mode = mode;
  el.modeEnroll.setAttribute("aria-pressed", String(mode === "enroll"));
  el.modeLogin.setAttribute("aria-pressed", String(mode === "login"));
  el.typeSection.hidden = false;
  el.progressWrap.hidden = mode !== "enroll";
  if (mode === "enroll") {
    el.typeTitle.textContent = "Enroll: type the phrase " + state.required + " times";
    setProgress(state.samples, state.required);
  } else {
    el.typeTitle.textContent = "Log in: type the phrase";
  }
  clearResult();
  resetTyping();
  el.typing.focus();
}

function resetTyping() {
  recorder.reset();
  el.typing.value = "";
}

function setBusy(busy) {
  state.busy = busy;
  el.typing.disabled = busy || state.countdown !== null;
  el.check.disabled = busy;
  el.del.disabled = busy;
}

// ---- countdown (locked / rate limited) -----------------------------------

function stopCountdown() {
  if (state.countdown !== null) {
    clearInterval(state.countdown);
    state.countdown = null;
  }
  el.typing.disabled = state.busy;
}

function startCountdown(seconds, kind, title, text) {
  stopCountdown();
  let left = Math.max(1, seconds);
  el.typing.disabled = true;
  const paint = () => {
    show(kind, title, text + " Try again in " + left + " s.");
  };
  paint();
  state.countdown = setInterval(() => {
    left -= 1;
    if (left <= 0) {
      stopCountdown();
      show("warn", "You can try again now.", "");
      el.typing.focus();
      return;
    }
    paint();
  }, 1000);
}

// ---- username + status -----------------------------------------------------

function readUser() {
  const value = el.user.value.trim();
  if (!isValidUserId(value)) {
    el.userError.hidden = false;
    el.userError.textContent = "Use 1 to 64 characters: letters, digits, . _ @ -";
    el.user.setAttribute("aria-invalid", "true");
    return null;
  }
  el.userError.hidden = true;
  el.user.removeAttribute("aria-invalid");
  return value;
}

function apiProblem(r) {
  if (r.status === 0) {
    show("bad", "Cannot reach the server.", "Check your connection and try again.");
    return true;
  }
  if (r.code === "model_unavailable") {
    show("bad", "The model is not loaded on the server.", "The administrator needs to train or install a checkpoint.");
    return true;
  }
  if (r.code === "rate_limited") {
    startCountdown(r.retryAfter, "warn", "Too many requests.", "Slow down.");
    return true;
  }
  return false;
}

async function checkUser() {
  const userId = readUser();
  if (userId === null) {
    return;
  }
  if (userId !== state.userId) {
    state.samples = 0;
    state.lastDebug = null;
    state.debugSeen = false;
    renderDebug();
  }
  state.userId = userId;
  stopCountdown();
  setBusy(true);
  const r = await api("GET", "/status?user_id=" + encodeURIComponent(userId));
  setBusy(false);
  el.del.hidden = true;
  if (r.ok) {
    const d = r.data;
    state.required = d.enroll_required;
    state.samples = d.enroll_samples;
    el.del.hidden = false;
    if (d.status === "active") {
      let text = userId + " is enrolled.";
      if (d.locked) {
        text += " Currently locked after too many failed attempts.";
      }
      el.statusLine.textContent = text;
      setMode("login");
    } else {
      el.statusLine.textContent = userId + " is partway through enrollment: " + d.enroll_samples + " of " + d.enroll_required + " samples.";
      setMode("enroll");
    }
    return;
  }
  if (r.code === "not_enrolled") {
    state.samples = 0;
    el.statusLine.textContent = userId + " is not enrolled yet. Enroll first.";
    setMode("enroll");
    return;
  }
  el.statusLine.textContent = "";
  if (!apiProblem(r)) {
    show("bad", "Something went wrong.", "The server answered " + r.status + " " + r.code + ".");
  }
}

// ---- attempts --------------------------------------------------------------

async function submitEnroll(events) {
  const r = await api("POST", "/enroll", { user_id: state.userId, events });
  if (r.ok) {
    const d = r.data;
    setProgress(d.samples, d.required);
    if (d.status === "complete") {
      el.statusLine.textContent = state.userId + " is enrolled.";
      el.del.hidden = false;
      show("ok", "Enrollment complete.", " Now switch to Log in and type the phrase the same way.");
      return;
    }
    if (d.status === "rejected" && d.reason === "inconsistent") {
      setProgress(0, d.required);
      show("warn", "Those samples were too different from each other.", " Enrollment restarted from 0. Type at a steady, natural pace and try again.");
      return;
    }
    show("ok", "Sample " + d.samples + " of " + d.required + " saved.", " Type the phrase again.");
    return;
  }
  if (apiProblem(r)) {
    return;
  }
  if (r.code === "already_enrolled") {
    show("warn", "Already enrolled.", " Use Log in, or delete your data to enroll again.");
  } else if (r.code === "replay") {
    show("warn", "That sample was identical to an earlier one.", " Type it again naturally; do not replay recorded input.");
  } else if (r.code === "length_mismatch") {
    show("warn", "That sample had a different number of keys.", " Start over and type the phrase without corrections.");
  } else if (r.code === "invalid_timings" || r.code === "validation_error") {
    show("warn", "That attempt could not be used.", " Start over and type the phrase in one go.");
  } else {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
}

async function submitLogin(events) {
  const r = await api("POST", "/verify", { user_id: state.userId, events });
  state.debugSeen = true;
  state.lastDebug = null;
  if (r.ok) {
    const d = r.data;
    if (typeof d.score === "number" && typeof d.threshold === "number") {
      state.lastDebug = { score: d.score, threshold: d.threshold };
    }
    renderDebug();
    if (d.accepted) {
      show("ok", "Accepted.", " Your password and your typing rhythm both matched.");
    } else if (d.retry_after_s) {
      startCountdown(d.retry_after_s, "bad", "Rejected, and now locked.", "Too many failed attempts.");
    } else {
      show("bad", "Rejected.", " The password was right but the rhythm did not match. Try again.");
    }
    return;
  }
  renderDebug();
  if (r.code === "locked") {
    startCountdown(r.retryAfter, "bad", "Locked.", "Too many failed attempts.");
  } else if (apiProblem(r)) {
    return;
  } else if (r.code === "not_enrolled") {
    state.samples = 0;
    el.statusLine.textContent = state.userId + " is not enrolled yet. Enroll first.";
    show("warn", "Not enrolled.", " Choose Enroll to record your rhythm first.");
  } else if (r.code === "replay") {
    show("bad", "Rejected: identical to an earlier attempt.", " Replayed input is not accepted.");
  } else if (r.code === "length_mismatch") {
    show("bad", "Rejected: wrong number of keys.", " Start over and type the phrase without corrections.");
  } else if (r.code === "scorer_error") {
    show("bad", "Your stored template is unusable.", " Delete your data and enroll again.");
  } else {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
}

async function handleResult(res) {
  if (res.status === "reset") {
    el.typing.value = "";
    if (res.reason === "wrong_password") {
      show("warn", "Wrong password.", " Type " + PHRASE + " exactly, then Enter. Start over.");
    } else if (res.reason === "incomplete") {
      show("warn", "Release all keys before ending.", " Start over.");
    } else {
      show("warn", "Start over.", " Corrections, pasting or leaving the field make the rhythm incomparable.");
    }
    return;
  }
  if (res.status !== "done") {
    return;
  }
  el.typing.value = "";
  if (state.userId === null || state.mode === null) {
    return;
  }
  setBusy(true);
  if (state.mode === "enroll") {
    await submitEnroll(res.events);
  } else {
    await submitLogin(res.events);
  }
  setBusy(false);
  if (state.countdown === null) {
    el.typing.focus();
  }
}

// ---- events ------------------------------------------------------------

el.typing.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault(); // no implicit form submit; Enter is part of the rhythm
  }
  if (state.busy || state.countdown !== null) {
    return;
  }
  const res = recorder.keyDown(e.key, e.code, performance.now(), e.repeat);
  handleResult(res);
});

el.typing.addEventListener("keyup", (e) => {
  if (state.busy || state.countdown !== null) {
    return;
  }
  const res = recorder.keyUp(e.key, e.code, performance.now());
  handleResult(res);
});

for (const name of ["paste", "cut", "drop"]) {
  el.typing.addEventListener(name, (e) => {
    e.preventDefault();
    handleResult(recorder.interrupt());
  });
}

el.typing.addEventListener("blur", () => {
  if (recorder.length > 0) {
    handleResult(recorder.interrupt());
  }
});

el.check.addEventListener("click", checkUser);
el.user.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    checkUser();
  }
});
el.modeEnroll.addEventListener("click", async () => {
  if (state.userId === null || state.userId !== el.user.value.trim()) {
    await checkUser();
  }
  if (state.userId !== null) {
    setMode("enroll");
  }
});
el.modeLogin.addEventListener("click", async () => {
  if (state.userId === null || state.userId !== el.user.value.trim()) {
    await checkUser();
  }
  if (state.userId !== null) {
    setMode("login");
  }
});
el.debug.addEventListener("change", renderDebug);

el.del.addEventListener("click", async () => {
  if (state.userId === null) {
    return;
  }
  if (!window.confirm("Delete all stored rhythm data for " + state.userId + "?")) {
    return;
  }
  setBusy(true);
  const r = await api("DELETE", "/user?user_id=" + encodeURIComponent(state.userId));
  setBusy(false);
  if (r.ok) {
    stopCountdown();
    state.samples = 0;
    el.del.hidden = true;
    el.statusLine.textContent = state.userId + " is not enrolled yet. Enroll first.";
    setMode("enroll");
    show("ok", "Your data was deleted.", "");
  } else if (r.status === 401) {
    show("warn", "Deleting needs an admin token.", " This server restricts deletion; ask its administrator.");
  } else if (r.code === "not_enrolled") {
    show("warn", "Nothing to delete.", " That user is not enrolled.");
  } else if (!apiProblem(r)) {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
});

renderDebug();

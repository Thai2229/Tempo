// Tempo landing page: try-it widget. Same capture logic as the demo page (capture.js), same-origin API calls only.
// The demo name is random and filled in; the widget works out by itself whether to enroll or to log in.
import { PHRASE, Recorder, isValidUserId } from "./capture.js";

const $ = (id) => document.getElementById(id);
const el = {
  user: $("user"),
  userError: $("user-error"),
  statusLine: $("status-line"),
  progressWrap: $("progress-wrap"),
  progress: $("progress"),
  progressText: $("progress-text"),
  typeTitle: $("type-title"),
  phrase: $("phrase"),
  typing: $("typing"),
  result: $("result"),
  del: $("delete"),
};

const state = {
  userId: null, // validated name the widget is working with
  mode: null, // "enroll" | "login" | null while unknown
  required: 10,
  samples: 0,
  busy: false,
  countdown: null, // interval id (locked / rate limited)
};

const recorder = new Recorder(PHRASE);
el.phrase.textContent = PHRASE;

function randomName() {
  const bytes = new Uint8Array(4);
  crypto.getRandomValues(bytes);
  let hex = "";
  for (const b of bytes) {
    hex += b.toString(16).padStart(2, "0");
  }
  return "visitor-" + hex;
}

// ---- API ---------------------------------------------------------------

async function api(method, path, body) {
  const init = { method, headers: {} };
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
  let retryAfter = parseInt(res.headers.get("Retry-After"), 10);
  if (Number.isNaN(retryAfter)) {
    retryAfter = 0;
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

function setProgress(samples, required) {
  state.samples = samples;
  state.required = required;
  el.progress.max = required;
  el.progress.value = samples;
  el.progressText.textContent = samples + " / " + required;
}

function setBusy(busy) {
  state.busy = busy;
  el.typing.disabled = busy || state.countdown !== null || state.mode === null;
  el.del.disabled = busy;
}

function setMode(mode) {
  state.mode = mode;
  el.progressWrap.hidden = mode !== "enroll";
  if (mode === "enroll") {
    el.typeTitle.textContent = "Enroll: type the phrase " + state.required + " times";
    setProgress(state.samples, state.required);
  } else {
    el.typeTitle.textContent = "Log in: type the phrase";
  }
  recorder.reset();
  el.typing.value = "";
  setBusy(state.busy);
}

// ---- countdown (locked / rate limited) -----------------------------------

function stopCountdown() {
  if (state.countdown !== null) {
    clearInterval(state.countdown);
    state.countdown = null;
  }
  setBusy(state.busy);
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

function apiProblem(r) {
  if (r.status === 0) {
    show("bad", "Cannot reach the server.", " Check your connection and try again.");
    return true;
  }
  if (r.code === "model_unavailable") {
    show("bad", "The model is not loaded on the server.", " The operator needs to install a checkpoint.");
    return true;
  }
  if (r.code === "rate_limited") {
    startCountdown(r.retryAfter, "warn", "Too many requests.", "Slow down.");
    return true;
  }
  return false;
}

// ---- name + status -----------------------------------------------------------

async function loadStatus() {
  const value = el.user.value.trim();
  if (!isValidUserId(value)) {
    el.userError.hidden = false;
    el.userError.textContent = "Use 1 to 64 characters: letters, digits, . _ @ -";
    el.user.setAttribute("aria-invalid", "true");
    state.userId = null;
    state.mode = null;
    el.statusLine.textContent = "";
    el.del.hidden = true;
    setBusy(state.busy);
    return;
  }
  el.userError.hidden = true;
  el.user.removeAttribute("aria-invalid");
  if (value !== state.userId) {
    state.samples = 0;
    clearResult();
  }
  state.userId = value;
  state.mode = null;
  stopCountdown();
  setBusy(true);
  const r = await api("GET", "/status?user_id=" + encodeURIComponent(value));
  if (value !== state.userId) {
    return; // the name changed while waiting
  }
  setBusy(false);
  if (r.ok) {
    const d = r.data;
    state.required = d.enroll_required;
    state.samples = d.enroll_samples;
    el.del.hidden = false;
    if (d.status === "active") {
      let text = value + " is enrolled. Type the phrase to log in.";
      if (d.locked) {
        text = value + " is enrolled but locked after too many failed attempts.";
      }
      el.statusLine.textContent = text;
      setMode("login");
    } else {
      el.statusLine.textContent = value + " is partway through enrollment: " + d.enroll_samples + " of " + d.enroll_required + " samples.";
      setMode("enroll");
    }
    return;
  }
  if (r.code === "not_enrolled") {
    state.samples = 0;
    el.del.hidden = true;
    el.statusLine.textContent = value + " is not enrolled yet. Type the phrase " + state.required + " times to enroll.";
    setMode("enroll");
    return;
  }
  el.statusLine.textContent = "";
  if (!apiProblem(r)) {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
}

// ---- attempts --------------------------------------------------------------

async function submitEnroll(events) {
  const r = await api("POST", "/enroll", { user_id: state.userId, events });
  if (r.ok) {
    const d = r.data;
    setProgress(d.samples, d.required);
    el.del.hidden = false;
    if (d.status === "complete") {
      el.statusLine.textContent = state.userId + " is enrolled. Type the phrase to log in.";
      setMode("login");
      show("ok", "Enrollment complete.", " Now type the phrase once more to log in.");
      return;
    }
    if (d.status === "rejected" && d.reason === "inconsistent") {
      setProgress(0, d.required);
      show("warn", "Those samples were too different from each other.", " Enrollment restarted from 0. Type at a steady, natural pace.");
      return;
    }
    show("ok", "Sample " + d.samples + " of " + d.required + " saved.", " Type the phrase again.");
    return;
  }
  if (apiProblem(r)) {
    return;
  }
  if (r.code === "already_enrolled") {
    show("warn", "Already enrolled.", " Reload the page for a new demo name.");
  } else if (r.code === "replay") {
    show("warn", "That sample was identical to an earlier one.", " Type it again naturally.");
  } else if (r.code === "length_mismatch") {
    show("warn", "That sample had a different number of keys.", " Type the phrase without corrections.");
  } else if (r.code === "invalid_timings" || r.code === "validation_error") {
    show("warn", "That attempt could not be used.", " Type the phrase in one go.");
  } else {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
}

async function submitLogin(events) {
  const r = await api("POST", "/verify", { user_id: state.userId, events });
  if (r.ok) {
    const d = r.data;
    if (d.accepted) {
      show("ok", "Accepted.", " Your typing rhythm matched.");
    } else if (d.retry_after_s) {
      startCountdown(d.retry_after_s, "bad", "Rejected, and now locked.", "Too many failed attempts.");
    } else {
      show("bad", "Rejected.", " The rhythm did not match. Try again.");
    }
    return;
  }
  if (r.code === "locked") {
    startCountdown(r.retryAfter, "bad", "Locked.", "Too many failed attempts.");
  } else if (apiProblem(r)) {
    return;
  } else if (r.code === "not_enrolled") {
    show("warn", "Not enrolled.", " Reload the page to start again.");
  } else if (r.code === "replay") {
    show("bad", "Rejected: identical to an earlier attempt.", " Replayed input is not accepted.");
  } else if (r.code === "length_mismatch") {
    show("bad", "Rejected: wrong number of keys.", " Type the phrase without corrections.");
  } else if (r.code === "scorer_error") {
    show("bad", "Your stored template is unusable.", " Reload the page for a new demo name.");
  } else {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
}

async function handleResult(res) {
  if (res.status === "reset") {
    el.typing.value = "";
    if (res.reason === "wrong_password") {
      show("warn", "Wrong phrase.", " Type " + PHRASE + " exactly, then Enter.");
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
    e.preventDefault(); // Enter is part of the rhythm, not a form submit
  }
  if (state.busy || state.countdown !== null || state.mode === null) {
    return;
  }
  handleResult(recorder.keyDown(e.key, e.code, performance.now(), e.repeat));
});

el.typing.addEventListener("keyup", (e) => {
  if (state.busy || state.countdown !== null || state.mode === null) {
    return;
  }
  handleResult(recorder.keyUp(e.key, e.code, performance.now()));
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

el.user.addEventListener("change", loadStatus);
el.user.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    loadStatus();
    el.typing.focus();
  }
});

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
    el.statusLine.textContent = state.userId + " is not enrolled yet. Type the phrase " + state.required + " times to enroll.";
    setMode("enroll");
    show("ok", "Your data was deleted.", "");
  } else if (r.status === 401) {
    show("warn", "Deleting needs the operator.", " This server restricts deletion to an admin token. Ask its operator to delete " + state.userId + ".");
  } else if (r.code === "not_enrolled") {
    show("warn", "Nothing to delete.", " That name is not enrolled.");
  } else if (!apiProblem(r)) {
    show("bad", "Something went wrong.", " The server answered " + r.status + " " + r.code + ".");
  }
});

el.user.value = randomName();
loadStatus();

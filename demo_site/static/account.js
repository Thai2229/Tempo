// Account page: enroll the typing rhythm with a ticket that OUR server issues to this logged-in session.
(function () {
  "use strict";
  const sitekey = document.body.dataset.sitekey;
  const start = document.getElementById("start-enroll");
  const area = document.getElementById("enroll-area");
  const pw = document.getElementById("enroll-pw");
  const progress = document.getElementById("enroll-progress");
  const status = document.getElementById("enroll-status");
  const message = document.getElementById("enroll-message");
  let reset = false;

  async function requestTicket() {
    const res = await fetch("/account/ticket", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset: reset }),
      credentials: "same-origin",
    });
    let data = {};
    try {
      data = await res.json();
    } catch (e) {
      data = {};
    }
    return { status: res.status, data: data };
  }

  async function run() {
    start.disabled = true;
    message.textContent = "";
    const t = await requestTicket();
    if (t.status === 409) {
      reset = true;
      start.disabled = false;
      start.textContent = "Replace my enrolled rhythm";
      message.textContent = "Already enrolled. Click again to enroll from scratch.";
      return;
    }
    if (t.status !== 200 || typeof t.data.ticket !== "string") {
      start.disabled = false;
      message.textContent = "Could not start enrollment (" + t.status + ").";
      return;
    }
    reset = false;
    area.hidden = false;
    pw.value = "";
    pw.focus();
    let failures = 0;
    while (failures < 5) {
      try {
        const r = await tempo.enroll(pw, { sitekey: sitekey, ticket: t.data.ticket, user_id: t.data.user_id });
        failures = 0;
        pw.value = "";
        progress.max = r.required;
        progress.value = r.samples;
        status.textContent = r.samples + " / " + r.required;
        if (r.done) {
          message.textContent = "Enrollment complete.";
          start.textContent = "Replace my enrolled rhythm";
          start.disabled = false;
          return;
        }
      } catch (err) {
        failures += 1;
        pw.value = "";
        message.textContent = "That attempt was not usable (" + err.code + "). Type it again.";
        if (err.code === "invalid_ticket" || err.code === "network" || err.code === "timeout") {
          break;
        }
      }
    }
    message.textContent = "Enrollment stopped (" + message.textContent + ")";
    start.disabled = false;
  }

  if (typeof tempo === "undefined") {
    message.textContent = "tempo.js did not load.";
    start.disabled = true;
  } else {
    start.addEventListener("click", run);
  }
})();

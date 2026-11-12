// Login page behaviour. A same-origin file, so no inline script is needed under the strict CSP.
// 1. when the username is entered, ask OUR server which opaque id keyauth should use (same answer shape for every name)
// 2. attach keyauth to the password field; it adds the hidden "keyauth_token" field to the form
(function () {
  "use strict";
  const sitekey = document.body.dataset.sitekey;
  const username = document.getElementById("username");
  const uid = document.getElementById("keyauth-uid");
  const password = document.getElementById("password");

  async function identify() {
    const name = username.value.trim();
    uid.value = "";
    if (name === "") {
      return;
    }
    try {
      const res = await fetch("/login/identify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: name }),
        credentials: "same-origin",
      });
      const data = await res.json();
      if (res.ok && typeof data.user_id === "string") {
        uid.value = data.user_id;
      }
    } catch (e) {
      uid.value = "";
    }
  }

  username.addEventListener("change", identify);
  if (username.value !== "") {
    identify();
  }
  if (typeof keyauth !== "undefined") {
    keyauth.attach(password, { sitekey: sitekey, userIdField: "#keyauth-uid" });
  }
  // if keyauth.js did not load, the form still works: the server sees an empty token and applies its policy
})();

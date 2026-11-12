// Tempo server-side helpers for your own backend. One file, no dependencies, Node 18+ (global fetch).
//
//   import { verify, issueTicket } from "./tempo-verify.mjs";
//   const r = await verify(process.env.TEMPO_SECRET, req.body.tempo_token, "https://tempo.example.com");
//   if (r.success && r.user_id === account.tempoId && r.rhythm_match) { /* rhythm looks right */ }
//   else { /* your policy: second factor, or let the password alone through */ }
//
// Neither function throws on network trouble. If Tempo is unreachable, answers badly or says no you get
// success: false and a short `reason` ("unreachable", "unauthorized", "invalid_token", "rate_limited",
// "bad_response", "no_token", "bad_config"), so YOU choose fail-open or fail-closed. Call verify() only after the
// password was checked, compare r.user_id with the account that is logging in; a token is single use and lives 60 s.
//
// Server calls send no Origin header (Tempo refuses those), do not follow redirects (the secret is never re-sent
// elsewhere) and never log or return the secret.

const MAX_RESPONSE_CHARS = 65536;

/** @returns {Promise<{status: number, data: object|null}>} status 0 = could not be reached (refused, DNS, timeout, TLS) */
async function post(baseUrl, path, payload, timeoutMs) {
  let response;
  let text;
  try {
    response = await fetch(baseUrl.replace(/\/+$/, "") + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      redirect: "manual",
      signal: AbortSignal.timeout(timeoutMs),
    });
    text = (await response.text()).slice(0, MAX_RESPONSE_CHARS);
  } catch {
    return { status: 0, data: null };
  }
  try {
    const data = JSON.parse(text);
    if (data === null || typeof data !== "object" || Array.isArray(data)) {
      return { status: response.status, data: null };
    }
    return { status: response.status, data };
  } catch {
    return { status: response.status, data: null };
  }
}

function errorReason(status, data) {
  if (status === 0) {
    return "unreachable";
  }
  if (status === 401) {
    return "unauthorized";
  }
  if (status === 429) {
    return "rate_limited";
  }
  if (status === 409 && data !== null && data.error && data.error.code === "already_enrolled") {
    return "already_enrolled";
  }
  return "bad_response";
}

function badBaseUrl(baseUrl) {
  return typeof baseUrl !== "string" || !/^https?:\/\//.test(baseUrl);
}

/**
 * Ask Tempo what the browser's token says (POST /v1/siteverify).
 * @returns {Promise<{success: boolean, user_id: string|null, rhythm_match: boolean, risk: string|null, reason: string|null}>}
 */
export async function verify(secret, token, baseUrl, timeoutSeconds = 3.0) {
  const fail = (reason) => ({ success: false, user_id: null, rhythm_match: false, risk: null, reason });
  if (badBaseUrl(baseUrl)) {
    return fail("bad_config");
  }
  if (typeof token !== "string" || token === "") {
    return fail("no_token"); // the page did not produce a token (fail-open/closed is yours)
  }
  const { status, data } = await post(baseUrl, "/v1/siteverify", { secret, token }, timeoutSeconds * 1000);
  if (status !== 200 || data === null) {
    return fail(errorReason(status, data));
  }
  if (data.success !== true) {
    if (typeof data.error === "string") {
      return fail(data.error);
    }
    return fail("invalid_token");
  }
  if (typeof data.user_id !== "string" || typeof data.rhythm_match !== "boolean" || typeof data.risk !== "string") {
    return fail("bad_response");
  }
  return { success: true, user_id: data.user_id, rhythm_match: data.rhythm_match, risk: data.risk, reason: null };
}

/**
 * Get an enrollment ticket for an already authenticated user (POST /v1/tickets).
 * reset: true deletes the user's existing rhythm template first (re-enroll, or erase it for good).
 * @returns {Promise<{success: boolean, ticket: string|null, expires_in: number|null, reason: string|null}>}
 */
export async function issueTicket(secret, userId, baseUrl, reset = false, timeoutSeconds = 3.0) {
  const fail = (reason) => ({ success: false, ticket: null, expires_in: null, reason });
  if (badBaseUrl(baseUrl)) {
    return fail("bad_config");
  }
  const { status, data } = await post(baseUrl, "/v1/tickets", { secret, user_id: userId, reset }, timeoutSeconds * 1000);
  if (status !== 200 || data === null) {
    return fail(errorReason(status, data));
  }
  if (typeof data.ticket !== "string" || !Number.isInteger(data.expires_in)) {
    return fail("bad_response");
  }
  return { success: true, ticket: data.ticket, expires_in: data.expires_in, reason: null };
}

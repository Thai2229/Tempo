"""HTML for the demo site. Plain strings, everything user-controlled is escaped, and there is no inline script:
the pages work under script-src 'self' <keyauth origin> (see app.security_headers)."""

from __future__ import annotations

from html import escape


def layout(title: str, body: str, sitekey: str, keyauth_url: str, scripts: list[str], user: str | None = None) -> str:
    tags = []
    if scripts:
        tags.append(f'<script src="{escape(keyauth_url)}/v1/keyauth.js" defer></script>')
        for name in scripts:
            tags.append(f'<script src="/static/{escape(name)}" defer></script>')
    nav = ""
    if user is not None:
        nav = (
            f'<div class="nav">Signed in as <strong>{escape(user)}</strong> '
            '<form method="post" action="/logout" class="inline"><button id="logout">Log out</button></form></div>'
        )
    # data-sitekey is the PUBLIC key; the site secret is never passed to a template
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)} - Demo Shop</title>"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<link rel="stylesheet" href="/static/site.css"></head>'
        f'<body data-sitekey="{escape(sitekey)}"><main><h1>Demo Shop</h1>{nav}{body}</main>{"".join(tags)}</body></html>'
    )


def error_line(message: str | None) -> str:
    if not message:
        return ""
    return f'<p class="error" role="alert">{escape(message)}</p>'


def register_form(message: str | None = None) -> str:
    return (
        "<h2>Create an account</h2>" + error_line(message) +
        '<form method="post" action="/register">'
        '<label>Username <input id="username" name="username" autocomplete="username" required maxlength="64"></label>'
        '<label>Password (8 to 100 characters) <input id="password" name="password" type="password" autocomplete="new-password" required minlength="8" maxlength="100"></label>'
        "<button>Register</button></form>"
        '<p><a href="/login">I already have an account</a></p>'
    )


def login_form(message: str | None = None) -> str:
    # The hidden "keyauth-uid" input has no name: it is never posted. The server resolves the account from
    # the username and the password, and compares the opaque id inside the token with the account's own.
    return (
        "<h2>Log in</h2>" + error_line(message) +
        '<form id="login-form" method="post" action="/login">'
        '<label>Username <input id="username" name="username" autocomplete="username" required maxlength="64"></label>'
        '<label>Password <input id="password" name="password" type="password" autocomplete="current-password" required maxlength="100"></label>'
        '<input type="hidden" id="keyauth-uid" value="">'
        "<button>Log in</button></form>"
        '<p class="hint">Press Enter to submit: the rhythm of your typing includes it.</p>'
        '<p><a href="/register">Create an account</a></p>'
    )


def confirm_form(message: str | None = None) -> str:
    return (
        "<h2>Unusual typing</h2>"
        "<p>Your typing rhythm did not look like yours. Please confirm it is you with the code we emailed.</p>"
        '<p class="hint">(Demo stub: no email is sent; the code is printed in the server log.)</p>'
        + error_line(message) +
        '<form method="post" action="/login/confirm">'
        '<label>Code <input id="code" name="code" inputmode="numeric" autocomplete="one-time-code" required maxlength="6"></label>'
        "<button>Confirm</button></form>"
    )


LOGIN_NOTES = {
    "rhythm": "You are logged in. Your typing rhythm was verified.",
    "code": "You are logged in. You confirmed it with an emailed code.",
    "password_only": "You are logged in with your password only (the typing check was not available).",
    "registered": "You are logged in. Welcome!",
}


def account_body(method: str) -> str:
    note = LOGIN_NOTES.get(method, "You are logged in.")
    return (
        "<h2>Your account</h2>"
        f'<p id="login-method" data-method="{escape(method)}">{escape(note)}</p>'
        "<h3>Typing rhythm</h3>"
        "<p>Type your password and press Enter, 10 times. Only timings are sent to keyauth, never the password.</p>"
        '<button id="start-enroll" type="button">Enroll my typing rhythm</button>'
        '<div id="enroll-area" hidden>'
        '<label>Password <input id="enroll-pw" type="password" autocomplete="current-password" maxlength="100"></label>'
        '<progress id="enroll-progress" max="10" value="0"></progress> <span id="enroll-status"></span>'
        "</div>"
        '<p id="enroll-message" role="status"></p>'
    )

"""Desktop-app OAuth flow for Google Calendar: human-run once, caches a token to disk."""

# No automated test: this opens a real browser OAuth consent screen and requires a
# human-provided Google Cloud OAuth client (client_secret.json) plus interactive login —
# it cannot be exercised in CI or offline tests.
# Run manually: `uv run python scripts/auth_google.py`.

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CLIENT_SECRETS_PATH = Path("client_secret.json")


def main() -> int:
    """Run the OAuth flow (or refresh cached credentials) and write token JSON to disk."""
    token_path_env = os.environ.get("GOOGLE_CREDENTIALS_PATH")
    if not token_path_env:
        print("Set GOOGLE_CREDENTIALS_PATH to the desired token cache path first.", file=sys.stderr)
        return 1
    token_path = Path(token_path_env)

    if not CLIENT_SECRETS_PATH.exists():
        print(
            f"Missing {CLIENT_SECRETS_PATH} — download an OAuth client (Desktop app type) "
            "from the Google Cloud Console first.",
            file=sys.stderr,
        )
        return 1

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.valid:
            print(f"Existing valid credentials found at {token_path}.")
            return 0
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            print(f"Refreshed credentials, saved to {token_path}.")
            return 0

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), SCOPES)
    creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json())
    print(f"Authenticated. Credentials cached at {token_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Real-mode calendar_create_event implementation backed by the Google Calendar API."""

from __future__ import annotations

import os
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from toolsmith.tools.sandbox.calendar_create_event import (
    CalendarCreateEventArgs,
    CalendarCreateEventResult,
)


class MissingCredentialsError(RuntimeError):
    """Raised when no cached Google credentials can be found on disk."""


class CalendarCreateEventRequestError(RuntimeError):
    """Raised when the Google Calendar API request fails."""


def _load_credentials() -> Credentials:
    """Load cached Google credentials from the path in GOOGLE_CREDENTIALS_PATH.

    Only checks that the path exists; the actual file contents are read
    internally by `Credentials.from_authorized_user_file`, never by this code.
    """
    path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
    if not path or not Path(path).exists():
        raise MissingCredentialsError(
            "No cached Google credentials found. Run scripts/auth_google.py to "
            "authenticate first (see GOOGLE_CREDENTIALS_PATH env var)."
        )
    return Credentials.from_authorized_user_file(path)


def calendar_create_event_real(args: CalendarCreateEventArgs) -> CalendarCreateEventResult:
    """Create a real Google Calendar event and return its id and status."""
    creds = _load_credentials()
    service = build("calendar", "v3", credentials=creds)

    event_body = {
        "summary": args.title,
        "start": {"dateTime": args.start.isoformat(), "timeZone": args.timezone},
        "end": {"dateTime": args.end.isoformat(), "timeZone": args.timezone},
    }

    try:
        response = service.events().insert(calendarId="primary", body=event_body).execute()
    except Exception as exc:
        raise CalendarCreateEventRequestError(
            f"Google Calendar API request failed: {exc}"
        ) from exc

    return CalendarCreateEventResult(
        event_id=response["id"], status=response.get("status", "confirmed")
    )

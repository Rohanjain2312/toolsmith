"""Tests for the real-mode calendar_create_event Google Calendar client (API always mocked)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from toolsmith.tools.real import calendar_create_event as calendar_create_event_real_module
from toolsmith.tools.real.calendar_create_event import (
    CalendarCreateEventRequestError,
    MissingCredentialsError,
    calendar_create_event_real,
)
from toolsmith.tools.sandbox.calendar_create_event import CalendarCreateEventArgs


def _args() -> CalendarCreateEventArgs:
    return CalendarCreateEventArgs(
        title="Team Sync",
        start=datetime(2026, 9, 1, 10, 0, 0),
        end=datetime(2026, 9, 1, 11, 0, 0),
        timezone="America/New_York",
    )


def _fake_service(
    execute_return: dict | None = None,
    execute_side_effect: Exception | None = None,
) -> MagicMock:
    service = MagicMock()
    execute_mock = service.events.return_value.insert.return_value.execute
    if execute_side_effect is not None:
        execute_mock.side_effect = execute_side_effect
    else:
        execute_mock.return_value = execute_return
    return service


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calendar_create_event_real_module, "_load_credentials", lambda: object())
    fake_service = _fake_service(execute_return={"id": "evt123", "status": "confirmed"})
    monkeypatch.setattr(
        calendar_create_event_real_module, "build", lambda *args, **kwargs: fake_service
    )

    result = calendar_create_event_real(_args())

    assert result.event_id == "evt123"
    assert result.status == "confirmed"


def test_missing_credentials_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CREDENTIALS_PATH", raising=False)

    with pytest.raises(MissingCredentialsError):
        calendar_create_event_real(_args())


def test_missing_credentials_file_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", "/nonexistent/path/token.json")

    with pytest.raises(MissingCredentialsError):
        calendar_create_event_real_module._load_credentials()


def test_api_failure_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calendar_create_event_real_module, "_load_credentials", lambda: object())
    fake_service = _fake_service(execute_side_effect=Exception("boom"))
    monkeypatch.setattr(
        calendar_create_event_real_module, "build", lambda *args, **kwargs: fake_service
    )

    with pytest.raises(CalendarCreateEventRequestError):
        calendar_create_event_real(_args())


def test_request_body_matches_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(calendar_create_event_real_module, "_load_credentials", lambda: object())
    fake_service = _fake_service(execute_return={"id": "evt123", "status": "confirmed"})
    monkeypatch.setattr(
        calendar_create_event_real_module, "build", lambda *args, **kwargs: fake_service
    )

    args = _args()
    calendar_create_event_real(args)

    _, kwargs = fake_service.events.return_value.insert.call_args
    body = kwargs["body"]
    assert body["summary"] == args.title
    assert body["start"] == {"dateTime": args.start.isoformat(), "timeZone": args.timezone}
    assert body["end"] == {"dateTime": args.end.isoformat(), "timeZone": args.timezone}

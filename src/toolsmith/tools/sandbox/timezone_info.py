"""Deterministic sandbox implementation of the timezone_info tool."""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from toolsmith.tools.schemas import ToolSpec, registry

# Fixed sandbox "now" (noon UTC on the project's fixed sandbox "today"). Never
# use datetime.now() here: the sandbox must be deterministic (same input ->
# identical output), so "current local time" is always derived from this
# constant rather than the real wall clock.
SANDBOX_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=ZoneInfo("UTC"))


class UnknownTimezoneError(ValueError):
    """Raised when a requested IANA timezone name isn't found in the tzdata."""


class TimezoneInfoArgs(BaseModel):
    """Arguments for a timezone_info call: an IANA name, or a lat/lon pair."""

    timezone: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def _require_timezone_or_coords(self) -> TimezoneInfoArgs:
        """Require either `timezone`, or both `lat` and `lon`, to be set."""
        has_timezone = self.timezone is not None
        has_coords = self.lat is not None and self.lon is not None
        if not has_timezone and not has_coords:
            raise ValueError("either timezone or both lat and lon must be provided")
        return self


class TimezoneInfoResult(BaseModel):
    """Result of a timezone_info call."""

    timezone: str
    utc_offset_minutes: int
    local_time: datetime


def timezone_info(args: TimezoneInfoArgs) -> TimezoneInfoResult:
    """Return the sandbox local time and UTC offset for a timezone name or lat/lon."""
    if args.timezone is not None:
        try:
            tz = ZoneInfo(args.timezone)
        except zoneinfo.ZoneInfoNotFoundError as exc:
            raise UnknownTimezoneError(f"unknown timezone: {args.timezone}") from exc
        tz_name = args.timezone
    else:
        # No offline reverse-geocoding timezone database is available, so we
        # deliberately approximate a fixed-offset zone from longitude alone
        # (15 degrees of longitude per UTC hour).
        offset_hours = round(args.lon / 15)
        tz = timezone(timedelta(hours=offset_hours))
        tz_name = f"UTC{offset_hours:+d}"

    local_time = SANDBOX_NOW.astimezone(tz)
    utc_offset = local_time.utcoffset()
    utc_offset_minutes = int(utc_offset.total_seconds() // 60) if utc_offset else 0

    return TimezoneInfoResult(
        timezone=tz_name,
        utc_offset_minutes=utc_offset_minutes,
        local_time=local_time,
    )


registry.register(
    ToolSpec(
        name="timezone_info",
        description=(
            "Look up the current local time and UTC offset for an IANA timezone "
            "name, or approximate one from a lat/lon pair."
        ),
        args_model=TimezoneInfoArgs,
        returns_model=TimezoneInfoResult,
        sandbox_fn=timezone_info,
    )
)

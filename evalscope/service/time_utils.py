"""Timestamp helpers for the service layer.

Metadata timestamps are persisted as timezone-aware UTC ISO-8601 strings.
Legacy metadata written before schema v15 was naive server-local time.  This
project historically ran in Asia/Shanghai, so v15 interprets naive legacy
values as UTC+08:00 by default before converting them to UTC.

Deployments with a different historical server timezone can override the
legacy fixed offset before the first v15 migration via
``EVALSCOPE_LEGACY_UTC_OFFSET_HOURS`` (for example ``-5`` or ``5.5``).
"""

import os
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
LEGACY_OFFSET_ENV = 'EVALSCOPE_LEGACY_UTC_OFFSET_HOURS'
DEFAULT_LEGACY_UTC_OFFSET_HOURS = 8.0


def legacy_timezone() -> timezone:
    """Return the fixed timezone used to interpret pre-v15 naive timestamps."""
    raw = os.environ.get(LEGACY_OFFSET_ENV, str(DEFAULT_LEGACY_UTC_OFFSET_HOURS))
    try:
        hours = float(raw)
    except ValueError as e:
        raise ValueError(f'{LEGACY_OFFSET_ENV} must be a numeric UTC offset, got {raw!r}') from e
    if not -24 < hours < 24:
        raise ValueError(f'{LEGACY_OFFSET_ENV} must be between -24 and 24 hours, got {hours}')
    return timezone(timedelta(hours=hours))


def utc_now_iso() -> str:
    """Return the current time as an offset-aware UTC ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def epoch_to_utc_iso(value: float | int) -> str:
    """Convert a Unix epoch value to an offset-aware UTC ISO-8601 string."""
    return datetime.fromtimestamp(float(value), UTC).isoformat()


def legacy_datetime_to_utc_iso(dt: datetime) -> str:
    """Convert a datetime to UTC, treating a naive value as legacy local time."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=legacy_timezone())
    return dt.astimezone(UTC).isoformat()


def normalize_persisted_timestamp(value: str | None) -> str | None:
    """Normalize one persisted timestamp to UTC.

    Offset-aware values preserve their actual instant.  Naive values are
    interpreted using :func:`legacy_timezone`.  Unparseable strings are left
    untouched so a migration never destroys opaque historical metadata.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value

    candidate = text[:-1] + '+00:00' if text.endswith(('Z', 'z')) else text
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return value
    return legacy_datetime_to_utc_iso(dt)

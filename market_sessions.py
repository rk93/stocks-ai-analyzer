"""Fail-closed NYSE session checks for prospective execution.

All inputs must carry a timezone.  Calendar dates are interpreted in New York,
while returned bounds are UTC.  Exchange holidays, early closes and daylight
saving transitions come from ``exchange_calendars`` rather than wall-clock
weekday rules.  If the calendar cannot answer, no execution is permitted.
"""
from __future__ import annotations

from functools import lru_cache
import math
from numbers import Real
from typing import Any

import pandas as pd

try:
    import exchange_calendars as _exchange_calendars
except ImportError:  # A missing optional runtime dependency must never open trading.
    _exchange_calendars = None


def _aware_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return None
        return timestamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        return None


@lru_cache(maxsize=8)
def _calendar(year: int):
    if _exchange_calendars is None:
        return None
    # Explicit bounds avoid silently relying on the library's moving default
    # range.  A year of bounds includes every session queried for that year.
    return _exchange_calendars.get_calendar(
        "XNYS", start=f"{year}-01-01", end=f"{year}-12-31"
    )


def session_bounds(timestamp: Any) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the NYSE session for the input's New York date, or ``None``.

    This also returns bounds for premarket/aftermarket timestamps on a valid
    session date.  Use :func:`in_regular_session` to check execution eligibility.
    Invalid or timezone-naive inputs, closed dates and calendar errors fail
    closed.  No fallback weekday schedule is used.
    """
    timestamp = _aware_timestamp(timestamp)
    if timestamp is None:
        return None
    try:
        local = timestamp.tz_convert("America/New_York")
        calendar = _calendar(local.year)
        if calendar is None:
            return None
        label = pd.Timestamp(local.date())
        if not calendar.is_session(label):
            return None
        opening = _aware_timestamp(calendar.session_open(label))
        closing = _aware_timestamp(calendar.session_close(label))
        if opening is None or closing is None or opening >= closing:
            return None
        return opening, closing
    except Exception:
        # Calendar lookup/coverage failures are execution-denial conditions.
        return None


def in_regular_session(timestamp: Any) -> bool:
    """Allow entry at the actual opening time, but never at/after the close."""
    timestamp = _aware_timestamp(timestamp)
    if timestamp is None:
        return False
    bounds = session_bounds(timestamp)
    return bounds is not None and bounds[0] <= timestamp < bounds[1]


def completed_bar(start: Any, observed_at: Any, minutes: Real = 5) -> bool:
    """Whether an entire regular-session bar existed when it was observed.

    ``start`` is the candle's opening timestamp.  Its end may equal the actual
    session close, including an early close.  Callers can mark the last bar
    ``session_end`` when ``start + duration == session_bounds(start)[1]``.
    Observation may happen later, but must be timezone-aware too.
    """
    start = _aware_timestamp(start)
    observed_at = _aware_timestamp(observed_at)
    if start is None or observed_at is None:
        return False
    if isinstance(minutes, bool) or not isinstance(minutes, Real):
        return False
    if not math.isfinite(minutes) or minutes <= 0:
        return False
    try:
        end = start + pd.Timedelta(minutes=float(minutes))
    except (TypeError, ValueError, OverflowError):
        return False
    bounds = session_bounds(start)
    return (
        bounds is not None
        and bounds[0] <= start < end <= bounds[1]
        and end <= observed_at
    )

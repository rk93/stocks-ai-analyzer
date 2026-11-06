"""Execution guards use the exchange calendar, never weekday approximations."""
from datetime import datetime, timezone

import pandas as pd
import pytest

import market_sessions as sessions
import paper_trader


def test_thanksgiving_is_closed():
    timestamp = "2026-11-26T15:00:00-05:00"
    assert sessions.session_bounds(timestamp) is None
    assert not sessions.in_regular_session(timestamp)
    assert not sessions.completed_bar(timestamp, "2026-11-26T15:10:00-05:00")


def test_black_friday_actual_early_close_and_last_completed_bar():
    opening, closing = sessions.session_bounds("2026-11-27T10:00:00-05:00")
    assert opening == pd.Timestamp("2026-11-27T14:30:00Z")
    assert closing == pd.Timestamp("2026-11-27T18:00:00Z")
    assert sessions.in_regular_session("2026-11-27T12:59:59-05:00")
    assert not sessions.in_regular_session("2026-11-27T13:00:00-05:00")
    assert not sessions.in_regular_session("2026-11-27T15:30:00-05:00")
    start = pd.Timestamp("2026-11-27T12:55:00-05:00")
    assert sessions.completed_bar(start, closing)
    assert start + pd.Timedelta(minutes=5) == closing  # session_end metadata
    assert not sessions.completed_bar(start, closing - pd.Timedelta(nanoseconds=1))
    assert not sessions.completed_bar("2026-11-27T12:56:00-05:00", closing)
    assert not sessions.completed_bar(closing, closing + pd.Timedelta(minutes=5))


def test_daylight_saving_transition_keeps_local_hours_correct():
    before = sessions.session_bounds("2026-10-30T12:00:00-04:00")
    after = sessions.session_bounds("2026-11-02T12:00:00-05:00")
    assert before == (
        pd.Timestamp("2026-10-30T13:30:00Z"),
        pd.Timestamp("2026-10-30T20:00:00Z"),
    )
    assert after == (
        pd.Timestamp("2026-11-02T14:30:00Z"),
        pd.Timestamp("2026-11-02T21:00:00Z"),
    )
    assert not sessions.in_regular_session("2026-11-02T14:29:59Z")
    assert sessions.in_regular_session("2026-11-02T14:30:00Z")
    assert sessions.in_regular_session("2026-11-02T20:59:59Z")
    assert not sessions.in_regular_session("2026-11-02T21:00:00Z")


def test_new_york_calendar_date_is_used_instead_of_utc_date():
    # Monday in UTC is still Sunday locally; it cannot obtain Monday bounds.
    assert sessions.session_bounds("2026-11-02T00:30:00Z") is None


@pytest.mark.parametrize("previous,expected_open", [
    ("2026-10-06T16:00:00-04:00", "2026-10-07T13:30:00Z"),
    ("2026-09-04T16:00:00-04:00", "2026-09-08T13:30:00Z"),  # Labor Day
    ("2026-10-30T16:00:00-04:00", "2026-11-02T14:30:00Z"),  # DST ends
    ("2026-11-25T16:00:00-05:00", "2026-11-27T14:30:00Z"),  # Thanksgiving
    ("2026-11-27T13:00:00-05:00", "2026-11-30T14:30:00Z"),  # half-day Friday
    ("2026-12-31T16:00:00-05:00", "2027-01-04T14:30:00Z"),  # year boundary
])
def test_next_session_open_uses_exchange_schedule(previous, expected_open):
    assert sessions.next_session_open(previous) == pd.Timestamp(expected_open)


@pytest.mark.parametrize("timestamp", [
    None, pd.NaT, "not-a-timestamp", "2026-11-02T15:00:00",
    datetime(2026, 11, 2, 15), 0, [], {},
])
def test_invalid_or_naive_timestamp_fails_closed(timestamp):
    assert sessions.session_bounds(timestamp) is None
    assert sessions.next_session_open(timestamp) is None
    assert not sessions.in_regular_session(timestamp)
    assert not sessions.completed_bar(timestamp, "2026-11-02T15:10:00Z")
    assert not sessions.completed_bar("2026-11-02T15:00:00Z", timestamp)


def test_aware_python_datetime_is_supported():
    assert sessions.in_regular_session(datetime(2026, 11, 2, 15, tzinfo=timezone.utc))


@pytest.mark.parametrize("minutes", [0, -5, float("nan"), float("inf"), "5", None, True])
def test_invalid_bar_duration_fails_closed(minutes):
    assert not sessions.completed_bar("2026-11-02T15:00:00Z", "2026-11-02T15:10:00Z", minutes)


def test_bar_must_be_wholly_inside_actual_session_and_already_completed():
    assert not sessions.completed_bar("2026-11-02T14:25:00Z", "2026-11-02T14:35:00Z")
    assert not sessions.completed_bar("2026-11-02T15:00:00Z", "2026-11-02T15:04:59Z")
    assert sessions.completed_bar("2026-11-02T15:00:00Z", "2026-11-02T15:05:00Z")
    assert sessions.completed_bar("2026-11-02T15:00:00Z", "2026-11-03T15:05:00Z")
    assert not sessions.completed_bar("2026-11-02T20:58:00Z", "2026-11-02T21:05:00Z")


def test_missing_calendar_dependency_fails_closed(monkeypatch):
    sessions._calendar.cache_clear()
    monkeypatch.setattr(sessions, "_exchange_calendars", None)
    try:
        assert sessions.session_bounds("2026-11-02T15:00:00Z") is None
        assert sessions.next_session_open("2026-11-02T15:00:00Z") is None
        assert not sessions.in_regular_session("2026-11-02T15:00:00Z")
        assert not sessions.completed_bar("2026-11-02T15:00:00Z", "2026-11-02T15:05:00Z")
    finally:
        sessions._calendar.cache_clear()


def test_calendar_lookup_errors_fail_closed(monkeypatch):
    def unavailable(year):
        raise RuntimeError("calendar coverage unavailable")
    monkeypatch.setattr(sessions, "_calendar", unavailable)
    assert sessions.session_bounds("2026-11-02T15:00:00Z") is None
    assert sessions.next_session_open("2026-11-02T15:00:00Z") is None
    assert not sessions.in_regular_session("2026-11-02T15:00:00Z")
    assert not sessions.completed_bar("2026-11-02T15:00:00Z", "2026-11-02T15:05:00Z")


def test_existing_paper_guard_uses_actual_exchange_session():
    assert paper_trader.in_regular_session is sessions.in_regular_session
    assert not paper_trader.fresh_quote(
        {"quote_available_at": "2026-11-26T10:00:00-05:00"},
        "2026-11-26T10:01:00-05:00",
    )
    assert not paper_trader.fresh_quote(
        {"quote_available_at": "2026-11-27T12:55:00-05:00"},
        "2026-11-27T13:00:00-05:00",
    )
    assert paper_trader.fresh_quote(
        {"quote_available_at": "2026-11-27T12:50:00-05:00"},
        "2026-11-27T12:51:00-05:00",
    )

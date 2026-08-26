import pandas as pd

from signal_engine import signal_label


def row(**overrides):
    base = {
        "quadrant": "CONTRARIAN BID",
        "quality_label": "HIGH",
        "quality_score": 80,
        "return_vs_spy_1m": -0.08,
        "normalized_skew": -0.05,
        "skew_change_5obs": None,
        "catalyst_flag": False,
        "days_to_earnings": 30,
        "history_observations": 1,
    }
    base.update(overrides)
    return pd.Series(base)


def test_new_signal_is_early_candidate_not_confirmed():
    label, score, reason = signal_label(row())
    assert label == "EARLY CANDIDATE"
    assert score < 60
    assert "1/5" in reason


def test_confirmed_candidate_requires_history_and_strengthening_skew():
    label, score, reason = signal_label(row(history_observations=7, skew_change_5obs=-0.025))
    assert label == "ADD CANDIDATE"
    assert score >= 65
    assert "confirmed" in reason


def test_mature_but_fading_signal_is_not_add_candidate():
    label, score, reason = signal_label(row(history_observations=7, skew_change_5obs=0.04))
    assert label == "WATCH"
    assert "fading" in reason


def test_near_earnings_overrides_candidate_state():
    label, _, _ = signal_label(row(history_observations=7, skew_change_5obs=-0.03, catalyst_flag=True, days_to_earnings=3))
    assert label == "EVENT RISK"

import math

import pandas as pd

from skew_map import bs_delta, bs_price, classify, implied_vol_from_mid
from signal_engine import signal_label


def test_atm_call_delta_is_reasonable():
    delta = bs_delta(100, 100, 30 / 365, 0.25, "call", rate=0.04)
    assert 0.45 < delta < 0.60


def test_put_delta_is_negative():
    delta = bs_delta(100, 95, 45 / 365, 0.30, "put", rate=0.04)
    assert -1 < delta < 0


def test_implied_vol_round_trip():
    t = 45 / 365
    mid = bs_price(100, 105, t, 0.32, "call", rate=0.04)
    recovered = implied_vol_from_mid(mid, 100, 105, t, "call", rate=0.04)
    assert abs(recovered - 0.32) < 1e-5


def test_invalid_mid_returns_nan():
    assert math.isnan(implied_vol_from_mid(0, 100, 100, 0.1, "call"))


def test_quadrants():
    assert classify(-0.05, -0.10) == "CONTRARIAN BID"
    assert classify(0.05, -0.10) == "CHASE"
    assert classify(0.05, 0.10) == "HEDGED RALLY"
    assert classify(-0.05, 0.10) == "FEAR"


def test_invalid_bs_inputs_return_nan():
    assert math.isnan(bs_delta(0, 100, 0.1, 0.2, "call"))


def test_entry_candidate_requires_good_contrarian_setup():
    row = pd.Series({
        "quadrant": "CONTRARIAN BID", "quality_label": "HIGH", "quality_score": 85,
        "return_vs_spy_1m": -0.08, "normalized_skew": -0.08,
        "skew_change_5obs": -0.03, "catalyst_flag": False, "days_to_earnings": 30,
    })
    label, score, reason = signal_label(row)
    assert label == "ADD CANDIDATE"
    assert score >= 58
    assert "call-side skew strengthening" in reason


def test_near_earnings_becomes_event_risk():
    row = pd.Series({
        "quadrant": "CONTRARIAN BID", "quality_label": "HIGH", "quality_score": 90,
        "return_vs_spy_1m": -0.08, "normalized_skew": -0.08,
        "skew_change_5obs": -0.03, "catalyst_flag": True, "days_to_earnings": 3,
    })
    label, _, _ = signal_label(row)
    assert label == "EVENT RISK"

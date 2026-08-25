import math

from skew_map import bs_delta, bs_price, classify, implied_vol_from_mid


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

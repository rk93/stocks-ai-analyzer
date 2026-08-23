import math

from skew_map import bs_delta, classify


def test_atm_call_delta_is_reasonable():
    delta = bs_delta(100, 100, 30 / 365, 0.25, "call", rate=0.04)
    assert 0.45 < delta < 0.60


def test_put_delta_is_negative():
    delta = bs_delta(100, 95, 45 / 365, 0.30, "put", rate=0.04)
    assert -1 < delta < 0


def test_quadrants():
    assert classify(-0.05, -0.10) == "CONTRARIAN BID"
    assert classify(0.05, -0.10) == "CHASE"
    assert classify(0.05, 0.10) == "HEDGED RALLY"
    assert classify(-0.05, 0.10) == "FEAR"


def test_invalid_bs_inputs_return_nan():
    assert math.isnan(bs_delta(0, 100, 0.1, 0.2, "call"))

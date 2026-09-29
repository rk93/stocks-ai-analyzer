from strategy_v4 import score_v4


def base_daily():
    return {
        "return_20d": 0.05,
        "ma20": 100.0,
        "ma50": 95.0,
        "atr_pct": 0.025,
        "prior_low": 98.0,
    }


def base_intra():
    return {
        "price": 105.0,
        "vwap_dist": 0.008,
        "volume_ratio_tod": 2.0,
        "momentum_15m": 0.001,
        "near_support": False,
        "breakout_or": True,
        "crossed_vwap": False,
        "minutes_from_open": 150,
        "gap": 0.01,
        "day_change": 0.02,
        "opening_range_low": 102.0,
    }


def bull_market():
    return {
        "price": 110.0,
        "ma20": 108.0,
        "ma50": 104.0,
        "ma200": 90.0,
        "atr_pct": 0.015,
        "return_20d": 0.04,
    }


def sideways_market():
    return {
        "price": 105.0,
        "ma20": 106.0,
        "ma50": 103.0,
        "ma200": 90.0,
        "atr_pct": 0.015,
        "return_20d": 0.01,
    }


def test_v4_allows_confirmed_bull_moderate_volume_setup():
    state, *_ = score_v4(base_daily(), base_intra(), bull_market())
    assert state == "ENTRY TRIGGERED"


def test_v4_rejects_sideways_regime():
    state, *_ = score_v4(base_daily(), base_intra(), sideways_market())
    assert state != "ENTRY TRIGGERED"


def test_v4_rejects_extreme_volume():
    intra = base_intra()
    intra["volume_ratio_tod"] = 2.8
    state, *_ = score_v4(base_daily(), intra, bull_market())
    assert state != "ENTRY TRIGGERED"

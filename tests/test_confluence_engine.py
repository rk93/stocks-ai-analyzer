import pandas as pd
from confluence_engine import score_row


def base():
    return pd.Series({
        "market_regime":"BULL_TREND",
        "v3_state":"ENTRY TRIGGERED",
        "volume_ratio_tod":1.9,
        "vwap_dist":0.006,
        "breakout_or":True,
        "trend_score":72,
        "rs_percentile":96,
        "trend_state":"STRONG TREND",
        "above_50d":True,
        "above_200d":True,
        "golden_cross":True,
        "quadrant":"HEDGED RALLY",
        "quality_label":"HIGH",
        "divergence_score":4,
        "days_to_earnings":30,
        "catalyst_flag":False,
    })


def test_trade_candidate_requires_multi_layer_confluence():
    out = score_row(base())
    assert out["decision"] == "TRADE CANDIDATE"


def test_earnings_veto_wins_over_good_signal():
    x = base(); x["days_to_earnings"] = 1
    out = score_row(x)
    assert out["decision"] == "AVOID EVENT RISK"


def test_sideways_market_prevents_intraday_trade_candidate():
    x = base(); x["market_regime"] = "SIDEWAYS"
    out = score_row(x)
    assert out["decision"] != "TRADE CANDIDATE"


def test_chase_options_prevents_trade_candidate():
    x = base(); x["quadrant"] = "CHASE"
    out = score_row(x)
    assert out["decision"] != "TRADE CANDIDATE"

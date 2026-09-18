import numpy as np

from strategy_v3 import classify_regime, score_v3


def test_classify_bull_trend():
    m={"price":110,"ma20":108,"ma50":105,"ma200":100,"atr_pct":0.015,"return_20d":0.04}
    assert classify_regime(m)=="BULL_TREND"


def test_classify_risk_off():
    m={"price":90,"ma20":94,"ma50":96,"ma200":100,"atr_pct":0.02,"return_20d":-0.08}
    assert classify_regime(m)=="RISK_OFF"


def test_v3_blocks_risk_off_even_with_setup():
    daily={"return_20d":0.05,"ma20":99,"ma50":95,"atr_pct":0.03}
    intra={"price":101,"vwap_dist":0.005,"volume_ratio_tod":2.1,"momentum_15m":0.01,
           "breakout_or":True,"crossed_vwap":False,"near_support":False,
           "minutes_from_open":150,"day_change":0.02,"gap":0.01,
           "opening_range_low":98}
    market={"price":90,"ma20":94,"ma50":96,"ma200":100,"atr_pct":0.02,"return_20d":-0.08}
    state,score,reason,stop,t1,t2=score_v3(daily,intra,market)
    assert state!="ENTRY TRIGGERED"
    assert "risk-off" in reason.lower()
    assert stop < intra["price"] < t1 < t2


def test_v3_can_trigger_in_bull_regime():
    daily={"return_20d":0.05,"ma20":99,"ma50":95,"atr_pct":0.03}
    intra={"price":101,"vwap_dist":0.005,"volume_ratio_tod":2.2,"momentum_15m":0.01,
           "breakout_or":True,"crossed_vwap":False,"near_support":False,
           "minutes_from_open":150,"day_change":0.02,"gap":0.01,
           "opening_range_low":98}
    market={"price":110,"ma20":108,"ma50":105,"ma200":100,"atr_pct":0.015,"return_20d":0.04}
    state,score,reason,stop,t1,t2=score_v3(daily,intra,market)
    assert state=="ENTRY TRIGGERED"
    assert score >= 68

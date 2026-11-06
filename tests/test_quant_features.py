import numpy as np
import pandas as pd
import pytest
from quant_features import feature_packet, intraday_features, regular_bars


def daily():
    idx=pd.bdate_range('2025-01-01',periods=260,tz='America/New_York')
    values=np.arange(100.,360.)
    return pd.DataFrame({'Open':values,'High':values+1,'Low':values-1,'Close':values,'Volume':1000},index=idx)


def intraday():
    idx=pd.date_range('2026-01-05 09:30',periods=20,freq='5min',tz='America/New_York')
    return pd.DataFrame({'Open':100.,'High':101.,'Low':99.,'Close':100.,'Volume':100.},index=idx)


def test_bar_close_unavailable_at_open_and_future_perturbation():
    bars=intraday();at=pd.Timestamp('2026-01-05 10:00',tz='America/New_York')
    bars.loc[at,['High','Close','Volume']]=[999,998,100000]
    features,asof=intraday_features(bars,{'prev_close':100},at)
    assert features['price']==100
    assert features['opening_range_high']==101
    assert pd.Timestamp(asof)==at
    later,_=intraday_features(bars,{'prev_close':100},at+pd.Timedelta(minutes=5))
    assert later['price']==998
    assert later['opening_range_high']==101 # 10:00 bar is outside opening range


def test_daily_excludes_entire_signal_session_and_future_rows():
    d=daily();signal=pd.Timestamp(d.index[-1].date(),tz='America/New_York')+pd.Timedelta(hours=10)
    packet=feature_packet(intraday(),d,d,signal)
    d.iloc[-1,d.columns.get_indexer(['High','Low','Close'])]=[1e9,1e9,1e9]
    changed=feature_packet(intraday(),d,d,signal)
    assert packet['daily']==changed['daily']
    assert packet['market']==changed['market']
    assert pd.Timestamp(packet['context']['daily_asof']).date()<signal.date()
    assert packet['daily']['rsi14']==100


def test_tod_volume_uses_prior_sessions_only():
    first=intraday();second=first.copy();second.index+=pd.Timedelta(days=1);second.Volume=200
    data=pd.concat([first,second])
    at=pd.Timestamp('2026-01-06 10:00',tz='America/New_York')
    result,_=intraday_features(data,{'prev_close':100},at)
    assert result['volume_ratio_tod']==2
    assert result['volume_history_sessions']==1
    future=second.copy();future.index+=pd.Timedelta(days=1);future.Volume=1e8
    other,_=intraday_features(pd.concat([data,future]),{'prev_close':100},at)
    assert result==other


def test_regular_session_excludes_close_labeled_bar_and_naive_input():
    d=intraday();d.index=pd.date_range('2026-01-05 15:50',periods=20,freq='5min',tz='America/New_York')
    assert len(regular_bars(d))==2
    with pytest.raises(ValueError):
        regular_bars(d.tz_localize(None))

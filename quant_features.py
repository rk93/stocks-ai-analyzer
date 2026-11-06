"""Point-in-time features shared by retrospective replay and prospective V4.

Yahoo 5-minute candle timestamps denote the OPEN. A close is available only
five minutes later. Daily observations always come from prior NY sessions.
No news, options, earnings, or historical constituent membership is inferred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = 'America/New_York'
BAR_DURATION = pd.Timedelta(minutes=5)


def utc_timestamp(value):
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError('An aware timestamp is required')
    return value.tz_convert('UTC')


def regular_bars(bars):
    if bars is None or bars.empty:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])
    out = bars.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        raise ValueError('Market candles must have timezone-aware timestamps')
    out.index = idx.tz_convert('UTC')
    out = out[~out.index.duplicated(keep='first')].sort_index()
    local = out.index.tz_convert(NY)
    minute = local.hour * 60 + local.minute
    return out[(minute >= 570) & (minute < 960)]


def daily_features(bars, signal_at):
    at = utc_timestamp(signal_at)
    if bars is None or bars.empty:
        return {}, None
    d = bars.copy()
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is None:
        raise ValueError('Daily candles must have timezone-aware timestamps')
    d.index = idx.tz_convert(NY)
    d = d.sort_index()
    d = d[d.index.date < at.tz_convert(NY).date()]
    d = d.dropna(subset=['High','Low','Close'])
    if len(d) < 50:
        return {}, None
    close = d['Close'].astype(float)
    tr = pd.concat([d.High-d.Low,(d.High-close.shift()).abs(),(d.Low-close.shift()).abs()],axis=1).max(axis=1)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    if loss.iloc[-1] == 0:
        rsi.iloc[-1] = 100 if gain.iloc[-1] > 0 else 50
    price = float(close.iloc[-1])
    result = {'prev_close':price,'price':price,'ma20':float(close.tail(20).mean()),
              'ma50':float(close.tail(50).mean()),
              'ma200':float(close.tail(200).mean()) if len(close)>=200 else np.nan,
              'atr_pct':float(tr.tail(14).mean()/price),'prior_high':float(d.High.iloc[-1]),
              'prior_low':float(d.Low.iloc[-1]),'return_20d':float(price/close.iloc[-21]-1),
              'rsi14':float(rsi.iloc[-1]),'history_sessions':len(d)}
    # Daily candle availability is its session's close, never the next midnight.
    asof = pd.Timestamp(d.index[-1].date(),tz=NY) + pd.Timedelta(hours=16)
    return result, asof.tz_convert('UTC').isoformat()


class PreparedIntraday:
    """Reusable volume/session index; values are still filtered at each cutoff."""
    def __init__(self,bars):
        reg=regular_bars(bars)
        local=reg.index.tz_convert(NY)
        self.sessions={date:group for date,group in reg.groupby(local.date)}
        volume=pd.DataFrame({'date':local.date,'minute':local.hour*60+local.minute,'volume':reg.Volume.to_numpy()})
        self.volume=volume.pivot(index='date',columns='minute',values='volume').sort_index()


def intraday_features(bars, daily, signal_at, *, prepared=None):
    at = utc_timestamp(signal_at)
    date = at.tz_convert(NY).date()
    if prepared is None:
        reg = regular_bars(bars)
        reg = reg[reg.index + BAR_DURATION <= at]
        if reg.empty:
            return {}, None
        dates = reg.index.tz_convert(NY).date
        today = reg[dates == date]
        prior = reg[dates < date]
    else:
        today=prepared.sessions.get(date,pd.DataFrame())
        if today.empty:return {},None
        today=today[today.index+BAR_DURATION<=at]
    if len(today) < 3 or not ((today.index[-3:][1:]-today.index[-3:][:-1]) == BAR_DURATION).all():
        return {}, None
    local = today.index.tz_convert(NY)
    opening = today[(local.hour*60+local.minute < 600)]
    typical = (today.High + today.Low + today.Close)/3
    vol = today.Volume.fillna(0).clip(lower=0)
    cumulative = vol.cumsum().replace(0,np.nan)
    vwap_series = (typical*vol).cumsum()/cumulative
    price = float(today.Close.iloc[-1]); vwap = float(vwap_series.iloc[-1])
    prev_close = daily.get('prev_close',np.nan)
    session_open = float(today.Open.iloc[0])
    if prepared is None:
        prior_dates = prior.index.tz_convert(NY).date
        sessions = sorted(set(prior_dates))[-20:]
        history = prior[np.isin(prior_dates,sessions)]
        times = set(local[-3:].time)
        matched = history[[t in times for t in history.index.tz_convert(NY).time]]
        if not matched.empty:
            groups = matched.groupby(matched.index.tz_convert(NY).date)
            complete_dates = [day for day, group in groups if len(group) == 3]
            matched = matched[np.isin(matched.index.tz_convert(NY).date,complete_dates)]
        expected = float(matched.Volume.mean()) if not matched.empty else np.nan
        avg = float(history.Volume.tail(78*3).mean()) if not history.empty else np.nan
        history_sessions=len(set(matched.index.tz_convert(NY).date))
    else:
        historical=prepared.volume[prepared.volume.index<date].tail(20)
        minutes=list(local[-3:].hour*60+local[-3:].minute)
        matched=historical.reindex(columns=minutes).dropna()
        expected=float(matched.to_numpy().mean()) if len(matched) else np.nan
        historical_values=historical.to_numpy().ravel()
        historical_values=historical_values[np.isfinite(historical_values)][-234:]
        avg=float(historical_values.mean()) if len(historical_values) else np.nan
        history_sessions=len(matched)
    # Mean of identical clock slots from prior sessions only; no future sessions.
    actual = float(vol.tail(3).mean())
    opening_high = float(opening.High.max()) if not opening.empty else np.nan
    opening_low = float(opening.Low.min()) if not opening.empty else np.nan
    prev_vwap = float(vwap_series.iloc[-2])
    support = [daily.get('ma20',np.nan),daily.get('ma50',np.nan),daily.get('prior_low',np.nan)]
    near_support = any(np.isfinite(x) and abs(price/x-1) <= .012 for x in support if x != 0)
    result = {'price':price,'session_open':session_open,
              'gap':session_open/prev_close-1 if prev_close>0 else np.nan,
              'day_change':price/prev_close-1 if prev_close>0 else np.nan,'session_return':price/session_open-1,'vwap':vwap,'vwap_dist':price/vwap-1,
              'volume_ratio':actual/avg if avg>0 else np.nan,
              'volume_ratio_tod':actual/expected if expected>0 else np.nan,
              'minutes_from_open':float((at.tz_convert(NY)-(pd.Timestamp(date,tz=NY)+pd.Timedelta(hours=9,minutes=30))).total_seconds()/60),
              'opening_range_high':opening_high,'opening_range_low':opening_low,
              'breakout_or':bool(at.tz_convert(NY).hour*60+at.tz_convert(NY).minute>=600 and price>opening_high),
              'crossed_vwap':bool(today.Close.iloc[-2]<=prev_vwap and price>vwap),
              'momentum_15m':price/float(today.Close.iloc[-4])-1 if len(today)>=4 and today.index[-1]-today.index[-4] == pd.Timedelta(minutes=15) else np.nan,
              'legacy_momentum_10m':price/float(today.Close.iloc[-3])-1,'near_support':near_support,'dollar_volume_15m':float((today.Close.tail(3)*vol.tail(3)).sum()),
              'volume_history_sessions':history_sessions}
    return result, (today.index[-1]+BAR_DURATION).isoformat()


def feature_packet(symbol_bars, daily_bars, market_daily, signal_at):
    """Return ``{daily,intra,market,context}`` using only available observations."""
    at = utc_timestamp(signal_at)
    daily, daily_asof = daily_features(daily_bars,at)
    market, market_asof = daily_features(market_daily,at)
    intra, intraday_asof = intraday_features(symbol_bars,daily,at)
    daily['relative_strength_20d'] = daily.get('return_20d',np.nan)-market.get('return_20d',np.nan)
    return {'daily':daily,'intra':intra,'market':market,
            'context':{'signal_at':at.isoformat(),'daily_asof':daily_asof,
                       'market_asof':market_asof,'intraday_asof':intraday_asof}}

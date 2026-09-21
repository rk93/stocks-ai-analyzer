from __future__ import annotations

"""Point-in-time backtest for the Day Opportunities strategy.

Replays the same score_row() logic on historical 5-minute bars. Yahoo/yfinance
limits intraday history to the recent ~60 days, so this engine deliberately
labels its coverage and never fabricates older intraday signals.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from day_opportunities import score_row
from strategy_v2 import score_v2
from strategy_v3 import score_v3

CONFIG=Path("skew_config.json")
TRADES=Path("data/backtest_trades.csv")
SUMMARY=Path("data/backtest_summary.json")
HORIZONS=(1,5,15,30,63)


def safe(v, default=np.nan):
    try:
        x=float(v); return x if np.isfinite(x) else default
    except Exception:return default


def regular(df):
    if df.empty:return df
    idx=pd.DatetimeIndex(df.index)
    if idx.tz is None:idx=idx.tz_localize("UTC")
    idx=idx.tz_convert("America/New_York")
    x=df.copy(); x.index=idx
    return x.between_time("09:30","16:00")


def daily_asof(daily, day):
    # Strict point-in-time: only sessions before the signal day are available.
    x=daily[pd.DatetimeIndex(daily.index).date < day]
    if len(x)<55:return None
    close=x.Close.dropna(); high=x.High.dropna(); low=x.Low.dropna()
    if len(close)<55:return None
    prev=safe(close.iloc[-1])
    tr=pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    return {"prev_close":prev,"ma20":safe(close.tail(20).mean()),"ma50":safe(close.tail(50).mean()),
            "atr_pct":safe(tr.tail(14).mean())/prev if prev else np.nan,
            "prior_high":safe(high.iloc[-1]),"prior_low":safe(low.iloc[-1]),
            "return_20d":safe(close.iloc[-1]/close.iloc[-21]-1)}


def market_asof(daily, day):
    x=daily[pd.DatetimeIndex(daily.index).date < day]
    if len(x)<205:return {}
    close=x.Close.dropna(); high=x.High.dropna(); low=x.Low.dropna()
    if len(close)<205:return {}
    price=safe(close.iloc[-1])
    tr=pd.concat([high-low,(high-close.shift(1)).abs(),(low-close.shift(1)).abs()],axis=1).max(axis=1)
    return {"price":price,"ma20":safe(close.tail(20).mean()),"ma50":safe(close.tail(50).mean()),
            "ma200":safe(close.tail(200).mean()),"atr_pct":safe(tr.tail(14).mean())/price if price else np.nan,
            "return_20d":safe(close.iloc[-1]/close.iloc[-21]-1)}


def intra_asof(today, prior_intraday, cutoff, dm):
    x=today[today.index<=cutoff].copy()
    if len(x)<3:return None
    typical=(x.High+x.Low+x.Close)/3
    cv=x.Volume.fillna(0).cumsum()
    vwap=(typical*x.Volume.fillna(0)).cumsum()/cv.replace(0,np.nan)
    price=safe(x.Close.iloc[-1]); cur_vwap=safe(vwap.iloc[-1])
    opening=x.between_time("09:30","10:00")
    or_high=safe(opening.High.max()) if not opening.empty else np.nan
    or_low=safe(opening.Low.min()) if not opening.empty else np.nan
    avg=safe(prior_intraday.Volume.tail(78*3).mean(),0) if not prior_intraday.empty else 0
    vr=safe(x.Volume.tail(3).mean()/avg) if avg>0 else np.nan
    prev=dm["prev_close"]; op=safe(x.Open.iloc[0])
    vdist=price/cur_vwap-1 if cur_vwap else np.nan
    last3=x.Close.tail(3)
    mom=safe(last3.iloc[-1]/last3.iloc[0]-1) if len(last3)>=3 else np.nan
    crossed=False
    if len(x)>=2 and np.isfinite(cur_vwap):
        crossed=safe(x.Close.iloc[-2])<=safe(vwap.iloc[-2]) and price>cur_vwap
    near=False
    for s in (dm["ma20"],dm["ma50"],dm["prior_low"]):
        if np.isfinite(s) and abs(price/s-1)<=.012:near=True;break
    # Time-of-day normalized volume: compare the latest 3 bars with the same
    # clock-time bars on prior sessions, avoiding the U-shaped intraday-volume bias.
    recent_times=set(x.tail(3).index.time)
    tod=prior_intraday[[t in recent_times for t in prior_intraday.index.time]]
    tod_avg=safe(tod.Volume.mean(),0) if not tod.empty else 0
    vr_tod=safe(x.Volume.tail(3).mean()/tod_avg) if tod_avg>0 else np.nan
    mins=(cutoff.hour*60+cutoff.minute)-(9*60+30)
    return {"price":price,"session_open":op,"gap":op/prev-1,"day_change":price/prev-1,
            "vwap":cur_vwap,"vwap_dist":vdist,"volume_ratio":vr,"volume_ratio_tod":vr_tod,"minutes_from_open":mins,
            "opening_range_high":or_high,"opening_range_low":or_low,
            "breakout_or":bool(np.isfinite(or_high) and price>or_high),"crossed_vwap":bool(crossed),
            "momentum_15m":mom,"near_support":near}


def exact_trade_path(intra, cutoff, entry, stop, t1, t2):
    """Walk future 5m bars. Entry occurs at cutoff close, so start next bar."""
    future=intra[intra.index>cutoff]
    risk=max(entry-stop,1e-9); first_1r=None; first_2r=None; first_stop=None
    for ts,b in future.iterrows():
        hi=safe(b.High); lo=safe(b.Low)
        # Same 5m candle touching both is genuinely unordered.
        if first_1r is None and hi>=t1:first_1r=ts
        if first_2r is None and hi>=t2:first_2r=ts
        if first_stop is None and lo<=stop:first_stop=ts
        if first_2r is not None and first_stop is not None:break
    if first_1r is not None and first_stop is not None and first_1r==first_stop:
        result="AMBIGUOUS"; r=np.nan
    elif first_1r is not None and (first_stop is None or first_1r<first_stop):
        result="WIN_1R"; r=1.0
    elif first_stop is not None:
        result="STOP"; r=-1.0
    else:
        result="OPEN"; r=np.nan
    hit2=first_2r is not None and (first_stop is None or first_2r<first_stop)
    return {"trade_result_1r":result,"realized_r_1r":r,"target_2r_before_stop":bool(hit2),
            "first_1r_at":first_1r.isoformat() if first_1r is not None else "",
            "first_2r_at":first_2r.isoformat() if first_2r is not None else "",
            "first_stop_at":first_stop.isoformat() if first_stop is not None else ""}

def outcome(daily, signal_day, entry, stop, t1, t2):
    future=daily[pd.DatetimeIndex(daily.index).date>signal_day]; out={}
    for n in HORIZONS:
        label="3m" if n==63 else f"{n}d"
        if len(future)<n:
            out.update({f"return_{label}":np.nan,f"mfe_{label}":np.nan,f"mae_{label}":np.nan,f"result_{label}":"PENDING"});continue
        w=future.iloc[:n]; hi=safe(w.High.max()); lo=safe(w.Low.min()); close=safe(w.Close.iloc[-1])
        out.update({f"return_{label}":close/entry-1,f"mfe_{label}":hi/entry-1,f"mae_{label}":lo/entry-1,
                    f"result_{label}":"UP" if close>entry else ("DOWN" if close<entry else "FLAT")})
    return out


def main():
    cfg=json.loads(CONFIG.read_text())
    symbols=cfg.get("day_universe") or cfg["core_universe"]
    rows=[]; coverage=[]
    try:
        market_daily=yf.Ticker("SPY").history(period="2y",interval="1d",auto_adjust=False)
    except Exception:
        market_daily=pd.DataFrame()
    for symbol in symbols:
        if symbol.startswith("^"):continue
        print("BACKTEST",symbol,flush=True)
        try:
            intra=regular(yf.Ticker(symbol).history(period="60d",interval="5m",prepost=False,auto_adjust=False))
            daily=yf.Ticker(symbol).history(period="1y",interval="1d",auto_adjust=False)
        except Exception as e:
            print("SKIP",symbol,e);continue
        if intra.empty or daily.empty:continue
        days=sorted(set(intra.index.date))
        coverage.extend(days)
        for day in days:
            dm=daily_asof(daily,day)
            if not dm:continue
            today=intra[intra.index.date==day]
            prior=intra[intra.index.date<day]
            # Mirrors the scheduled scanner cadence after the opening range exists.
            checkpoints=today[(today.index.minute%30==0)&(today.index.time>=pd.Timestamp("10:00").time())].index
            mm=market_asof(market_daily,day) if not market_daily.empty else {}
            for version in ("V1","V2","V3"):
                signal=None
                for cutoff in checkpoints:
                    im=intra_asof(today,prior,cutoff,dm)
                    if not im:continue
                    if version=="V1": state,score,reason,stop,t1,t2=score_row(dm,im)
                    elif version=="V2": state,score,reason,stop,t1,t2=score_v2(dm,im)
                    else: state,score,reason,stop,t1,t2=score_v3(dm,im,mm)
                    if state=="ENTRY TRIGGERED":
                        signal=(cutoff,im,score,reason,stop,t1,t2);break
                if not signal:continue
                cutoff,im,score,reason,stop,t1,t2=signal
                row={"strategy":version,"symbol":symbol,"signal_date":str(day),"signal_time":cutoff.isoformat(),
                     "score":score,"entry":im["price"],"stop":stop,"target_1r":t1,"target_2r":t2,
                     "reason":reason,"volume_ratio":im["volume_ratio"],"volume_ratio_tod":im["volume_ratio_tod"],
                     "vwap_dist":im["vwap_dist"],"day_change_at_signal":im["day_change"]}
                row.update(exact_trade_path(intra,cutoff,im["price"],stop,t1,t2))
                row.update(outcome(daily,day,im["price"],stop,t1,t2));rows.append(row)

    df=pd.DataFrame(rows)
    TRADES.parent.mkdir(parents=True,exist_ok=True);df.to_csv(TRADES,index=False)
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"method":"exact_recent_intraday",
             "note":"5-minute point-in-time replay; historical intraday coverage is limited by Yahoo/yfinance.",
             "coverage_start":str(min(coverage)) if coverage else None,"coverage_end":str(max(coverage)) if coverage else None,
             "signals":len(df),"strategies":{},"horizons":{}}
    if not df.empty:
        for version,g in df.groupby("strategy"):
            resolved=g[g.trade_result_1r.isin(["WIN_1R","STOP"])]
            wins=int((resolved.trade_result_1r=="WIN_1R").sum()); losses=int((resolved.trade_result_1r=="STOP").sum())
            gross_win=float(resolved.loc[resolved.realized_r_1r>0,"realized_r_1r"].sum())
            gross_loss=abs(float(resolved.loc[resolved.realized_r_1r<0,"realized_r_1r"].sum()))
            summary["strategies"][version]={"signals":len(g),"resolved_1r":len(resolved),"wins_1r":wins,"stops":losses,
                "ambiguous_5m":int((g.trade_result_1r=="AMBIGUOUS").sum()),
                "win_rate_1r":wins/len(resolved) if len(resolved) else None,
                "expectancy_r":safe(resolved.realized_r_1r.mean()) if len(resolved) else None,
                "profit_factor_r":gross_win/gross_loss if gross_loss else None,
                "target_2r_before_stop_rate":safe(g.target_2r_before_stop.mean()) if len(g) else None}
        # Horizon returns describe signal follow-through; exact 1R trade sequencing is above.
        summary["horizons_by_strategy"]={}
        for version,g in df.groupby("strategy"):
            summary["horizons_by_strategy"][version]={}
            for n in HORIZONS:
                label="3m" if n==63 else f"{n}d"; col=f"result_{label}"
                resolved=g[g[col].isin(["UP","DOWN"])]
                summary["horizons_by_strategy"][version][label]={
                    "resolved":len(resolved),
                    "correct":int((resolved[col]=="UP").sum()),
                    "wrong":int((resolved[col]=="DOWN").sum()),
                    "pending":int((g[col]=="PENDING").sum()),
                    "win_rate":safe((resolved[col]=="UP").mean()) if len(resolved) else None,
                    "avg_return":safe(g[f"return_{label}"].mean()) if g[f"return_{label}"].notna().any() else None,
                    "median_return":safe(g[f"return_{label}"].median()) if g[f"return_{label}"].notna().any() else None,
                    "avg_mfe":safe(g[f"mfe_{label}"].mean()) if g[f"mfe_{label}"].notna().any() else None,
                    "avg_mae":safe(g[f"mae_{label}"].mean()) if g[f"mae_{label}"].notna().any() else None,
                }
        for n in HORIZONS:
            label="3m" if n==63 else f"{n}d"; col=f"result_{label}"
            resolved=df[df[col].isin(["UP","DOWN"])]
            summary["horizons"][label]={"resolved":len(resolved),"correct":int((resolved[col]=="UP").sum()),
                "wrong":int((resolved[col]=="DOWN").sum()),"ambiguous":0,
                "pending":int((df[col]=="PENDING").sum()),
                "win_rate":safe((resolved[col]=="UP").mean()) if len(resolved) else None,
                "avg_return":safe(df[f"return_{label}"].mean()) if df[f"return_{label}"].notna().any() else None,
                "median_return":safe(df[f"return_{label}"].median()) if df[f"return_{label}"].notna().any() else None,
                "avg_mfe":safe(df[f"mfe_{label}"].mean()) if df[f"mfe_{label}"].notna().any() else None,
                "avg_mae":safe(df[f"mae_{label}"].mean()) if df[f"mae_{label}"].notna().any() else None}
    SUMMARY.write_text(json.dumps(summary,allow_nan=False,indent=2))

if __name__=="__main__":main()

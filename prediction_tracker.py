from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HIST = Path("data/day_opportunities_history.csv")
OUT = Path("data/prediction_tracker.csv")


def _f(v):
    try:
        x=float(v); return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _session(symbol, start, end):
    # Daily bars are sufficient for the first auditable 1-day tracker.
    d=yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False)
    return d.dropna(subset=["Open","High","Low","Close"]) if not d.empty else d


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not HIST.exists():
        pd.DataFrame().to_csv(OUT,index=False); return
    h=pd.read_csv(HIST)
    if h.empty:
        h.to_csv(OUT,index=False); return

    h["generated_at"]=pd.to_datetime(h["generated_at"],utc=True,errors="coerce")
    h=h.dropna(subset=["generated_at","symbol"])
    h["signal_date"]=h["generated_at"].dt.tz_convert("America/New_York").dt.date

    # Freeze one prediction per stock/day: first ENTRY TRIGGERED, otherwise latest WATCH.
    h["priority"]=(h["state"]=="ENTRY TRIGGERED").astype(int)
    h=h.sort_values(["signal_date","symbol","priority","generated_at"])
    chosen=[]
    for _,g in h.groupby(["signal_date","symbol"],sort=False):
        trig=g[g.state=="ENTRY TRIGGERED"]
        chosen.append(trig.iloc[0] if not trig.empty else g.iloc[-1])
    signals=pd.DataFrame(chosen)

    rows=[]
    today=datetime.now(timezone.utc).date()
    for _,r in signals.iterrows():
        sd=r["signal_date"]
        # Need at least the next trading session; request a generous calendar window.
        start=(pd.Timestamp(sd)+pd.Timedelta(days=1)).date().isoformat()
        end=(pd.Timestamp(sd)+pd.Timedelta(days=8)).date().isoformat()
        try:
            d=_session(str(r.symbol),start,end)
        except Exception:
            d=pd.DataFrame()
        nextbar=d.iloc[0] if not d.empty else None
        entry=_f(r.get("price")); stop=_f(r.get("stop")); t1=_f(r.get("target_1r")); t2=_f(r.get("target_2r"))
        outcome="PENDING"; ret=np.nan; max_move=np.nan; adverse=np.nan; target_hit=False; stop_hit=False
        eval_date=None; close=np.nan
        if nextbar is not None and np.isfinite(entry) and entry>0:
            eval_date=str(pd.Timestamp(nextbar.name).date())
            hi=_f(nextbar.High); lo=_f(nextbar.Low); close=_f(nextbar.Close)
            ret=close/entry-1
            max_move=hi/entry-1; adverse=lo/entry-1
            target_hit=np.isfinite(t1) and hi>=t1
            stop_hit=np.isfinite(stop) and lo<=stop
            if r.state=="ENTRY TRIGGERED":
                # Daily OHLC cannot know ordering if both levels trade the same day.
                if target_hit and stop_hit: outcome="AMBIGUOUS"
                elif target_hit: outcome="CORRECT"
                elif stop_hit: outcome="WRONG"
                else: outcome="CORRECT" if ret>0 else ("WRONG" if ret<0 else "UNRESOLVED")
            else:
                outcome="OBSERVED"
        rows.append({
            "signal_date":str(sd),"evaluation_date":eval_date,"symbol":r.symbol,
            "prediction":r.state,"score":_f(r.get("score")),"signal_price":entry,
            "stop":stop,"target_1r":t1,"target_2r":t2,"next_close":close,
            "return_1d":ret,"max_favourable_1d":max_move,"max_adverse_1d":adverse,
            "target_hit":target_hit,"stop_hit":stop_hit,"result":outcome,
            "original_reason":r.get("reason",""),"signal_time":r.get("generated_at"),
        })
    pd.DataFrame(rows).sort_values(["signal_date","score"],ascending=[False,False]).to_csv(OUT,index=False)


if __name__=="__main__":
    main()

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

LEDGER=Path("data/predictions.csv")
LEGACY=Path("data/day_opportunities_history.csv")
OUT=Path("data/prediction_tracker.csv")
HORIZONS=(1,5,15,30)

def f(v):
    try:
        x=float(v); return x if np.isfinite(x) else np.nan
    except Exception: return np.nan

def source():
    if LEDGER.exists():
        d=pd.read_csv(LEDGER)
        if not d.empty: return d
    if not LEGACY.exists(): return pd.DataFrame()
    h=pd.read_csv(LEGACY)
    if h.empty:return h
    h=h[h.state=="ENTRY TRIGGERED"].copy()
    h["generated_at"]=pd.to_datetime(h.generated_at,utc=True,errors="coerce")
    h["signal_date"]=h.generated_at.dt.tz_convert("America/New_York").dt.date.astype(str)
    h=h.sort_values("generated_at").drop_duplicates(["signal_date","symbol"],keep="first")
    h["prediction_id"]=h.apply(lambda r:f"legacy-{r.signal_date}-{r.symbol}",axis=1)
    return h

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    s=source()
    if s.empty: pd.DataFrame().to_csv(OUT,index=False); return
    rows=[]
    for _,r in s.iterrows():
        sd=str(r.get("signal_date") or pd.to_datetime(r.generated_at,utc=True).tz_convert("America/New_York").date())
        entry=f(r.get("price")); stop=f(r.get("stop")); t1=f(r.get("target_1r")); t2=f(r.get("target_2r"))
        start=(pd.Timestamp(sd)+pd.Timedelta(days=1)).date().isoformat()
        end=(pd.Timestamp(sd)+pd.Timedelta(days=60)).date().isoformat()
        try:d=yf.Ticker(str(r.symbol)).history(start=start,end=end,interval="1d",auto_adjust=False).dropna(subset=["High","Low","Close"])
        except Exception:d=pd.DataFrame()
        out={"prediction_id":r.get("prediction_id"),"signal_date":sd,"signal_time":r.get("generated_at"),
             "symbol":r.symbol,"prediction":r.get("state"),"score":f(r.get("score")),"signal_price":entry,
             "stop":stop,"target_1r":t1,"target_2r":t2,"original_reason":r.get("reason","")}
        for n in HORIZONS:
            col=str(n)
            if len(d)>=n and np.isfinite(entry) and entry>0:
                w=d.iloc[:n]; close=f(w.Close.iloc[-1]); hi=f(w.High.max()); lo=f(w.Low.min())
                out[f"evaluation_date_{col}d"]=str(pd.Timestamp(w.index[-1]).date())
                out[f"return_{col}d"]=close/entry-1; out[f"max_favourable_{col}d"]=hi/entry-1; out[f"max_adverse_{col}d"]=lo/entry-1
                out[f"target1_hit_{col}d"]=bool(np.isfinite(t1) and hi>=t1); out[f"target2_hit_{col}d"]=bool(np.isfinite(t2) and hi>=t2)
                out[f"stop_hit_{col}d"]=bool(np.isfinite(stop) and lo<=stop)
                # OHLC across a window cannot prove order when both are touched.
                if out[f"target1_hit_{col}d"] and out[f"stop_hit_{col}d"]: result="AMBIGUOUS"
                elif out[f"target1_hit_{col}d"]: result="CORRECT"
                elif out[f"stop_hit_{col}d"]: result="WRONG"
                else: result="CORRECT" if out[f"return_{col}d"]>0 else ("WRONG" if out[f"return_{col}d"]<0 else "UNRESOLVED")
                out[f"result_{col}d"]=result
            else: out[f"result_{col}d"]="PENDING"
        rows.append(out)
    pd.DataFrame(rows).sort_values(["signal_date","score"],ascending=[False,False]).to_csv(OUT,index=False)

if __name__=="__main__":main()

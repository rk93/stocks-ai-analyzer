from __future__ import annotations
"""Experimental Day Opportunities V2.

V1 remains the live strategy. V2 is deliberately isolated for backtest/forward
validation so research changes cannot silently alter live ENTRY TRIGGERED calls.
"""
import numpy as np

def _finite(x):
    try:return np.isfinite(float(x))
    except Exception:return False

def score_v2(daily: dict, intra: dict):
    score=0.0; reasons=[]
    price=float(intra["price"]); vdist=float(intra.get("vwap_dist",np.nan)); vr=float(intra.get("volume_ratio_tod",np.nan))
    atr=float(daily.get("atr_pct",np.nan)); mom=float(intra.get("momentum_15m",np.nan))
    # Structural trend: modest weight, rather than stacking many correlated momentum bonuses.
    trend=0
    if _finite(daily.get("return_20d")) and daily["return_20d"]>0: trend+=1
    if price>daily.get("ma20",np.inf): trend+=1
    if price>daily.get("ma50",np.inf): trend+=1
    score += (0,6,12,18)[trend]
    if trend>=2: reasons.append(f"trend {trend}/3")

    # V2's primary setup is a confirmed opening-range break while close remains near VWAP.
    if intra.get("breakout_or") and _finite(vdist) and 0<=vdist<=0.018:
        score+=28; reasons.append("confirmed OR breakout")
    # A one-bar VWAP cross is context only; V1 overweighted this.
    elif intra.get("crossed_vwap") and _finite(vdist) and 0<=vdist<=0.008:
        score+=8; reasons.append("VWAP reclaim context")

    # Compare volume with historical bars at the same time of day.
    if _finite(vr) and vr>=2: score+=22; reasons.append(f"time-normalized volume {vr:.1f}x")
    elif _finite(vr) and vr>=1.4: score+=14; reasons.append(f"time-normalized volume {vr:.1f}x")

    # Momentum is confirmation, not another major independent signal.
    if _finite(mom) and 0.002<=mom<=0.02: score+=6; reasons.append("controlled 15m momentum")
    if intra.get("near_support"): score+=4; reasons.append("near support")
    if _finite(atr) and .015<=atr<=.06: score+=5; reasons.append("tradable daily range")

    # Explicit anti-chase / failed-breakout penalties.
    if _finite(vdist) and vdist>.018: score-=25; reasons.append("too extended from VWAP")
    dc=float(intra.get("day_change",np.nan))
    if _finite(dc) and dc>.05: score-=18; reasons.append("already >5% up")
    gap=float(intra.get("gap",np.nan))
    if _finite(gap) and gap>.06: score-=12; reasons.append("large gap")

    score=float(max(0,min(100,score)))
    # Research hypothesis from V1 diagnostics: wait until initial opening volatility settles.
    after_confirmation=bool(intra.get("minutes_from_open",0)>=120)  # 11:30 ET
    trigger=(score>=62 and after_confirmation and intra.get("breakout_or")
             and _finite(vdist) and 0<=vdist<=.018 and _finite(vr) and vr>=1.4)
    state="ENTRY TRIGGERED" if trigger else ("WATCH" if score>=45 else "NO TRADE")

    atr_dollars=price*atr if _finite(atr) else price*.025
    stop_dist=min(max(.006*price,.35*atr_dollars),.015*price)
    or_low=intra.get("opening_range_low",np.nan)
    stop=max(price-stop_dist,or_low if _finite(or_low) and or_low<price else price-stop_dist)
    risk=max(price-stop,price*.004)
    return state,round(score,1),"; ".join(reasons[:7]),stop,price+risk,price+2*risk

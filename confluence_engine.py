from __future__ import annotations

"""Research-only multi-layer confluence engine.

Combines independent evidence instead of hiding everything in one score:
- intraday execution quality (day_opportunities)
- swing/relative-strength quality (trending)
- options/skew positioning (skew map)
- catalyst / earnings-event risk

It does NOT place trades. It emits a compact research decision:
TRADE CANDIDATE / SWING CANDIDATE / WAIT FOR TRIGGER / AVOID / NO SETUP.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DAY = Path("data/day_opportunities_latest.csv")
TREND = Path("data/trending_latest.csv")
SKEW = Path("data/skew_latest.csv")
OUT = Path("data/confluence_latest.csv")


def num(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def text(v):
    return "" if pd.isna(v) else str(v)


def score_row(r: pd.Series) -> dict:
    reasons = []

    # ---- Intraday layer (0-35) ----
    intra = 0.0
    regime = text(r.get("market_regime"))
    v3_state = text(r.get("v3_state"))
    rvol = num(r.get("volume_ratio_tod"))
    vdist = num(r.get("vwap_dist"))
    breakout = bool(r.get("breakout_or", False))

    if v3_state == "ENTRY TRIGGERED":
        intra += 16; reasons.append("V3 entry")
    elif v3_state == "WATCH":
        intra += 7; reasons.append("V3 watch")
    if breakout:
        intra += 6; reasons.append("opening-range breakout")
    if np.isfinite(vdist) and 0 <= vdist <= 0.01:
        intra += 5; reasons.append("near VWAP")
    if np.isfinite(rvol) and 1.4 <= rvol < 2.5:
        intra += 8; reasons.append("healthy relative volume")
    elif np.isfinite(rvol) and rvol >= 2.5:
        intra += 2; reasons.append("extreme volume")

    # ---- Swing / RS layer (0-35) ----
    swing = 0.0
    trend_score = num(r.get("trend_score"), 0.0)
    rs = num(r.get("rs_percentile"), 0.0)
    trend_state = text(r.get("trend_state"))
    if trend_score >= 70:
        swing += 12
    elif trend_score >= 65:
        swing += 8
    elif trend_score >= 60:
        swing += 4
    if rs >= 95:
        swing += 10; reasons.append("RS >=95")
    elif rs >= 90:
        swing += 7; reasons.append("RS >=90")
    elif rs >= 80:
        swing += 3
    if bool(r.get("above_50d", False)) and bool(r.get("above_200d", False)):
        swing += 6; reasons.append("above 50D/200D")
    if bool(r.get("golden_cross", False)):
        swing += 3
    if trend_state == "STRONG TREND":
        swing += 4; reasons.append("strong trend")

    # ---- Options layer (-10 to +20) ----
    options = 0.0
    quadrant = text(r.get("quadrant"))
    quality = text(r.get("quality_label"))
    divergence = num(r.get("divergence_score"), 0.0)
    if quality == "HIGH":
        options += 8; reasons.append("high-quality options")
    elif quality == "MEDIUM":
        options += 4
    if quadrant == "CONTRARIAN BID":
        options += 8; reasons.append("contrarian options bid")
    elif quadrant == "HEDGED RALLY":
        options += 5; reasons.append("hedged rally")
    elif quadrant == "CHASE":
        options -= 8; reasons.append("options chase")
    elif quadrant == "FEAR":
        options -= 2
    if divergence >= 8:
        options += 4; reasons.append("skew divergence")

    # ---- Market / event layer ----
    market = 0.0
    if regime == "BULL_TREND":
        market += 10; reasons.append("bull regime")
    elif regime == "SIDEWAYS":
        market -= 6; reasons.append("sideways regime")
    elif regime in {"RISK_OFF", "HIGH_VOL", "UNKNOWN"}:
        market -= 15; reasons.append("hostile regime")

    event = 0.0
    dte = num(r.get("days_to_earnings"))
    catalyst = bool(r.get("catalyst_flag", False))
    hard_veto = False
    if np.isfinite(dte) and dte <= 2:
        hard_veto = True; event -= 25; reasons.append("earnings <=2d")
    elif np.isfinite(dte) and dte <= 7:
        event -= 10; reasons.append("earnings <=7d")
    if catalyst:
        event += 3

    total = intra + swing + options + market + event

    # Keep setup types explicit: a strong swing setup does not need an intraday trigger yet.
    if hard_veto:
        decision = "AVOID EVENT RISK"
    elif regime in {"RISK_OFF", "HIGH_VOL", "UNKNOWN"}:
        decision = "AVOID"
    elif (v3_state == "ENTRY TRIGGERED" and regime == "BULL_TREND" and
          swing >= 14 and total >= 55 and quadrant != "CHASE"):
        decision = "TRADE CANDIDATE"
    elif swing >= 22 and total >= 40 and quadrant != "CHASE":
        decision = "SWING CANDIDATE"
    elif swing >= 16 or v3_state in {"ENTRY TRIGGERED", "WATCH"}:
        decision = "WAIT FOR TRIGGER"
    else:
        decision = "NO SETUP"

    return {
        "decision": decision,
        "confluence_score": round(total, 1),
        "intraday_score": round(intra, 1),
        "swing_score": round(swing, 1),
        "options_score": round(options, 1),
        "market_event_score": round(market + event, 1),
        "confluence_reason": "; ".join(reasons[:10]),
    }


def main():
    if not DAY.exists():
        print("No day-opportunity data")
        return
    try:
        day = pd.read_csv(DAY)
    except pd.errors.EmptyDataError:
        day = pd.DataFrame()
    try:
        trend = pd.read_csv(TREND) if TREND.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        trend = pd.DataFrame()
    try:
        skew = pd.read_csv(SKEW) if SKEW.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        skew = pd.DataFrame()

    if day.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUT, index=False)
        return

    x = day.copy()
    if not trend.empty:
        keep = [c for c in ["symbol","trend_score","trend_state","rs_percentile","above_50d","above_200d","golden_cross","sector","analyst_recommendation"] if c in trend.columns]
        x = x.merge(trend[keep].drop_duplicates("symbol"), on="symbol", how="left")
    if not skew.empty:
        keep = [c for c in ["symbol","quality_label","quadrant","divergence_score","catalyst_flag","days_to_earnings","skew_percentile"] if c in skew.columns]
        x = x.merge(skew[keep].drop_duplicates("symbol"), on="symbol", how="left")

    scored = x.apply(score_row, axis=1, result_type="expand")
    x = pd.concat([x, scored], axis=1)
    rank = {"TRADE CANDIDATE":0,"SWING CANDIDATE":1,"WAIT FOR TRIGGER":2,"AVOID EVENT RISK":3,"AVOID":4,"NO SETUP":5}
    x["_rank"] = x["decision"].map(rank).fillna(9)
    x = x.sort_values(["_rank","confluence_score"], ascending=[True,False]).drop(columns="_rank")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(OUT, index=False)
    cols = [c for c in ["symbol","decision","confluence_score","market_regime","v3_state","trend_state","rs_percentile","quadrant","days_to_earnings","confluence_reason"] if c in x.columns]
    print(x[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

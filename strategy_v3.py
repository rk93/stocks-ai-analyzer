from __future__ import annotations
"""Experimental V3 strategy.

V3 is deliberately deterministic.  Agent/research context may be attached to a
signal for explanation, but it is never allowed to bypass the hard execution
gate.  This keeps historical and forward results reproducible.
"""
import numpy as np

from strategy_v2 import score_v2


def _finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def classify_regime(market: dict) -> str:
    """Classify broad US equity conditions from point-in-time SPY metrics."""
    price = float(market.get("price", np.nan))
    ma20 = float(market.get("ma20", np.nan))
    ma50 = float(market.get("ma50", np.nan))
    ma200 = float(market.get("ma200", np.nan))
    atr = float(market.get("atr_pct", np.nan))
    ret20 = float(market.get("return_20d", np.nan))

    if not all(_finite(x) for x in (price, ma20, ma50, ma200)):
        return "UNKNOWN"
    if _finite(atr) and atr >= 0.025:
        return "HIGH_VOL"
    if price < ma200 and ret20 < -0.04:
        return "RISK_OFF"
    if price > ma20 > ma50 > ma200 and ret20 > 0:
        return "BULL_TREND"
    if price > ma200 and ret20 >= -0.02:
        return "SIDEWAYS"
    return "RECOVERY"


def score_v3(daily: dict, intra: dict, market: dict | None = None):
    """V2 setup plus regime/risk gates.

    Returns the same tuple contract as V1/V2:
    state, score, reason, stop, target_1r, target_2r.
    """
    v2_state, v2_score, v2_reason, stop, t1, t2 = score_v2(daily, intra)
    market = market or {}
    regime = classify_regime(market)
    score = float(v2_score)
    reasons = [v2_reason] if v2_reason else []

    if regime == "BULL_TREND":
        score += 8
        reasons.append("broad market bull trend")
    elif regime == "RECOVERY":
        score += 3
        reasons.append("broad market recovery")
    elif regime == "SIDEWAYS":
        reasons.append("sideways market: neutral")
    elif regime == "HIGH_VOL":
        score -= 10
        reasons.append("high-volatility regime penalty")
    elif regime == "RISK_OFF":
        score -= 20
        reasons.append("risk-off regime penalty")
    else:
        score -= 5
        reasons.append("market regime unavailable")

    price = float(intra.get("price", np.nan))
    vdist = float(intra.get("vwap_dist", np.nan))
    vr = float(intra.get("volume_ratio_tod", np.nan))
    gap = float(intra.get("gap", np.nan))
    atr = float(daily.get("atr_pct", np.nan))

    # Hard gates intentionally remain deterministic.
    liquid_confirmation = _finite(vr) and vr >= 1.4
    near_vwap = _finite(vdist) and 0 <= vdist <= 0.015
    breakout = bool(intra.get("breakout_or"))
    settled = int(intra.get("minutes_from_open", 0) or 0) >= 120
    acceptable_gap = not _finite(gap) or gap <= 0.05
    acceptable_atr = not _finite(atr) or atr <= 0.065

    if near_vwap:
        score += 4
        reasons.append("tight VWAP proximity")
    if _finite(vr) and vr >= 2.0:
        score += 4
        reasons.append("strong time-normalized volume")
    if not acceptable_gap:
        score -= 12
        reasons.append("gap gate failed")
    if not acceptable_atr:
        score -= 8
        reasons.append("volatility gate failed")

    score = float(max(0, min(100, score)))
    hard_gate = (
        score >= 68
        and settled
        and breakout
        and near_vwap
        and liquid_confirmation
        and acceptable_gap
        and acceptable_atr
        and regime not in {"RISK_OFF", "HIGH_VOL", "UNKNOWN"}
    )
    state = "ENTRY TRIGGERED" if hard_gate else ("WATCH" if score >= 50 else "NO TRADE")

    # Keep V2 risk geometry so strategy comparisons isolate the entry logic.
    return state, round(score, 1), "; ".join([r for r in reasons if r][:8]), stop, t1, t2

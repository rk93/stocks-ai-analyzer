from __future__ import annotations
"""Experimental V4 strategy.

V4 is a deliberately small refinement of V3 based on walk-forward diagnostics:
- retain V3's deterministic setup/risk geometry
- only enter in a confirmed broad-market BULL_TREND
- require elevated but non-extreme time-normalized volume (1.4x <= RVOL < 2.5x)

This version is research-only. It must earn its way through backtest and forward
validation before any execution path uses it.
"""
import numpy as np

from strategy_v3 import classify_regime, score_v3


def _finite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def score_v4(daily: dict, intra: dict, market: dict | None = None):
    state3, score3, reason3, stop, t1, t2 = score_v3(daily, intra, market)
    market = market or {}
    regime = classify_regime(market)
    vr = float(intra.get("volume_ratio_tod", np.nan))

    reasons = [reason3] if reason3 else []
    reasons.append("V4 bull-regime gate")
    reasons.append("V4 moderate-volume gate")

    bull_gate = regime == "BULL_TREND"
    volume_gate = _finite(vr) and 1.4 <= vr < 2.5
    hard_gate = state3 == "ENTRY TRIGGERED" and bull_gate and volume_gate

    # Keep the V3 score for interpretability; V4 changes authorization, not ranking.
    state = "ENTRY TRIGGERED" if hard_gate else ("WATCH" if score3 >= 50 else "NO TRADE")
    return state, float(score3), "; ".join([r for r in reasons if r][:10]), stop, t1, t2

"""Portfolio-aware signal layer on top of the options skew scanner.

Educational research only. Labels are research states, not trade instructions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

LATEST = Path("data/skew_latest.csv")
SIGNALS = Path("data/signals_latest.csv")
PORTFOLIO = Path("portfolio_config.json")


def _price_only(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    h = t.history(period="3mo", auto_adjust=True)
    if len(h) < 22:
        raise ValueError("insufficient price history")
    spot = float(h.Close.iloc[-1])
    ret_1m = float(h.Close.iloc[-1] / h.Close.iloc[-22] - 1)
    avg_vol = float(h.Volume.iloc[-21:-1].mean())
    volume_ratio = float(h.Volume.iloc[-1] / avg_vol) if avg_vol > 0 else np.nan
    return {"spot": spot, "return_1m": ret_1m, "volume_ratio": volume_ratio}


def signal_label(row: pd.Series) -> tuple[str, float, str]:
    q = str(row.get("quadrant", ""))
    quality = str(row.get("quality_label", "LOW"))
    qs = float(row.get("quality_score", 0) or 0)
    rel = float(row.get("return_vs_spy_1m", 0) or 0)
    skew = float(row.get("normalized_skew", 0) or 0)
    change = row.get("skew_change_5obs")
    change = float(change) if pd.notna(change) else None
    catalyst = bool(row.get("catalyst_flag", False))
    days = row.get("days_to_earnings")

    score = 0.0
    reasons = []
    if q == "CONTRARIAN BID":
        score += 40
        reasons.append("price/options divergence")
    elif q == "CHASE":
        score += 18
        reasons.append("momentum and calls agree")
    elif q == "HEDGED RALLY":
        score += 8
        reasons.append("rally carries downside hedging")
    else:
        score -= 12
        reasons.append("price weakness and puts agree")

    score += {"HIGH": 25, "MEDIUM": 15, "LOW": 0}.get(quality, 0)
    if quality != "LOW": reasons.append(f"{quality.lower()} quote quality")
    if rel < -0.03 and q == "CONTRARIAN BID": score += min(15, abs(rel) * 100)
    if skew < -0.03: score += min(10, abs(skew) * 50)
    if change is not None:
        if change < -0.015:
            score += 15
            reasons.append("call-side skew strengthening")
        elif change > 0.03:
            score -= 12
            reasons.append("bullish skew fading")
    else:
        reasons.append("history still building")

    if catalyst and days is not None and float(days) <= 10:
        score -= 30
        reasons.append(f"earnings in {int(float(days))}d")
        label = "EVENT RISK"
    elif q == "CONTRARIAN BID" and quality in {"HIGH", "MEDIUM"} and score >= 58:
        label = "ADD CANDIDATE"
    elif q == "FEAR":
        label = "DON'T ADD"
    elif q in {"CHASE", "HEDGED RALLY"}:
        label = "HOLD / WATCH"
    else:
        label = "WATCH"

    return label, round(max(0, min(100, score)), 1), "; ".join(reasons)


def build() -> pd.DataFrame:
    latest = pd.read_csv(LATEST) if LATEST.exists() else pd.DataFrame()
    if not latest.empty:
        labels = latest.apply(signal_label, axis=1, result_type="expand")
        labels.columns = ["research_state", "entry_score", "signal_reason"]
        latest = pd.concat([latest, labels], axis=1)
        latest["in_portfolio"] = False

    portfolio = json.loads(PORTFOLIO.read_text(encoding="utf-8"))["holdings"]
    portfolio_symbols = {x["symbol"] for x in portfolio}
    if not latest.empty:
        latest.loc[latest.symbol.isin(portfolio_symbols), "in_portfolio"] = True

    extra = []
    known = set(latest.symbol) if not latest.empty else set()
    for item in portfolio:
        symbol = item["symbol"]
        if symbol in known:
            continue
        try:
            p = _price_only(symbol)
            extra.append({
                "symbol": symbol, "sector": "Portfolio", **p,
                "quadrant": "PRICE ONLY", "quality_label": "N/A", "quality_score": np.nan,
                "normalized_skew": np.nan, "skew_change_5obs": np.nan,
                "return_vs_spy_1m": np.nan, "divergence_score": np.nan,
                "catalyst_flag": False, "days_to_earnings": np.nan,
                "research_state": "HOLD / WATCH", "entry_score": 0.0,
                "signal_reason": "portfolio coverage available; reliable options skew unavailable",
                "in_portfolio": True
            })
        except Exception as exc:
            extra.append({
                "symbol": symbol, "sector": "Portfolio", "quadrant": "UNAVAILABLE",
                "quality_label": "N/A", "research_state": "WATCH", "entry_score": 0.0,
                "signal_reason": f"data unavailable: {exc}", "in_portfolio": True
            })
    out = pd.concat([latest, pd.DataFrame(extra)], ignore_index=True, sort=False) if extra else latest
    SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SIGNALS, index=False)
    return out


if __name__ == "__main__":
    df = build()
    cols = [c for c in ["symbol", "research_state", "entry_score", "quadrant", "quality_label", "signal_reason"] if c in df]
    print(df[cols].sort_values("entry_score", ascending=False).to_string(index=False))

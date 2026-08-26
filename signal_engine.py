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
HISTORY = Path("data/skew_history.csv")
SIGNALS = Path("data/signals_latest.csv")
PORTFOLIO = Path("portfolio_config.json")
MIN_CONFIRMATION_OBS = 5


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
    rel = float(row.get("return_vs_spy_1m", 0) or 0)
    skew = float(row.get("normalized_skew", 0) or 0)
    change = row.get("skew_change_5obs")
    change = float(change) if pd.notna(change) else None
    catalyst = bool(row.get("catalyst_flag", False))
    days = row.get("days_to_earnings")
    obs = int(float(row.get("history_observations", 0) or 0))

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
    if quality != "LOW":
        reasons.append(f"{quality.lower()} quote quality")
    if rel < -0.03 and q == "CONTRARIAN BID":
        score += min(15, abs(rel) * 100)
    if skew < -0.03:
        score += min(10, abs(skew) * 50)

    mature = obs >= MIN_CONFIRMATION_OBS
    strengthening = change is not None and change < -0.005
    if not mature:
        reasons.append(f"history building ({obs}/{MIN_CONFIRMATION_OBS} observations)")
        # A fresh one-day signal should never look as certain as a confirmed setup.
        score = min(score, 59.9)
    elif change is None:
        reasons.append("history count met but 5-observation change unavailable")
        score -= 8
    elif strengthening:
        score += 15
        reasons.append("call-side skew strengthening; history confirmed")
    elif change > 0.03:
        score -= 12
        reasons.append("bullish skew fading")
    else:
        reasons.append("history mature but skew not yet strengthening")

    if catalyst and days is not None and float(days) <= 10:
        score -= 30
        reasons.append(f"earnings in {int(float(days))}d")
        label = "EVENT RISK"
    elif q == "CONTRARIAN BID" and quality in {"HIGH", "MEDIUM"}:
        if not mature and score >= 45:
            label = "EARLY CANDIDATE"
        elif mature and strengthening and score >= 65:
            label = "ADD CANDIDATE"
        else:
            label = "WATCH"
    elif q == "FEAR":
        label = "DON'T ADD"
    elif q in {"CHASE", "HEDGED RALLY"}:
        label = "HOLD / WATCH"
    else:
        label = "WATCH"

    return label, round(max(0, min(100, score)), 1), "; ".join(reasons)


def _history_counts() -> dict[str, int]:
    if not HISTORY.exists():
        return {}
    h = pd.read_csv(HISTORY)
    if h.empty or "symbol" not in h.columns:
        return {}
    if "run_date" in h.columns:
        return h.groupby("symbol")["run_date"].nunique().astype(int).to_dict()
    return h.groupby("symbol").size().astype(int).to_dict()


def build() -> pd.DataFrame:
    latest = pd.read_csv(LATEST) if LATEST.exists() else pd.DataFrame()
    counts = _history_counts()
    if not latest.empty:
        latest["history_observations"] = latest["symbol"].map(counts).fillna(0).astype(int)
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
                "history_observations": counts.get(symbol, 0),
                "research_state": "HOLD / WATCH", "entry_score": 0.0,
                "signal_reason": "portfolio coverage available; reliable options skew unavailable",
                "in_portfolio": True
            })
        except Exception as exc:
            extra.append({
                "symbol": symbol, "sector": "Portfolio", "quadrant": "UNAVAILABLE",
                "quality_label": "N/A", "history_observations": counts.get(symbol, 0),
                "research_state": "WATCH", "entry_score": 0.0,
                "signal_reason": f"data unavailable: {exc}", "in_portfolio": True
            })
    out = pd.concat([latest, pd.DataFrame(extra)], ignore_index=True, sort=False) if extra else latest
    SIGNALS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SIGNALS, index=False)
    return out


if __name__ == "__main__":
    df = build()
    cols = [c for c in ["symbol", "research_state", "entry_score", "history_observations", "quadrant", "quality_label", "signal_reason"] if c in df]
    print(df[cols].sort_values("entry_score", ascending=False).to_string(index=False))

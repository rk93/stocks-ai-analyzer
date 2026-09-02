from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CONFIG = Path("skew_config.json")
OUT = Path("data/day_opportunities_latest.csv")
HIST = Path("data/day_opportunities_history.csv")


def _safe(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("America/New_York")
    out = df.copy()
    out.index = idx
    return out.between_time("09:30", "16:00")


def _daily_metrics(symbol: str):
    d = yf.Ticker(symbol).history(period="6mo", interval="1d", auto_adjust=False)
    if len(d) < 55:
        return {}
    close = d["Close"].dropna()
    high = d["High"].dropna()
    low = d["Low"].dropna()
    if len(close) < 55:
        return {}
    prev_close = _safe(close.iloc[-2])
    ma20 = _safe(close.tail(20).mean())
    ma50 = _safe(close.tail(50).mean())
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = _safe(tr.tail(14).mean())
    return {
        "prev_close": prev_close,
        "ma20": ma20,
        "ma50": ma50,
        "atr_pct": atr14 / prev_close if prev_close else np.nan,
        "prior_high": _safe(high.iloc[-2]),
        "prior_low": _safe(low.iloc[-2]),
        "return_20d": _safe(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan,
    }


def _intraday_metrics(symbol: str, daily: dict):
    raw = yf.Ticker(symbol).history(period="5d", interval="5m", prepost=True, auto_adjust=False)
    if raw.empty:
        return None
    reg = _regular_session(raw)
    if reg.empty:
        return None
    today = reg[reg.index.date == reg.index[-1].date()].copy()
    if len(today) < 3:
        return None

    typical = (today["High"] + today["Low"] + today["Close"]) / 3.0
    cumvol = today["Volume"].replace(0, np.nan).fillna(0).cumsum()
    vwap = (typical * today["Volume"].fillna(0)).cumsum() / cumvol.replace(0, np.nan)
    price = _safe(today["Close"].iloc[-1])
    cur_vwap = _safe(vwap.iloc[-1])

    opening = today.between_time("09:30", "10:00")
    or_high = _safe(opening["High"].max()) if not opening.empty else np.nan
    or_low = _safe(opening["Low"].min()) if not opening.empty else np.nan

    past = reg[reg.index.date != today.index[-1].date()]
    avg_bar_vol = _safe(past["Volume"].tail(78 * 3).mean(), 0.0)
    volume_ratio = _safe(today["Volume"].tail(3).mean() / avg_bar_vol) if avg_bar_vol > 0 else np.nan

    prev_close = daily.get("prev_close", np.nan)
    session_open = _safe(today["Open"].iloc[0])
    gap = session_open / prev_close - 1 if prev_close and np.isfinite(prev_close) else np.nan
    day_change = price / prev_close - 1 if prev_close and np.isfinite(prev_close) else np.nan
    vwap_dist = price / cur_vwap - 1 if cur_vwap and np.isfinite(cur_vwap) else np.nan

    last3 = today["Close"].tail(3)
    momentum_15m = _safe(last3.iloc[-1] / last3.iloc[0] - 1) if len(last3) >= 3 else np.nan
    crossed_vwap = False
    if len(today) >= 2 and np.isfinite(cur_vwap):
        prev_vwap = _safe(vwap.iloc[-2])
        crossed_vwap = _safe(today["Close"].iloc[-2]) <= prev_vwap and price > cur_vwap

    breakout_or = np.isfinite(or_high) and price > or_high
    near_support = False
    supports = [daily.get("ma20"), daily.get("ma50"), daily.get("prior_low")]
    for s in supports:
        if s and np.isfinite(s) and abs(price / s - 1) <= 0.012:
            near_support = True
            break

    return {
        "price": price,
        "session_open": session_open,
        "gap": gap,
        "day_change": day_change,
        "vwap": cur_vwap,
        "vwap_dist": vwap_dist,
        "volume_ratio": volume_ratio,
        "opening_range_high": or_high,
        "opening_range_low": or_low,
        "breakout_or": bool(breakout_or),
        "crossed_vwap": bool(crossed_vwap),
        "momentum_15m": momentum_15m,
        "near_support": bool(near_support),
    }


def score_row(daily: dict, intra: dict):
    score = 0.0
    reasons = []
    price = intra["price"]
    vdist = intra["vwap_dist"]
    vr = intra["volume_ratio"]
    atr = daily.get("atr_pct", np.nan)

    if intra["crossed_vwap"]:
        score += 22; reasons.append("VWAP reclaim")
    elif np.isfinite(vdist) and 0 <= vdist <= 0.012:
        score += 14; reasons.append("holding above VWAP")
    if intra["breakout_or"]:
        score += 18; reasons.append("opening-range breakout")
    if np.isfinite(vr) and vr >= 2:
        score += 20; reasons.append(f"volume {vr:.1f}x")
    elif np.isfinite(vr) and vr >= 1.4:
        score += 12; reasons.append(f"volume {vr:.1f}x")
    if np.isfinite(intra["momentum_15m"]) and intra["momentum_15m"] > 0.004:
        score += 10; reasons.append("15m momentum")
    if intra["near_support"]:
        score += 8; reasons.append("near support")
    if np.isfinite(daily.get("return_20d", np.nan)) and daily["return_20d"] > 0:
        score += 7; reasons.append("positive 1M trend")
    if np.isfinite(atr) and atr >= 0.025:
        score += 7; reasons.append("enough daily range")
    if price > daily.get("ma20", np.inf):
        score += 4
    if price > daily.get("ma50", np.inf):
        score += 4

    # Penalise chasing: good signal should occur near VWAP / support, not after a vertical move.
    if np.isfinite(vdist) and vdist > 0.025:
        score -= 22; reasons.append("extended above VWAP")
    if np.isfinite(intra["day_change"]) and intra["day_change"] > 0.07:
        score -= 15; reasons.append("already >7% up")
    if np.isfinite(intra["gap"]) and intra["gap"] > 0.08:
        score -= 10; reasons.append("large gap")

    score = float(max(0, min(100, score)))
    trigger = (
        score >= 68
        and np.isfinite(vdist) and 0 <= vdist <= 0.025
        and np.isfinite(vr) and vr >= 1.4
        and (intra["crossed_vwap"] or intra["breakout_or"])
    )
    state = "ENTRY TRIGGERED" if trigger else ("WATCH" if score >= 48 else "NO TRADE")

    atr_dollars = price * atr if np.isfinite(atr) else price * 0.025
    stop_dist = min(max(0.006 * price, 0.35 * atr_dollars), 0.015 * price)
    stop = max(price - stop_dist, intra.get("opening_range_low") if np.isfinite(intra.get("opening_range_low", np.nan)) and intra.get("opening_range_low") < price else price - stop_dist)
    risk = max(price - stop, price * 0.004)
    target1 = price + risk
    target2 = price + 2 * risk

    return state, round(score, 1), "; ".join(reasons[:7]), stop, target1, target2


def main():
    cfg = json.loads(CONFIG.read_text())
    symbols = cfg.get("day_universe") or cfg.get("core_universe", [])
    symbols = [s for s in symbols if not s.startswith("^")]
    rows = []
    ts = datetime.now(timezone.utc).isoformat()
    for symbol in symbols:
        try:
            daily = _daily_metrics(symbol)
            if not daily:
                continue
            intra = _intraday_metrics(symbol, daily)
            if not intra:
                continue
            state, score, reason, stop, t1, t2 = score_row(daily, intra)
            row = {"generated_at": ts, "symbol": symbol, **daily, **intra,
                   "state": state, "score": score, "reason": reason,
                   "stop": stop, "target_1r": t1, "target_2r": t2}
            rows.append(row)
        except Exception as e:
            print(f"SKIP {symbol}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT, index=False)
        return
    df = df.sort_values(["score", "volume_ratio"], ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Persist only actionable/watch snapshots, one row per symbol per run.
    keep = df[df["state"].isin(["ENTRY TRIGGERED", "WATCH"])].copy()
    if not keep.empty:
        if HIST.exists():
            old = pd.read_csv(HIST)
            keep = pd.concat([old, keep], ignore_index=True).tail(5000)
        keep.to_csv(HIST, index=False)

    print(df[["symbol", "state", "score", "day_change", "volume_ratio", "vwap_dist", "reason"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

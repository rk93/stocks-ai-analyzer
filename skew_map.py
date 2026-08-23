"""Options skew analytics engine.

Educational research tooling only. A skew reading is positioning/relative pricing,
not a prediction or trade recommendation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

DEFAULT_CONFIG = Path("skew_config.json")
HISTORY_PATH = Path("data/skew_history.csv")
LATEST_PATH = Path("data/skew_latest.csv")


@dataclass
class SkewResult:
    run_date: str
    symbol: str
    sector: str
    expiry: str
    dte: int
    spot: float
    return_1m: float
    return_vs_spy_1m: float
    volume_ratio: float
    atm_iv: float
    put_iv: float
    call_iv: float
    raw_skew: float
    normalized_skew: float
    put_delta: float
    call_delta: float
    put_strike: float
    call_strike: float
    put_oi: int
    call_oi: int
    put_volume: int
    call_volume: int
    put_spread_pct: float
    call_spread_pct: float
    quality_score: float
    quality_label: str
    catalyst_flag: bool
    days_to_earnings: Optional[int]
    quadrant: str
    skew_change_5obs: Optional[float] = None
    skew_change_20obs: Optional[float] = None
    skew_percentile: Optional[float] = None
    divergence_score: Optional[float] = None


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def bs_delta(spot: float, strike: float, t: float, iv: float, option_type: str,
             rate: float = 0.04, dividend_yield: float = 0.0) -> float:
    """Black-Scholes delta estimate used only to choose comparable chain strikes."""
    if min(spot, strike, t, iv) <= 0:
        return np.nan
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    if option_type == "call":
        return float(math.exp(-dividend_yield * t) * norm.cdf(d1))
    return float(math.exp(-dividend_yield * t) * (norm.cdf(d1) - 1.0))


def _spread_pct(row: pd.Series) -> float:
    bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
    mid = (bid + ask) / 2
    return (ask - bid) / mid if mid > 0 and ask >= bid else 9.99


def _select_expiry(ticker: yf.Ticker, min_dte: int, max_dte: int) -> tuple[str, int]:
    today = date.today()
    candidates = []
    for exp in ticker.options:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if min_dte <= dte <= max_dte:
            candidates.append((abs(dte - (min_dte + max_dte) / 2), exp, dte))
    if not candidates:
        raise ValueError(f"No expiry in configured {min_dte}-{max_dte} DTE window")
    _, exp, dte = min(candidates)
    return exp, dte


def _prepare_chain(df: pd.DataFrame, spot: float, dte: int, option_type: str,
                   min_oi: int) -> pd.DataFrame:
    out = df.copy()
    out["impliedVolatility"] = pd.to_numeric(out["impliedVolatility"], errors="coerce")
    out["openInterest"] = pd.to_numeric(out.get("openInterest", 0), errors="coerce").fillna(0)
    out["volume"] = pd.to_numeric(out.get("volume", 0), errors="coerce").fillna(0)
    out = out[(out.impliedVolatility > 0.01) & (out.impliedVolatility < 5.0)]
    out["delta_est"] = out.apply(
        lambda r: bs_delta(spot, float(r.strike), dte / 365.0, float(r.impliedVolatility), option_type), axis=1
    )
    out["spread_pct"] = out.apply(_spread_pct, axis=1)
    # Keep sparse rows available for scoring, but prefer liquid rows for selection.
    out["liquid"] = out.openInterest >= min_oi
    return out


def _pick_delta(df: pd.DataFrame, target_abs_delta: float, option_type: str) -> pd.Series:
    otm = df[df.strike > 0].copy()
    if option_type == "call":
        target = target_abs_delta
    else:
        target = -target_abs_delta
    otm["delta_distance"] = (otm.delta_est - target).abs()
    preferred = otm[otm.liquid]
    source = preferred if not preferred.empty else otm
    if source.empty:
        raise ValueError(f"No usable {option_type} strikes")
    return source.sort_values(["delta_distance", "spread_pct"]).iloc[0]


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> float:
    vals = []
    for df in (calls, puts):
        if not df.empty:
            row = df.iloc[(df.strike - spot).abs().argsort()[:1]]
            if not row.empty:
                vals.append(float(row.iloc[0].impliedVolatility))
    if not vals:
        raise ValueError("No ATM IV")
    return float(np.mean(vals))


def _price_context(symbol: str, benchmark: str = "SPY") -> tuple[float, float, float, float]:
    def stats(s: str):
        hist = yf.Ticker(s).history(period="3mo", auto_adjust=True)
        if len(hist) < 22:
            raise ValueError(f"Insufficient price history for {s}")
        ret = float(hist.Close.iloc[-1] / hist.Close.iloc[-22] - 1)
        avg_vol = float(hist.Volume.iloc[-21:-1].mean())
        vr = float(hist.Volume.iloc[-1] / avg_vol) if avg_vol > 0 else np.nan
        return float(hist.Close.iloc[-1]), ret, vr
    spot, ret, vr = stats(symbol)
    _, spy_ret, _ = stats(benchmark)
    return spot, ret, ret - spy_ret, vr


def _earnings_distance(ticker: yf.Ticker) -> Optional[int]:
    try:
        cal = ticker.calendar
        if cal is None:
            return None
        earnings = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not earnings:
            return None
        dt = pd.Timestamp(earnings[0]).date()
        return (dt - date.today()).days
    except Exception:
        return None


def _quality(row_put: pd.Series, row_call: pd.Series, cfg: dict) -> tuple[float, str]:
    q = cfg["quality"]
    oi = min(float(row_put.openInterest), float(row_call.openInterest))
    spread = max(float(row_put.spread_pct), float(row_call.spread_pct))
    delta_error = max(abs(abs(float(row_put.delta_est)) - cfg["target_delta"]),
                      abs(abs(float(row_call.delta_est)) - cfg["target_delta"]))
    oi_score = min(1.0, oi / max(q["good_open_interest"], 1))
    spread_score = max(0.0, 1.0 - spread / q["max_spread_pct"])
    delta_score = max(0.0, 1.0 - delta_error / q["max_delta_error"])
    score = round(100 * (0.45 * oi_score + 0.35 * spread_score + 0.20 * delta_score), 1)
    label = "HIGH" if score >= 70 else "MEDIUM" if score >= 45 else "LOW"
    return score, label


def classify(return_1m: float, normalized_skew: float) -> str:
    if return_1m < 0 and normalized_skew < 0:
        return "CONTRARIAN BID"
    if return_1m >= 0 and normalized_skew < 0:
        return "CHASE"
    if return_1m >= 0 and normalized_skew >= 0:
        return "HEDGED RALLY"
    return "FEAR"


def analyze_symbol(symbol: str, cfg: dict) -> SkewResult:
    symbol = symbol.upper().strip()
    ticker = yf.Ticker(symbol)
    spot, ret, rel_ret, volume_ratio = _price_context(symbol, cfg["benchmark"])
    expiry, dte = _select_expiry(ticker, cfg["min_dte"], cfg["max_dte"])
    chain = ticker.option_chain(expiry)
    calls = _prepare_chain(chain.calls, spot, dte, "call", cfg["quality"]["min_open_interest"])
    puts = _prepare_chain(chain.puts, spot, dte, "put", cfg["quality"]["min_open_interest"])
    # OTM restriction is essential: call above spot, put below spot.
    call = _pick_delta(calls[calls.strike > spot], cfg["target_delta"], "call")
    put = _pick_delta(puts[puts.strike < spot], cfg["target_delta"], "put")
    atm_iv = _atm_iv(calls, puts, spot)
    raw = float(put.impliedVolatility - call.impliedVolatility)
    normalized = raw / atm_iv if atm_iv else np.nan
    ceiling = cfg["sanity_ceiling_normalized"]
    if not np.isfinite(normalized) or abs(normalized) > ceiling:
        raise ValueError(f"Skew {normalized:.3f} exceeds sanity ceiling {ceiling}")
    quality_score, quality_label = _quality(put, call, cfg)
    days_to_earnings = _earnings_distance(ticker)
    catalyst = days_to_earnings is not None and 0 <= days_to_earnings <= dte
    try:
        sector = ticker.info.get("sector", "Unknown") or "Unknown"
    except Exception:
        sector = "Unknown"
    return SkewResult(
        run_date=date.today().isoformat(), symbol=symbol, sector=sector, expiry=expiry, dte=dte,
        spot=spot, return_1m=ret, return_vs_spy_1m=rel_ret, volume_ratio=volume_ratio,
        atm_iv=atm_iv, put_iv=float(put.impliedVolatility), call_iv=float(call.impliedVolatility),
        raw_skew=raw, normalized_skew=normalized, put_delta=float(put.delta_est),
        call_delta=float(call.delta_est), put_strike=float(put.strike), call_strike=float(call.strike),
        put_oi=int(put.openInterest), call_oi=int(call.openInterest), put_volume=int(put.volume),
        call_volume=int(call.volume), put_spread_pct=float(put.spread_pct),
        call_spread_pct=float(call.spread_pct), quality_score=quality_score, quality_label=quality_label,
        catalyst_flag=catalyst, days_to_earnings=days_to_earnings, quadrant=classify(ret, normalized)
    )


def enrich_with_history(latest: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = latest.copy()
    for idx, row in out.iterrows():
        h = history[history.symbol == row.symbol].sort_values("run_date") if not history.empty else pd.DataFrame()
        values = pd.to_numeric(h.get("normalized_skew", pd.Series(dtype=float)), errors="coerce").dropna()
        current = float(row.normalized_skew)
        if len(values) >= 5:
            out.at[idx, "skew_change_5obs"] = current - float(values.iloc[-5])
        if len(values) >= 20:
            out.at[idx, "skew_change_20obs"] = current - float(values.iloc[-20])
        if len(values) >= 10:
            out.at[idx, "skew_percentile"] = float((values <= current).mean() * 100)
        # Highest score: weak price + rotation toward calls, adjusted by chain confidence.
        change = out.at[idx, "skew_change_5obs"]
        change = float(change) if pd.notna(change) else 0.0
        price_component = max(0.0, -float(row.return_vs_spy_1m))
        call_rotation = max(0.0, -change)
        out.at[idx, "divergence_score"] = round(
            100 * (0.55 * min(price_component / 0.15, 1) + 0.45 * min(call_rotation / 0.25, 1))
            * float(row.quality_score) / 100, 1
        )
    return out


def run_scan(symbols: list[str], cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()
    results, errors = [], []
    for symbol in dict.fromkeys(symbols):
        try:
            results.append(asdict(analyze_symbol(symbol, cfg)))
            print(f"OK {symbol}")
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})
            print(f"SKIP {symbol}: {exc}")
    latest = pd.DataFrame(results)
    if latest.empty:
        return latest, errors
    latest = enrich_with_history(latest, history)
    latest.to_csv(LATEST_PATH, index=False)
    combined = pd.concat([history, latest], ignore_index=True) if not history.empty else latest.copy()
    combined = combined.drop_duplicates(subset=["run_date", "symbol"], keep="last")
    combined.to_csv(HISTORY_PATH, index=False)
    pd.DataFrame(errors).to_csv("data/skew_errors.csv", index=False)
    return latest, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run options skew map")
    parser.add_argument("--symbols", nargs="*", help="Override configured universe")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    symbols = args.symbols or cfg["universe"]
    latest, errors = run_scan(symbols, cfg)
    if not latest.empty:
        cols = ["symbol", "quadrant", "return_1m", "normalized_skew", "quality_label", "catalyst_flag", "divergence_score"]
        print(latest[cols].sort_values("divergence_score", ascending=False).to_string(index=False))
    print(f"Completed: {len(latest)} names, {len(errors)} skipped")


if __name__ == "__main__":
    main()

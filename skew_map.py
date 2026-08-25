"""Options skew analytics engine.

Educational research tooling only. A skew reading is positioning/relative pricing,
not a prediction or trade recommendation.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from market_data_provider import YahooMarketDataProvider

DEFAULT_CONFIG = Path("skew_config.json")
HISTORY_PATH = Path("data/skew_history.csv")
LATEST_PATH = Path("data/skew_latest.csv")
PROVIDER = YahooMarketDataProvider()


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
    price_source: str
    options_source: str = "yfinance-validated-mid"
    skew_change_5obs: Optional[float] = None
    skew_change_20obs: Optional[float] = None
    skew_percentile: Optional[float] = None
    divergence_score: Optional[float] = None


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def bs_price(spot: float, strike: float, t: float, iv: float, option_type: str,
             rate: float = 0.04, dividend_yield: float = 0.0) -> float:
    if min(spot, strike, t, iv) <= 0:
        return np.nan
    root_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * iv * iv) * t) / (iv * root_t)
    d2 = d1 - iv * root_t
    disc_q = math.exp(-dividend_yield * t)
    disc_r = math.exp(-rate * t)
    if option_type == "call":
        return spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2)
    return strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, iv: float, option_type: str,
             rate: float = 0.04, dividend_yield: float = 0.0) -> float:
    if min(spot, strike, t, iv) <= 0:
        return np.nan
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    if option_type == "call":
        return float(math.exp(-dividend_yield * t) * norm.cdf(d1))
    return float(math.exp(-dividend_yield * t) * (norm.cdf(d1) - 1.0))


def implied_vol_from_mid(mid: float, spot: float, strike: float, t: float,
                         option_type: str, rate: float = 0.04) -> float:
    """Recover IV from a validated option mid-price using Black-Scholes."""
    if not all(np.isfinite(x) for x in (mid, spot, strike, t)) or min(mid, spot, strike, t) <= 0:
        return np.nan
    intrinsic = max(0.0, spot - strike * math.exp(-rate * t)) if option_type == "call" else \
        max(0.0, strike * math.exp(-rate * t) - spot)
    if mid <= intrinsic + 1e-6 or mid >= spot:
        return np.nan
    try:
        f = lambda vol: bs_price(spot, strike, t, vol, option_type, rate=rate) - mid
        return float(brentq(f, 0.01, 5.0, maxiter=100))
    except Exception:
        return np.nan


def _quote_metrics(row: pd.Series) -> tuple[float, float]:
    bid = float(pd.to_numeric(row.get("bid", np.nan), errors="coerce"))
    ask = float(pd.to_numeric(row.get("ask", np.nan), errors="coerce"))
    if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0 or ask <= 0 or ask < bid:
        return np.nan, np.inf
    mid = (bid + ask) / 2
    spread = (ask - bid) / mid if mid > 0 else np.inf
    return mid, spread


def _prepare_chain(df: pd.DataFrame, spot: float, dte: int, option_type: str, cfg: dict) -> pd.DataFrame:
    q = cfg["quality"]
    out = df.copy()
    for col in ("strike", "bid", "ask", "openInterest", "volume"):
        out[col] = pd.to_numeric(out.get(col, np.nan), errors="coerce")
    out["openInterest"] = out["openInterest"].fillna(0)
    out["volume"] = out["volume"].fillna(0)
    out[["mid", "spread_pct"]] = out.apply(
        lambda r: pd.Series(_quote_metrics(r)), axis=1
    )
    max_spread = q.get("hard_max_spread_pct", q["max_spread_pct"])
    min_mid = q.get("min_option_mid", 0.05)
    out = out[
        out["strike"].gt(0)
        & out["mid"].ge(min_mid)
        & out["spread_pct"].le(max_spread)
        & out["openInterest"].ge(q["min_open_interest"])
    ].copy()
    if out.empty:
        return out
    t = dte / 365.0
    out["iv_calc"] = out.apply(
        lambda r: implied_vol_from_mid(float(r.mid), spot, float(r.strike), t, option_type), axis=1
    )
    out = out[out.iv_calc.between(q.get("min_iv", 0.03), q.get("max_iv", 3.0))].copy()
    out["delta_est"] = out.apply(
        lambda r: bs_delta(spot, float(r.strike), t, float(r.iv_calc), option_type), axis=1
    )
    out = out[np.isfinite(out.delta_est)].copy()
    out["liquid"] = out.openInterest >= q["good_open_interest"]
    return out


def _pick_delta(df: pd.DataFrame, target_abs_delta: float, option_type: str, cfg: dict) -> pd.Series:
    if df.empty:
        raise ValueError(f"No validated {option_type} quotes")
    target = target_abs_delta if option_type == "call" else -target_abs_delta
    out = df.copy()
    out["delta_distance"] = (out.delta_est - target).abs()
    max_error = cfg["quality"]["max_delta_error"]
    out = out[out.delta_distance <= max_error]
    if out.empty:
        raise ValueError(f"No {option_type} quote close enough to {target_abs_delta:.0%} delta")
    preferred = out[out.liquid]
    source = preferred if not preferred.empty else out
    return source.sort_values(["delta_distance", "spread_pct", "openInterest"], ascending=[True, True, False]).iloc[0]


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, cfg: dict) -> float:
    candidates = []
    for df in (calls, puts):
        if not df.empty:
            tmp = df.copy()
            tmp["moneyness_distance"] = (tmp.strike - spot).abs() / spot
            tmp = tmp[tmp.moneyness_distance <= cfg["quality"].get("max_atm_moneyness", 0.03)]
            if not tmp.empty:
                candidates.append(float(tmp.sort_values(["moneyness_distance", "spread_pct"]).iloc[0].iv_calc))
    if not candidates:
        raise ValueError("No validated ATM quote for normalization")
    return float(np.mean(candidates))


def _price_context(symbol: str, benchmark: str) -> tuple[float, float, float, float, str]:
    stock = PROVIDER.price_context(symbol)
    bench = PROVIDER.price_context(benchmark)
    return stock.spot, stock.return_1m, stock.return_1m - bench.return_1m, stock.volume_ratio, stock.source


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
    ticker = PROVIDER.ticker(symbol)
    spot, ret, rel_ret, volume_ratio, price_source = _price_context(symbol, cfg["benchmark"])
    expiry, dte = PROVIDER.select_expiry(ticker, cfg["min_dte"], cfg["max_dte"])
    chain = PROVIDER.option_chain(ticker, expiry)
    calls = _prepare_chain(chain.calls, spot, dte, "call", cfg)
    puts = _prepare_chain(chain.puts, spot, dte, "put", cfg)
    call = _pick_delta(calls[calls.strike > spot], cfg["target_delta"], "call", cfg)
    put = _pick_delta(puts[puts.strike < spot], cfg["target_delta"], "put", cfg)
    atm_iv = _atm_iv(calls, puts, spot, cfg)
    raw = float(put.iv_calc - call.iv_calc)
    normalized = raw / atm_iv if atm_iv else np.nan
    ceiling = cfg["sanity_ceiling_normalized"]
    if not np.isfinite(normalized) or abs(normalized) > ceiling:
        raise ValueError(f"Skew {normalized:.3f} exceeds sanity ceiling {ceiling}")
    quality_score, quality_label = _quality(put, call, cfg)
    days_to_earnings = PROVIDER.earnings_distance(ticker)
    catalyst = days_to_earnings is not None and 0 <= days_to_earnings <= dte
    return SkewResult(
        run_date=date.today().isoformat(), symbol=symbol, sector=PROVIDER.sector(ticker), expiry=expiry, dte=dte,
        spot=spot, return_1m=ret, return_vs_spy_1m=rel_ret, volume_ratio=volume_ratio,
        atm_iv=atm_iv, put_iv=float(put.iv_calc), call_iv=float(call.iv_calc), raw_skew=raw,
        normalized_skew=normalized, put_delta=float(put.delta_est), call_delta=float(call.delta_est),
        put_strike=float(put.strike), call_strike=float(call.strike), put_oi=int(put.openInterest),
        call_oi=int(call.openInterest), put_volume=int(put.volume), call_volume=int(call.volume),
        put_spread_pct=float(put.spread_pct), call_spread_pct=float(call.spread_pct),
        quality_score=quality_score, quality_label=quality_label, catalyst_flag=catalyst,
        days_to_earnings=days_to_earnings, quadrant=classify(ret, normalized), price_source=price_source
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
    pd.DataFrame(errors).to_csv("data/skew_errors.csv", index=False)
    if latest.empty:
        return latest, errors
    latest = enrich_with_history(latest, history)
    latest.to_csv(LATEST_PATH, index=False)
    combined = pd.concat([history, latest], ignore_index=True) if not history.empty else latest.copy()
    combined = combined.drop_duplicates(subset=["run_date", "symbol"], keep="last")
    combined.to_csv(HISTORY_PATH, index=False)
    return latest, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run options skew map")
    parser.add_argument("--symbols", nargs="*", help="Override configured universe")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = load_config(Path(args.config))
    latest, errors = run_scan(args.symbols or cfg["universe"], cfg)
    if not latest.empty:
        cols = ["symbol", "quadrant", "return_1m", "normalized_skew", "quality_label", "catalyst_flag", "divergence_score"]
        print(latest[cols].sort_values("divergence_score", ascending=False).to_string(index=False))
    print(f"Completed: {len(latest)} names, {len(errors)} skipped")


if __name__ == "__main__":
    main()

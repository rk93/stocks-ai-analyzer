"""Trending-stock screener using free Yahoo market data.

Educational research only. The RS percentile is our own cross-sectional momentum
proxy and is not IBD's proprietary RS Rating.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CONFIG = Path("skew_config.json")
OUT = Path("data/trending_latest.csv")
CONTEXT = Path("data/market_context.json")

SECTOR_ETFS = {
    "Technology": "XLK", "Financial Services": "XLF", "Energy": "XLE",
    "Healthcare": "XLV", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Industrials": "XLI", "Utilities": "XLU", "Real Estate": "XLRE",
    "Basic Materials": "XLB", "Communication Services": "XLC",
}
MARKET_TICKERS = {"S&P futures": "ES=F", "Nasdaq futures": "NQ=F", "Dow futures": "YM=F", "VIX": "^VIX"}


def _safe_num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _ret(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def _download(symbols: list[str], period="1y") -> dict[str, pd.DataFrame]:
    data = yf.download(symbols, period=period, auto_adjust=True, group_by="ticker", threads=True, progress=False)
    out = {}
    for s in symbols:
        try:
            if len(symbols) == 1:
                df = data.copy()
            else:
                df = data[s].copy()
            if not df.empty and "Close" in df:
                out[s] = df.dropna(how="all")
        except Exception:
            continue
    return out


def market_context() -> dict:
    rows = []
    for label, symbol in MARKET_TICKERS.items():
        try:
            h = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
            last = float(h.Close.iloc[-1])
            chg = float(last / h.Close.iloc[-2] - 1) if len(h) >= 2 else np.nan
            rows.append({"label": label, "symbol": symbol, "value": last, "change_1d": chg})
        except Exception as exc:
            rows.append({"label": label, "symbol": symbol, "value": None, "change_1d": None, "error": str(exc)})
    vix = next((x for x in rows if x["label"] == "VIX"), None)
    regime = "UNKNOWN"
    if vix and vix.get("value") is not None:
        vv = float(vix["value"])
        regime = "LOW FEAR" if vv < 15 else "NORMAL" if vv < 20 else "ELEVATED" if vv < 30 else "HIGH FEAR"
    payload = {"market": rows, "vix_regime": regime}
    CONTEXT.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def sector_rotation(histories: dict[str, pd.DataFrame]) -> dict[str, float]:
    result = {}
    for sector, etf in SECTOR_ETFS.items():
        h = histories.get(etf)
        result[sector] = _ret(h.Close, 5) if h is not None and not h.empty else np.nan
    return result


def technical_row(symbol: str, h: pd.DataFrame) -> dict:
    close = h.Close.dropna()
    vol = h.Volume.fillna(0) if "Volume" in h else pd.Series(dtype=float)
    if len(close) < 200:
        raise ValueError("<200 sessions")
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    sma50_prev = float(close.rolling(50).mean().iloc[-6]) if len(close) >= 205 else np.nan
    sma200_prev = float(close.rolling(200).mean().iloc[-6]) if len(close) >= 205 else np.nan
    avg20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else np.nan
    volume_ratio = float(vol.iloc[-1] / avg20) if avg20 and avg20 > 0 else np.nan
    high_52w_prior = float(close.iloc[-253:-1].max()) if len(close) >= 253 else float(close.iloc[:-1].max())
    ath_distance = float(close.iloc[-1] / high_52w_prior - 1) if high_52w_prior else np.nan
    breakout = bool(close.iloc[-1] >= high_52w_prior and volume_ratio >= 1.5)
    golden = bool(sma50 > sma200)
    golden_recent = bool(np.isfinite(sma50_prev) and np.isfinite(sma200_prev) and sma50_prev <= sma200_prev and sma50 > sma200)
    return {
        "symbol": symbol, "spot": float(close.iloc[-1]), "return_1m": _ret(close, 21),
        "return_3m": _ret(close, 63), "return_6m": _ret(close, 126),
        "volume_ratio": volume_ratio, "sma50": sma50, "sma200": sma200,
        "above_50d": bool(close.iloc[-1] > sma50), "above_200d": bool(close.iloc[-1] > sma200),
        "golden_cross": golden, "golden_cross_recent": golden_recent,
        "distance_from_52w_high": ath_distance, "breakout": breakout,
    }


def enrich_fundamentals(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    out = df.copy()
    cols = ["sector", "revenue_growth", "earnings_growth", "profit_margin", "free_cashflow",
            "debt_to_cash", "institutional_ownership", "analyst_recommendation", "analyst_count"]
    for c in cols:
        if c not in out: out[c] = np.nan if c != "sector" and c != "analyst_recommendation" else None
    idxs = out.sort_values("technical_score", ascending=False).head(top_n).index
    for idx in idxs:
        symbol = out.at[idx, "symbol"]
        try:
            info = yf.Ticker(symbol).info or {}
            cash = _safe_num(info.get("totalCash")); debt = _safe_num(info.get("totalDebt"))
            out.at[idx, "sector"] = info.get("sector") or "Unknown"
            out.at[idx, "revenue_growth"] = _safe_num(info.get("revenueGrowth"))
            out.at[idx, "earnings_growth"] = _safe_num(info.get("earningsGrowth"))
            out.at[idx, "profit_margin"] = _safe_num(info.get("profitMargins"))
            out.at[idx, "free_cashflow"] = _safe_num(info.get("freeCashflow"))
            out.at[idx, "debt_to_cash"] = debt / cash if np.isfinite(debt) and np.isfinite(cash) and cash > 0 else np.nan
            out.at[idx, "institutional_ownership"] = _safe_num(info.get("heldPercentInstitutions"))
            out.at[idx, "analyst_recommendation"] = info.get("recommendationKey")
            out.at[idx, "analyst_count"] = _safe_num(info.get("numberOfAnalystOpinions"))
        except Exception:
            continue
    return out


def score_rows(df: pd.DataFrame, sector_5d: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    # Our own RS proxy: 6m return percentile among the scanned universe.
    out["rs_percentile"] = out["return_6m"].rank(pct=True, method="average") * 100
    scores, reasons = [], []
    for _, r in out.iterrows():
        score = 0.0; why = []
        vr = _safe_num(r.get("volume_ratio")); rs = _safe_num(r.get("rs_percentile"))
        if np.isfinite(vr) and vr >= 2: score += min(18, 9 + (vr - 2) * 3); why.append(f"volume {vr:.1f}x")
        elif np.isfinite(vr) and vr >= 1.5: score += 6
        if np.isfinite(rs) and rs >= 80: score += 20; why.append(f"RS proxy {rs:.0f}")
        elif np.isfinite(rs) and rs >= 65: score += 10
        if bool(r.get("above_50d")): score += 8
        if bool(r.get("above_200d")): score += 8
        if bool(r.get("golden_cross")): score += 6
        if bool(r.get("golden_cross_recent")): score += 5; why.append("recent golden cross")
        if bool(r.get("breakout")): score += 18; why.append("52w breakout + volume")
        elif _safe_num(r.get("distance_from_52w_high")) >= -0.03: score += 6; why.append("near 52w high")
        sector = r.get("sector")
        sret = sector_5d.get(sector, np.nan) if isinstance(sector, str) else np.nan
        if np.isfinite(sret) and sret > 0.01: score += 7; why.append("sector inflow proxy")
        rev = _safe_num(r.get("revenue_growth")); eg = _safe_num(r.get("earnings_growth")); pm = _safe_num(r.get("profit_margin")); fcf = _safe_num(r.get("free_cashflow")); dc = _safe_num(r.get("debt_to_cash"))
        fundamental_points = 0
        if np.isfinite(rev) and rev > .20: fundamental_points += 8; why.append("revenue >20%")
        if np.isfinite(eg) and eg > .20: fundamental_points += 8; why.append("earnings >20%")
        if np.isfinite(pm) and pm > 0: fundamental_points += 3
        if np.isfinite(fcf) and fcf > 0: fundamental_points += 5
        if np.isfinite(dc) and dc < 2: fundamental_points += 3
        score += min(22, fundamental_points)
        rec = str(r.get("analyst_recommendation") or "").lower()
        if rec in {"strong_buy", "buy"}: score += 4; why.append("analyst consensus positive")
        scores.append(round(min(100, score), 1)); reasons.append("; ".join(why) if why else "no strong trend confirmations")
    out["trend_score"] = scores; out["trend_reason"] = reasons
    out["trend_state"] = np.select(
        [out.trend_score >= 70, out.trend_score >= 55, out.trend_score >= 40],
        ["STRONG TREND", "TRENDING", "WATCH"], default="WEAK / MIXED")
    return out


def build() -> pd.DataFrame:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    symbols = [x for x in cfg["universe"] if x not in {"SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","SMH","SOXX","ARKK"}]
    download_symbols = list(dict.fromkeys(symbols + list(SECTOR_ETFS.values())))
    histories = _download(download_symbols, "1y")
    rows = []
    for symbol in symbols:
        try: rows.append(technical_row(symbol, histories[symbol]))
        except Exception: continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Preliminary technical score determines which names receive slower fundamentals calls.
    df["technical_score"] = (
        df["return_6m"].rank(pct=True).fillna(0) * 35
        + df["above_50d"].astype(int) * 12 + df["above_200d"].astype(int) * 12
        + np.minimum(df["volume_ratio"].fillna(0), 5) / 5 * 20
        + df["breakout"].astype(int) * 21
    ).round(1)
    df = enrich_fundamentals(df)
    sectors = sector_rotation(histories)
    df["sector_5d"] = df["sector"].map(sectors)
    df = score_rows(df, sectors).sort_values(["trend_score", "rs_percentile"], ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    market_context()
    return df


if __name__ == "__main__":
    result = build()
    if not result.empty:
        cols = ["symbol","trend_state","trend_score","rs_percentile","volume_ratio","above_50d","above_200d","breakout","trend_reason"]
        print(result[cols].head(30).to_string(index=False))

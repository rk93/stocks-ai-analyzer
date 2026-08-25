"""Free market-data adapters used by the skew scanner.

The primary source is Yahoo Finance through yfinance. yahooquery is used only as a
secondary price-history fallback. Options are deliberately validated here before
being passed to the analytics layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from yahooquery import Ticker as YQTicker
except Exception:  # optional at import time for local compatibility
    YQTicker = None


@dataclass
class PriceContext:
    spot: float
    return_1m: float
    volume_ratio: float
    source: str


class YahooMarketDataProvider:
    """Yahoo-backed provider with conservative fallbacks and validation."""

    def price_context(self, symbol: str) -> PriceContext:
        try:
            hist = yf.Ticker(symbol).history(period="3mo", auto_adjust=True)
            return self._price_context_from_frame(hist, "yfinance")
        except Exception as first_error:
            if YQTicker is None:
                raise first_error
            try:
                hist = YQTicker(symbol).history(period="3mo", interval="1d")
                if isinstance(hist.index, pd.MultiIndex):
                    hist = hist.reset_index(level=0, drop=True)
                hist = hist.rename(columns={"close": "Close", "volume": "Volume"})
                return self._price_context_from_frame(hist, "yahooquery")
            except Exception:
                raise first_error

    @staticmethod
    def _price_context_from_frame(hist: pd.DataFrame, source: str) -> PriceContext:
        if hist is None or len(hist) < 22:
            raise ValueError("Insufficient price history")
        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
        volume = pd.to_numeric(hist["Volume"], errors="coerce")
        if len(close) < 22:
            raise ValueError("Insufficient valid closes")
        ret = float(close.iloc[-1] / close.iloc[-22] - 1)
        recent_volume = volume.tail(21)
        avg_vol = float(recent_volume.iloc[:-1].mean()) if len(recent_volume) >= 2 else np.nan
        vr = float(recent_volume.iloc[-1] / avg_vol) if np.isfinite(avg_vol) and avg_vol > 0 else np.nan
        return PriceContext(float(close.iloc[-1]), ret, vr, source)

    def ticker(self, symbol: str) -> yf.Ticker:
        return yf.Ticker(symbol)

    @staticmethod
    def select_expiry(ticker: yf.Ticker, min_dte: int, max_dte: int) -> tuple[str, int]:
        today = date.today()
        candidates = []
        for exp in ticker.options:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
            if min_dte <= dte <= max_dte:
                candidates.append((abs(dte - (min_dte + max_dte) / 2), exp, dte))
        if not candidates:
            raise ValueError(f"No expiry in configured {min_dte}-{max_dte} DTE window")
        _, expiry, dte = min(candidates)
        return expiry, dte

    @staticmethod
    def option_chain(ticker: yf.Ticker, expiry: str):
        return ticker.option_chain(expiry)

    @staticmethod
    def sector(ticker: yf.Ticker) -> str:
        try:
            return ticker.info.get("sector", "Unknown") or "Unknown"
        except Exception:
            return "Unknown"

    @staticmethod
    def earnings_distance(ticker: yf.Ticker) -> Optional[int]:
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

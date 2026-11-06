"""Conservative, standalone long-only execution research (no broker connections).

Bars are timestamped at their *open*, must be timezone aware, and are restricted
to US regular hours. ``signal_time`` is when all signal inputs were available,
not the label of an unfinished candle. For a 10:00 five-minute candle it is
10:05. A fill at the next eligible open is executable, not the signal close.
``frozen_at`` can impose a later, prospective publication cutoff.

All exits are triggered market orders, including targets: spread and adverse
slippage apply to both sides. OHLC collisions resolve stop-first; trailing stops
and trend exits cannot use a bar's high/close before that bar is complete. The
caller must supply consistently adjusted point-in-time prices and a window
which does not borrow observations from a held-out split. No missing history,
correlations or executable intrabar ordering is invented here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from market_sessions import next_session_open


@dataclass(frozen=True)
class ExecutionConfig:
    initial_capital: float = 10_000.0
    risk_fraction: float = 0.005
    max_position_dollars: float = 1_000.0
    max_open_positions: int = 5
    max_total_risk_fraction: float = 0.025
    max_sector_fraction: float | None = 0.30
    max_abs_correlation: float | None = 0.85
    slippage_bps: float = 7.0
    spread_bps: float = 2.0
    commission_bps: float = 0.0
    commission_per_order: float = 0.0
    stop_model: str = "signal"  # signal, fixed, atr
    fixed_stop_pct: float = 0.02
    atr_multiple: float = 2.0
    target_r: float | None = 1.0
    trailing_atr_multiple: float | None = None
    trailing_pct: float | None = None
    partial_fraction: float = 0.0
    partial_target_r: float = 1.0
    max_holding_sessions: int | None = 5
    max_holding_bars: int | None = None
    trend_exit_column: str | None = None
    bar_minutes: int = 5
    execution_delay_bars: int = 0
    max_entry_delay_minutes: float | None = 30.0
    require_contiguous_bars: bool = True
    allow_fractional_shares: bool = False
    market_timezone: str = "America/New_York"
    cash_annual_rate: float = 0.0
    min_daily_returns_for_ratios: int = 60

    def __post_init__(self):
        for name in ("initial_capital", "risk_fraction", "max_position_dollars",
                     "max_total_risk_fraction", "fixed_stop_pct", "atr_multiple"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("slippage_bps", "spread_bps", "commission_bps", "commission_per_order"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.slippage_bps + self.spread_bps / 2 >= 10_000:
            raise ValueError("execution friction must be below 100%")
        if self.commission_bps >= 10_000 or self.fixed_stop_pct >= 1:
            raise ValueError("commission and fixed stop must be below 100%")
        if self.stop_model not in {"signal", "fixed", "atr"}:
            raise ValueError("stop_model must be signal, fixed or atr")
        for name in ("target_r", "trailing_atr_multiple", "trailing_pct", "max_entry_delay_minutes"):
            val = getattr(self, name)
            if val is not None and (not math.isfinite(val) or val <= 0):
                raise ValueError(f"{name} must be positive or None")
        if self.trailing_pct is not None and self.trailing_pct >= 1:
            raise ValueError("trailing_pct must be below 1")
        if not 0 <= self.partial_fraction < 1 or self.partial_target_r <= 0:
            raise ValueError("invalid partial-exit parameters")
        if self.partial_fraction and (self.target_r is not None and self.partial_target_r >= self.target_r):
            raise ValueError("partial target must precede final target")
        if self.max_sector_fraction is not None and not 0 < self.max_sector_fraction <= 1:
            raise ValueError("max_sector_fraction must be in (0, 1] or None")
        if self.max_abs_correlation is not None and not 0 < self.max_abs_correlation <= 1:
            raise ValueError("max_abs_correlation must be in (0, 1] or None")
        if not math.isfinite(self.cash_annual_rate) or self.cash_annual_rate <= -1:
            raise ValueError("cash_annual_rate must be finite and above -1")
        for name in ("bar_minutes", "max_open_positions", "min_daily_returns_for_ratios"):
            if getattr(self, name) < 1 or int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_holding_sessions", "max_holding_bars"):
            val = getattr(self, name)
            if val is not None and (val < 1 or int(val) != val):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.execution_delay_bars < 0 or int(self.execution_delay_bars) != self.execution_delay_bars:
            raise ValueError("execution_delay_bars must be a nonnegative integer")


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError("timestamps must be valid and timezone aware")
    return ts.tz_convert("UTC")


def _iso(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def _number(value: Any) -> float | None:
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def _bars(frame: pd.DataFrame, cfg: ExecutionConfig, end_time=None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC"))
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("OHLC requires a timezone-aware DatetimeIndex of bar opens")
    df = frame.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("duplicate OHLC columns")
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError("open, high, low and close columns are required")
    df.index = df.index.tz_convert("UTC")
    if df.index.has_duplicates:
        raise ValueError("duplicate OHLC timestamps")
    df = df.sort_index()
    local = df.index.tz_convert(cfg.market_timezone)
    minutes = local.hour * 60 + local.minute
    df = df[(local.dayofweek < 5) & (minutes >= 570) & (minutes < 960)]
    if end_time is not None:
        df = df[df.index + pd.Timedelta(minutes=cfg.bar_minutes) <= _timestamp(end_time)]
    values = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise ValueError("OHLC prices must be finite and positive")
    if ((values.high < values[["open", "close", "low"]].max(axis=1)) |
        (values.low > values[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC high/low are inconsistent")
    df[["open", "high", "low", "close"]] = values
    return df


def _fill(price: float, side: str, cfg: ExecutionConfig) -> float:
    adverse = (cfg.slippage_bps + cfg.spread_bps / 2) / 10_000
    return price * (1 + adverse if side == "buy" else 1 - adverse)


def _commission(price: float, qty: float, cfg: ExecutionConfig) -> float:
    return price * qty * cfg.commission_bps / 10_000 + cfg.commission_per_order


def _prepare(df: pd.DataFrame, signal: Mapping[str, Any], cfg: ExecutionConfig) -> dict:
    signal_time = _timestamp(signal["signal_time"])
    available = max(signal_time, _timestamp(signal["frozen_at"])) if signal.get("frozen_at") else signal_time
    pos = int(df.index.searchsorted(available, side="left")) + cfg.execution_delay_bars
    correlations = {}
    for symbol, value in (signal.get("correlations") or {}).items():
        corr = _number(value)
        if corr is not None:
            if abs(corr) > 1:
                raise ValueError("correlations must be within [-1, 1]")
            correlations[str(symbol)] = corr
    common = {"signal_time": _iso(signal_time), "available_at": _iso(available),
              "signal_id": str(signal.get("signal_id", f"{signal.get('symbol', '')}|{signal_time.isoformat()}")),
              "symbol": str(signal.get("symbol", "")), "sector": str(signal.get("sector") or "UNKNOWN"),
              "score": _number(signal.get("score")),
              "market_regime": str(signal["market_regime"]) if pd.notna(signal.get("market_regime")) else None,
              "correlations": correlations}
    if pos >= len(df):
        return {**common, "status": "REJECTED", "reason": "NO_EXECUTABLE_BAR"}
    entry_time = df.index[pos]
    if cfg.max_entry_delay_minutes is not None and (entry_time - available).total_seconds() / 60 > cfg.max_entry_delay_minutes:
        return {**common, "status": "REJECTED", "reason": "STALE_ENTRY"}
    raw_entry = float(df.iloc[pos].open)
    entry = _fill(raw_entry, "buy", cfg)
    atr = _number(signal.get("atr"))
    if cfg.stop_model == "fixed":
        stop = raw_entry * (1 - cfg.fixed_stop_pct)
    elif cfg.stop_model == "atr":
        stop = raw_entry - cfg.atr_multiple * atr if atr is not None and atr > 0 else None
    else:
        stop = _number(signal.get("stop_price", signal.get("stop")))
    if stop is None or stop <= 0 or stop >= raw_entry:
        return {**common, "status": "REJECTED", "reason": "INVALID_STOP_AT_FILL"}
    if cfg.trailing_atr_multiple is not None and (atr is None or atr <= 0):
        return {**common, "status": "REJECTED", "reason": "MISSING_FROZEN_ATR"}
    return {**common, "status": "READY", "entry_time": _iso(entry_time), "entry_price": entry,
            "raw_entry_price": raw_entry, "stop_price": stop, "atr": atr, "entry_index": pos,
            "initial_risk_per_share": entry - stop,
            "net_stop_risk_per_share": entry - _fill(stop, "sell", cfg) +
            (entry + _fill(stop, "sell", cfg)) * cfg.commission_bps / 10_000}


def _simulate(df: pd.DataFrame, prepared: dict, qty: float, cfg: ExecutionConfig) -> dict:
    result = {k: v for k, v in prepared.items() if k != "entry_index"}
    if prepared["status"] == "REJECTED":
        return result
    qty = float(qty)
    if not math.isfinite(qty) or qty <= 0 or (not cfg.allow_fractional_shares and qty != math.floor(qty)):
        raise ValueError("quantity must be positive (and integral unless fractional shares enabled)")
    entry = prepared["entry_price"]
    risk = prepared["initial_risk_per_share"]
    target = entry + risk * cfg.target_r if cfg.target_r is not None else None
    partial_target = entry + risk * cfg.partial_target_r
    stop = prepared["stop_price"]
    remaining, partial_done, pending_trend = qty, False, False
    entry_commission = _commission(entry, qty, cfg)
    legs: list[dict] = []
    sessions: list[Any] = []
    ambiguous = False
    data_quality_flags: list[str] = []
    unresolved_from = None
    gap_details = None
    start = prepared["entry_index"]
    last_i = start

    def sell(raw_price, amount, reason, ts, phase, bar_ts):
        nonlocal remaining
        price = _fill(float(raw_price), "sell", cfg)
        legs.append({"time": _iso(ts), "bar_time": _iso(bar_ts), "phase": phase,
                     "quantity": float(amount), "price": price, "raw_price": float(raw_price),
                     "commission": _commission(price, amount, cfg), "reason": reason})
        remaining = max(0.0, remaining - amount)

    for i in range(start, len(df)):
        last_i = i
        ts, bar = df.index[i], df.iloc[i]
        end = ts + pd.Timedelta(minutes=cfg.bar_minutes)
        session = ts.tz_convert(cfg.market_timezone).date()
        if cfg.require_contiguous_bars and i > start:
            previous_ts, previous_bar = df.index[i - 1], df.iloc[i - 1]
            previous_end = previous_ts + pd.Timedelta(minutes=cfg.bar_minutes)
            previous_local = previous_end.tz_convert(cfg.market_timezone)
            previous_flag = previous_bar.get("session_end", False)
            previous_session_complete = (pd.notna(previous_flag) and bool(previous_flag)) or (
                previous_local.hour * 60 + previous_local.minute >= 960)
            if previous_ts.tz_convert(cfg.market_timezone).date() == session:
                missing = ts != previous_end
            else:
                expected_open = next_session_open(previous_ts)
                missing = not previous_session_complete or expected_open is None or ts != expected_open
            if missing:
                # Unknown bars could already have exited the trade. Do not use
                # a subsequent recovery/target to manufacture a resolved path.
                data_quality_flags.append("MISSING_BARS_UNRESOLVED")
                unresolved_from = previous_end
                gap_details = {"previous_bar": _iso(previous_ts), "next_observation": _iso(ts)}
                last_i = i - 1
                break
        if not sessions or sessions[-1] != session:
            sessions.append(session)
        # Opening gaps and previously scheduled close-known trend exits are executable now.
        if float(bar.open) <= stop:
            sell(bar.open, remaining, "STOP_GAP", ts, "open", ts)
            break
        if pending_trend:
            sell(bar.open, remaining, "TREND", ts, "open", ts)
            break
        # Opening prints have known ordering before later highs/lows. Credit
        # no positive gap improvement, but do not invent a stop-first collision
        # when a target was already executable at the open.
        if cfg.partial_fraction and not partial_done and float(bar.open) >= partial_target:
            amount = qty * cfg.partial_fraction
            if not cfg.allow_fractional_shares:
                amount = math.floor(amount)
            if amount > 0:
                sell(partial_target, amount, "PARTIAL_TARGET", ts, "open", ts)
            partial_done = True
        if target is not None and float(bar.open) >= target:
            sell(target, remaining, "TARGET", ts, "open", ts)
            break
        # If both barriers are touched, no favourable OHLC ordering is inferred.
        hit_stop = float(bar.low) <= stop
        first_target = partial_target if cfg.partial_fraction and not partial_done else target
        hit_target = first_target is not None and float(bar.high) >= first_target
        if hit_stop:
            ambiguous = ambiguous or hit_target
            sell(stop, remaining, "STOP_AMBIGUOUS" if hit_target else "STOP", end, "intrabar", ts)
            break
        if cfg.partial_fraction and not partial_done and float(bar.high) >= partial_target:
            amount = qty * cfg.partial_fraction
            if not cfg.allow_fractional_shares:
                amount = math.floor(amount)
            if amount > 0:
                sell(partial_target, amount, "PARTIAL_TARGET", end, "intrabar", ts)
            partial_done = True
        if target is not None and float(bar.high) >= target:
            # Even an opening gap above a target is credited only at the target.
            sell(target, remaining, "TARGET", ts if bar.open >= target else end,
                 "open" if bar.open >= target else "intrabar", ts)
            break
        held_bars = i - start + 1
        end_local = end.tz_convert(cfg.market_timezone)
        session_end_flag = bar.get("session_end", False)
        known_session_end = (pd.notna(session_end_flag) and bool(session_end_flag)) or (end_local.hour * 60 + end_local.minute >= 960)
        time_due = (cfg.max_holding_bars is not None and held_bars >= cfg.max_holding_bars) or (
            cfg.max_holding_sessions is not None and len(sessions) >= cfg.max_holding_sessions and known_session_end)
        if time_due:
            sell(bar.close, remaining, "TIME", end, "close", ts)
            break
        # Any updated stop first becomes active in the following bar.
        if cfg.trailing_atr_multiple is not None:
            stop = max(stop, float(bar.high) - cfg.trailing_atr_multiple * prepared["atr"])
        if cfg.trailing_pct is not None:
            stop = max(stop, float(bar.high) * (1 - cfg.trailing_pct))
        col = cfg.trend_exit_column.lower() if cfg.trend_exit_column else None
        trend_value = bar.get(col, False) if col else False
        pending_trend = pd.notna(trend_value) and bool(trend_value)

    sold = qty - remaining
    sale_value = sum(x["price"] * x["quantity"] for x in legs)
    exit_commissions = sum(x["commission"] for x in legs)
    realised = sale_value - entry * sold - exit_commissions - entry_commission * sold / qty
    mark = float(df.iloc[last_i].close)
    mark_time = df.index[last_i] + pd.Timedelta(minutes=cfg.bar_minutes)
    unrealised = (mark - entry) * remaining - entry_commission * remaining / qty
    closed = remaining <= 1e-10
    exit_ts = _timestamp(legs[-1]["time"]) if closed else None
    holding_end = exit_ts if closed else mark_time
    friction = (entry - prepared["raw_entry_price"]) * qty + sum(
        (x["raw_price"] - x["price"]) * x["quantity"] for x in legs)
    return {**result, "status": "CLOSED" if closed else "OPEN", "quantity": qty,
            "remaining_quantity": remaining, "target_price": target, "last_stop_price": stop,
            "entry_commission": entry_commission, "initial_risk": risk * qty,
            "net_stop_risk": prepared["net_stop_risk_per_share"] * qty + 2 * cfg.commission_per_order,
            "exit_time": _iso(exit_ts), "exit_price": sale_value / sold if sold else None,
            "exit_reason": legs[-1]["reason"] if closed else None, "ambiguous": ambiguous,
            "data_quality_flags": data_quality_flags, "unresolved_from": _iso(unresolved_from), "gap_details": gap_details,
            "net_pnl": realised, "unrealized_pnl": unrealised if not closed else 0.0,
            "net_r": realised / (risk * qty) if closed else None,
            "gross_pnl": sum((x["raw_price"] - prepared["raw_entry_price"]) * x["quantity"] for x in legs),
            "commissions": entry_commission + exit_commissions, "slippage_spread_cost": friction,
            "transaction_cost": entry_commission + exit_commissions + friction,
            "return_pct": realised / (entry * qty + entry_commission) if closed else None,
            "bars_held": last_i - start + 1, "holding_sessions": len(sessions),
            "holding_minutes": (holding_end - _timestamp(prepared["entry_time"])).total_seconds() / 60,
            "mark_price": mark, "mark_time": _iso(mark_time), "legs": legs}


def simulate_trade(bars: pd.DataFrame, signal_time, stop_price=None, atr=None,
                   config: ExecutionConfig | None = None, quantity: float = 1,
                   *, signal: Mapping[str, Any] | None = None, end_time=None) -> dict:
    """Replay one frozen signal. An incomplete path remains OPEN, never forced closed.

    Optional ``signal`` adds symbol, signal_id, sector, score, market_regime and
    frozen_at metadata. A supplied end_time excludes unfinished bars at that
    cutoff. Invalid datasets raise ValueError; unexecutable signals are rejected.
    """
    cfg = config or ExecutionConfig()
    s = dict(signal or {})
    s.update(signal_time=signal_time)
    if stop_price is not None:
        s["stop_price"] = stop_price
    if atr is not None:
        s["atr"] = atr
    df = _bars(bars, cfg, end_time)
    return _simulate(df, _prepare(df, s, cfg), quantity, cfg)


def _metrics(trades: list[dict], equity: list[dict], rejected: list[dict], cfg: ExecutionConfig) -> dict:
    closed = [t for t in trades if t["status"] == "CLOSED"]
    pnls = np.array([t["net_pnl"] for t in closed], dtype=float)
    rs = np.array([t["net_r"] for t in closed], dtype=float)
    vals = np.array([cfg.initial_capital] + [x["equity"] for x in equity], dtype=float)
    drawdowns = vals / np.maximum.accumulate(vals) - 1
    profits = float(pnls[pnls > 0].sum())
    losses = float(-pnls[pnls < 0].sum())
    final = float(vals[-1])
    full_daily = [x for x in equity if x["session_complete"]]
    daily = pd.Series([x["equity"] for x in full_daily], dtype=float).pct_change().dropna()
    ratios_ok = len(daily) >= cfg.min_daily_returns_for_ratios
    excess = daily - ((1 + cfg.cash_annual_rate) ** (1 / 252) - 1)
    std = float(excess.std(ddof=1)) if len(excess) > 1 else 0
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2))) if len(excess) else 0
    notionals = sum(t["entry_price"] * t["quantity"] + sum(
        leg["price"] * leg["quantity"] for leg in t["legs"]) for t in trades)
    average_equity = float(np.mean(vals))
    # Entry-notional time weighting captures intraday round trips which disappear
    # in end-of-day snapshots. Calendar time includes overnight/weekend exposure.
    capital_minutes = 0.0
    intervals = []
    span_minutes = 0.0
    if trades and equity:
        span_start = _timestamp(equity[0]["evaluation_start"]) if equity[0].get("evaluation_start") else min(_timestamp(t["entry_time"]) for t in trades)
        span_end = _timestamp(equity[-1]["time"])
        span_minutes = (span_end - span_start).total_seconds() / 60
        for t in trades:
            begin = _timestamp(t["entry_time"])
            for leg in t["legs"]:
                end = _timestamp(leg["time"])
                capital_minutes += t["raw_entry_price"] * leg["quantity"] * (end - begin).total_seconds() / 60
            if t["remaining_quantity"]:
                capital_minutes += t["raw_entry_price"] * t["remaining_quantity"] * (span_end - begin).total_seconds() / 60
            intervals.append((begin, _timestamp(t["exit_time"]) if t["exit_time"] else span_end))
    occupied_minutes = 0.0
    if intervals:
        left, right = sorted(intervals)[0]
        for begin, end in sorted(intervals)[1:]:
            if begin <= right:
                right = max(right, end)
            else:
                occupied_minutes += (right - left).total_seconds() / 60
                left, right = begin, end
        occupied_minutes += (right - left).total_seconds() / 60
    return {"signals": len(trades) + len(rejected), "executed_trades": len(trades), "closed_trades": len(closed),
            "open_trades": len(trades) - len(closed), "rejected_signals": len(rejected),
            "starting_capital": cfg.initial_capital, "ending_equity": final, "net_pnl": final - cfg.initial_capital,
            "total_return": final / cfg.initial_capital - 1, "win_rate": float(np.mean(pnls > 0)) if len(pnls) else None,
            "average_trade_return": float(np.mean([t["return_pct"] for t in closed])) if closed else None,
            "median_trade_return": float(np.median([t["return_pct"] for t in closed])) if closed else None,
            "net_expectancy_r": float(rs.mean()) if len(rs) else None,
            "median_r": float(np.median(rs)) if len(rs) else None,
            "net_expectancy_dollars": float(pnls.mean()) if len(pnls) else None,
            "profit_factor": profits / losses if losses else None,
            "profit_factor_note": "undefined without realised losses" if not losses else None,
            "max_drawdown": float(-drawdowns.min()), "drawdown_frequency": "daily close / final partial session",
            "sharpe": float(np.sqrt(252) * excess.mean() / std) if ratios_ok and std > 0 else None,
            "sortino": float(np.sqrt(252) * excess.mean() / downside) if ratios_ok and downside > 0 else None,
            "annualized_return": ((final / cfg.initial_capital) ** (252 / len(full_daily)) - 1)
            if len(full_daily) >= 252 and final > 0 else None,
            "daily_observations": len(full_daily), "risk_ratio_note":
            None if ratios_ok else f"requires at least {cfg.min_daily_returns_for_ratios} complete daily returns",
            "average_exposure": float(np.mean([x["invested_value"] / x["equity"] for x in equity if x["equity"] > 0])) if equity else 0.0,
            "exposure_definition": "mean daily-close market value / equity; see time-weighted utilisation for intraday trades",
            "average_capital_utilisation": capital_minutes / (cfg.initial_capital * span_minutes) if span_minutes > 0 else 0.0,
            "capital_utilisation_definition": "calendar-time-weighted entry notional / initial capital",
            "time_in_market": occupied_minutes / span_minutes if span_minutes > 0 else 0.0,
            "turnover": notionals / (2 * average_equity) if average_equity > 0 else None,
            "turnover_definition": "(buys + sells) / (2 * mean equity), unannualized",
            "average_holding_sessions": float(np.mean([t["holding_sessions"] for t in closed])) if closed else None,
            "average_holding_minutes": float(np.mean([t["holding_minutes"] for t in closed])) if closed else None,
            "largest_loss": float(min(0.0, pnls.min())) if len(pnls) else None,
            "transaction_costs": float(sum(t["transaction_cost"] for t in trades)),
            "ambiguous_trades": sum(t["ambiguous"] for t in trades)}


def simulate_portfolio(signals, bars_by_symbol: Mapping[str, pd.DataFrame],
                       config: ExecutionConfig | None = None, *, end_time=None) -> dict:
    """Allocate frozen signals chronologically with identical cash/risk constraints.

    Entries sharing a timestamp are ordered by score descending, then stable ID.
    Exits release cash only when their open/complete-bar outcome was knowable.
    Stops limit planned risk, not unbounded overnight-gap losses. There is no
    leverage, no pyramiding and no inferred correlation/ETF look-through data.
    Inputs and the immutable config are never mutated.
    """
    cfg = config or ExecutionConfig()
    frames = {symbol: _bars(frame, cfg, end_time) for symbol, frame in bars_by_symbol.items()}
    source = signals.to_dict("records") if isinstance(signals, pd.DataFrame) else list(signals)
    rejected: list[dict] = []
    candidates: list[dict] = []
    seen = set()
    for s in source:
        symbol = str(s.get("symbol", ""))
        prepared = _prepare(frames.get(symbol, _bars(pd.DataFrame(), cfg)), s, cfg)
        if prepared["signal_id"] in seen:
            rejected.append({**prepared, "status": "REJECTED", "reason": "DUPLICATE_SIGNAL"})
            continue
        seen.add(prepared["signal_id"])
        if prepared["status"] == "REJECTED":
            rejected.append(prepared)
        else:
            candidates.append(prepared)
    candidates.sort(key=lambda x: (x["entry_time"], -(x["score"] if x["score"] is not None else -math.inf), x["signal_id"]))
    # Events contain no ordering of simultaneous intrabar highs and lows.
    events: list[tuple] = []
    serial = 0

    def event(ts, priority, kind, payload):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (_timestamp(ts), priority, serial, kind, payload))

    for p in candidates:
        event(p["entry_time"], 1, "entry", p)
    first = min((frame.index[0] for frame in frames.values() if not frame.empty), default=None)
    sessions: dict[Any, pd.Timestamp] = {}
    if first is not None:
        for frame in frames.values():
            for ts in frame.index[frame.index + pd.Timedelta(minutes=cfg.bar_minutes) >= first]:
                end = ts + pd.Timedelta(minutes=cfg.bar_minutes)
                date = ts.tz_convert(cfg.market_timezone).date()
                sessions[date] = max(sessions.get(date, end), end)
        for ts in sorted(sessions.values()):
            event(ts, 2, "mark", None)
    cash = cfg.initial_capital
    active: dict[str, dict] = {}
    trades, equity = [], []

    def mark_position(pos, ts):
        frame = frames[pos["symbol"]]
        open_idx = int(frame.index.searchsorted(ts, side="right")) - 1
        if open_idx >= 0 and frame.index[open_idx] + pd.Timedelta(minutes=cfg.bar_minutes) > ts:
            return float(frame.iloc[open_idx].open), frame.index[open_idx]
        idx = int((frame.index + pd.Timedelta(minutes=cfg.bar_minutes)).searchsorted(ts, side="right")) - 1
        if idx < 0 or frame.index[idx] < _timestamp(pos["trade"]["entry_time"]):
            return pos["trade"]["raw_entry_price"], None
        return float(frame.iloc[idx].close), frame.index[idx] + pd.Timedelta(minutes=cfg.bar_minutes)

    while events:
        ts, _, _, kind, payload = heapq.heappop(events)
        if kind == "exit":
            pos_id, leg = payload
            pos = active[pos_id]
            cash += leg["price"] * leg["quantity"] - leg["commission"]
            pos["remaining"] = max(0.0, pos["remaining"] - leg["quantity"])
            if pos["remaining"] <= 1e-10:
                del active[pos_id]
            continue
        if kind == "mark":
            invested = 0.0
            stale = []
            unresolved = []
            for pos in active.values():
                price, mark_time = mark_position(pos, ts)
                invested += price * pos["remaining"]
                if mark_time is None or (ts - mark_time) > pd.Timedelta(minutes=cfg.bar_minutes):
                    stale.append(pos["symbol"])
                unresolved_from = pos["trade"]["unresolved_from"]
                if unresolved_from is not None and ts >= _timestamp(unresolved_from):
                    unresolved.append(pos["symbol"])
            local = ts.tz_convert(cfg.market_timezone)
            equity.append({"time": _iso(ts), "equity": cash + invested, "cash": cash,
                           "invested_value": invested, "open_positions": len(active),
                           "session_complete": local.hour * 60 + local.minute >= 960 or any("session_end" in frame and bool(frame.iloc[-1].session_end) and frame.index[-1]+pd.Timedelta(minutes=cfg.bar_minutes)==ts for frame in frames.values() if not frame.empty),
                           "evaluation_start": _iso(first),
                           "stale_symbols": sorted(set(stale)), "unresolved_symbols": sorted(set(unresolved)),
                           "equity_reliable": not unresolved})
            continue
        p = payload
        reason = None
        if any(pos["trade"]["unresolved_from"] is not None and ts >= _timestamp(pos["trade"]["unresolved_from"]) for pos in active.values()):
            reason = "UNRESOLVED_OPEN_POSITION"
        elif any(pos["symbol"] == p["symbol"] for pos in active.values()):
            reason = "SYMBOL_ALREADY_OPEN"
        elif len(active) >= cfg.max_open_positions:
            reason = "POSITION_LIMIT"
        elif cfg.max_abs_correlation is not None and any(
            abs(p["correlations"].get(pos["symbol"], 0.0)) >= cfg.max_abs_correlation for pos in active.values()
        ):
            reason = "CORRELATED_EXPOSURE"
        portfolio_equity = cash + sum(mark_position(pos, ts)[0] * pos["remaining"] for pos in active.values())
        remaining_risk = sum(pos["trade"]["net_stop_risk"] * pos["remaining"] / pos["trade"]["quantity"] for pos in active.values())
        risk_budget = min(portfolio_equity * cfg.risk_fraction,
                          portfolio_equity * cfg.max_total_risk_fraction - remaining_risk)
        max_notional = min(cfg.max_position_dollars, cash - cfg.commission_per_order)
        if cfg.max_sector_fraction is not None:
            sector_value = sum(mark_position(pos, ts)[0] * pos["remaining"] for pos in active.values() if pos["sector"] == p["sector"])
            max_notional = min(max_notional, portfolio_equity * cfg.max_sector_fraction - sector_value)
        buy_unit_cost = p["entry_price"] * (1 + cfg.commission_bps / 10_000)
        qty = min(max_notional / buy_unit_cost, (risk_budget - 2 * cfg.commission_per_order) / p["net_stop_risk_per_share"])
        qty = max(0.0, qty if cfg.allow_fractional_shares else math.floor(qty))
        if not reason and qty <= 0:
            reason = "CAPITAL_OR_RISK_LIMIT"
        if reason:
            rejected.append({k: v for k, v in {**p, "status": "REJECTED", "reason": reason}.items() if k != "entry_index"})
            continue
        trade = _simulate(frames[p["symbol"]], p, qty, cfg)
        cash -= trade["entry_price"] * qty + trade["entry_commission"]
        trades.append(trade)
        active[p["signal_id"]] = {"symbol": p["symbol"], "sector": p["sector"], "remaining": qty, "trade": trade}
        for leg in trade["legs"]:
            event(leg["time"], 0, "exit", (p["signal_id"], leg))
    metrics = _metrics(trades, equity, rejected, cfg)
    reliable = not any(t["data_quality_flags"] for t in trades)
    metrics["equity_path_reliable"] = reliable
    metrics["unresolved_data_gap_trades"] = sum(bool(t["data_quality_flags"]) for t in trades)
    if not reliable:
        metrics["indicative_total_return"] = metrics["total_return"]
        metrics["indicative_net_pnl"] = metrics["net_pnl"]
        for key in ("total_return", "net_pnl", "max_drawdown", "sharpe", "sortino", "annualized_return"):
            metrics[key] = None
        metrics["risk_ratio_note"] = "unresolved missing-bar execution paths; indicative equity is not executable P/L"
    return {"schema_version": 1, "config": asdict(cfg), "trades": trades, "rejections": rejected,
            "equity": equity, "metrics": metrics,
            "assumptions": {"entry": "first regular-session bar open at/after signal availability and frozen_at",
                            "timestamps": "UTC; input bar-open timestamps, signal inputs completed before entry",
                            "same_bar_collision": "stop first; outcome recorded at bar end",
                            "targets": "market-trigger exits with adverse costs; no favourable gap improvement",
                            "trailing": "updated from completed bars; active from following open",
                            "missing_data": "open paths retained; stale position marks explicitly listed",
                            "contiguity": "stop at first unknown held-path bar" if cfg.require_contiguous_bars else
                            "disabled explicitly; sparse-path fills assume no unobserved barrier touches",
                            "calendar": "NYSE calendar verifies the next scheduled session opening across held-path gaps; holidays and early closes supported; calendar failures leave paths unresolved" if cfg.require_contiguous_bars else
                            "scheduled-session continuity disabled explicitly; sparse paths do not establish executable performance",
                            "correlation_controls": "supplied point-in-time candidate correlations only; missing mappings are not inferred",
                            "cash_benchmark": "nominal zero-interest cash" if cfg.cash_annual_rate == 0 else
                            "constant annual cash-rate assumption; not observed point-in-time risk-free returns",
                            "risk": "fixed-fraction planned stop loss including costs; gaps can exceed budget",
                            "scope": "research only; no broker or live-order integration"}}

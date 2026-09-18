from __future__ import annotations
"""Forward paper-trading harness for V1/V2/V3.

Default mode is fully local and costs nothing.  Alpaca is optional and can only
use the paper endpoint; this module deliberately contains no live-trading URL.
"""
import csv
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

SIGNALS = Path("data/day_opportunities_latest.csv")
STATE = Path("data/paper_state.json")
TRADES = Path("data/paper_trades.csv")
EQUITY = Path("data/paper_equity.csv")
SUMMARY = Path("data/paper_summary.json")

STARTING_CASH = float(os.getenv("PAPER_STARTING_CASH", "10000"))
MAX_POSITION_DOLLARS = float(os.getenv("PAPER_MAX_POSITION_DOLLARS", "1000"))
MAX_OPEN_POSITIONS = int(os.getenv("PAPER_MAX_OPEN_POSITIONS", "5"))
SLIPPAGE_BPS = float(os.getenv("PAPER_SLIPPAGE_BPS", "7"))
EXIT_AT_R = float(os.getenv("PAPER_EXIT_AT_R", "1"))
MODE = os.getenv("PAPER_EXECUTION_MODE", "local").lower()
ALPACA_STRATEGY = os.getenv("ALPACA_PAPER_STRATEGY", "V3").upper()
ALPACA_BASE = "https://paper-api.alpaca.markets"


def now():
    return datetime.now(timezone.utc).isoformat()


def f(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def empty_state():
    return {
        "schema_version": 1,
        "created_at": now(),
        "starting_cash": STARTING_CASH,
        "benchmarks": {},
        "portfolios": {
            v: {"cash": STARTING_CASH, "positions": {}, "realized_pnl": 0.0}
            for v in ("V1", "V2", "V3")
        },
    }


def load_state():
    if not STATE.exists():
        return empty_state()
    try:
        s = json.loads(STATE.read_text())
        for v in ("V1", "V2", "V3"):
            s.setdefault("portfolios", {}).setdefault(
                v, {"cash": STARTING_CASH, "positions": {}, "realized_pnl": 0.0}
            )
        return s
    except Exception:
        return empty_state()


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2, allow_nan=False))


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def slippage(price, side):
    bps = SLIPPAGE_BPS / 10000.0
    return price * (1 + bps if side == "buy" else 1 - bps)


def current_prices(df):
    return {
        str(r.symbol): f(r.price)
        for r in df.itertuples()
        if np.isfinite(f(getattr(r, "price", np.nan)))
    }


def benchmark_prices():
    out = {}
    for symbol in ("SPY", "QQQ"):
        try:
            h = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=True)
            if not h.empty:
                out[symbol] = f(h.Close.iloc[-1])
        except Exception:
            pass
    return out


def strategy_fields(strategy):
    if strategy == "V1":
        return "state", "score", "reason", "stop", "target_1r", "target_2r"
    p = strategy.lower()
    return f"{p}_state", f"{p}_score", f"{p}_reason", f"{p}_stop", f"{p}_target_1r", f"{p}_target_2r"


def alpaca_headers():
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def submit_alpaca_bracket(symbol, qty, stop, target, client_order_id):
    """Submit to Alpaca PAPER only. Never falls back to a live endpoint."""
    headers = alpaca_headers()
    if MODE != "alpaca" or not headers:
        return {"status": "disabled"}
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
        "order_class": "bracket",
        "take_profit": {"limit_price": f"{target:.2f}"},
        "stop_loss": {"stop_price": f"{stop:.2f}"},
        "client_order_id": client_order_id[:48],
    }
    r = requests.post(f"{ALPACA_BASE}/v2/orders", json=payload, headers=headers, timeout=20)
    if r.status_code >= 300:
        return {"status": "error", "code": r.status_code, "message": r.text[:300]}
    j = r.json()
    return {"status": j.get("status", "submitted"), "id": j.get("id")}


def close_positions(state, prices):
    for strategy, pf in state["portfolios"].items():
        for symbol, p in list(pf["positions"].items()):
            price = prices.get(symbol)
            if not np.isfinite(f(price)):
                continue
            stop, target = f(p["stop"]), f(p["target"])
            reason = None
            if price <= stop:
                reason = "STOP"
            elif price >= target:
                reason = f"TARGET_{EXIT_AT_R:g}R"
            if not reason:
                p["last_price"] = price
                continue
            exit_price = slippage(price, "sell")
            qty = int(p["qty"])
            proceeds = qty * exit_price
            pnl = proceeds - qty * float(p["entry_price"])
            pf["cash"] += proceeds
            pf["realized_pnl"] = float(pf.get("realized_pnl", 0)) + pnl
            append_csv(TRADES, {
                "trade_id": p["trade_id"], "strategy": strategy, "symbol": symbol,
                "entry_time": p["entry_time"], "exit_time": now(),
                "entry_price": round(float(p["entry_price"]), 4),
                "exit_price": round(exit_price, 4), "qty": qty,
                "pnl": round(pnl, 2), "pnl_pct": round(exit_price / float(p["entry_price"]) - 1, 6),
                "exit_reason": reason, "signal_score": p.get("score"),
                "market_regime": p.get("market_regime", ""),
            })
            del pf["positions"][symbol]


def open_positions(state, df, prices):
    for strategy in ("V1", "V2", "V3"):
        pf = state["portfolios"][strategy]
        if len(pf["positions"]) >= MAX_OPEN_POSITIONS:
            continue
        state_col, score_col, reason_col, stop_col, t1_col, t2_col = strategy_fields(strategy)
        if state_col not in df.columns:
            continue
        candidates = df[df[state_col] == "ENTRY TRIGGERED"].copy()
        if score_col in candidates:
            candidates = candidates.sort_values(score_col, ascending=False)
        for _, r in candidates.iterrows():
            if len(pf["positions"]) >= MAX_OPEN_POSITIONS:
                break
            symbol = str(r["symbol"])
            if symbol in pf["positions"]:
                continue
            price = prices.get(symbol)
            if not np.isfinite(f(price)) or price <= 0:
                continue
            entry = slippage(price, "buy")
            qty = int(min(MAX_POSITION_DOLLARS, pf["cash"]) // entry)
            if qty < 1:
                continue
            stop = f(r.get(stop_col))
            t1 = f(r.get(t1_col))
            t2 = f(r.get(t2_col))
            if not np.isfinite(stop) or stop >= entry:
                continue
            raw_target = t1 if EXIT_AT_R <= 1 else t2
            if not np.isfinite(raw_target) or raw_target <= entry:
                raw_target = entry + EXIT_AT_R * (entry - stop)
            target = float(raw_target)
            trade_id = f"{strategy}-{symbol}-{uuid.uuid4().hex[:12]}"
            cost = qty * entry
            pf["cash"] -= cost
            broker = {"status": "local"}
            if strategy == ALPACA_STRATEGY:
                broker = submit_alpaca_bracket(symbol, qty, stop, target, trade_id)
            pf["positions"][symbol] = {
                "trade_id": trade_id, "symbol": symbol, "qty": qty,
                "entry_time": now(), "entry_price": entry, "last_price": price,
                "stop": stop, "target": target, "score": f(r.get(score_col)),
                "reason": str(r.get(reason_col, "")),
                "market_regime": str(r.get("market_regime", "UNKNOWN")),
                "broker_status": broker.get("status"), "broker_order_id": broker.get("id"),
            }


def portfolio_value(pf, prices):
    mv = 0.0
    for symbol, p in pf["positions"].items():
        px = prices.get(symbol, f(p.get("last_price"), f(p["entry_price"])))
        mv += int(p["qty"]) * float(px)
    return float(pf["cash"]) + mv


def main():
    if not SIGNALS.exists():
        print("No day-opportunity signals yet.")
        return
    df = pd.read_csv(SIGNALS)
    if df.empty:
        return
    state = load_state()
    prices = current_prices(df)
    close_positions(state, prices)
    open_positions(state, df, prices)

    bench = benchmark_prices()
    for symbol, px in bench.items():
        if symbol not in state["benchmarks"] and np.isfinite(px):
            state["benchmarks"][symbol] = {"start_price": px, "start_time": now()}
    snapshot = {"timestamp": now()}
    summaries = {}
    for strategy, pf in state["portfolios"].items():
        equity = portfolio_value(pf, prices)
        snapshot[f"{strategy}_equity"] = round(equity, 2)
        summaries[strategy] = {
            "equity": round(equity, 2),
            "return": equity / float(state.get("starting_cash", STARTING_CASH)) - 1,
            "cash": round(float(pf["cash"]), 2),
            "open_positions": len(pf["positions"]),
            "realized_pnl": round(float(pf.get("realized_pnl", 0)), 2),
            "positions": list(pf["positions"].values()),
        }
    for symbol, px in bench.items():
        base = f(state["benchmarks"].get(symbol, {}).get("start_price"))
        snapshot[f"{symbol}_return"] = px / base - 1 if np.isfinite(base) and base > 0 else np.nan

    append_csv(EQUITY, snapshot)
    hist = pd.read_csv(EQUITY) if EQUITY.exists() else pd.DataFrame()
    for strategy in ("V1", "V2", "V3"):
        col = f"{strategy}_equity"
        if col in hist and hist[col].notna().any():
            curve = hist[col].astype(float)
            dd = curve / curve.cummax() - 1
            summaries[strategy]["max_drawdown"] = f(dd.min(), 0.0)
    closed = pd.read_csv(TRADES) if TRADES.exists() and TRADES.stat().st_size else pd.DataFrame()
    for strategy in ("V1", "V2", "V3"):
        g = closed[closed.strategy == strategy] if not closed.empty and "strategy" in closed else pd.DataFrame()
        if not g.empty:
            wins = g[g.pnl > 0]
            losses = g[g.pnl < 0]
            summaries[strategy]["closed_trades"] = len(g)
            summaries[strategy]["win_rate"] = len(wins) / len(g)
            summaries[strategy]["profit_factor"] = (
                float(wins.pnl.sum()) / abs(float(losses.pnl.sum())) if not losses.empty else None
            )
        else:
            summaries[strategy].update({"closed_trades": 0, "win_rate": None, "profit_factor": None})

    benchmark_summary = {}
    for symbol, px in bench.items():
        base = f(state["benchmarks"].get(symbol, {}).get("start_price"))
        benchmark_summary[symbol] = {
            "start_price": base if np.isfinite(base) else None,
            "price": px,
            "return": px / base - 1 if np.isfinite(base) and base > 0 else None,
        }

    SUMMARY.write_text(json.dumps({
        "generated_at": now(),
        "mode": MODE,
        "alpaca_strategy": ALPACA_STRATEGY,
        "starting_cash": float(state.get("starting_cash", STARTING_CASH)),
        "slippage_bps": SLIPPAGE_BPS,
        "max_position_dollars": MAX_POSITION_DOLLARS,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "exit_at_r": EXIT_AT_R,
        "benchmarks": benchmark_summary,
        "strategies": summaries,
        "note": "Local shadow portfolios are the source of truth. Alpaca, when enabled, is paper-only and mirrors the selected strategy.",
    }, indent=2, allow_nan=False))
    save_state(state)
    print(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()

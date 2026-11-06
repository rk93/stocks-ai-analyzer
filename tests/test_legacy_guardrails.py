"""Future-facing fixes preserve baseline rules and append-only historical data.

All data/quotes are controlled fixtures; no real feeds, broker calls or tracked
research records are read or written. Baseline fingerprints are from 5ef7ae6.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import day_opportunities as scanner
import paper_trader as paper
import prediction_tracker as tracker


ROOT = Path(__file__).resolve().parents[1]
AT = "2026-11-05T17:02:00+00:00"


class ClockPandas:
    """Override this module's clock without mutating pandas for other callers."""
    def __init__(self, at):
        stamp = pd.Timestamp(at)
        self.Timestamp = SimpleNamespace(now=lambda tz=None: stamp.tz_convert(tz) if tz else stamp.tz_localize(None))

    def __getattr__(self, key):
        return getattr(pd, key)


def bars(index, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close - 0.1, "High": close + 0.5, "Low": close - 0.5,
                         "Close": close, "Volume": np.full(len(index), 1000.)}, index=index)


def mock_feed(monkeypatch, frame, at=AT):
    monkeypatch.setattr(scanner, "pd", ClockPandas(at))
    monkeypatch.setattr(scanner.yf, "Ticker", lambda symbol: SimpleNamespace(history=lambda **kwargs: frame.copy()))


def test_live_daily_and_market_features_exclude_current_and_future_sessions(monkeypatch):
    index = pd.bdate_range(end="2026-11-06", periods=230, tz="America/New_York")
    frame = bars(index, np.linspace(80, 120, len(index)))
    frame.loc[index[-2]:, ["Open", "High", "Low", "Close"]] = 99999
    mock_feed(monkeypatch, frame)
    prior = frame[frame.index.date < pd.Timestamp(AT).tz_convert("America/New_York").date()]
    daily, market = scanner._daily_metrics("A"), scanner._market_metrics()
    assert daily["prev_close"] == prior.Close.iloc[-1]
    assert daily["prior_high"] == prior.High.iloc[-1]
    assert daily["prior_low"] == prior.Low.iloc[-1]
    assert daily["ma20"] == pytest.approx(prior.Close.tail(20).mean())
    assert daily["return_20d"] == pytest.approx(prior.Close.iloc[-1] / prior.Close.iloc[-21] - 1)
    assert market["price"] == prior.Close.iloc[-1]
    assert market["ma200"] == pytest.approx(prior.Close.tail(200).mean())
    assert market["return_20d"] == daily["return_20d"]


def test_live_intraday_uses_completed_bars_and_thirty_minute_opening_range(monkeypatch):
    prior_index = pd.date_range("2026-11-04 09:30", "2026-11-04 15:55", freq="5min", tz="America/New_York")
    today_index = pd.date_range("2026-11-05 09:30", "2026-11-05 12:05", freq="5min", tz="America/New_York")
    frame = bars(prior_index.append(today_index), np.full(len(prior_index) + len(today_index), 100))
    # At 12:02 NY the 12:00 and 12:05 opening-labelled bars are unavailable.
    frame.loc[today_index[-2]:, "Close"] = 99999
    # The 10:00 opening-labelled bar is outside the 09:30–10:00 opening range.
    frame.loc[pd.Timestamp("2026-11-05 10:00", tz="America/New_York"), "High"] = 5000
    mock_feed(monkeypatch, frame)
    result = scanner._intraday_metrics("A", {"prev_close": 99, "ma20": 98, "ma50": 97, "prior_low": 96})
    assert result["price"] == 100
    assert pd.Timestamp(result["quote_available_at"]) == pd.Timestamp("2026-11-05T17:00:00Z")
    assert result["data_cutoff"] == result["quote_available_at"]
    assert result["opening_range_high"] == 100.5
    assert result["feature_method"] == "completed_5m_prior_daily_v1"


def test_regular_session_excludes_1600_opening_label():
    index = pd.date_range("2026-11-05 15:55", periods=3, freq="5min", tz="America/New_York")
    selected = scanner._regular_session(bars(index, [1, 2, 3]))
    assert list(selected.index) == [index[0]]


def test_prior_session_quotes_are_not_current_intraday_research(monkeypatch):
    index = pd.date_range("2026-11-04 09:30", periods=10, freq="5min", tz="America/New_York")
    mock_feed(monkeypatch, bars(index, np.full(10, 100)))
    assert scanner._intraday_metrics("A", {"prev_close": 99}) is None


def signal_row(**updates):
    result = {"symbol": "A", "price": 100, "generated_at": AT, "quote_available_at": "2026-11-05T17:00:00+00:00", "data_cutoff": "2026-11-05T17:00:00+00:00"}
    for prefix in ("", "v2_", "v3_"):
        result.update({prefix + "state": "ENTRY TRIGGERED", prefix + "score": 75,
                       prefix + "reason": "frozen research", prefix + "stop": 98,
                       prefix + "target_1r": 101, prefix + "target_2r": 102})
    result.update(updates)
    return result


@pytest.mark.parametrize("at,quote", [
    (AT, None), (AT, "bad"), (AT, "2026-11-05T17:00:00"),
    (AT, "2026-11-05T16:45:00+00:00"), (AT, "2026-11-05T17:05:00+00:00"),
    (AT, "2026-11-04T17:00:00+00:00"),
    ("2026-11-05T22:00:00+00:00", "2026-11-05T21:59:00+00:00"),
    ("2026-11-07T17:02:00+00:00", "2026-11-07T17:00:00+00:00"),
])
def test_offhours_stale_missing_future_and_naive_quotes_cannot_trade(monkeypatch, at, quote):
    monkeypatch.setattr(paper, "now", lambda: at)
    row = signal_row(quote_available_at=quote)
    df = pd.DataFrame([row])
    state = paper.empty_state()
    before = deepcopy(state)
    assert not paper.fresh_quote(row, at)
    assert paper.current_prices(df) == {}
    # Even a caller-provided price dict cannot bypass the row's quote provenance.
    paper.open_positions(state, df, {"A": 100})
    assert state == before


def test_actual_slipped_entry_has_exact_configured_1point5r_target(monkeypatch):
    monkeypatch.setattr(paper, "now", lambda: AT)
    monkeypatch.setattr(paper, "EXIT_AT_R", 1.5)
    monkeypatch.setattr(paper, "SLIPPAGE_BPS", 7)
    monkeypatch.setattr(paper, "submit_alpaca_bracket", lambda *args: {"status": "disabled"})
    state = paper.empty_state()
    df = pd.DataFrame([signal_row()])
    paper.open_positions(state, df, paper.current_prices(df))
    for strategy in ("V1", "V2", "V3"):
        position = state["portfolios"][strategy]["positions"]["A"]
        entry = position["entry_price"]
        assert entry == pytest.approx(100.07)
        assert position["target"] - entry == pytest.approx(1.5 * (entry - position["stop"]))
        assert position["execution_method"] == "guarded_sampled_quote_v2"
        assert state["portfolios"][strategy]["cash"] >= paper.STARTING_CASH - paper.MAX_POSITION_DOLLARS
    # A repeated signal cannot create a second position or debit capital twice.
    before = deepcopy(state)
    paper.open_positions(state, df, paper.current_prices(df))
    assert state == before


@pytest.mark.parametrize("contents", ["{broken", "{}", '{"schema_version":1,"portfolios":{}}', '{"schema_version":2,"portfolios":{"V1":{"cash":0,"positions":{}}}}'])
def test_corrupt_or_incompatible_paper_state_cannot_reset_history(monkeypatch, tmp_path, contents):
    path = tmp_path / "state.json"
    path.write_text(contents)
    monkeypatch.setattr(paper, "STATE", path)
    with pytest.raises((ValueError, TypeError)):
        paper.load_state()
    assert path.read_text() == contents


def test_missing_state_initializes_equal_capital_but_existing_state_is_preserved(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    monkeypatch.setattr(paper, "STATE", path)
    state = paper.load_state()
    assert {portfolio["cash"] for portfolio in state["portfolios"].values()} == {paper.STARTING_CASH}
    state["portfolios"]["V1"]["cash"] = 9123.45
    state["seen_signals"] = ["V1|A|2026-11-04"]
    state["preserved_extension"] = {"original": "audit"}
    paper.save_state(state)
    assert paper.load_state() == state


def test_trade_append_preserves_previous_bytes_and_one_header(tmp_path):
    path = tmp_path / "trades.csv"
    paper.append_csv(path, {"id": "old", "pnl": -5})
    previous = path.read_bytes()
    paper.append_csv(path, {"id": "new", "pnl": 2})
    assert path.read_bytes().startswith(previous)
    assert len(pd.read_csv(path)) == 2
    assert path.read_text().count("id,pnl") == 1


def test_tracker_uses_strategy_frozen_fields_without_mutating_prediction_ledger(monkeypatch, tmp_path):
    ledger, output = tmp_path / "predictions.csv", tmp_path / "tracker.csv"
    row = {"prediction_id": "frozen-v3", "signal_date": "2026-11-04", "generated_at": "2026-11-04T17:00:00Z", "symbol": "A", "price": 100,
           "strategy_version": "V3", "state": "WATCH", "score": 12, "reason": "V1 context", "stop": 90, "target_1r": 110, "target_2r": 120,
           "prediction_state": "ENTRY TRIGGERED", "prediction_score": 80, "prediction_reason": "original V3 rationale", "prediction_stop": 98,
           "prediction_target_1r": 102, "prediction_target_2r": 104, "data_cutoff": "2026-11-04T16:55:00Z"}
    pd.DataFrame([row]).to_csv(ledger, index=False)
    original = ledger.read_bytes()
    monkeypatch.setattr(tracker, "LEDGER", ledger)
    monkeypatch.setattr(tracker, "OUT", output)
    future = bars(pd.date_range("2026-11-05", periods=2, freq="B", tz="America/New_York"), [101, 102])
    monkeypatch.setattr(tracker.yf, "Ticker", lambda symbol: SimpleNamespace(history=lambda **kwargs: future))
    tracker.main()
    result = pd.read_csv(output).iloc[0]
    assert result["prediction"] == "ENTRY TRIGGERED"
    assert result["score"] == 80
    assert result["original_reason"] == "original V3 rationale"
    assert result["stop"] == 98 and result["target_1r"] == 102 and result["target_2r"] == 104
    assert result["strategy_version"] == "V3"
    assert result["data_availability_cutoff"] == row["data_cutoff"]
    assert ledger.read_bytes() == original


def test_scanner_preserves_first_frozen_signal_on_repeated_same_day_scans(monkeypatch, tmp_path):
    config, output, history, ledger = (tmp_path / name for name in ("config.json", "latest.csv", "history.csv", "predictions.csv"))
    config.write_text(json.dumps({"day_universe": ["A"]}))
    for name, path in (("CONFIG", config), ("OUT", output), ("HIST", history), ("PREDICTIONS", ledger)):
        monkeypatch.setattr(scanner, name, path)
    monkeypatch.setattr(scanner, "datetime", SimpleNamespace(now=lambda tz=None: pd.Timestamp(AT).to_pydatetime()))
    monkeypatch.setattr(scanner, "_daily_metrics", lambda symbol: {"prev_close": 99, "return_20d": 0.07, "ma20": 98, "ma50": 95, "atr_pct": 0.03})
    monkeypatch.setattr(scanner, "_market_metrics", lambda: {"price": 110, "ma20": 108, "ma50": 105, "ma200": 100, "atr_pct": 0.015, "return_20d": 0.04})
    intra = {"price": 100, "vwap_dist": 0.005, "volume_ratio": 2.1, "volume_ratio_tod": 1.8, "momentum_15m": 0.007,
             "breakout_or": True, "crossed_vwap": True, "near_support": True, "minutes_from_open": 150,
             "day_change": 0.02, "gap": 0.01, "opening_range_low": 97, "data_cutoff": "2026-11-05T17:00:00Z"}
    monkeypatch.setattr(scanner, "_intraday_metrics", lambda symbol, daily: deepcopy(intra))
    prediction_id = hashlib.sha256(b"2026-11-05|A|V1|ENTRY TRIGGERED").hexdigest()[:20]
    frozen = {"prediction_id": prediction_id, "signal_date": "2026-11-05", "symbol": "A", "strategy_version": "V1", "prediction_reason": "earlier frozen rationale",
              "prediction_score": 76, "prediction_stop": 95, "price": 97, "frozen_at": "2026-11-05T16:30:00Z", "data_cutoff": "2026-11-05T16:25:00Z"}
    pd.DataFrame([frozen]).to_csv(ledger, index=False)
    scanner.main()
    first = pd.read_csv(ledger).set_index("prediction_id")
    assert len(first) == 3
    for key, value in frozen.items():
        if key != "prediction_id":
            assert first.loc[prediction_id, key] == value
    intra["price"] = 100.5
    scanner.main()
    second = pd.read_csv(ledger).set_index("prediction_id")
    # The legacy CSV writer may widen numeric dtypes when concatenating the new
    # scan. Every previously frozen value and its identity must remain equal.
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


@pytest.mark.parametrize("filename,name,expected", [
    ("day_opportunities.py", "score_row", "6ec9cd19d605e4d6f83c2a648c60e0abac85d9dbc4dcf74bfa3c5e21c8b1a1df"),
    ("strategy_v2.py", "_finite", "4626d688c6f356482d1923490beb82114958b91ee1af7709731a755ab7a895cd"),
    ("strategy_v2.py", "score_v2", "3cc94548284eb662602d4f003ea0a5b8b960dfb8f1378f1c34a42e94652c481d"),
    ("strategy_v3.py", "_finite", "4626d688c6f356482d1923490beb82114958b91ee1af7709731a755ab7a895cd"),
    ("strategy_v3.py", "classify_regime", "78570076dc640e5c403127af280e340976e23fa581f2d5a59f2e12a55cab7b1d"),
    ("strategy_v3.py", "score_v3", "9cc210f76e1b8a8f9fa1e16ba1df7e47e7957058a83b3852fb831a321dd80eeb"),
])
def test_historical_baseline_scoring_ast_is_unchanged(filename, name, expected):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == expected


def test_historical_backtest_engine_is_unchanged():
    assert hashlib.sha256((ROOT / "backtest_engine.py").read_bytes()).hexdigest() == "d306e38ab5a8469110dcb01bc2fbd6979208a9c968a3989264c9589a7672b4c2"

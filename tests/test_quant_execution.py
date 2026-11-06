"""Execution invariants: timing, unfavourable paths, cash and risk accounting."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

import market_sessions
from quant_execution import ExecutionConfig, simulate_portfolio, simulate_trade


def frame(rows, times=None):
    if times is None:
        times = pd.date_range("2026-10-05 09:30", periods=len(rows), freq="5min", tz="America/New_York")
    else:
        times = pd.DatetimeIndex(times)
        if times.tz is None:
            times = times.tz_localize("America/New_York")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=times)


def config(**kwargs):
    # Simplified sparse fixtures are not claimed as real replay evidence.
    defaults = dict(slippage_bps=0, spread_bps=0, max_holding_sessions=None, max_sector_fraction=None,
                    require_contiguous_bars=False)
    defaults.update(kwargs)
    return ExecutionConfig(**defaults)


def signal(symbol, when="2026-10-05 09:30-04:00", **kwargs):
    return {"symbol": symbol, "signal_time": when, "stop_price": 98, **kwargs}


def test_completed_candle_cutoff_enters_next_open_and_ignores_signal_close():
    bars = frame([(90, 110, 80, 109), (100, 101, 99, 100), (100, 103, 99, 102)])
    trade = simulate_trade(bars, bars.index[1], 98, config=config())
    assert trade["raw_entry_price"] == 100
    assert pd.Timestamp(trade["entry_time"]) == bars.index[1]
    assert trade["exit_reason"] == "TARGET"
    assert trade["net_r"] == 1


def test_prospective_frozen_timestamp_prevents_retroactive_open_fill():
    bars = frame([(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 101, 102)])
    trade = simulate_trade(bars, bars.index[0], 98, config=config(),
                           signal={"frozen_at": bars.index[0] + pd.Timedelta(minutes=1)})
    assert pd.Timestamp(trade["entry_time"]) == bars.index[1]
    assert trade["raw_entry_price"] == 101


def test_missing_next_bar_and_stale_next_session_are_rejected():
    bars = frame([(100, 101, 99, 100)])
    assert simulate_trade(bars, bars.index[-1] + pd.Timedelta(minutes=5), 98)["reason"] == "NO_EXECUTABLE_BAR"
    future = frame([(100, 101, 99, 100)], ["2026-10-06 09:30"])
    assert simulate_trade(future, bars.index[0], 98)["reason"] == "STALE_ENTRY"


def test_unfinished_final_bar_is_excluded_at_evaluation_boundary():
    bars = frame([(100, 101, 99, 100), (100, 103, 99, 102)])
    trade = simulate_trade(bars, bars.index[0], 98, config=config(), end_time=bars.index[1])
    assert trade["status"] == "OPEN"
    assert trade["net_r"] is None
    assert pd.Timestamp(trade["mark_time"]) == bars.index[1]


@pytest.mark.parametrize("r", [1, 1.5, 2])
def test_reward_models_have_exact_independent_targets(r):
    bars = frame([(100, 105, 99, 100)])
    trade = simulate_trade(bars, bars.index[0], 98, config=config(target_r=r))
    assert trade["target_price"] == 100 + 2 * r
    assert trade["net_r"] == r


def test_same_bar_collision_is_pessimistic_and_identified():
    bars = frame([(100, 103, 97, 100)])
    trade = simulate_trade(bars, bars.index[0], 98, config=config())
    assert trade["exit_reason"] == "STOP_AMBIGUOUS"
    assert trade["ambiguous"] is True
    assert trade["net_r"] == -1
    assert pd.Timestamp(trade["exit_time"]) == bars.index[0] + pd.Timedelta(minutes=5)


def test_opening_gap_through_stop_cannot_fill_at_the_old_stop():
    bars = frame([(100, 101, 99, 100), (94, 96, 93, 95)])
    cfg = config(slippage_bps=7, spread_bps=2)
    trade = simulate_trade(bars, bars.index[0], 98, config=cfg)
    assert trade["exit_reason"] == "STOP_GAP"
    assert trade["exit_price"] == pytest.approx(94 * (1 - .0008))
    assert trade["net_r"] < -2
    assert trade["net_pnl"] < -trade["net_stop_risk"]


def test_known_opening_target_precedes_later_stop_without_favourable_gap_credit():
    bars = frame([(100, 101, 99, 100), (104, 105, 95, 99)])
    trade = simulate_trade(bars, bars.index[0], 98, config=config())
    assert trade["exit_reason"] == "TARGET"
    assert trade["exit_price"] == 102
    assert trade["ambiguous"] is False
    assert pd.Timestamp(trade["exit_time"]) == bars.index[1]


def test_both_sides_charge_spread_slippage_and_commission():
    bars = frame([(100, 105, 99, 104)])
    cfg = config(slippage_bps=7, spread_bps=2, commission_per_order=.50, commission_bps=1)
    trade = simulate_trade(bars, bars.index[0], 98, config=cfg, quantity=10)
    buy = 100 * 1.0008
    target = buy + buy - 98
    sell = target * .9992
    fees = (buy + sell) * 10 * .0001 + 1
    assert trade["entry_price"] == pytest.approx(buy)
    assert trade["exit_price"] == pytest.approx(sell)
    assert trade["commissions"] == pytest.approx(fees)
    assert trade["net_pnl"] == pytest.approx((sell - buy) * 10 - fees)
    assert trade["gross_pnl"] - trade["transaction_cost"] == pytest.approx(trade["net_pnl"])


def test_atr_and_fixed_stops_use_actual_entry_open_and_frozen_atr():
    bars = frame([(100, 101, 99, 100)])
    fixed = simulate_trade(bars, bars.index[0], 10, config=config(stop_model="fixed", fixed_stop_pct=.03))
    atr = simulate_trade(bars, bars.index[0], 10, atr=2, config=config(stop_model="atr", atr_multiple=2))
    assert fixed["stop_price"] == 97
    assert atr["stop_price"] == 96
    assert simulate_trade(bars, bars.index[0], config=config(stop_model="atr"))["reason"] == "INVALID_STOP_AT_FILL"


def test_open_below_signal_stop_is_rejected_instead_of_buying_already_failed_setup():
    bars = frame([(97, 98, 96, 97)])
    assert simulate_trade(bars, bars.index[0], 98, config=config())["reason"] == "INVALID_STOP_AT_FILL"


def test_trailing_stop_is_activated_only_after_high_bar_completes():
    bars = frame([(100, 110, 99, 105), (105, 106, 104, 105)])
    trade = simulate_trade(bars, bars.index[0], 95, config=config(target_r=None, trailing_pct=.05))
    assert pd.Timestamp(trade["exit_time"]) == bars.index[1] + pd.Timedelta(minutes=5)
    assert trade["exit_price"] == pytest.approx(104.5)


def test_atr_trailing_requires_frozen_volatility():
    bars = frame([(100, 101, 99, 100)])
    assert simulate_trade(bars, bars.index[0], 98, config=config(trailing_atr_multiple=2))["reason"] == "MISSING_FROZEN_ATR"


def test_close_known_trend_exit_fills_following_open():
    bars = frame([(100, 101, 95, 96), (94, 97, 93, 96)])
    bars["trend_exit"] = [True, False]
    trade = simulate_trade(bars, bars.index[0], 90, config=config(target_r=None, trend_exit_column="trend_exit"))
    assert trade["exit_reason"] == "TREND"
    assert trade["exit_price"] == 94
    assert pd.Timestamp(trade["exit_time"]) == bars.index[1]


def test_explicit_bar_time_exit_charges_costs_on_close():
    bars = frame([(100, 101, 99, 100), (100, 101, 99, 100.5)])
    trade = simulate_trade(bars, bars.index[0], 98, config=config(target_r=None, max_holding_bars=2))
    assert trade["exit_reason"] == "TIME"
    assert trade["exit_price"] == 100.5
    assert trade["holding_minutes"] == 10


def test_truncated_fifth_session_is_open_until_a_known_session_close():
    times = pd.date_range("2026-10-05 09:30", periods=5, freq="B", tz="America/New_York")
    bars = frame([(100, 101, 99, 100)] * 5, times)
    cfg = config(target_r=None, max_holding_sessions=5)
    trade = simulate_trade(bars, bars.index[0], 98, config=cfg)
    assert trade["status"] == "OPEN"
    assert trade["holding_sessions"] == 5
    bars["session_end"] = [False] * 4 + [True]
    assert simulate_trade(bars, bars.index[0], 98, config=cfg)["exit_reason"] == "TIME"


def test_session_close_exit_and_null_session_flag():
    bars = frame([(100, 101, 99, 100), (100, 101, 99, 100)], ["2026-10-05 09:30", "2026-10-05 15:55"])
    bars["session_end"] = [np.nan, np.nan]
    trade = simulate_trade(bars, bars.index[0], 98, config=config(target_r=None, max_holding_sessions=1))
    assert trade["exit_reason"] == "TIME"
    assert pd.Timestamp(trade["exit_time"]).tz_convert("America/New_York").hour == 16


def test_partial_exits_account_for_each_quantity_and_order():
    bars = frame([(100, 103, 99, 102), (102, 105, 101, 104)])
    trade = simulate_trade(bars, bars.index[0], 98, quantity=10,
                           config=config(target_r=2, partial_fraction=.5, commission_per_order=.5))
    assert [x["quantity"] for x in trade["legs"]] == [5, 5]
    assert [x["price"] for x in trade["legs"]] == [102, 104]
    assert trade["net_pnl"] == 28.5
    assert trade["commissions"] == 1.5


def test_partial_gap_target_before_intrabar_stop_releases_only_partial_cash_at_open():
    bars = frame([(100, 101, 99, 100), (103, 103, 97, 100)])
    trade = simulate_trade(bars, bars.index[0], 98, quantity=10, config=config(target_r=2, partial_fraction=.5))
    assert [x["reason"] for x in trade["legs"]] == ["PARTIAL_TARGET", "STOP"]
    assert [x["phase"] for x in trade["legs"]] == ["open", "intrabar"]
    assert trade["net_pnl"] == 0


def test_partial_collision_does_not_credit_unproven_profit():
    bars = frame([(100, 103, 97, 100)])
    trade = simulate_trade(bars, bars.index[0], 98, quantity=10, config=config(target_r=2, partial_fraction=.5))
    assert len(trade["legs"]) == 1
    assert trade["net_pnl"] == -20


def test_fractional_quantity_requires_explicit_configuration():
    bars = frame([(100, 103, 99, 102)])
    with pytest.raises(ValueError, match="integral"):
        simulate_trade(bars, bars.index[0], 98, quantity=.5)
    trade = simulate_trade(bars, bars.index[0], 98, quantity=.5, config=config(allow_fractional_shares=True))
    assert trade["net_pnl"] == 1


def test_risk_sizing_and_notional_cap_and_cash_conservation():
    bars = frame([(100, 111, 99, 110)])
    result = simulate_portfolio([signal("A", stop_price=90)], {"A": bars}, config())
    trade = result["trades"][0]
    assert trade["quantity"] == 5  # 0.5% of 10,000 / $10 stop risk
    assert trade["net_stop_risk"] == 50
    assert result["metrics"]["ending_equity"] == 10_050
    assert result["equity"][-1]["cash"] == 10_000 + trade["net_pnl"]
    capped = simulate_portfolio([signal("A", stop_price=99)], {"A": bars}, config(risk_fraction=.1))
    assert capped["trades"][0]["quantity"] == 10


def test_commissions_are_included_in_cash_and_stop_risk_budgets():
    bars = frame([(100, 111, 99, 110)])
    result = simulate_portfolio([signal("A", stop_price=90)], {"A": bars},
                                config(commission_per_order=1, commission_bps=5))
    trade = result["trades"][0]
    assert trade["quantity"] == 4
    assert trade["net_stop_risk"] <= 50
    assert result["equity"][-1]["cash"] == pytest.approx(10_000 + trade["net_pnl"])


def test_position_limit_total_risk_and_sector_limit_are_independent():
    bars = frame([(100, 101, 99, 100)])
    signals = [signal(str(i), stop_price=90) for i in range(6)]
    frames = {str(i): bars for i in range(6)}
    result = simulate_portfolio(signals, frames, config(max_total_risk_fraction=.1))
    assert len(result["trades"]) == 5
    assert result["rejections"][0]["reason"] == "POSITION_LIMIT"
    risk = simulate_portfolio(signals, frames, config(max_total_risk_fraction=.005))
    assert len(risk["trades"]) == 1
    sector = simulate_portfolio([signal("A", sector="Tech"), signal("B", sector="Tech")],
                                {"A": bars, "B": bars}, config(max_sector_fraction=.1))
    assert len(sector["trades"]) == 1
    assert sector["rejections"][0]["reason"] == "CAPITAL_OR_RISK_LIMIT"


def test_same_symbol_and_duplicate_signal_are_distinguished():
    bars = frame([(100, 101, 99, 100), (100, 101, 99, 100)])
    s = signal("A", signal_id="first")
    result = simulate_portfolio([s, s, signal("A", bars.index[1], signal_id="second")], {"A": bars}, config())
    assert len(result["trades"]) == 1
    assert {x["reason"] for x in result["rejections"]} == {"DUPLICATE_SIGNAL", "SYMBOL_ALREADY_OPEN"}


def test_correlation_controls_use_only_supplied_point_in_time_pairs():
    bars = frame([(100, 101, 99, 100)])
    known = simulate_portfolio([signal("A", score=90), signal("B", score=80, correlations={"A": .9})],
                               {"A": bars, "B": bars}, config())
    assert known["rejections"][0]["reason"] == "CORRELATED_EXPOSURE"
    unknown = simulate_portfolio([signal("A"), signal("B")], {"A": bars, "B": bars}, config())
    assert len(unknown["trades"]) == 2


def test_intrabar_cash_cannot_be_reused_at_earlier_same_bar_open():
    bars = frame([(100, 103, 99, 102), (100, 103, 99, 102)])
    result = simulate_portfolio([signal("A", score=90), signal("B", score=80), signal("C", bars.index[1])],
                                {"A": bars, "B": bars, "C": bars}, config(max_open_positions=1))
    assert [t["symbol"] for t in result["trades"]] == ["A", "C"]
    assert result["rejections"][0]["symbol"] == "B"
    assert result["metrics"]["average_capital_utilisation"] > 0
    assert result["metrics"]["average_exposure"] == 0  # correctly labelled daily-close sampling


def test_partial_open_position_equity_reconciles_realized_and_unrealized_pnl():
    bars = frame([(100, 103, 99, 102)])
    result = simulate_portfolio([signal("A")], {"A": bars}, config(target_r=2, partial_fraction=.5, commission_per_order=.5))
    trade = result["trades"][0]
    assert trade["status"] == "OPEN"
    assert result["metrics"]["ending_equity"] == pytest.approx(10_000 + trade["net_pnl"] + trade["unrealized_pnl"])
    assert result["metrics"]["closed_trades"] == 0
    assert result["metrics"]["net_expectancy_r"] is None


def test_stale_marks_are_disclosed_and_short_history_has_no_annualized_claim():
    a = frame([(100, 101, 99, 100)], ["2026-10-05 15:55"])
    b = frame([(100, 101, 99, 100)], ["2026-10-06 15:55"])
    result = simulate_portfolio([signal("A", a.index[0])], {"A": a, "B": b}, config())
    assert result["equity"][-1]["stale_symbols"] == ["A"]
    assert result["metrics"]["sharpe"] is None
    assert result["metrics"]["sortino"] is None
    assert result["metrics"]["annualized_return"] is None


def test_empty_data_missing_symbol_and_json_are_explicit():
    result = simulate_portfolio([signal("MISSING")], {}, config())
    assert result["rejections"][0]["reason"] == "NO_EXECUTABLE_BAR"
    assert result["metrics"]["ending_equity"] == 10_000
    json.dumps(result, allow_nan=False)
    assert simulate_portfolio([], {}, config())["metrics"]["executed_trades"] == 0


def test_inputs_and_configuration_remain_immutable_and_replay_is_deterministic():
    bars = frame([(100, 103, 99, 102)])
    original = bars.copy(deep=True)
    signals = [signal("A", correlations={"B": .9})]
    cfg = config()
    first = simulate_portfolio(signals, {"A": bars}, cfg)
    second = simulate_portfolio(signals, {"A": bars}, cfg)
    pd.testing.assert_frame_equal(bars, original)
    assert first == second
    assert signals[0]["correlations"] == {"B": .9}
    assert cfg.initial_capital == 10_000
    json.dumps(first, allow_nan=False)


def test_regular_session_timezone_conversion_and_invalid_bars():
    bars = frame([(50, 51, 49, 50), (100, 101, 99, 100), (500, 501, 499, 500)],
                 ["2026-10-05 09:25", "2026-10-05 09:30", "2026-10-05 16:00"])
    trade = simulate_trade(bars.tz_convert("UTC"), bars.index[0], 98, config=config())
    assert trade["raw_entry_price"] == 100
    assert trade["bars_held"] == 1
    with pytest.raises(ValueError, match="timezone"):
        simulate_trade(bars.tz_localize(None), bars.index[0], 98)
    with pytest.raises(ValueError, match="duplicate"):
        simulate_trade(pd.concat([bars, bars]), bars.index[0], 98)
    broken = bars.copy()
    broken.iloc[1, broken.columns.get_loc("High")] = 90
    with pytest.raises(ValueError, match="inconsistent"):
        simulate_trade(broken, bars.index[0], 98)


def test_missing_intraday_bars_cannot_skip_a_stop_and_credit_a_later_target():
    bars = frame([(100, 101, 99, 100), (100, 105, 99, 104)],
                 ["2026-10-05 09:30", "2026-10-05 09:40"])
    cfg = config(require_contiguous_bars=True)
    trade = simulate_trade(bars, bars.index[0], 98, config=cfg)
    assert trade["status"] == "OPEN"
    assert trade["legs"] == []
    assert trade["data_quality_flags"] == ["MISSING_BARS_UNRESOLVED"]
    assert pd.Timestamp(trade["mark_time"]) == bars.index[0] + pd.Timedelta(minutes=5)
    result = simulate_portfolio([signal("A"), signal("B", bars.index[1])], {"A": bars, "B": bars}, cfg)
    assert result["metrics"]["equity_path_reliable"] is False
    assert result["metrics"]["total_return"] is None
    assert result["equity"][-1]["unresolved_symbols"] == ["A"]
    assert result["rejections"][0]["reason"] == "UNRESOLVED_OPEN_POSITION"


def test_missing_session_tail_or_open_is_unresolved_but_holiday_gap_is_not_inferred():
    complete = frame([(100, 101, 99, 100), (96, 97, 95, 96)],
                     ["2026-09-04 15:55", "2026-09-08 09:30"])
    cfg = config(require_contiguous_bars=True)
    trade = simulate_trade(complete, complete.index[0], 98, config=cfg)
    assert trade["exit_reason"] == "STOP_GAP"  # weekend and Labor Day are not unknown intraday bars
    missing_tail = frame([(100, 101, 99, 100), (103, 104, 102, 103)],
                         ["2026-10-05 15:50", "2026-10-06 09:30"])
    assert simulate_trade(missing_tail, missing_tail.index[0], 98, config=cfg)["status"] == "OPEN"
    missing_open = frame([(100, 101, 99, 100), (103, 104, 102, 103)],
                         ["2026-10-05 15:55", "2026-10-06 09:35"])
    assert simulate_trade(missing_open, missing_open.index[0], 98, config=cfg)["status"] == "OPEN"


def test_entire_missing_scheduled_session_cannot_credit_later_target():
    tuesday = pd.date_range("2026-10-06 09:30", "2026-10-06 15:55", freq="5min", tz="America/New_York")
    thursday = pd.DatetimeIndex(["2026-10-08 09:30"], tz="America/New_York")
    bars = frame([(100, 101, 99, 100)] * len(tuesday) + [(103, 104, 102, 103)],
                 tuesday.append(thursday))
    cfg = config(require_contiguous_bars=True)
    trade = simulate_trade(bars, bars.index[0], 98, config=cfg)
    assert trade["status"] == "OPEN"
    assert trade["legs"] == []
    assert trade["net_r"] is None
    assert trade["bars_held"] == 78
    assert trade["data_quality_flags"] == ["MISSING_BARS_UNRESOLVED"]
    assert pd.Timestamp(trade["mark_time"]) == pd.Timestamp("2026-10-06T16:00:00-04:00")
    assert pd.Timestamp(trade["unresolved_from"]) == pd.Timestamp("2026-10-06T16:00:00-04:00")
    result = simulate_portfolio([signal("A", bars.index[0])], {"A": bars}, cfg)
    assert result["metrics"]["equity_path_reliable"] is False
    assert result["metrics"]["total_return"] is None
    assert result["metrics"]["net_pnl"] is None
    assert result["equity"][-1]["unresolved_symbols"] == ["A"]


@pytest.mark.parametrize("previous,last_bar,following", [
    ("2026-09-04 15:55", False, "2026-09-08 09:30"),  # weekend plus Labor Day
    ("2026-11-25 15:55", False, "2026-11-27 09:30"),  # Thanksgiving
    ("2026-11-27 12:55", True, "2026-11-30 09:30"),  # early close plus weekend
    ("2026-12-31 15:55", False, "2027-01-04 09:30"),  # New Year plus weekend
])
def test_expected_next_session_open_allows_known_overnight_target(previous, last_bar, following):
    bars = frame([(100, 101, 99, 100), (103, 104, 102, 103)], [previous, following])
    bars["session_end"] = [last_bar, False]
    trade = simulate_trade(bars, bars.index[0], 98, config=config(require_contiguous_bars=True))
    assert trade["status"] == "CLOSED"
    assert trade["exit_reason"] == "TARGET"
    assert pd.Timestamp(trade["exit_time"]) == bars.index[1]
    assert trade["data_quality_flags"] == []


def test_inter_session_calendar_failure_keeps_held_path_unresolved(monkeypatch):
    bars = frame([(100, 101, 99, 100), (103, 104, 102, 103)],
                 ["2026-10-06 15:55", "2026-10-07 09:30"])
    monkeypatch.setattr(market_sessions, "_calendar", lambda year: None)
    trade = simulate_trade(bars, bars.index[0], 98, config=config(require_contiguous_bars=True))
    assert trade["status"] == "OPEN"
    assert trade["legs"] == []
    assert trade["data_quality_flags"] == ["MISSING_BARS_UNRESOLVED"]


def test_archive_may_start_at_entry_without_inventing_pre_entry_missing_data():
    bars = frame([(100, 103, 99, 102)], ["2026-10-05 12:00"])
    trade = simulate_trade(bars, bars.index[0], 98, config=config(require_contiguous_bars=True))
    assert trade["status"] == "CLOSED"
    assert trade["data_quality_flags"] == []


@pytest.mark.parametrize("kwargs", [{"target_r": 0}, {"partial_fraction": .5}, {"stop_model": "unknown"},
                                    {"slippage_bps": -1}, {"max_holding_bars": 0}, {"max_abs_correlation": 2}])
def test_invalid_experiment_configuration_cannot_silently_run(kwargs):
    with pytest.raises(ValueError):
        replace(ExecutionConfig(), **kwargs)

def test_cash_only_sessions_are_included_in_equal_window_curves():
    bars=frame([(100,101,99,100),(100,101,99,100)],['2026-10-05 15:55','2026-10-06 15:55'])
    empty=simulate_portfolio([],{'A':bars},config=config())
    assert len(empty['equity'])==2
    assert all(p['equity']==10000 and p['invested_value']==0 for p in empty['equity'])
    delayed=simulate_portfolio([signal('A','2026-10-06 15:55-04:00')],{'A':bars},config=config())
    assert len(delayed['equity'])==2
    assert delayed['equity'][0]['equity']==10000
    assert delayed['equity'][0]['open_positions']==0

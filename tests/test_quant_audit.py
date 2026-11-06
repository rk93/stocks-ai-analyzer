import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from quant_audit import (INPUTS, block_interval, build_report, frozen_value,
                         join_frozen, publish)


def trade(**updates):
    row = {"trade_id": "one", "strategy": "V3", "symbol": "TEST", "entry_time": "2026-10-01T15:00:00Z",
           "exit_time": "2026-10-02T15:00:00Z", "entry_price": 100.07, "exit_price": 98.93,
           "qty": 5, "pnl": -5.70, "pnl_pct": -.0114, "signal_score": 82,
           "market_regime": "BULL_TREND", "exit_reason": "STOP"}
    return {**row, **updates}


def prediction(**updates):
    row = {"prediction_id": "frozen", "strategy_version": "V3", "symbol": "TEST", "signal_date": "2026-10-01",
           "generated_at": "2026-10-01T14:59:00Z", "frozen_at": "2026-10-01T14:59:01Z",
           "price": 100, "score": 99, "stop": 50, "reason": "generic V1",
           "prediction_score": 82, "prediction_stop": 98, "prediction_reason": "frozen V3",
           "prediction_target_1r": 102, "v3_score": 81, "v3_stop": 97,
           "vwap_dist": .004, "volume_ratio_tod": 1.8}
    return {**row, **updates}


def replay(**updates):
    row = {"strategy": "V3", "symbol": "TEST", "signal_date": "2026-10-01", "signal_time": "2026-10-01T14:00:00Z",
           "score": 82, "entry": 100, "stop": 98, "target_1r": 102,
           "realized_r_1r": -1, "trade_result_1r": "STOP"}
    return {**row, **updates}


class FrozenJoinTests(unittest.TestCase):
    def test_only_strategy_specific_original_fields(self):
        row = prediction()
        self.assertEqual(frozen_value(row, "V3", "reason"), "frozen V3")
        self.assertEqual(frozen_value(row, "V3", "stop"), 98)
        row["prediction_stop"] = None
        self.assertEqual(frozen_value(row, "V3", "stop"), 97)
        self.assertIsNone(frozen_value({"stop": 50}, "V3", "stop"))

    def test_later_frozen_record_is_never_attached(self):
        source = prediction(frozen_at="2026-10-01T16:00:00Z")
        result = join_frozen(pd.DataFrame([trade()]), pd.DataFrame([source])).iloc[0]
        self.assertEqual(result.join_status, "not_frozen_before_entry")
        self.assertIsNone(result.vwap_dist)
        self.assertIsNone(result.prediction_id)

    def test_later_generated_record_is_never_attached(self):
        source = prediction(generated_at="2026-10-01T16:00:00Z")
        result = join_frozen(pd.DataFrame([trade()]), pd.DataFrame([source])).iloc[0]
        self.assertEqual(result.join_status, "not_frozen_before_entry")

    def test_wrong_strategy_and_future_day_cannot_fill_unknown_fields(self):
        source = pd.DataFrame([prediction(strategy_version="V1"), prediction(signal_date="2026-10-02")])
        result = join_frozen(pd.DataFrame([trade()]), source).iloc[0]
        self.assertEqual(result.join_status, "missing")
        self.assertIsNone(result.vwap_dist)

    def test_duplicate_original_records_are_explicitly_ambiguous(self):
        source = pd.DataFrame([prediction(), prediction(prediction_id="second")])
        result = join_frozen(pd.DataFrame([trade()]), source).iloc[0]
        self.assertEqual(result.join_status, "ambiguous_multiple_frozen_records")

    def test_ny_trading_date_not_utc_calendar_date(self):
        p = trade(entry_time="2026-10-02T02:00:00Z")
        result = join_frozen(pd.DataFrame([p]), pd.DataFrame([prediction()])).iloc[0]
        self.assertEqual(result.signal_date, "2026-10-01")
        self.assertEqual(result.join_status, "unique_prior_frozen")
        self.assertTrue(result.outside_regular_hours)
        self.assertFalse(result.original_geometry_confirmed)

    def test_half_open_regular_session_and_weekends(self):
        entries = ["2026-10-01T13:29:00Z", "2026-10-01T13:30:00Z", "2026-10-01T19:59:00Z",
                   "2026-10-01T20:00:00Z", "2026-10-03T15:00:00Z"]
        result = join_frozen(pd.DataFrame([trade(entry_time=t) for t in entries]), pd.DataFrame())
        self.assertEqual(result.outside_regular_hours.tolist(), [True, False, False, True, True])


class DiagnosticsTests(unittest.TestCase):
    def test_ambiguous_and_open_trades_preserved_but_not_counted_as_wins(self):
        history = pd.DataFrame([replay(), replay(trade_result_1r="WIN_1R", realized_r_1r=1),
                                replay(trade_result_1r="AMBIGUOUS", realized_r_1r=None),
                                replay(trade_result_1r="OPEN", realized_r_1r=None)])
        report = build_report(pd.DataFrame([trade()]), history, pd.DataFrame([prediction()]))
        result = report["strategies"]["V3"]["backtest"]
        self.assertEqual((result["signals"], result["count"], result["ambiguous"], result["pending"]), (4, 2, 1, 1))
        self.assertEqual(result["expectancy_r"], 0)
        self.assertEqual(len(report["trades"]["backtest"]), 4)

    def test_winning_more_than_half_can_still_have_negative_expectancy(self):
        rows = [trade(pnl=1, pnl_pct=.01), trade(pnl=1, pnl_pct=.01), trade(pnl=-4, pnl_pct=-.04)]
        report = build_report(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())
        result = report["strategies"]["V3"]["paper"]
        self.assertAlmostEqual(result["win_rate"], 2/3)
        self.assertEqual(result["mean_pnl"], -2/3)
        self.assertEqual(result["payoff_ratio"], .25)
        self.assertEqual(result["break_even_win_rate"], .8)

    def test_date_block_uncertainty_does_not_invent_independent_days(self):
        self.assertIsNone(block_interval([1]*100, ["2026-10-01"]*100))
        interval = block_interval([1]*100+[-1]*100, ["a"]*100+["b"]*100)
        self.assertEqual(interval, [-1, 1])
        self.assertEqual(interval, block_interval([1]*100+[-1]*100, ["a"]*100+["b"]*100))

    def test_scores_are_never_relabelled_probabilities(self):
        report = build_report(pd.DataFrame([trade()]), pd.DataFrame([replay()]), pd.DataFrame([prediction()]))
        self.assertIsNone(report["methodology"]["brier_score"])
        self.assertEqual(report["methodology"]["probability_calibration"], "not_available")
        self.assertEqual(report["strategies"]["V3"]["score_buckets"]["paper"][0]["bucket"], "80–89")
        self.assertEqual(report["coverage"]["backtest_features"]["rsi"], {"known": 0, "missing": 1})

    def test_empty_inputs_are_unknown_without_profitability_claims(self):
        report = build_report(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertEqual(report["coverage"]["paper_closed_trades"], 0)
        self.assertIsNone(report["strategies"]["V1"]["paper"]["win_rate"])
        json.dumps(report, allow_nan=False)

    def test_publish_is_deterministic_and_never_mutates_ledgers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"data").mkdir()
            pd.DataFrame([trade()]).to_csv(root/INPUTS[0], index=False)
            pd.DataFrame([replay()]).to_csv(root/INPUTS[1], index=False)
            pd.DataFrame([prediction()]).to_csv(root/INPUTS[2], index=False)
            (root/INPUTS[3]).write_text(json.dumps({"generated_at": "2026-11-05T20:00:00Z", "strategies": {}}))
            (root/INPUTS[4]).write_text(json.dumps({"generated_at": "2026-11-05T21:00:00Z"}))
            before = {p: (root/p).read_bytes() for p in INPUTS}
            first = publish(root)
            content = (root/"docs/data/quant-audit.json").read_bytes()
            second = publish(root)
            self.assertEqual(first, second)
            self.assertEqual(content, (root/"docs/data/quant-audit.json").read_bytes())
            for p in INPUTS:
                self.assertEqual(before[p], (root/p).read_bytes())
                metadata = next(x for x in first["inputs"] if x["path"] == p)
                self.assertEqual(metadata["sha256"], hashlib.sha256(before[p]).hexdigest())
            with self.assertRaisesRegex(ValueError, "source ledger"):
                publish(root, root/INPUTS[0])
            self.assertEqual(before[INPUTS[0]], (root/INPUTS[0]).read_bytes())

    def test_missing_or_invalid_sources_are_actionable_incomplete(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"data").mkdir()
            (root/INPUTS[0]).write_text("unrelated\nvalue\n")
            report = publish(root)
            self.assertEqual(report["pipeline_status"], "incomplete")
            self.assertTrue(any("missing required columns" in error for error in report["input_errors"]))
            self.assertEqual(report["coverage"]["paper_closed_trades"], 0)


if __name__ == "__main__":
    unittest.main()

"""Static publication preserves evidence cohorts and never manufactures marks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from v4_publish import OPTIONAL_SOURCES, SOURCES, build_report, canonical_hash, prospective_view, publish


CONFIG = {"schema_version": 1, "experiment_id": "frozen-example", "status": "research-only"}
HASH = canonical_hash(CONFIG)


def inputs():
    signal = {"signal_id": "frozen", "strategy": "V4", "symbol": "AAPL", "signal_time": "2026-10-01T15:00:00Z",
              "rationale": "Prior-session trend", "context": {"intraday_asof": "2026-10-01T15:00:00Z"},
              "checks": [{"name": "trend", "passed": True}, {"name": "volume", "passed": False}]}
    trade = {"signal_id": "frozen", "symbol": "AAPL", "entry_time": "2026-10-01T15:00:00Z", "status": "CLOSED",
             "quantity": 2, "entry_price": 100, "stop_price": 98, "target_price": 102, "net_r": -1.1,
             "net_pnl": -4.4, "market_regime": "BULL_TREND"}
    replay = {"generated_at": "2026-11-05T21:00:00Z", "market_data_at": "2026-11-04T21:00:00Z", "config_hash": HASH,
              "status": "retrospective", "manifest_sha256": "sealed-archive-hash", "source_sha256": {"strategy_v4.py": "frozen-code"},
              "comparisons": [{"strategy": "V4", "window": "retrospective_tail", "signals": [signal],
                               "trades": [trade], "equity": [{"time": "2026-10-01T20:00:00Z", "equity": 9995.6}],
                               "metrics": {"signals": 1, "executed_trades": 1, "closed_trades": 1, "net_expectancy_r": -1.1},
                               "confidence_interval": {"lower": -.7, "upper": .2, "method": "date blocks"}}],
              "validation": {"chronological_windows": {"retrospective_tail": ["2026-10-01"]}},
              "experiments": [{"id": "two-r", "window": "development", "status": "failed", "metrics": {"net_expectancy_r": -2}}]}
    forward = {"generated_at": "2026-11-06T12:00:00Z", "created_at": "2026-11-06T12:00:00Z", "config_hash": HASH,
               "status": "research_only_inconclusive", "starting_cash": 10000,
               "strategies": {version: {"metrics": {"ending_equity": 10000, "closed_trades": 0}, "trades": [], "equity": []}
                              for version in ("V1", "V2", "V3", "V4")},
               "signals": [], "benchmarks": {"CASH": {"equity": 10000, "return": 0}}}
    legacy = {"generated_at": "2026-11-05T20:00:00Z", "strategies": {"V1": {"equity": 9797.29, "return": -.020271,
              "closed_trades": 66, "open_positions": 0, "win_rate": .515, "max_drawdown": -.03, "profit_factor": .72}}}
    audit = {"as_of": "2026-11-05T21:00:00Z", "pipeline_status": "complete", "strategies": {}}
    return replay, forward, audit, legacy, deepcopy(CONFIG)


def write_inputs(root, values=None):
    values = dict(zip(("replay", "forward", "audit", "legacy_forward", "config"), values or inputs()))
    for name, value in values.items():
        path = root/SOURCES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))


class CohortPublicationTests(unittest.TestCase):
    def test_empty_is_inconclusive_without_synthetic_observations(self):
        report = build_report()
        self.assertEqual(report["status"], "research_only_inconclusive")
        self.assertEqual(report["pipeline_status"], "incomplete")
        self.assertIsNone(report["generated_at"])
        self.assertEqual(report["prospective"]["equity"], [])
        self.assertEqual(report["prospective"]["signals"], [])
        self.assertIn("no untouched historical test", report["validation"]["untouched_test_status"])
        json.dumps(report, allow_nan=False)

    def test_legacy_replay_and_new_forward_never_share_a_cohort(self):
        values = inputs()
        report = build_report(*values)
        self.assertEqual([row["source"] for row in report["comparisons"]], ["corrected_equal_execution_replay", "legacy_forward"])
        self.assertEqual(report["comparisons"][1]["metrics"]["closed_trades"], 66)
        self.assertIsNone(report["comparisons"][1]["metrics"]["net_expectancy_r"])
        self.assertEqual(len(report["retrospective"]["trades"]), 1)
        self.assertEqual(report["prospective"]["closed_trades"], [])
        self.assertEqual(report["prospective"]["equity"], [])
        self.assertEqual(report["prospective"]["metrics"]["V1"]["closed_trades"], 0)
        self.assertEqual(report["retrospective"]["manifest_sha256"], "sealed-archive-hash")
        self.assertEqual(report["retrospective"]["source_sha256"]["strategy_v4.py"], "frozen-code")

    def test_confidence_interval_and_failed_experiment_preserved(self):
        report = build_report(*inputs())
        row = report["comparisons"][0]
        self.assertEqual(row["metrics"]["expectancy_r_ci95"], [-.7, .2])
        self.assertEqual(row["bootstrap"]["method"], "date blocks")
        self.assertEqual(report["experiments"][0]["status"], "failed")
        self.assertNotIn("signals", row)  # Full feature packets remain in replay download.
        self.assertNotIn("trades", row)  # Drill-down has one compact copy.
        trade = report["retrospective"]["trades"][0]
        self.assertEqual(trade["reason"], "Prior-session trend")
        self.assertEqual(trade["supports"], ["trend"])
        self.assertEqual(trade["rejections"], ["volume"])

    def test_hash_mismatch_omits_only_incompatible_cohort_and_keeps_provenance(self):
        values = inputs()
        values[0]["config_hash"] = "different"
        report = build_report(*values)
        self.assertEqual([row["source"] for row in report["comparisons"]], ["legacy_forward"])
        self.assertEqual(report["source_states"]["replay"]["config_hash"], "different")
        self.assertEqual(report["prospective"]["config_hash"], HASH)
        self.assertEqual(report["pipeline_status"], "incomplete")
        self.assertTrue(any("configuration hash" in s for s in report["validation"]["limitations"]))

    def test_publication_timestamp_does_not_invent_market_cutoff(self):
        report = build_report(*inputs())
        self.assertEqual(report["generated_at"], "2026-11-06T12:00:00+00:00")
        self.assertEqual(report["market_data_at"], "2026-11-04T21:00:00Z")
        self.assertIsNone(report["prospective"]["data_cutoff"])
        self.assertIsNone(report["prospective"]["collection_started_at"])

    def test_input_objects_are_never_mutated(self):
        values = inputs()
        original = deepcopy(values)
        build_report(*values)
        self.assertEqual(values, original)

    def test_paused_forward_and_upstream_failure_remain_explicit(self):
        values = inputs()
        values[1].update(status="paused_reconciliation_required", status_at="2026-11-06T13:00:00Z", warnings=["Corporate action"])
        values[2]["pipeline_status"] = "incomplete"
        report = build_report(*values)
        self.assertEqual(report["prospective"]["status"], "paused_reconciliation_required")
        self.assertEqual(report["prospective"]["status_at"], "2026-11-06T13:00:00Z")
        self.assertEqual(report["prospective"]["generated_at"], values[1]["generated_at"])
        self.assertTrue(any("paused" in s for s in report["validation"]["limitations"]))
        self.assertTrue(any("Corporate action" in s for s in report["validation"]["limitations"]))
        self.assertEqual(report["pipeline_status"], "incomplete")

    def test_charts_join_exact_observation_times_without_forward_filling(self):
        forward = inputs()[1]
        forward["strategies"]["V1"]["equity"] = [{"time": "2026-11-06T15:00:00Z", "equity": 9990}]
        forward["strategies"]["V4"]["equity"] = [{"time": "2026-11-06T15:05:00Z", "equity": 10001}]
        forward["benchmarks"]["SPY"] = {"asof": "2026-11-06T15:05:00Z", "equity": 10010}
        report = prospective_view(forward)
        self.assertEqual(len(report["equity"]), 2)
        self.assertNotIn("V4_equity", report["equity"][0])
        self.assertNotIn("V1_equity", report["equity"][1])
        self.assertEqual(report["equity"][1]["SPY_equity"], 10010)
        self.assertIsNone(report["data_cutoff"])
        self.assertEqual(report["latest_equity_observation_at"], "2026-11-06T15:05:00+00:00")

    def test_unreliable_stale_and_undated_marks_are_not_chart_evidence(self):
        forward = inputs()[1]
        forward["strategies"]["V4"]["equity"] = [
            {"time": "2026-11-06T15:00:00Z", "equity": 10005, "equity_reliable": False},
            {"time": "2026-11-06T15:05:00Z", "equity": 10005, "stale_symbols": ["AAPL"]},
            {"equity": 10006},
        ]
        result = prospective_view(forward)
        self.assertEqual([r["V4_equity"] for r in result["equity"]], [None, None])
        self.assertEqual(len(result["warnings"]), 3)
        self.assertEqual(forward["strategies"]["V4"]["equity"][0]["equity"], 10005)

    def test_nonfinite_metrics_do_not_become_profitability_evidence(self):
        values = inputs()
        values[0]["comparisons"][0]["metrics"]["profit_factor"] = float("inf")
        values[0]["comparisons"][0]["confidence_interval"] = {"lower": float("nan"), "upper": 1}
        result = build_report(*values)
        self.assertIsNone(result["comparisons"][0]["metrics"]["profit_factor"])
        self.assertIsNone(result["comparisons"][0]["confidence_interval"])
        json.dumps(result, allow_nan=False)


class FilePublicationTests(unittest.TestCase):
    def test_deterministic_and_no_source_mutations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root)
            original = {name: (root/path).read_bytes() for name, path in SOURCES.items() if name not in OPTIONAL_SOURCES}
            first = publish(root)
            output = root/"docs/data/v4-research.json"
            body = output.read_bytes()
            self.assertEqual(first["pipeline_status"], "complete")
            self.assertEqual(publish(root), first)
            self.assertEqual(output.read_bytes(), body)
            self.assertEqual(original, {name: (root/SOURCES[name]).read_bytes() for name in original})
            for source in first["sources"]:
                if source["name"] in OPTIONAL_SOURCES:
                    continue
                self.assertEqual(source["sha256"], hashlib.sha256(original[source["name"]]).hexdigest())
                self.assertTrue(source["valid"])
            self.assertFalse(output.with_name(output.name+".tmp").exists())

    def test_missing_files_produce_an_explicit_incomplete_empty_report(self):
        with TemporaryDirectory() as directory:
            result = publish(Path(directory))
            self.assertEqual(len(result["input_errors"]), 5)
            self.assertEqual(result["comparisons"], [])
            self.assertEqual(result["prospective"]["signals"], [])
            self.assertEqual(result["pipeline_status"], "incomplete")

    def test_malformed_json_root_container_and_nonstandard_nan_fail_closed(self):
        for invalid in ("{", "[]", '{"strategies":[]}', '{"signals":[1]}', '{"starting_cash":NaN}'):
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                root = Path(directory)
                write_inputs(root)
                (root/SOURCES["forward"]).write_text(invalid)
                result = publish(root)
                self.assertTrue(any("v4-forward.json" in error for error in result["input_errors"]))
                self.assertEqual(result["prospective"]["signals"], [])
                self.assertEqual(result["prospective"]["equity"], [])
                self.assertEqual(result["pipeline_status"], "incomplete")

    def test_source_and_market_age_are_separate_and_never_reset_by_publication(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = inputs()
            values[0]["generated_at"] = "2026-11-06T10:00:00Z"
            values[0]["market_data_at"] = "2026-10-01T20:00:00Z"
            write_inputs(root, values)
            result = publish(root)
            source = next(row for row in result["sources"] if row["name"] == "replay")
            self.assertFalse(source["stale_at_report"])
            self.assertTrue(source["market_data_stale_at_report"])
            self.assertEqual(result["market_data_at"], "2026-10-01T20:00:00Z")
            self.assertTrue(any("dated source" in row for row in result["validation"]["limitations"]))

    def test_candidate_sample_is_optional_and_never_changes_metrics(self):
        values = inputs()
        candidates = {"config_hash": HASH, "manifest_sha256": values[0]["manifest_sha256"], "status": "retrospective_sample",
                      "decisions": [{"strategy": "V4", "symbol": "AAPL", "accepted": False, "rejections": ["Regime gate"]}],
                      "limitations": ["Predetermined illustration grid; not exhaustive"]}
        result = build_report(*values, candidates=candidates)
        self.assertEqual(result["comparisons"], build_report(*values)["comparisons"])
        self.assertEqual(result["retrospective"]["candidate_sample"]["count"], 1)
        self.assertEqual(result["retrospective"]["decisions"][0]["source"], "retrospective_sample")
        self.assertEqual(result["prospective"]["signals"], [])
        self.assertIn(candidates["limitations"][0], result["validation"]["limitations"])
        candidates["manifest_sha256"] = "different-archive"
        self.assertEqual(build_report(*values, candidates=candidates)["retrospective"]["candidate_sample"]["count"], 0)

    def test_output_cannot_overwrite_source_config_or_any_legacy_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root)
            for target in list(SOURCES.values())+["data/predictions.csv", "data/new-ledger.json"]:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    publish(root, root/target)


if __name__ == "__main__":
    unittest.main()

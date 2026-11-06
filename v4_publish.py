"""Publish static V4 research views without running strategies or editing ledgers.

The aggregate retains execution cohorts, source dates and configuration hashes.
It does not turn retrospective windows into untouched tests or create forward
observations. Run ``python v4_publish.py`` after the source publishers.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SOURCES = {
    "replay": "docs/data/v4-replay.json",
    "forward": "docs/data/v4-forward.json",
    "audit": "docs/data/quant-audit.json",
    "legacy_forward": "data/paper_summary.json",
    "config": "research/v4_config.json",
    "candidates": "docs/data/v4-candidates.json",
}
OPTIONAL_SOURCES = {"candidates"}
VERSIONS = ("V1", "V2", "V3", "V4")


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def time(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def latest(values):
    valid = [time(v) for v in values]
    return max((v for v in valid if v is not None), default=None)


def items(value):
    return value if isinstance(value, list) else []


def mapping(value):
    return value if isinstance(value, dict) else {}


def records(value):
    return [row for row in items(value) if isinstance(row, dict)]


def unique(values):
    return list(dict.fromkeys(str(v) for v in values if v))


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_hash(config):
    body = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def validate_payload(name, value):
    """Fail closed on malformed containers, before reading nested financial rows."""
    containers = {
        "replay": {"comparisons": list, "benchmarks": list, "experiments": list, "decisions": list, "validation": dict},
        "forward": {"strategies": dict, "benchmarks": dict, "signals": list, "warnings": list},
        "audit": {"strategies": dict}, "legacy_forward": {"strategies": dict}, "config": {},
        "candidates": {"decisions": list, "limitations": list},
    }
    for key, kind in containers[name].items():
        if key in value and not isinstance(value[key], kind):
            raise ValueError(f"{key} must be a {kind.__name__}")
    for key in ("comparisons", "benchmarks", "experiments", "decisions", "signals"):
        if isinstance(value.get(key), list) and any(not isinstance(row, dict) for row in value[key]):
            raise ValueError(f"{key} must contain objects")
    for key in ("strategies", "benchmarks"):
        if isinstance(value.get(key), dict) and any(not isinstance(row, dict) for row in value[key].values()):
            raise ValueError(f"{key} must map names to objects")
    return value


def interval(source):
    if isinstance(source, list) and len(source) == 2:
        return [number(v) for v in source] if all(number(v) is not None for v in source) else None
    if isinstance(source, dict):
        ends = [source.get("lower"), source.get("upper")]
        return [number(v) for v in ends] if all(number(v) is not None for v in ends) else None
    return None


def metrics_aliases(source, confidence=None):
    result = deepcopy(source) if isinstance(source, dict) else {}
    aliases = {"equity": "ending_equity", "return": "total_return"}
    for target, origin in aliases.items():
        if target not in result and origin in result:
            result[target] = result[origin]
    if "executed_trades" not in result and number(result.get("trades")) is not None:
        result["executed_trades"] = result["trades"]
    if confidence is not None:
        result["confidence_interval"] = interval(confidence)
        result["expectancy_r_ci95"] = interval(confidence)
    return result


def decision_view(signal, strategy, source, window=None):
    result = deepcopy(signal)
    result.update(strategy=strategy, source=source, window=window)
    result.setdefault("stop", result.get("stop_price"))
    result.setdefault("target", result.get("target_1r", result.get("target_price")))
    result.setdefault("reason", result.get("rationale"))
    result.setdefault("data_cutoff", result.get("context", {}).get("intraday_asof") if isinstance(result.get("context"), dict) else None)
    checks = items(result.get("checks"))
    if checks:
        result.setdefault("supports", [c.get("name") for c in checks if isinstance(c, dict) and c.get("passed")])
        result.setdefault("rejections", [c.get("name") for c in checks if isinstance(c, dict) and not c.get("passed")])
    return result


def trade_view(trade, signal, strategy, source, window=None):
    result = decision_view(signal or {}, strategy, source, window)
    result.update(deepcopy(trade))
    result.update(strategy=strategy, source=source, window=window)
    result["stop"] = trade.get("stop_price", result.get("stop"))
    result["target"] = trade.get("target_price", result.get("target"))
    result.setdefault("qty", trade.get("quantity"))
    # The engine's last observed mark must remain missing if it was not logged.
    result.setdefault("last_price", trade.get("mark_price"))
    result.setdefault("data_cutoff", trade.get("mark_time"))
    result["accepted"] = trade.get("status") != "REJECTED"
    if trade.get("status") == "REJECTED":
        result["decision"] = "Execution rejected"
        result["rejections"] = unique(items(result.get("rejections"))+[trade.get("reason")])
    # Full legs, correlation matrices and timestamp packets remain in the
    # source download. Avoid repeating them in each static drill-down row.
    for field in ("legs", "correlations", "context"):
        result.pop(field, None)
    if result.get("rationale") == result.get("reason"):
        result.pop("rationale", None)
    return result


def prospective_view(forward):
    forward = deepcopy(forward)
    result = {"created_at": forward.get("created_at"), "collection_started_at": forward.get("collection_started_at"),
              "generated_at": forward.get("generated_at"), "config_hash": forward.get("config_hash"),
              "starting_cash": forward.get("starting_cash", 10000), "status": forward.get("status", "awaiting_prospective_collection"),
              "equity": [], "positions": [], "closed_trades": [], "signals": items(forward.get("signals")),
              "metrics": {}, "strategies": mapping(forward.get("strategies")), "benchmarks": mapping(forward.get("benchmarks")),
              "warnings": items(forward.get("warnings")), "data_cutoff": forward.get("data_cutoff"), "note": forward.get("note")}
    result["status_at"] = forward.get("status_at")
    lookup = {s.get("signal_id", s.get("decision_id")): s for s in result["signals"] if isinstance(s, dict)}
    snapshots = {}
    cuts = []
    for version, run in result["strategies"].items():
        if not isinstance(run, dict):
            continue
        result["metrics"][version] = metrics_aliases(run.get("metrics"))
        for point in records(run.get("equity")):
            stamp = time(point.get("time", point.get("timestamp")))
            if stamp is None:
                result["warnings"].append(f"{version}: undated equity mark excluded from chart")
                continue
            key = stamp.isoformat()
            row = snapshots.setdefault(key, {"timestamp": key})
            reliable = point.get("equity_reliable") is not False and not items(point.get("stale_symbols"))
            row[version+"_equity"] = number(point.get("equity")) if reliable else None
            if not reliable:
                result["warnings"].append(f"{version}: unreliable or stale observed mark withheld from the chart at {key}")
            cuts.append(stamp)
        for trade in records(run.get("trades")):
            view = trade_view(trade, lookup.get(trade.get("signal_id")), version, "prospective_paper")
            if trade.get("status") == "CLOSED":
                result["closed_trades"].append(view)
            elif trade.get("status") == "OPEN":
                result["positions"].append(view)
    # Join only exact observed times. No interpolation/forward-fill or synthetic
    # starting curve is inserted when the cohort has no observations.
    for version, benchmark in result["benchmarks"].items():
        if not isinstance(benchmark, dict):
            continue
        points = items(benchmark.get("equity"))
        if not points and benchmark.get("asof"):
            points = [{"time": benchmark["asof"], "equity": benchmark.get("equity")}]
        for point in records(points):
            stamp = time(point.get("time", point.get("timestamp")))
            if stamp is not None and number(point.get("equity")) is not None:
                key = stamp.isoformat()
                snapshots.setdefault(key, {"timestamp": key})[version+"_equity"] = number(point["equity"])
                cuts.append(stamp)
    # A dated portfolio mark is not proof of a fresh underlying market quote.
    # Only the forward publisher may provide its actual source-data cutoff.
    result["latest_equity_observation_at"] = max(cuts).isoformat() if cuts else None
    result["equity"] = [snapshots[k] for k in sorted(snapshots)]
    result["warnings"] = unique(result["warnings"])
    return result


def build_report(replay=None, forward=None, audit=None, legacy=None, config=None, metadata=None, candidates=None):
    replay, forward, audit, legacy, config = [deepcopy(v) if isinstance(v, dict) else {} for v in (replay, forward, audit, legacy, config)]
    candidates = deepcopy(mapping(candidates))
    warnings = []
    try:
        expected = canonical_hash(config) if config else None
    except (TypeError, ValueError):
        config, expected = {}, None
        warnings.append("Frozen configuration is malformed and cannot be hashed.")
    if not config:
        warnings.append("Frozen configuration unavailable; no current V4 configuration can be verified.")
    source_states = {name: {"status": value.get("status"), "pipeline_status": value.get("pipeline_status"),
                            "generated_at": value.get("generated_at", value.get("as_of")),
                            "market_data_at": value.get("market_data_at", value.get("data_cutoff")),
                            "config_hash": value.get("config_hash")}
                     for name, value in (("replay", replay), ("forward", forward), ("audit", audit), ("legacy_forward", legacy), ("candidates", candidates))}
    for name, value in (("replay", replay), ("forward", forward), ("candidates", candidates)):
        if value and (not expected or value.get("config_hash") != expected):
            warnings.append(f"{name}: configuration hash is missing or differs from the frozen configuration; cohort omitted from combined comparisons.")
            if name == "replay":
                replay = {}
            elif name == "forward":
                forward = {}
            else:
                candidates = {}
    if candidates and replay and (not candidates.get("manifest_sha256") or candidates.get("manifest_sha256") != replay.get("manifest_sha256")):
        warnings.append("Candidate explanation sample uses a different or missing archive hash; sample omitted.")
        candidates = {}
    if str(forward.get("status", "")).startswith("paused"):
        warnings.append("Prospective collection is paused: " + str(forward.get("status")))
    warnings.extend("Prospective cohort: " + str(v) for v in items(forward.get("warnings")))
    for name, value in (("replay", replay), ("forward", forward), ("audit", audit)):
        if value.get("pipeline_status") not in (None, "complete", "ok"):
            warnings.append(f"{name}: source pipeline status {value['pipeline_status']}")
    comparisons, trades, decisions, regimes = [], [], [], []
    for run in items(replay.get("comparisons")):
        if not isinstance(run, dict):
            continue
        version, window = run.get("strategy"), run.get("window")
        # Keep the comparison compact. Full frozen feature packets, execution
        # rejections and trade legs live in the linked immutable replay artifact.
        row = {k: deepcopy(run[k]) for k in ("strategy", "window", "metrics", "equity", "config", "assumptions") if k in run}
        row.update(source="corrected_equal_execution_replay", metrics=metrics_aliases(run.get("metrics"), run.get("confidence_interval")),
                   bootstrap=run.get("confidence_interval"), confidence_interval=interval(run.get("confidence_interval")))
        comparisons.append(row)
        signals = {s.get("signal_id"): s for s in items(run.get("signals")) if isinstance(s, dict)}
        local_trades = []
        for trade in records(run.get("trades")):
            view = trade_view(trade, signals.get(trade.get("signal_id")), version, "corrected_equal_execution_replay", window)
            trades.append(view)
            local_trades.append(view)
        for rejected in records(run.get("rejections")):
            decisions.append(trade_view(rejected, signals.get(rejected.get("signal_id")), version, "retrospective_execution_rejection", window))
        groups = {}
        for trade in local_trades:
            if trade.get("status") == "CLOSED":
                groups.setdefault(trade.get("market_regime") or "Unknown", []).append(trade)
        for regime, group in sorted(groups.items()):
            rs = [number(t.get("net_r")) for t in group]
            rs = [v for v in rs if v is not None]
            pnls = [number(t.get("net_pnl")) for t in group]
            pnls = [v for v in pnls if v is not None]
            regimes.append({"strategy": version, "window": window, "regime": regime, "source": "corrected_equal_execution_replay",
                            "metrics": {"closed_trades": len(group), "net_expectancy_r": sum(rs)/len(rs) if rs else None,
                                        "win_rate": sum(v > 0 for v in pnls)/len(pnls) if pnls else None},
                            "interpretation": "Retrospective association; correlated trades and multiple comparisons; no causal filter claim."})
    for row in items(replay.get("benchmarks")):
        if isinstance(row, dict):
            comparisons.append({**deepcopy(row), "source": "retrospective_benchmark_price_return", "metrics": metrics_aliases(row.get("metrics"))})
    # Exact V4 decisions are exposed only if their archived context exists.
    for signal in items(replay.get("decisions")):
        if isinstance(signal, dict):
            decisions.append(decision_view(signal, signal.get("strategy", "V4"), "retrospective_candidate_decision", signal.get("window")))
    for signal in records(candidates.get("decisions")):
        decisions.append(decision_view(signal, signal.get("strategy", "V4"), "retrospective_sample", signal.get("window")))
    for version in ("V1", "V2", "V3"):
        source = mapping(legacy.get("strategies")).get(version)
        if isinstance(source, dict):
            counts = source.get("closed_trades")
            metrics = {k: source.get(k) for k in ("equity", "return", "closed_trades", "win_rate", "profit_factor", "realized_pnl")}
            metrics.update(signals=None, executed_trades=number(counts)+number(source.get("open_positions", 0)) if number(counts) is not None and number(source.get("open_positions", 0)) is not None else None,
                           net_expectancy_r=None, confidence_interval=None,
                           max_drawdown=abs(number(source["max_drawdown"])) if number(source.get("max_drawdown")) is not None else None)
            comparisons.append({"strategy": version, "source": "legacy_forward", "window": "original prospective observations under legacy execution",
                                "generated_at": legacy.get("generated_at"), "metrics": metrics,
                                "assumptions": "Original sampled-price execution, original allocation and historical costs. Not the same protocol as the new V1–V4 cohort; original per-trade R unavailable."})
        record = mapping(mapping(audit.get("strategies")).get(version))
        for group in records(mapping(mapping(record.get("strata")).get("paper")).get("market_regime")):
            regimes.append({"strategy": version, "regime": group.get("bucket"), "source": "legacy_forward",
                            "window": "original forward paper history", "metrics": deepcopy(group)})
    experiments = []
    for experiment in records(replay.get("experiments")):
        row = deepcopy(experiment)
        row["metrics"] = metrics_aliases(row.get("metrics"), row.get("confidence_interval"))
        row["bootstrap"] = row.get("confidence_interval")
        row["confidence_interval"] = interval(row.get("confidence_interval"))
        row.setdefault("status", "retrospective exploratory; not a validated improvement")
        row.setdefault("assumptions", row.get("config"))
        experiments.append(row)
    source_dates = [x.get("generated_at", x.get("as_of")) for x in (replay, forward, audit, legacy, candidates)]
    as_of = latest(source_dates)
    generated = as_of.isoformat() if as_of else None
    sources = deepcopy(records(metadata))
    for source in sources:
        stamp = time(source.get("generated_at"))
        source["age_at_report_days"] = (as_of-stamp).total_seconds()/86400 if as_of and stamp else None
        source["stale_at_report"] = source["age_at_report_days"] > 4 if source["age_at_report_days"] is not None else None
        quote = time(source.get("market_data_at"))
        source["market_data_age_at_report_days"] = (as_of-quote).total_seconds()/86400 if as_of and quote else None
        source["market_data_stale_at_report"] = source["market_data_age_at_report_days"] > 4 if source["market_data_age_at_report_days"] is not None else None
        if source["stale_at_report"] or source["market_data_stale_at_report"]:
            warnings.append(f"{source.get('name', 'source')}: dated source or market observations exceed four calendar days at report time; not live data.")
    missing = [name for name, value in (("replay", replay), ("forward", forward), ("audit", audit), ("legacy_forward", legacy), ("config", config)) if not value]
    validation = mapping(replay.get("validation"))
    limitations = unique(items(validation.get("limitations"))+items(candidates.get("limitations"))+warnings+[
        "No untouched historical test exists: all historical windows and the current universe have already been inspected. Prospective evidence begins only with actually frozen future decisions.",
        "Legacy forward, corrected historical replay and the new prospective cohort use separate data/execution protocols. Preserve source labels; do not interpret a cross-cohort ranking as causal superiority.",
        "Publication timestamps are copied from source artifacts. They are not market-data timestamps, live quotes or evidence of a fresh observation.",
        "Missing marks remain gaps. No initial equity curve, unobserved benchmark path, trade, probability, confidence interval or outcome is fabricated.",
        "Candidate explanations are a predefined retrospective date/symbol/cutoff sample, not exhaustive signals or a performance sample; they never enter comparison metrics.",
        "Configuration mismatches fail closed for combined V4 comparisons; missing source files are disclosed. Research-only status never implies validated positive expectancy."])
    return json_clean({"schema_version": 1, "status": "research_only_inconclusive", "generated_at": generated,
        "pipeline_status": "incomplete" if missing or warnings else "complete", "missing_inputs": missing,
        "experiment_id": config.get("experiment_id"), "config_hash": expected, "config": config,
        "sources": sources, "source_states": source_states, "market_data_at": replay.get("market_data_at"),
        "freshness": {"reference_at": generated, "age_basis": "latest input publication; market cutoffs retained separately", "stale_after_calendar_days": 4},
        "validation": {"status": "inconclusive; retrospective historical research and a separately tracked prospective cohort",
                       "untouched_test_status": "no untouched historical test; prospective only",
                       "chronological_windows": validation.get("chronological_windows", {}),
                       "purge_embargo_sessions": validation.get("purge_embargo_sessions", []),
                       "limitations": limitations},
        "comparisons": comparisons, "experiments": experiments,
        "retrospective": {"status": "already_observed_exploratory", "trades": trades, "regimes": regimes, "decisions": decisions,
                          "candidate_sample": {"status": candidates.get("status"), "count": len(records(candidates.get("decisions"))),
                                               "manifest_sha256": candidates.get("manifest_sha256"), "sampling": candidates.get("sampling"),
                                               "limitations": items(candidates.get("limitations"))},
                          "coverage": replay.get("coverage", {}), "manifest_path": replay.get("manifest_path"),
                          "manifest_sha256": replay.get("manifest_sha256"), "source_sha256": replay.get("source_sha256", {}),
                          "execution_config": replay.get("primary_execution_config", {}), "experiment_config_hash": replay.get("experiment_config_hash")},
        "prospective": prospective_view(forward), "limitations": ["No automatic promotion and no real-money order execution."]})


def publish(root=Path("."), output=None):
    root = Path(root)
    payloads, metadata, errors = {}, [], []
    for name, relative in SOURCES.items():
        path = root/relative
        info = {"name": name, "path": relative, "available": path.is_file(), "valid": False, "sha256": None, "generated_at": None}
        if path.is_file():
            try:
                raw = path.read_bytes()
                info["sha256"] = hashlib.sha256(raw).hexdigest()
                value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {token}")))
                if not isinstance(value, dict):
                    raise ValueError("expected a JSON object")
                payloads[name] = validate_payload(name, value)
                info["valid"] = True
                info["generated_at"] = value.get("generated_at", value.get("as_of"))
                info["market_data_at"] = value.get("market_data_at", value.get("data_cutoff"))
                info["status"] = value.get("status")
                info["pipeline_status"] = value.get("pipeline_status")
                info["config_hash"] = value.get("config_hash")
            except (ValueError, TypeError, OSError) as exc:
                info["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(f"{relative}: {info['error']}")
        elif name not in OPTIONAL_SOURCES:
            errors.append(f"{relative}: missing")
        metadata.append(info)
    result = build_report(payloads.get("replay"), payloads.get("forward"), payloads.get("audit"),
                          payloads.get("legacy_forward"), payloads.get("config"), metadata, payloads.get("candidates"))
    result["input_errors"] = errors
    if errors:
        result["pipeline_status"] = "incomplete"
        result["validation"]["limitations"].extend(errors)
    destination = Path(output) if output else root/"docs/data/v4-research.json"
    forbidden = {(root/p).resolve() for p in SOURCES.values()}
    if destination.resolve() in forbidden or (root/"data").resolve() in destination.resolve().parents:
        raise ValueError("Publisher output must not overwrite source data or configuration")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":"))+"\n")
    temporary.replace(destination)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = publish(args.root, args.output)
    print(json.dumps({"status": result["status"], "pipeline_status": result["pipeline_status"],
                      "comparisons": len(result["comparisons"]), "prospective_signals": len(result["prospective"]["signals"])}))

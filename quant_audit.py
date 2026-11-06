"""Read-only, deterministic diagnostics for the published research ledgers.

Run ``python quant_audit.py`` to publish a *new* audit artifact. No market-data
requests, historical strategy reruns, ledger repairs or parameter fitting occur.
The output contains every source trade, including unresolved replay signals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
SEED = 20261105
BOOTSTRAPS = 2000
VERSIONS = ("V1", "V2", "V3")
INPUTS = ("data/paper_trades.csv", "data/backtest_trades.csv", "data/predictions.csv",
          "data/paper_summary.json", "data/backtest_summary.json")
REQUIRED = {
    INPUTS[0]: {"strategy", "symbol", "entry_time", "exit_time", "entry_price", "exit_price", "qty", "pnl", "pnl_pct", "signal_score", "market_regime"},
    INPUTS[1]: {"strategy", "symbol", "signal_date", "signal_time", "score", "entry", "realized_r_1r", "trade_result_1r"},
    INPUTS[2]: {"symbol", "signal_date", "generated_at", "frozen_at"},
}
FROZEN_FEATURES = ("vwap_dist", "volume_ratio", "volume_ratio_tod", "gap", "day_change",
                   "atr_pct", "return_20d", "minutes_from_open", "momentum_15m")


def number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def clean(value):
    """Produce strict JSON; unavailable values are null, never zero or NaN."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return number(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def timestamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def frozen_value(row, strategy, field):
    """Never use the generic V1 score/levels to explain a V2 or V3 trade."""
    value = row.get("prediction_" + field)
    if pd.notna(value) and str(value).strip():
        return value
    prefix = "" if strategy == "V1" else strategy.lower() + "_"
    value = row.get(prefix + field)
    return value if pd.notna(value) and str(value).strip() else None


def join_frozen(paper, predictions):
    """Join only unique same-NY-date strategy records already frozen at entry.

    The old closed ledger contains no prediction ID or original stop/target.
    Even a unique match is contextual evidence, not proof of execution geometry.
    Never attach a later prediction or cross-strategy record to fill a gap.
    """
    output = paper.copy()
    if output.empty:
        return output
    frozen = predictions.copy()
    if not frozen.empty:
        frozen["_version"] = frozen.get("strategy_version", pd.Series("V1", index=frozen.index)).fillna("V1").replace("", "V1")
        frozen["_generated"] = pd.to_datetime(frozen.get("generated_at"), utc=True, errors="coerce")
        frozen["_frozen"] = pd.to_datetime(frozen.get("frozen_at"), utc=True, errors="coerce")
    records = []
    for _, trade in output.iterrows():
        record = trade.to_dict()
        entry = timestamp(trade.get("entry_time"))
        local = entry.tz_convert(NY) if pd.notna(entry) else pd.NaT
        day = local.date().isoformat() if pd.notna(local) else None
        record.update(signal_date=day, join_status="missing", prediction_id=None,
                      original_geometry_confirmed=False, entry_hour=str(local.hour) if pd.notna(local) else "Unknown")
        if pd.notna(local):
            minute = local.hour * 60 + local.minute
            record["outside_regular_hours"] = bool(local.weekday() >= 5 or not 570 <= minute < 960)
        else:
            record["outside_regular_hours"] = None
        exit_time = timestamp(trade.get("exit_time"))
        record["holding_hours"] = (exit_time-entry).total_seconds()/3600 if pd.notna(entry) and pd.notna(exit_time) else None
        for field in FROZEN_FEATURES:
            record[field] = None
        if not frozen.empty and day:
            candidates = frozen[(frozen["_version"] == trade.get("strategy")) &
                                (frozen["symbol"] == trade.get("symbol")) &
                                (frozen["signal_date"] == day)]
            eligible = candidates[(candidates["_generated"] <= entry) & (candidates["_frozen"] <= entry)]
            if len(eligible) == 1:
                source = eligible.iloc[0]
                record.update(join_status="unique_prior_frozen", prediction_id=source.get("prediction_id"),
                              frozen_at=source.get("frozen_at"), signal_generated_at=source.get("generated_at"),
                              signal_lag_minutes=(entry-source["_generated"]).total_seconds()/60,
                              frozen_score=number(frozen_value(source, str(trade["strategy"]), "score")),
                              frozen_stop=number(frozen_value(source, str(trade["strategy"]), "stop")),
                              frozen_target_1r=number(frozen_value(source, str(trade["strategy"]), "target_1r")),
                              original_reason=frozen_value(source, str(trade["strategy"]), "reason"),
                              frozen_signal_price=number(source.get("price")))
                for field in FROZEN_FEATURES:
                    record[field] = number(source.get(field))
                # This is an agreement diagnostic, not an asserted execution join.
                score, stored_score = number(trade.get("signal_score")), record["frozen_score"]
                record["score_matches_frozen"] = score == stored_score if score is not None and stored_score is not None else None
                entry_price, signal_price = number(trade.get("entry_price")), record["frozen_signal_price"]
                record["entry_vs_frozen_price_return"] = entry_price/signal_price-1 if entry_price and signal_price else None
            elif len(eligible) > 1:
                record["join_status"] = "ambiguous_multiple_frozen_records"
            elif not candidates.empty:
                record["join_status"] = "not_frozen_before_entry"
        records.append(record)
    return pd.DataFrame(records)


def block_interval(values, dates, seed=SEED, repetitions=BOOTSTRAPS):
    """Percentile CI of a row-weighted mean, resampling whole entry-date blocks.

    Same-day cross-sectional dependence is preserved. Adjacent date dependence
    from overlapping holdings remains unmodelled; this is descriptive uncertainty.
    """
    frame = pd.DataFrame({"value": pd.to_numeric(pd.Series(list(values)), errors="coerce"),
                          "date": pd.Series(list(dates))}).dropna()
    grouped = frame.groupby("date", sort=True)["value"].agg(["sum", "count"])
    if len(grouped) < 2:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(repetitions, len(grouped)))
    samples = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped["count"].to_numpy()[indices].sum(axis=1)
    return np.quantile(samples, [.025, .975]).tolist()


def metrics(frame, kind):
    if frame.empty:
        return {"count": 0, "win_rate": None, "win_rate_ci95": None}
    value_column = "pnl" if kind == "paper" else "realized_r_1r"
    values = pd.to_numeric(frame[value_column], errors="coerce")
    eligible = values.notna()
    values = values[eligible]
    dates = frame.loc[eligible, "signal_date"]
    count = len(values)
    wins, losses = values[values > 0], values[values < 0]
    result = {"count": count, "entry_date_clusters": int(dates.nunique()),
              "wins": len(wins), "losses": len(losses),
              "win_rate": float((values > 0).mean()) if count else None,
              "win_rate_ci95": block_interval((values > 0).astype(float), dates),
              "profit_factor": float(wins.sum()/-losses.sum()) if len(losses) and losses.sum() else None}
    if kind == "paper":
        avg_win = float(wins.mean()) if len(wins) else None
        avg_loss = float(-losses.mean()) if len(losses) else None
        result.update(total_pnl=float(values.sum()), mean_pnl=number(values.mean()),
                      median_pnl=number(values.median()), pnl_mean_ci95=block_interval(values, dates),
                      mean_win=avg_win, mean_loss=avg_loss,
                      payoff_ratio=avg_win/avg_loss if avg_win is not None and avg_loss else None,
                      break_even_win_rate=avg_loss/(avg_win+avg_loss) if avg_win is not None and avg_loss else None,
                      mean_return=number(pd.to_numeric(frame.get("pnl_pct"), errors="coerce").mean()),
                      median_return=number(pd.to_numeric(frame.get("pnl_pct"), errors="coerce").median()),
                      mean_holding_hours=number(pd.to_numeric(frame.get("holding_hours"), errors="coerce").mean()),
                      largest_loss=number(values.min()),
                      entry_outside_regular_hours=int(frame["outside_regular_hours"].fillna(False).sum()),
                      unique_prior_frozen_matches=int((frame["join_status"] == "unique_prior_frozen").sum()))
        capital = pd.to_numeric(frame["entry_price"], errors="coerce") * pd.to_numeric(frame["qty"], errors="coerce")
        exit_notional = pd.to_numeric(frame["exit_price"], errors="coerce") * pd.to_numeric(frame["qty"], errors="coerce")
        result.update(average_entry_notional=number(capital.mean()), gross_traded_notional=number((capital+exit_notional).sum()))
        # Sensitivity only: historical per-trade cost settings were not archived.
        raw_entry = pd.to_numeric(frame["entry_price"], errors="coerce") / 1.0007
        raw_exit = pd.to_numeric(frame["exit_price"], errors="coerce") / .9993
        no_slippage = (raw_exit-raw_entry) * pd.to_numeric(frame["qty"], errors="coerce")
        result["pnl_without_assumed_7bps_each_side"] = number(no_slippage.sum())
        result["assumed_slippage_drag"] = number(no_slippage.sum()-values.sum())
        result["slippage_sensitivity_note"] = "Algebraic removal of the default 7 bps each side, not a rerun or verified original per-trade cost configuration."
    else:
        result.update(expectancy_r=number(values.mean()), median_r=number(values.median()),
                      expectancy_r_ci95=block_interval(values, dates))
    return result


def score_bucket(value):
    value = number(value)
    if value is None:
        return "Unknown"
    if value < 60:
        return "<60"
    if value >= 90:
        return "90+"
    lower = int(value//10)*10
    return f"{lower}–{lower+9}"


def feature_bucket(value, cutpoints, labels):
    value = number(value)
    if value is None:
        return "Unknown"
    return labels[int(np.searchsorted(cutpoints, value, side="right"))]


def strata(frame, field, kind, transform=None):
    if frame.empty:
        return []
    series = frame[field] if field in frame else pd.Series("Unknown", index=frame.index)
    groups = series.map(transform) if transform else series.fillna("Unknown").replace("", "Unknown").astype(str)
    result = []
    for bucket in sorted(groups.unique()):
        subset = frame[groups == bucket]
        result.append({"bucket": bucket, **metrics(subset, kind)})
    return result


def dataset_stats(frame, fields):
    return {field: {"known": int(frame[field].notna().sum()) if field in frame else 0,
                    "missing": int(frame[field].isna().sum()) if field in frame else len(frame)}
            for field in fields}


def build_report(paper, backtest, predictions, paper_summary=None, metadata=None):
    paper_summary = paper_summary or {}
    joined = join_frozen(paper, predictions)
    replay = backtest.copy()
    if not replay.empty:
        times = pd.to_datetime(replay["signal_time"], utc=True, errors="coerce").dt.tz_convert(NY)
        replay["entry_hour"] = times.dt.hour.map(lambda x: str(int(x)) if pd.notna(x) else "Unknown")
        # The saved name differs from the scanner ledger; retain the original too.
        replay["day_change"] = replay.get("day_change_at_signal")
        if {"stop", "target_1r"}.issubset(replay):
            risk = pd.to_numeric(replay.entry, errors="coerce")-pd.to_numeric(replay.stop, errors="coerce")
            reward = pd.to_numeric(replay.target_1r, errors="coerce")-pd.to_numeric(replay.entry, errors="coerce")
            replay["actual_target_reward_to_risk"] = (reward/risk).where(risk > 0)
    strategies = {}
    moderate_volume = lambda x: feature_bucket(x, [1.4, 2.5, 5], ["<1.4", "1.4–2.5", "2.5–5", "5+"])
    distance = lambda x: feature_bucket(x, [0, .005, .015, .03], ["<0%", "0–0.5%", "0.5–1.5%", "1.5–3%", "3%+"])
    for version in VERSIONS:
        p = joined[joined.strategy == version] if not joined.empty else joined
        all_b = replay[replay.strategy == version] if not replay.empty else replay
        b = all_b[all_b.trade_result_1r.isin(["WIN_1R", "STOP"])] if not all_b.empty else all_b
        ps, bs = metrics(p, "paper"), metrics(b, "backtest")
        bs.update(signals=len(all_b), ambiguous=int((all_b.trade_result_1r == "AMBIGUOUS").sum()) if not all_b.empty else 0,
                  pending=int((~all_b.trade_result_1r.isin(["WIN_1R", "STOP", "AMBIGUOUS"])).sum()) if not all_b.empty else 0)
        if "actual_target_reward_to_risk" in all_b:
            bs["nominal_1r_target_exceeds_actual_risk"] = int((all_b.actual_target_reward_to_risk > 1.00001).sum())
            bs["max_target_reward_to_risk"] = number(all_b.actual_target_reward_to_risk.max())
        official = paper_summary.get("strategies", {}).get(version, {})
        ps["official_realized_pnl"] = official.get("realized_pnl")
        ps["ledger_rounding_difference"] = ps.get("total_pnl", 0)-official["realized_pnl"] if "realized_pnl" in official else None
        strategies[version] = {"paper": ps, "backtest": bs,
            "score_buckets": {"paper": strata(p, "signal_score", "paper", score_bucket),
                              "backtest": strata(b, "score", "backtest", score_bucket)},
            "strata": {"paper": {"market_regime": strata(p, "market_regime", "paper"),
                                   "entry_hour": strata(p, "entry_hour", "paper"),
                                   "volume_ratio_tod": strata(p, "volume_ratio_tod", "paper", moderate_volume),
                                   "vwap_dist": strata(p, "vwap_dist", "paper", distance),
                                   "holding_hours": strata(p, "holding_hours", "paper", lambda x: feature_bucket(x, [24, 72, 168], ["<24h", "24–72h", "72–168h", "168h+"])),
                                   "gap": strata(p, "gap", "paper", lambda x: feature_bucket(x, [-.02, 0, .02, .05], ["<-2%", "-2–0%", "0–2%", "2–5%", "5%+"])),
                                   "atr_pct": strata(p, "atr_pct", "paper", lambda x: feature_bucket(x, [.02, .04, .065], ["<2%", "2–4%", "4–6.5%", "6.5%+"])),
                                   "day_change": strata(p, "day_change", "paper", lambda x: feature_bucket(x, [0, .02, .05], ["<0%", "0–2%", "2–5%", "5%+"]))},
                       "backtest": {"entry_hour": strata(b, "entry_hour", "backtest"),
                                    "volume_ratio_tod": strata(b, "volume_ratio_tod", "backtest", moderate_volume),
                                    "vwap_dist": strata(b, "vwap_dist", "backtest", distance),
                                    "day_change": strata(b, "day_change_at_signal", "backtest", lambda x: feature_bucket(x, [0, .02, .05], ["<0%", "0–2%", "2–5%", "5%+"]))}},
            "largest_losses": p.nsmallest(5, "pnl").to_dict("records") if not p.empty else []}
    overlap, shared_forward = [], []
    if not joined.empty and not replay.empty:
        for version in VERSIONS:
            # Matching dates do NOT imply matching entries: this is coverage only.
            pairs = joined[joined.strategy == version].merge(replay[replay.strategy == version],
                        on=["strategy", "symbol", "signal_date"], suffixes=("_paper", "_backtest"))
            resolved = pairs[pairs.trade_result_1r.isin(["WIN_1R", "STOP"])]
            overlap.append({"strategy": version, "same_symbol_date_pairs": len(pairs),
                            "resolved_pairs": len(resolved),
                            "opposite_win_loss": int(((resolved.pnl > 0) != (resolved.realized_r_1r > 0)).sum()),
                            "entry_prices_differ": int((abs(resolved.entry_price/resolved.entry-1) > .001).sum()),
                            "note": "Coverage pairing only: times, prices, selection, fills and risk budgets differ."})
    if not joined.empty:
        # Clustered, simultaneous cross-strategy trades do not represent independent
        # confirmations. Pair by symbol/minute only, then expose exact fill agreement.
        forward = joined.copy()
        forward["_key"] = pd.to_datetime(forward.entry_time, utc=True, errors="coerce").dt.floor("min").astype(str)+"|"+forward.symbol
        first = forward[forward.strategy == "V1"]
        for version in ("V2", "V3"):
            other = forward[forward.strategy == version]
            if first["_key"].duplicated().any() or other["_key"].duplicated().any():
                shared_forward.append({"strategy": version, "status": "ambiguous_duplicate_keys"})
                continue
            pairs = first.merge(other, on="_key", suffixes=("_v1", "_other"))
            shared_forward.append({"strategy": version, "same_symbol_minute": len(pairs),
                "identical_recorded_entry_exit_prices": int(((pairs.entry_price_v1 == pairs.entry_price_other) & (pairs.exit_price_v1 == pairs.exit_price_other)).sum()),
                "v1_shared_pnl": number(pairs.pnl_v1.sum()), "other_shared_pnl": number(pairs.pnl_other.sum()),
                "v1_unshared_pnl": number(first.loc[~first._key.isin(other._key), "pnl"].sum()),
                "other_unshared_pnl": number(other.loc[~other._key.isin(first._key), "pnl"].sum()),
                "note": "Descriptive decomposition, not random assignment or an independent paired test."})
    sources = metadata or []
    as_of = max((s["generated_at"] for s in sources if s.get("generated_at")), default=None)
    report = {"schema_version": 1, "as_of": as_of, "status": "research_only", "inputs": sources,
        "methodology": {"bootstrap": "entry-date block percentile bootstrap of row-weighted means",
                        "bootstrap_seed": SEED, "bootstrap_repetitions": BOOTSTRAPS,
                        "confidence_level": .95, "probability_calibration": "not_available",
                        "brier_score": None, "execution_pnl": "paper ledger only; replay is gross independent 1R signal outcome",
                        "frozen_join": "unique strategy/symbol/NY date, generated_at and frozen_at <= paper entry; no future/current feature attachment"},
        "strategies": strategies, "comparisons": {"historical_forward_coverage": overlap, "v1_shared_forward": shared_forward},
        "coverage": {"paper_closed_trades": len(paper), "backtest_signals": len(backtest),
                     "backtest_resolved_trades": int(backtest.trade_result_1r.isin(["WIN_1R", "STOP"]).sum()) if not backtest.empty else 0,
                     "predictions": len(predictions),
                     "frozen_join_status": joined.join_status.value_counts().to_dict() if not joined.empty else {},
                     "frozen_score_disagrees_with_paper": int((joined.score_matches_frozen == False).sum()) if "score_matches_frozen" in joined else 0,
                     "maximum_frozen_context_lag_minutes": number(joined.signal_lag_minutes.max()) if "signal_lag_minutes" in joined else None,
                     "paper_features": dataset_stats(joined, FROZEN_FEATURES),
                     "backtest_features": dataset_stats(replay, (*FROZEN_FEATURES, "market_regime", "rsi", "macd", "relative_strength", "earnings_date", "options_skew", "sector")),
                     "paper_missing_point_in_time_features": ["rsi", "macd", "ma200", "relative_strength_to_SPY_QQQ", "earnings_date", "options_skew", "sector", "bid_ask", "exit_candle_path", "original_stop_target_in_closed_ledger", "market_data_cutoff"],
                     "paper_signal_dates": [joined.signal_date.min(), joined.signal_date.max()] if not joined.empty else None,
                     "backtest_signal_dates": [replay.signal_date.min(), replay.signal_date.max()] if not replay.empty else None,
                     "backtest_unique_symbol_dates": int(len(replay.groupby(["symbol", "signal_date"]))) if not replay.empty else 0},
        "findings": [
            {"id": "execution_protocol_mismatch", "evidence": f"{int(joined.outside_regular_hours.fillna(False).sum()) if not joined.empty else 0}/{len(joined)} recorded paper entries are outside the weekday 09:30–16:00 New York clock window.",
             "interpretation": "The old simulator buys sampled scanner prices on workflow execution, not verified contemporaneous tradable fills. Replay uses intraday checkpoint closes. Attribution to strategy rules alone is invalid."},
            {"id": "loss_asymmetry", "evidence": "Inspect mean_win, mean_loss, payoff_ratio and break_even_win_rate for each paper strategy.",
             "interpretation": "A win rate above 50% can still lose money: periodic sampled exits can overshoot stops, while replay caps outcomes at exactly +1R/-1R. Fixed-dollar sizing is not equal-risk sizing."},
            {"id": "scores_are_not_probabilities", "evidence": "Score bucket outcomes and date-block confidence intervals are published without probability labels or Brier scores.",
             "interpretation": "Higher research scores do not establish calibrated win probabilities. Buckets are retrospective associations with correlated observations and multiple comparisons."},
            {"id": "unmatched_experiments", "evidence": "Historical replay samples all first checkpoint signals in the current universe; forward portfolios cap positions at five and $1,000 each with integer shares and score-prioritised candidates.",
             "interpretation": "Different periods, candidate selection, capital utilisation, exposure, slippage and execution paths explain why historical rankings need not match forward rankings. V1 is an essential prospective comparator."}],
        "limitations": [
            "Historical and paper data have already been inspected in prior research; no interval in this audit is an untouched test set.",
            "Entry-date bootstrap preserves same-day dependence but not all serial dependence from overlapping holdings; intervals are descriptive, not a profitability test.",
            "All bucket comparisons are exploratory; small samples and multiple testing prohibit causal claims or filter selection as confirmed improvement.",
            "Frozen scanner features were recorded before matched entries, but the original market-data availability cutoff was not logged; point-in-time completeness cannot be certified.",
            "A unique prediction join gives frozen context. The closed paper ledger lacks original prediction ID/stop/target, so original risk, exact stop overshoot and gap-versus-delayed-exit attribution are unavailable.",
            "Replay assumes checkpoint close entry, exact stop/target prices, no costs, no capital limit and excludes ambiguous bars. Intraday timestamps denote bar starts, exposing completion-time ambiguity in legacy code.",
            "Forward paper entries/exits use sampled latest prices and 7 bps adverse slippage each side; intervening stop/target order, overnight exit gaps and market-data age were not recorded.",
            "Current universe and rolling Yahoo 5m coverage are not a survivor-free historical market archive. Corporate actions and adjusted benchmark versus raw signal prices need a consistent future protocol.",
            "Future-close horizon returns and legacy prediction hit labels are not executable P/L and are excluded from these win-rate comparisons.",
            "No RSI/MACD/options/earnings/sector causal analysis, fair equal-capital Sharpe/Sortino or benchmark attribution can be recovered from these ledgers alone. Missing values remain unknown.",
            "Summing rounded closed-trade P/L may differ by cents from full-precision state/summary totals. This report preserves both without rewriting either.",
            "Prospective execution corrections define a new protocol; they cannot repair or retroactively validate the legacy forward record."],
        "trades": {"paper": joined.to_dict("records"), "backtest": replay.to_dict("records")}}
    return clean(report)


def publish(root=Path("."), output=None):
    root = Path(root)
    frames, payloads, metadata, errors = {}, {}, [], []
    for relative in INPUTS:
        path = root/relative
        item = {"path": relative, "available": path.exists(), "sha256": None, "rows": 0}
        if path.exists():
            raw = path.read_bytes()
            item["sha256"] = hashlib.sha256(raw).hexdigest()
            try:
                if path.suffix == ".csv":
                    frame = pd.read_csv(path)
                    item["rows"] = len(frame)
                    absent = REQUIRED[relative]-set(frame.columns)
                    if absent:
                        errors.append(f"{relative}: missing required columns {', '.join(sorted(absent))}")
                    else:
                        frames[relative] = frame
                else:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        errors.append(f"{relative}: expected an object")
                    else:
                        payloads[relative] = payload
                        item["generated_at"] = payload.get("generated_at")
            except (ValueError, pd.errors.ParserError) as exc:
                errors.append(f"{relative}: {type(exc).__name__}")
        else:
            errors.append(f"{relative}: missing")
        metadata.append(item)
    report = build_report(frames.get(INPUTS[0], pd.DataFrame()), frames.get(INPUTS[1], pd.DataFrame()),
                          frames.get(INPUTS[2], pd.DataFrame()), payloads.get(INPUTS[3]), metadata)
    report["pipeline_status"] = "incomplete" if errors else "complete"
    report["input_errors"] = errors
    destination = Path(output) if output else root/"docs/data/quant-audit.json"
    # Refuse any accidental output to one of the immutable input paths.
    if destination.resolve() in {(root/p).resolve() for p in INPUTS}:
        raise ValueError("Audit output must not overwrite a source ledger")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, allow_nan=False, sort_keys=True, separators=(",", ":"))+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = publish(args.root, args.output)
    print(json.dumps({"status": result["pipeline_status"], "coverage": result["coverage"]}, indent=2))

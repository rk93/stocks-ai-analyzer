"""Isolated, preregistered V4 experiment; no imported V1/V2/V3 scoring logic.

All gates are hypotheses, not established sources of alpha. The score describes
coverage of required evidence and is never a probability. A caller must supply
point-in-time features and their availability timestamps. This module never
fetches data or fills missing historical metadata with today's values.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo


CONFIG_PATH = Path(__file__).resolve().parent / "research" / "v4_config.json"
# Changing the registered rules requires an explicitly versioned new experiment.
REGISTERED_CONFIG_HASH = "d36d64a7cc8a538a17170c30a3b60f666ed3ecd72f7ee47e99a5b5304ce0fe6c"


def config_hash(config: Mapping[str, Any]) -> str:
    """Hash every rule and assumption, including simulation/risk constraints."""
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _registered_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config_hash(config) != REGISTERED_CONFIG_HASH:
        raise ValueError("Registered V4 configuration changed; create a new versioned experiment and explicitly register its hash")
    return config


def load_config() -> dict:
    """Return an independent copy so caller mutations cannot alter later signals."""
    return deepcopy(_registered_config())


STRATEGY_VERSION = _registered_config()["strategy_version"]
CONFIG_HASH = config_hash(_registered_config())


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _time(value: Any) -> datetime | None:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def classify_regime_v4(market: Mapping[str, Any] | None, config: Mapping[str, Any] | None = None) -> str:
    """Fail closed if *any* necessary regime evidence is unavailable."""
    rules = (config or _registered_config())["market"]
    metrics = {name: _number((market or {}).get(name)) for name in ("price", "ma20", "ma50", "ma200", "atr_pct", "return_20d")}
    if any(value is None for value in metrics.values()):
        return "UNKNOWN"
    price, ma20, ma50, ma200, atr, ret20 = (metrics[k] for k in ("price", "ma20", "ma50", "ma200", "atr_pct", "return_20d"))
    if min(price, ma20, ma50, ma200) <= 0 or atr <= 0:
        return "UNKNOWN"
    if atr >= rules["maximum_atr_pct"]:
        return "HIGH_VOL"
    if price > ma20 > ma50 > ma200 and ret20 > 0:
        return "BULL_TREND"
    if price < ma200 and ret20 < -0.04:
        return "RISK_OFF"
    return "OTHER"


def evaluate_v4(
    daily: Mapping[str, Any],
    intra: Mapping[str, Any],
    market: Mapping[str, Any] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict:
    """Explain acceptance/rejection with frozen config identity and finite levels.

    ``context`` requires timezone-aware ``signal_at``, ``daily_asof``,
    ``market_asof`` and ``intraday_asof``. Intraday as-of is the availability
    time of the last *completed* bar, not its opening label. Daily features must
    come from an earlier New York session. Optional earnings is a mapping with
    ``announced_at`` and ``event_at``; optional spread uses ``spread_bps`` and
    ``spread_observed_at``. Missing optional data is disclosed, never fabricated.

    Portfolio-level cash/exposure/correlation budgets belong to the execution
    allocator; this function authorizes only a research candidate, not a fill.
    """
    cfg = deepcopy(dict(config)) if config is not None else load_config()
    identity = config_hash(cfg)
    if identity != REGISTERED_CONFIG_HASH and (
        cfg.get("strategy_version") == STRATEGY_VERSION
        or cfg.get("experiment_id") == _registered_config()["experiment_id"]
    ):
        raise ValueError("Different V4 rules require distinct strategy_version and experiment_id; the registered experiment is immutable")
    ctx = dict(context or {})
    mkt = market or {}
    checks: list[dict] = []
    unavailable: list[str] = []

    def check(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    names = {
        "daily": (daily, ("prev_close", "ma20", "ma50", "atr_pct", "return_20d")),
        "intraday": (intra, ("price", "gap", "day_change", "vwap_dist", "volume_ratio_tod", "minutes_from_open", "opening_range_high", "momentum_15m", "dollar_volume_15m", "volume_history_sessions")),
        "market": (mkt, ("price", "ma20", "ma50", "ma200", "atr_pct", "return_20d")),
    }
    data = {group: {key: _number(source.get(key)) for key in keys} for group, (source, keys) in names.items()}
    missing = [f"{group}.{key}" for group, values in data.items() for key, value in values.items() if value is None]
    check("required_features", not missing, "finite required features" if not missing else "missing/invalid: " + ", ".join(missing))
    d, i, m = data["daily"], data["intraday"], data["market"]
    signal_at = _time(ctx.get("signal_at"))
    available = {key: _time(ctx.get(key)) for key in ("daily_asof", "market_asof", "intraday_asof")}
    valid_times = signal_at is not None and all(value is not None for value in available.values())
    time_reason = "aware availability timestamps are required"
    if valid_times:
        ny = ZoneInfo(cfg["market"]["exchange_timezone"])
        signal_date = signal_at.astimezone(ny).date()
        daily_ok = all(
            stamp <= signal_at
            and stamp.astimezone(ny).date() < signal_date
            and (signal_date - stamp.astimezone(ny).date()).days <= cfg["data"]["maximum_daily_age_days"]
            for key, stamp in available.items() if key != "intraday_asof"
        )
        intra_age = (signal_at - available["intraday_asof"]).total_seconds() / 60
        valid_times = daily_ok and 0 <= intra_age <= cfg["data"]["maximum_intraday_age_minutes"]
        time_reason = "prior-session daily/market data and completed intraday availability within cutoff" if valid_times else "future, same-session daily, or stale feature availability"
    check("point_in_time", valid_times, time_reason)
    regime = classify_regime_v4(mkt, cfg)
    check("market_regime", regime == "BULL_TREND", f"market regime {regime}; requires ordered MA20 > MA50 > MA200, positive 20d return and ATR below cap")

    def finite(*values: float | None) -> bool:
        return all(value is not None for value in values)

    trend = finite(d["prev_close"], d["ma20"], d["ma50"], d["return_20d"]) and d["prev_close"] > d["ma20"] > d["ma50"] > 0 and d["return_20d"] > 0
    check("stock_trend", trend, "previous close > MA20 > MA50 and positive prior-session 20d return")
    relative_strength = d["return_20d"] - m["return_20d"] if finite(d["return_20d"], m["return_20d"]) else None
    check("relative_strength", relative_strength is not None and relative_strength > cfg["entry"]["minimum_relative_strength_20d"], f"20d return minus market: {relative_strength}; must exceed {cfg['entry']['minimum_relative_strength_20d']}")
    price = i["price"]
    breakout = finite(price, i["opening_range_high"]) and price > i["opening_range_high"] > 0 and intra.get("breakout_or") is True
    check("opening_range", breakout, "completed close above the frozen opening-range high")
    vdist = i["vwap_dist"]
    check("vwap_entry", vdist is not None and 0 <= vdist <= cfg["entry"]["maximum_vwap_distance"], f"VWAP distance {vdist}; allowed 0 to {cfg['entry']['maximum_vwap_distance']}")
    vr, history = i["volume_ratio_tod"], i["volume_history_sessions"]
    check("volume_confirmation", finite(vr, history) and vr >= cfg["entry"]["minimum_volume_ratio_tod"] and history >= cfg["data"]["minimum_volume_history_sessions"], f"same-clock RVOL {vr}, history sessions {history}; minimum {cfg['entry']['minimum_volume_ratio_tod']}x over {cfg['data']['minimum_volume_history_sessions']} sessions")
    minutes = i["minutes_from_open"]
    actual_minutes = None
    regular_weekday = False
    if signal_at is not None:
        local_signal = signal_at.astimezone(ZoneInfo(cfg["market"]["exchange_timezone"]))
        actual_minutes = local_signal.hour * 60 + local_signal.minute + local_signal.second / 60 - 570
        regular_weekday = local_signal.weekday() < 5
    # Holidays/early closes are the market-data adapter's responsibility. This
    # check still prevents callers inventing a session minute at night/weekends.
    session_ok = finite(minutes, actual_minutes) and regular_weekday and abs(minutes - actual_minutes) < 1 and cfg["entry"]["minimum_minutes_from_open"] <= minutes <= cfg["entry"]["maximum_minutes_from_open"]
    check("session_window", session_ok, f"minutes from regular-session open {minutes}, observed clock {actual_minutes}; require settled liquid session")
    gap, change, momentum = i["gap"], i["day_change"], i["momentum_15m"]
    entry_ok = finite(gap, change, momentum) and abs(gap) <= cfg["entry"]["maximum_absolute_gap"] and change <= cfg["entry"]["maximum_day_change"] and cfg["entry"]["minimum_momentum_15m"] <= momentum <= cfg["entry"]["maximum_momentum_15m"]
    check("entry_extension", entry_ok, f"gap {gap}, day change {change}, 15m momentum {momentum}; bounded gap and short-term extension")
    dollars = i["dollar_volume_15m"]
    check("observed_liquidity", dollars is not None and dollars >= cfg["entry"]["minimum_dollar_volume_15m"], f"last completed 15m USD volume {dollars}; minimum {cfg['entry']['minimum_dollar_volume_15m']}")
    atr = d["atr_pct"]
    risk_ok = finite(price, atr) and price > 0 and cfg["risk"]["minimum_atr_pct"] <= atr <= cfg["risk"]["maximum_atr_pct"]
    check("atr_risk", risk_ok, f"prior-session ATR/close {atr}; finite bounded volatility required")

    if "earnings" in ctx:
        earnings = ctx["earnings"] if isinstance(ctx["earnings"], Mapping) else {}
        known_at, event_at = _time(earnings.get("announced_at")), _time(earnings.get("event_at"))
        earnings_valid = signal_at is not None and known_at is not None and event_at is not None and known_at <= signal_at
        no_near_event = earnings_valid and not 0 <= (event_at - signal_at).total_seconds() / 3600 <= cfg["entry"]["earnings_blackout_hours_when_known"]
        check("known_earnings", no_near_event, "known-at-signal earnings outside next 24h" if no_near_event else "earnings event imminent or event availability invalid/future")
    else:
        unavailable.append("Point-in-time earnings calendar unavailable; no earnings-avoidance benefit claimed")
    if "spread_bps" in ctx or "spread_observed_at" in ctx:
        spread, observed = _number(ctx.get("spread_bps")), _time(ctx.get("spread_observed_at"))
        spread_ok = signal_at is not None and observed is not None and spread is not None and 0 <= (signal_at - observed).total_seconds() / 60 <= cfg["data"]["maximum_intraday_age_minutes"] and 0 <= spread <= cfg["entry"]["maximum_spread_bps_when_known"]
        check("known_spread", spread_ok, f"observed spread {spread}bps; fresh point-in-time quote required when supplied")
    else:
        unavailable.append("Point-in-time bid/ask spread unavailable; dollar-volume proxy and explicit simulation costs used")
    unavailable.extend(["Options and constituent overlap/correlation are not available in this candidate gate", "Evidence score is uncalibrated; no probability or profitability claim"])

    rejections = [f"{item['name']}: {item['evidence']}" for item in checks if not item["passed"]]
    reasons = [f"{item['name']}: {item['evidence']}" for item in checks if item["passed"]]
    accepted = not rejections
    # Required evidence is equally weighted to avoid fitting retrospective scores.
    score = round(100 * sum(item["passed"] for item in checks) / len(checks), 1)
    state = "ENTRY TRIGGERED" if accepted else ("WATCH" if score >= 50 else "NO TRADE")
    stop = target_1r = target_2r = risk_per_share = None
    if risk_ok:
        stop_pct = min(max(cfg["risk"]["stop_atr_multiple"] * atr, cfg["risk"]["minimum_stop_pct"]), cfg["risk"]["maximum_stop_pct"])
        risk_per_share = price * stop_pct
        stop = price - risk_per_share
        target_1r = price + risk_per_share
        target_2r = price + 2 * risk_per_share
    return {
        "strategy_version": cfg["strategy_version"], "experiment_id": cfg["experiment_id"],
        "config_hash": identity, "status": "research-only", "accepted": accepted,
        "state": state, "score": score, "score_kind": cfg["score_kind"],
        "regime": regime, "checks": checks, "reasons": reasons, "rejections": rejections,
        "unavailable": unavailable, "relative_strength_20d": relative_strength,
        "stop": stop, "target_1r": target_1r, "target_2r": target_2r,
        "risk_per_share": risk_per_share,
        "signal_at": signal_at.isoformat() if signal_at is not None else None,
    }


def score_v4(daily: Mapping[str, Any], intra: Mapping[str, Any], market: Mapping[str, Any] | None = None, *, context: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None) -> tuple:
    """V1-compatible six-field presentation contract; invalid levels are NaN."""
    result = evaluate_v4(daily, intra, market, context=context, config=config)
    explanation = "; ".join(result["rejections"] or result["reasons"])
    levels = [result[key] if result[key] is not None else math.nan for key in ("stop", "target_1r", "target_2r")]
    return result["state"], result["score"], explanation, *levels

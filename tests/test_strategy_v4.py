"""Contract and fail-closed checks, independent of historical return outcomes."""
from copy import deepcopy
import json
import math

import pytest

from strategy_v4 import (
    CONFIG_HASH,
    CONFIG_PATH,
    STRATEGY_VERSION,
    classify_regime_v4,
    config_hash,
    evaluate_v4,
    load_config,
    score_v4,
)


@pytest.fixture
def packet():
    return (
        {"prev_close": 100, "ma20": 98, "ma50": 95, "atr_pct": 0.03, "return_20d": 0.07},
        {"price": 101, "gap": 0.005, "day_change": 0.02, "vwap_dist": 0.005,
         "volume_ratio_tod": 1.8, "minutes_from_open": 150, "opening_range_high": 100.5,
         "breakout_or": True, "momentum_15m": 0.007, "dollar_volume_15m": 3000000,
         "volume_history_sessions": 8},
        {"price": 110, "ma20": 108, "ma50": 105, "ma200": 100, "atr_pct": 0.015, "return_20d": 0.04},
        {"signal_at": "2026-11-05T17:00:00+00:00", "daily_asof": "2026-11-04T21:00:00+00:00",
         "market_asof": "2026-11-04T21:00:00+00:00", "intraday_asof": "2026-11-05T17:00:00+00:00"},
    )


def evaluate(packet):
    daily, intra, market, context = packet
    return evaluate_v4(daily, intra, market, context=context)


def test_deterministic_immutable_config_and_explanations(packet):
    original = deepcopy(packet)
    first = evaluate(packet)
    assert first == evaluate(packet)
    assert packet == original
    assert first["accepted"]
    assert first["state"] == "ENTRY TRIGGERED"
    assert first["score"] == 100
    assert "not a probability" in first["score_kind"]
    assert first["strategy_version"] == STRATEGY_VERSION
    assert first["config_hash"] == CONFIG_HASH
    assert first["rejections"] == []
    assert first["unavailable"]
    assert all(check["passed"] for check in first["checks"])
    # Public explanations never mislabel score as calibrated confidence.
    assert "uncalibrated" in " ".join(first["unavailable"])


def test_config_copy_and_canonical_hash_are_stable():
    original = load_config()
    mutated = load_config()
    mutated["entry"]["minimum_volume_ratio_tod"] = 99
    assert load_config() == original
    assert CONFIG_HASH == config_hash(json.loads(CONFIG_PATH.read_text()))
    assert CONFIG_HASH == config_hash(dict(reversed(list(original.items()))))
    assert CONFIG_HASH != config_hash(mutated)


def test_scenario_cannot_impersonate_registered_rules(packet):
    daily, intra, market, context = packet
    changed = load_config()
    changed["entry"]["minimum_volume_ratio_tod"] = 99
    with pytest.raises(ValueError, match="immutable"):
        evaluate_v4(daily, intra, market, context=context, config=changed)
    changed["strategy_version"] = "V4.separate-scenario"
    changed["experiment_id"] = "separate-scenario-not-registered-v4"
    result = evaluate_v4(daily, intra, market, context=context, config=changed)
    assert not result["accepted"]
    assert result["config_hash"] != CONFIG_HASH


@pytest.mark.parametrize("group,key", [
    (0, "prev_close"), (0, "ma20"), (0, "ma50"), (0, "return_20d"), (0, "atr_pct"),
    (1, "price"), (1, "gap"), (1, "day_change"), (1, "vwap_dist"),
    (1, "volume_ratio_tod"), (1, "minutes_from_open"), (1, "opening_range_high"),
    (1, "momentum_15m"), (1, "dollar_volume_15m"), (1, "volume_history_sessions"),
    (2, "price"), (2, "ma20"), (2, "ma50"), (2, "ma200"), (2, "atr_pct"), (2, "return_20d"),
])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "bad", True])
def test_missing_or_invalid_core_features_never_authorize(packet, group, key, bad):
    packet[group][key] = bad
    result = evaluate(packet)
    assert not result["accepted"]
    assert any("required_features" in reason for reason in result["rejections"])
    # The JSON research contract uses null, never NaN/Infinity.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("market,expected", [
    ({"atr_pct": 0.03}, "HIGH_VOL"),
    ({"price": 90, "return_20d": -0.08}, "RISK_OFF"),
    ({"price": 106}, "OTHER"),
    ({"return_20d": None}, "UNKNOWN"),
    ({"atr_pct": -0.1}, "UNKNOWN"),
])
def test_regime_is_a_hard_gate(packet, market, expected):
    packet[2].update(market)
    assert classify_regime_v4(packet[2]) == expected
    assert not evaluate(packet)["accepted"]


@pytest.mark.parametrize("key,value", [
    ("vwap_dist", -0.001), ("vwap_dist", 0.01501),
    ("volume_ratio_tod", 1.399), ("volume_history_sessions", 4),
    ("gap", -0.051), ("gap", 0.051), ("day_change", 0.051),
    ("momentum_15m", -0.001), ("momentum_15m", 0.021),
    ("minutes_from_open", 119), ("minutes_from_open", 331),
    ("breakout_or", False), ("opening_range_high", 101),
    ("dollar_volume_15m", 999999),
])
def test_each_entry_quality_guard_is_binding(packet, key, value):
    packet[1][key] = value
    assert not evaluate(packet)["accepted"]


def test_does_not_reuse_retrospective_pr9_volume_cap(packet):
    packet[1]["volume_ratio_tod"] = 4
    assert evaluate(packet)["accepted"]


def test_trend_and_cross_sectional_strength_are_independent_gates(packet):
    packet[0]["return_20d"] = 0.03
    assert not evaluate(packet)["accepted"]
    packet[0]["return_20d"] = 0.07
    packet[0]["ma20"] = 94
    assert not evaluate(packet)["accepted"]


@pytest.mark.parametrize("key,value", [
    ("signal_at", "2026-11-05T17:00:00"),
    ("intraday_asof", "2026-11-05T17:05:00+00:00"),
    ("intraday_asof", "2026-11-05T16:45:00+00:00"),
    ("daily_asof", "2026-11-05T13:00:00+00:00"),
    ("market_asof", "2026-11-06T21:00:00+00:00"),
    ("daily_asof", "2026-10-20T20:00:00+00:00"),
    ("market_asof", None),
])
def test_future_stale_naive_and_same_session_daily_data_rejected(packet, key, value):
    packet[3][key] = value
    result = evaluate(packet)
    assert not result["accepted"]
    assert any("point_in_time" in reason for reason in result["rejections"])


def test_no_timestamp_provenance_cannot_trigger(packet):
    daily, intra, market, _ = packet
    assert not evaluate_v4(daily, intra, market)["accepted"]


@pytest.mark.parametrize("at", ["2026-11-05T23:00:00+00:00", "2026-11-07T17:00:00+00:00"])
def test_claimed_session_minutes_cannot_override_actual_clock(packet, at):
    packet[3]["signal_at"] = at
    packet[3]["intraday_asof"] = at
    assert not evaluate(packet)["accepted"]


@pytest.mark.parametrize("atr,expected_pct", [(0.005, 0.006), (0.03, 0.015), (0.06, 0.03)])
def test_stop_target_risk_geometry_is_exact(packet, atr, expected_pct):
    packet[0]["atr_pct"] = atr
    result = evaluate(packet)
    entry = packet[1]["price"]
    assert result["accepted"]
    assert result["risk_per_share"] == pytest.approx(entry * expected_pct)
    assert entry - result["stop"] == pytest.approx(result["target_1r"] - entry)
    assert result["target_2r"] - entry == pytest.approx(2 * (entry - result["stop"]))


@pytest.mark.parametrize("atr", [0, -0.03, 0.061])
def test_invalid_risk_has_no_tradeable_levels(packet, atr):
    packet[0]["atr_pct"] = atr
    result = evaluate(packet)
    assert not result["accepted"]
    assert result["stop"] is None
    assert result["target_1r"] is None


def test_known_earnings_blackout_and_future_feature_leakage(packet):
    packet[3]["earnings"] = {"announced_at": "2026-11-01T12:00:00+00:00", "event_at": "2026-11-05T21:00:00+00:00"}
    assert not evaluate(packet)["accepted"]
    packet[3]["earnings"]["event_at"] = "2026-11-10T21:00:00+00:00"
    assert evaluate(packet)["accepted"]
    packet[3]["earnings"]["announced_at"] = "2026-11-06T12:00:00+00:00"
    assert not evaluate(packet)["accepted"]


@pytest.mark.parametrize("spread,at", [(30, "2026-11-05T17:00:00+00:00"), (10, "2026-11-05T17:01:00+00:00"), (10, "2026-11-05T16:40:00+00:00"), (float("nan"), "2026-11-05T17:00:00+00:00")])
def test_optional_spread_cannot_bypass_availability_or_liquidity(packet, spread, at):
    packet[3].update(spread_bps=spread, spread_observed_at=at)
    assert not evaluate(packet)["accepted"]


def test_presentation_contract_and_missing_price(packet):
    daily, intra, market, context = packet
    state, score, reason, stop, target1, target2 = score_v4(daily, intra, market, context=context)
    assert state == "ENTRY TRIGGERED" and score == 100 and reason
    assert stop < intra["price"] < target1 < target2
    intra["price"] = None
    result = score_v4(daily, intra, market, context=context)
    assert result[0] != "ENTRY TRIGGERED"
    assert all(math.isnan(level) for level in result[3:])

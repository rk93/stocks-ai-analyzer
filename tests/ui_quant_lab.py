"""V4 read-only evidence views. Run against `python -m http.server 8765 --directory docs`."""
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = os.environ.get("UI_BASE_URL", "http://127.0.0.1:8765/")


def fixture():
    return {
        "schema_version": 1,
        "generated_at": "2026-11-05T10:00:00Z",
        "status": "research_only_inconclusive",
        "experiment_id": "v4-frozen-example",
        "config_hash": "a" * 64,
        "config": {"version": "V4", "slippage_bps": 5, "entry": "next_bar_open"},
        "validation": {
            "status": "inconclusive",
            "untouched_test_status": "awaiting_future_observations",
            "limitations": ["Already-observed dates are retrospective evidence."],
        },
        "comparisons": [
            {"strategy": version, "source": "corrected_equal_execution_replay",
             "signals": 20, "trades": 10, "expectancy_r": -.15,
             "profit_factor": .75, "win_rate": .4, "max_drawdown": -.04,
             "confidence_interval": [-.7, .2]}
            for version in ["V1", "V2", "V3", "V4"]
        ] + [{"strategy": "V1", "source": "legacy_forward", "trades": 66,
              "profit_factor": .72, "win_rate": .515}],
        "experiments": [{"id": "fixed-2r", "name": "Fixed 2R exit", "window": "development",
                         "status": "failed_hypothesis", "metrics": {"trades": 9, "expectancy_r": -.3},
                         "assumptions": ["Adverse costs", "Stop first on ambiguous candles"]}],
        "retrospective": {
            "regimes": [{"strategy": "V4", "regime": "bullish", "source": "retrospective",
                         "count": 5, "win_rate": .4, "expectancy_r": -.12}],
            "trades": [
                {"strategy": "V4", "symbol": "AAPL", "signal_time": "2026-10-20T15:00:00Z",
                 "market_regime": "bullish", "exit_reason": "STOP", "net_r": -1.1,
                 "original_reason": "Trend confirmed", "supports": ["Relative strength"],
                 "contradictions": ["Extended entry"], "invalidation": "Frozen stop",
                 "entry_price": 100, "stop": 98, "target": 104, "score": 72},
                {"strategy": "V1", "symbol": "MSFT", "signal_time": "2026-10-21T15:00:00Z",
                 "market_regime": "sideways", "exit_reason": "TARGET", "net_r": .9,
                 "original_reason": "<img src=x onerror=alert(1)>", "score": 65},
            ],
            "decisions": [{"strategy": "V4", "symbol": "NVDA", "market_regime": "bearish",
                           "accepted": False, "reason": "Unsupported market regime",
                           "rejections": ["Regime gate"]}],
        },
        "prospective": {"created_at": "2026-11-05T10:00:00Z", "starting_cash": 10000,
                        "signals": [], "closed_trades": [], "positions": [], "equity": []},
    }


AUDIT = {
    "schema_version": 1,
    "findings": [{"id": "payoff_asymmetry", "evidence": "Mean loss exceeds mean win.",
                  "interpretation": "A 51.5% win rate can still lose money."}],
    "strategies": {"V1": {"score_buckets": {"paper": [
        {"bucket": "70-79", "count": 8, "win_rate": .5,
         "win_rate_ci95": [.21, .79], "mean_pnl": -2},
        {"bucket": "80-89", "count": 0, "win_rate": None, "mean_pnl": None},
    ]}}},
}


with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=os.environ.get("CHROMIUM_EXECUTABLE") or shutil.which("chromium"),
        headless=True, args=["--no-sandbox"],
    )
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    report = fixture()
    outage = False
    requests, errors = [], []

    def serve_report(route):
        if outage:
            route.fulfill(status=503, body="Research unavailable")
        else:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(report))

    context.route("**/data/v4-research.json", serve_report)
    context.route("**/data/quant-audit.json", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(AUDIT)))
    context.on("request", lambda request: requests.append((request.method, request.url)))
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.clock.set_fixed_time(datetime(2026, 11, 5, 12, tzinfo=timezone.utc))
    page.goto(BASE + "v4.html")
    page.wait_for_selector("[data-ql-trade]")
    assert page.get_by_text("V4 is experimental research only.", exact=True).count() == 1
    assert page.get_by_text("Awaiting prospective observations.", exact=False).count() == 1
    assert page.get_by_role("heading", name="Corrected historical execution comparison").count() == 1
    assert page.get_by_role("heading", name="Original forward paper history · separate execution").count() == 1
    assert page.get_by_text("-0.7R to 0.2R", exact=True).count() == 4
    assert page.get_by_text("21% to 79%", exact=True).count() == 1
    assert page.get_by_text("Insufficient observations", exact=True).count() > 0
    assert page.get_by_text("failed hypothesis", exact=True).count() == 1
    assert page.locator("#prospective .metric").all_text_contents() == ["0", "0", "0"]
    assert page.locator(".ql-curve").count() == 0
    # Filters operate on evidence source and frozen regime; no strategy is recommended.
    page.get_by_label("Filter strategy", exact=True).select_option("V4")
    assert page.locator("[data-ql-count]").inner_text() == "2 records shown"
    page.get_by_label("Filter regime", exact=True).select_option("sideways")
    assert page.locator("[data-ql-no-match]").is_visible()
    page.get_by_label("Filter regime", exact=True).select_option("bullish")
    assert page.locator("[data-ql-count]").inner_text() == "1 records shown"
    page.get_by_text("Inspect evidence for AAPL", exact=True).click()
    assert page.get_by_text("Supports: Relative strength", exact=True).is_visible()
    assert page.get_by_text("Contradictions / rejection: Extended entry", exact=True).is_visible()
    assert page.get_by_text("Invalidation: Frozen stop", exact=True).is_visible()
    assert page.locator("img").count() == 0  # Original rationale remains escaped text.
    page.get_by_role("button", name="Light mode", exact=True).click()
    assert page.locator("body").evaluate("e => e.classList.contains('light')")
    assert page.get_by_role("button", name="Dark mode", exact=True).get_attribute("aria-pressed") == "true"
    page.get_by_role("button", name="Dark mode", exact=True).click()
    screenshots = Path("research")
    screenshots.mkdir(exist_ok=True)
    page.screenshot(path=str(screenshots / "v4-desktop.png"), full_page=True)
    for width in [1440, 768, 390]:
        page.set_viewport_size({"width": width, "height": 900})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
    page.screenshot(path=str(screenshots / "v4-mobile.png"), full_page=True)
    # Missing marks do not fabricate connected performance; valid forward snapshots do render.
    report["prospective"]["equity"] = [
        {"timestamp": "2026-11-05T10:00:00Z", "V1_equity": 10000, "V4_equity": 10000},
        {"timestamp": "2026-11-06T21:00:00Z", "V1_equity": 9990, "V4_equity": 10005},
    ]
    page.get_by_role("button", name="Reload published research").click()
    page.wait_for_selector(".ql-curve")
    assert "Prospective paper equity" in page.locator(".ql-curve").get_attribute("aria-label")
    # A newly published report cannot disguise a paused, dated financial view.
    report["prospective"].update(
        status="paused_reconciliation_required", status_at="2026-11-06T12:00:00Z",
        data_cutoff="2026-11-04T20:00:00Z", collection_started_at="2026-11-05T14:31:00Z",
        warnings=["Corporate action requires reconciliation <script>alert(1)</script>"],
    )
    page.get_by_role("button", name="Reload published research").click()
    page.wait_for_function("document.querySelector('[data-ql-prospective-status]').textContent.includes('paused reconciliation required')")
    assert "2026-11-04T20:00:00Z" in page.locator("[data-ql-prospective-status]").inner_text()
    assert page.get_by_text("Registration does not backfill missed observations.", exact=False).is_visible()
    assert page.get_by_role("list", name="Prospective collection warnings").is_visible()
    assert page.locator(".quant-lab script").count() == 0
    # Staleness is computed from publication age; empty payloads stay honest and usable.
    report = {"schema_version": 1, "generated_at": "2026-10-01T10:00:00Z",
              "status": "research_only_inconclusive", "prospective": {"starting_cash": 10000}}
    page.get_by_role("button", name="Reload published research").click()
    page.wait_for_function("document.querySelector('[data-ql-freshness]').textContent.includes('STALE')")
    assert page.get_by_text("No experiment register is published.", exact=False).is_visible()
    assert page.get_by_text("No V4 trade or candidate decision records are published.", exact=False).is_visible()
    outage = True
    page.get_by_role("button", name="Reload published research").click()
    page.wait_for_function("document.querySelector('#ql-load-status').textContent.includes('unavailable')")
    assert page.get_by_text("V4 report unavailable.", exact=False).is_visible()
    assert page.get_by_role("button", name="Reload published research").is_enabled()
    outage = False
    report = fixture()
    page.get_by_role("button", name="Reload published research").click()
    page.wait_for_selector("[data-ql-trade]")
    # Main workspace integrations do not require the historical dataset to load V4.
    context.route("**/data/backtest.json", lambda route: route.fulfill(status=503, body="Unavailable"))
    page.goto(BASE + "#lab")
    page.wait_for_selector(".quant-lab")
    assert page.get_by_role("heading", name="Corrected historical execution comparison").count() == 1
    assert page.get_by_text("Baseline dataset unavailable.", exact=False).is_visible()
    page.goto(BASE + "#insights")
    page.wait_for_selector("h1")
    assert page.get_by_role("heading", name="V4 research & prediction quality").count() == 1
    for legacy in ["backtest.html", "paper-trading.html", "prediction-tracker.html"]:
        page.goto(BASE + legacy)
        assert page.get_by_role("link", name="Open V4 research ↗", exact=True).count() == 1
    assert not errors, errors
    assert all(method == "GET" and url.startswith(BASE) for method, url in requests), requests
    browser.close()
    print("PASS: V4 evidence separation; uncertainty; score non-probability semantics; filters/drill-down; empty/stale/outage retry; 3 widths; workspace/legacy links; no writes or browser errors")

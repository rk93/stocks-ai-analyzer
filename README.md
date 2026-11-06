# stocks-ai-analyzer

Stock research toolkit combining the existing Nifty 500 momentum/ML analysis with an options-market **Skew Map** for optionable equities.

> Educational research only. Skew describes relative option pricing/positioning; it does not predict price direction and is not financial advice.

## Options Skew Map

The skew module compares approximately 25-delta OTM put IV with approximately 25-delta OTM call IV using a consistent 28–60 DTE expiry rule.

`normalized skew = (put IV - call IV) / ATM IV`

It combines skew with one-month price performance to classify each stock:

| Price | Skew | Quadrant | Research interpretation |
|---|---|---|---|
| Down | Calls bid | CONTRARIAN BID | Price/options disagreement; investigate |
| Up | Calls bid | CHASE | Momentum and options agree |
| Up | Puts bid | HEDGED RALLY | Rally with expensive protection |
| Down | Puts bid | FEAR | Weak tape and expensive protection |

### Improvements over a basic skew spreadsheet

- Black-Scholes delta estimation to select probability-comparable strikes
- Fixed DTE selection rule
- Both raw vol-point skew and ATM-normalized skew
- 1-month return and relative return vs SPY
- 5-observation and 20-observation skew change
- Historical skew percentile after enough observations accumulate
- Divergence ranking focused on weak relative price + rotation toward calls
- Open-interest, bid/ask-spread and delta-fit quality score
- Hard sanity ceiling for suspicious option marks
- Earnings/catalyst flag when earnings falls inside the measured expiry
- Sector agreement table using **raw** skew for cross-sector comparison
- Persistent daily history
- Interactive Streamlit dashboard
- Scheduled weekday GitHub Action
- Unit tests for core calculations

## Run locally

```bash
python -m pip install -r requirements.txt
python skew_map.py
streamlit run skew_dashboard.py
```

Scan selected symbols:

```bash
python skew_map.py --symbols AAPL NVDA MU AMD MSFT
```

Edit `skew_config.json` to change the universe, target delta, DTE window, quality thresholds, sanity ceiling and benchmark.

## Data outputs

- `data/skew_latest.csv` — latest board
- `data/skew_history.csv` — observations used for change/percentile calculations
- `data/skew_errors.csv` — names deliberately skipped when no reliable reading can be produced

The first run has no historical skew changes. Five-observation change appears after enough runs; longer-term readings improve as history accumulates.

## Dashboard

The dashboard provides:

1. Price × normalized-skew quadrant radar
2. Contrarian/hedged/catalyst KPI counts
3. Research ranking with divergence and quality scores
4. Sector agreement view
5. Per-stock skew history
6. Live symbol entry and scan

Run with `streamlit run skew_dashboard.py` and open the local Streamlit URL.

## Automation

`.github/workflows/skew-map.yml` runs after the US regular session Monday–Friday and commits the latest observations back to the repository. It can also be started manually from GitHub Actions.

Free option-chain sources can be delayed, incomplete or rate-limited. The application therefore refuses to score a name when the configured expiry/strikes cannot produce a defensible reading. For serious trading use, replace the yfinance adapter with a licensed real-time options feed while keeping the analytics layer unchanged.

## Existing Nifty 500 analyzer

The original project remains available and continues to provide:

- Daily Nifty 500 analysis
- Multi-period performance tracking
- Top-performer reports
- Machine-learning recommendations
- Telegram notifications
- Existing portfolio analysis scripts

Its original entry point remains:

```bash
python main.py
```


## Forward paper trading

The repository now includes a forward-test harness for three isolated strategies:

- **V1** — existing baseline
- **V2** — opening-range / time-normalized-volume experiment
- **V3** — V2 plus deterministic broad-market regime and risk gates

Each shadow portfolio starts with `$10,000` by default, limits each position to `$1,000`, applies adverse slippage, and records an append-only trade/equity history. The Paper Trading dashboard is published at `docs/paper-trading.html`.

Run locally:

```bash
python day_opportunities.py
python paper_trader.py
```

### Optional Alpaca paper mirror

Local shadow portfolios are always the comparison source of truth. To mirror V3 into Alpaca Paper Trading, add GitHub repository secrets `ALPACA_API_KEY` and `ALPACA_API_SECRET`, then set repository variable `PAPER_EXECUTION_MODE=alpaca`.

The broker adapter is hard-wired to `https://paper-api.alpaca.markets`; there is intentionally no live-trading endpoint in this module.

Useful environment overrides:

```text
PAPER_STARTING_CASH=10000
PAPER_MAX_POSITION_DOLLARS=1000
PAPER_MAX_OPEN_POSITIONS=5
PAPER_SLIPPAGE_BPS=7
PAPER_EXIT_AT_R=1
ALPACA_PAPER_STRATEGY=V3
```

Every V1/V2/V3 signal is versioned and frozen in the prediction ledger before outcomes are evaluated.

## V4 quantitative research

V4 is an isolated, explainable **experimental strategy**, with a trade-level loss
audit, a reproducible historical replay and a new equal-capital V1–V4 prospective
paper cohort. It has not established positive expectancy. All historical windows
have already been inspected; the retrospective tail is not an untouched test.
Original V1/V2/V3 histories remain separate and preserved.

Use the Python/package versions recorded in
[v4_environment.json](research/v4_environment.json) and the accompanying
[constraints](research/v4_constraints.txt):

```bash
python -m pip install -r requirements.txt -c research/v4_constraints.txt
python quant_audit.py
python v4_replay.py --output /tmp/v4-replay.json
python -m pytest -q
```

Replay defaults to the sealed
[enriched manifest](research/v4_inputs/capture-20261105/enriched-manifest.json)
and runs offline; `/tmp/v4-replay.json` leaves published results intact. The
[V4 research view](docs/v4.html) separates corrected historical results, original
forward history and the prospective cohort. Registration alone creates no trades
or equity history. Actual collection uses `python quant_forward.py`; its frozen
code/configuration checks prevent silently restarting an existing cohort.

Read the [findings and delivery](research/QUANT_DELIVERY.md),
[loss audit](research/QUANT_ROOT_CAUSES.md),
[validation and reproduction](research/V4_VALIDATION.md) and
[research sources](research/V4_SOURCES.md). The
[published replay](docs/data/v4-replay.json) contains the complete experiment
register, including unsuccessful variants, execution assumptions and source hashes.

## Personal research workspace

The GitHub Pages entry point now provides Overview, Discover, My Portfolio,
Strategy Lab and Insights. Run locally with `python -m http.server 8765 --directory docs`.
No Node build or backend is required. Existing advanced dashboard URLs remain available.

Holdings, journal entries, watchlists and preferences are stored only in this
browser's versioned IndexedDB. Import a CSV positions snapshot, map columns and
review before committing. Supported transaction histories can be reconciled into a new local ledger or
appended to an existing ledger for the same account. Confirm complete share
history from zero holdings, verified opening cash and net settlement totals
before committing. Conflicts, unsupported actions and missing balances block
the import. FIFO realised P/L is research accounting, not tax reporting. Export private
JSON backups before changing browser/device or clearing storage.

Prices entered into a private portfolio are manual snapshots. Cross-currency
aggregates are withheld without verified FX (except explicit GBX-to-GBP units).
No live quotes, total-return history or strategy profitability are implied.
Frozen prediction rationale and versioned close-return observations are published
separately; original predictions and existing paper ledgers remain unchanged.

Read the [audit and delivery limits](research/AUDIT.md) and
[experimental research register](research/EXPERIMENTS.md).
Run `node --test tests/*.test.mjs` and `PYTHONPATH=. python -m pytest -q tests`.
Optional browser regression: install Playwright, run the local server and
`python tests/ui_smoke.py` (Chromium expected at `/usr/bin/chromium`).

See the [transaction accounting contract](research/TRANSACTION_ACCOUNTING.md) for
broker schema evidence, fee/FX treatment, duplicate handling and storage migration.
Run `python tests/ui_accounting.py` with the local server for its browser regression.


## Dated market prices and allocation modelling

Choose **Published completed-session closes** in a portfolio to value covered
instruments using the shared public price/FX snapshot. Matching remains local;
private symbols are not sent to providers, and manual quotes are preserved.
A separate current-allocation model shows historical volatility, drawdown, beta
and correlations with explicit coverage/assumptions. These are hypothetical model
results, not your achieved returns. Historical acquisition FX is never replaced
by current FX to invent cost basis or P/L.

See [market data and modelling](research/MARKET_DATA_AND_RISK.md).
Run `python portfolio_market_data.py` to refresh the fixed public universe, or let
the nightly research workflow refresh it. `python tests/ui_market_intelligence.py`
checks the new browser flow with the local server running.

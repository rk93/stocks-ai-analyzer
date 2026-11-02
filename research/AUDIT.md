# Product and research audit

Audited source: `3d946b799e5f851a73505ea74a4e5edb1369ea34`, main branch retrieved 2 November 2026.

## Useful foundation

- Static `docs/` publication and Python-produced JSON already suit GitHub Pages; no backend or frontend build is necessary.
- Skew quality, delta/expiry filtering, catalyst flags, five-observation confirmation and signal reasons are useful transparent research inputs.
- V1/V2/V3 entry functions, point-in-time daily features, 5-minute path ambiguity handling, horizon follow-through and paper shadow portfolios provide a baseline worth preserving.
- The prediction ledger freezes first signals by strategy/symbol/day. Paper CSV writers append trades/equity. Neither is rewritten by this overhaul.
- Existing unit tests cover confluence, signal gating, options validity and V3 regime gates.

## Findings and response

| Area | Finding | Response |
| --- | --- | --- |
| Navigation | Six separately styled HTML dashboards, duplicated formatting/fetch/render code and competing tools | Shared five-section entry workspace; legacy URLs retained with workspace navigation and ticker links |
| Ownership | Public portfolio_config.json has 18 configured symbols; both legacy analysers contain different hardcoded portfolios | Public app starts empty; private IndexedDB holdings; configured options sample explicitly labelled |
| FX | Legacy PortfolioAnalyzer silently uses 1.25 GBPUSD fallback | New calculations never use fallback FX; GBX/GBP conversion is explicit; missing FX withholds totals |
| Import | No user import/reconciliation journey | Local CSV mapping, validation, preview, duplicate fingerprint, new-snapshot-only commit and failed-import preservation |
| Research performance | 31 October summary: 69/187/214 signals, negative expectancy in all versions | Baselines immutable; negative expectancy prominent; Wilson intervals, after-cost sensitivity and date-block diagnostics |
| Outcome semantics | Horizon returns are not executable P/L; unresolved and ambiguous trades affect denominators | Separate displays and explicit resolved/pending counts; no fabricated execution equity |
| Prediction tracker | Derived tracker recomputes observations; reads generic state/score/reason rather than frozen strategy-specific fields | Existing ledger preserved; new UI labels records as published. New prediction-research sidecar restores frozen strategy-specific rationale and appends versioned daily close observations; legacy hit/miss labels are not promoted as trade P/L |
| ML | ml_model.py randomly splits concatenated stock rows; target pct_change crosses symbol boundaries | Not included as calibrated predictive probability in the new interface. Retire confidence claims until grouped chronological labels and validation are implemented |
| Freshness | Payload generated_at often represents publication, not each instrument's quote cutoff | Separate manifest with unknown market timestamp, source coverage, hashes, pipeline publication state and conservative publication-age flag |
| History | Yahoo intraday window is limited; configured universe may create selection bias; earnings and fundamentals lack archived availability | Explicit limits, no OOS claim or new profitable strategy claim |
| Persistence | No private browser schema/backups | Versioned IndexedDB, atomic writes, validated backup replacement, deletion/reset and independent watchlists/journal |
| Workflow | Legacy main.yml uses retired artifact v2; refresh publishes existing schemas without a provenance contract | Existing data pipelines retained; add provenance/diagnostics to refresh, frontend tests to test workflow. Legacy main workflow upgrade remains separate to avoid changing notification behaviour |

## Verified performance

Generated 31 October 2026 01:44 UTC, coverage 7 August–30 October 2026.

| Version | Signals | Resolved | Win rate | Expectancy | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 69 | 69 | 40.6% | -0.188R | 0.68 |
| V2 | 187 | 186 | 44.6% | -0.108R | 0.81 |
| V3 | 214 | 212 | 44.8% | -0.104R | 0.81 |

At a hypothetical 0.05R round-trip cost: -0.238R, -0.158R and -0.154R.
The exploratory volume filter does not establish an improvement. See EXPERIMENTS.md.

## Delivered behaviour

- Overview, Discover, My Portfolio, Strategy Lab and Insights; dark/light preference; mobile bottom navigation; modal keyboard focus and labels; local symbol search; saved sort/sector; module hiding/order.
- Multiple named portfolios; manual CRUD; GBP default; per-share cost, entered-price valuation and cash; weights and concentration review; missing currency coverage explained.
- CSV delimiter/quoted multiline parsing; column mapping; row errors; editable ticker preview; event-history detection; reconciliation; exact duplicate protection; independent snapshots avoid overlapping import double counting.
- Atomic IndexedDB persistence; backups validated before explicit replacement; full reset; no holdings network transmission; independent watchlist and thesis journal.
- Consistent stock workspace, observed price/skew history, range selector, fundamentals with timestamp caveat, supporting/contradicting evidence, comparison and journal/watchlist actions.
- Strategy horizon comparisons, intervals, actual paper equity and open positions, benchmarks, individual replay outcomes, cost sensitivity and chronological diagnostics.
- Provenance sidecar produced without rewriting baseline data; legacy deep links remain operational.

## Explicitly incomplete requirements / data gates

This is a substantive first release, not a claim that every item in the full overhaul brief is implemented.

- XLSX parsing remains unshipped. Supported transaction accounting now requires a reviewable reconciliation flow. No real user broker export was supplied; adapters are mapped against published existing fields and synthetic tests, not represented as validated across all Trading 212 variants. The second release adds explicit transaction reconciliation for complete share histories with verified opening cash, net settlement totals, duplicate-safe appends and FIFO research realised P/L. See TRANSACTION_ACCOUNTING.md. Share transfers, corporate actions, multi-currency cash ledgers and jurisdiction-specific tax rules remain unsupported.
- Portfolio metrics use entered prices, not automatically currency-verified public quotes. Position-only snapshots still lack realised P/L and income history. Transaction-based portfolios now show booked FIFO realised P/L and net income. Daily/market total-return history, beta, covariance, verified sector/ETF look-through, asset classes and historical scenarios remain unavailable rather than invented. Cash contributions and purchases at entered prices are arithmetic scenarios, not return/risk simulations. Published sector classifications are displayed with an unclassified bucket; ETF look-through is unavailable.
- Aggregation across independent snapshots is intentionally not offered: overlapping snapshots must not be double counted. Multiple portfolios are switchable.
- Price history is archived scanner observations, not complete OHLC. RSI/MACD/ATR, attributed news, full catalyst calendars and reliable exchange normalization await a richer point-in-time dataset. Users must enter exchange-qualified symbols where ambiguous.
- Strategy capital/risk execution simulation, regime/sector/time slicing, calibrated prediction charts, prospective walk-forward V4 research and executable outcome accounting remain future work. No robust OOS profitability has been demonstrated.
- Browser storage is device/origin-specific, not encrypted cross-device sync. A future sync design needs authenticated per-user authorization, encrypted transport/storage, deletion/export, audit controls and explicit consent. No login, public-write shortcut or live trading is introduced.

## Validation

24 Python tests pass (existing tests plus diagnostics and frozen-prediction tests); 20 Node portfolio/accounting tests pass.
Chromium checks cover routes, stock details/comparison, persistence after reload, CSV commit,
duplicate rejection, invalid-import preservation, legacy navigation and overflow at desktop/tablet/mobile widths.
See `tests/ui_smoke.py` for reproducible browser checks.

## Second release

See [Transaction accounting](TRANSACTION_ACCOUNTING.md) for the supported import contract, source-format evidence, schema-1 migration, append-only local records and cross-tab write protection.

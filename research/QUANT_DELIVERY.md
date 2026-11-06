# Quantitative audit and V4 delivery

V4 is implemented, reproducibly replayed and registered for a separate prospective paper cohort. **It has not demonstrated a robust positive expected return.** All 13 development execution variants have negative net expectancy. The validation window contains only four closed V4 trades and the retrospective tail only two; neither establishes an improvement. V4 remains research-only and no real-money execution is introduced.

## Why the current strategies lose

The [complete root-cause audit](QUANT_ROOT_CAUSES.md) reads every available closed paper trade and recent resolved replay trade without changing a ledger. Its machine-readable evidence is [quant-audit.json](../docs/data/quant-audit.json), including source hashes, trade joins, feature coverage, score buckets and uncertainty.

- Legacy forward execution is a sampled scanner price, often recorded outside the regular session: 197 of 228 paper entries fall outside the weekday New York 09:30–16:00 clock window. Historical replay enters and resolves five-minute candles using different rules. These are separate experiments, not interchangeable performance samples.
- Average dollar losses exceed average wins in every version. V1 wins 51.5% of trades but needs approximately 59.7% at its observed payoff mix to break even. Removing the assumed 7 bps per-side friction algebraically still leaves all three ledgers negative.
- V1's smaller forward loss is partly selection and exposure: V1 and V3 share 38 symbol/time fills with identical recorded entry/exit prices and the same −$78.20 P/L; their unmatched trades account for the ranking difference. This is a descriptive decomposition, not evidence that adding or removing an indicator caused the losses.
- Legacy replay ignores capital competition and execution costs and can label nominal 1R targets differently from actual stop risk. Daily, intraday and refresh timestamps were not consistently separated. Original scoring functions and published V1/V2/V3 historical outputs remain intact; prospective guards prevent using stale quotes and recalculate targets from actual filled risk.
- Higher scores do not consistently imply better outcomes. Research scores are not probabilities. Missing timestamped RSI, options, earnings, sector and execution-path observations cannot be recovered by attaching today's metadata to old trades.

The original published paper state remains V1 −$202.71, V2 −$233.84 and V3 −$399.83 from $10,000 each. Summing rounded trade CSV amounts differs by five cents per version; both observations are retained and explained rather than reconciled by rewriting history.

The latest legacy audit reads the upstream 6 November refresh: the paper summary is dated 02:30:14Z and the legacy backtest 02:33:43Z. Its 449 historical signals comprise V1 64, V2 182 and V3 203 (448 resolved, one ambiguous), with gross 1R expectancies −0.250R, −0.088R and −0.109R. This rolling-source baseline differs from the sealed 5 November archive used for the corrected comparison below; that experiment retains its captured prices and hashes. The 228 closed paper trades and their P/L are unchanged by the upstream refresh. A newer source timestamp does not imply a fresh executable quote or an improvement in strategy quality.

## V4 logic and research register

[strategy_v4.py](../strategy_v4.py) and [v4_config.json](v4_config.json) isolate `V4.0.0-experimental` / `v4-pit-trend-risk-001`. Configuration SHA-256 is `d36d64a7cc8a538a17170c30a3b60f666ed3ecd72f7ee47e99a5b5304ce0fe6c`.

V4 combines a bullish prior-session market trend, stock trend and positive relative strength, confirmed opening range and same-clock volume, controlled VWAP/gap/momentum extension, observed dollar liquidity and ATR-based risk. Checks expose their inputs and acceptance/rejection reasons. The score measures uncalibrated evidence coverage. Unknown earnings/spread observations are disclosed; absent historical options, sentiment, MACD and event data do not become invented filters. A relative-volume upper cap selected by the earlier V4 proposal is excluded because it was selected after seeing this period.

[Research sources](V4_SOURCES.md) document LEAN, Backtrader, vectorbt, backtesting.py, Qlib and FinRL, with the relevant execution, point-in-time and validation ideas and their limits. No third-party strategy is imported as evidence of profitability.

The immutable experiment register is published in [v4-replay.json](../docs/data/v4-replay.json). All development variants remain visible: 1R/1.5R/2R targets, half-ATR/fixed-2% stops, trailing 1%, one/three/five-session exits, prior-MA20 exit, half profit-taking at 1R then 2R, notional-cap sizing and 0.25% risk sizing. All have negative mean net R on this development sample. The least-negative variant, a one-session time exit, still has −0.425R across 11 closed trades. It is not selected or promoted after inspecting these results.

## Corrected equal-execution comparison

Every version and a simple trend baseline receives $10,000 per window, five positions, a $1,000 notional cap, 0.5% planned entry risk, 2.5% aggregate risk, integer shares and the same correlation guard. Stops use actual fill risk. Each side pays 7 bps adverse slippage plus an assumed 1 bp half-spread; commission is zero. Entries use the first eligible future regular-session bar open. Stop/target collisions resolve stop-first and remain flagged; gap-through stops use adverse opening prices. Missing path bars prevent a fabricated resolution.

| Strategy | Development closed / net R | Validation closed / net R | Retrospective tail closed / net R |
|---|---:|---:|---:|
| V1 | 34 / −0.519 | 11 / −0.394 | 10 / −0.473 |
| V2 | 79 / −0.435 | 26 / −0.098 | 25 / −0.469 |
| V3 | 79 / −0.517 | 32 / −0.061 | 27 / −0.454 |
| **V4 experimental** | **9 / −0.900** | **4 / +0.112** | **2 / −0.059** |
| Simple trend | 77 / −0.243 | 45 / −0.240 | 39 / −0.460 |

Net R is mean realised R among closed trades, after costs. Window-end open positions remain in portfolio value and are excluded from resolved expectancy. Development runs 13 August–23 September 2026; validation 1–14 October; retrospective tail 22 October–4 November. Five sessions at each boundary are excluded. Every historical date and the universe have already been inspected; the final historical window is **not an untouched test**.

V4 marked returns are −1.122%, +0.111% and −0.042%, respectively. Lower absolute drawdown or losses primarily reflect much lower exposure and sample size; they do not demonstrate superior unit-risk expectancy. The corresponding V1 returns are −1.619%, −0.512% and −0.477%. V1 remains a required reference in the independent prospective cohort.

SPY price returns after the specified benchmark friction are −1.073%, +0.526%, −0.113%; QQQ +2.049%, −2.007%, −3.218%. These are fully invested price-return references, with different exposure from the capped strategies. Cash assumes disclosed nominal 0% interest because no historical yield series is archived.

The report supplies win rate, mean/median returns, expectancy in currency and R, profit factor, costs, drawdown, exposure, turnover, holding period, largest loss, regime/feature associations and equity/trade detail. Confidence intervals use fixed-seed 2,000 moving five-session block resamples, clustering trades by entry date. V4 has too few closed trades for these intervals in every window; unavailable intervals remain null. Short windows also suppress Sharpe/Sortino and annualised returns. Statistical uncertainty, revised Yahoo prices, current-universe selection and absent historical quotes remain material limitations.

## Reproduction and source vintages

The [validation methodology](V4_VALIDATION.md) explains point-in-time availability, universe limitations, purging, execution and uncertainty. The replay uses [enriched-manifest.json](v4_inputs/capture-20261105/enriched-manifest.json), which retains the original 72-file vintage and adds a separately hashed longer SPY daily warm-up. Its SHA-256 is `16fb00e4dc05acc19e58d988c574acdd3060b7016c2b231a560ea8b7497111a1`. The original [manifest.json](v4_inputs/capture-20261105/manifest.json) is preserved. The archived five-minute coverage contains 59 completed sessions for 36 symbols; the unfinished 5 November session is excluded.

Use the observed Python/package versions in [v4_environment.json](v4_environment.json) and the matching [constraints](v4_constraints.txt) for numerical reproduction. The enriched manifest is the replay default; the explicit argument below makes the input vintage visible.

```sh
python -m pip install -r requirements.txt -c research/v4_constraints.txt
python quant_audit.py
python v4_replay.py --manifest research/v4_inputs/capture-20261105/enriched-manifest.json --output /tmp/v4-replay.json
python v4_candidates.py
python v4_publish.py
python -m pytest -q
```

Replay is offline and deterministic for the sealed inputs, exact source hashes and fixed configurations. The report retains the experiment configuration hash `5c80bf3b48dc4b1621ce3e81d9631eef50d4a1ad9490af43e48563d78c2f151b`. Data acquisition creates new named vintages; it does not silently replace a sealed archive. No historical signal or loss is deleted or rewritten.

## Forward evaluation and application

[quant_forward.py](../quant_forward.py) owns a new append-only `data/quant_forward/` cohort, separate from the original V1/V2/V3 state, trades, predictions and equity. V1–V4 share capital and execution assumptions. Registration freezes code/configuration hashes; observations begin only when collection actually starts. Decisions are frozen after actual source retrieval, feature computation and input archival, immediately before ledger append; freshness and session eligibility are rechecked at that persistence clock before eligible future entry fills; no replay candidate is backdated into this ledger. Original executions and closed observations cannot be silently revised. Incompatible code or corporate actions pause evaluation for review.

The [active manifest](../data/quant_forward/manifest.json) is the authoritative protocol identifier. Four registration-only prelaunch records were retired before any collection, decision or execution: one for cash-only/full-window exposure reporting, one to freeze the public universe and dependency constraints, one to verify completely missing intervening exchange sessions, and one to stamp decisions after processing rather than before it. Their original manifests, sole registration events and exact retirement reasons are preserved in `research/v4_inputs/prelaunch-registration-001/`, `research/v4_inputs/prelaunch-registration-002/`, `research/v4_inputs/prelaunch-registration-003/` and `research/v4_inputs/prelaunch-registration-004/`. This did not restart a losing cohort or remove any performance observation. Full-window equity includes cash-only sessions, and exposure denominators begin at the first observed bar rather than the first entry.

The refresh workflow prevents automatic cancellation of running collectors and requires a successful commit/push of observations and published data before Pages deployment. A failed persistence step blocks publication. Its run-specific `prospective-paper-observations-*` recovery artifact attempts to preserve the prospective state for 30 days even after failure, excluding the transient lock. Recover and reconcile that manifest/append-only state before retrying a failed persistence run; do not reset records, overwrite conflicting observations or backfill missed decisions. An abruptly terminated runner can still interrupt artifact upload.

The initial prospective artifact has **zero frozen decisions, zero trades and no historical equity curve**. $10,000 is the configured starting capital, not evidence of an observed path. SPY/QQQ benchmarks also await post-collection bars. Registration alone does not backfill missed observations. Use `python quant_forward.py --initialize-only` to initialise a new cohort, and `python quant_forward.py` for actual collection; an existing cohort refuses incompatible code/configuration rather than resetting its history.

[v4.html](../docs/v4.html) and the main Strategy Lab/Insights expose the same evidence. Backtest, Prediction Tracker and Paper Trading link to it. The interface separates original forward history, corrected historical replay and the genuinely prospective cohort; includes confidence/sample disclosures, experiment register, regime and score diagnostics, equity/drawdown views, filters and trade rationale; and identifies paused collection, warnings and source cutoffs. Missing marks remain chart gaps. Publication time never substitutes for a quote timestamp. [v4_publish.py](../v4_publish.py) produces the static [v4-research.json](../docs/data/v4-research.json) aggregate without fetching data or running a strategy.

The optional [v4-candidates.json](../docs/data/v4-candidates.json) explanation sample uses a predefined date/symbol/cutoff grid and the same sealed archive/configuration. Accepted and rejected checks illustrate the strategy rules; this is not an exhaustive signal sample, is never used for performance metrics and adds no prospective observations.

## Validation evidence and release state

The publisher suite passes **16 tests and 13 subtests**, covering cohort separation, immutable input/source hashes, exact-time chart joins, absent prospective observations, stale versus newly published data, unavailable/non-finite metrics, malformed inputs, hash mismatches, sampled candidate isolation, paused collection and protected output paths. Browser tests pass at 1440, 768 and 390 pixels, including uncertainty labels, empty and stale states, source filters, evidence drill-down, escaping, paused-cohort warnings, retry after outage, theme controls, workspace/legacy links and GET-only browser requests.

Final local checks pass: **353 Python tests plus 13 subtests**, all three JavaScript regression files, all four browser suites at three responsive widths, and Python compilation. The final replay source hashes and active prospective registration were verified, and historical baseline/data files have no diff against the refreshed main branch. Remote CI is recorded in the pull request after it runs.

The broader suites cover signal determinism, point-in-time daily/candle features, execution/costs/gaps/collisions, risk sizing, historical baseline guards, append-only forward persistence and replay isolation. The pull request records their final results and CI status. This document makes no claim about a remote check that has not run. Merge remains pending passing checks and review of these inconclusive findings; no production promotion is automatic.

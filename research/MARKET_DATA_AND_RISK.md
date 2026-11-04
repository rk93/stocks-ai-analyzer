# Public market-data valuation and allocation model

## Coverage and privacy

`portfolio_market_data.py` publishes `docs/data/portfolio-market.json`, schema 1.
It fetches only the repository's public core universe, public configured sample
symbols, SPY/QQQ and the explicit public additions RR.L and RELIANCE.NS. The initial
snapshot covers 42 instruments and nine FX pairs. No browser holdings, uploaded
rows or private symbols are supplied to Python or a remote market-data provider.
Every browser downloads the same public file; portfolio matching runs locally.

Existing portfolios remain in manual-price mode. Users explicitly choose
**Published completed-session closes** per portfolio. Switching modes never
replaces entered quotes, transaction events, cash balances or acquisition costs.
Uncovered instruments remain unavailable; the app does not make symbol-specific
network requests to fill private portfolio gaps.

## Source and session contract

Yahoo Finance through yfinance provides daily OHLC, adjusted close, volume,
exchange, quote currency and exchange timezone. Currency and exchange are read
from provider metadata, not inferred from a ticker suffix. For example, CSPX.L
is USD-quoted despite its London listing, while RR.L quotes in GBX/pence.
Requested and returned provider symbols must match. Explicit listing-exchange
mismatches in positions are withheld; common MIC/provider aliases are normalised.
An execution venue is not necessarily the listing exchange. Ambiguous broker
symbols still require manual mapping; this is not an ISIN-level security master.

The pipeline rejects missing currencies, invalid bars and unknown timezones.
A current-date stock bar is included only when provider regular-session-end
metadata confirms closure and a 15-minute buffer has passed. If this cannot be
confirmed, the current-date bar is excluded. FX bars always use prior dates to
avoid partial 24-hour observations. Publication time is never called quote time.

Nightly market-data collection is attached to the existing 22:15 UTC refresh,
plus manual/push refreshes. Intraday research scans do not repeatedly fetch a year
of portfolio history. A failed source can retain its prior dated observations,
with `retained_after_error`, its original fetch/session dates and a separate retry
error. Retrying does not make old quotes fresh. The manifest records market-data
session ranges, provenance and coverage without inventing exact quote timestamps.

## Valuation and FX

Published marks must have a verified matching quote currency, an ordered valid
history, and no future date. Prices more than four calendar days old are withheld.
This is a conservative age policy, not a complete exchange-holiday calendar.
Every holding must have coverage before an aggregate portfolio value or weights
are shown; covered rows remain inspectable when another holding is missing.

FX observations are explicitly configured USD-per-unit pairs:
`GBPUSD=X`, `EURUSD=X`, `INRUSD=X`, `CADUSD=X`, `AUDUSD=X`, `CHFUSD=X`,
`JPYUSD=X`, `HKDUSD=X` and `SGDUSD=X`. The quoted currency must be USD.
For currency A to base B, the cross rate is `USD_per_A / USD_per_B`.
USD is a mathematical identity and GBX is exactly one hundredth of GBP.

For each close date, FX is taken at or before that date, never from a future row,
with at most four days' backward carry. Current valuations also reject FX older
than four days relative to today. Both close and FX dates are visible. Mixed
exchange closes and backward-carried FX are dated marks, not simultaneous
executable prices or live quotes. Price and FX dates may differ.

Snapshot-native acquisition cost is **not** converted at current FX and presented
as historical base-currency cost. Unknown historical FX means aggregate cost and
unrealised P/L remain unavailable even when market value is available. A reconciled
transaction ledger retains its actual net-settlement book basis independently of
market FX. Manual same-currency and explicit GBP/GBX costs remain supported.

## Risk model — not actual portfolio performance

The portfolio panel shows a retrospective hypothetical model of the **current
allocation**, rebalanced each aligned observation. It uses provider-adjusted
closes, dated historical FX, current value weights and flat base-currency cash.
It does not reconstruct earlier holdings, trade timing or deposits/withdrawals.
No trading costs, spread, rebalancing costs or cash interest are simulated.

All holdings require complete current marks and historical adjusted-close/FX
coverage. Returns use common date endpoints across holdings, with no future FX,
at least 60 aligned observations, and no interval longer than seven days. Sample
covariance gives volatility and beta; volatility assumes 252 aligned observations
per year. Cross-market closing times and holiday calendars make this an
approximation. The displayed date range and sample count are part of every model.

The index starts at 100. Its compounded path provides hypothetical drawdown and
return; these are not the user's achieved returns. The benchmark is fully invested
SPY converted to the same base currency over the same observed endpoints. Cash
weights/exposure may differ. Beta is withheld when benchmark coverage/variance is
insufficient. Correlations use the same aligned observations for all pairs;
0.85 is the displayed high-correlation threshold. Correlation does not establish
ETF constituent overlap or a stable future relationship.

Current weights were chosen after the historical observations, so these results
are not out-of-sample evidence of a strategy edge. Adjusted historical prices can
also be revised by the provider. No V1/V2/V3 baseline or historical prediction/paper
record is changed or re-evaluated using this new dataset.

## Stock context

The consistent stock workspace adds daily close history and dated indicators:
Wilder RSI/ATR (14), EMA MACD (12/26/9), and 20/50/200-session simple averages when
covered. MACD is seeded from the first available close; finite history affects
warm-up. Missing history remains unavailable, rather than being reconstructed
from sparse options-skew snapshots. The original scanner and daily dataset have
separate provenance and may have different observation dates.

Instrument labels and quote-currency weights are descriptive. An ETF label does
not imply knowledge of its equity/bond/commodity mix; underlying ETF holdings and
actual economic currency exposure remain unverified.

## Validation and remaining work

Five deterministic Python publisher tests cover session cutoffs, metadata,
invalid bars, source-error retention, fixed-universe collection and atomic files.
Eleven market-intelligence tests cover FX direction/dates/pence, current freshness,
no fabricated historical cost, currency/exchange mismatches, missing coverage,
model covariance/cash/flat cases and technical indicators. The browser suite uses
a synthetic public feed to verify valuation, modelling, persistence, stale/error
states and no transmission or mutation of private positions.

Actual time-weighted portfolio returns, detailed price/FX attribution, trading
calendar alignment, ETF look-through and prospectively validated strategy changes
remain separate work. The new risk model is not a substitute for those features.

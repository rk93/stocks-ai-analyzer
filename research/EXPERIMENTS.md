# Prospective experiment register

V1, V2 and V3 source and published historical records are preserved byte-for-byte.
No V4 production signal is introduced in this change.

## EXP-001: time-normalised volume confirmation

Hypothesis: requiring volume_ratio_tod >= 1.5 improves resolved 1R expectancy after 0.05R round-trip cost.

This hypothesis was inspected retrospectively against the existing 470 signals.
All three filtered baseline groups remain negative after the assumed cost:
V1 -0.222R, V2 -0.148R, V3 -0.141R. This is not evidence to promote the filter.
The sample excludes signals rejected by the original entry logic, so it cannot
evaluate a full replacement strategy. Results are in `docs/data/research-diagnostics.json`.

## Validation gate for any new strategy

1. Archive the exact code/config hash, dated universe, input features and availability cutoff.
2. Register the entry/exit rules and costs before collecting an untouched future window.
3. Fit only on chronological development folds; purge overlapping holding periods and embargo boundaries.
4. Test the frozen rule once on a later untouched window; retain rejected experiments and all comparisons.
5. Use next-bar tradable prices, spreads, slippage, commissions, gaps and pessimistic ambiguous fills.
6. Simulate capital, position limits, correlated exposure, turnover and end-of-test liquidation.
7. Compare with buy-and-hold benchmarks on identical dates/cashflows; report uncertainty and unresolved coverage.
8. Promote only with sufficient independent observations and positive after-cost evidence. Do not optimise to one metric.

RSI/regime, relative strength, sector momentum, skew divergence, earnings avoidance,
ATR stops and holding-period alternatives require point-in-time feature archives
and execution paths that are not fully present in the current published contract.
Do not backfill current fundamentals, today's earnings calendar or current ETF
constituents into historical decisions.

## Limits of current diagnostics

The 70/30 split is by distinct signal dates, with 7 October 2026 as the later
window boundary. It is descriptive because prior strategy development used this
period. Date-block bootstrap preserves same-day cross-sectional dependence, but
not dependence across overlapping holding periods. Costs in R are a sensitivity
assumption, not broker fee estimation. Unresolved/ambiguous observations are
excluded and counted. No portfolio equity curve is inferred by summing overlapping
trade R values; the UI uses the existing forward paper ledger instead.

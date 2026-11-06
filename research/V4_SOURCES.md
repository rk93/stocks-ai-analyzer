# V4 methodology: primary-source review

Reviewed on 2026-11-05. The links below pin the source revisions actually read;
they are engineering references, not evidence that a strategy has positive
expectancy. No third-party strategy code, trained model, dependency, or reported
performance was imported. In particular, a backtesting framework cannot repair
missing point-in-time observations or make retrospectively chosen filters an
untouched experiment.

## QuantConnect LEAN

Source revision: `d1c09284c438b70eb965b7dc40f44e7dca69e7b5`.

- [EquityFillModel: market fills](https://github.com/QuantConnect/Lean/blob/d1c09284c438b70eb965b7dc40f44e7dca69e7b5/Common/Orders/Fills/EquityFillModel.cs#L124)
- [EquityFillModel: stop-market fills](https://github.com/QuantConnect/Lean/blob/d1c09284c438b70eb965b7dc40f44e7dca69e7b5/Common/Orders/Fills/EquityFillModel.cs#L182)
- [ConstantSlippageModel](https://github.com/QuantConnect/Lean/blob/d1c09284c438b70eb965b7dc40f44e7dca69e7b5/Common/Orders/Slippage/ConstantSlippageModel.cs)

Inspected the bid/ask selection, stale-data checks, exchange-open check, adverse
slippage, and stop-gap handling. A sell stop gapped through at the open fills at
the lower opening price less slippage, rather than the stop price. The source
also explicitly describes OHLC ordering limitations for stop-limit orders.

Application here: keep signal time, available-data cutoff, order eligibility time,
and fill time distinct. Reject stale observations and use adverse opening prices
when stops are crossed by a gap. A percentage slippage assumption is a sensitivity
input; it is not a measured spread or market-impact estimate. LEAN's equity model
does not establish how this repository's missing intraday path actually unfolded.

## Backtrader

Source revision: `b853d7c90b6721476eb5a5ea3135224e33db1f14`.

- [BackBroker: execution defaults and slippage parameters](https://github.com/mementum/backtrader/blob/b853d7c90b6721476eb5a5ea3135224e33db1f14/backtrader/brokers/bbroker.py#L36)
- [Market and stop execution](https://github.com/mementum/backtrader/blob/b853d7c90b6721476eb5a5ea3135224e33db1f14/backtrader/brokers/bbroker.py#L853)

Ordinary market orders execute using the next bar's open; `_try_exec_market`
requires the bar timestamp to follow order creation unless an explicit cheat
mode is enabled. `_try_exec_stop` uses the opening price for a gap through the
stop. The broker exposes commissions, cash checks, slippage, optional fill-volume
handling, and explicit cheat-on-close/open switches. Default slippage is zero;
slippage can also be capped by bar prices depending on configuration.

Application here: same-close fills cannot be justified using indicators that
require that close. Capital, risk limits, cash reservation and fill assumptions
must be identical when comparing versions. A library default must be recorded
explicitly rather than being treated as realistic execution. No Backtrader
strategy or parameter set is adopted.

## vectorbt

Source revision: `c923f74c9e4d1e07e626ae9002f7382cd8264833`.

- [Portfolio: signal timing and stop priority](https://github.com/polakowo/vectorbt/blob/c923f74c9e4d1e07e626ae9002f7382cd8264833/vectorbt/portfolio/base.py#L2307)
- [Portfolio: intrabar look-ahead warning](https://github.com/polakowo/vectorbt/blob/c923f74c9e4d1e07e626ae9002f7382cd8264833/vectorbt/portfolio/base.py#L2653)
- [Order simulation: adverse slippage](https://github.com/polakowo/vectorbt/blob/c923f74c9e4d1e07e626ae9002f7382cd8264833/vectorbt/portfolio/nb.py#L89)
- [Benchmark value: fee/slippage limitation](https://github.com/polakowo/vectorbt/blob/c923f74c9e4d1e07e626ae9002f7382cd8264833/vectorbt/portfolio/base.py#L4745)

The documentation embedded in `Portfolio` tells users to shift close-derived
signals forward and use a price after the signal. `from_signals` gives stop
signals priority and permits only one order per symbol per bar. Fees, fixed fees
and slippage are explicit simulation inputs. Its convenience market benchmark
does not include fees or slippage; the source advises a separate portfolio for
that comparison.

Application here: vectorised speed is useful for a fixed, registered hypothesis
matrix, but it must not encourage repeated selection on the final test. Stop
priority is a simulator convention, not recovered candle ordering. Use the same
capital and cost treatment for SPY/QQQ baselines; distinguish a cost-free reference
index from an investable benchmark. Broad parameter sweeps would need a complete
experiment register and genuinely later validation, not only the best result.

## backtesting.py

Source revision: `ca2e2611621e472542ba90f7243a1fa06a7d7108`.

- [Broker order processing](https://github.com/kernc/backtesting.py/blob/ca2e2611621e472542ba90f7243a1fa06a7d7108/backtesting/backtesting.py#L875)
- [Warning about unresolved same-bar contingent exits](https://github.com/kernc/backtesting.py/blob/ca2e2611621e472542ba90f7243a1fa06a7d7108/backtesting/backtesting.py#L1037)

Inspected `_process_orders`, spread/commission adjustments, margin checks and
contingent exits. The engine uses a pessimistic convention for a stop-limit
ordering case. For a parent stop/limit entry and potentially competing same-bar
SL/TP, it explicitly warns that the intrabar path cannot be asserted and defers
the affected contingent execution to a later matching bar. Market execution can
use the next open or an explicitly enabled previous close.

Application here: retain ambiguous outcomes or apply a disclosed conservative
rule; never count the favourable level first merely because both lie inside an
OHLC candle. The particular deferral used by this framework is not automatically
the right convention for this repository. Include gap loss and costs in realised
R, since a stop distance is a risk budget rather than a guaranteed loss cap.

## Microsoft Qlib

Source revision: `54355232463878d2eebb91fe0ee5fa7fa1f5976c`.

- [LightGBM Alpha158 workflow configuration](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- [RollingGen and trunc_segments](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/workflow/task/gen.py#L128)
- [Forward-return label definitions](https://github.com/microsoft/qlib/blob/54355232463878d2eebb91fe0ee5fa7fa1f5976c/qlib/contrib/data/handler.py#L89)

The example separates training (2008–2014), validation (2015–2016), and testing
(2017 onward), with separate processor-fit dates, transaction costs and a
benchmark. `RollingGen` supports expanding or sliding training windows and a
`trunc_days` parameter intended to prevent future-information leakage.
`trunc_segments` shortens non-test segments relative to the test start. Forward
return expressions are labels, not permissible contemporaneous features.

Application here: define chronological development, validation and final-test
windows before evaluating the candidate. Purge observations whose outcome window
reaches the next segment, and disclose any additional embargo in trading sessions
using the maximum holding/label horizon. Fit any normalisation only on training
data. Qlib's example dates, China-market costs and trained LightGBM configuration
are not transferable assumptions for this app. Neither a date split nor an
embargo reverses leakage caused by choosing rules after reviewing all outcomes.

## FinRL

Source revision: `7e98df461cbc7772f16a545a2f13b521e5d0b8f4`.

- [Stock-market environments and train/validate/test/trade pipeline](https://github.com/AI4Finance-Foundation/FinRL/blob/7e98df461cbc7772f16a545a2f13b521e5d0b8f4/docs/source/start/three_layer/environments.rst)
- [StockTradingEnv: buy/sell costs, cash, actions and state transition](https://github.com/AI4Finance-Foundation/FinRL/blob/7e98df461cbc7772f16a545a2f13b521e5d0b8f4/finrl/meta/env_stock_trading/env_stocktrading.py#L115)
- [Example train/test/trade entry points](https://github.com/AI4Finance-Foundation/FinRL/blob/7e98df461cbc7772f16a545a2f13b521e5d0b8f4/finrl/main.py)
- [Configured dates](https://github.com/AI4Finance-Foundation/FinRL/blob/7e98df461cbc7772f16a545a2f13b521e5d0b8f4/finrl/config.py#L9)

The documentation describes training, validation, historical testing and later
paper/live environments. The inspected environment charges distinct buy and sell
costs, limits purchases by available cash, processes discrete actions, and updates
portfolio rewards through time. This is useful separation of data, policy,
execution and accounting responsibilities.

Important limitation: the inspected example configuration uses the same dates for
`TEST_*` and `TRADE_*`. It does not by itself guarantee a separate untouched third
sample. The environment's state and action fill-price timing also needs an
explicit availability audit before reuse. A reinforcement-learning agent would
add considerable model-selection and sample-size requirements to the short
history available here. V4 therefore borrows the staged evaluation discipline,
not an RL agent, reward function or broker integration. All local execution stays
paper-only.

## Existing repository experiment: unmerged PR #9

- [PR #9: experimental bull-regime volume gate](https://github.com/rk93/stocks-ai-analyzer/pull/9)
- [Inspected strategy at its head revision](https://github.com/rk93/stocks-ai-analyzer/blob/dd4ed6150d3983009fd595cb474b61886ddca133/strategy_v4.py)

At review, PR #9 was open and unmerged, with head
`dd4ed6150d3983009fd595cb474b61886ddca133` on
`feature/strategy-v4-bull-volume-gate`. It calls V3, retaining its score and
geometry, then requires `BULL_TREND` and `1.4 <= volume_ratio_tod < 2.5` for entry.
Its own description says those gates were derived from current V3 diagnostics
and that the selected subset performed positively on both sides of a chronological
split.

This is a prior hypothesis worth retaining in the experiment register, not an
independently validated V4 winner. If the gates were chosen after seeing the
diagnostics across the split, the later portion is already exposed to selection.
The PR has not been merged or silently promoted by this work. Its historical
claim must be kept separate from a newly frozen configuration and genuinely
prospective observations.

## Consequences for this experiment

1. Use simple market/trend/relative-strength/entry-quality hypotheses and record
   rejection reasons. A research score is not a probability or calibration result.
2. Freeze configuration, source revision, data hashes, availability cutoff and
   experiment identifiers. Record all tried variants, including failures.
3. Compare versions and simple benchmarks under one disclosed executable replay
   contract, including capital, exposure limits, fees, slippage and gap handling.
   Legacy reported outcomes remain intact and separately labelled.
4. Treat overlapping trades and market sessions as dependent observations when
   estimating uncertainty. Count independent dates/regimes as well as trades;
   use chronological evaluation and clustered uncertainty estimates where feasible.
5. When point-in-time features or reliable bars are absent, report coverage loss
   and an inconclusive result. Do not reconstruct earnings, volume, regime or
   intraday order using information published after the signal.
6. Reserve positive-expectancy and promotion claims for adequate independent
   evidence after realistic costs. None of the reviewed framework implementations
   supplies that evidence for V4.

## Registered local candidate

The separate local candidate is `V4.0.0-experimental`, experiment
`v4-pit-trend-risk-001`. Its complete configuration is
[`v4_config.json`](v4_config.json), SHA-256
`d36d64a7cc8a538a17170c30a3b60f666ed3ecd72f7ee47e99a5b5304ce0fe6c`.
`strategy_v4.py` verifies this registered hash and returns it with every evaluation.
Different scenarios must retain their distinct configuration identities; changes
to registered rules require another explicit experiment version.

The candidate requires finite prior-session market trend/volatility and stock
trend, positive stock-minus-SPY 20-day return, a completed opening-range breakout,
non-extended VWAP/momentum/gap conditions, same-clock volume confirmation with at
least five prior sessions, and observed completed-bar dollar volume. These are
economic guardrails and unvalidated hypotheses, registered without adjusting them
to replay outcomes. More gates can reduce coverage and introduce selection risk;
they do not establish better returns. No RSI, MACD, options or sentiment bonus is
added without point-in-time evidence and independent validation. The proposed
upper RVOL cap from PR #9 is not adopted as a proven improvement.

The ATR stop is half a prior-session daily ATR, bounded at 0.6–3% of signal price.
One R is exactly the actual signal-price-to-stop distance; 1R/2R targets use that
distance consistently. This changes baseline risk geometry as well as entry
authorization, so an overall result alone cannot attribute improvement to an
individual filter. Execution comparison and entry/risk ablations must disclose
that distinction. Proposed levels are research levels; actual fills, costs and
gaps determine realised risk and P/L.

Missing or future core evidence and stale availability timestamps block an entry.
An unavailable historical earnings calendar or bid/ask quote is disclosed; no
historical avoidance benefit is claimed. If optional event/quote information is
supplied, it must have been available at the signal and pass its own gate. Market
holiday and early-close eligibility belongs to the point-in-time data adapter.
Cross-position correlation and sector/ETF overlap are not inferred from current
metadata or claimed by this candidate function. The portfolio allocator must
enforce its separate cash/exposure/risk budgets.

The displayed score is the percentage of required checks satisfied, equally
weighted. It is an uncalibrated explanation of evidence coverage, never a
probability or forecast of profitability. A failed hard gate cannot be overridden
by a high score. This candidate is research-only, and the already-inspected
historical period is exploratory rather than an untouched final test.

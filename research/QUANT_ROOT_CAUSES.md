# Quantitative loss audit: 6 November 2026 snapshot

The published evidence does not establish a profitable strategy. V1 has the
smallest observed forward loss; V2 has the least-negative recent gross replay
expectancy. Those are different experiments. The most actionable finding is a
material execution-protocol mismatch, followed by adverse payoff asymmetry and
uncalibrated scores. Additional indicators do not establish an advantage.

This audit is descriptive. None of the available historical period is an
untouched final test: previous strategy development, dashboards and an earlier
V4 proposal already inspected these dates. The code changes accompanying this
research apply prospectively; they do not repair the old fills or establish a
comparable continuation of the old protocol.

## Reproduction and preservation

Run from the repository root:

```sh
python quant_audit.py
python -m pytest -q tests/test_quant_audit.py
```

The generator reads five existing source files and writes only the new
`docs/data/quant-audit.json` artifact. It makes no network requests, reruns no
strategy and changes no source ledger. The artifact records each input's SHA-256,
row count and source summary timestamps. There is no wall-clock-dependent field;
identical inputs produce identical bytes. Its `as_of` is the latest source
summary timestamp, not a claim that prices are live. Missing or malformed input
is an explicit incomplete pipeline state. Output cannot overwrite an input path.

The snapshot contains all 228 closed paper trades, all 449 recent legacy replay signals
(448 resolved and one ambiguous), and 336 frozen prediction rows. No losing
trades, unsupported features, or inconvenient experiments are discarded. All
source trades remain available under `trades` for drill-down. The replay covers
signal dates 14 August–4 November; the forward closed trades entered on
21 September–4 November. These are approximately six weeks of forward experience,
not six weeks of independent market regimes.

The audit reads the latest upstream refresh: paper summary timestamp
2026-11-06T02:30:14.568382Z and legacy backtest timestamp
2026-11-06T02:33:43.754953Z. The backtest summary reports market coverage
13 August–5 November; its actual signal rows span 14 August–4 November. These
rolling Yahoo inputs differ from the separately sealed 5 November V4 archive,
whose completed-session replay ends on 4 November. The corrected V4 experiment
retains that archive and its hashes rather than silently substituting newer
prices. Compare performance within a source/execution cohort, not across these
different input vintages.

## What the observed results mean

| Forward paper ledger | V1 | V2 | V3 |
|---|---:|---:|---:|
| Closed trades | 66 | 79 | 83 |
| Entry dates | 28 | 31 | 31 |
| Win rate | 51.5% | 44.3% | 41.0% |
| Average win | $15.21 | $15.62 | $15.46 |
| Average absolute loss | $22.50 | $17.74 | $18.89 |
| Win/loss payoff ratio | 0.676 | 0.880 | 0.818 |
| Break-even win rate at observed payoff | 59.7% | 53.2% | 55.0% |
| Profit factor | 0.718 | 0.700 | 0.568 |
| Summed rounded trade P/L | −$202.76 | −$233.89 | −$399.88 |
| Average P/L per trade | −$3.07 | −$2.96 | −$4.82 |
| 95% date-block interval, average P/L | [−$9.58, $1.98] | [−$7.82, $1.70] | [−$9.88, −$0.11] |
| Mean holding hours | 47.25 | 46.27 | 47.65 |

The published state summaries retain full-precision running P/L of −$202.71,
−$233.84 and −$399.83. Each differs by five cents from summing individually
rounded CSV trades. The artifact reports both totals and the difference; neither
source is rewritten. The break-even rates above describe observed dollar payoff
mixes, not stable future probabilities.

| Legacy gross 1R replay | V1 | V2 | V3 |
|---|---:|---:|---:|
| Signals | 64 | 182 | 203 |
| Resolved 1R outcomes | 64 | 182 | 202 |
| Win rate | 37.5% | 45.6% | 44.6% |
| Expectancy | −0.250R | −0.088R | −0.109R |
| 95% date-block expectancy interval | [−0.455, −0.032]R | [−0.283, 0.096]R | [−0.294, 0.067]R |
| Profit factor | 0.600 | 0.838 | 0.804 |

The reference prompt's V1/V2 replay figures are older. These values are
recomputed from the actual 6 November upstream ledger, not copied from that prompt.
All three point estimates are negative before executable transaction costs.

Intervals resample whole entry dates 2,000 times with fixed seed 20261105,
retaining each sampled date's complete cross-section and calculating a
row-weighted mean. They preserve within-day co-movement, but not all serial
dependence from overlapping holdings. With fewer than two entry-date clusters,
an interval is unavailable. These are exploratory percentile intervals, not
independent-trade significance tests, model-selection gates, or proof that V3
will lose in every future regime.

## Why historical and forward rankings differ

### 1. Legacy paper timestamps frequently fall outside the trading session

197/228 recorded entries fall outside the weekday 09:30–16:00 New York clock
window: V1 65/66, V2 64/79 and V3 68/83. The check correctly converts UTC, so a
02:31 UTC entry belongs to the previous New York date. It is a clock/session
diagnostic; it does not pretend to reconstruct holidays or early closes.

In the legacy `paper_trader.py` producing these records, `current_prices` reads
the latest scanner price and `open_positions` buys it whenever the workflow runs.
There is no verified contemporaneous executable quote. The schedule includes
after-hours refreshes; workflow delays and manual runs add further variation.
The signal generation timestamp is not the price's availability timestamp. A
quote sampled after the close can represent an old regular-session bar.

`backtest_engine.py` instead enters the close of the first qualifying 30-minute
checkpoint, then evaluates future five-minute bars. Matching only a symbol and
date yields just 10 V1, 37 V2 and 37 V3 pairs. Of these, win/loss classifications
disagree on 4, 16 and 16 pairs; prices differ by more than 10 bps on 10, 33 and 33.
These pairs are coverage diagnostics, not identical trades.

The precise counterfactual fill cannot be reconstructed from this ledger. It
would be incorrect to replace old entries with attractive intraday prices or
claim that fixing session checks makes the strategy profitable.

### 2. The replay and paper stop/target models are materially different

The legacy replay yields exactly +1R or −1R after ordering target/stop touches in
five-minute bars. Same-bar stop and target touches are ambiguous and excluded
from the resolved denominator. It does not charge costs, fill gap-through stops
at the opening price, allocate capital or simulate portfolio-level selection.
It can hold overnight until a touch occurs anywhere in its remaining coverage.

The legacy paper simulator only checks the scanner's sampled price on refresh.
If it is below a stop, it sells at that sampled price less 7 bps, not at the stop.
It may miss both an intervening target and a later stop. A stop can be exceeded
substantially before the next refresh. A sampled target can overshoot too, but
the observed average losses exceed average wins in every strategy.

V1's 51.5% win rate therefore remains below its observed 59.7% break-even rate.
Its worst two trades, ORCL −$75.91 and PLTR −$63.56, together contribute −$139.47.
V3 shares the ORCL loss and also loses $41.63 in AMZN and $39.75 in GE. Those
are observed loss contributors, not proof that earnings or opening gaps caused
them: the exit path and point-in-time event data were not saved.

Removing the default 7 bps per-side slippage algebraically, while leaving fills
and trade selection fixed, still yields negative totals of approximately
−$127.94, −$142.28 and −$304.32. This sensitivity assumes the current default
applied to every historical trade; original per-trade cost configuration is not
archived. Costs worsen results but do not, by themselves, explain away losses.

The V2/V3 geometry also permits an opening-range low above the nominal
ATR-derived stop. The target risk floor can then differ from the actual
entry-to-stop distance. The legacy replay still labels winners +1R. A common
future execution harness must calculate R from the actual fill and original
risk, preserve both, and model the actual target payoff rather than infer it
from a label. Existing signals and labels must remain immutable.
In this snapshot the nominal “1R” target exceeds actual initial risk in 2/64 V1,
3/182 V2 and 3/203 V3 signals (maximum ratio 1.322 in each version). This is a
measurement inconsistency, but its limited incidence cannot explain all losses.

### 3. Candidate populations, capital and exposure differ

The replay takes every first qualifying symbol/day signal in today's configured
universe. The paper portfolios prioritise higher scores, cap positions at five,
use at most $1,000 per position, skip instruments for which an integer share
cannot be bought, and block repeat symbol/day signals. Fixed dollars do not
mean fixed risk; integer-share average entry notionals are only $811/$829/$824.
Repeated symbols, simultaneous entries and common market days also correlate
the observations. The 449 replay rows represent only 246 unique symbol/dates.

V1/V2 share 39 symbol/minute paper trades, with identical recorded entry/exit
prices and +$57.44 each. V1/V3 share 38 such trades and −$78.20 each. On unshared
trades, V1 loses −$124.56 versus V3 −$321.68. This accounts for the observed V1/V3
ranking mechanically: their common fills do not produce the difference.
Unshared selection, timing, holding capacity and risk mix differ, so it is not
a causal estimate of one filter's value.

A fair prospective comparison must run V1 alongside V2/V3/V4 on the same
timestamped inputs with the same capital, costs, risk limits and execution
protocol. Existing equity curves must remain visible as a separate legacy cohort.

### 4. Some data and metric contracts do not support causal attribution

The replay's `daily_asof` and `market_asof` correctly restrict daily inputs to
earlier sessions. However, legacy intraday bars are labelled by their start
time; treating their OHLC as known at that exact checkpoint can expose up to one
bar's future information. The old opening-range interval includes the 10:00
bar. A prospective fix should use completed bars and a half-open range, while
preserving the original results. The original scanner's daily/market sampling
also differs from historical as-of reconstruction; this changes feature values.

Signal prices use raw Yahoo OHLC while benchmarks use adjusted closes. Without
a frozen corporate-action ledger and historical universe membership, a
survivor-free, corporate-action-consistent long-term replay cannot be certified.
Rolling Yahoo five-minute coverage changes on each refresh. Downloading current
fundamentals, today's universe or revised history does not turn these records
into point-in-time observations.

`trending_engine.py` calculates relative-strength and trend context using
current adjusted history and current fundamental metadata. `skew_map.py` records
options diagnostics and replaces the same symbol/day history observation with
the latest run. `confluence_engine.py` joins current symbol snapshots without a
common availability cutoff and rewards “healthy” volume and bull regimes by
hand-set weights. These can be useful research views, but they are not immutable
features for historical execution attribution. Neither their options score nor
their current sector/event fields may be backfilled into old trades.

The legacy prediction tracker combines daily-close direction with daily-window
target/stop labels. Unordered OHLC windows and future-close returns cannot be
treated as executed trade P/L. This audit excludes those correctness labels and
uses resolved replay outcomes and actual paper ledger P/L separately.

## Feature and calibration findings

All 228 paper trades have exactly one same-strategy, same-symbol, same-New-York-
date prediction whose generation and freezing times precede the recorded entry.
The audit joins that frozen context only. It prefers `prediction_*` fields and
uses V2/V3-specific fallbacks; it never substitutes generic V1 scores or levels.
The old closed ledger lacks the original prediction ID, stop and target, and
the first frozen prediction can precede a later actual signal by many hours.
Six trades have a different recorded entry score from the first frozen score;
the largest lag is 642 minutes. This is explicitly exposed in the trade records.
The join is therefore contextual, not proof that those were the executed levels.
The underlying market-data availability cutoff is unknown throughout.

The JSON reports score buckets and stratifies available features by entry hour,
regime, time-normalised volume, VWAP distance, signal-day move, opening gap,
daily ATR and holding time. Every group includes counts and date-block
uncertainty. Group patterns are exploratory associations, confounded by symbol,
market date, score-selection and exposure; changing filters on their basis
would be retrospective model selection.

| V3 paper score | Trades | Win rate | Mean P/L |
|---|---:|---:|---:|
| 60–69 | 10 | 50.0% | −$2.72 |
| 70–79 | 24 | 45.8% | −$3.68 |
| 80–89 | 38 | 34.2% | −$6.28 |
| 90+ | 11 | 45.5% | −$4.17 |

Higher score has not produced monotonic improvement. V2's 80–89 bucket has only
four forward trades and loses $22.86 per trade; its historical 60–69 bucket has
104 outcomes and +0.135R, versus 76 outcomes and −0.395R at 70–79. These are not
calibrated probabilities. No Brier score, log loss or probability reliability
claim is valid when the source provides only hand-weighted research scores.
For future calibration, freeze the outcome definition, fit the mapping only in
development folds, and evaluate it prospectively without refitting the test.

The proposed bull/moderate-volume gate deserves particular caution:

- Paper bull-regime totals are negative in every strategy: V1 −$278.57/39,
  V2 −$220.39/43 and V3 −$368.72/48. A bull filter is not established protection.
- V3's frozen time-normalised volume [1.4, 2.5) group loses $305.68 over 42
  trades, versus $25.37 over 35 at [2.5, 5). Moderate volume alone is not a
  demonstrated improvement over the current forward sample.
- V3 frozen VWAP distance 0–0.5% has +$95.98 across 38 trades, whereas
  0.5–1.5% has −$495.86 across 45. V2 shows a similar association. This is a
  candidate for preregistration, not a validated rule: both selection and exit
  protocols are confounded, and the same period has now been inspected.

The report explicitly lists unsupported analyses. Historical replay rows do not
save regime, ATR, relative strength, RSI, MACD, earnings, options or sector.
Paper predictions save ATR/trend inputs but lack frozen RSI/MACD, 200-day MA,
benchmark-relative strength, earnings proximity, options, spread, sector and
exit paths. Original per-trade risk, gap versus delayed-exit loss attribution,
stops-before-recovery and intrabar partial/trailing-exit counterfactuals cannot
be recovered reliably. Missing fields remain unknown rather than being
reconstructed using today's information.

## Decisions this evidence supports

1. Preserve V1/V2/V3 signals, closed trades, equity snapshots and historical
   labels unchanged. Publish diagnostics as a new, reproducible artifact.
2. Correct execution and feature timing prospectively. Freeze completed-bar
   cutoffs, original signal IDs, stops, targets, sizing, costs and protocol
   version; enforce regular-session and quote-freshness entry checks.
3. Introduce isolated V4 research and a simultaneous common-protocol V1 control.
   Preregister immutable rules and a truly future validation start. Preserve
   all rejections and negative results. No observed historical subperiod can
   become untouched merely by naming it “test.”
4. Use archived executable paths for costed stop/target, gap, trailing, partial
   and holding-period experiments. Report unavailable data before producing a
   result. Do not reconstruct profitable paths from daily highs and lows.
5. Keep V4 research-only until economically meaningful prospective evidence
   accumulates across independent market conditions. A positive retrospective
   subgroup, a confidence threshold, or more signal components is insufficient.

The present conclusion is negative for demonstrated profitability and
inconclusive for the value of proposed entry filters. Fixing measurement and
starting a clean prospective comparison is a concrete improvement even when
the honest strategy result remains inconclusive.

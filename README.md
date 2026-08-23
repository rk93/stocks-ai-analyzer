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

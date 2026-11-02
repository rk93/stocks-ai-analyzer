# Local transaction accounting — second release

## Source formats inspected

The adapter was grounded in the public Trading 212 example header at
[portfolio-performance-trading212-addon/resources/activity2026.csv](https://github.com/jcor1965/portfolio-performance-trading212-addon/blob/main/resources/activity2026.csv)
and its [extractor field handling](https://github.com/jcor1965/portfolio-performance-trading212-addon/blob/main/src/org/trading212/ppaddon/Trading212CSVExtractor.java).
Only the schema and field semantics were used; no personal transaction rows or
third-party implementation code are copied into this repository.

The inspected header distinguishes `Time (UTC)`, `ID`, `No. of shares`,
`Price / share`, `Currency (Price / share)`, `Total`, `Currency (Total)`,
`Result` and withholding-tax fields. `Result` is not cash settlement.
Header matching supports `Time` and `Time (UTC)` variants, prioritises specific
currency headers over generic `Currency`, and still allows manual column mapping.
The committed CSV fixture uses synthetic securities, quantities and amounts.
This is not certification of every broker/export version.

## Supported journey

1. Upload locally, check mapped columns and correct symbols.
2. Choose a new transaction-based portfolio or an existing transaction ledger for
   the **same account**. Position snapshots cannot receive transaction appends.
3. Choose a single net-settlement currency and verified opening cash. Confirm that
   all share activity is represented from zero opening holdings, that totals are
   booked cash amounts inclusive of fees/taxes, and that offset-free times are UTC.
4. Review row errors, new/duplicate counts, closing cash, remaining cost and FIFO
   realised research P/L before saving.
5. Save atomically to IndexedDB. No holdings, transactions, IDs or uploaded files
   are transmitted. Quotes remain missing until manually entered.

Supported actions: market/limit/stop buys and sells, deposits, withdrawals,
ordinary dividend variants, cash interest, standalone fees and taxes. Corporate
actions, share transfers, reversals, shorts, margin and explicit FX conversions
stop the import. Exports containing these need separate accounting or a positions
snapshot; do not remove or fabricate records to force reconciliation.

## Accounting contract

`ledger.version = 1`, `method = FIFO_NET_SETTLEMENT`. Original normalised events
and broker IDs are retained. Chronological replay derives shares, cash, remaining
native entry prices, remaining settlement-currency book basis and realised P/L.
A sell allocates the earliest remaining net acquisition costs proportionately.
Fractional shares and partial lot closes are supported without dropping tiny lots.
Trades at identical timestamps in the same symbol are blocked because FIFO order
cannot be established. Cash-event ties retain source/import order.

Native quote currency and settled cash currency are separate. For example, a USD
security acquired for a booked GBP total carries that actual GBP book cost;
no historical exchange rate is invented. Its current GBP market value remains
unavailable until verified current FX exists. GBX/GBp prices are pence and convert
explicitly to GBP. Prices are never populated from the last trade as if current.

Trade/dividend totals must already include embedded fees/taxes. The mapped fee
field is retained for audit in its original units and is **not** deducted again.
Standalone fee/tax events debit cash once. Negative credit events, missing totals,
invalid calendar dates, insufficient shares/cash and inconsistent same-currency
trade totals block the entire commit.

These are research FIFO book figures, **not tax calculations**. In particular,
UK same-day, 30-day and section-104 share matching is not implemented. Returns,
annualised performance, market-risk statistics and currency attribution are not
inferred from these accounting balances.

## Duplicate and conflict policy

A broker action/ID identifies an event within the selected account ledger.
Exactly matching existing IDs are skipped and counted. A changed amount, symbol,
price or timestamp with the same ID is a conflict: historical records are not
silently corrected. Rebuild a new separate ledger from a reconciled corrected
export if necessary. IDs are not assumed to be unique across different accounts.

Records without IDs can be imported if unique. Identical no-ID events are
ambiguous: they could be overlapping exports or two genuine identical trades.
The importer blocks them instead of guessing. It offers a new independent
full-history portfolio rather than adding overlapping snapshots together.

## Storage and migration

Workspace schema 2 adds a revision counter and ledger validation. Schema-1 data
and backups migrate in memory without dropping portfolios, watchlists or journal
entries; the next successful write persists the migrated state. The IndexedDB
upgrade reuses the existing object store. Blocked upgrades display an actionable
error. Invalid/unreadable stored data disables saving rather than overwriting it.

Each write reads and checks the stored revision inside the same IndexedDB
read/write transaction. A stale tab is rejected with a reload message, preserving
the other tab's newer data. Backup restoration retains the local revision guard.
A ledger backup must replay to its stored cash, quantities and cost basis before
it can replace a workspace. Manual editing of ledger holdings/cash is blocked;
quote updates and portfolio renaming preserve the transaction history.

## Validation

- 13 accounting/domain tests plus seven existing portfolio tests cover settlement
  currency/header priority, partial FIFO sales, income/cash flows, fees counted
  once, fractional lots, pence, overlaps, conflicting IDs, ambiguous no-ID events,
  timestamp ties, malformed dates, missing shares/cash and backup migration.
- `tests/ui_accounting.py` checks a real schema-1 IndexedDB upgrade, history import,
  explicit confirmations, quote-only edits, overlap append, conflict atomicity,
  selected-portfolio persistence, cross-tab write rejection, mobile overflow and
  absence of holdings network transmission.
- Existing Python research tests and `tests/ui_smoke.py` remain regression gates.

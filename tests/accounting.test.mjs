import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  parseCSV,
  detect,
  blank,
  validateState,
  valuation,
} from "../docs/assets/portfolio.mjs";
import {
  parseTransactions,
  reconcileHistory,
  replayLedger,
  transactionTime,
  validateLedgerPortfolio,
} from "../docs/assets/accounting.mjs";
const csv = readFileSync(
  new URL("./fixtures/trading212-history.csv", import.meta.url),
  "utf8",
);
const parsed = parseCSV(csv),
  map = detect(parsed.headers);
const result = (p = parsed, m = map, o = {}) => reconcileHistory(p, m, o);
const short = (rows) =>
  parseCSV(
    "Action,Time,Ticker,Quantity,Price / share,Currency,Total,Settlement currency,ID\n" +
      rows,
  );
const run = (rows) => {
  const p = short(rows);
  return result(p, detect(p.headers));
};
const portfolio = (r) => ({
  id: "p1",
  name: "History",
  currency: "GBP",
  cash: r.cash,
  positions: r.positions,
  ledger: r.ledger,
});
test("documented export headers map native quote and booked settlement separately", () => {
  assert.equal(parsed.headers[map.date], "Time (UTC)");
  assert.equal(parsed.headers[map.currency], "Currency (Price / share)");
  assert.equal(parsed.headers[map.settlementCurrency], "Currency (Total)");
  assert.equal(parsed.headers[map.amount], "Total");
  assert.equal(parsed.headers[map.brokerId], "ID");
});
test("FIFO across partial lots, net FX settlements, dividends and cash flows reconcile", () => {
  const r = result();
  assert.deepEqual(r.errors, []);
  assert.equal(r.added, 8);
  assert.equal(r.cash, 925.5);
  assert.equal(r.positions.length, 1);
  assert.equal(r.positions[0].quantity, 0.5);
  assert.equal(r.positions[0].cost, 120);
  assert.equal(r.positions[0].basisBase, 48.5);
  assert.equal(r.summary.realizedPnl, 69.5);
  assert.equal(r.summary.dividends, 5.5);
  assert.equal(r.summary.interest, 1);
  assert.equal(r.summary.fees, 2);
  assert.equal(r.summary.netContributions, 900);
  assert.equal(r.positions[0].price, null);
  assert.equal(valuation(portfolio(r)).value, null);
  assert.equal(valuation(portfolio(r)).cost, 48.5);
  validateLedgerPortfolio(portfolio(r));
});
test("overlapping exports deduplicate by stable IDs without mutating prior ledger", () => {
  const first = result(),
    p = portfolio(first),
    before = JSON.stringify(p);
  const overlap = result(parsed, map, { existing: p });
  assert.equal(overlap.duplicates, 8);
  assert.equal(overlap.added, 0);
  assert.deepEqual(overlap.errors, []);
  assert.equal(overlap.cash, 925.5);
  assert.equal(JSON.stringify(p), before);
  const next = short("Deposit,2026-01-09 10:00:00,,,,,50,GBP,cash-3");
  const r = result(next, detect(next.headers), { existing: p });
  assert.deepEqual(r.errors, []);
  assert.equal(r.cash, 975.5);
  assert.equal(r.ledger.transactions.length, 9);
});
test("conflicting IDs never overwrite historical records", () => {
  const p = portfolio(result()),
    changed = structuredClone(parsed);
  changed.rows[0][map.amount] = "1001";
  const r = result(changed, map, { existing: p });
  assert.match(r.errors.join(" "), /conflicts/);
  assert.equal(r.ledger, null);
  assert.equal(p.cash, 925.5);
});
test("identical events without broker IDs block rather than silently lose real fills", () => {
  const rows = "Deposit,2026-01-01 10:00:00,,,,,100,GBP,";
  const r = run(rows + "\n" + rows);
  assert.match(r.errors.join(" "), /without a broker ID/);
  assert.equal(r.ledger, null);
});
test("unsupported corporate actions, missing dates, invalid leap dates and settlement FX fail", () => {
  for (const row of [
    "Stock split,2026-01-01 10:00:00,TEST,2,5,GBP,10,GBP,id",
    "Deposit,2026-02-30 10:00:00,,,,,10,GBP,id",
    "Deposit,2026-01-01,,,,,10,GBP,id",
    "Deposit,2026-01-01 10:00:00,,,,,10,USD,id",
  ]) {
    const r = run(row);
    assert.ok(r.errors.length, row);
    assert.equal(r.ledger, null);
  }
  assert.equal(
    transactionTime("2026-01-01 09:00:00"),
    "2026-01-01T09:00:00.000Z",
  );
  assert.equal(
    transactionTime("2026-01-01T10:00:00+01:00"),
    "2026-01-01T09:00:00.000Z",
  );
});
test("incomplete holdings and negative cash are reconciliation errors", () => {
  assert.match(
    run("Sell,2026-01-01 10:00:00,TEST,1,10,GBP,10,GBP,id").errors.join(" "),
    /exceeds/,
  );
  assert.match(
    run("Buy,2026-01-01 10:00:00,TEST,1,10,GBP,10,GBP,id").errors.join(" "),
    /cash would become/,
  );
  assert.ok(
    result(parsed, map, { existing: { name: "Snapshot" } }).errors.length,
  );
});
test("same-symbol timestamp ties do not invent FIFO ordering", () => {
  const r = run(
    "Deposit,2026-01-01 09:00:00,,,,,100,GBP,cash\nBuy,2026-01-01 10:00:00,TEST,1,10,GBP,10,GBP,b1\nBuy,2026-01-01 10:00:00,TEST,1,20,GBP,20,GBP,b2",
  );
  assert.match(r.errors.join(" "), /ambiguous FIFO ordering/);
});
test("pence quotes retain net settlement basis and do not deduct embedded charges twice", () => {
  const r = run(
    "Deposit,2026-01-01 09:00:00,,,,,2000,GBP,cash\nBuy,2026-01-01 10:00:00,RR.L,100,1000,GBp,1001,GBP,b1",
  );
  assert.deepEqual(r.errors, []);
  const p = portfolio(r);
  p.positions[0].price = 1100;
  assert.equal(valuation(p).cost, 1001);
  assert.equal(valuation(p).value, 2099);
});
test("schema 1 migration is non-destructive and ledger backups must reconcile", () => {
  const old = { ...blank(), schema: 1 };
  delete old.revision;
  const before = JSON.stringify(old),
    s = validateState(old);
  assert.equal(s.schema, 2);
  assert.equal(s.revision, 0);
  assert.equal(JSON.stringify(old), before);
  const p = portfolio(result()),
    state = { ...blank(), portfolios: [p] };
  validateState(state);
  p.cash++;
  assert.throws(() => validateState(state), /does not reconcile/);
});
test("fractional lots close without dust or loss of tiny remaining shares", () => {
  const r = run(
    "Deposit,2026-01-01 09:00:00,,,,,100,GBP,cash\nBuy,2026-01-01 10:00:00,TEST,0.1,10,GBP,1,GBP,b1\nBuy,2026-01-02 10:00:00,TEST,0.2,10,GBP,2,GBP,b2\nSell,2026-01-03 10:00:00,TEST,0.3,20,GBP,6,GBP,s1",
  );
  assert.deepEqual(r.errors, []);
  assert.equal(r.positions.length, 0);
  assert.ok(Math.abs(r.summary.realizedPnl - 3) < 1e-10);
  const tiny = run(
    "Deposit,2026-01-01 09:00:00,,,,,1,GBP,cash\nBuy,2026-01-01 10:00:00,TEST,0.000000001,10,GBP,0.01,GBP,b1",
  );
  assert.deepEqual(tiny.errors, []);
  assert.equal(tiny.positions.length, 1);
});
test("embedded fees are counted exactly once and inconsistent native totals fail", () => {
  const p = parseCSV(
    "Action,Time,Ticker,Quantity,Price / share,Currency,Total,Settlement currency,ID,Fees\nDeposit,2026-01-01 09:00:00,,,,,200,GBP,cash,0\nBuy,2026-01-01 10:00:00,TEST,1,100,GBP,101,GBP,b1,1\nSell,2026-01-02 10:00:00,TEST,1,110,GBP,109,GBP,s1,1",
  );
  const r = result(p, detect(p.headers));
  assert.deepEqual(r.errors, []);
  assert.equal(r.cash, 208);
  assert.equal(r.summary.realizedPnl, 8);
  assert.equal(r.summary.fees, 0);
  const bad = run(
    "Deposit,2026-01-01 09:00:00,,,,,200,GBP,cash\nBuy,2026-01-01 10:00:00,TEST,10,100,GBP,5,GBP,b1",
  );
  assert.match(bad.errors.join(" "), /inconsistent/);
});
test("a referenced broker currency wins over generic Currency regardless of header order", () => {
  const p = parseCSV(
    "Currency,Currency (Total),Total,Time (UTC),Action\nUSD,GBP,100,2026-01-01 09:00:00,Deposit",
  );
  const m = detect(p.headers);
  assert.equal(m.settlementCurrency, 1);
  assert.deepEqual(parseTransactions(p, m).errors, []);
});

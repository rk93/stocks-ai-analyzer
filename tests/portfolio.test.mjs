import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseCSV,
  detect,
  reconcile,
  valuation,
  validateState,
  blank,
  validatePosition,
} from "../docs/assets/portfolio.mjs";
test("quoted CSV, BOM, CRLF and embedded newlines", () => {
  const p = parseCSV(
    '\uFEFFTicker;Quantity;Currency;Average price;Note\r\nAAPL;2;USD;100;"hello;\nworld"',
  );
  assert.equal(p.rows[0][4], "hello;\nworld");
  assert.equal(reconcile(p, detect(p.headers)).positions[0].quantity, 2);
});
test("rejects incomplete and malformed imports atomically", () => {
  const p = parseCSV(
    "Ticker,Quantity,Currency,Average price\nAAPL,-1,USD,100\nMSFT,2,XYZ,10\nNVDA,,USD,5",
  );
  const r = reconcile(p, detect(p.headers));
  assert.equal(r.errors.length, 3);
  assert.equal(r.positions.length, 0);
  assert.throws(() => parseCSV('a,b\n"unterminated,2'));
});
test("Trading 212 history never silently becomes holdings", () => {
  const p = parseCSV(
    "Action,Time,Ticker,No. of shares,Price / share,Currency (Price / share)\nMarket buy,2026-01-01,AAPL,2,100,USD\nDividend,2026-01-02,AAPL,,,USD\nDeposit,2026-01-03,,,,GBP",
  );
  const r = reconcile(p, detect(p.headers));
  assert.equal(r.events.length, 3);
  assert.equal(r.positions.length, 0);
  assert.equal(r.history, true);
  assert.ok(r.errors.length);
});
test("overlapping lots and unsupported dates are errors", () => {
  const p = parseCSV(
    "Ticker,Quantity,Currency,Average price,Date\nAAPL,1,USD,10,2026-01-01\nAAPL,2,USD,12,2026-01-02\nMSFT,1,USD,1,not-a-date",
  );
  assert.equal(reconcile(p, detect(p.headers)).errors.length, 2);
});
test("GBX converts to GBP; missing FX withholds totals", () => {
  const p = {
    currency: "GBP",
    cash: 20,
    positions: [
      { symbol: "RR.L", quantity: 100, currency: "GBX", cost: 100, price: 120 },
    ],
  };
  assert.equal(valuation(p).value, 140);
  assert.equal(valuation(p).cost, 100);
  p.positions.push({
    symbol: "AAPL",
    currency: "USD",
    quantity: 1,
    cost: 1,
    price: 2,
  });
  assert.equal(valuation(p).value, null);
});
test("missing prices are distinct from zero prices", () => {
  assert.equal(
    valuation({
      currency: "GBP",
      cash: 0,
      positions: [
        { symbol: "A", quantity: 1, currency: "GBP", cost: 1, price: null },
      ],
    }).value,
    null,
  );
  assert.throws(() =>
    validatePosition({
      symbol: "<script>",
      quantity: 1,
      currency: "GBP",
      cost: 1,
      price: 1,
    }),
  );
});
test("backup schema validation preserves compatibility boundary", () => {
  assert.equal(validateState(blank()).schema, 2);
  assert.throws(() => validateState({ ...blank(), schema: 3 }));
  assert.throws(() =>
    validateState({
      ...blank(),
      portfolios: [
        {
          id: "a",
          name: "X",
          cash: 0,
          currency: "GBP",
          positions: [{ symbol: "A", quantity: NaN }],
        },
      ],
    }),
  );
});

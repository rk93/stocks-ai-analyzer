import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fxRate,
  publicMark,
  publicValuation,
  portfolioRisk,
  stockIndicators,
} from "../docs/assets/market-intelligence.mjs";
const now = new Date("2026-11-04T23:00:00Z");
function history(n = 100, start = 100, factor = 1) {
  const rows = [];
  let d = new Date("2026-06-01T00:00:00Z"),
    price = start;
  while (rows.length < n) {
    if (d.getUTCDay() !== 0 && d.getUTCDay() !== 6) {
      price *= 1 + factor * (rows.length % 2 === 0 ? 0.01 : -0.005);
      rows.push({
        date: d.toISOString().slice(0, 10),
        open: price,
        high: price + 2,
        low: price - 2,
        close: price,
        adjusted_close: price,
        volume: 100,
      });
    }
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return rows;
}
const stock = (symbol, rows, currency = "USD") => ({
  symbol,
  currency,
  status: "ok",
  history: rows,
  last_session: rows.at(-1).date,
});
function dataset() {
  const h = history();
  h.push({ ...h.at(-1), date: "2026-11-03" });
  return {
    schema_version: 1,
    instruments: {
      A: stock("A", h),
      B: stock(
        "B",
        h.map((r) => ({
          ...r,
          close: r.close * 2,
          adjusted_close: r.adjusted_close * 2,
          high: r.high * 2,
          low: r.low * 2,
          open: r.open * 2,
        })),
      ),
      SPY: stock("SPY", h),
    },
    fx: {
      GBP: {
        status: "ok",
        currency: "USD",
        unit_currency: "GBP",
        history: h.map((r) => ({ date: r.date, usd_per_unit: 1.25 })),
      },
    },
  };
}
const pos = { symbol: "A", currency: "USD", quantity: 2, cost: 80, price: 7 };
test("FX uses USD-per-unit direction, explicit pence, historical dates and no future rates", () => {
  const d = dataset();
  assert.equal(fxRate(d, "USD", "GBP", "2026-11-03").rate, 0.8);
  assert.equal(fxRate(d, "GBP", "USD", "2026-11-03").rate, 1.25);
  assert.equal(fxRate(d, "GBX", "GBP", "2026-11-03").rate, 0.01);
  assert.equal(fxRate(d, "GBP", "GBX", "2026-11-03").rate, 100);
  assert.equal(fxRate(d, "USD", "GBP", "2026-05-01").rate, null);
  assert.equal(fxRate(d, "USD", "GBP", "2026-11-09").rate, null);
  d.fx.GBP.history.push({ date: "2026-11-04", usd_per_unit: 10 });
  assert.equal(fxRate(d, "USD", "GBP", "2026-11-03").rate, 0.8);
});
test("published marks reject stale/future data, mismatched currencies and missing symbols", () => {
  const d = dataset(),
    r = publicMark(d, pos, "GBP", now);
  assert.ok(r.value > 0);
  assert.equal(r.fxRate, 0.8);
  assert.equal(r.date, "2026-11-03");
  assert.match(
    publicMark(d, { ...pos, currency: "GBP" }, "GBP", now).reason,
    /mismatch/,
  );
  assert.equal(publicMark(d, pos, "GBP", new Date("2026-11-09")).value, null);
  assert.equal(publicMark(d, pos, "GBP", new Date("2026-11-01")).value, null);
  assert.equal(
    publicMark(d, { ...pos, symbol: "UNKNOWN" }, "GBP", now).value,
    null,
  );
});
test("current FX never fabricates historical foreign acquisition cost or overwrites manual prices", () => {
  const d = dataset(),
    p = { currency: "GBP", cash: 10, positions: [pos] },
    before = JSON.stringify(p);
  const r = publicValuation(p, d, now);
  assert.ok(r.value > 10);
  assert.equal(r.cost, null);
  assert.equal(r.missingCost, 1);
  assert.equal(JSON.stringify(p), before);
  const ledger = { ...p, ledger: {}, positions: [{ ...pos, basisBase: 123 }] };
  assert.equal(publicValuation(ledger, d, now).cost, 123);
});
test("one missing mark withholds totals and retains per-position evidence", () => {
  const p = {
    currency: "USD",
    cash: 10,
    positions: [pos, { ...pos, symbol: "MISSING" }],
  };
  const r = publicValuation(p, dataset(), now);
  assert.equal(r.value, null);
  assert.equal(r.missing, 1);
  assert.ok(r.rows[0].value > 0);
  assert.equal(r.rows[1].value, null);
});
test("risk rejects insufficient/gapped history and manual mode", () => {
  const d = dataset(),
    p = { currency: "USD", cash: 0, positions: [pos], quoteMode: "published" };
  assert.match(portfolioRisk(p, d, now).reason, /gap/);
  assert.equal(
    portfolioRisk({ ...p, quoteMode: "manual" }, d, now).available,
    false,
  );
  d.instruments.A = stock("A", d.instruments.A.history.slice(-10));
  assert.equal(portfolioRisk(p, d, now).available, false);
});
test("fixed-weight model handles correlations, benchmark beta, cash drag and loss paths", () => {
  const h = history(112);
  const end = h.at(-1).date,
    clock = new Date(end + "T23:00:00Z");
  const d = {
    schema_version: 1,
    instruments: { A: stock("A", h), B: stock("B", h), SPY: stock("SPY", h) },
    fx: {},
  };
  const p = {
    currency: "USD",
    cash: 0,
    quoteMode: "published",
    positions: [pos],
  };
  const r = portfolioRisk(p, d, clock);
  assert.equal(r.available, true);
  assert.equal(r.observations, 111);
  assert.ok(Math.abs(r.beta - 1) < 1e-10);
  assert.ok(r.volatility > 0);
  assert.ok(r.drawdown < 0);
  assert.ok(Math.abs(r.modelReturn - r.benchmarkReturn) < 1e-10);
  const two = portfolioRisk(
    { ...p, positions: [pos, { ...pos, symbol: "B" }] },
    d,
    clock,
  );
  assert.ok(Math.abs(two.pairs[0].correlation - 1) < 1e-10);
  const cash = portfolioRisk(
    { ...p, cash: pos.quantity * h.at(-1).close },
    d,
    clock,
  );
  assert.ok(Math.abs(cash.beta - 0.5) < 1e-10);
  assert.ok(Math.abs(cash.volatility / r.volatility - 0.5) < 1e-10);
});
test("flat series yield zero volatility and undefined beta, not artificial scores", () => {
  const h = history(70).map((r) => ({ ...r, close: 100, adjusted_close: 100 })),
    clock = new Date(h.at(-1).date + "T23:00:00Z");
  const d = {
    schema_version: 1,
    instruments: { A: stock("A", h), SPY: stock("SPY", h) },
    fx: {},
  };
  const r = portfolioRisk(
    { currency: "USD", cash: 0, quoteMode: "published", positions: [pos] },
    d,
    clock,
  );
  assert.equal(r.volatility, 0);
  assert.equal(r.drawdown, 0);
  assert.equal(r.beta, null);
});
test("Wilder RSI / ATR and EMA MACD have deterministic constant and rising cases", () => {
  const flat = history(70).map((r) => ({
    ...r,
    open: 100,
    close: 100,
    high: 102,
    low: 98,
  }));
  const r = stockIndicators(stock("A", flat));
  assert.equal(r.rsi, 50);
  assert.equal(r.macd, 0);
  assert.equal(r.macdSignal, 0);
  assert.ok(Math.abs(r.atr - 4) < 1e-10);
  assert.equal(r.ma200, null);
  const up = history(70).map((r, i) => ({
    ...r,
    open: 100 + i,
    close: 100 + i,
    high: 102 + i,
    low: 98 + i,
  }));
  const u = stockIndicators(stock("A", up));
  assert.equal(u.rsi, 100);
  assert.ok(u.macd > 0);
  assert.equal(stockIndicators(stock("A", up.slice(-20))).available, false);
});
test("unordered or invalid calendar observations and unverified FX units are rejected", () => {
  const d = dataset();
  d.instruments.A.history[0].date = "2026-02-31";
  assert.equal(publicMark(d, pos, "USD", now).value, null);
  const f = dataset();
  f.fx.GBP.currency = "EUR";
  assert.equal(fxRate(f, "USD", "GBP", "2026-11-03").rate, null);
});
test("current FX freshness is checked against today as well as against the price date", () => {
  const d = dataset();
  d.instruments.A.history.at(-1).date = "2026-10-30";
  d.instruments.A.last_session = "2026-10-30";
  d.fx.GBP.history.at(-1).date = "2026-10-26";
  const m = publicMark(d, pos, "GBP", new Date("2026-11-03T23:00:00Z"));
  assert.equal(m.value, null);
  assert.match(m.reason, /FX observation/);
});
test("explicit listing exchange mismatches are withheld while known MIC/provider aliases match", () => {
  const d = dataset();
  d.instruments.A.exchange = "NMS";
  assert.ok(publicMark(d, { ...pos, exchange: "XNAS" }, "GBP", now).value > 0);
  const bad = publicMark(d, { ...pos, exchange: "LSE" }, "GBP", now);
  assert.equal(bad.value, null);
  assert.match(bad.reason, /exchange requires review/);
});

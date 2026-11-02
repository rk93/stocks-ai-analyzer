import { currency } from "./financial-domain.mjs";

export const LEDGER_VERSION = 1;
const EPS = 1e-8;
const symbolPattern = /^[A-Z0-9^][A-Z0-9.^=:/_-]{0,29}$/;
const types = [
  "BUY",
  "SELL",
  "DEPOSIT",
  "WITHDRAWAL",
  "DIVIDEND",
  "INTEREST",
  "FEE",
  "TAX",
];
export const transactionType = (value) => {
  const v = String(value).trim().toLowerCase();
  if (/^(?:market |limit |stop |stop limit )?buy$/.test(v)) return "BUY";
  if (/^(?:market |limit |stop |stop limit )?sell$/.test(v)) return "SELL";
  if (/^dividend(?: \([^()]+\))?$/.test(v)) return "DIVIDEND";
  return (
    {
      deposit: "DEPOSIT",
      withdrawal: "WITHDRAWAL",
      withdraw: "WITHDRAWAL",
      interest: "INTEREST",
      "interest on cash": "INTEREST",
      fee: "FEE",
      fees: "FEE",
      tax: "TAX",
      "withholding tax": "TAX",
    }[v] || null
  );
};
export const quoteCurrency = (value) => {
  const v = String(value).trim();
  return v === "GBp" ? "GBX" : v.toUpperCase();
};
function decimal(raw, label, { optional = false } = {}) {
  const v = String(raw ?? "").trim();
  if (!v && optional) return null;
  if (
    !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(v) ||
    !Number.isFinite(Number(v))
  )
    throw Error(`${label} must be an unambiguous decimal (e.g. 1234.56)`);
  return Number(v);
}
export function transactionTime(value) {
  const v = String(value).trim();
  const m =
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(\.\d{1,3})?(Z|[+-]\d{2}:\d{2})?$/.exec(
      v,
    );
  if (!m)
    throw Error(
      "Time must include YYYY-MM-DD HH:mm:ss; undated or date-only events cannot be ordered reliably",
    );
  const [year, month, day, hour, minute, second] = m.slice(1, 7).map(Number);
  const days = new Date(Date.UTC(year, month, 0)).getUTCDate();
  if (
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day > days ||
    hour > 23 ||
    minute > 59 ||
    second > 59
  )
    throw Error("Invalid calendar date or clock time");
  const ts = Date.parse(v.replace(" ", "T") + (m[8] ? "" : "Z"));
  if (!Number.isFinite(ts)) throw Error("Invalid timestamp");
  return new Date(ts).toISOString();
}
const core = (e) => ({
  type: e.type,
  time: e.time,
  symbol: e.symbol,
  exchange: e.exchange,
  quantity: e.quantity,
  unitPrice: e.unitPrice,
  quoteCurrency: e.quoteCurrency,
  amount: e.amount,
  settlementCurrency: e.settlementCurrency,
  reportedFees: e.reportedFees,
  brokerId: e.brokerId,
});
const canonical = (e) => JSON.stringify(core(e));
export const eventKey = (e) =>
  e.brokerId ? `broker:${e.type}:${e.brokerId}` : `row:${canonical(e)}`;
export function validateEvent(e, base) {
  if (
    !e ||
    !types.includes(e.type) ||
    !currency(base) ||
    base === "GBX" ||
    e.settlementCurrency !== base ||
    !Number.isFinite(e.amount) ||
    e.amount < 0 ||
    typeof e.brokerId !== "string" ||
    typeof e.exchange !== "string" ||
    typeof e.symbol !== "string" ||
    (e.symbol !== "" && !symbolPattern.test(e.symbol)) ||
    !Number.isFinite(e.reportedFees) ||
    e.reportedFees < 0
  )
    throw Error("Invalid ledger transaction or settlement currency");
  if (transactionTime(e.time) !== e.time)
    throw Error("Ledger timestamps must be canonical UTC");
  if (
    ["BUY", "SELL"].includes(e.type) &&
    (!symbolPattern.test(e.symbol) ||
      !currency(e.quoteCurrency) ||
      !Number.isFinite(e.quantity) ||
      e.quantity <= 0 ||
      !Number.isFinite(e.unitPrice) ||
      e.unitPrice <= 0 ||
      e.amount <= 0)
  )
    throw Error(
      "Trade requires symbol, positive shares, quote currency and non-negative price",
    );
  if (
    !["BUY", "SELL"].includes(e.type) &&
    (e.quantity !== null || e.unitPrice !== null || e.quoteCurrency !== null)
  )
    throw Error("Cash events cannot alter shares");
  if (e.key !== eventKey(e)) throw Error("Ledger event identity mismatch");
  return e;
}
export function parseTransactions(parsed, map, base = "GBP") {
  const events = [],
    errors = [],
    warnings = [];
  if (!currency(base) || base === "GBX")
    return {
      events,
      errors: [
        "Choose a supported settlement currency in whole currency units.",
      ],
      warnings,
    };
  for (const [i, row] of parsed.rows.entries()) {
    try {
      const get = (k) => (map[k] >= 0 ? (row[map[k]] || "").trim() : "");
      if (row.length !== parsed.headers.length)
        throw Error("Column count does not match header");
      const type = transactionType(get("type"));
      if (!type)
        throw Error(
          `Unsupported action “${get("type")}”. Transfers, FX conversions, reversals and corporate actions require separate reconciliation`,
        );
      const settlement = quoteCurrency(get("settlementCurrency"));
      if (
        !currency(settlement) ||
        (settlement !== base && !(settlement === "GBX" && base === "GBP"))
      )
        throw Error(
          `Net settlement currency ${settlement || "(missing)"} differs from ${base}; no FX rate will be inferred`,
        );
      const raw = decimal(get("amount"), "Net settlement total");
      if (raw < 0 && ["SELL", "DEPOSIT", "DIVIDEND", "INTEREST"].includes(type))
        throw Error(
          "Negative credit amount may be a reversal; reconcile this row separately",
        );
      const amount = Math.abs(raw) * (settlement === "GBX" ? 0.01 : 1),
        trade = ["BUY", "SELL"].includes(type);
      const fees =
        decimal(get("fees"), "Reported fees", { optional: true }) ?? 0;
      if (fees < 0)
        throw Error("Negative fees may be a refund; reconcile separately");
      const e = {
        type,
        time: transactionTime(get("date")),
        symbol: get("symbol").toUpperCase(),
        exchange: get("exchange"),
        quantity: trade ? decimal(get("quantity"), "Quantity") : null,
        unitPrice: trade ? decimal(get("cost"), "Trade price per share") : null,
        quoteCurrency: trade ? quoteCurrency(get("currency")) : null,
        amount,
        settlementCurrency: base,
        reportedFees: fees,
        brokerId: get("brokerId"),
        sourceRow: i + 2,
      };
      e.key = eventKey(e);
      validateEvent(e, base);
      if (
        trade &&
        (e.quoteCurrency === base ||
          (e.quoteCurrency === "GBX" && base === "GBP"))
      ) {
        const gross =
            e.quantity * e.unitPrice * (e.quoteCurrency === "GBX" ? 0.01 : 1),
          tolerance = Math.max(0.02, Math.abs(gross) * 1e-10);
        if (
          (type === "BUY" && amount < gross - tolerance) ||
          (type === "SELL" && amount > gross + tolerance)
        )
          throw Error(
            "Net total is inconsistent with shares × price and trade direction. Check quantity, price units, and whether Total is booked cash rather than Result",
          );
      }
      events.push(e);
    } catch (e) {
      errors.push(`Row ${i + 2}: ${e.message}`);
    }
  }
  if (events.some((e) => !e.brokerId))
    warnings.push(
      "Some records have no broker ID. Identical rows cannot be distinguished from separate identical transactions; any duplicate without an ID blocks the import.",
    );
  warnings.push(
    "Net settlement totals are authoritative and must already include trade fees and withholding taxes. Mapped fees are shown for reference and are not deducted a second time.",
  );
  return { events, errors, warnings };
}
export function mergeEvents(existing, incoming, base) {
  const events = existing.map((e) => ({ ...e })),
    byKey = new Map(),
    errors = [];
  let duplicates = 0,
    added = 0;
  for (const e of events) {
    validateEvent(e, base);
    if (byKey.has(e.key)) throw Error("Duplicate stored ledger identity");
    byKey.set(e.key, e);
  }
  for (const e of incoming) {
    validateEvent(e, base);
    const prior = byKey.get(e.key);
    if (prior) {
      if (!e.brokerId)
        errors.push(
          `Row ${e.sourceRow}: identical transaction without a broker ID is ambiguous. Map unique broker IDs or rebuild a separate full-history portfolio.`,
        );
      else if (canonical(prior) !== canonical(e))
        errors.push(
          `Row ${e.sourceRow}: broker ID ${e.brokerId} conflicts with an existing record; no historical record will be overwritten`,
        );
      else duplicates++;
    } else {
      events.push({ ...e });
      byKey.set(e.key, e);
      added++;
    }
  }
  return { events, errors, duplicates, added };
}
export function replayLedger(ledger) {
  if (
    ledger?.version !== LEDGER_VERSION ||
    ledger.method !== "FIFO_NET_SETTLEMENT" ||
    ledger.completeFromZero !== true ||
    !currency(ledger.currency) ||
    ledger.currency === "GBX" ||
    !Number.isFinite(ledger.openingCash) ||
    ledger.openingCash < 0 ||
    !Array.isArray(ledger.transactions)
  )
    throw Error("Unsupported or incomplete ledger");
  let cash = ledger.openingCash,
    realizedPnl = 0,
    dividends = 0,
    interest = 0,
    fees = 0,
    taxes = 0,
    netContributions = ledger.openingCash;
  const lots = new Map(),
    seen = new Set(),
    tradeTimes = new Set(),
    closedLots = [],
    activity = [];
  const events = ledger.transactions
    .map((e, i) => ({ ...validateEvent(e, ledger.currency), sequence: i }))
    .sort((a, b) => a.time.localeCompare(b.time) || a.sequence - b.sequence);
  for (const e of events) {
    const row = `Row ${e.sourceRow || "?"} (${e.time})`;
    if (seen.has(e.key)) throw Error(`${row}: duplicate event identity`);
    seen.add(e.key);
    const key = JSON.stringify([e.symbol, e.quoteCurrency, e.exchange]);
    if (["BUY", "SELL"].includes(e.type)) {
      const timing = JSON.stringify([e.symbol, e.time]);
      if (tradeTimes.has(timing))
        throw Error(
          `${row}: multiple trades in ${e.symbol} at the same timestamp have ambiguous FIFO ordering`,
        );
      tradeTimes.add(timing);
      if (!lots.has(key)) lots.set(key, []);
      const position = lots.get(key);
      if (e.type === "BUY") {
        position.push({
          quantity: e.quantity,
          unitNative: e.unitPrice,
          baseCost: e.amount,
        });
        cash -= e.amount;
      } else {
        const available = position.reduce((n, l) => n + l.quantity, 0),
          qeps = Number.EPSILON * Math.max(available, e.quantity) * 8;
        if (e.quantity > available + qeps)
          throw Error(
            `${row}: selling ${e.quantity} ${e.symbol} exceeds ${available} available shares. Import complete history from zero holdings; shorts and share transfers are unsupported`,
          );
        let remaining =
            Math.abs(e.quantity - available) <= qeps ? available : e.quantity,
          basis = 0;
        while (remaining > 0 && position.length) {
          const lot = position[0],
            take = Math.min(remaining, lot.quantity),
            allocated = lot.baseCost * (take / lot.quantity);
          basis += allocated;
          lot.baseCost -= allocated;
          lot.quantity -= take;
          remaining -= take;
          if (lot.quantity === 0) position.shift();
        }
        const pnl = e.amount - basis;
        realizedPnl += pnl;
        cash += e.amount;
        closedLots.push({
          symbol: e.symbol,
          time: e.time,
          quantity: e.quantity,
          proceeds: e.amount,
          basis,
          pnl,
        });
      }
    } else if (e.type === "DEPOSIT") {
      cash += e.amount;
      netContributions += e.amount;
    } else if (e.type === "WITHDRAWAL") {
      cash -= e.amount;
      netContributions -= e.amount;
    } else if (e.type === "DIVIDEND") {
      cash += e.amount;
      dividends += e.amount;
    } else if (e.type === "INTEREST") {
      cash += e.amount;
      interest += e.amount;
    } else if (e.type === "FEE") {
      cash -= e.amount;
      fees += e.amount;
    } else if (e.type === "TAX") {
      cash -= e.amount;
      taxes += e.amount;
    }
    if (cash < -EPS)
      throw Error(
        `${row}: cash would become ${cash.toFixed(2)} ${ledger.currency}. Add verified opening cash or include missing deposits; margin is unsupported`,
      );
    if (!Number.isFinite(cash))
      throw Error(`${row}: cash exceeds numeric limits`);
    if (cash < 0 && cash >= -EPS) cash = 0;
    activity.push({ ...e, cashAfter: cash });
  }
  const positions = [];
  for (const [key, position] of lots) {
    const quantity = position.reduce((n, l) => n + l.quantity, 0);
    if (quantity <= 0) continue;
    const [symbol, quoteCurrency, exchange] = JSON.parse(key);
    positions.push({
      symbol,
      exchange,
      currency: quoteCurrency,
      quantity,
      cost:
        position.reduce((n, l) => n + l.quantity * l.unitNative, 0) / quantity,
      basisBase: position.reduce((n, l) => n + l.baseCost, 0),
      price: null,
      asOf: null,
    });
  }
  // Different venues/quote currencies for the same symbol are not silently combined.
  return {
    positions,
    cash,
    realizedPnl,
    dividends,
    interest,
    fees,
    taxes,
    netContributions,
    closedLots,
    activity,
  };
}
export function validateLedgerPortfolio(p) {
  const r = replayLedger(p.ledger),
    close = (a, b) =>
      Number.isFinite(a) &&
      Number.isFinite(b) &&
      Math.abs(a - b) <=
        Number.EPSILON *
          Math.max(Math.abs(a), Math.abs(b), Number.MIN_VALUE) *
          16;
  if (
    p.currency !== p.ledger.currency ||
    !close(p.cash, r.cash) ||
    p.positions.length !== r.positions.length
  )
    throw Error("Portfolio does not reconcile with its transaction ledger");
  for (const expected of r.positions) {
    const stored = p.positions.find(
      (x) =>
        x.symbol === expected.symbol &&
        x.currency === expected.currency &&
        (x.exchange || "") === expected.exchange,
    );
    if (
      !stored ||
      !close(stored.quantity, expected.quantity) ||
      !close(stored.cost, expected.cost) ||
      !close(stored.basisBase, expected.basisBase)
    )
      throw Error(`Ledger holdings do not reconcile for ${expected.symbol}`);
  }
  return r;
}
export function reconcileHistory(
  parsed,
  map,
  { base = "GBP", openingCash = 0, existing = null } = {},
) {
  const parsedResult = parseTransactions(parsed, map, base),
    prior = existing?.ledger;
  const result = {
    ...parsedResult,
    positions: [],
    cash: null,
    duplicates: 0,
    added: 0,
    ledger: null,
    summary: null,
  };
  if (existing && !prior) {
    result.errors.push(
      "Only transaction-based portfolios can receive additional transaction imports. Create a new independent portfolio.",
    );
    return result;
  }
  if (prior && prior.currency !== base) {
    result.errors.push(
      "Settlement currency must match the destination ledger.",
    );
    return result;
  }
  if (result.errors.length) return result;
  try {
    const merged = mergeEvents(
      prior?.transactions || [],
      parsedResult.events,
      base,
    );
    result.errors.push(...merged.errors);
    result.duplicates = merged.duplicates;
    result.added = merged.added;
    if (result.errors.length) return result;
    const ledger = {
      version: LEDGER_VERSION,
      method: "FIFO_NET_SETTLEMENT",
      currency: base,
      openingCash: prior?.openingCash ?? openingCash,
      completeFromZero: true,
      transactions: merged.events,
    };
    const summary = replayLedger(ledger);
    result.ledger = ledger;
    result.summary = summary;
    result.cash = summary.cash;
    result.positions = summary.positions.map((r) => {
      const quote = existing?.positions.find(
        (p) =>
          p.symbol === r.symbol &&
          p.currency === r.currency &&
          (p.exchange || "") === r.exchange,
      );
      return quote ? { ...r, price: quote.price, asOf: quote.asOf } : r;
    });
  } catch (e) {
    result.errors.push(e.message);
  }
  return result;
}

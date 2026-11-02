import { currency, validatePosition } from "./financial-domain.mjs";
import { validateLedgerPortfolio } from "./accounting.mjs";
export { currency, validatePosition };
// Financial data stays in this origin's IndexedDB. No holdings are sent over the network.
export const SCHEMA = 2;
export const blank = () => ({
  schema: SCHEMA,
  revision: 0,
  portfolios: [],
  watchlist: [],
  journal: [],
  imports: [],
  preferences: { currency: "GBP", hidden: [] },
});
export function validateState(s) {
  if (s?.schema === 1)
    s = { ...structuredClone(s), schema: SCHEMA, revision: 0 };
  if (
    s?.schema !== SCHEMA ||
    !Number.isSafeInteger(s.revision) ||
    s.revision < 0 ||
    !Array.isArray(s.portfolios) ||
    !Array.isArray(s.watchlist) ||
    !Array.isArray(s.journal) ||
    !Array.isArray(s.imports) ||
    !s.preferences
  )
    throw Error("Unsupported or invalid backup schema");
  if (new Set(s.portfolios.map((p) => p.id)).size !== s.portfolios.length)
    throw Error("Duplicate portfolio IDs");
  for (const p of s.portfolios) {
    if (
      typeof p.id !== "string" ||
      typeof p.name !== "string" ||
      !Array.isArray(p.positions) ||
      !Number.isFinite(p.cash) ||
      p.cash < 0 ||
      !currency(p.currency)
    )
      throw Error("Invalid portfolio");
    for (const r of p.positions) validatePosition(r);
    if (p.ledger) validateLedgerPortfolio(p);
  }
  if (
    !Array.isArray(s.preferences.hidden) ||
    s.preferences.hidden.some(
      (x) => !["brief", "research", "strategy", "watchlist"].includes(x),
    )
  )
    throw Error("Invalid module preferences");
  if (
    s.preferences.order &&
    (!Array.isArray(s.preferences.order) ||
      s.preferences.order.length !== 4 ||
      new Set(s.preferences.order).size !== 4 ||
      s.preferences.order.some(
        (x) => !["brief", "research", "strategy", "watchlist"].includes(x),
      ))
  )
    throw Error("Invalid module order");
  if (
    s.watchlist.some(
      (x) => typeof x !== "string" || !/^[A-Z0-9.^=:/_-]{1,30}$/.test(x),
    )
  )
    throw Error("Invalid watchlist symbols");
  for (const j of s.journal)
    if (
      !j ||
      typeof j.symbol !== "string" ||
      typeof j.thesis !== "string" ||
      typeof j.date !== "string"
    )
      throw Error("Invalid journal record");
  return s;
}
export async function openStore() {
  return new Promise((resolve, reject) => {
    const q = indexedDB.open("stocks-research-private", SCHEMA);
    let blocked = false;
    q.onupgradeneeded = () => {
      if (!q.result.objectStoreNames.contains("state"))
        q.result.createObjectStore("state");
    };
    q.onblocked = () => {
      blocked = true;
      reject(
        Error(
          "An older workspace tab is blocking the storage upgrade. Close other Northstar tabs and reload; existing data is preserved.",
        ),
      );
    };
    q.onsuccess = () => {
      if (blocked) {
        q.result.close();
        return;
      }
      q.result.onversionchange = () => q.result.close();
      resolve(q.result);
    };
    q.onerror = () =>
      reject(
        Error(
          "Local database unavailable. Enable browser storage to save portfolios.",
        ),
      );
  });
}
export async function readStore(db) {
  return new Promise((resolve, reject) => {
    const r = db.transaction("state").objectStore("state").get("main");
    r.onsuccess = () => {
      try {
        resolve(r.result ? validateState(r.result) : blank());
      } catch (error) {
        reject(error);
      }
    };
    r.onerror = () => reject(r.error);
  });
}
export async function saveStore(db, s) {
  validateState(s);
  return new Promise((resolve, reject) => {
    const t = db.transaction("state", "readwrite"),
      store = t.objectStore("state");
    let saved,
      conflict = null;
    const read = store.get("main");
    read.onsuccess = () => {
      const revision = read.result?.revision ?? 0;
      if (revision !== s.revision) {
        conflict = Error(
          "Workspace changed in another tab. Reload before saving; no changes were written.",
        );
        t.abort();
        return;
      }
      saved = { ...s, revision: revision + 1 };
      store.put(saved, "main");
    };
    t.oncomplete = () => resolve(saved);
    t.onerror = () => reject(conflict || t.error);
    t.onabort = () => reject(conflict || t.error || Error("Save aborted"));
  });
}
export function parseCSV(text) {
  text = text.replace(/^\uFEFF/, "");
  const first = text.split(/\r?\n/)[0];
  const delimiter = [",", ";", "\t"].sort(
    (a, b) => first.split(b).length - first.split(a).length,
  )[0];
  const rows = [];
  let row = [],
    value = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '"') {
      if (quoted && text[i + 1] === '"') {
        value += '"';
        i++;
      } else if (quoted || value === "") quoted = !quoted;
      else throw Error("Unexpected quote near row " + (rows.length + 1));
    } else if (!quoted && (c === delimiter || c === "\n" || c === "\r")) {
      row.push(value.trim());
      value = "";
      if (c !== delimiter) {
        if (c === "\r" && text[i + 1] === "\n") i++;
        if (row.some(Boolean)) rows.push(row);
        row = [];
      }
    } else value += c;
  }
  if (quoted) throw Error("Unclosed quoted field");
  row.push(value.trim());
  if (row.some(Boolean)) rows.push(row);
  if (rows.length < 2)
    throw Error("CSV needs a header and at least one data row");
  const headers = rows.shift();
  if (
    new Set(headers.map((h) => h.trim().toLowerCase())).size !== headers.length
  )
    throw Error("Duplicate column names");
  return { headers, rows, delimiter };
}
export const aliases = {
  symbol: ["ticker", "symbol", "instrument"],
  quantity: ["no. of shares", "quantity", "shares", "qty"],
  currency: [
    "currency (price / share)",
    "currency",
    "currency (price)",
    "currency code",
  ],
  cost: [
    "average price",
    "average cost",
    "cost basis per share",
    "price / share",
    "cost",
  ],
  price: ["current price", "market price", "price"],
  type: ["action", "transaction type", "type"],
  date: ["time", "time (utc)", "date", "trade date"],
  fees: ["fees", "fee", "transaction fee"],
  amount: ["total", "net amount", "amount"],
  settlementCurrency: [
    "currency (total)",
    "settlement currency",
    "amount currency",
    "currency",
  ],
  brokerId: ["id", "transaction id", "trade id", "reference"],
  exchange: ["exchange", "market", "mic"],
};
export function detect(headers) {
  return Object.fromEntries(
    Object.entries(aliases).map(([k, names]) => [
      k,
      names
        .map((name) =>
          headers.findIndex((h) => h.trim().toLowerCase() === name),
        )
        .find((i) => i >= 0) ?? -1,
    ]),
  );
}
export function reconcile(parsed, map) {
  const errors = [],
    positions = [],
    events = [];
  let history = false;
  for (const [i, row] of parsed.rows.entries()) {
    const get = (k) => (map[k] >= 0 ? row[map[k]] || "" : "");
    const type = get("type").toLowerCase();
    if (row.length !== parsed.headers.length) {
      errors.push(`Row ${i + 2}: column count does not match header`);
      continue;
    }
    if (type) {
      history = true;
      if (/dividend|deposit|withdraw|interest|fee|buy|sell/.test(type)) {
        events.push({
          row: i + 2,
          type,
          symbol: get("symbol"),
          date: get("date"),
        });
      } else
        errors.push(
          `Row ${i + 2}: unsupported transaction type “${get("type")}”`,
        );
      continue;
    }
    try {
      const number = (k) => {
        const v = get(k);
        if (!v.trim()) throw Error(`Missing ${k}`);
        const n = Number(v);
        if (!Number.isFinite(n)) throw Error(`Invalid ${k}: ${v}`);
        return n;
      };
      if (get("date") && !Number.isFinite(Date.parse(get("date"))))
        throw Error("Invalid date");
      const r = validatePosition({
        symbol: get("symbol").toUpperCase(),
        quantity: number("quantity"),
        currency: get("currency"),
        cost: number("cost"),
        price: get("price") ? number("price") : null,
        exchange: get("exchange"),
        asOf: get("date") || null,
      });
      if (
        positions.some(
          (p) => p.symbol === r.symbol && p.currency === r.currency,
        )
      )
        throw Error(
          "Duplicate symbol/currency; consolidate lots before importing",
        );
      positions.push(r);
    } catch (e) {
      errors.push(`Row ${i + 2}: ${e.message}`);
    }
  }
  if (history)
    errors.push(
      "Transaction history detected. Trades and cash events are previewed separately, but cannot be committed as holdings: opening balances, corporate actions and complete history must first be reconciled. Export a positions snapshot instead.",
    );
  return { positions, errors, events, history };
}
export function valuation(p) {
  let value = p.cash,
    cost = 0,
    missing = 0,
    missingCost = 0;
  const rows = p.positions.map((r) => {
    const scale =
      r.currency === "GBX" && p.currency === "GBP"
        ? 0.01
        : r.currency === p.currency
          ? 1
          : null;
    const v =
      scale !== null && Number.isFinite(r.price)
        ? r.quantity * r.price * scale
        : null;
    const c =
      p.ledger && Number.isFinite(r.basisBase)
        ? r.basisBase
        : scale !== null
          ? r.quantity * r.cost * scale
          : null;
    if (v === null) missing++;
    else value += v;
    if (c === null) missingCost++;
    else cost += c;
    return { ...r, value: v, basis: c };
  });
  return {
    rows,
    value: missing ? null : value,
    cost: missingCost ? null : cost,
    missing,
    missingCost,
    cash: p.cash,
  };
}
export async function fingerprint(text) {
  return Array.from(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)),
    ),
  )
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
}

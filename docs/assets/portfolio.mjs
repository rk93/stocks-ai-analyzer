// Financial data stays in this origin's IndexedDB. No holdings are sent over the network.
export const SCHEMA = 1;
export const blank = () => ({
  schema: SCHEMA,
  portfolios: [],
  watchlist: [],
  journal: [],
  imports: [],
  preferences: { currency: "GBP", hidden: [] },
});
export function validateState(s) {
  if (
    s?.schema !== SCHEMA ||
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
export const currency = (c) =>
  /^[A-Z]{3}$/.test(c) &&
  (() => {
    try {
      new Intl.NumberFormat("en", { style: "currency", currency: c });
      return [
        "GBP",
        "USD",
        "EUR",
        "INR",
        "CAD",
        "AUD",
        "CHF",
        "JPY",
        "HKD",
        "SGD",
        "GBX",
      ].includes(c);
    } catch {
      return false;
    }
  })();
export function validatePosition(r) {
  if (
    !/^[A-Z0-9^][A-Z0-9.^=:/_-]{0,29}$/.test(r.symbol) ||
    !Number.isFinite(r.quantity) ||
    r.quantity <= 0 ||
    !currency(r.currency) ||
    !Number.isFinite(r.cost) ||
    r.cost < 0 ||
    !(r.price == null || (Number.isFinite(r.price) && r.price >= 0))
  )
    throw Error(
      "Check symbol, positive quantity, currency and non-negative unit cost/price",
    );
  return r;
}
export async function openStore() {
  return new Promise((resolve, reject) => {
    const q = indexedDB.open("stocks-research-private", SCHEMA);
    q.onupgradeneeded = () => q.result.createObjectStore("state");
    q.onsuccess = () => resolve(q.result);
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
    r.onsuccess = () => resolve(r.result ? validateState(r.result) : blank());
    r.onerror = () => reject(r.error);
  });
}
export async function saveStore(db, s) {
  validateState(s);
  return new Promise((resolve, reject) => {
    const t = db.transaction("state", "readwrite");
    t.objectStore("state").put(s, "main");
    t.oncomplete = resolve;
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || Error("Save aborted"));
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
  if (new Set(headers).size !== headers.length)
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
  date: ["time", "date", "trade date"],
  fees: ["fees", "fee", "transaction fee"],
};
export function detect(headers) {
  return Object.fromEntries(
    Object.entries(aliases).map(([k, names]) => [
      k,
      headers.findIndex((h) => names.includes(h.toLowerCase())),
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
        exchange: "",
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
    missing = 0;
  const rows = p.positions.map((r) => {
    const scale =
      r.currency === "GBX" && p.currency === "GBP"
        ? 0.01
        : r.currency === p.currency
          ? 1
          : null;
    const v =
      scale !== null && r.price !== null ? r.quantity * r.price * scale : null;
    const c = scale !== null ? r.quantity * r.cost * scale : null;
    if (v === null || c === null) missing++;
    if (v !== null) value += v;
    if (c !== null) cost += c;
    return { ...r, value: v, basis: c };
  });
  return {
    rows,
    value: missing ? null : value,
    cost: missing ? null : cost,
    missing,
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

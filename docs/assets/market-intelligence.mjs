// All matches and portfolio calculations run locally against a public fixed dataset.
const DAY = 86400000,
  MAX_AGE = 4;
const dayNumber = (d) => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(d || "")) return NaN;
  const t = Date.parse(d + "T00:00:00Z");
  return Number.isFinite(t) && new Date(t).toISOString().slice(0, 10) === d
    ? t / DAY
    : NaN;
};
export const normalizeCurrency = (c) =>
  c === "GBp" ? "GBX" : String(c || "").toUpperCase();
const units = (c) =>
  normalizeCurrency(c) === "GBX"
    ? { code: "GBP", scale: 0.01 }
    : { code: normalizeCurrency(c), scale: 1 };
const good = (n) => Number.isFinite(n) && n > 0;
function seriesRows(rows, field) {
  return (
    Array.isArray(rows) &&
    rows.length > 0 &&
    rows.every(
      (r, i) =>
        Number.isFinite(dayNumber(r.date)) &&
        good(r[field]) &&
        (!i || r.date > rows[i - 1].date),
    )
  );
}
export function fxRate(data, from, to, date) {
  const a = units(from),
    b = units(to);
  if (a.code === b.code)
    return {
      rate: a.scale / b.scale,
      dates: [],
      method: "same currency / explicit pence units",
    };
  if (data?.schema_version !== 1 || !Number.isFinite(dayNumber(date)))
    return { rate: null, reason: "FX dataset or valuation date unavailable" };
  const at = (c) => {
    if (c === "USD") return { rate: 1, date: null };
    const r = data.fx?.[c];
    if (
      !["ok", "retained_after_error"].includes(r?.status) ||
      r.unit_currency !== c ||
      r.currency !== "USD" ||
      !seriesRows(r.history, "usd_per_unit")
    )
      return null;
    const row = r.history.findLast((x) => x.date <= date);
    if (!row || dayNumber(date) - dayNumber(row.date) > MAX_AGE) return null;
    return {
      rate: row.usd_per_unit,
      date: row.date,
      retained: r.status === "retained_after_error",
    };
  };
  const x = at(a.code),
    y = at(b.code);
  if (!x || !y)
    return {
      rate: null,
      reason:
        "Missing or older-than-four-day FX observation at or before price date",
    };
  return {
    rate: ((x.rate / y.rate) * a.scale) / b.scale,
    dates: [x.date, y.date].filter(Boolean),
    retained: x.retained || y.retained,
    method: "USD-per-unit cross rate; no future FX",
  };
}
export function canonicalExchange(value) {
  const v = String(value || "")
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "");
  return (
    {
      NMS: "NASDAQ",
      NGM: "NASDAQ",
      NCM: "NASDAQ",
      XNAS: "NASDAQ",
      NASDAQGS: "NASDAQ",
      NASDAQGM: "NASDAQ",
      NASDAQCM: "NASDAQ",
      NASDAQGLOBALSELECTMARKET: "NASDAQ",
      NYQ: "NYSE",
      XNYS: "NYSE",
      NEWYORKSTOCKEXCHANGE: "NYSE",
      PCX: "ARCA",
      NYSEARCA: "ARCA",
      ARCX: "ARCA",
      XLON: "LSE",
      LONDON: "LSE",
      LONDONSTOCKEXCHANGE: "LSE",
      NSI: "NSE",
      XNSE: "NSE",
    }[v] || v
  );
}
export function publicMark(data, position, base, now = new Date()) {
  const r = data?.instruments?.[position.symbol],
    today = now.toISOString().slice(0, 10);
  if (
    data?.schema_version !== 1 ||
    !r ||
    !["ok", "retained_after_error"].includes(r.status) ||
    r.symbol !== position.symbol
  )
    return {
      value: null,
      reason: "Instrument absent or unavailable in the public universe",
    };
  if (
    position.exchange &&
    (!r.exchange ||
      canonicalExchange(position.exchange) !== canonicalExchange(r.exchange))
  )
    return {
      value: null,
      reason: `Listing exchange requires review: position ${position.exchange}, provider ${r.exchange || "unknown"}`,
    };
  if (normalizeCurrency(r.currency) !== normalizeCurrency(position.currency))
    return {
      value: null,
      reason: `Quote currency mismatch: position ${position.currency}, provider ${r.currency || "unknown"}`,
    };
  if (!seriesRows(r.history, "close"))
    return { value: null, reason: "Invalid or unordered price history" };
  const q = r.history.at(-1),
    age = dayNumber(today) - dayNumber(q.date);
  if (age < 0 || age > MAX_AGE)
    return {
      value: null,
      reason: "Future-dated or older-than-four-day closing price",
      date: q.date,
    };
  const fx = fxRate(data, r.currency, base, q.date);
  if (fx.rate === null)
    return { value: null, reason: fx.reason, date: q.date, price: q.close };
  if (fx.dates.some((date) => dayNumber(today) - dayNumber(date) > MAX_AGE))
    return {
      value: null,
      reason:
        "Current valuation FX observation is older than four calendar days",
      date: q.date,
      price: q.close,
      fxDates: fx.dates,
    };
  const value = position.quantity * q.close * fx.rate;
  if (!Number.isFinite(value))
    return {
      value: null,
      reason: "Position exceeds numeric calculation limits",
    };
  return {
    value,
    price: q.close,
    date: q.date,
    fxRate: fx.rate,
    fxDates: fx.dates,
    retained: r.status === "retained_after_error" || fx.retained,
    source: "published completed-session close",
    reason: null,
  };
}
export function publicValuation(portfolio, data, now = new Date()) {
  let total = portfolio.cash,
    cost = 0,
    missing = 0,
    missingCost = 0;
  const rows = portfolio.positions.map((r) => {
    const mark = publicMark(data, r, portfolio.currency, now),
      same = units(r.currency).code === units(portfolio.currency).code;
    const basis =
      portfolio.ledger && Number.isFinite(r.basisBase)
        ? r.basisBase
        : same
          ? (r.quantity * r.cost * units(r.currency).scale) /
            units(portfolio.currency).scale
          : null;
    if (mark.value === null) missing++;
    else total += mark.value;
    if (basis === null) missingCost++;
    else cost += basis;
    return { ...r, value: mark.value, basis, mark };
  });
  return {
    rows,
    value: missing ? null : total,
    cost: missingCost ? null : cost,
    missing,
    missingCost,
    cash: portfolio.cash,
    mode: "published",
  };
}
function mean(x) {
  return x.reduce((a, b) => a + b, 0) / x.length;
}
function covariance(a, b) {
  const ma = mean(a),
    mb = mean(b);
  return a.reduce((n, v, i) => n + (v - ma) * (b[i] - mb), 0) / (a.length - 1);
}
function correlation(a, b) {
  const den = Math.sqrt(covariance(a, a) * covariance(b, b));
  return den > 0 ? covariance(a, b) / den : null;
}
function historyInBase(data, symbol, base) {
  const r = data?.instruments?.[symbol];
  if (
    !r ||
    !["ok", "retained_after_error"].includes(r.status) ||
    !seriesRows(r.history, "close")
  )
    return null;
  const m = new Map();
  for (const bar of r.history) {
    if (!good(bar.adjusted_close)) continue;
    const fx = fxRate(data, r.currency, base, bar.date);
    if (fx.rate !== null) m.set(bar.date, bar.adjusted_close * fx.rate);
  }
  return m;
}
export function portfolioRisk(p, data, now = new Date()) {
  const v = publicValuation(p, data, now);
  if (p.quoteMode !== "published")
    return {
      available: false,
      reason:
        "Choose published closing prices to model current allocation risk.",
    };
  if (v.value === null || v.value <= 0 || !p.positions.length)
    return {
      available: false,
      reason:
        "Complete fresh price and FX coverage is required for every holding.",
    };
  const maps = p.positions.map((r) =>
    historyInBase(data, r.symbol, p.currency),
  );
  if (maps.some((m) => !m?.size))
    return {
      available: false,
      reason: "Adjusted-close or historical FX coverage is missing.",
    };
  const dates = [...maps[0].keys()]
    .filter((d) => maps.every((m) => m.has(d)))
    .sort();
  const points = [];
  for (let i = 1; i < dates.length; i++) {
    if (dayNumber(dates[i]) - dayNumber(dates[i - 1]) > 7) continue;
    points.push({
      date: dates[i],
      prior: dates[i - 1],
      returns: maps.map((m) => m.get(dates[i]) / m.get(dates[i - 1]) - 1),
    });
  }
  if (points.length < 60)
    return {
      available: false,
      reason: `Only ${points.length} aligned observations; at least 60 are required.`,
      observations: points.length,
    };
  // Missing price/FX history cannot be silently omitted from the model path.
  if (points.length !== dates.length - 1)
    return {
      available: false,
      reason: "A gap greater than seven days interrupts aligned history.",
    };
  const weights = v.rows.map((r) => r.value / v.value),
    assetReturns = maps.map((_, i) => points.map((p) => p.returns[i]));
  const returns = points.map((p) =>
    p.returns.reduce((n, r, i) => n + r * weights[i], 0),
  );
  let level = 100,
    peak = 100,
    drawdown = 0;
  const equity = [100];
  for (const r of returns) {
    level *= 1 + r;
    equity.push(level);
    peak = Math.max(peak, level);
    drawdown = Math.min(drawdown, level / peak - 1);
  }
  const pairs = [];
  for (let i = 0; i < maps.length; i++)
    for (let j = i + 1; j < maps.length; j++)
      pairs.push({
        a: p.positions[i].symbol,
        b: p.positions[j].symbol,
        correlation: correlation(assetReturns[i], assetReturns[j]),
      });
  const benchmark = historyInBase(data, "SPY", p.currency);
  let beta = null,
    benchmarkReturn = null;
  if (benchmark && dates.every((d) => benchmark.has(d))) {
    const br = points.map(
        (p) => benchmark.get(p.date) / benchmark.get(p.prior) - 1,
      ),
      variance = covariance(br, br);
    if (variance > 0) beta = covariance(returns, br) / variance;
    benchmarkReturn = benchmark.get(dates.at(-1)) / benchmark.get(dates[0]) - 1;
  }
  return {
    available: true,
    observations: points.length,
    start: dates[0],
    end: dates.at(-1),
    volatility:
      Math.sqrt(Math.max(0, covariance(returns, returns))) * Math.sqrt(252),
    drawdown,
    beta,
    benchmarkReturn,
    modelReturn: level / 100 - 1,
    equity,
    pairs: pairs.sort((a, b) => (b.correlation ?? -2) - (a.correlation ?? -2)),
    weights,
    methodology:
      "Historical hypothetical model: current weights rebalanced each aligned session, cash earns zero, provider-adjusted closes and historical FX, no trading costs. Not actual portfolio performance. 252 aligned observations/year is an approximation; cross-market closes are non-synchronous.",
  };
}
export function stockIndicators(instrument, now = new Date()) {
  const h = instrument?.history;
  if (
    !Array.isArray(h) ||
    h.length < 51 ||
    !seriesRows(h, "close") ||
    h.some((b) => !good(b.high) || !good(b.low) || b.high < b.low)
  )
    return {
      available: false,
      reason: "At least 51 valid daily OHLC observations are needed.",
    };
  if (h.at(-1).date > now.toISOString().slice(0, 10))
    return {
      available: false,
      reason: "Future-dated price history cannot support indicators.",
    };
  const c = h.map((b) => b.close),
    ema = (values, n) => {
      let value = values[0];
      return values.map(
        (v, i) => (value = i ? value + (2 / (n + 1)) * (v - value) : v),
      );
    };
  let gain = 0,
    loss = 0,
    atr = 0;
  for (let i = 1; i < c.length; i++) {
    const d = c[i] - c[i - 1],
      tr = Math.max(
        h[i].high - h[i].low,
        Math.abs(h[i].high - c[i - 1]),
        Math.abs(h[i].low - c[i - 1]),
      );
    if (i <= 14) {
      gain += Math.max(d, 0) / 14;
      loss += Math.max(-d, 0) / 14;
      atr += tr / 14;
    } else {
      gain = (gain * 13 + Math.max(d, 0)) / 14;
      loss = (loss * 13 + Math.max(-d, 0)) / 14;
      atr = (atr * 13 + tr) / 14;
    }
  }
  const e12 = ema(c, 12),
    e26 = ema(c, 26),
    macd = e12.map((v, i) => v - e26[i]),
    signal = ema(macd, 9),
    rsi = loss === 0 ? (gain === 0 ? 50 : 100) : 100 - 100 / (1 + gain / loss);
  return {
    available: true,
    date: h.at(-1).date,
    rsi,
    macd: macd.at(-1),
    macdSignal: signal.at(-1),
    atr,
    ma20: mean(c.slice(-20)),
    ma50: mean(c.slice(-50)),
    ma200: c.length >= 200 ? mean(c.slice(-200)) : null,
    methodology:
      "Daily close indicators; Wilder RSI/ATR 14, EMA MACD 12/26/9 seeded from first available close. Finite history affects EMA warm-up; not a trading recommendation.",
  };
}

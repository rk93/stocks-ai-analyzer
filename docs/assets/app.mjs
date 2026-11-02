import { reconcileHistory, replayLedger } from "./accounting.mjs";
import {
  blank,
  openStore,
  readStore,
  saveStore,
  validateState,
  validatePosition,
  parseCSV,
  detect,
  reconcile,
  valuation,
  fingerprint,
} from "./portfolio.mjs";
const $ = (s) => document.querySelector(s),
  esc = (x) =>
    String(x ?? "—").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    ),
  pct = (x) => (x == null ? "—" : (Number(x) * 100).toFixed(1) + "%"),
  num = (x) =>
    x == null
      ? "—"
      : Number(x).toLocaleString("en-GB", { maximumFractionDigits: 2 }),
  money = (x, c = "GBP") =>
    x == null
      ? "Unavailable"
      : new Intl.NumberFormat("en-GB", {
          style: "currency",
          currency: c,
        }).format(x),
  id = () => crypto.randomUUID();
let importEpoch = 0;
let storageError = "";
let stockRange = "all",
  db,
  state = blank(),
  data = {},
  errors = [],
  selected = "",
  route = "overview",
  query = "",
  sector = "",
  sort = "score",
  horizon = "1d",
  importData = null;
const titles = {
  overview: "Overview",
  discover: "Discover",
  portfolio: "My Portfolio",
  lab: "Strategy Lab",
  insights: "Insights",
};
const toast = (m) => {
  const t = $("#toast");
  t.textContent = m;
  t.style.display = "block";
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.style.display = "none"), 5000);
};
async function commit(fn) {
  const next = structuredClone(state);
  try {
    fn(next);
    next.revision = state.revision;
    if (!db) throw Error("Storage unavailable; changes cannot be saved");
    state = await saveStore(db, next);
    render();
    return true;
  } catch (e) {
    toast(e.message);
    return false;
  }
}
const stocks = () => data.trending?.stocks || [],
  signals = () => data.skew?.signals || [],
  find = (s) => ({
    ...stocks().find((r) => r.symbol === s),
    ...signals().find((r) => r.symbol === s),
    symbol: s,
  }),
  ticker = (s) =>
    `<button class="ticker" data-stock="${esc(s)}">${esc(s)} ↗</button>`,
  selectedP = () =>
    state.portfolios.find((p) => p.id === selected) ||
    state.portfolios.find(
      (p) => p.id === state.preferences.activePortfolioId,
    ) ||
    state.portfolios[0];
const panel = (title, body, extra = "") =>
  `<section class="panel"><div class="section-head"><h2>${title}</h2>${extra}</div>${body}</section>`;
function chart(series, labels = []) {
  const colors = ["#b9ef8a", "#7abbd2", "#c1a0ef"];
  const vals = series.flat().filter(Number.isFinite);
  if (!vals.length)
    return '<p class="empty">No chart observations available.</p>';
  let min = Math.min(...vals),
    max = Math.max(...vals);
  if (max === min) {
    max++;
    min--;
  }
  return `<svg viewBox="0 0 700 220" role="img" aria-label="Historical observations. ${esc(labels.join(", "))}. Vertical scale ${num(min)} to ${num(max)}"><text x="2" y="15" fill="currentColor" font-size="10">${num(max)}</text><text x="2" y="207" fill="currentColor" font-size="10">${num(min)}</text>${[40, 100, 160, 210].map((y) => `<path d="M45 ${y}H690" stroke="var(--border)"/>`).join("")}${series.map((s, j) => `<polyline fill="none" stroke="${colors[j % 3]}" stroke-width="2.5" points="${s.map((v, i) => (Number.isFinite(v) ? `${45 + (i / Math.max(1, s.length - 1)) * 640},${205 - ((v - min) / (max - min)) * 180}` : "")).join(" ")}"/>`).join("")}</svg><div class="legend">${labels.map((l, i) => `<span style="color:${colors[i % 3]}">● ${esc(l)}</span>`).join("")}</div>`;
}
function head(title, sub, actions = "") {
  return `<div class="page-head"><div><h1>${title}</h1><p>${sub}</p></div><div class="actions">${actions}</div></div>`;
}
function freshness() {
  return `<div class="notice">${data.manifest?.datasets?.trending?.stale_publication ? "STALE PUBLICATION · " : ""}Published snapshot: ${esc(data.trending?.generated_at || "unavailable")}. This is scheduled research, not a live quote feed. ${errors.length ? `Unavailable datasets: ${esc(errors.join(", "))}. <button data-action="retry">Retry</button>` : ""}</div>`;
}
function summaryCards(p) {
  const v = valuation(p);
  return `<div class="cards">${panel("Portfolio value", `<div class="metric">${money(v.value, p.currency)}</div><small>Cash ${money(p.cash, p.currency)} · ${v.rows.length} positions</small>`)}${panel("Cost basis", `<div class="metric">${money(v.cost, p.currency)}</div><small>${p.ledger ? "Remaining FIFO book basis from net settlements" : "Position cost, excluding unreported fees"}</small>`)}${panel("Unrealised P/L", `<div class="metric ${v.value !== null && v.value - p.cash - v.cost >= 0 ? "positive" : "negative"}">${money(v.value === null ? null : v.value - p.cash - v.cost, p.currency)}</div><small>Based on your entered prices · not live</small>`)}</div>${v.missing ? `<div class="notice">${v.missing} positions have missing prices or unsupported currency conversion. Portfolio totals and weights are withheld. Enter prices in the position currency; no FX rates are assumed.</div>` : ""}`;
}
function market() {
  const m = data.trending?.context?.market || [];
  return `<div class="market">${m.map((r) => `<div><small>${esc(r.label)} <span class="tag">SNAPSHOT</span></small><strong>${num(r.value)} <small class="${r.change_1d >= 0 ? "positive" : "negative"}">${pct(r.change_1d)}</small></strong></div>`).join("") || "<div>Market snapshot unavailable</div>"}</div>`;
}
function researchRows(rows) {
  return (
    rows
      .map(
        (r) =>
          `<div class="list-row"><span class="symbol-icon">${esc(r.symbol?.slice(0, 3))}</span><div class="grow">${ticker(r.symbol)}<p>${esc(r.sector || "Sector unavailable")}</p></div><div class="right"><span class="tag">${esc(r.research_state || "Monitor")}</span><p class="sub">1M ${pct(r.return_1m)}</p></div><button data-watch="${esc(r.symbol)}" aria-label="Toggle watchlist ${esc(r.symbol)}">${state.watchlist.includes(r.symbol) ? "★" : "☆"}</button></div>`,
      )
      .join("") ||
    '<p class="empty">No published candidates. Explore the full market universe.</p>'
  );
}
function overview() {
  const p = selectedP(),
    hidden = state.preferences.hidden || [],
    modules = {
      brief: panel(
        "Your research briefing",
        `<div class="list-row"><span class="tag">01</span><div><b>Strategy evidence needs caution</b><p>V1, V2 and V3 have negative expectancy in the published 1R replay. Review assumptions before experimenting.</p></div></div><div class="list-row"><span class="tag">02</span><div><b>${p ? esc(p.name) + " is stored privately" : "Make the research yours"}</b><p>${p ? "Review holdings, concentration and missing price coverage in My Portfolio." : "Import a positions CSV or explore a clearly labelled demo. No account needed."}</p></div></div><a class="tag" href="#insights">Read all insights ↗</a>`,
      ),
      research: panel(
        "On the research radar",
        researchRows(
          signals()
            .filter((r) =>
              ["ADD CANDIDATE", "EARLY CANDIDATE"].includes(r.research_state),
            )
            .slice(0, 5),
        ),
        `<a class="sub" href="#discover">Explore all ↗</a>`,
      ),
      strategy: panel(
        "Evidence before conviction",
        `<p class="muted">Latest published 1R expectancy · before execution costs</p>${Object.entries(
          data.backtest?.summary?.strategies || {},
        )
          .map(
            ([k, v]) =>
              `<div class="list-row"><b class="grow">${k}</b><span class="negative">${v.expectancy_r?.toFixed(3)}R</span><small>${v.resolved_1r} resolved</small></div>`,
          )
          .join(
            "",
          )}<p class="notice">Attractive signals do not establish profitability. This short replay is not independent out-of-sample validation.</p><a class="tag" href="#lab">Open Strategy Lab ↗</a>`,
      ),
      watchlist: panel(
        "Your watchlist",
        state.watchlist.length
          ? researchRows(state.watchlist.slice(0, 6).map(find))
          : '<p class="empty">Star a symbol in Discover to build your private watchlist.</p>',
      ),
    };
  return (
    head(
      "A clearer view of your investments.",
      "Your portfolio. Transparent research. Better-informed decisions.",
      `<span class="tag">● PRIVATE WORKSPACE</span><button data-action="customize">Customise</button>`,
    ) +
    market() +
    (!p
      ? `<section class="banner"><div><span class="eyebrow">RESEARCH THAT STARTS WITH YOU</span><h2>Put your portfolio in perspective.</h2><p>Connect the market to what you own. Explore allocation, understand concentration and keep your investment thesis in view.</p></div><div class="actions"><button class="primary" data-action="import">↑ Upload portfolio</button><button data-action="demo">Try demo ↗</button><a class="tag" href="#discover">Explore markets</a></div></section>`
      : summaryCards(p)) +
    `<div class="two-col">${(state.preferences.order || Object.keys(modules))
      .filter((k) => !hidden.includes(k))
      .map((k) => modules[k] || "")
      .join("")}</div>` +
    freshness()
  );
}
function discover() {
  let rows = stocks()
    .map((r) => ({ ...r, ...signals().find((s) => s.symbol === r.symbol) }))
    .filter(
      (r) =>
        (!query ||
          (r.symbol + " " + (r.sector || ""))
            .toLowerCase()
            .includes(query.toLowerCase())) &&
        (!sector || r.sector === sector),
    );
  rows.sort((a, b) =>
    sort === "symbol"
      ? a.symbol.localeCompare(b.symbol)
      : (b[sort === "score" ? "technical_score" : "return_1m"] ?? -Infinity) -
        (a[sort === "score" ? "technical_score" : "return_1m"] ?? -Infinity),
  );
  return (
    head(
      "Follow the evidence.",
      "Screen the published universe. Open any symbol to inspect its research.",
    ) +
    `<div class="tools"><input id="filter" aria-label="Search symbol or sector" placeholder="Search symbol or sector…" value="${esc(query)}"><select id="sector" aria-label="Sector"><option value="">All sectors</option>${[
      ...new Set(
        stocks()
          .map((r) => r.sector)
          .filter(Boolean),
      ),
    ]
      .sort()
      .map(
        (s) => `<option ${sector === s ? "selected" : ""}>${esc(s)}</option>`,
      )
      .join("")}</select><select id="sort" aria-label="Sort results">${[
      ["score", "Technical score"],
      ["return", "1M return"],
      ["symbol", "Symbol"],
    ]
      .map(
        ([v, l]) =>
          `<option value="${v}" ${sort === v ? "selected" : ""}>${l}</option>`,
      )
      .join("")}</select></div>` +
    panel(
      `${rows.length} research names`,
      `<div class="scroll"><table><thead><tr><th>Company / symbol</th><th>Snapshot price</th><th>1M return</th><th>Technical score</th><th>Research state</th><th>Watchlist</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${ticker(r.symbol)}<p class="muted">${esc(r.sector)}</p></td><td>${num(r.spot)}<p class="muted">Quote currency unverified</p></td><td class="${r.return_1m >= 0 ? "positive" : "negative"}">${pct(r.return_1m)}</td><td>${num(r.technical_score)} / 100</td><td><span class="tag">${esc(r.research_state || "Monitor")}</span></td><td><button data-watch="${esc(r.symbol)}" aria-label="Toggle watchlist ${esc(r.symbol)}">${state.watchlist.includes(r.symbol) ? "★" : "☆"}</button></td></tr>`).join("")}</tbody></table></div>`,
    ) +
    freshness()
  );
}
function health(p, v) {
  const alerts = [];
  if (v.value === null)
    alerts.push(
      "Insufficient Data: complete prices and FX coverage to calculate concentration.",
    );
  else
    for (const r of v.rows) {
      if (r.value / v.value > 0.25)
        alerts.push(
          `${r.symbol}: ${pct(r.value / v.value)} of total value exceeds the 25% concentration review threshold.`,
        );
    }
  const currencies = [...new Set(p.positions.map((r) => r.currency))];
  if (currencies.some((c) => c !== p.currency))
    alerts.push(
      "Review Risk: multiple quote currencies. Quote currency is not the same as underlying economic exposure.",
    );
  for (const r of p.positions) {
    const s = find(r.symbol);
    if (s.catalyst_flag)
      alerts.push(
        `${r.symbol}: published event risk, ${s.days_to_earnings ?? "unknown"} days to earnings at snapshot time.`,
      );
    if (s.return_1m < 0 && s.normalized_skew > 0)
      alerts.push(
        `${r.symbol}: price weakness and relatively expensive put protection; review downside risk.`,
      );
  }
  return (
    alerts.map((a) => `<p class="notice">${esc(a)}</p>`).join("") ||
    '<p class="muted">No threshold alerts from available data. This does not establish that the portfolio is low risk.</p>'
  );
}
function portfolio() {
  const p = selectedP();
  return (
    head(
      "Your portfolio, in focus.",
      "Private on this device. No login. No financial uploads to a server.",
      `<button data-action="new">+ New portfolio</button><button class="primary" data-action="import">↑ Import CSV</button><button data-action="backup">Export backup</button><button data-action="restore">Restore backup</button>`,
    ) +
    (!p
      ? panel(
          "Start with what you own",
          '<div class="empty"><p>Import a broker positions snapshot, add holdings manually, or explore a sample.</p><button data-action="demo">Try labelled demo</button></div>',
        )
      : (() => {
          const v = valuation(p);
          const total = v.value;
          return (
            `<div class="tools"><select id="portfolio-select" aria-label="Selected portfolio">${state.portfolios.map((r) => `<option value="${esc(r.id)}" ${r.id === p.id ? "selected" : ""}>${esc(r.name)}${r.demo ? " · DEMO" : ""}</option>`).join("")}</select>${p.ledger ? "" : '<button data-action="position">+ Add position</button>'}<button data-action="edit-portfolio">${p.ledger ? "Edit name" : "Edit name, currency & cash"}</button><button data-action="delete-portfolio">Delete portfolio</button></div>` +
            (p.demo
              ? '<p class="notice">Sample portfolio · fictional quantities, costs and prices. Not your holdings and not live market data.</p>'
              : "") +
            summaryCards(p) +
            ledgerPanel(p) +
            `<div class="two-col">${panel("Allocation by holding", v.rows.map((r) => `<div class="list-row"><div class="grow">${ticker(r.symbol)}<div class="bar"><span style="width:${total && r.value !== null ? Math.max(0, Math.min(100, (r.value / total) * 100)) : 0}%"></span></div></div><span>${total && r.value !== null ? pct(r.value / total) : "Unavailable"}</span></div>`).join("") + '<p class="sub">Weights include cash in the denominator. Missing FX prevents aggregate weights.</p>')}${panel("Portfolio health", health(p, v))}</div>` +
            allocationPanel(p, v) +
            panel(
              "Your positions",
              `<div class="scroll"><table><thead><tr><th>Symbol</th><th>Quantity</th><th>${p.ledger ? "Average native entry (excl. settlement charges)" : "Unit cost"}</th><th>Entered price</th><th>Value (${esc(p.currency)})</th><th>Research</th><th>Manage</th></tr></thead><tbody>${v.rows.map((r, i) => `<tr><td>${ticker(r.symbol)}<p>${esc(r.currency)}</p></td><td>${num(r.quantity)}</td><td>${num(r.cost)}</td><td>${num(r.price)}<p class="muted">${esc(r.asOf || "Date not provided")}</p></td><td>${money(r.value, p.currency)}</td><td>${esc(find(r.symbol).research_state || "Insufficient Data")}</td><td><button data-edit="${i}">Edit</button> ${p.ledger ? "" : `<button data-remove="${i}" aria-label="Remove ${esc(r.symbol)}">×</button>`}</td></tr>`).join("")}</tbody></table></div>`,
            ) +
            `<div class="two-col">${panel("What if you add more?", `<p class="muted">Model a new contribution in ${esc(p.currency)}, held as cash or allocated to an existing position. Entered prices remain unchanged; fees and future returns are excluded.</p><label>Additional cash<input id="what-amount" type="number" min="0" value="500"></label><label>Hypothetical allocation<select id="what-symbol"><option value="">Hold as cash</option>${p.positions.map((r, i) => `<option value="${i}">${esc(r.symbol)}</option>`).join("")}</select></label><button data-action="what">Calculate allocation scenario</button><p id="what-result" role="status"></p>`)}${panel("Coverage & calculation limits", '<p class="muted">Transaction-based portfolios show booked FIFO realised P/L and net income. Position-only snapshots lack this history. Market total return, daily movement, beta, volatility, drawdown, correlations and ETF overlap still require verified historical prices, FX or constituent data; unavailable metrics are not estimated.</p>')}</div>`
          );
        })()) +
    `<div class="notice">Privacy: holdings, watchlists and journal entries are saved in this browser’s IndexedDB. Backups contain sensitive financial data. Clearing browser storage removes your workspace. Published public research is fetched from this site only. <button data-action="reset">Remove all local data</button></div>`
  );
}
function allocationPanel(p, v) {
  const groups = {};
  for (const r of v.rows) {
    const sector = find(r.symbol).sector || "Unclassified";
    groups[sector] = (groups[sector] || 0) + (r.value || 0);
  }
  return panel(
    "Sector exposure · published classifications",
    v.value === null
      ? '<p class="muted">Complete price and FX coverage to calculate sector weights.</p>'
      : Object.entries(groups)
          .map(
            ([k, a]) =>
              `<div class="list-row"><span class="grow">${esc(k)}<div class="bar"><span style="width:${v.value ? (100 * a) / v.value : 0}%"></span></div></span><span>${v.value ? pct(a / v.value) : "0%"}</span></div>`,
          )
          .join("") +
          '<p class="sub">Cash is included in the total denominator. Unknown sectors remain unclassified. ETF underlying constituents are not inferred.</p>',
  );
}
function lab() {
  const s = data.backtest?.summary;
  if (!s)
    return head(
      "Strategy Lab",
      "Research dataset unavailable. Retry from Overview.",
    );
  return (
    head(
      "Test conviction against evidence.",
      "Historical baselines preserved. Experimental assumptions are never production signals.",
      `<a class="tag" href="paper-trading.html">Paper trading ↗</a><a class="tag" href="backtest.html">Detailed replay ↗</a>`,
    ) +
    `<div class="notice">Coverage ${esc(s.coverage_start)} to ${esc(s.coverage_end)} · ${esc(s.method)}. ${esc(s.note)} All three baselines have negative 1R expectancy before costs. No validated profitable strategy is established.</div>` +
    panel(
      "Immutable baseline comparison",
      `<div class="scroll"><table><thead><tr><th>Version</th><th>Signals / resolved</th><th>1R win rate</th><th>Expectancy</th><th>Profit factor</th><th>95% win interval*</th></tr></thead><tbody>${Object.entries(
        s.strategies,
      )
        .map(([k, v]) => {
          const n = v.resolved_1r,
            p = v.win_rate_1r,
            z = 1.96,
            d = 1 + (z * z) / n,
            c = (p + (z * z) / (2 * n)) / d,
            w = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
          return `<tr><td>${k}</td><td>${v.signals} / ${n}</td><td>${pct(p)}</td><td class="negative">${v.expectancy_r.toFixed(3)}R</td><td>${v.profit_factor_r.toFixed(2)}</td><td>${pct(c - w)} – ${pct(c + w)}</td></tr>`;
        })
        .join(
          "",
        )}</tbody></table></div><p class="sub">*Wilson binomial interval assumes independent outcomes. Correlated symbols and dates reduce effective sample size. This is descriptive uncertainty, not proof of an edge.</p>`,
    ) +
    `<div class="tools"><label>Future price horizon<select id="horizon">${["1d", "5d", "15d", "30d", "3m"].map((h) => `<option ${horizon === h ? "selected" : ""}>${h}</option>`).join("")}</select></label></div>` +
    panel(
      "Signal follow-through · not executable P/L",
      `<div class="cards">${Object.entries(s.horizons_by_strategy || {})
        .map(([k, v]) =>
          panel(
            k,
            `<div class="metric">${pct(v[horizon]?.avg_return)}</div><p>Mean future-price return</p><small>${v[horizon]?.resolved ?? 0} resolved · ${v[horizon]?.pending ?? 0} pending<br>Median ${pct(v[horizon]?.median_return)}</small>`,
          ),
        )
        .join("")}</div>`,
    ) +
    `<div class="two-col">${panel(
      "Published paper equity",
      chart(
        ["V1", "V2", "V3"].map((k) =>
          (data["paper-trading"]?.equity || []).map((r) => r[k + "_equity"]),
        ),
        ["V1", "V2", "V3"],
      ) +
        '<p class="sub">Chronological ledger observations; spacing represents observations, not elapsed time. See detailed paper trading for costs and execution assumptions.</p>',
    )}${panel("Cost sensitivity", `<p class="muted">Subtract a hypothetical round-trip cost expressed in R from each resolved trade. This sensitivity is not an execution backtest.</p><label>Round-trip cost (R)<input id="cost-r" type="number" min="0" step="0.01" value="0.05"></label><button data-action="cost">Calculate sensitivity</button><div id="cost-result" role="status"></div>`)}</div>` +
    diagnosticPanel() +
    paperPanel() +
    panel(
      "Individual replay outcomes",
      `<div class="scroll"><table><thead><tr><th>Signal date</th><th>Symbol</th><th>Version</th><th>1R outcome</th><th>R</th><th>Rationale</th></tr></thead><tbody>${(
        data.backtest.trades || []
      )
        .slice(-40)
        .reverse()
        .map(
          (r) =>
            `<tr><td>${esc(r.signal_date)}</td><td>${ticker(r.symbol)}</td><td>${esc(r.strategy)}</td><td>${esc(r.trade_result_1r)}</td><td>${num(r.realized_r_1r)}</td><td>${esc(r.reason)}</td></tr>`,
        )
        .join("")}</tbody></table></div>`,
    ) +
    panel(
      "Research gates before another version",
      '<p class="muted">Freeze V1–V3. Record hypotheses before testing. Use date-separated development and untouched test windows, date-block uncertainty, next-bar execution, pessimistic ambiguous fills, explicit costs and point-in-time universes. Regime, earnings, options divergence and relative-strength experiments need archived features; current snapshots cannot supply historical evidence.</p>',
    )
  );
}
function diagnosticPanel() {
  const d = data["research-diagnostics"];
  if (!d)
    return panel(
      "Research diagnostics",
      "<p>Diagnostics unavailable. No experimental conclusions can be drawn.</p>",
    );
  return panel(
    "Chronological diagnostics · descriptive only",
    `<p class="muted">Date split ${esc(d.split_date)}. Assumed round-trip cost ${num(d.cost_assumption_r)}R. Existing data was already observed during development; later dates are not an untouched test set.</p><div class="scroll"><table><thead><tr><th>Version</th><th>Earlier dates expectancy</th><th>Later dates expectancy</th><th>After-cost date-block 95% interval</th></tr></thead><tbody>${Object.entries(
      d.strategies,
    )
      .map(
        ([k, v]) =>
          `<tr><td>${k}</td><td>${num(v.earlier_70pct_dates.expectancy_after_assumed_cost_r)}R (${v.earlier_70pct_dates.resolved})</td><td>${num(v.later_30pct_dates.expectancy_after_assumed_cost_r)}R (${v.later_30pct_dates.resolved})</td><td>${v.all.date_block_bootstrap_95_interval_r?.map((x) => num(x) + "R").join(" to ") || "Insufficient dates"}</td></tr>`,
      )
      .join(
        "",
      )}</tbody></table></div><p class="notice">The exploratory time-normalised volume filter remains a hypothesis. No V4 is promoted. Complete limitations and results: <a href="data/research-diagnostics.json">download diagnostics</a>.</p>`,
  );
}
function paperPanel() {
  const p = data["paper-trading"]?.summary;
  if (!p) return "";
  return panel(
    "Paper ledger · simulated capital",
    `<p class="muted">${esc(p.generated_at)} · ${num(p.slippage_bps)} bps published slippage assumption. Stale prices may affect marked equity. No live orders are created by this interface.</p><div class="scroll"><table><thead><tr><th>Version</th><th>Equity (USD)</th><th>Return</th><th>Realised P/L</th><th>Max drawdown</th><th>Closed / open</th></tr></thead><tbody>${Object.entries(
      p.strategies || {},
    )
      .map(
        ([k, v]) =>
          `<tr><td>${k}</td><td>${money(v.equity, "USD")}</td><td>${pct(v.return)}</td><td>${money(v.realized_pnl, "USD")}</td><td>${pct(v.max_drawdown)}</td><td>${v.closed_trades} / ${v.open_positions}</td></tr>`,
      )
      .join(
        "",
      )}</tbody></table></div><p class="sub">Published benchmark returns: ${Object.entries(
      p.benchmarks || {},
    )
      .map(([k, v]) => k + " " + pct(v.return))
      .join(
        " · ",
      )}. Strategy and benchmark cashflows/exposure differ.</p><h3>Open simulated positions</h3>${Object.entries(
      p.strategies || {},
    )
      .flatMap(([k, v]) =>
        (v.positions || []).map(
          (r) =>
            `<div class="list-row"><b>${esc(k)}</b><div class="grow">${ticker(r.symbol)}<p>${num(r.qty)} shares · entered ${esc(r.entry_time)} · last marked ${money(r.last_price, "USD")}</p></div><span class="tag">PAPER ONLY</span></div>`,
        ),
      )
      .join("")}`,
  );
}
function insights() {
  return (
    head(
      "Keep the reasoning in view.",
      "A decision journal, published predictions and evidence-backed context.",
      `<button class="primary" data-action="journal">+ Record a decision</button><a class="tag" href="prediction-tracker.html">Prediction tracker ↗</a>`,
    ) +
    `<div class="two-col">${panel("Decision journal", state.journal.map((j, i) => `<article class="list-row"><div class="grow">${ticker(j.symbol)} <span class="tag">${esc(j.action)}</span><p>${esc(j.date)}</p><p>${esc(j.thesis)}</p><p>Invalidation: ${esc(j.invalidation)}</p><p>Later observation: ${esc(j.outcome || "Not recorded")}</p></div><button data-journal="${i}">Review</button></article>`).join("") || '<div class="empty">Record your thesis and what would change your mind. Your entries stay private.</div>')}${panel("Research context", `<p>What changed?</p><p class="muted">The published snapshot is ${esc(data.trending?.generated_at || "unavailable")}. No real-time news claims are generated.</p><p>What matters to my portfolio?</p><p class="muted">${selectedP() ? `${selectedP().positions.length} holdings are available for matching against public research. Review freshness before using any signal.` : "Add a local portfolio for holdings-specific research."}</p><p>What deserves attention?</p><p class="muted">Negative baseline expectancy, missing data and concentration are reasons to investigate further.</p>`)}</div>` +
    panel(
      "Frozen predictions · observed close returns",
      `<div class="scroll"><table><thead><tr><th>Timestamp</th><th>Symbol</th><th>Strategy / original prediction</th><th>Original rationale</th><th>1D observed return</th></tr></thead><tbody>${(
        data["prediction-research"]?.predictions || []
      )
        .slice(0, 25)
        .map(
          (r) =>
            `<tr><td>${esc(r.signal_time)}</td><td>${ticker(r.symbol)}</td><td>${esc(r.strategy_version)} · ${esc(r.prediction)}</td><td>${esc(r.original_reason)}</td><td>${pct(r.observations?.find((o) => o.horizon_sessions === 1)?.close_return)}</td></tr>`,
        )
        .join("")}</tbody></table></div>`,
    ) +
    freshness()
  );
}
function render() {
  route = location.hash.slice(1) || "overview";
  if (!titles[route]) route = "overview";
  $("#crumb").textContent = titles[route];
  document.title = titles[route] + " · Northstar";
  document
    .querySelectorAll("nav a")
    .forEach((a) => a.classList.toggle("active", a.hash === "#" + route));
  $("#main").innerHTML =
    (storageError
      ? `<p class="notice error" role="alert">Local storage unavailable: ${esc(storageError)}. Saving is disabled to preserve any existing data.</p>`
      : "") + { overview, discover, portfolio, lab, insights }[route]();
}
function modal(html) {
  $("#modal-body").innerHTML = html;
  if (!$("#modal").open) $("#modal").showModal();
}
const close = () => $("#modal").close();
function stock(s) {
  const r = find(s),
    hist = (data.skew?.history || [])
      .filter((x) => x.symbol === s && x.normalized_skew != null)
      .sort((a, b) => String(a.run_date).localeCompare(String(b.run_date)))
      .filter(
        (x) =>
          stockRange === "all" ||
          Date.parse(x.run_date) >=
            Date.parse(data.skew?.generated_at) - Number(stockRange) * 86400000,
      );
  modal(
    `<span class="tag">STOCK INTELLIGENCE</span><h1>${esc(s)}</h1><p class="muted">${esc(r.sector || "Sector unavailable")} · published snapshot</p><div class="cards">${panel("Snapshot price", `<div class="metric">${num(r.spot)}</div><small>Quote currency not verified</small>`)}${panel("1M price return", `<div class="metric">${pct(r.return_1m)}</div>`)}${panel("Research state", `<span class="tag">${esc(r.research_state || "Insufficient Data")}</span>`)}</div><h2>Why is this stock highlighted?</h2><p>${esc(r.signal_reason || "This symbol is available in the research universe or your local workspace; no current entry classification is established.")}</p><div class="two-col">${panel("Supporting evidence", `<p>Technical score: ${num(r.technical_score)}</p><p>Above 50-day average: ${r.above_50d == null ? "Unavailable" : r.above_50d ? "Yes" : "No"}</p><p>Relative to SPY (1M): ${pct(r.return_vs_spy_1m)}</p><p>Options quality: ${esc(r.quality_label || "Unavailable")}</p>`)}${panel("Contradictions & invalidation", `<p>${r.catalyst_flag ? "Event distortion is flagged." : "Event coverage may be incomplete."}</p><p>Skew alone is not a directional forecast. Weak quote quality, stale observations or loss of trend support invalidate reliance on this snapshot.</p><p class="muted">No strategy here has established out-of-sample profitability.</p>`)}</div><h2>Published price observations</h2><div class="tools"><select id="stock-range" data-symbol="${esc(s)}" aria-label="Observation range">${[
      ["30", "Last 30 days"],
      ["90", "Last 90 days"],
      ["all", "All observations"],
    ]
      .map(
        ([v, l]) =>
          `<option value="${v}" ${stockRange === v ? "selected" : ""}>${l}</option>`,
      )
      .join(
        "",
      )}</select><button data-compare="${esc(s)}">Compare symbols</button></div>${chart([hist.filter((r) => Number.isFinite(r.spot)).map((r) => r.spot)], ["Snapshot price · not OHLC bars"])}<h2>Available fundamentals</h2><div class="cards">${panel("Revenue growth", `<div class="metric">${pct(r.revenue_growth)}</div>`)}${panel("Profit margin", `<div class="metric">${pct(r.profit_margin)}</div>`)}${panel("Earnings growth", `<div class="metric">${pct(r.earnings_growth)}</div>`)}</div><p class="sub">Fundamental reporting period and source timestamp are not published; verify against company filings.</p><h2>Observed options skew</h2>${chart([hist.map((r) => r.normalized_skew)], ["Normalised skew"])}<p class="sub">${esc(hist[0]?.run_date || "No history")} → ${esc(hist.at(-1)?.run_date || "")} · observations, not continuous time. Price history, RSI, MACD and ATR are unavailable in the published contract; no synthetic chart is substituted.</p><div class="actions"><button data-watch="${esc(s)}">${state.watchlist.includes(s) ? "Remove from" : "Add to"} watchlist</button><button data-journal-symbol="${esc(s)}">Record thesis</button></div>`,
  );
}
function portfolioForm() {
  const p = selectedP();
  modal(
    `<h2>${p ? "Edit" : "Create"} portfolio</h2><form id="portfolio-form"><label>Name<input name="name" required maxlength="80" value="${esc(p?.name || "My portfolio")}"></label><label>Base currency<select name="currency">${["GBP", "USD", "EUR", "INR", "CAD", "AUD", "CHF", "JPY", "HKD", "SGD"].map((c) => `<option ${p?.currency === c ? "selected" : ""}>${c}</option>`).join("")}</select></label><label>Cash<input name="cash" type="number" step="any" min="0" value="${p?.cash || 0}" required></label><button class="primary">Save portfolio</button></form>`,
  );
  if (p?.ledger) {
    $('#portfolio-form [name="currency"]').disabled = true;
    $('#portfolio-form [name="cash"]').readOnly = true;
  }
}
function positionForm(i) {
  const p = selectedP();
  if (!p) {
    toast("Create a portfolio first");
    return;
  }
  if (p.ledger && i == null) {
    toast(
      "Transaction-based holdings come from the ledger. Import more records to change shares.",
    );
    return;
  }
  const r = p.positions[i] || {};
  modal(
    `<h2>${i == null ? "Add" : "Edit"} position</h2><p class="muted">Use an exchange-qualified symbol when ambiguous (e.g. RR.L or RELIANCE.NS). Prices are entered manually.</p><form id="position-form" data-index="${i ?? ""}" class="grid-form">${[
      ["symbol", "Symbol", r.symbol || "", "text"],
      ["quantity", "Quantity", r.quantity ?? "", "number"],
      [
        "currency",
        "Quote currency (GBP, USD, GBX…)",
        r.currency || p.currency,
        "text",
      ],
      ["cost", "Cost per share", r.cost ?? "", "number"],
      ["price", "Current price (optional)", r.price ?? "", "number"],
      ["asOf", "Price date (optional)", r.asOf || "", "date"],
    ]
      .map(
        ([n, l, v, t]) =>
          `<label>${l}<input name="${n}" type="${t}" value="${esc(v)}" ${!["price", "asOf"].includes(n) ? "required" : ""} ${t === "number" ? 'step="any" min="0"' : ""}></label>`,
      )
      .join(
        "",
      )}<div class="full error" id="form-error"></div><button class="primary">Save position</button></form>`,
  );
  if (p.ledger) {
    for (const name of ["symbol", "quantity", "currency", "cost"])
      $('#position-form [name="' + name + '"]').readOnly = true;
    $("#position-form").insertAdjacentHTML(
      "afterbegin",
      '<p class="notice full">Shares and cost come from the ledger. Only quote and quote date can be changed here.</p>',
    );
  }
}
function importStart() {
  importEpoch++;
  importData = null;
  modal(
    `<span class="tag">PRIVATE IMPORT · STEP 1 OF 3</span><h2>Bring your holdings into focus.</h2><p class="muted">CSV positions snapshots or supported transaction histories from Trading 212 or another broker. Map and review before saving. Files stay in this browser. XLSX is not supported; export as CSV.</p><div class="drop" id="drop"><label>Choose or drop a CSV<input id="csv-file" type="file" accept=".csv,text/csv"></label></div><p class="notice">Transaction histories need net settlement totals, complete trade history from zero holdings and verified opening cash. Reconcile a new portfolio or append to a transaction-based portfolio with duplicate checks. Position snapshots remain separate.</p><div id="import-error" class="error" role="alert"></div>`,
  );
}
async function importFile(file) {
  const epoch = ++importEpoch;
  try {
    if (file.size > 5e6) throw Error("File exceeds 5 MB limit");
    const text = await file.text(),
      hash = await fingerprint(text);
    if (epoch !== importEpoch || !$("#modal").open) return;
    const parsed = parseCSV(text);
    importData = { parsed, hash, map: detect(parsed.headers) };
    if (state.imports.includes(hash)) {
      $("#import-error").innerHTML =
        'This exact file was already imported. Your existing portfolio is unchanged. <p>Use the existing portfolio, or review import options to inspect broker IDs and choose a destination. Independent portfolios are never aggregated.</p><button data-action="mapping">Review import options</button>';
      return;
    }
    mapping();
  } catch (e) {
    if (epoch === importEpoch && $("#import-error"))
      $("#import-error").textContent = e.message;
  }
}
function mapping() {
  const { parsed, map } = importData;
  modal(
    `<span class="tag">STEP 2 OF 3 · COLUMN MAPPING</span><h2>Check the detected columns.</h2><p class="muted">Delimiter ${parsed.delimiter === "\t" ? "tab" : esc(parsed.delimiter)} · ${parsed.rows.length} rows. For positions, cost is per share, not total lot cost. For transaction history, map cost to Price / share, amount to net Total, and settlement currency to Currency (Total). Correct ambiguous tickers in the preview.</p><div class="grid-form">${Object.keys(
      map,
    )
      .map(
        (k) =>
          `<label>${esc(k)}<select data-map="${k}"><option value="-1">Not provided</option>${parsed.headers.map((h, i) => `<option value="${i}" ${map[k] === i ? "selected" : ""}>${esc(h)}</option>`).join("")}</select></label>`,
      )
      .join(
        "",
      )}</div><h3>Raw preview (first 5 rows)</h3><div class="scroll"><table><thead><tr>${parsed.headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${parsed.rows
      .slice(0, 5)
      .map(
        (r, ri) =>
          `<tr>${r.map((v, ci) => `<td>${ci === map.symbol ? `<input data-correct="${ri}" aria-label="Correct ticker row ${ri + 2}" value="${esc(v)}">` : esc(v)}</td>`).join("")}</tr>`,
      )
      .join(
        "",
      )}</tbody></table></div><div class="grid-form"><label>Correct any CSV row number (header is row 1)<input id="correct-row-number" type="number" min="2" max="${parsed.rows.length + 1}" value="2"></label><label>Corrected exchange-qualified ticker<input id="correct-row-symbol" placeholder="e.g. RR.L"></label></div><button data-action="correct-row">Apply ticker correction</button><p class="sub">Fees are not included in per-share cost. Reconcile fee-inclusive basis in your broker export before importing.</p><button class="primary" data-action="reconcile">Review reconciliation</button>`,
  );
}
function importReview() {
  const action = importData.parsed.headers.findIndex(
    (h) => h.trim().toLowerCase() === "action",
  );
  if (
    (importData.map.type >= 0 &&
      importData.parsed.rows.some((r) => r[importData.map.type]?.trim())) ||
    (action >= 0 && importData.parsed.rows.some((r) => r[action]?.trim()))
  )
    return historySetup();
  importData.result = reconcile(importData.parsed, importData.map);
  const { positions, errors, events } = importData.result;
  modal(
    `<span class="tag">STEP 3 OF 3 · RECONCILIATION</span><h2>Review before saving.</h2><p>${positions.length} valid positions · ${events.length} transaction events · ${errors.length} errors</p><div class="error">${errors.map((e) => `<p>${esc(e)}</p>`).join("")}</div>${events
      .slice(0, 10)
      .map(
        (e) =>
          `<p class="sub">Row ${e.row}: ${esc(e.type)} · ${esc(e.symbol)} · ${esc(e.date)}</p>`,
      )
      .join(
        "",
      )}<div class="scroll"><table><thead><tr><th>Symbol</th><th>Quantity</th><th>Currency</th><th>Unit cost</th><th>Price</th></tr></thead><tbody>${positions.map((r) => `<tr><td>${esc(r.symbol)}</td><td>${num(r.quantity)}</td><td>${esc(r.currency)}</td><td>${num(r.cost)}</td><td>${num(r.price)}</td></tr>`).join("")}</tbody></table></div><label>New snapshot portfolio name<input id="import-name" value="Imported portfolio" maxlength="80"></label><p class="notice">A new independent snapshot will be created. Previous portfolios are preserved. Overlapping snapshots are never combined. Save a backup before deleting old snapshots.</p><button data-action="mapping">Back to mapping</button> <button class="primary" data-action="import-save" ${errors.length || !positions.length ? "disabled" : ""}>Save new portfolio</button>`,
  );
}
function historySetup() {
  const c = importData.historyConfig || {
    base: "GBP",
    openingCash: 0,
    destination: "",
  };
  modal(
    `<span class="tag">TRANSACTION IMPORT · RECONCILIATION ASSUMPTIONS</span><h2>Rebuild from booked transactions.</h2><p class="muted">Use a complete account history starting with zero shares. Trades use FIFO research cost basis in the settlement currency. This is not a jurisdiction-specific tax calculation.</p><label>Destination<select id="history-destination"><option value="">New transaction-based portfolio</option>${state.portfolios
      .filter((p) => p.ledger)
      .map(
        (p) =>
          `<option value="${esc(p.id)}" ${p.id === c.destination ? "selected" : ""}>${esc(p.name)}</option>`,
      )
      .join(
        "",
      )}</select></label><div class="grid-form"><label>Settlement currency<select id="history-currency">${["GBP", "USD", "EUR", "INR", "CAD", "AUD", "CHF", "JPY", "HKD", "SGD"].map((x) => `<option ${c.base === x ? "selected" : ""}>${x}</option>`).join("")}</select></label><label>Verified cash before the first record<input id="history-cash" type="number" min="0" step="any" value="${c.openingCash}"></label></div><p class="notice">Total / Amount must be the net booked cash amount, including trade fees and dividend taxes. Do not map the broker's “Result” (profit) as Total. Cross-currency trades work only when their actual settled cash total is recorded in this one settlement currency; exchange rates are never inferred. Share transfers, splits, FX conversions, shorts and margin are blocked.</p><label class="check"><input id="history-complete" type="checkbox">These records, plus the destination ledger if selected, cover all share activity from zero opening holdings for the same account.</label><label class="check"><input id="history-net" type="checkbox">The mapped totals are net booked amounts; fees/taxes are included once. Opening cash is verified, and timestamps without an offset are UTC.</label><div id="history-error" class="error" role="alert"></div><button data-action="mapping">Back to columns</button> <button class="primary" data-action="history-preview">Reconcile transactions</button>`,
  );
  if (c.destination) configureHistoryDestination();
}
function configureHistoryDestination() {
  const p = state.portfolios.find(
    (p) => p.id === $("#history-destination").value,
  );
  $("#history-currency").disabled = !!p;
  $("#history-cash").readOnly = !!p;
  if (p) {
    $("#history-currency").value = p.currency;
    $("#history-cash").value = p.ledger.openingCash;
  }
}
function historyResult() {
  const c = importData.historyConfig;
  return reconcileHistory(importData.parsed, importData.map, {
    base: c.base,
    openingCash: c.openingCash,
    existing: c.destination
      ? state.portfolios.find((p) => p.id === c.destination)
      : null,
  });
}
function historyPreview() {
  if (!$("#history-complete").checked || !$("#history-net").checked) {
    $("#history-error").textContent =
      "Confirm the source history, net totals and timestamp assumptions before reconciliation.";
    return;
  }
  const opening = $("#history-cash").value;
  if (
    opening.trim() === "" ||
    !Number.isFinite(Number(opening)) ||
    Number(opening) < 0
  ) {
    $("#history-error").textContent =
      "Enter verified non-negative opening cash.";
    return;
  }
  importData.historyConfig = {
    base: $("#history-currency").value,
    openingCash: Number(opening),
    destination: $("#history-destination").value,
  };
  const r = historyResult(),
    c = importData.historyConfig;
  importData.historyResult = r;
  modal(
    `<span class="tag">TRANSACTION IMPORT · REVIEW</span><h2>${r.errors.length ? "Resolve the reconciliation errors." : "Your history reconciles."}</h2><p>${r.added} new records · ${r.duplicates} exact broker-ID duplicates skipped · ${r.errors.length} errors</p><div class="error" role="alert">${r.errors.map((e) => `<p>${esc(e)}</p>`).join("")}</div>${r.warnings.map((w) => `<p class="notice">${esc(w)}</p>`).join("")}${r.summary ? `<div class="cards">${panel("Closing cash", `<div class="metric">${money(r.cash, c.base)}</div>`)}${panel("FIFO realised P/L", `<div class="metric">${money(r.summary.realizedPnl, c.base)}</div><small>Research accounting, not tax reporting</small>`)}${panel("Net dividends", `<div class="metric">${money(r.summary.dividends, c.base)}</div>`)}</div><div class="scroll"><table><thead><tr><th>Symbol</th><th>Shares</th><th>Quote currency</th><th>Remaining book cost (${esc(c.base)})</th></tr></thead><tbody>${r.positions.map((p) => `<tr><td>${esc(p.symbol)}</td><td>${num(p.quantity)}</td><td>${esc(p.currency)}</td><td>${money(p.basisBase, c.base)}</td></tr>`).join("")}</tbody></table></div><p class="notice">Quotes remain unpriced until you enter a current price. Replaying history does not turn trade prices into current market prices.</p>` : ""}${!c.destination ? '<label>Portfolio name<input id="history-name" value="Imported transaction history" maxlength="80"></label>' : "<p>New records will be appended to the selected local ledger; existing records remain unchanged.</p>"}<button data-action="history-setup">Back to assumptions</button> <button data-action="mapping">Back to columns</button> <button class="primary" data-action="history-save" ${r.errors.length || !r.ledger || !r.added ? "disabled" : ""}>${c.destination ? "Append reconciled records" : "Save reconciled portfolio"}</button>`,
  );
}
async function historySave() {
  const r = historyResult(),
    c = importData.historyConfig;
  if (r.errors.length || !r.ledger || !r.added) {
    toast("Reconciliation failed or no new records to save.");
    return;
  }
  const name =
    $("#history-name")?.value.trim() || "Imported transaction history";
  let pid = c.destination || id();
  if (
    await commit((s) => {
      const existing = c.destination
        ? s.portfolios.find((p) => p.id === c.destination)
        : null;
      if (c.destination && !existing)
        throw Error("Destination no longer exists. Reconcile again.");
      const p = {
        ...(existing || { id: pid, name }),
        currency: c.base,
        cash: r.cash,
        positions: r.positions,
        ledger: r.ledger,
        importedAt: new Date().toISOString(),
      };
      if (existing) s.portfolios.splice(s.portfolios.indexOf(existing), 1, p);
      else s.portfolios.push(p);
      s.preferences.activePortfolioId = pid;
      if (!s.imports.includes(importData.hash)) s.imports.push(importData.hash);
    })
  ) {
    selected = pid;
    close();
    location.hash = "portfolio";
    render();
    toast("Transactions reconciled and saved locally.");
  }
}
function ledgerPanel(p) {
  if (!p.ledger) return "";
  const r = replayLedger(p.ledger);
  return panel(
    "Transaction-based accounting",
    `<p class="notice">FIFO research accounting in ${esc(p.currency)}. Booked net settlements include costs once. This is not tax reporting. Shares, cash and book basis come from the ledger; only quotes can be edited directly.</p><div class="cards">${panel("Realised P/L", `<div class="metric">${money(r.realizedPnl, p.currency)}</div><small>${r.closedLots.length} sale records · FIFO</small>`)}${panel("Net dividends & interest", `<div class="metric">${money(r.dividends + r.interest, p.currency)}</div><small>Dividends ${money(r.dividends, p.currency)} · interest ${money(r.interest, p.currency)}</small>`)}${panel("Net contributions", `<div class="metric">${money(r.netContributions, p.currency)}</div><small>Includes opening cash, deposits less withdrawals</small>`)}</div><p class="sub">Standalone fees ${money(r.fees, p.currency)} · standalone taxes ${money(r.taxes, p.currency)}. Embedded charges remain in net trade/dividend totals.</p><button data-action="ledger">Review transaction ledger</button> <button data-action="import">Import more transactions</button>`,
  );
}
function ledgerDetails() {
  const p = selectedP();
  if (!p?.ledger) return;
  const r = replayLedger(p.ledger);
  modal(
    `<h2>${esc(p.name)} · transaction ledger</h2><p class="muted">${r.activity.length} records · FIFO net-settlement methodology · ${esc(p.currency)} cash account. Identical times retain file order for cash events; same-symbol trade ties are blocked.</p><div class="scroll"><table><thead><tr><th>UTC time</th><th>Action</th><th>Symbol</th><th>Shares</th><th>Net amount</th><th>Cash after</th><th>Broker ID</th><th>Source fee field*</th></tr></thead><tbody>${r.activity
      .slice()
      .reverse()
      .map(
        (e) =>
          `<tr><td>${esc(e.time)}</td><td>${esc(e.type)}</td><td>${esc(e.symbol || "—")}</td><td>${num(e.quantity)}</td><td>${money(e.amount, p.currency)}</td><td>${money(e.cashAfter, p.currency)}</td><td>${esc(e.brokerId || "Not provided")}</td><td>${num(e.reportedFees)}</td></tr>`,
      )
      .join(
        "",
      )}</tbody></table></div><h3>FIFO sale outcomes</h3><div class="scroll"><table><thead><tr><th>Symbol / UTC time</th><th>Shares</th><th>Net proceeds</th><th>Allocated cost</th><th>Realised P/L</th></tr></thead><tbody>${r.closedLots.map((e) => `<tr><td>${esc(e.symbol)}<p>${esc(e.time)}</p></td><td>${num(e.quantity)}</td><td>${money(e.proceeds, p.currency)}</td><td>${money(e.basis, p.currency)}</td><td>${money(e.pnl, p.currency)}</td></tr>`).join("")}</tbody></table></div><p class="notice">*Source fee fields retain their original units, which may differ from the settlement currency. They are not deducted twice. Average quote-currency entry prices exclude settlement-currency charges. Remaining book basis and realised P/L use actual settled amounts. No current FX conversion or tax-lot rule is inferred.</p>`,
  );
}
function journalForm(symbol = "", i = null) {
  const j = i === null ? {} : state.journal[i];
  modal(
    `<h2>Investment decision journal</h2><p class="sub">Original decisions are preserved on review; saving a review updates only the later observation.</p><form id="journal-form" data-index="${i ?? ""}"><label>Symbol<input name="symbol" required value="${esc(j.symbol || symbol)}"></label><label>Decision<select name="action">${["Watch", "Buy thesis", "Sell thesis", "Review risk"].map((a) => `<option ${j.action === a ? "selected" : ""}>${a}</option>`).join("")}</select></label><label>Original thesis<textarea name="thesis" required>${esc(j.thesis || "")}</textarea></label><label>What would invalidate it?<textarea name="invalidation" required>${esc(j.invalidation || "")}</textarea></label><label>Later observation / outcome<textarea name="outcome">${esc(j.outcome || "")}</textarea></label><button class="primary">Save privately</button></form>`,
  );
}
function compare(
  first,
  second = stocks().find((r) => r.symbol !== first)?.symbol,
) {
  const rows = [find(first), find(second)];
  modal(
    `<h2>Compare published evidence</h2><label>Compare ${esc(first)} with<select id="compare-second" data-first="${esc(first)}">${stocks()
      .filter((r) => r.symbol !== first)
      .map(
        (r) =>
          `<option ${r.symbol === second ? "selected" : ""}>${esc(r.symbol)}</option>`,
      )
      .join(
        "",
      )}</select></label><div class="scroll"><table><thead><tr><th>Measure</th>${rows.map((r) => `<th>${ticker(r.symbol)}</th>`).join("")}</tr></thead><tbody>${[
      ["1M return", "return_1m", pct],
      ["Relative to SPY", "return_vs_spy_1m", pct],
      ["Technical score", "technical_score", num],
      ["Revenue growth", "revenue_growth", pct],
      ["Profit margin", "profit_margin", pct],
      ["Options quality", "quality_label", esc],
      ["Normalised skew", "normalized_skew", num],
      ["Research state", "research_state", esc],
    ]
      .map(
        ([l, k, f]) =>
          `<tr><td>${l}</td>${rows.map((r) => `<td>${f(r[k])}</td>`).join("")}</tr>`,
      )
      .join(
        "",
      )}</tbody></table></div><p class="notice">Same published snapshot; missing values stay unavailable. Absolute skew between unrelated companies is not directly comparable. This comparison is not a portfolio recommendation.</p>`,
  );
}
const actions = {
  "history-preview": historyPreview,
  "history-setup": historySetup,
  "history-save": historySave,
  ledger: ledgerDetails,
  "correct-row": () => {
    const i = Number($("#correct-row-number").value) - 2,
      s = $("#correct-row-symbol").value.trim().toUpperCase();
    if (
      !Number.isInteger(i) ||
      !importData.parsed.rows[i] ||
      importData.map.symbol < 0 ||
      !/^[A-Z0-9.^=:/_-]{1,30}$/.test(s)
    ) {
      toast("Choose a valid data row, ticker column and symbol.");
      return;
    }
    importData.parsed.rows[i][importData.map.symbol] = s;
    mapping();
    toast("Ticker corrected in preview.");
  },
  import: importStart,
  demo: async () => {
    const p = {
      id: id(),
      name: "Sample · research portfolio",
      demo: true,
      currency: "GBP",
      cash: 500,
      positions: [
        {
          symbol: "MSFT",
          quantity: 3,
          currency: "GBP",
          cost: 300,
          price: 320,
          asOf: "2026-10-30",
        },
        {
          symbol: "RR.L",
          quantity: 100,
          currency: "GBX",
          cost: 1000,
          price: 1100,
          asOf: "2026-10-30",
        },
      ],
    };
    if (
      await commit((s) => {
        s.portfolios.push(p);
        s.preferences.activePortfolioId = p.id;
      })
    ) {
      selected = p.id;
      location.hash = "portfolio";
      render();
    }
  },
  new: () => {
    selected = "";
    modal(
      '<h2>Create portfolio</h2><form id="new-form"><label>Name<input name="name" required maxlength="80"></label><button class="primary">Create in GBP</button></form>',
    );
  },
  position: () => positionForm(),
  "edit-portfolio": portfolioForm,
  "delete-portfolio": () => {
    const p = selectedP();
    if (p)
      modal(
        `<h2>Delete ${esc(p.name)}?</h2><p>This removes this portfolio from this browser. Export a backup first if needed.</p><button data-action="confirm-delete">Delete portfolio</button>`,
      );
  },
  "confirm-delete": async () => {
    const p = selectedP();
    if (
      await commit(
        (s) => (s.portfolios = s.portfolios.filter((x) => x.id !== p.id)),
      )
    )
      close();
  },
  backup: () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(state, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "northstar-private-backup.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
  restore: () =>
    modal(
      '<h2>Restore a private backup</h2><p>Validated backups replace your whole local workspace. Export a backup of current data first.</p><input id="backup-file" type="file" accept=".json"><div class="error" id="backup-error"></div>',
    ),
  reset: () =>
    modal(
      '<h2>Remove all local data?</h2><p>All portfolios, import fingerprints, watchlists, preferences and journal entries will be removed from this browser.</p><button data-action="confirm-reset">Remove everything</button>',
    ),
  "confirm-reset": async () => {
    if (
      await commit((s) => {
        for (const k of Object.keys(s)) delete s[k];
        Object.assign(s, blank());
      })
    ) {
      close();
      selected = "";
      query = "";
      sector = "";
      sort = "score";
      document.body.classList.remove("light");
      render();
      toast("All local workspace data removed");
    }
  },
  journal: () => journalForm(),
  mapping,
  reconcile: importReview,
  "import-save": async () => {
    const result = reconcile(importData.parsed, importData.map);
    if (result.errors.length || !result.positions.length) return;
    const p = {
      id: id(),
      name: $("#import-name").value.trim() || "Imported portfolio",
      currency: "GBP",
      cash: 0,
      positions: result.positions,
      importedAt: new Date().toISOString(),
    };
    if (
      await commit((s) => {
        s.portfolios.push(p);
        s.preferences.activePortfolioId = p.id;
        if (!s.imports.includes(importData.hash))
          s.imports.push(importData.hash);
      })
    ) {
      selected = p.id;
      close();
      location.hash = "portfolio";
      render();
      toast("Snapshot saved locally");
    }
  },
  what: () => {
    const amount = Number($("#what-amount").value),
      p = selectedP(),
      v = valuation(p),
      symbol = $("#what-symbol").value,
      out = $("#what-result");
    if (!Number.isFinite(amount) || amount < 0) {
      out.textContent = "Enter a non-negative contribution.";
      return;
    }
    if (v.value === null) {
      out.textContent = "Complete prices and FX coverage first.";
      return;
    }
    const total = v.value + amount;
    if (symbol === "") {
      out.textContent = `Hypothetical value ${money(total, p.currency)}. Cash ${money(p.cash + amount, p.currency)} (${total ? pct((p.cash + amount) / total) : "0%"}). No return or risk reduction is assumed.`;
      return;
    }
    const r = v.rows[Number(symbol)],
      scale =
        r.currency === "GBX" && p.currency === "GBP"
          ? 0.01
          : r.currency === p.currency
            ? 1
            : null;
    if (!r || !r.price || scale === null) {
      out.textContent =
        "A positive entered price and verified currency units are required.";
      return;
    }
    out.textContent = `${r.symbol}: hypothetical purchase of ${num(amount / (r.price * scale))} shares. New holding value ${money(r.value + amount, p.currency)}; weight ${total ? pct((r.value + amount) / total) : "0%"}. Portfolio value ${money(total, p.currency)}. ${total && (r.value + amount) / total > 0.25 ? "Above the 25% concentration review threshold. " : ""}No future return, liquidity or risk estimate is assumed.`;
  },
  cost: () => {
    const c = Number($("#cost-r").value);
    $("#cost-result").textContent =
      Number.isFinite(c) && c >= 0
        ? Object.entries(data.backtest.summary.strategies)
            .map(
              ([k, v]) =>
                `${k}: ${(v.expectancy_r - c).toFixed(3)}R after hypothetical cost`,
            )
            .join(" · ")
        : "Enter a non-negative cost.";
  },
  customize: () =>
    modal(
      `<h2>Make room for what matters.</h2><p class="muted">Move modules up to prioritise them. Hide modules you do not use.</p>${(state.preferences.order || ["brief", "research", "strategy", "watchlist"]).map((k, i) => `<div class="list-row"><span class="grow">${esc(k)}</span><button data-hide="${k}">${state.preferences.hidden?.includes(k) ? "Show" : "Hide"}</button><button data-up="${i}" ${i === 0 ? "disabled" : ""}>↑ Move up</button></div>`).join("")}`,
    ),
  retry: () => loadData(),
};
document.addEventListener("click", async (e) => {
  const t = e.target.closest("button,[data-stock]");
  if (!t) return;
  try {
    if (t.dataset.stock) return stock(t.dataset.stock);
    if (t.dataset.compare) return compare(t.dataset.compare);
    if (t.dataset.watch) {
      await commit((s) => {
        const x = t.dataset.watch;
        s.watchlist = s.watchlist.includes(x)
          ? s.watchlist.filter((v) => v !== x)
          : [...s.watchlist, x];
      });
      toast("Watchlist updated");
      return;
    }
    if (t.dataset.edit !== undefined)
      return positionForm(Number(t.dataset.edit));
    if (t.dataset.remove !== undefined) {
      const p = selectedP();
      if (p.ledger) {
        toast(
          "Ledger positions cannot be removed manually; import a sale or create a separate manual portfolio.",
        );
        return;
      }
      await commit((s) =>
        s.portfolios
          .find((x) => x.id === p.id)
          .positions.splice(Number(t.dataset.remove), 1),
      );
      return;
    }
    if (t.dataset.journal !== undefined)
      return journalForm("", Number(t.dataset.journal));
    if (t.dataset.journalSymbol) return journalForm(t.dataset.journalSymbol);
    if (t.dataset.hide) {
      await commit((s) => {
        const h = s.preferences.hidden || (s.preferences.hidden = []),
          k = t.dataset.hide;
        s.preferences.hidden = h.includes(k)
          ? h.filter((x) => x !== k)
          : [...h, k];
      });
      return actions.customize();
    }
    if (t.dataset.up) {
      await commit((s) => {
        const order = s.preferences.order || [
            "brief",
            "research",
            "strategy",
            "watchlist",
          ],
          i = Number(t.dataset.up);
        [order[i - 1], order[i]] = [order[i], order[i - 1]];
        s.preferences.order = order;
      });
      return actions.customize();
    }
    if (t.dataset.action) await actions[t.dataset.action]?.();
  } catch (e) {
    toast(e.message);
  }
});
document.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target,
    v = Object.fromEntries(new FormData(f));
  try {
    let ok = false;
    if (f.id === "new-form") {
      const p = {
        id: id(),
        name: v.name,
        currency: "GBP",
        cash: 0,
        positions: [],
      };
      ok = await commit((s) => {
        s.portfolios.push(p);
        s.preferences.activePortfolioId = p.id;
      });
      selected = p.id;
    }
    if (f.id === "portfolio-form") {
      const p = selectedP();
      if (
        p.ledger &&
        ((v.currency ?? p.currency) !== p.currency || Number(v.cash) !== p.cash)
      )
        throw Error(
          "Cash and settlement currency come from the ledger; import transactions to change them.",
        );
      ok = await commit((s) =>
        Object.assign(
          s.portfolios.find((x) => x.id === p.id),
          {
            name: v.name,
            currency: v.currency ?? p.currency,
            cash: Number(v.cash),
          },
        ),
      );
    }
    if (f.id === "position-form") {
      const r = validatePosition({
          symbol: v.symbol.toUpperCase().trim(),
          quantity: Number(v.quantity),
          currency: v.currency.trim(),
          cost: Number(v.cost),
          price: v.price === "" ? null : Number(v.price),
          asOf: v.asOf || null,
        }),
        p = selectedP(),
        i = f.dataset.index;
      ok = await commit((s) => {
        const a = s.portfolios.find((x) => x.id === p.id).positions;
        if (p.ledger) {
          const old = a[Number(i)];
          if (
            i === "" ||
            !old ||
            r.symbol !== old.symbol ||
            r.quantity !== old.quantity ||
            r.currency !== old.currency ||
            r.cost !== old.cost
          )
            throw Error(
              "Ledger shares, currency and cost cannot be edited. Update the quote only.",
            );
          a[Number(i)] = { ...old, price: r.price, asOf: r.asOf };
          return;
        }
        if (
          a.some(
            (x, j) =>
              x.symbol === r.symbol &&
              x.currency === r.currency &&
              String(j) !== i,
          )
        )
          throw Error("Position already exists; edit it instead");
        i === "" ? a.push(r) : a.splice(Number(i), 1, r);
      });
    }
    if (f.id === "journal-form") {
      const j = {
        ...v,
        symbol: v.symbol.trim().toUpperCase(),
        date: new Date().toISOString(),
      };
      if (!/^[A-Z0-9.^=:/_-]{1,30}$/.test(j.symbol))
        throw Error("Invalid symbol");
      ok = await commit((s) => {
        const i = f.dataset.index;
        i === ""
          ? s.journal.push(j)
          : s.journal.splice(Number(i), 1, {
              ...s.journal[i],
              outcome: v.outcome,
            });
      });
    }
    if (ok) {
      close();
      render();
    }
  } catch (e) {
    const error = $("#form-error");
    if (error) error.textContent = e.message;
    else toast(e.message);
  }
});
document.addEventListener("change", async (e) => {
  const t = e.target;
  if (t.id === "history-destination") return configureHistoryDestination();
  if (t.id === "stock-range") {
    stockRange = t.value;
    return stock(t.dataset.symbol);
  }
  if (t.id === "compare-second") return compare(t.dataset.first, t.value);
  if (t.id === "csv-file" && t.files[0]) return importFile(t.files[0]);
  if (t.dataset.map) {
    importData.map[t.dataset.map] = Number(t.value);
    return;
  }
  if (t.id === "portfolio-select") {
    selected = t.value;
    await commit((s) => (s.preferences.activePortfolioId = selected));
  }
  if (t.id === "sector") {
    sector = t.value;
    await commit((s) => (s.preferences.sector = sector));
  }
  if (t.id === "sort") {
    sort = t.value;
    await commit((s) => (s.preferences.sort = sort));
  }
  if (t.id === "horizon") {
    horizon = t.value;
    render();
  }
  if (t.id === "backup-file" && t.files[0]) {
    try {
      if (t.files[0].size > 5e6) throw Error("Backup exceeds 5 MB");
      const b = validateState(JSON.parse(await t.files[0].text()));
      modal(
        '<h2>Validated backup</h2><p>This replaces the entire local workspace.</p><button id="restore-confirm">Replace local workspace</button>',
      );
      $("#restore-confirm").onclick = async () => {
        if (
          await commit((s) => {
            for (const k of Object.keys(s)) delete s[k];
            Object.assign(s, b);
          })
        ) {
          selected = state.preferences.activePortfolioId || "";
          sector = state.preferences.sector || "";
          sort = state.preferences.sort || "score";
          document.body.classList.toggle("light", !!state.preferences.light);
          close();
          render();
        }
      };
    } catch (e) {
      $("#backup-error").textContent = e.message;
    }
  }
});
document.addEventListener("input", (e) => {
  const t = e.target;
  if (t.id === "filter") {
    query = t.value;
    const pos = t.selectionStart;
    render();
    $("#filter").focus();
    $("#filter").setSelectionRange(pos, pos);
  }
  if (t.dataset.correct !== undefined)
    importData.parsed.rows[Number(t.dataset.correct)][importData.map.symbol] =
      t.value;
  if (t.id === "symbol-search") {
    const q = t.value.toUpperCase();
    $("#search-results").innerHTML = stocks()
      .filter((r) => r.symbol.includes(q))
      .slice(0, 20)
      .map(
        (r) =>
          `<div class="list-row">${ticker(r.symbol)} <small>${esc(r.sector)}</small></div>`,
      )
      .join("");
  }
});
document.addEventListener("dragover", (e) => {
  if (e.target.closest("#drop")) {
    e.preventDefault();
    $("#drop").classList.add("over");
  }
});
document.addEventListener("drop", (e) => {
  if (e.target.closest("#drop")) {
    e.preventDefault();
    if (e.dataTransfer.files[0]) importFile(e.dataTransfer.files[0]);
  }
});
$("#modal").addEventListener("close", () => {
  importData = null;
  importEpoch++;
});
$("#close-modal").onclick = close;
$("#search-open").onclick = () => {
  modal(
    '<h2>Find a stock</h2><input id="symbol-search" aria-label="Search symbols" placeholder="Type a ticker…" autocomplete="off"><div id="search-results"></div>',
  );
  $("#symbol-search").focus();
};
$("#theme").onclick = async () => {
  const light = !document.body.classList.contains("light");
  document.body.classList.toggle("light", light);
  await commit((s) => (s.preferences.light = light));
};
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) {
    e.preventDefault();
    $("#search-open").click();
  }
});
window.addEventListener("hashchange", () => {
  render();
  openStockHash();
});
function openStockHash() {
  if (location.hash.startsWith("#stock=")) {
    const symbol = decodeURIComponent(location.hash.slice(7));
    if (/^[A-Z0-9.^=:/_-]{1,30}$/.test(symbol)) stock(symbol);
  }
}
async function loadData() {
  errors = [];
  await Promise.allSettled(
    [
      "trending",
      "skew",
      "backtest",
      "paper-trading",
      "prediction-tracker",
      "prediction-research",
      "manifest",
      "research-diagnostics",
    ].map(async (k) => {
      try {
        const r = await fetch(`data/${k}.json`);
        if (!r.ok) throw Error(r.status);
        const d = await r.json();
        data[k] = d;
      } catch {
        errors.push(k);
      }
    }),
  );
  render();
}
try {
  db = await openStore();
  state = await readStore(db);
  sector = state.preferences.sector || "";
  sort = state.preferences.sort || "score";
  document.body.classList.toggle("light", !!state.preferences.light);
} catch (e) {
  storageError = e.message;
  db?.close();
  db = null;
  toast(e.message);
}
await loadData();
openStockHash();

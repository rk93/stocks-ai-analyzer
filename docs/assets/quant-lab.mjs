// Public, read-only research views. This module never writes a strategy ledger.
const esc = (value) =>
  String(value ?? "Unavailable").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const list = (value) => (Array.isArray(value) ? value : []);
const finite = (value) =>
  value !== null &&
  value !== undefined &&
  value !== "" &&
  Number.isFinite(Number(value));
const number = (value, digits = 2) =>
  finite(value)
    ? Number(value).toLocaleString("en-GB", { maximumFractionDigits: digits })
    : "Unavailable";
const percent = (value) =>
  finite(value) ? `${(100 * Number(value)).toFixed(1)}%` : "Unavailable";
const dollars = (value) =>
  finite(value)
    ? new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(value)
    : "Unavailable";
const pick = (object, ...keys) =>
  keys
    .map((key) => object?.[key])
    .find((value) => value !== undefined && value !== null);
const human = (value) => String(value ?? "Unavailable").replaceAll("_", " ");
const box = (title, body, extra = "") =>
  `<section class="panel"><div class="section-head"><h2>${esc(title)}</h2>${extra}</div>${body}</section>`;
const empty = (text) => `<p class="empty">${esc(text)}</p>`;
const table = (heads, rows) =>
  `<div class="scroll" tabindex="0" role="region" aria-label="${esc(heads[0])} research table"><table><thead><tr>${heads.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
const sequence = (value) =>
  Array.isArray(value)
    ? value
    : value && typeof value === "object"
      ? Object.entries(value)
          .filter(([, entry]) => entry && typeof entry === "object" && !Array.isArray(entry))
          .map(([strategy, entry]) => ({ strategy, ...entry }))
      : [];
const metric = (row, ...keys) => pick(row?.metrics, ...keys) ?? pick(row, ...keys);
function interval(value, unit = "R") {
  const ends = Array.isArray(value)
    ? value
    : value
      ? [pick(value, "lower", "low", "lo"), pick(value, "upper", "high", "hi")]
      : [];
  return ends.length === 2 && ends.every(finite)
    ? `${number(ends[0], 3)}${unit} to ${number(ends[1], 3)}${unit}`
    : "Insufficient observations";
}
function ci(row) {
  return metric(
    row,
    "expectancy_r_ci95",
    "expectancy_ci95_r",
    "bootstrap_ci_r",
    "bootstrap_95_ci_r",
    "confidence_interval",
    "date_block_bootstrap_95_interval_r",
  );
}
function signalCount(row) {
  const value = metric(row, "signals", "signal_count", "candidates");
  return Array.isArray(value) ? value.length : value;
}
function sample(row) {
  return metric(
    row,
    "executed_trades",
    "closed_trades",
    "trade_count",
    "trades",
    "count",
    "n",
  );
}
function expectancy(row) {
  return metric(row, "net_expectancy_r", "expectancy_net_r", "expectancy_r");
}
function reportFreshness(report) {
  const timestamp = pick(report, "generated_at", "as_of");
  const age = Date.now() - Date.parse(timestamp);
  const stale = !Number.isFinite(age) || age < 0 || age > 4 * 86400000;
  return `<p class="notice" data-ql-freshness>${stale ? "DATED / STALE RESEARCH · " : "DATED RESEARCH · "}Published ${esc(timestamp ?? "timestamp unavailable")}. Publication time is not a live market quote. Research older than four calendar days is flagged; this age rule is not an exchange-calendar check.</p>`;
}
function comparisonTable(rows) {
  if (!rows.length)
    return empty(
      "Comparable replay results are not available. A missing result is not evidence of profitability.",
    );
  return table(
    [
      "Strategy / source",
      "Signals / trades",
      "Net expectancy",
      "Profit factor",
      "Win rate",
      "Maximum drawdown",
      "95% expectancy interval",
    ],
    rows
      .map(
        (r) =>
          `<tr><td><b>${esc(pick(r, "strategy", "version", "name"))}</b><p class="sub">${esc(human(r.source ?? "Source unavailable"))}${r.window ? " · " + esc(human(r.window)) : ""}</p></td><td>${number(signalCount(r), 0)} / ${number(sample(r), 0)}</td><td>${finite(expectancy(r)) ? `${number(expectancy(r), 3)}R` : "Unavailable"}</td><td>${number(metric(r, "profit_factor", "profit_factor_r"))}</td><td>${percent(metric(r, "win_rate", "win_rate_1r"))}</td><td>${percent(metric(r, "max_drawdown"))}</td><td>${interval(ci(r))}</td></tr>`,
      )
      .join(""),
  );
}
function comparisons(report) {
  const rows = sequence(report.comparisons);
  const legacy = rows.filter((r) =>
    /legacy|published|original/i.test(r.source ?? ""),
  );
  const benchmarks = rows.filter((r) => /benchmark/i.test(r.source ?? ""));
  const replay = rows.filter((r) => !legacy.includes(r) && !benchmarks.includes(r));
  return (
    box(
      "Corrected historical execution comparison",
      `<p class="muted">Retrospective replay of already observed dates. Compare rows only when capital, risk, costs and execution rules match. Later historical dates do not become an untouched test merely by being labelled out of sample.</p>${comparisonTable(replay)}`,
    ) +
    (legacy.length
      ? box(
          "Original forward paper history · separate execution",
          `<p class="muted">V1 is a required reference. These published ledgers predate V4 and use their original execution rules. They are preserved observations, not an equal-execution V4 trial.</p>${comparisonTable(legacy)}`,
        )
      : "") +
    (benchmarks.length ? box("Benchmark references · exposure differs", `<p class="muted">SPY and QQQ are price-return references, with their published execution assumptions. The nominal cash reference is not a Treasury total-return series. Tactical strategies and fully invested benchmarks have different exposure.</p>${table(["Benchmark / window","Starting capital","Ending equity","Net return","Drawdown"],benchmarks.map((r)=>`<tr><td>${esc(r.strategy)} / ${esc(human(r.window))}</td><td>${dollars(metric(r,"starting_capital"))}</td><td>${dollars(metric(r,"ending_equity","equity"))}</td><td>${percent(metric(r,"total_return","return"))}</td><td>${percent(metric(r,"max_drawdown"))}</td></tr>`).join(""))}`) : "")
  );
}
function curve(equity, {mode = "Prospective paper", drawdown = false, initialCapital = 10000} = {}) {
  let rows = list(equity);
  const names = ["V1", "V2", "V3", "V4", "TREND", "SPY", "QQQ", "CASH"];
  const present = names.filter((name) =>
    rows.some((r) => finite(r[`${name}_equity`])),
  );
  if (drawdown) {
    const peaks = Object.fromEntries(present.map((name)=>[name,initialCapital]));
    rows = rows.map((row)=> {
      const result = {...row};
      for (const name of present) {
        const value = row[`${name}_equity`];
        if (!finite(value)) continue;
        peaks[name] = Math.max(peaks[name], Number(value));
        result[`${name}_equity`] = peaks[name] > 0 ? Number(value) / peaks[name] - 1 : null;
      }
      return result;
    });
  }
  const values = rows.flatMap((r) =>
    present
      .map((name) => r[`${name}_equity`])
      .filter(finite)
      .map(Number),
  );
  if (rows.length < 2 || !values.length)
    return empty(
      mode === "Prospective paper" ? "Equity curves require at least two dated prospective observations. No historical signals are inserted into this forward cohort." : "Dated historical equity observations are unavailable. No curve is inferred from trade statistics.",
    );
  let lo = Math.min(...values),
    hi = Math.max(...values);
  if (lo === hi) {
    lo -= drawdown ? 0.01 : 1;
    hi += drawdown ? 0.01 : 1;
  }
  const colors = [
    "#7abbd2",
    "#c1a0ef",
    "#f1c77b",
    "#b9ef8a",
    "#f79898",
    "#d1d8e1",
    "#90a19a",
    "#e0ac84",
  ];
  const paths = present
    .map((name) => {
      let active = false;
      const path = rows
        .map((r, i) => {
          const value = r[`${name}_equity`];
          if (!finite(value)) {
            active = false;
            return "";
          }
          const start = active ? "L" : "M";
          active = true;
          return `${start}${50 + (i / (rows.length - 1)) * 630},${200 - ((Number(value) - lo) / (hi - lo)) * 170}`;
        })
        .join(" ");
      return `<path d="${path}" fill="none" stroke="${colors[names.indexOf(name)]}" stroke-width="2.5"/>`;
    })
    .join("");
  const format = drawdown ? percent : (value)=>number(value,0);
  return `<svg class="ql-curve" viewBox="0 0 700 230" role="img" aria-label="${esc(mode)} ${drawdown ? "drawdown from peak" : "equity in USD"}. ${esc(present.join(", "))}. Range ${format(lo)} to ${format(hi)}."><text x="4" y="20" fill="currentColor" font-size="11">${format(hi)}</text><text x="4" y="205" fill="currentColor" font-size="11">${format(lo)}</text><path d="M50 205H680" stroke="var(--border)"/>${paths}</svg><div class="legend">${present.map((name) => `<span style="color:${colors[names.indexOf(name)]}">● ${name}</span>`).join("")}</div><p class="sub">${esc(rows[0]?.timestamp ?? rows[0]?.date ?? rows[0]?.time)} to ${esc(rows.at(-1)?.timestamp ?? rows.at(-1)?.date ?? rows.at(-1)?.time)} · observation spacing, not elapsed time. Missing marks leave gaps.${drawdown ? " Drawdown is relative to the running peak including published initial capital; intraday extremes may be larger." : ""}</p>`;
}
function replayCurves(report) {
  const groups = new Map();
  for (const row of sequence(report.comparisons)) {
    if (/legacy|original|published/i.test(row.source ?? "")) continue;
    if (!list(row.equity).length) continue;
    const window = row.window ?? "Window unavailable";
    if (!groups.has(window)) groups.set(window, new Map());
    const points = groups.get(window);
    for (const observation of row.equity) {
      const time = pick(observation,"time","timestamp","date");
      if (!time) continue;
      if (!points.has(time)) points.set(time,{timestamp:time});
      const reliable = observation.equity_reliable !== false && !list(observation.stale_symbols).length;
      points.get(time)[`${row.strategy}_equity`] = reliable ? observation.equity : null;
    }
  }
  return box("Historical equity & drawdown · never forward evidence", groups.size ? `<p class="muted">Each chronological window restarts the same published capital. Curves are not joined across purged gaps. Unreliable or stale marks are withheld. A benchmark's exposure differs from a tactical strategy.</p>${[...groups].map(([window,points],i)=> {
    const rows = [...points.values()].sort((a,b)=>String(a.timestamp).localeCompare(String(b.timestamp)));
    const capital = metric(sequence(report.comparisons).find((r)=>r.window === window),"starting_capital") ?? 10000;
    return `<details ${i === 0 ? "open" : ""}><summary>${esc(human(window))} · ${rows.length} dated observations</summary><div class="two-col">${box("Retrospective equity (USD)",curve(rows,{mode:"Retrospective replay"}))}${box("Retrospective drawdown",curve(rows,{mode:"Retrospective replay",drawdown:true,initialCapital:capital}))}</div></details>`;
  }).join("")}` : empty("Dated historical equity observations are unavailable. No curve is inferred from summary returns."));
}
function executionDiagnostics(report) {
  const rows = sequence(report.comparisons).filter((r)=>!/legacy|original|published|benchmark/i.test(r.source ?? ""));
  return box("Execution diagnostics · closed trades and capital exposure", rows.length ? `<p class="muted">Win rate and trade expectancy use closed outcomes. Executed trades can include right-censored open positions. Risk ratios are withheld when the published sample is too short; no annualised return is inferred here.</p>${table(["Version / window","Closed / open","Net return","Mean / median trade return","Mean net P&L","Sharpe / Sortino","Capital utilisation / turnover","Mean holding sessions","Largest loss / costs"],rows.map((r)=>`<tr><td>${esc(r.strategy)} / ${esc(human(r.window))}</td><td>${number(metric(r,"closed_trades"),0)} / ${number(metric(r,"open_trades"),0)}</td><td>${percent(metric(r,"total_return","return"))}</td><td>${percent(metric(r,"average_trade_return"))} / ${percent(metric(r,"median_trade_return"))}</td><td>${dollars(metric(r,"net_expectancy_dollars"))}</td><td>${number(metric(r,"sharpe"))} / ${number(metric(r,"sortino"))}<p class="sub">${esc(metric(r,"risk_ratio_note") ?? "See sample and method in report")}</p></td><td>${percent(metric(r,"average_capital_utilisation"))} / ${number(metric(r,"turnover"))}×</td><td>${number(metric(r,"average_holding_sessions"))}</td><td>${dollars(metric(r,"largest_loss"))} / ${dollars(metric(r,"transaction_costs"))}</td></tr>`).join(""))}<p class="sub">Capital utilisation is calendar-time-weighted entry notional / initial capital. Turnover is (buys + sells) / twice mean equity over this window. Neither is annualised.</p>` : empty("Execution diagnostics unavailable."));
}
function prospective(report, compact) {
  const p = report.prospective ?? {};
  const signals = list(p.signals),
    closed = list(p.closed_trades),
    positions = list(p.positions);
  const rows = sequence(p.metrics?.strategies ?? p.metrics);
  const created = pick(p, "created_at", "createdAt", "frozen_at");
  const warnings = list(p.warnings);
  const status = human(p.status ?? "awaiting prospective collection");
  const cohortStatus = `<p class="notice" data-ql-prospective-status><b>Collection status: ${esc(status)}.</b> ${p.status_at ? `Status checked ${esc(p.status_at)}. ` : ""}Last source-data cutoff ${esc(p.data_cutoff ?? "unavailable; no market observations published")}. This is a dated paper cohort, not a live quote.</p><p class="sub">Collection actually started ${esc(p.collection_started_at ?? "not yet started")}. Registration does not backfill missed observations.</p>${warnings.length ? `<ul class="notice" aria-label="Prospective collection warnings">${warnings.map((warning)=>`<li>${esc(warning)}</li>`).join("")}</ul>` : ""}`;
  return box(
    "Genuinely prospective · new equal-capital paper cohort",
    `<p class="notice">${signals.length || closed.length ? "Prospective observations must follow their frozen signal timestamp." : "Awaiting prospective observations."} V1–V4 start a separate experiment with ${dollars(p.starting_cash)} each. The earlier V1–V3 ledger is separate. Paper simulation only.</p><p class="sub">Cohort created ${esc(created ?? "not yet published")} · frozen configuration ${esc(p.config_hash ?? report.config_hash ?? "unavailable")}</p>${cohortStatus}<div class="cards">${box("Frozen signals", `<div class="metric">${signals.length}</div>`)}${box("Closed prospective trades", `<div class="metric">${closed.length}</div>`)}${box("Open prospective positions", `<div class="metric">${positions.length}</div>`)}</div>${curve(p.equity)}${rows.length ? table(["Strategy", "Equity", "Net return", "Drawdown", "Closed trades"], rows.map((r) => `<tr><td>${esc(r.strategy)}</td><td>${dollars(metric(r, "equity", "portfolio_value", "ending_equity"))}</td><td>${percent(metric(r, "return", "net_return", "total_return"))}</td><td>${percent(metric(r, "max_drawdown"))}</td><td>${number(sample(r), 0)}</td></tr>`).join("")) : ""}${compact ? '<p><a class="tag" href="v4.html#prospective">Inspect the prospective cohort ↗</a></p>' : prospectivePositions(positions)}`,
  );
}
function prospectivePositions(positions) {
  return positions.length
    ? `<h3>Open paper positions</h3>${table(["Strategy / symbol", "Entered", "Quantity", "Entry / last mark", "Stop / target"], positions.map((r) => `<tr><td>${esc(r.strategy)} / ${esc(r.symbol)}</td><td>${esc(r.entry_time)}</td><td>${number(r.qty ?? r.quantity)}</td><td>${number(r.entry_price)} / ${number(r.last_price)}</td><td>${number(r.stop)} / ${number(r.target)}</td></tr>`).join(""))}`
    : "";
}
function validation(report) {
  const v = report.validation ?? {};
  const limitations = [...new Set([...list(v.limitations), ...list(report.limitations)])];
  return box(
    "Validation status & frozen assumptions",
    `<p class="notice"><b>V4 is experimental research only.</b> ${esc(human(v.status ?? report.status ?? "inconclusive"))}. No validated positive expected return is established.</p><dl class="ql-facts"><div><dt>Untouched test</dt><dd>${esc(human(v.untouched_test_status ?? "Awaiting independent future observations"))}</dd></div><div><dt>Configuration hash</dt><dd class="ql-hash">${esc(report.config_hash ?? "Unavailable")}</dd></div><div><dt>Experiment / schema</dt><dd>${esc(report.experiment_id ?? report.config?.experiment_id ?? "Unavailable")} / ${esc(report.schema_version ?? "Unavailable")}</dd></div></dl>${limitations.length ? `<details class="ql-limitations"><summary>Coverage and methodology limits · ${limitations.length} disclosures</summary><ul>${limitations.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></details>` : '<p class="muted">Historical coverage, point-in-time features and independent observations limit the conclusions. Read the experiment configuration before comparing metrics.</p>'}<details><summary>Inspect frozen configuration</summary><pre class="ql-config">${esc(JSON.stringify(report.config ?? report.primary_execution_config ?? {}, null, 2))}</pre></details><p><a class="tag" href="data/v4-research.json">Download V4 report ↗</a> <a class="tag" href="data/quant-audit.json">Download root-cause audit ↗</a> <a class="tag" href="data/v4-replay.json">Full historical replay & frozen inputs ↗</a> <a class="tag" href="data/v4-forward.json">Prospective cohort artifact ↗</a></p>`,
  );
}
function experiments(report) {
  const rows = list(report.experiments);
  return box(
    "Experiment register · includes unsuccessful hypotheses",
    rows.length
      ? table(
          [
            "Experiment / window",
            "Status",
            "Executed / net expectancy",
            "95% interval",
            "Assumptions",
          ],
          rows
            .map(
              (r) =>
                `<tr><td><b>${esc(r.name ?? r.id)}</b><p class="sub">${esc(typeof r.window === "object" ? JSON.stringify(r.window) : (r.window ?? "Window unavailable"))}</p></td><td>${esc(human(r.status ?? "retrospective exploratory"))}</td><td>${number(sample(r), 0)} / ${finite(expectancy(r)) ? number(expectancy(r), 3) + "R" : "Unavailable"}</td><td>${interval(ci(r))}</td><td>${esc(evidence(r.assumptions) ?? r.description ?? r.hypothesis ?? evidence(r.config) ?? "Unavailable")}</td></tr>`,
            )
            .join(""),
        )
      : empty(
          "No experiment register is published. Untested rules cannot support an improvement claim.",
        ),
  );
}
function regimes(report, audit) {
  let rows = sequence(report.retrospective?.regimes ?? report.regimes);
  if (!rows.length)
    rows = Object.entries(audit?.strategies ?? {}).flatMap(([strategy, r]) =>
      list(r.strata?.paper?.market_regime).map((b) => ({
        strategy,
        regime: b.bucket,
        source: "Original forward paper ledger",
        ...b,
      })),
    );
  return box(
    "Market regime diagnostics · descriptive associations",
    `<p class="muted">These groups describe observed associations. Small, correlated groups cannot establish a causal filter or a stable edge.</p>${rows.length ? table(["Strategy / regime", "Source", "Trades", "Win rate", "Net expectancy / mean P&L"], rows.map((r) => `<tr><td>${esc(r.strategy)} / ${esc(r.regime ?? r.bucket)}</td><td>${esc(human(r.source))}</td><td>${number(sample(r), 0)}</td><td>${percent(metric(r, "win_rate"))}</td><td>${finite(expectancy(r)) ? number(expectancy(r), 3) + "R" : dollars(metric(r, "mean_pnl"))}</td></tr>`).join("")) : empty("Frozen point-in-time regime features are unavailable for this sample. Current market conditions are not substituted for historical features.")}`,
  );
}
function calibration(audit) {
  const rows = Object.entries(audit?.strategies ?? {}).flatMap(
    ([strategy, r]) =>
      ["paper", "backtest"].flatMap((source) =>
        list(r.score_buckets?.[source]).map((b) => ({
          strategy,
          source,
          ...b,
        })),
      ),
  );
  return box(
    "Prediction scores · observed outcomes, not probabilities",
    `<p class="notice">A research score is not a calibrated probability. A higher score does not establish a higher chance of profit. Original paper outcomes and historical replay outcomes use different evaluation methods.</p>${rows.length ? scoreChart(rows) + table(["Strategy / score bucket", "Sample source", "Resolved", "Observed win rate", "95% win interval", "Mean P&L / expectancy"], rows.map((r) => `<tr><td>${esc(r.strategy)} / ${esc(r.bucket)}</td><td>${esc(r.source === "paper" ? "Original forward paper" : "Historical replay")}</td><td>${number(r.count, 0)}</td><td>${percent(r.win_rate)}</td><td>${interval(r.win_rate_ci95?.map?.((v) => (finite(v) ? Number(v) * 100 : v)) ?? [], "%")}</td><td>${finite(r.expectancy_r) ? number(r.expectancy_r, 3) + "R" : dollars(r.mean_pnl)}</td></tr>`).join("")) : empty("Score-bucket observations are unavailable. No probability or calibration is inferred.")}`,
  );
}
function scoreChart(rows) {
  const values = rows.filter((r) => finite(r.win_rate) && Number(r.count) > 0);
  if (!values.length) return "";
  const height = 48 + values.length * 28;
  return `<svg class="ql-scores" viewBox="0 0 700 ${height}" role="img" aria-label="Observed win rate by strategy, sample source and research score bucket. Circles show observed rates; lines show descriptive 95 percent intervals. Research scores are not probabilities."><path d="M230 24V${height - 20}M450 24V${height - 20}M670 24V${height - 20}" stroke="var(--border)"/><text x="220" y="15" fill="currentColor" font-size="11">0%</text><text x="440" y="15" fill="currentColor" font-size="11">50%</text><text x="650" y="15" fill="currentColor" font-size="11">100%</text>${values
    .map((r, i) => {
      const y = 40 + i * 28,
        x = 230 + Math.max(0, Math.min(1, Number(r.win_rate))) * 440;
      const range = list(r.win_rate_ci95);
      const line =
        range.length === 2 && range.every(finite)
          ? `<path d="M${230 + Math.max(0, Number(range[0])) * 440} ${y}H${230 + Math.min(1, Number(range[1])) * 440}" stroke="${r.source === "paper" ? "#7abbd2" : "#b9ef8a"}" stroke-width="2"/>`
          : "";
      return `<g><title>${esc(r.strategy)} ${esc(r.source)} ${esc(r.bucket)}: ${percent(r.win_rate)}, ${number(r.count, 0)} resolved; descriptive interval ${interval(
        range.map((v) => Number(v) * 100),
        "%",
      )}</title><text x="0" y="${y + 4}" fill="currentColor" font-size="11">${esc(r.strategy)} ${esc(r.source)} ${esc(r.bucket)} · n=${number(r.count, 0)}</text>${line}<circle cx="${x}" cy="${y}" r="4" fill="${r.source === "paper" ? "#7abbd2" : "#b9ef8a"}"/></g>`;
    })
    .join(
      "",
    )}</svg><p class="sub">Observed win-rate intervals by score bucket. Paper and replay outcomes remain separate. Intervals are descriptive and reflect the audit's resampling method.</p>`;
}
function findings(audit) {
  return box(
    "Why the baselines lose · audit evidence",
    list(audit?.findings).length
      ? list(audit.findings)
          .map(
            (r) =>
              `<article class="ql-finding"><h3>${esc(human(r.title ?? r.id))}</h3><p>${esc(typeof r.evidence === "object" ? JSON.stringify(r.evidence) : r.evidence)}</p><p class="muted">${esc(r.interpretation)}</p></article>`,
          )
          .join("") + `<details><summary>Audit provenance & limitations</summary><p>Audit as of ${esc(audit.as_of ?? "Unavailable")}. Frozen historical features may describe the captured signal context rather than the actual legacy paper entry.</p><ul>${list(audit.limitations).map((s)=>`<li>${esc(s)}</li>`).join("")}</ul></details>`
      : empty(
          "Root-cause audit unavailable. Published V1–V3 results remain visible in their original dashboards.",
        ),
  );
}
function evidence(value) {
  return Array.isArray(value)
    ? value.join(" · ")
    : typeof value === "object" && value
      ? JSON.stringify(value)
      : value;
}
function tradeRows(report) {
  const fallback = sequence(report.comparisons).filter((r) => !/legacy|original|published|benchmark/i.test(r.source ?? "")).flatMap((r) => list(r.trades).map((trade) => ({...trade,strategy:r.strategy,window:r.window})));
  const historical = list(report.retrospective?.trades ?? fallback).map((r) => ({
    ...r, source: "Retrospective replay",
  }));
  const forward = list(report.prospective?.closed_trades).map((r) => ({
    ...r, source: "Prospective paper",
  }));
  const decisions = list(
    report.retrospective?.decisions ?? report.decisions,
  ).map((r) => ({ ...r, source: "Retrospective decision" }));
  return [...forward, ...historical, ...decisions];
}
function trades(report) {
  const rows = tradeRows(report);
  if (!rows.length)
    return box(
      "Trade & decision drill-down",
      empty(
        "No V4 trade or candidate decision records are published. No historical record is presented as a prospective prediction.",
      ),
    );
  const unique = (key) =>
    [...new Set(rows.map((r) => String(r[key] ?? "Unavailable")))].sort();
  const select = (key, title) =>
    `<label>${title}<select data-ql-filter="${key}" aria-label="${title}"><option value="">All</option>${unique(
      key,
    )
      .map((v) => `<option value="${esc(v)}">${esc(v)}</option>`)
      .join("")}</select></label>`;
  return box(
    "Trade & decision drill-down",
    `<div class="tools">${select("strategy", "Filter strategy")}${select("source", "Filter evidence source")}${select("market_regime", "Filter regime")}</div><p class="sub" data-ql-count role="status">${rows.length} records shown</p><div class="ql-trades">${table(["Signal / symbol", "Version / regime", "Evidence source", "Outcome / net R", "Original evidence"], rows.map((r) => `<tr data-ql-trade data-strategy="${esc(r.strategy ?? "Unavailable")}" data-source="${esc(r.source)}" data-market_regime="${esc(r.market_regime ?? "Unavailable")}"><td data-label="Signal / symbol">${esc(pick(r, "signal_time", "signal_date", "entry_time", "timestamp"))}<br><a class="ticker" href="index.html#stock=${encodeURIComponent(r.symbol ?? "")}">${esc(r.symbol)}</a></td><td data-label="Version / regime">${esc(r.strategy ?? "V4")}<p class="sub">${esc(r.market_regime ?? "Unavailable")}</p></td><td data-label="Evidence source">${esc(r.source)}</td><td data-label="Outcome / net R">${esc(pick(r, "exit_reason", "outcome", "result", "decision") ?? (r.accepted === true ? "Accepted" : r.accepted === false ? "Rejected" : "Unavailable"))}<p>${finite(pick(r, "net_r", "realized_r", "realized_r_1r")) ? number(pick(r, "net_r", "realized_r", "realized_r_1r"), 3) + "R" : "R unavailable"}</p></td><td data-label="Original evidence"><details><summary>Inspect evidence for ${esc(r.symbol)}</summary><p>${esc(evidence(pick(r, "explanation", "original_reason", "reason", "rationale")) ?? "Original rationale unavailable")}</p><p>Supports: ${esc(evidence(pick(r, "supports", "evidence", "features")) ?? "Not archived")}</p><p>Contradictions / rejection: ${esc(evidence(pick(r, "contradictions", "rejections", "reject_reasons", "reasons")) ?? "Not archived")}</p><p>Invalidation: ${esc(evidence(r.invalidation) ?? "Use the frozen stop and experiment exit rules; no additional condition archived")}</p><p>Entry ${number(r.entry_price ?? r.entry)} · stop ${number(r.stop)} · target ${number(r.target)} · score ${number(r.score)} (research score)</p><p class="sub">Data cutoff ${esc(r.data_cutoff ?? r.data_available_at ?? "Unavailable")} · configuration ${esc(r.config_hash ?? report.config_hash ?? "Unavailable")}</p></details></td></tr>`).join(""))}</div><p class="empty" data-ql-no-match hidden>No records match these filters. Change a filter to inspect the available evidence.</p><p class="sub">${rows.length} published records. The report contains the reproducible full sample; outcome counts may differ from rejected-candidate counts.</p>`,
  );
}
export function renderQuantSummary(report, audit) {
  return box(
    "V4 research & prediction quality",
    `<span class="tag">EXPERIMENTAL · INCONCLUSIVE</span><p>V4 is isolated from the original V1–V3 records. The new forward cohort ${list(report?.prospective?.signals).length ? "contains frozen prospective signals" : "is awaiting prospective observations"}.</p><p class="muted">Research scores are not probabilities. Inspect score buckets, loss drivers and frozen candidate explanations.</p><a class="tag" href="v4.html">Open quantitative research ↗</a>${report ? `<p class="sub">Report ${esc(report.generated_at ?? report.as_of ?? "timestamp unavailable")} · ${list(audit?.findings).length} audit findings</p>` : '<p class="notice">V4 research report is unavailable; no conclusion can be drawn.</p>'}`,
  );
}
export function renderQuantLab(report, audit, { compact = false } = {}) {
  if (!report || typeof report !== "object" || Array.isArray(report))
    return `<div class="quant-lab">${box("V4 experimental research", '<span class="tag">RESEARCH ONLY</span>' + empty("V4 report unavailable. Retry the published datasets or inspect the original baseline dashboards. No V4 performance is inferred.") + '<a class="tag" href="v4.html">Open V4 research & retry ↗</a>')}</div>`;
  return `<div class="quant-lab" aria-label="V4 experimental quantitative research">${reportFreshness(report)}${validation(report)}${comparisons(report)}<div id="prospective">${prospective(report, compact)}</div>${compact ? '<p><a class="tag" href="v4.html">Open experiments, regime diagnostics, calibration & trade evidence ↗</a></p>' : replayCurves(report) + findings(audit) + experiments(report) + regimes(report, audit) + calibration(audit) + trades(report)}</div>`;
}
const bound = new WeakSet();
export function bindQuantLab(root = document) {
  if (bound.has(root)) return;
  bound.add(root);
  root.addEventListener("change", (event) => {
    if (!event.target.matches("[data-ql-filter]")) return;
    const container = event.target.closest(".quant-lab");
    const filters = [...container.querySelectorAll("[data-ql-filter]")];
    let count = 0;
    container.querySelectorAll("[data-ql-trade]").forEach((row) => {
      row.hidden = filters.some(
        (field) =>
          field.value && row.dataset[field.dataset.qlFilter] !== field.value,
      );
      if (!row.hidden) count++;
    });
    const status = container.querySelector("[data-ql-count]");
    if (status) status.textContent = `${count} records shown`;
    const emptyState = container.querySelector("[data-ql-no-match]");
    if (emptyState) emptyState.hidden = count > 0;
  });
}

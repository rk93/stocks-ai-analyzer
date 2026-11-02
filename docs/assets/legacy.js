// Keep legacy URLs available while routing portfolio ownership to the private workspace.
const bar = document.createElement("nav");
bar.setAttribute("aria-label", "Research platform navigation");
bar.style.cssText =
  "padding:14px 20px;background:#12191e;border-bottom:1px solid #34433c;display:flex;gap:20px;flex-wrap:wrap;font:13px system-ui;color:#b9ef8a";
for (const [label, route] of [
  ["✳ Northstar", "overview"],
  ["Discover", "discover"],
  ["My Portfolio · private", "portfolio"],
  ["Strategy Lab", "lab"],
  ["Insights", "insights"],
]) {
  const a = document.createElement("a");
  a.href = "index.html#" + route;
  a.textContent = label;
  a.style.color = "inherit";
  bar.append(a);
}
document.body.prepend(bar);
// Existing legacy tables use their first column for symbols. Expose a consistent stock workspace.
const enrich = () =>
  document.querySelectorAll("tbody tr").forEach((row) => {
    const cell = row.cells[0];
    if (!cell || cell.querySelector("a")) return;
    const bold = cell.querySelector("b");
    if (!bold) return;
    const symbol = bold.textContent.trim();
    if (!/^[A-Z0-9.^=:/_-]{1,30}$/.test(symbol)) return;
    const a = document.createElement("a");
    a.href = "index.html#stock=" + encodeURIComponent(symbol);
    a.textContent = symbol + " ↗";
    a.style.color = "#b9ef8a";
    bold.replaceWith(a);
  });
const observer = new MutationObserver(enrich);
observer.observe(document.body, { childList: true, subtree: true });
enrich();

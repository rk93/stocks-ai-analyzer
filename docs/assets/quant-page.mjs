import { renderQuantLab, bindQuantLab } from "./quant-lab.mjs";
bindQuantLab();
let generation = 0;
async function load() {
  const request = ++generation;
  const status = document.querySelector("#ql-load-status");
  const button = document.querySelector("#ql-retry");
  button.disabled = true;
  status.textContent = "Loading dated research…";
  const results = await Promise.allSettled(
    ["v4-research", "quant-audit"].map(async (name) => {
      const response = await fetch(`data/${name}.json`, { cache: "no-store" });
      if (!response.ok) throw Error(name);
      return response.json();
    }),
  );
  if (request !== generation) return;
  const [report, audit] = results.map((r) =>
    r.status === "fulfilled" ? r.value : null,
  );
  document.querySelector("#ql-content").innerHTML = renderQuantLab(
    report,
    audit,
  );
  status.textContent = results.some((r) => r.status === "rejected")
    ? "Some research datasets are unavailable. Reload to retry; original baseline dashboards remain accessible above."
    : "Published research loaded. Dates and coverage are shown with each source.";
  button.disabled = false;
}
document.querySelector("#ql-retry").addEventListener("click", load);
document.querySelector("#ql-theme").addEventListener("click", (event) => {
  const light = document.body.classList.toggle("light");
  event.target.setAttribute("aria-pressed", String(light));
  event.target.textContent = light ? "Dark mode" : "Light mode";
});
load();

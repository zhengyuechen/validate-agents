"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const txt = (s) => document.createTextNode(s);
const spinner = () => el("span", "spin");

const api = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  },
  async send(method, path, body) {
    const r = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    return data;
  },
};

let loadedConfig = null;
let activeRunId = null;
let poller = null;
let timer = null;
let runStartedAt = null;
let logRunId = null;
let logCursor = 0;

function showView(name) {
  $("#view-welcome").hidden = name !== "welcome";
  $("#view-results").hidden = name !== "results";
  $("#view-config").hidden = name !== "config";
  $("#nav-config").classList.toggle("active", name === "config");
}

function renderGrounding(cfg) {
  const backend = cfg?.grounding?.backend ?? "none";
  const faithful = backend !== "none";
  const badge = $("#grounding-badge");
  badge.replaceChildren(el("span", "dot " + (faithful ? "faithful" : "local")), txt(`grounding: ${backend}`));
}

function statusTag(status) {
  const v = String(status || "unknown").toLowerCase();
  return `tag status ${v}`;
}

function pct(x) {
  return x == null ? "maturity --" : `maturity ${Math.round(Number(x) * 100)}%`;
}

async function loadRuns() {
  const list = $("#runs-list");
  list.replaceChildren();
  let runs = [];
  try { runs = await api.get("/api/runs"); } catch { /* none yet */ }
  if (!runs.length) { list.appendChild(el("li", "empty", "No artifacts yet.")); return runs; }
  for (const r of runs) {
    const li = el("li");
    const b = el("button");
    b.dataset.id = r.id;
    b.appendChild(el("span", "run-goal", r.seed || "(untitled)"));
    b.appendChild(el("span", "run-meta", `${r.id} · ${r.claims} claims · ${r.status}`));
    b.addEventListener("click", () => selectRun(r.id));
    if (r.id === activeRunId) b.classList.add("active");
    li.appendChild(b);
    list.appendChild(li);
  }
  return runs;
}

async function selectRun(id) {
  activeRunId = id;
  document.querySelectorAll(".runs-list button").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  let run;
  try { run = await api.get(`/api/runs/${id}`); } catch { return; }
  const art = run.artifact || {};
  $("#result-id").textContent = id;
  $("#result-seed").textContent = run.seed || "(untitled artifact)";
  $("#status-pill").className = statusTag(run.status);
  $("#status-pill").textContent = run.status || "unknown";
  $("#maturity-pill").textContent = pct(run.maturity);
  $("#load-pill").textContent = `load-bearing ${run.load_bearing || "--"}`;
  renderBlocker(run.blocker);
  renderFormalClaim(art);
  renderClaims(art.claim_graph || []);
  renderAttacks(art.attacks || []);
  renderPredictions(art.predictions || []);
  renderValidationPlan(art.validation_plan);
  $("#report").textContent = run.report || "No markdown report was produced for this run.";
  $("#bibtex").textContent = run.bibtex || "";
  $("#bib-section").hidden = !run.bibtex;
  loadEventsFor(id, true);
  showView("results");
}

function renderBlocker(blocker) {
  const node = $("#blocker-line");
  if (!blocker) { node.hidden = true; return; }
  node.hidden = false;
  node.textContent = `blocker: ${blocker.reason || "-"}${blocker.claim_id ? ` · ${blocker.claim_id}` : ""}`;
}

function renderFormalClaim(art) {
  const fc = art.formal_claim;
  if (!fc) {
    $("#formal-claim").textContent = "No formal claim was recorded.";
    return;
  }
  $("#formal-claim").replaceChildren(
    el("p", null, fc.statement || ""),
    el("p", "muted-line", `regime: ${fc.regime || "--"} · falsifiable: ${fc.falsifiable ? "yes" : "no"}`)
  );
}

function claimChecks(checks) {
  if (!checks || !checks.length) return "--";
  return checks.map((c) => `${c.lens}:${c.verdict}${c.independent_sources ? `/${c.independent_sources}` : ""}`).join(", ");
}

function renderClaims(claims) {
  const tb = $("#claim-table tbody"); tb.replaceChildren();
  $("#claim-count").textContent = claims.length ? `n = ${claims.length}` : "";
  for (const c of claims) {
    const tr = el("tr");
    tr.appendChild(el("td", "mono", c.id));
    tr.appendChild(el("td", null, c.type));
    const st = el("td"); st.appendChild(el("span", statusTag(c.status), c.status || "pending")); tr.appendChild(st);
    tr.appendChild(el("td", "mono check-cell", claimChecks(c.checks)));
    tr.appendChild(el("td", "title-cell", c.statement || ""));
    tb.appendChild(tr);
  }
}

function renderAttacks(attacks) {
  const tb = $("#attack-table tbody"); tb.replaceChildren();
  $("#attack-count").textContent = attacks.length ? `${attacks.length}` : "";
  for (const a of attacks) {
    const tr = el("tr");
    tr.appendChild(el("td", null, a.type || ""));
    tr.appendChild(el("td", null, a.severity || ""));
    tr.appendChild(el("td", null, a.status || ""));
    tr.appendChild(el("td", "mono", a.target_claim_id || "none"));
    tr.appendChild(el("td", "title-cell", a.basis || ""));
    tb.appendChild(tr);
  }
}

function renderPredictions(preds) {
  const tb = $("#prediction-table tbody"); tb.replaceChildren();
  $("#prediction-count").textContent = preds.length ? `${preds.length}` : "";
  for (const p of preds) {
    const tr = el("tr");
    tr.appendChild(el("td", "title-cell", p.observable || ""));
    tr.appendChild(el("td", null, p.effect_size || ""));
    tr.appendChild(el("td", null, p.measurable ? "yes" : "no"));
    tb.appendChild(tr);
  }
}

function renderValidationPlan(plan) {
  const node = $("#validation-plan");
  if (!plan) {
    node.textContent = "No validation plan was recorded.";
    return;
  }
  node.replaceChildren(
    el("p", null, plan.decisive_test || ""),
    el("p", "muted-line", `confirm if: ${plan.confirm_if || "--"}`),
    el("p", "muted-line", `refute if: ${plan.refute_if || "--"} · cost: ${plan.cost || "--"}`)
  );
}

function clip(s, n) {
  s = String(s == null ? "" : s);
  return s.length > n ? s.slice(0, n - 1) + "..." : s;
}

function eventSummary(e) {
  switch (e.event) {
    case "run_started": return ["run", `started · ${clip(e.seed, 42)}`];
    case "run_done": return ["done", `complete · ${e.status} · ${e.claims || 0} claims`];
    case "run_error": return ["error", `failed · ${clip(e.error, 64)}`];
    case "entry_gate": return ["warn", `entry gate · ${e.reason || ""}`];
    case "entry_ok": return ["task", `entry ok · ${e.claims} claims · ${e.coverage}`];
    case "check": return [e.verdict === "pass" ? "review" : "warn", `${e.claim} · ${e.lens} · ${e.verdict}`];
    case "fanout": return ["review", `fanout ${e.claim} · ${e.lens} · ${e.verdict}`];
    case "fanout_limited": return ["warn", `fanout limited · ${e.claim}`];
    case "repair": return ["evolve", `repair · ${(e.targets || []).join(", ") || "none"} · ${e.ok ? "ok" : "failed"}`];
    case "final": return ["done", `final · ${e.status} · ${e.load_bearing || "--"}`];
    case "arbiter_mismatch": return ["warn", `arbiter mismatch · ${e.narrated} vs ${e.computed}`];
    default: return ["task", e.event || "event"];
  }
}

function appendEvents(events) {
  const log = $("#event-log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
  for (const e of events) {
    const [cat, text] = eventSummary(e);
    const li = el("li", "event-row");
    li.appendChild(el("span", "event-tick", e.time ? e.time.slice(11, 19) : "·"));
    li.appendChild(el("span", `event-dot ${cat}`));
    li.appendChild(el("span", "event-text", text));
    log.appendChild(li);
  }
  if (atBottom) log.scrollTop = log.scrollHeight;
}

async function loadEventsFor(runId, reset) {
  if (reset) { logRunId = runId; logCursor = 0; $("#event-log").replaceChildren(); }
  if (logRunId !== runId) return;
  let body;
  try { body = await api.get(`/api/runs/${runId}/events?since=${logCursor}`); } catch { return; }
  if (body.events && body.events.length) { appendEvents(body.events); logCursor = body.next; }
  $("#log-meta").textContent = logCursor ? `${logCursor} events` : "no events yet";
}

function formatElapsed(ms) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function startTimer() {
  if (timer) clearInterval(timer);
  runStartedAt = Date.now();
  $("#run-timer").hidden = false;
  timer = setInterval(() => { $("#run-timer").textContent = `elapsed ${formatElapsed(Date.now() - runStartedAt)}`; }, 1000);
}

function stopTimer(label = "elapsed") {
  if (timer) clearInterval(timer);
  timer = null;
  if (runStartedAt != null) $("#run-timer").textContent = `${label} ${formatElapsed(Date.now() - runStartedAt)}`;
}

async function launchRun(e) {
  e.preventDefault();
  const seed = $("#seed").value.trim();
  if (!seed) return;
  const btn = $("#run-btn");
  const status = $("#run-status");
  btn.disabled = true;
  startTimer();
  status.hidden = false;
  status.className = "run-status";
  status.replaceChildren(spinner(), txt("queued..."));
  let runId;
  try {
    const references = $("#references").value.trim();
    ({ run_id: runId } = await api.send("POST", "/api/runs", { seed, references: references || null }));
  } catch (err) {
    status.className = "run-status is-error";
    status.textContent = `Could not start: ${err.message}`;
    stopTimer("failed after");
    btn.disabled = false;
    return;
  }
  loadEventsFor(runId, true);
  if (poller) clearInterval(poller);
  poller = setInterval(() => pollStatus(runId, btn), 1500);
  pollStatus(runId, btn);
}

async function pollStatus(runId, btn) {
  loadEventsFor(runId, false);
  let s;
  try { s = await api.get(`/api/runs/${runId}/status`); } catch { return; }
  const status = $("#run-status");
  if (s.status === "running" || s.status === "queued") {
    status.className = "run-status";
    status.replaceChildren(spinner(), txt(`${s.status}...`));
    return;
  }
  clearInterval(poller); poller = null; btn.disabled = false;
  if (s.status === "done") {
    stopTimer("completed in");
    status.textContent = "Validation complete.";
    loadRuns().then(() => selectRun(runId));
  } else {
    stopTimer("failed after");
    status.className = "run-status is-error";
    status.textContent = `Run failed: ${s.error || "unknown error"}`;
  }
}

const SELECTS = { backend: ["arxiv", "none", "tavily"] };

function fieldInput(path, key, value) {
  const wrap = el("label", "field");
  wrap.appendChild(el("span", "field-label", key));
  let input;
  if (SELECTS[key]) {
    input = el("select");
    SELECTS[key].forEach((o) => {
      const opt = el("option", null, o);
      opt.value = o;
      if (o === value) opt.selected = true;
      input.appendChild(opt);
    });
    input.dataset.type = "string";
  } else if (typeof value === "boolean") {
    input = el("select");
    ["true", "false"].forEach((o) => {
      const opt = el("option", null, o);
      opt.value = o;
      if (String(value) === o) opt.selected = true;
      input.appendChild(opt);
    });
    input.dataset.type = "bool";
  } else if (typeof value === "number" || value === null) {
    input = el("input");
    input.type = "number";
    input.step = "any";
    input.value = value == null ? "" : value;
    input.dataset.type = "number";
  } else {
    input = el("input");
    input.type = "text";
    input.value = value;
    input.dataset.type = "string";
  }
  input.dataset.path = path;
  wrap.appendChild(input);
  return wrap;
}

function renderConfigForm(cfg) {
  const root = $("#config-groups"); root.replaceChildren();
  for (const [key, value] of Object.entries(cfg)) {
    const fs = el("fieldset", "config-group");
    fs.appendChild(el("legend", null, key));
    const ff = el("div", "config-fields");
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      for (const [sk, sv] of Object.entries(value)) ff.appendChild(fieldInput(`${key}.${sk}`, sk, sv));
    } else {
      ff.appendChild(fieldInput(key, key, value));
    }
    fs.appendChild(ff);
    root.appendChild(fs);
  }
}

function setPath(obj, path, val) {
  const parts = path.split(".");
  let o = obj;
  for (let i = 0; i < parts.length - 1; i++) o = o[parts[i]];
  o[parts[parts.length - 1]] = val;
}

async function saveConfig(e) {
  e.preventDefault();
  const cfg = JSON.parse(JSON.stringify(loadedConfig));
  const msg = $("#config-msg");
  msg.className = "config-msg";
  for (const input of document.querySelectorAll("#config-groups [data-path]")) {
    const t = input.dataset.type, raw = input.value;
    let val;
    if (t === "number") val = raw === "" ? null : Number(raw);
    else if (t === "bool") val = raw === "true";
    else val = raw;
    setPath(cfg, input.dataset.path, val);
  }
  try {
    await api.send("PUT", "/api/config", cfg);
    loadedConfig = cfg;
    msg.className = "config-msg ok"; msg.textContent = "Saved.";
    renderGrounding(cfg);
  } catch (err) {
    msg.className = "config-msg err"; msg.textContent = err.message;
  }
}

async function init() {
  $("#run-form").addEventListener("submit", launchRun);
  $("#view-config").addEventListener("submit", saveConfig);
  $("#nav-config").addEventListener("click", () => {
    if (!loadedConfig) return;
    renderConfigForm(loadedConfig); showView("config");
    document.querySelectorAll(".runs-list button").forEach((b) => b.classList.remove("active"));
    activeRunId = null;
  });
  try {
    const { config } = await api.get("/api/config");
    loadedConfig = config;
    renderGrounding(config);
  } catch { $("#grounding-badge").textContent = "config unavailable"; }
  const runs = await loadRuns();
  if (runs && runs.length) selectRun(runs[0].id);
  else showView("welcome");
}

document.addEventListener("DOMContentLoaded", init);

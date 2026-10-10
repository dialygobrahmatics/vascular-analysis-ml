// Doctor's review screen. All rules are enforced by the server; this script only draws
// the state and sends one change at a time.
"use strict";

const STUDY = document.getElementById("app").dataset.study;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let S = null;            // server state
let runId = null;        // selected run
let findingId = null;    // selected finding
let tool = "view";
let pending = [];        // clicks collected by the current tool
let refClicks = [];      // shift-clicks on a width profile
const masks = {};        // kind -> {w, h, data: Uint8Array}
const dirty = { vessel: false, device: false };
let frames = [];         // Image objects of the selected run
let cur = 0, timer = null;
const canEdit = () => !!S;  // no roles: whoever reviews can edit; the log records who
const signed = () => S && S.study.state === "signed";

// ------------------------------------------------------------------ server --
// what the corner notice says while / after each kind of change
const MESSAGES = [
  [/\/sign$/, "Signing and creating the report…", "Signed"],
  [/\/measure$/, "Measuring…", "Measured"],
  [/\/points$/, "Recalculating…", "Recalculated"],
  [/\/mask\//, "Saving the outline and re-measuring…", "Outline saved"],
  [/\/bestframe$/, "Redrawing the outline on the new frame…", "Best frame changed"],
  [/\/laterality$/, "Saving the side…", "Side saved"],
  [/\/calibration$/, "Calibrating…", "Calibration saved"],
  [/\/ack$/, "Saving your answer…", "Warning answered"],
  [/\/segment$/, "Saving…", "Segment saved"],
  [/\/danger$/, "Saving…", "Answer saved"],
  [/\/finding\/\d+$/, "Saving…", "Saved"],
];

async function api(url, body, raw) {
  const opts = body === undefined ? {} : raw
    ? { method: "POST", headers: { "Content-Type": "image/png" }, body }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const call = async () => {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      const msg = (j.error || `Request failed (${r.status})`) + (j.blockers ? " " + j.blockers.join(" ") : "");
      showError(msg);
      throw new Error(j.error || `Request failed (${r.status})`);
    }
    showError("");
    return j;
  };
  if (body === undefined) return call();  // loading the page state: no notice
  const [, busy, ok] = MESSAGES.find(([rx]) => rx.test(url)) || [null, "Saving…", "Saved"];
  return window.feedback.track(call, busy, ok);
}
function showError(msg) { $("error").innerHTML = msg ? `<div class="error">${esc(msg)}</div>` : ""; }

async function refresh(state) {
  S = state || (await api(`/api/study/${STUDY}`));
  if (runId === null && S.runs.length) runId = S.runs[0].id;
  renderAll();
}

function renderAll() {
  $("title").textContent = `Study ${S.study.pseudonym || S.study.id}`;
  renderSide(); renderWarnings(); renderRuns(); renderSign(); renderSegments(); renderDanger(); renderFindings(); drawOverlay();
  renderCalib();
}

// -------------------------------------------------------------------- side --
function renderSide() {
  const L = S.laterality;
  const cls = L.status === "confirmed" ? "ok" : L.status === "conflict" ? "stop" : "ask";
  const votes = (L.votes || []).map((v) => `<li>${esc(v.source)}: <b>${esc(v.side)}</b> <span class="muted small">${esc(v.detail || "")}</span></li>`).join("");
  let controls = "";
  if (canEdit() && !signed()) {
    controls = `<div class="toolbar">
      <span>Side from the procedure note:</span>
      <select id="note-side"><option value="">not stated</option><option ${S.study.note_side === "LEFT" ? "selected" : ""}>LEFT</option><option ${S.study.note_side === "RIGHT" ? "selected" : ""}>RIGHT</option></select>
      <span>Your decision:</span>
      <button class="secondary" data-side="LEFT">LEFT arm</button><button class="secondary" data-side="RIGHT">RIGHT arm</button>
      ${L.status === "conflict" ? `<input id="override" placeholder="Why the disagreeing source is wrong (required)" style="min-width:300px" value="${esc(S.study.override_reason || "")}">` : ""}
    </div>`;
  }
  $("side").innerHTML = `<div class="banner ${cls}">${esc(L.message)}</div><ul class="small">${votes}</ul>${controls}
    ${L.status !== "confirmed" ? '<div class="small muted">Nothing is measured until the side is confirmed.</div>' : ""}`;
  $("side").querySelectorAll("[data-side]").forEach((b) => b.addEventListener("click", async () => {
    const body = { side: b.dataset.side, note_side: $("note-side").value };
    if ($("override")) body.override_reason = $("override").value;
    await refresh(await api(`/api/study/${STUDY}/laterality`, body));
  }));
}

// ---------------------------------------------------------------- warnings --
function renderWarnings() {
  const box = $("warnings");
  if (!S.warnings.length) { box.innerHTML = ""; return; }
  box.innerHTML = `<div class="card"><h3>Warnings: check and answer each one</h3>${S.warnings.map((w) => w.acknowledged_by
    ? `<div class="warning done">✓ ${esc(w.message)}<div class="small muted">${esc(w.acknowledged_by)}: ${esc(w.response)}</div></div>`
    : `<div class="warning"><b>${esc(w.message)}</b>${canEdit() && !signed() ? `<div class="toolbar"><input data-w="${w.id}" placeholder="What did you find when you checked?" style="flex:1"><button data-ack="${w.id}">I have checked this</button></div>` : ""}</div>`).join("")}</div>`;
  box.querySelectorAll("[data-ack]").forEach((b) => b.addEventListener("click", async () => {
    const response = box.querySelector(`[data-w="${b.dataset.ack}"]`).value;
    await refresh(await api(`/api/warning/${b.dataset.ack}/ack`, { response }));
  }));
}

// -------------------------------------------------------------------- runs --
function run() { return S.runs.find((r) => r.id === runId); }

function renderRuns() {
  $("runs").innerHTML = S.runs.map((r) => `<button class="${r.id === runId ? "active" : ""}" data-run="${r.id}">Run ${r.idx} · ${r.info.is_dsa ? "DSA" : "cine"} · ${r.info.n_frames} fr</button>`).join("");
  $("runs").querySelectorAll("[data-run]").forEach((b) => b.addEventListener("click", () => selectRun(Number(b.dataset.run))));
  const r = run();
  if (r) {
    $("run-info").textContent = `${r.info.description_text || ""} · best frame ${r.best.index} (${r.best.confidence}: ${r.best.reason}) · outline: ${r.draft_notes?.segmentation || ""} ${(r.draft_notes?.notes || []).join(" ")}`;
  }
}

async function selectRun(id) {
  if ((dirty.vessel || dirty.device) && !confirm("The outline has unsaved changes. Discard them?")) return;
  runId = id; findingId = null; pending = []; refClicks = [];
  renderRuns(); renderFindings(); renderCalib();
  await loadRun();
}

async function loadRun() {
  const r = run();
  if (!r) return;
  stop();
  frames = Array.from({ length: r.info.n_frames }, (_, i) => { const im = new Image(); im.src = `/run/${r.id}/frame/${i}.jpg`; return im; });
  $("scrub").max = r.info.n_frames - 1;
  showFrame(r.best.index);
  drawCurve();
  const best = frames[r.best.index];
  await new Promise((res) => (best.complete ? res() : (best.onload = res)));
  const base = $("base"), over = $("over");
  base.width = over.width = best.naturalWidth;
  base.height = over.height = best.naturalHeight;
  base.getContext("2d").drawImage(best, 0, 0);
  for (const kind of ["vessel", "device"]) { masks[kind] = await loadMask(r.id, kind); dirty[kind] = false; }
  drawOverlay();
}

async function loadMask(rid, kind) {
  const im = new Image();
  im.src = `/run/${rid}/mask/${kind}.png?t=${Date.now()}`;
  await new Promise((res) => (im.onload = res));
  const c = document.createElement("canvas");
  c.width = im.naturalWidth; c.height = im.naturalHeight;
  const g = c.getContext("2d");
  g.drawImage(im, 0, 0);
  const px = g.getImageData(0, 0, c.width, c.height).data;
  const data = new Uint8Array(c.width * c.height);
  for (let i = 0; i < data.length; i++) data[i] = px[i * 4 + 3] > 0 ? 1 : 0;
  return { w: c.width, h: c.height, data };
}

// ------------------------------------------------------------------ player --
function showFrame(n) {
  const r = run();
  cur = Math.max(0, Math.min(n, frames.length - 1));
  if (frames[cur]) $("player").src = frames[cur].src;
  $("scrub").value = cur;
  $("frame-no").textContent = `frame ${cur} / ${frames.length - 1}${cur === r.best.index ? " (best)" : ""}${r.best.moved[cur] ? " · moved" : ""}`;
  drawCurve();
}
function stop() { clearInterval(timer); timer = null; $("play").textContent = "Play"; }
$("play").addEventListener("click", () => {
  if (timer) return stop();
  const fps = run()?.info.fps || 10;
  timer = setInterval(() => showFrame(cur + 1 >= frames.length ? 0 : cur + 1), 1000 / fps);
  $("play").textContent = "Pause";
});
$("scrub").addEventListener("input", (e) => { stop(); showFrame(Number(e.target.value)); });
$("set-best").addEventListener("click", async () => {
  if (!confirm(`Use frame ${cur} as the best frame? The outline is redrawn and measurements on this run are repeated.`)) return;
  await api(`/api/run/${runId}/bestframe`, { index: cur });
  await refresh(); await loadRun();
});

function drawCurve() {
  const r = run(); if (!r) return;
  const c = r.best.curve, n = c.length, mx = Math.max(...c, 1e-9);
  const x = (i) => (n > 1 ? (i / (n - 1)) * 400 : 200), y = (v) => 85 - (v / mx) * 78;
  const moved = r.best.moved.map((m, i) => (m ? `<line x1="${x(i)}" x2="${x(i)}" y1="80" y2="90" stroke="#ff5a5a" stroke-width="2"/>` : "")).join("");
  $("curve").innerHTML = `<polyline fill="none" stroke="#ff9f1c" stroke-width="2" points="${c.map((v, i) => `${x(i)},${y(v)}`).join(" ")}"/>
    ${moved}<line x1="${x(r.best.index)}" x2="${x(r.best.index)}" y1="0" y2="90" stroke="#3ea6ff" stroke-width="2"/>
    <line x1="${x(cur)}" x2="${x(cur)}" y1="0" y2="90" stroke="#fff" stroke-width="1" stroke-dasharray="3 3"/>`;
}
$("curve").addEventListener("click", (e) => {
  const r = run(); if (!r) return;
  stop(); showFrame(Math.round((e.offsetX / $("curve").clientWidth) * (r.best.curve.length - 1)));
});

// ------------------------------------------------------------- canvas tools --
document.querySelectorAll("[data-tool]").forEach((b) => b.addEventListener("click", () => {
  tool = b.dataset.tool; pending = [];
  document.querySelectorAll("[data-tool]").forEach((x) => x.classList.toggle("active", x === b));
  renderCalib(); drawOverlay();
}));
["show-vessel", "show-device"].forEach((id) => $(id).addEventListener("change", drawOverlay));

function toImage(e) {
  const c = $("over");
  return [(e.offsetY / c.clientHeight) * c.height, (e.offsetX / c.clientWidth) * c.width]; // [row, col]
}

let painting = false;
const brushKind = () => (tool === "add" || tool === "erase" ? "vessel" : "device");
function paint(e) {
  const m = masks[brushKind()]; if (!m) return;
  const [r0, c0] = toImage(e), rad = Number($("brush").value), val = tool === "add" || tool === "device" ? 1 : 0;
  for (let r = Math.max(0, Math.floor(r0 - rad)); r <= Math.min(m.h - 1, r0 + rad); r++)
    for (let c = Math.max(0, Math.floor(c0 - rad)); c <= Math.min(m.w - 1, c0 + rad); c++)
      if ((r - r0) ** 2 + (c - c0) ** 2 <= rad * rad) m.data[r * m.w + c] = val;
  dirty[brushKind()] = true;
  drawOverlay();
}
const over = $("over");
over.addEventListener("mousedown", (e) => {
  if (!canEdit() || signed()) return;
  if (["add", "erase", "device", "undevice"].includes(tool)) { painting = true; paint(e); return; }
  if (["measure", "marker", "devcal"].includes(tool)) { pending.push(toImage(e)); drawOverlay(); if (pending.length === 2) finishClicks(); }
});
over.addEventListener("mousemove", (e) => { if (painting) paint(e); });
window.addEventListener("mouseup", () => { painting = false; });

async function finishClicks() {
  const [p1, p2] = pending.map((p) => p.map((v) => Math.round(v * 10) / 10));
  pending = [];
  try {
    if (tool === "measure") {
      if (dirty.vessel || dirty.device) await saveOutline();
      const res = await api(`/api/run/${runId}/measure`, { start: p1, end: p2 });
      findingId = res.id;
    } else if (tool === "marker") {
      const mm = parseFloat(prompt("Known length between the two points, in mm (from the ruler or marker bands):") || "");
      if (!(mm > 0)) { drawOverlay(); return; }
      const st = await api(`/api/run/${runId}/calibration`, { method: "marker", p1, p2, known_mm: mm });
      if (st.message) alert(st.message);
    } else if (tool === "devcal") {
      const dev = $("dev-select")?.value;
      if (!dev) { alert("Choose a measured device first."); drawOverlay(); return; }
      const st = await api(`/api/run/${runId}/calibration`, { method: "device", device_id: dev, p1, p2 });
      if (st.message) alert(st.message);
    }
  } finally { await refresh(); }
}

async function saveOutline() {
  for (const kind of ["vessel", "device"]) {
    if (!dirty[kind]) continue;
    const m = masks[kind], c = document.createElement("canvas");
    c.width = m.w; c.height = m.h;
    const g = c.getContext("2d"), img = g.createImageData(m.w, m.h);
    for (let i = 0; i < m.data.length; i++) if (m.data[i]) { img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255; img.data[i * 4 + 3] = 255; }
    g.putImageData(img, 0, 0);
    const blob = await new Promise((res) => c.toBlob(res, "image/png"));
    await api(`/api/run/${runId}/mask/${kind}`, blob, true);
    dirty[kind] = false;
  }
}
$("save-outline").addEventListener("click", async () => { await saveOutline(); await refresh(); });

function drawOverlay() {
  const c = $("over"); if (!c.width || !S) return;
  const g = c.getContext("2d");
  g.clearRect(0, 0, c.width, c.height);
  const img = g.createImageData(c.width, c.height);
  for (const [kind, rgb, show] of [["vessel", [255, 159, 28], $("show-vessel").checked], ["device", [230, 60, 230], $("show-device").checked]]) {
    const m = masks[kind]; if (!show || !m || m.w !== c.width) continue;
    for (let i = 0; i < m.data.length; i++) if (m.data[i]) { img.data[i * 4] = rgb[0]; img.data[i * 4 + 1] = rgb[1]; img.data[i * 4 + 2] = rgb[2]; img.data[i * 4 + 3] = 70; }
  }
  g.putImageData(img, 0, 0);
  const lw = Math.max(1, c.width / 500);
  for (const f of S.findings.filter((f) => f.run_id === runId && f.data.ok)) {
    const sel = f.id === findingId, P = f.data.profile;
    g.globalAlpha = sel ? 1 : 0.5;
    line(g, P.centres, sel ? "#ffffff" : "#bbbbbb", lw);
    for (const [a, b] of f.data.ref_ranges || []) line(g, P.centres.slice(a, b + 1), "#7CFC00", lw * 3);
    if (f.data.mld_points) line(g, f.data.mld_points, "#00e5ff", lw * 2.5);
    if (f.data.ref_points) line(g, f.data.ref_points, "#7CFC00", lw * 2);
    const m = P.centres[f.data.mld_index];
    g.fillStyle = "#00e5ff"; g.font = `${12 * lw}px sans-serif`;
    g.fillText(`#${f.id}${f.segment !== null ? " [" + f.segment + "]" : ""}`, m[1] + 8 * lw, m[0] - 8 * lw);
  }
  g.globalAlpha = 1;
  for (const p of pending) { g.fillStyle = "#ffb020"; g.beginPath(); g.arc(p[1], p[0], 4 * lw, 0, 7); g.fill(); }
}
function line(g, pts, color, w) {
  if (!pts || pts.length < 2) return;
  g.strokeStyle = color; g.lineWidth = w; g.beginPath();
  pts.forEach(([r, c], i) => (i ? g.lineTo(c, r) : g.moveTo(c, r)));
  g.stroke();
}

function renderCalib() {
  const r = run(); if (!r) return;
  const c = r.calibration || {};
  let html = `<div class="${c.reliable ? "" : "error"}"><b>Calibration:</b> ${c.reliable ? `${c.mm_per_px.toFixed(4)} mm/px (${esc(c.method)}, ±${(100 * (c.rel_error || 0)).toFixed(1)}%)` : "mm not available"} - ${esc(c.detail)}</div>`;
  if (tool === "marker") html += `<div class="muted">Click the two ends of a ruler span or marker of known size.</div>`;
  if (tool === "devcal") html += S.devices.length
    ? `<div>Device: <select id="dev-select"><option value="">choose</option>${S.devices.map((d) => `<option value="${esc(d.id)}">${esc(d.name)} (${d.measured_outer_mm} mm, measured by ${esc(d.measured_by)})</option>`).join("")}</select> then click across its two outer edges.</div>`
    : `<div class="error">No micrometer-measured devices are listed yet (access/rules/devices.json). Label sizes are never used.</div>`;
  if (tool === "measure") html += `<div class="muted">Click where the vessel segment starts (upstream, in the direction of blood flow), then where it ends.</div>`;
  $("calib").innerHTML = html;
}

// ---------------------------------------------------------------- findings --
const fmt = (v, e, u) => (v === null || v === undefined ? null : `${v}${e ? ` ± ${e}` : ""} ${u}`);

function renderFindings() {
  const list = S.findings.filter((f) => f.run_id === runId);
  const segOpts = (sel) => `<option value="">— segment —</option>` + S.segments.segments.map((s) => `<option value="${s.id}" ${String(sel) === String(s.id) ? "selected" : ""}>[${s.id}] ${esc(s.name)}</option>`).join("");
  $("findings").innerHTML = !list.length
    ? `<p class="muted">${S.laterality.status === "confirmed" ? "No measurement yet: use “Measure” on the best frame." : "Measurements start once the side is confirmed."}</p>`
    : list.map((f) => {
      const d = f.data;
      if (!d.ok) return `<div class="finding"><b>#${f.id}</b> could not be measured: ${esc((d.notes || []).join(" "))}</div>`;
      const mm = d.mm_available;
      const nums = [
        ["Narrowest (MLD)", mm ? fmt(d.mld_mm, d.mld_mm_err, "mm") : `${d.mld_px} px`],
        ["Normal (RVD)", d.percentage_available ? (mm ? fmt(d.rvd_mm, d.rvd_mm_err, "mm") : `${d.rvd_px} px`) : "not available"],
        ["Narrowing (%DS)", d.percentage_available ? fmt(d.ds_percent, d.ds_err, "%") : "percentage not available"],
        ["Length", d.percentage_available ? (mm ? fmt(d.length_mm, d.length_mm_err, "mm") : `${d.length_px} px`) : "-"],
      ];
      const can = canEdit() && !signed();
      return `<div class="finding ${f.id === findingId ? "selected" : ""}" data-f="${f.id}">
        <div class="toolbar"><b>#${f.id}</b><span class="pill">${esc(f.status)}</span><span class="conf ${d.confidence}">${esc(d.confidence)} confidence</span>
          <span class="small muted">run ${d.run_idx}, frame ${d.frame} · MLD point ${esc(d.mld_source)} · reference ${esc(d.ref_side)}</span></div>
        <div class="nums">${nums.map(([l, v]) => `<div><div class="numlbl">${l}</div><div class="num">${esc(v)}</div></div>`).join("")}</div>
        ${!mm ? `<div class="small" style="color:#ffd98a">mm not available: only pixels and percentages are shown (no reliable calibration).</div>` : ""}
        ${d.confidence_reasons?.length ? `<div class="small muted">Confidence lowered because: ${esc(d.confidence_reasons.join("; "))}</div>` : ""}
        ${d.notes?.length ? `<div class="small">${esc(d.notes.join(" "))}</div>` : ""}
        <svg class="profile" data-prof="${f.id}" viewBox="0 0 600 130" preserveAspectRatio="none"></svg>
        <div class="toolbar">
          <select data-seg="${f.id}" ${can ? "" : "disabled"}>${segOpts(f.segment)}</select>
          <button data-acc="${f.id}" ${can ? "" : "disabled"}>Accept</button>
          <button class="secondary" data-rej="${f.id}" ${can ? "" : "disabled"}>Reject</button>
          <span class="small muted" id="t-${f.id}"></span>
        </div></div>`;
    }).join("");
  list.filter((f) => f.data.ok).forEach(drawProfile);
  $("findings").querySelectorAll("[data-f]").forEach((el) => el.addEventListener("click", (e) => {
    if (e.target.closest("select,button,svg")) return;
    findingId = Number(el.dataset.f); renderFindings(); drawOverlay();
  }));
  $("findings").querySelectorAll("[data-seg]").forEach((el) => el.addEventListener("change", async () => refresh(await api(`/api/finding/${el.dataset.seg}`, { segment: el.value }))));
  $("findings").querySelectorAll("[data-acc]").forEach((el) => el.addEventListener("click", async () => refresh(await api(`/api/finding/${el.dataset.acc}`, { status: "accepted" }))));
  $("findings").querySelectorAll("[data-rej]").forEach((el) => el.addEventListener("click", async () => refresh(await api(`/api/finding/${el.dataset.rej}`, { status: "rejected" }))));
}

function drawProfile(f) {
  const svg = $("findings").querySelector(`[data-prof="${f.id}"]`); if (!svg) return;
  const P = f.data.profile, w = P.width_px, n = w.length;
  const vals = w.filter((v) => v !== null), mx = Math.max(...vals) * 1.1;
  const x = (i) => (i / (n - 1)) * 600, y = (v) => 125 - (v / mx) * 118;
  const pts = w.map((v, i) => (v === null ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(" ");
  const refs = (f.data.ref_ranges || []).map(([a, b]) => `<rect x="${x(a)}" width="${Math.max(2, x(b) - x(a))}" y="0" height="130" fill="#7CFC00" opacity=".15"/>`).join("");
  const [s0, s1] = f.data.span || [0, 0];
  svg.innerHTML = `${refs}<rect x="${x(s0)}" width="${Math.max(2, x(s1) - x(s0))}" y="0" height="130" fill="#00e5ff" opacity=".1"/>
    <polyline fill="none" stroke="#ff9f1c" stroke-width="2" points="${pts}"/>
    <line x1="${x(f.data.mld_index)}" x2="${x(f.data.mld_index)}" y1="0" y2="130" stroke="#00e5ff" stroke-width="2"/>
    ${refClicks.length && refClicks[0].f === f.id ? `<line x1="${x(refClicks[0].i)}" x2="${x(refClicks[0].i)}" y1="0" y2="130" stroke="#7CFC00" stroke-dasharray="4 3"/>` : ""}
    <text x="4" y="12" fill="#8ba0bd" font-size="11">width along the vessel (start → end)</text>`;
  svg.addEventListener("click", async (e) => {
    if (!canEdit() || signed()) return;
    const i = Math.round((e.offsetX / svg.clientWidth) * (n - 1));
    findingId = f.id;
    const keep = f.inputs || {};
    let body;
    if (e.shiftKey) {
      refClicks.push({ f: f.id, i });
      if (refClicks.length < 2 || refClicks[0].f !== f.id) { if (refClicks[0].f !== f.id) refClicks = [{ f: f.id, i }]; renderFindings(); return; }
      const [a, b] = [refClicks[0].i, refClicks[1].i].sort((p, q) => p - q);
      refClicks = [];
      body = { mld_index: keep.mld_index ?? null, ref_indices: Array.from({ length: b - a + 1 }, (_, k) => a + k) };
    } else {
      body = { mld_index: i, ref_indices: keep.ref_indices ?? null };
    }
    const res = await api(`/api/finding/${f.id}/points`, body);
    await refresh();
    const t = $(`t-${f.id}`); if (t) t.textContent = `recalculated in ${res.seconds} s`;
  });
}

// ---------------------------------------------------------- segments / danger --
function renderSegments() {
  const can = canEdit() && !signed();
  $("segments").innerHTML = S.segments.segments.map((s) => {
    const val = S.study.segment_status[String(s.id)] || "";
    const n = S.findings.filter((f) => f.status === "accepted" && String(f.segment) === String(s.id)).length;
    return `<tr class="seg-row"><td>[${s.id}] ${esc(s.name)}${n ? ` <span class="pill">${n} measured</span>` : ""}</td>
      <td><select data-segst="${s.id}" ${can ? "" : "disabled"}><option value="">— not answered —</option>
      ${S.segments.statuses.map((o) => `<option value="${o.id}" ${val === o.id ? "selected" : ""}>${esc(o.label)}</option>`).join("")}</select></td></tr>`;
  }).join("") + `<tr><td colspan="2" class="small muted">No dye in a segment is “not visible”, never “open”.</td></tr>`;
  $("segments").querySelectorAll("[data-segst]").forEach((el) => el.addEventListener("change", async () =>
    refresh(await api(`/api/study/${STUDY}/segment`, { segment: el.dataset.segst, status: el.value || null }))));
}

function renderDanger() {
  const can = canEdit() && !signed();
  $("danger").innerHTML = S.danger_items.items.map((d) => `<div class="danger-item"><div>${esc(d.name)}</div>
    ${S.danger_items.answers.map((a) => `<label><input type="radio" name="dg-${d.id}" value="${a.id}" data-dg="${d.id}" ${S.study.danger[d.id] === a.id ? "checked" : ""} ${can ? "" : "disabled"}> ${esc(a.label)}</label>`).join("")}</div>`).join("");
  $("danger").querySelectorAll("[data-dg]").forEach((el) => el.addEventListener("change", async () =>
    refresh(await api(`/api/study/${STUDY}/danger`, { item: el.dataset.dg, answer: el.value }))));
}

// -------------------------------------------------------------------- sign --
function renderSign() {
  const box = $("sign-card");
  if (signed()) {
    box.innerHTML = `<div class="signed"><b>Signed</b> by ${esc(S.study.signed_by)} on ${new Date(S.study.signed_at * 1000).toLocaleString()}.
      <div class="toolbar"><a class="btn" href="/study/${STUDY}/report.pdf">Report (PDF)</a><a class="btn" href="/study/${STUDY}/report.json">Data (JSON)</a></div></div>`;
    return;
  }
  const b = S.blockers;
  box.innerHTML = `<h3>Sign</h3>${b.length ? `<div class="small">Still needed before signing:</div><ul class="blockers small">${b.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : `<div class="banner ok">Everything is answered. Ready to sign.</div>`}
    <div class="toolbar"><input id="sign-name" placeholder="Type your name to sign: ${esc(S.user.username)}" ${b.length ? "disabled" : ""} style="min-width:260px"><button id="sign-btn" ${b.length ? "disabled" : ""}>Sign and create the report</button></div>
    <div class="small muted">Nothing leaves the system without your signature. After signing, the study can no longer be changed.</div>`;
  $("sign-btn")?.addEventListener("click", async () => refresh(await api(`/api/study/${STUDY}/sign`, { name: $("sign-name").value })));
}

window.addEventListener("beforeunload", (e) => { if (dirty.vessel || dirty.device) { e.preventDefault(); e.returnValue = ""; } });
refresh().then(loadRun);

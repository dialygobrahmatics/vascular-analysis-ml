// Import a study: files chosen from this computer (folder / .dcm / .zip), or a folder path.
"use strict";
const $ = (id) => document.getElementById(id);
let chosen = [];

const mb = (n) => (n / 1048576).toFixed(1);

function pick(list, kind) {
  // a chosen folder often also holds DICOMDIR indexes, thumbnails or notes; the server
  // keeps only real DICOM files, so everything is sent and checked there
  chosen = Array.from(list);
  const total = chosen.reduce((s, f) => s + f.size, 0);
  $("picked").textContent = chosen.length
    ? `${chosen.length} file(s) from ${kind}, ${mb(total)} MB`
    : "Nothing chosen yet";
  $("upload-btn").disabled = !chosen.length;
  $("upload-status").textContent = "";
}
$("pick-folder").addEventListener("change", (e) => pick(e.target.files, "the folder"));
$("pick-files").addEventListener("change", (e) => pick(e.target.files, "your selection"));

async function followJob(job, statusEl) {
  while (true) {
    await new Promise((res) => setTimeout(res, 1000));
    const s = await (await fetch(`/api/jobs/${job}`)).json();
    statusEl.textContent = `${s.message} (${s.percent ?? 0}%)`;
    setBar(s.percent ?? 0);
    if (s.state === "done") return s;
    if (s.state === "error") throw new Error(s.message);
  }
}

function setBar(pct) {
  $("upload-progress").hidden = false;
  $("upload-progress").querySelector(".bar").style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function fail(statusEl, btn, msg) {
  statusEl.innerHTML = `<div class="error"></div>`;
  statusEl.querySelector(".error").textContent = msg;
  window.feedback.error(msg);
  btn.classList.remove("busy");
  btn.disabled = false;
}

function finished(s) {
  window.feedback.ok("Study imported");
  const sid = (s.study_ids || [])[0];
  setTimeout(() => (location.href = sid ? `/study/${sid}` : "/"), 700);
}

$("upload-btn").addEventListener("click", () => {
  const btn = $("upload-btn"), status = $("upload-status");
  if (!chosen.length) return;
  btn.disabled = true;
  btn.classList.add("busy");
  window.feedback.info("Uploading the study…", 600000);
  const form = new FormData();
  for (const f of chosen) form.append("files", f, f.webkitRelativePath || f.name);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/import-upload");
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    const pct = (e.loaded / e.total) * 100;
    setBar(pct);
    status.textContent = `Uploading: ${Math.round(pct)}% (${mb(e.loaded)} of ${mb(e.total)} MB)`;
  };
  xhr.onerror = () => fail(status, btn, "The upload failed. Please try again.");
  xhr.onload = async () => {
    let j = {};
    try { j = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
    if (xhr.status !== 200) return fail(status, btn, j.error || `Upload failed (HTTP ${xhr.status})`);
    status.textContent = `${j.files} DICOM file(s) received. Analysing…`;
    window.feedback.info("Analysing the runs… this takes a few seconds per run", 600000);
    setBar(0);
    try { finished(await followJob(j.job, status)); } catch (e) { fail(status, btn, e.message); }
  };
  xhr.send(form);
});

$("import-btn").addEventListener("click", async () => {
  const btn = $("import-btn"), status = $("import-status");
  const folder = $("folder").value.trim();
  if (!folder) return;
  btn.disabled = true;
  btn.classList.add("busy");
  window.feedback.info("Importing the study… this takes a few seconds per run", 600000);
  try {
    const r = await fetch("/api/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ folder }) });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "Import failed");
    finished(await followJob(j.job, status));
  } catch (e) { fail(status, btn, e.message); }
});

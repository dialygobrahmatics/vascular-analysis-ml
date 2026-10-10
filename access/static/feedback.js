// Visible feedback for every action: the clicked button shows a spinner while the
// request runs, and a notice in the corner says "Saving…", then "Saved ✓" or the error.
"use strict";
(function () {
  const toast = document.createElement("div");
  toast.id = "toast";
  if (document.body) document.body.appendChild(toast);
  else document.addEventListener("DOMContentLoaded", () => document.body.appendChild(toast));
  let hideTimer = null, pending = 0, lastControl = null;

  // remember the control the person just used, so its request can show on it
  for (const ev of ["click", "change"]) {
    document.addEventListener(ev, (e) => {
      const el = e.target.closest("button, select, input");
      if (el) lastControl = el;
    }, true);
  }

  function show(kind, text, ms) {
    clearTimeout(hideTimer);
    toast.className = "show " + (kind || "");
    toast.innerHTML = kind === "busy" ? `<span class="dot"></span>` : kind === "ok" ? "✓ " : kind === "err" ? "✕ " : "";
    toast.appendChild(document.createTextNode(text));
    if (ms) hideTimer = setTimeout(() => (toast.className = ""), ms);
  }

  window.feedback = {
    // wraps one request: spinner on the button that started it, notice while it runs
    async track(promiseFn, busyText = "Saving…", okText = "Saved") {
      const btn = lastControl && lastControl.tagName === "BUTTON" ? lastControl : null;
      lastControl = null;
      pending++;
      if (btn) btn.classList.add("busy");
      show("busy", busyText);
      try {
        const out = await promiseFn();
        pending--;
        if (!pending) show("ok", okText, 1600);
        return out;
      } catch (e) {
        pending--;
        show("err", e.message || "Something went wrong", 5000);
        throw e;
      } finally {
        if (btn) btn.classList.remove("busy");
      }
    },
    info(text, ms = 2500) { show("", text, ms); },
    ok(text, ms = 1600) { show("ok", text, ms); },
    error(text, ms = 5000) { show("err", text, ms); },
  };
})();

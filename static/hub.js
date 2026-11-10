// Hub page: shows the signed-in reader's progress on each course card that
// opts in with data-progress-url. The request is a simple cross-origin GET
// with credentials and no custom headers, so the browser sends no preflight.
// Any failure leaves the card unchanged.
(() => {
  "use strict";
  const TIMEOUT_MS = 5000;

  function summary(data) {
    if (!data || typeof data !== "object") return null;
    const { done, total, last } = data;
    if (!Number.isInteger(done) || !Number.isInteger(total) || done < 0 || total <= 0 || done > total) return null;
    const title = last && typeof last.title === "string" ? last.title.trim() : "";
    return { done, total, title };
  }

  function render(out, s) {
    const line = document.createElement("span");
    line.className = "hub-progress-count";
    line.textContent = `${s.done} of ${s.total} done`;
    const track = document.createElement("div");
    track.className = "progress-track";
    const fill = document.createElement("div");
    fill.className = "progress-fill";
    fill.style.width = `${Math.round((s.done / s.total) * 100)}%`;
    track.append(fill);
    out.replaceChildren(line, track);
    if (s.title) {
      const last = document.createElement("span");
      last.className = "hub-progress-last";
      last.textContent = `Last: ${s.title}`;
      out.append(last);
    }
  }

  async function load(card) {
    const out = card.querySelector("[data-progress-out]");
    const url = card.dataset.progressUrl;
    if (!out || !url) return;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    try {
      // redirect: "follow" is the default; an Authelia redirect ends in a
      // CORS failure or a non-JSON page, and both land in the catch below.
      const res = await fetch(url, { credentials: "include", signal: ctl.signal });
      if (res.status !== 200) return;
      const s = summary(await res.json());
      if (s) render(out, s);
    } catch {
      // Network error, CORS error, timeout, or bad JSON: leave the card as is.
    } finally {
      clearTimeout(timer);
    }
  }

  document.querySelectorAll("[data-progress-url]").forEach((card) => { load(card); });
})();

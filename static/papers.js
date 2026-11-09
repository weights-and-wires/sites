// Paper pages: reading progress, the full prerequisite chain, and the
// exercise editors and test runner. Progress lives in localStorage only.
(() => {
  "use strict";
  const KEY = "papers.done";
  const TIMEOUT_MS = 20000;

  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
    },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode */ } },
  };
  const doneSet = () => new Set(store.get(KEY, []));
  window.PapersProgress = { doneSet, KEY };

  // Mark every element that names a paper with data-paper-id as done or not.
  function paintProgress() {
    const done = doneSet();
    document.querySelectorAll("[data-paper-id]").forEach((el) => {
      const isDone = done.has(el.dataset.paperId);
      el.classList.toggle("is-done", isDone);
      const state = el.querySelector(".progress-state");
      if (state) state.textContent = isDone ? "Done" : "";
    });
    const box = document.querySelector("[data-progress-toggle]");
    if (box) box.checked = done.has(box.dataset.progressToggle);
    const count = document.querySelector("[data-chain-done]");
    if (count) {
      const ids = [...document.querySelectorAll(".chain-list [data-paper-id]")].map((el) => el.dataset.paperId);
      count.textContent = `${ids.filter((id) => done.has(id)).length} of ${ids.length} done`;
    }
  }

  const toggle = document.querySelector("[data-progress-toggle]");
  if (toggle) {
    toggle.addEventListener("change", () => {
      const done = doneSet();
      if (toggle.checked) done.add(toggle.dataset.progressToggle); else done.delete(toggle.dataset.progressToggle);
      store.set(KEY, [...done]);
      paintProgress();
    });
  }
  window.addEventListener("storage", (e) => { if (e.key === KEY) paintProgress(); });

  document.querySelectorAll("[data-chain-toggle]").forEach((btn) => {
    const panel = document.getElementById(btn.getAttribute("aria-controls"));
    btn.addEventListener("click", () => {
      const open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!open));
      panel.hidden = open;
      btn.textContent = open ? btn.dataset.closedLabel : "Hide full chain";
    });
  });
  paintProgress();

  // Exercises.
  const blocks = document.querySelectorAll("[data-exercise]");
  if (!blocks.length) return;
  const page = document.body.dataset;
  let worker = null;
  let seq = 0;

  function runInWorker(submission, tests, onStatus) {
    return new Promise((resolve) => {
      if (!worker) worker = new Worker(page.workerUrl);
      const id = ++seq;
      const timer = setTimeout(() => {
        worker.terminate();
        worker = null;
        resolve({ type: "error", text: `Stopped after ${TIMEOUT_MS / 1000} s. Look for an infinite loop or a very large array.` });
      }, TIMEOUT_MS);
      const w = worker;
      const onMessage = (e) => {
        if (e.data.id !== id) return;
        if (e.data.type === "status") { onStatus(e.data.text); return; }
        clearTimeout(timer);
        w.removeEventListener("message", onMessage);
        resolve(e.data);
      };
      w.addEventListener("message", onMessage);
      w.addEventListener("error", (e) => {
        clearTimeout(timer);
        worker = null;
        resolve({ type: "error", text: e.message || "The runner failed to start." });
      }, { once: true });
      w.postMessage({ id, harnessURL: page.harnessUrl, submission, tests });
    });
  }

  function text(tag, cls, value) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    el.textContent = value;
    return el;
  }

  blocks.forEach((block) => {
    const key = `papers.code.${block.dataset.exercise}`;
    const starter = block.querySelector("template[data-starter]").content.textContent;
    const tests = block.querySelector("template[data-tests]")?.content.textContent || "";
    const host = block.querySelector("[data-editor]");
    const fallback = host.querySelector("textarea");
    const saved = store.get(key, null);
    let editor = {
      get: () => fallback.value,
      set: (v) => { fallback.value = v; },
    };
    if (saved !== null) editor.set(saved);
    if (window.PapersEditor) {
      const doc = fallback.value;
      host.textContent = "";
      editor = window.PapersEditor.mount(host, doc, {
        label: `Code for ${block.querySelector("h3").textContent}`,
        onChange: (v) => store.set(key, v),
      });
    } else {
      fallback.addEventListener("input", () => store.set(key, fallback.value));
    }

    const out = block.querySelector("[data-output]");
    const run = block.querySelector("[data-run]");
    const reset = block.querySelector("[data-reset]");
    const solBtn = block.querySelector("[data-show-solution]");
    const sol = block.querySelector("[data-solution]");

    reset?.addEventListener("click", () => {
      if (confirm("Replace your code with the starter?")) { editor.set(starter); store.set(key, starter); }
    });
    solBtn?.addEventListener("click", () => {
      sol.hidden = !sol.hidden;
      solBtn.textContent = sol.hidden ? "Show solution" : "Hide solution";
    });

    run?.addEventListener("click", async () => {
      run.disabled = true;
      out.hidden = false;
      out.replaceChildren(text("p", "runner-status", "Starting…"));
      const res = await runInWorker(editor.get(), tests, (s) => { out.firstChild.textContent = s; });
      run.disabled = false;
      if (solBtn) solBtn.disabled = false;
      out.replaceChildren();
      if (res.type === "error") {
        out.append(text("p", "runner-error", res.text));
        return;
      }
      const passed = res.results.filter((r) => r.ok).length;
      const all = passed === res.results.length;
      out.append(text("p", all ? "runner-summary is-pass" : "runner-summary is-fail",
        all ? `All ${passed} tests passed.` : `${passed} of ${res.results.length} tests passed.`));
      const list = document.createElement("ul");
      list.className = "runner-results";
      res.results.forEach((r) => {
        const li = text("li", r.ok ? "is-pass" : "is-fail", "");
        li.append(text("span", "runner-mark", r.ok ? "PASS" : "FAIL"), text("code", "", r.name));
        if (!r.ok) li.append(text("pre", "runner-message", r.message));
        list.append(li);
      });
      out.append(list);
      if (res.stdout && res.stdout.trim()) {
        const d = document.createElement("details");
        d.append(text("summary", "", "Printed output"), text("pre", "runner-stdout", res.stdout));
        out.append(d);
      }
    });
  });
})();

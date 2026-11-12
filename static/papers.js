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
  function text(tag, cls, value) {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    el.textContent = value;
    return el;
  }
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


  // Reading paths: checkboxes that mark a paper read, and progress counts.
  function paintPaths() {
    const done = doneSet();
    document.querySelectorAll("[data-mark-done]").forEach((box) => { box.checked = done.has(box.dataset.markDone); });
    document.querySelectorAll("[data-path-papers]").forEach((el) => {
      const ids = el.dataset.pathPapers.split(" ").filter(Boolean);
      const n = ids.filter((id) => done.has(id)).length;
      const label = el.querySelector("[data-path-progress]");
      if (label) label.textContent = n ? `${n} of ${ids.length} read` : `${ids.length} papers`;
      const fill = el.querySelector("[data-path-fill]");
      if (fill) fill.style.width = `${Math.round((100 * n) / ids.length)}%`;
    });
  }
  document.querySelectorAll("[data-mark-done]").forEach((box) => {
    box.addEventListener("change", () => {
      const done = doneSet();
      if (box.checked) done.add(box.dataset.markDone); else done.delete(box.dataset.markDone);
      store.set(KEY, [...done]);
      paintProgress();
      paintPaths();
    });
  });
  paintPaths();
  window.addEventListener("storage", (e) => { if (e.key === KEY) paintPaths(); });

  // The prerequisite graph: highlight ancestors and descendants on hover or
  // focus, and filter by track and tier.
  const svg = document.querySelector("[data-graph]");
  if (svg) {
    const nodes = [...svg.querySelectorAll(".graph-node")];
    const edges = [...svg.querySelectorAll(".graph-edges path")];
    const needs = new Map(nodes.map((n) => [n.dataset.id, n.dataset.needs.split(" ").filter(Boolean)]));
    const next = new Map(nodes.map((n) => [n.dataset.id, []]));
    needs.forEach((list, id) => list.forEach((r) => next.get(r).push(id)));
    const reach = (start, adj) => {
      const seen = new Set();
      const stack = [...adj.get(start)];
      while (stack.length) { const id = stack.pop(); if (!seen.has(id)) { seen.add(id); stack.push(...adj.get(id)); } }
      return seen;
    };
    const clear = () => {
      svg.classList.remove("is-focused");
      svg.querySelectorAll(".is-anc, .is-desc, .is-self").forEach((el) => el.classList.remove("is-anc", "is-desc", "is-self"));
    };
    const focusNode = (node) => {
      clear();
      const id = node.dataset.id;
      const anc = reach(id, needs);
      const desc = reach(id, next);
      svg.classList.add("is-focused");
      node.classList.add("is-self");
      nodes.forEach((n) => {
        if (anc.has(n.dataset.id)) n.classList.add("is-anc");
        if (desc.has(n.dataset.id)) n.classList.add("is-desc");
      });
      edges.forEach((e) => {
        const f = e.dataset.from, t = e.dataset.to;
        if ((anc.has(f) || f === id) && (anc.has(t) || t === id)) e.classList.add("is-anc");
        if ((desc.has(t) || t === id) && (desc.has(f) || f === id)) e.classList.add("is-desc");
      });
    };
    nodes.forEach((n) => {
      n.addEventListener("mouseenter", () => focusNode(n));
      n.addEventListener("focus", () => focusNode(n));
      n.addEventListener("mouseleave", clear);
      n.addEventListener("blur", clear);
    });
    const done = doneSet();
    nodes.forEach((n) => n.classList.toggle("is-done", done.has(n.dataset.id)));
    const form = document.querySelector("[data-graph-filters]");
    form?.addEventListener("change", () => {
      const tracks = new Set(new FormData(form).getAll("track"));
      const tiers = new Set(new FormData(form).getAll("tier"));
      const hidden = new Set();
      nodes.forEach((n) => {
        const off = !tracks.has(n.dataset.track) || !tiers.has(n.dataset.tier);
        n.classList.toggle("is-filtered", off);
        if (off) { hidden.add(n.dataset.id); n.setAttribute("tabindex", "-1"); } else { n.removeAttribute("tabindex"); }
      });
      edges.forEach((e) => e.classList.toggle("is-filtered", hidden.has(e.dataset.from) || hidden.has(e.dataset.to)));
    });
  }

  // Search: press / anywhere, or click the search button.
  const dialog = document.querySelector("[data-search]");
  if (dialog) {
    const input = dialog.querySelector("[data-search-input]");
    const list = dialog.querySelector("[data-search-results]");
    let index = null;
    let active = 0;
    const load = async () => {
      if (!index) {
        try { index = await (await fetch("/search.json")).json(); } catch { index = []; }
      }
      return index;
    };
    const score = (entry, terms) => {
      let total = 0;
      for (const t of terms) {
        const title = entry.title.toLowerCase();
        const heads = (entry.headings || []).join(" ").toLowerCase();
        const body = entry.excerpt.toLowerCase();
        const url = entry.url.toLowerCase();
        let s = 0;
        if (title.includes(t)) s += title.startsWith(t) ? 12 : 8;
        if (url.includes(t)) s += 6;
        if (heads.includes(t)) s += 3;
        if (body.includes(t)) s += 1;
        if (!s) return 0;
        total += s;
      }
      return total;
    };
    const render = async () => {
      const terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      const data = await load();
      const hits = terms.length
        ? data.map((e) => [score(e, terms), e]).filter(([s]) => s > 0).sort((a, b) => b[0] - a[0]).slice(0, 12).map(([, e]) => e)
        : [];
      active = 0;
      list.replaceChildren(...hits.map((e, i) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.className = "search-result";
        a.href = e.url;
        a.id = `search-hit-${i}`;
        a.setAttribute("role", "option");
        a.setAttribute("aria-selected", String(i === 0));
        a.append(text("span", "search-result-title", e.title), text("span", "search-result-detail", e.excerpt.slice(0, 140)));
        li.append(a);
        return li;
      }));
      if (terms.length && !hits.length) list.replaceChildren(text("li", "search-result-detail empty-state", "No matches."));
    };
    const open = () => { if (!dialog.open) { dialog.showModal(); input.select(); load(); } };
    document.querySelector("[data-search-open]")?.addEventListener("click", open);
    document.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (e.key === "/" && !dialog.open && !["input", "textarea"].includes(tag) && !e.target.isContentEditable) {
        e.preventDefault();
        open();
      }
    });
    input.addEventListener("input", render);
    input.addEventListener("keydown", (e) => {
      const links = [...list.querySelectorAll("a")];
      if (!links.length) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        active = (active + (e.key === "ArrowDown" ? 1 : links.length - 1)) % links.length;
        links.forEach((l, i) => l.setAttribute("aria-selected", String(i === active)));
        links[active].scrollIntoView({ block: "nearest" });
      } else if (e.key === "Enter") {
        e.preventDefault();
        location.href = links[active].href;
      }
    });
    dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  }

  // Exercises and deep-dive cells share one Pyodide worker.
  const blocks = document.querySelectorAll("[data-exercise]");
  const cells = document.querySelectorAll("[data-cell]");
  if (!blocks.length && !cells.length) return;
  const page = document.body.dataset;
  let worker = null;
  let seq = 0;

  // runInWorker sends msg (tests, or a cell's code) to the worker and resolves
  // with its reply, or with an error after TIMEOUT_MS.
  function runInWorker(msg, onStatus) {
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
      w.postMessage({ id, harnessURL: page.harnessUrl, ...msg });
    });
  }

  // mountEditor puts the code editor (or the plain textarea fallback) in a
  // block and keeps its text in localStorage under key.
  function mountEditor(block, key, label) {
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
      editor = window.PapersEditor.mount(host, doc, { label, onChange: (v) => store.set(key, v) });
    } else {
      fallback.addEventListener("input", () => store.set(key, fallback.value));
    }
    return editor;
  }

  cells.forEach((cell) => {
    const key = `dives.code.${cell.dataset.cell}`;
    const original = cell.querySelector("template[data-code]").content.textContent;
    const editor = mountEditor(cell, key, `Python cell ${cell.dataset.cell}`);
    const out = cell.querySelector("[data-output]");
    const run = cell.querySelector("[data-run]");
    cell.querySelector("[data-reset]")?.addEventListener("click", () => {
      if (confirm("Replace your code with the original?")) { editor.set(original); store.set(key, original); }
    });
    run.addEventListener("click", async () => {
      run.disabled = true;
      out.hidden = false;
      out.replaceChildren(text("p", "runner-status", "Starting…"));
      const res = await runInWorker({ code: editor.get() }, (s) => { out.firstChild.textContent = s; });
      run.disabled = false;
      out.replaceChildren();
      if (res.type === "error") {
        out.append(text("p", "runner-error", res.text));
        return;
      }
      out.append(text("pre", "runner-stdout", res.stdout.trim() || "(no output)"));
      // A dive's widgets listen for this to draw what the cell computed.
      cell.dispatchEvent(new CustomEvent("cell:result", { bubbles: true, detail: { name: cell.dataset.cell, result: res.result, stdout: res.stdout } }));
    });
  });

  blocks.forEach((block) => {
    const key = `papers.code.${block.dataset.exercise}`;
    const starter = block.querySelector("template[data-starter]").content.textContent;
    const tests = block.querySelector("template[data-tests]")?.content.textContent || "";
    const editor = mountEditor(block, key, `Code for ${block.querySelector("h3").textContent}`);

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
      const res = await runInWorker({ submission: editor.get(), tests }, (s) => { out.firstChild.textContent = s; });
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

// Widgets for the "Qwen, minus the E" dive. Toy numbers here are ours.
(() => {
  "use strict";
  const D = window.Dives;
  const fmt = (x, d = 2) => (x >= 0 ? "" : "-") + Math.abs(x).toFixed(d);
  const sgn = (x, d = 2) => (x >= 0 ? "+" : "-") + Math.abs(x).toFixed(d);

  function text(ctx, s, x, y, color, font, align) {
    ctx.fillStyle = color;
    ctx.font = font;
    ctx.textAlign = align || "left";
    ctx.textBaseline = "middle";
    ctx.fillText(s, x, y);
  }

  // ---------------------------------------------------------------- 1. mask
  D.mount("letter-mask", (el) => {
    el.append(D.el("p", "widget-title", "Letter mask on a next-token distribution"));
    const toks = ["the", "walk", "a", "dog", "here", "run", "see", "and"];
    const hasE = toks.map((t) => t.includes("e"));
    const presets = {
      likes: [3.0, 1.0, 0.5, 0.0, 2.2, 0.8, 1.8, 0.4],
      plain: [0.5, 1.0, 1.2, 1.0, 0.4, 1.1, 0.3, 1.2],
    };
    const controls = D.el("div", "widget-controls");
    el.append(controls);
    let masked = true;
    const logits = presets.likes.slice();
    const sliders = [];
    const plot = D.canvas(el, { height: (w) => 8 * 40 + 16, label: "Probability of each candidate token, before and after the mask" });
    const out = D.readout(el);
    const update = () => { logits.forEach((_, i) => { logits[i] = sliders[i].get(); }); plot.redraw(); };
    D.toggle(controls, { label: "Apply the 'e' mask", checked: true, onChange: (v) => { masked = v; plot.redraw(); } });
    D.button(controls, "Model that favors e-words", () => { presets.likes.forEach((v, i) => sliders[i].set(v)); });
    D.button(controls, "Even model", () => { presets.plain.forEach((v, i) => sliders[i].set(v)); });
    toks.forEach((t, i) => {
      sliders.push(D.slider(controls, {
        label: `Logit: ${t}`, min: -2, max: 4, step: 0.1, value: logits[i],
        format: (v) => v.toFixed(1), onInput: update,
      }));
    });
    const softmax = (arr) => {
      const m = Math.max(...arr.filter((v) => isFinite(v)));
      const e = arr.map((v) => (isFinite(v) ? Math.exp(v - m) : 0));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    };
    plot.draw = (ctx, w, h) => {
      const c = D.colors();
      const before = softmax(logits);
      const after = softmax(logits.map((v, i) => (masked && hasE[i] ? -Infinity : v)));
      const left = 52, right = w - 44, bw = right - left;
      const font = `12px ${c.sans}`;
      text(ctx, "top bar: model's own probability   bottom bar: probability when sampling", left, 8, c.muted, `11px ${c.sans}`);
      toks.forEach((t, i) => {
        const y = 22 + i * 40;
        text(ctx, t, left - 8, y + 16, hasE[i] ? c.error : c.ink, `13px ${c.mono}`, "right");
        ctx.fillStyle = c.line;
        ctx.fillRect(left, y + 2, Math.max(1, before[i] * bw), 13);
        const dead = masked && hasE[i];
        if (dead) {
          text(ctx, "-inf", left + 4, y + 25, c.error, font);
        } else {
          ctx.fillStyle = c.forest;
          ctx.fillRect(left, y + 18, Math.max(1, after[i] * bw), 13);
        }
        text(ctx, `${(before[i] * 100).toFixed(0)}%`, left + before[i] * bw + 5, y + 8, c.muted, font);
        if (!dead) text(ctx, `${(after[i] * 100).toFixed(0)}%`, left + after[i] * bw + 5, y + 24, c.ink, font);
      });
      const lost = before.reduce((a, p, i) => a + (hasE[i] ? p : 0), 0);
      const top = (arr) => toks[arr.indexOf(Math.max(...arr))];
      out.textContent = masked
        ? `The mask removes ${(lost * 100).toFixed(0)}% of the model's probability. Favorite before: "${top(before)}". Favorite after: "${top(after)}". Every allowed token gains the same factor, ${(1 / (1 - lost)).toFixed(2)}x.`
        : `The mask is off, so ${(lost * 100).toFixed(0)}% of the probability sits on tokens that contain an 'e'. Sampling breaks the rule about that often at this step.`;
    };
    plot.redraw();
    el.append(D.el("p", "widget-caption", "Drag a logit up on an e-word and watch the model lose its favorite. The mask keeps the order of the allowed words, but it can leave only poor ones."));
  });

  // ------------------------------------------------------------- 2. scorer
  D.mount("group-scorer", (el) => {
    el.append(D.el("p", "widget-title", "GRPO group scorer"));
    const controls = D.el("div", "widget-controls");
    el.append(controls);
    const init = [[5, 5], [4, 4], [5, 3], [2, 2], [4, 5], [3, 4], [5, 2], [4, 3]];
    const rows = init.map((cr) => ({ c: cr[0], r: cr[1] }));
    let mode = "sum";
    D.select(controls, {
      label: "Reward", value: "sum",
      options: [["sum", "Sum: ((c-1)+(r-1))/8"], ["min", "Min: (min(c,r)-1)/4 (our scaling)"]],
      onChange: (v) => { mode = v; plot.redraw(); },
    });
    const table = D.el("div", "qwn-table");
    controls.append(table);
    const mk = (i, key) => {
      const s = document.createElement("select");
      s.setAttribute("aria-label", `Answer ${i + 1} ${key === "c" ? "coherence" : "responds"} score`);
      for (let v = 1; v <= 5; v++) { const o = document.createElement("option"); o.value = v; o.textContent = v; s.append(o); }
      s.value = rows[i][key];
      s.addEventListener("change", () => { rows[i][key] = Number(s.value); plot.redraw(); });
      return s;
    };
    const head = D.el("div", "qwn-row qwn-head");
    head.append(D.el("span", "", "Answer"), D.el("span", "", "Coherence"), D.el("span", "", "Responds"));
    table.append(head);
    rows.forEach((_, i) => {
      const row = D.el("div", "qwn-row");
      row.append(D.el("span", "", `${i + 1}`), mk(i, "c"), mk(i, "r"));
      table.append(row);
    });
    const plot = D.canvas(el, { height: 8 * 30 + 40, label: "Reward and advantage for each answer in the group" });
    const out = D.readout(el);
    plot.draw = (ctx, w, h) => {
      const c = D.colors();
      const R = rows.map((r) => (mode === "sum" ? (r.c - 1 + r.r - 1) / 8 : (Math.min(r.c, r.r) - 1) / 4));
      const mean = R.reduce((a, b) => a + b, 0) / R.length;
      const sd = Math.sqrt(R.reduce((a, b) => a + (b - mean) ** 2, 0) / R.length);
      const A = R.map((x) => (sd < 1e-9 ? 0 : (x - mean) / sd));
      const use = rows.map((r) => r.c >= 4 && r.r >= 4);
      const mid = w * 0.62, scale = Math.max(1, w * 0.3 - 6) / 2.2;
      text(ctx, "answer (c, r) reward", 4, 10, c.muted, `11px ${c.sans}`);
      text(ctx, "advantage", mid, 10, c.muted, `11px ${c.sans}`, "center");
      ctx.strokeStyle = c.strongLine; ctx.beginPath(); ctx.moveTo(mid, 18); ctx.lineTo(mid, 8 * 30 + 24); ctx.stroke();
      rows.forEach((r, i) => {
        const y = 24 + i * 30;
        text(ctx, `${i + 1} (${r.c},${r.r}) ${R[i].toFixed(2)}`, 4, y + 12, c.ink, `12px ${c.mono}`);
        const len = A[i] * scale;
        ctx.fillStyle = A[i] >= 0 ? c.forest : c.error;
        ctx.fillRect(Math.min(mid, mid + len), y + 3, Math.max(1, Math.abs(len)), 18);
        text(ctx, sgn(A[i]), A[i] >= 0 ? mid + len + 4 : mid + 6, y + 12, c.ink, `11px ${c.mono}`);
        // usable marker
        ctx.beginPath(); ctx.arc(w - 10, y + 12, 5, 0, 7);
        if (use[i]) { ctx.fillStyle = c.forest; ctx.fill(); } else { ctx.strokeStyle = c.muted; ctx.stroke(); }
      });
      text(ctx, "usable", w - 10, 10, c.muted, `11px ${c.sans}`, "right");
      let msg = `Group mean ${mean.toFixed(2)}, population std ${sd.toFixed(2)}. Filled dot: both scores at least 4, so the answer counts as usable.`;
      if (sd < 1e-9) msg = "Every reward is equal, so every advantage is 0 and this group teaches the model nothing.";
      const r53 = rows.findIndex((r, i) => (r.c === 5 && r.r === 3) || (r.c === 3 && r.r === 5));
      const tie = rows.some((a, i) => rows.some((b, j) => i < j && R[i] === R[j] && use[i] !== use[j]));
      if (tie) msg += " Two answers tie on reward but only one is usable: the sum cannot tell them apart.";
      else if (mode === "min" && r53 >= 0) msg += " Under the min, the lopsided answer now ranks below the balanced one.";
      out.textContent = msg;
    };
    plot.redraw();
    el.append(D.el("p", "widget-caption", "Under the sum, a (5,3) answer and a (4,4) answer earn the same reward; switch to the min and the unusable one drops. The usable test needs both scores at 4 or more, which is a min, not a sum."));
  });

  // ----------------------------------------------------------- 3. toy curves
  D.mount("toy-curves", (el) => {
    el.append(D.el("p", "widget-title", "Toy GRPO training curves (from the cell above)"));
    const controls = D.el("div", "widget-controls");
    el.append(controls);
    let data = null, which = "usable";
    D.select(controls, {
      label: "Plot", value: "usable",
      options: [["usable", "Usable rate"], ["len", "Words per answer"]],
      onChange: (v) => { which = v; plot.redraw(); },
    });
    const plot = D.canvas(el, { aspect: 0.55, label: "Training curves for sum and min rewards in the toy" });
    const out = D.readout(el);
    plot.draw = (ctx, w, h) => {
      const c = D.colors();
      if (!data) { text(ctx, "Run the toy cell above to draw the curves.", w / 2, h / 2, c.muted, `13px ${c.sans}`, "center"); return; }
      const L = 40, B = h - 26, T = 12, Rr = w - 12;
      const k = which === "usable" ? 0 : 1;
      const all = [...data.sum, ...data.min].map((p) => p[k]);
      const hi = which === "usable" ? 1 : Math.ceil(Math.max(...all) + 0.5);
      const lo = 0;
      ctx.strokeStyle = c.line; ctx.fillStyle = c.muted; ctx.font = `11px ${c.sans}`;
      for (let i = 0; i <= 4; i++) {
        const y = B - (B - T) * (i / 4);
        ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(Rr, y); ctx.stroke();
        const v = lo + (hi - lo) * (i / 4);
        text(ctx, which === "usable" ? `${(v * 100).toFixed(0)}%` : v.toFixed(1), L - 5, y, c.muted, `11px ${c.sans}`, "right");
      }
      const n = data.sum.length - 1;
      const X = (i) => L + (Rr - L) * (i / n);
      const Y = (v) => B - (B - T) * ((v - lo) / (hi - lo));
      [["sum", c.series[0], "sum reward"], ["min", c.series[1], "min reward"]].forEach(([key, col, name], s) => {
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
        data[key].forEach((p, i) => (i ? ctx.lineTo(X(i), Y(p[k])) : ctx.moveTo(X(i), Y(p[k]))));
        ctx.stroke();
        ctx.fillStyle = col; ctx.fillRect(L + 6 + s * 100, T + 2, 12, 3);
        text(ctx, name, L + 22 + s * 100, T + 4, c.ink, `11px ${c.sans}`);
      });
      ctx.lineWidth = 1;
      text(ctx, "0", L, B + 14, c.muted, `11px ${c.sans}`, "center");
      text(ctx, `${n * data.every} GRPO steps`, Rr, B + 14, c.muted, `11px ${c.sans}`, "right");
      const a = data.sum, b = data.min, last = a.length - 1;
      out.textContent = `Start: ${(a[0][0] * 100).toFixed(0)}% usable, ${a[0][1].toFixed(1)} words. End (sum): ${(a[last][0] * 100).toFixed(0)}%, ${a[last][1].toFixed(1)} words. End (min): ${(b[last][0] * 100).toFixed(0)}%, ${b[last][1].toFixed(1)} words.`;
    };
    document.addEventListener("cell:result", (e) => {
      if (e.detail.name === "qwn/toy" && e.detail.result && e.detail.result.sum) { data = e.detail.result; plot.redraw(); }
    });
    plot.redraw();
    el.append(D.el("p", "widget-caption", "Our toy learns to be usable under the mask, but its answers do not shrink, so it does not reproduce the post's short-answer drift."));
  });

  // ------------------------------------------------------------ 4. memory
  D.mount("memory-calc", (el) => {
    el.append(D.el("p", "widget-title", "Fine-tuning memory napkin (2B-parameter model)"));
    const controls = D.el("div", "widget-controls");
    el.append(controls);
    const st = { lora: false, ckpt: false, chunk: false, q4: false };
    const redo = () => plot.redraw();
    const batch = D.slider(controls, { label: "Batch size", min: 1, max: 16, value: 4, onInput: redo });
    const seq = D.slider(controls, { label: "Sequence length", min: 512, max: 8192, step: 256, value: 2048, onInput: redo });
    D.toggle(controls, { label: "LoRA (train about 1% of parameters)", checked: false, onChange: (v) => { st.lora = v; if (!v) { st.q4 = false; q4.set(false); } redo(); } });
    D.toggle(controls, { label: "Gradient checkpointing", checked: false, onChange: (v) => { st.ckpt = v; redo(); } });
    D.toggle(controls, { label: "Chunked cross-entropy", checked: false, onChange: (v) => { st.chunk = v; redo(); } });
    const q4 = D.toggle(controls, { label: "4-bit base weights (QLoRA, needs LoRA)", checked: false, onChange: (v) => { st.q4 = v && st.lora; if (v && !st.lora) q4.input.checked = false; redo(); } });
    const plot = D.canvas(el, { height: 150, label: "Stacked bar of training memory by component" });
    const list = D.el("div", "qwn-legend");
    el.append(list);
    const out = D.readout(el);
    plot.draw = (ctx, w, h) => {
      const c = D.colors();
      const b = batch.get(), L = seq.get(), GB = 1e9;
      const params = 2e9, d = 2048, layers = 28, V = 150000;
      const weights = params * (st.q4 ? 0.5 : 2) / GB;
      const train = st.lora ? 20e6 : params;
      const grads = train * 2 / GB, adam = train * 8 / GB;
      const act = st.ckpt ? (layers * b * L * d * 2 + b * L * d * 2 * 12) / GB : b * L * d * 2 * 12 * layers / GB;
      const logit = st.chunk ? 0.3 : b * L * V * 2 / GB;
      const parts = [["Weights", weights], ["Gradients", grads], ["Optimizer", adam], ["Activations", act], ["Logits", logit]];
      const total = parts.reduce((a, p) => a + p[1], 0);
      const max = 60, left = 10, bw = w - 20;
      const X = (g) => left + bw * Math.min(1, g / max);
      let x = 0;
      parts.forEach((p, i) => {
        ctx.fillStyle = c.series[i];
        ctx.fillRect(X(x), 40, Math.max(0, X(x + p[1]) - X(x)), 40);
        x += p[1];
      });
      ctx.strokeStyle = c.strongLine; ctx.strokeRect(left, 40, bw, 40);
      [16, 40].forEach((g) => {
        ctx.strokeStyle = c.ink; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(X(g), 28); ctx.lineTo(X(g), 92); ctx.stroke(); ctx.setLineDash([]);
        text(ctx, `${g} GB`, X(g), 16, c.ink, `11px ${c.sans}`, g > 30 ? "right" : "left");
      });
      text(ctx, `Total ${total.toFixed(1)} GB`, left, 112, c.ink, `bold 14px ${c.sans}`);
      text(ctx, total > 16 ? "does not fit in 16 GB" : "fits in 16 GB before CUDA overhead", left, 132, total > 16 ? c.error : c.forest, `12px ${c.sans}`);
      list.replaceChildren(...parts.map((p, i) => {
        const row = D.el("span", "qwn-chip");
        const sw = D.el("i"); sw.style.background = c.series[i];
        row.append(sw, document.createTextNode(` ${p[0]} ${p[1].toFixed(2)} GB`));
        return row;
      }));
      out.textContent = "With all options off at batch 4 and length 2048, the total is 37.7 GB, which matches the post's napkin. With LoRA, checkpointing, and chunked loss on, the components sum to about 6 GB; the post reports about 7 to 8 GB with CUDA overhead.";
    };
    plot.redraw();
    el.append(D.el("p", "widget-caption", "Turn the options on one at a time and see which bucket each one removes. Activations and optimizer states are the largest buckets in the full-fine-tune bar."));
  });
})();

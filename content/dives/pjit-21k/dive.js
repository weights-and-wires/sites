// Widgets for the pjit-21k deep dive. Every number here is arithmetic from the
// post's settings or our own illustration; none of it is a result from the post.
(() => {
  "use strict";
  const { el, colors, canvas, slider, select, toggle, button, readout } = Dives;

  // ---- math helpers -------------------------------------------------------
  const erf = (x) => {
    // Abramowitz and Stegun 7.1.26, max error about 1.5e-7.
    const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
    return s * y;
  };
  const Phi = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
  const logit = (p) => Math.log(p / (1 - p));
  const massBetween = (mu, s, a, b) => Phi((logit(b) - mu) / s) - Phi((logit(a) - mu) / s);
  const density = (mu, s, t) => {
    const u = (logit(t) - mu) / s;
    return Math.exp(-0.5 * u * u) / (s * Math.sqrt(2 * Math.PI) * t * (1 - t));
  };
  const meanT = (mu, s) => {
    let tot = 0, n = 800;
    const lo = mu - 8 * s, h = (16 * s) / n;
    for (let i = 0; i <= n; i++) {
      const z = lo + i * h;
      tot += (1 / (1 + Math.exp(-z))) * Math.exp(-0.5 * ((z - mu) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
    }
    return tot * h;
  };
  const pct = (p) => (p < 0.001 ? `${(100 * p).toFixed(3)}%` : `${(100 * p).toFixed(2)}%`);
  const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
  const fmtSci = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M` : fmtInt(n));

  function rng(seed) {
    let a = seed >>> 0;
    const u = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
  }

  // ---- 1. token budget ----------------------------------------------------
  Dives.mount("token-budget", (root) => {
    root.append(el("p", "widget-title", "Tokens and attention pairs"));
    const controls = el("div", "widget-controls");
    root.append(controls);
    const sideSel = select(controls, { label: "Image side", options: [[256, "256 px"], [512, "512 px"], [1024, "1024 px"]], value: 512, onChange: () => plot.redraw() || show() });
    const patchSel = select(controls, { label: "Patch side", options: [[8, "8 px"], [16, "16 px"], [32, "32 px"]], value: 32, onChange: () => plot.redraw() || show() });
    const text = toggle(controls, { label: "Add 256 text tokens", checked: false, onChange: () => plot.redraw() || show() });
    const plot = canvas(root, { aspect: 0.5, label: "Bars compare attention pair counts for three patch sizes on a log scale" });
    const out = readout(root);
    const state = () => {
      const side = Number(sideSel.get()), p = Number(patchSel.get());
      const img = (side / p) ** 2, tx = text.get() ? 256 : 0;
      return { side, p, img, n: img + tx, tx };
    };
    function show() {
      const s = state();
      const values = s.p * s.p * 3;
      out.textContent =
        `${s.side / s.p} x ${s.side / s.p} = ${fmtInt(s.img)} image tokens` + (s.tx ? ` + ${s.tx} text = ${fmtInt(s.n)}` : "") + "\n" +
        `attention score pairs: ${fmtInt(s.n)}^2 = ${fmtInt(s.n * s.n)}\n` +
        `values per patch: ${s.p}*${s.p}*3 = ${fmtInt(values)}\n` +
        `into a 1,024-wide bottleneck: ${(values / 1024).toFixed(2)}:1; into 256: ${(values / 256).toFixed(2)}:1`;
    }
    plot.draw = (ctx, w, h) => {
      const c = colors(), s = state();
      const left = 96, right = w - 12, top = 14, rowH = (h - top - 30) / 3;
      const logMin = 3, logMax = 9.5;
      ctx.font = `12px ${c.sans}`;
      ctx.textBaseline = "middle";
      [8, 16, 32].forEach((p, i) => {
        const n = (s.side / p) ** 2 + s.tx, pairs = n * n;
        const frac = (Math.log10(pairs) - logMin) / (logMax - logMin);
        const y = top + i * rowH;
        ctx.fillStyle = p === s.p ? c.ink : c.muted;
        ctx.textAlign = "right";
        ctx.fillText(`${p} px patch`, left - 8, y + rowH / 2 - 8);
        ctx.fillStyle = c.muted;
        ctx.fillText(`${fmtInt(n)} tokens`, left - 8, y + rowH / 2 + 8);
        ctx.fillStyle = p === s.p ? c.accent : c.forestSoft;
        ctx.fillRect(left, y + 6, Math.max(2, frac * (right - left)), rowH - 12);
        ctx.textAlign = "left";
        ctx.fillStyle = c.ink;
        const label = fmtSci(pairs);
        const x = left + frac * (right - left);
        if (x + 8 + ctx.measureText(label).width < right) ctx.fillText(label, x + 6, y + rowH / 2);
        else { ctx.fillStyle = c.ink; ctx.textAlign = "right"; ctx.fillText(label, x - 6, y + rowH / 2); }
      });
      ctx.strokeStyle = c.line; ctx.fillStyle = c.muted; ctx.textAlign = "center";
      for (let e = 3; e <= 9; e += 2) {
        const x = left + ((e - logMin) / (logMax - logMin)) * (right - left);
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, h - 24); ctx.stroke();
        ctx.fillText(`10^${e}`, x, h - 12);
      }
    };
    root.append(el("p", "widget-caption", "Halving the patch count per side (16 px to 32 px) cuts the tokens 4x and the score matrix 16x. The 3,072 values in a 32 px patch are why the post widens the bottleneck from 256 to 1,024. The post's 512 px model is 32 px patches plus 256 text tokens: 512 tokens."));
    show(); plot.redraw();
  });

  // ---- 2. pyramid targets -------------------------------------------------
  Dives.mount("pyramid-targets", (root) => {
    root.append(el("p", "widget-title", "Pyramid targets under noise (1-D toy)"));
    const controls = el("div", "widget-controls");
    root.append(controls);
    const D = 32;
    const clean = Array.from({ length: D }, (_, i) => 1.2 * Math.sin((2 * Math.PI * i) / D) + 0.7 * Math.sin((4 * Math.PI * i) / D + 1) + 0.35 * Math.sin((2 * Math.PI * 12 * i) / D));
    const g = rng(7);
    const eps = Array.from({ length: D }, () => g());
    const tS = slider(controls, { label: "Noise level t", min: 0.05, max: 0.9, step: 0.01, value: 0.5, format: (v) => v.toFixed(2), onInput: () => { plot.redraw(); show(); } });
    const sS = slider(controls, { label: "Noise scale", min: 1, max: 4, step: 0.5, value: 2, format: (v) => v.toFixed(1), onInput: () => { plot.redraw(); show(); } });
    const plot = canvas(root, { height: (w) => Math.round(Math.max(250, w * 0.62)), label: "Three rows compare a pooled clean target with the pooled noisy input at three resolutions" });
    const out = readout(root);
    const rows = [{ f: 4, name: "block 10 target", note: "pool 4" }, { f: 2, name: "block 16 target", note: "pool 2" }, { f: 1, name: "block 22 target", note: "full" }];
    const pool = (a, f) => Array.from({ length: a.length / f }, (_, j) => { let s = 0; for (let k = 0; k < f; k++) s += a[j * f + k]; return s / f; });
    function calc() {
      const t = tS.get(), sg = sS.get(), k = 1 / (1 - t);
      const z = clean.map((x, i) => (1 - t) * x + t * sg * eps[i]);
      const est = z.map((v) => v * k); // naive estimate: rescale the noisy input
      const sd = (t * sg) / (1 - t);
      return rows.map((r) => {
        const tg = pool(clean, r.f), es = pool(est, r.f);
        const mse = tg.reduce((a, v, j) => a + (v - es[j]) ** 2, 0) / tg.length;
        return { ...r, tg, es, mse, expect: (sd * sd) / r.f };
      });
    }
    function show() {
      const rs = calc();
      out.textContent = rs.map((r) => `${r.name.padEnd(16)} ${String(D / r.f).padStart(2)} values  MSE ${r.mse.toFixed(2).padStart(6)}  (expected ${r.expect.toFixed(2)})`).join("\n") +
        `\nloss weight at this t: 1 / max(t, 0.1)^2 = ${(1 / Math.max(tS.get(), 0.1) ** 2).toFixed(2)}`;
    }
    plot.draw = (ctx, w, h) => {
      const c = colors(), rs = calc();
      const left = 8, right = w - 8, rowH = (h - 8) / 3, lim = 3.5;
      ctx.font = `12px ${c.sans}`; ctx.textBaseline = "top";
      rs.forEach((r, i) => {
        const y0 = 4 + i * rowH, top = y0 + 18, bot = y0 + rowH - 6;
        const Y = (v) => top + ((lim - Math.max(-lim, Math.min(lim, v))) / (2 * lim)) * (bot - top);
        const cw = (right - left) / r.tg.length;
        ctx.fillStyle = c.ink; ctx.textAlign = "left";
        ctx.fillText(`${r.name}: ${r.tg.length} values (${r.note})`, left, y0);
        ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(left, Y(0)); ctx.lineTo(right, Y(0)); ctx.stroke();
        ctx.fillStyle = c.accent;
        r.es.forEach((v, j) => { ctx.beginPath(); ctx.arc(left + (j + 0.5) * cw, Y(v), 2.6, 0, 7); ctx.fill(); });
        ctx.strokeStyle = c.forest; ctx.lineWidth = 2; ctx.beginPath();
        r.tg.forEach((v, j) => { const x1 = left + j * cw, x2 = x1 + cw; if (j === 0) ctx.moveTo(x1, Y(v)); else ctx.lineTo(x1, Y(v)); ctx.lineTo(x2, Y(v)); });
        ctx.stroke(); ctx.lineWidth = 1;
      });
    };
    root.append(el("p", "widget-caption", "Green is the clean target at each resolution; orange dots are the noisy input rescaled by 1/(1 - t), pooled the same way. Pooling by f cuts the noise power by f, so the coarse targets sit closer to the dots. This is our illustration of why a low-resolution target is an easier early job, not a measurement from the post; points beyond the plot range are clamped to its edge."));
    show(); plot.redraw();
  });

  // ---- 3. timestep schedule -----------------------------------------------
  Dives.mount("schedule", (root) => {
    root.append(el("p", "widget-title", "Timestep schedule explorer"));
    const controls = el("div", "widget-controls");
    root.append(controls);
    const P1 = [0.8, 0.8], P2 = [-0.2, 1.0], SWITCH = 80, TOTAL = 660;
    let lock = false;
    const sync = () => { plot.redraw(); show(); };
    const prog = slider(controls, { label: "Samples seen", min: 0, max: TOTAL, step: 5, value: 0, format: (v) => `${v}M`, onInput: (v) => { const [m, s] = v < SWITCH ? P1 : P2; lock = true; mu.set(m); sd.set(s); lock = false; sync(); } });
    const mu = slider(controls, { label: "Mean (mu)", min: -1, max: 1.5, step: 0.05, value: P1[0], format: (v) => v.toFixed(2), onInput: () => { if (!lock) sync(); } });
    const sd = slider(controls, { label: "Spread (s)", min: 0.4, max: 1.5, step: 0.05, value: P1[1], format: (v) => v.toFixed(2), onInput: () => { if (!lock) sync(); } });
    const lo = slider(controls, { label: "Band from", min: 0.02, max: 0.5, step: 0.005, value: 0.075, format: (v) => v.toFixed(3), onInput: () => { if (lo.get() >= hi.get()) hi.set(Math.min(0.9, lo.get() + 0.01)); else sync(); } });
    const hi = slider(controls, { label: "Band to", min: 0.03, max: 0.9, step: 0.005, value: 0.125, format: (v) => v.toFixed(3), onInput: () => { if (hi.get() <= lo.get()) lo.set(Math.max(0.02, hi.get() - 0.01)); else sync(); } });
    const btns = el("div", "widget-controls");
    root.append(btns);
    button(btns, "Phase 1: LogitNormal(0.8, 0.8)", () => { lock = true; mu.set(P1[0]); sd.set(P1[1]); lock = false; sync(); });
    button(btns, "Phase 2: LogitNormal(-0.2, 1.0)", () => { lock = true; mu.set(P2[0]); sd.set(P2[1]); lock = false; sync(); });
    const plot = canvas(root, { aspect: 0.5, label: "Density of the timestep t for phase 1, phase 2, and the current setting, with a shaded band" });
    const out = readout(root);
    function show() {
      const m = mu.get(), s = sd.get(), a = lo.get(), b = hi.get();
      const cur = massBetween(m, s, a, b), p1 = massBetween(...P1, a, b), p2 = massBetween(...P2, a, b);
      out.textContent =
        `current: share in [${a.toFixed(3)}, ${b.toFixed(3)}] = ${pct(cur)}, mean t = ${meanT(m, s).toFixed(2)}\n` +
        `P(t < 0.1) = ${pct(massBetween(m, s, 1e-9, 0.1))}, P(t > 0.9) = ${pct(massBetween(m, s, 0.9, 1 - 1e-9))}\n` +
        `phase 1 in band: ${pct(p1)}; phase 2 in band: ${pct(p2)}; ratio ${(p2 / p1).toFixed(0)}x`;
    }
    plot.draw = (ctx, w, h) => {
      const c = colors();
      const L = 30, R = w - 10, T = 12, B = h - 26;
      const N = 240, ts = Array.from({ length: N }, (_, i) => (i + 0.5) / N);
      const curves = [[P1, c.accent], [P2, c.muted], [[mu.get(), sd.get()], c.forest]].map(([p, col]) => ({ col, ys: ts.map((t) => density(p[0], p[1], t)) }));
      const yMax = Math.min(8, Math.max(2.5, ...curves.flatMap((k) => k.ys)) * 1.05);
      const X = (t) => L + t * (R - L), Y = (v) => B - (Math.min(v, yMax) / yMax) * (B - T);
      ctx.globalAlpha = 0.3; ctx.fillStyle = c.accent;
      ctx.fillRect(X(lo.get()), T, Math.max(1, X(hi.get()) - X(lo.get())), B - T);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = c.line; ctx.beginPath(); ctx.moveTo(L, B); ctx.lineTo(R, B); ctx.stroke();
      ctx.font = `12px ${c.sans}`; ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
      [0, 0.25, 0.5, 0.75, 1].forEach((t) => ctx.fillText(String(t), X(t), B + 3));
      ctx.textAlign = "left"; ctx.fillText("clean", L, B + 14);
      ctx.textAlign = "right"; ctx.fillText("pure noise", R, B + 14);
      curves.forEach((k, i) => {
        ctx.strokeStyle = k.col; ctx.lineWidth = i === 2 ? 2.5 : 1.5; ctx.setLineDash(i === 2 ? [] : [5, 4]);
        ctx.beginPath(); k.ys.forEach((v, j) => (j ? ctx.lineTo(X(ts[j]), Y(v)) : ctx.moveTo(X(ts[j]), Y(v)))); ctx.stroke();
      });
      ctx.setLineDash([]); ctx.lineWidth = 1;
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      [["phase 1", c.accent], ["phase 2", c.muted], ["current", c.forest]].forEach(([n, col], i) => { ctx.fillStyle = col; ctx.fillText(n, L + 6 + i * 64, T + 2); });
    };
    root.append(el("p", "widget-caption", "Move the samples slider past 80M and the density slides toward clean (small t). The shaded band, centered on t = 0.1, goes from about 0.03% to 3% of the timesteps. The post labels this band 0.08 to 0.13; the default 0.075 to 0.125 reproduces its 3.0% and 107x."));
    show(); plot.redraw();
  });

  // ---- 4. toy result (fed by the cell) ------------------------------------
  Dives.mount("toy-result", (root) => {
    root.append(el("p", "widget-title", "Toy pyramid readouts: our own result"));
    const plot = canvas(root, { aspect: 0.55, label: "Grouped bars of final-head error with three readouts relative to the final head alone" });
    const out = readout(root);
    let data = null;
    out.textContent = "Run the second Python cell above to fill this in.";
    plot.draw = (ctx, w, h) => {
      const c = colors();
      ctx.font = `12px ${c.sans}`; ctx.textBaseline = "middle";
      if (!data) { ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.fillText("no data yet", w / 2, h / 2); return; }
      const L = 40, R = w - 10, T = 14, B = h - 30, lo = 0.9, hi = 1.1;
      const Y = (v) => B - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (B - T);
      ctx.strokeStyle = c.line; ctx.fillStyle = c.muted; ctx.textAlign = "right";
      [0.9, 1.0, 1.1].forEach((v) => { ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), L - 6, Y(v)); });
      ctx.strokeStyle = c.ink; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(L, Y(1)); ctx.lineTo(R, Y(1)); ctx.stroke(); ctx.setLineDash([]);
      const gw = (R - L) / data.t.length;
      data.t.forEach((t, i) => {
        const base = data.one.mean[i], r = data.three.mean[i] / base, e = data.three.sd[i] / base;
        const e1 = data.one.sd[i] / base;
        const x = L + gw * (i + 0.5);
        ctx.fillStyle = c.forest; ctx.fillRect(x - 26, Y(1 + e1), 14, Y(1 - e1) - Y(1 + e1)); // reference spread
        ctx.fillStyle = c.accent; ctx.fillRect(x + 12, Y(Math.max(r, 1)), 14, Math.max(2, Math.abs(Y(r) - Y(1))));
        ctx.strokeStyle = c.ink; ctx.beginPath(); ctx.moveTo(x + 19, Y(r - e)); ctx.lineTo(x + 19, Y(r + e)); ctx.stroke();
        ctx.fillStyle = c.ink; ctx.textAlign = "center"; ctx.fillText(`t = ${t}`, x, B + 14);
      });
    };
    document.addEventListener("cell:result", (e) => {
      if (e.detail.name !== "pjit-21k/toy_pyramid" || !e.detail.result) return;
      data = e.detail.result;
      const r = data.three.mean.map((v, i) => (v / data.one.mean[i]).toFixed(2)).join(", ");
      out.textContent = `final-head MSE with readouts / without, at t = 0.3, 0.6, 0.9: ${r}\n` +
        `gradient norm on block 0 at init, with / without readouts: ${(data.grad.three / data.grad.one).toFixed(2)}x`;
      plot.redraw();
    });
    root.append(el("p", "widget-caption", "Green marks the spread of the final-head-only runs around 1.0; orange is the three-readout mean with its own seed spread. Each bar sits inside the other's spread, so the toy shows no benefit. The gradient reaching block 0 does grow, which is the mechanism the post points to, but the toy does not turn that into a lower loss."));
  });

  // ---- 5. budget ----------------------------------------------------------
  Dives.mount("budget", (root) => {
    root.append(el("p", "widget-title", "Training bill calculator"));
    const controls = el("div", "widget-controls");
    root.append(controls);
    const POST_SAMPLES = 660.24, POST_HOURS = 8139, POST_FT = 276;
    const upd = () => { plot.redraw(); show(); };
    const smp = slider(controls, { label: "Pretraining samples", min: 50, max: 1000, step: 0.01, value: POST_SAMPLES, format: (v) => `${v.toFixed(2)}M`, onInput: upd });
    const bs = select(controls, { label: "Global batch", options: [[1024, "1,024"], [2048, "2,048"], [4096, "4,096"]], value: 2048, onChange: upd });
    const price = slider(controls, { label: "$ per H100-hour", min: 1, max: 4, step: 0.1, value: 2.5, format: (v) => `$${v.toFixed(2)}`, onInput: upd });
    const pt = toggle(controls, { label: "Add 276 h of post-training", checked: true, onChange: upd });
    button(controls, "Reset to the post's run", () => { smp.set(POST_SAMPLES); bs.set(2048); price.set(2.5); pt.set(true); });
    const plot = canvas(root, { height: 70, label: "A bar split into pretraining and post-training cost" });
    const out = readout(root);
    const calc = () => {
      const hoursPerM = POST_HOURS / POST_SAMPLES;
      const hours = smp.get() * hoursPerM, post = pt.get() ? POST_FT : 0;
      return { hours, post, steps: (smp.get() * 1e6) / Number(bs.get()), pre$: hours * price.get(), post$: post * price.get() };
    };
    function show() {
      const k = calc();
      out.textContent = `steps: ${fmtInt(k.steps)}   pretraining: ${fmtInt(k.hours)} h = $${fmtInt(k.pre$)}\n` +
        `post-training: ${k.post} h = $${fmtInt(k.post$)}   total: $${fmtInt(k.pre$ + k.post$)}\n` +
        `on 64 GPUs with no downtime: ${(k.hours / 64 / 24).toFixed(1)} days (the post's hours include restarts)`;
    }
    plot.draw = (ctx, w, h) => {
      const c = colors(), k = calc(), tot = k.pre$ + k.post$;
      const max = 60000, L = 8, R = w - 8, x1 = L + (k.pre$ / max) * (R - L), x2 = x1 + (k.post$ / max) * (R - L);
      ctx.fillStyle = c.surfaceMuted; ctx.fillRect(L, 14, R - L, 28);
      ctx.fillStyle = c.forest; ctx.fillRect(L, 14, Math.max(0, Math.min(x1, R) - L), 28);
      ctx.fillStyle = c.accent; ctx.fillRect(Math.min(x1, R), 14, Math.max(0, Math.min(x2, R) - Math.min(x1, R)), 28);
      ctx.font = `12px ${c.sans}`; ctx.fillStyle = c.muted; ctx.textBaseline = "top"; ctx.textAlign = "left";
      ctx.fillText("$0", L, 48); ctx.textAlign = "right"; ctx.fillText(`bar scale: $${fmtInt(max)}`, R, 48);
    };
    root.append(el("p", "widget-caption", "The default reproduces the post's $21,038. Our assumption: hours scale linearly with samples at the post's 12.3 hours per million, which holds only if restarts and throughput stay the same. The post doesn't split the hours between the 512 px and 1024 px stages, and 1024 px images probably cost more each, so treat sample counts as approximate."));
    show(); plot.redraw();
  });
})();

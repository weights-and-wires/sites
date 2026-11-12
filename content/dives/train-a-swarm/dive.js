// Widgets for the train-a-swarm deep dive. The toy model is ours, not the post's:
// a team of n agents does work at an Amdahl's-law speed and pays a setup tax first.
(() => {
  "use strict";

  const speed = (n, p) => 1 / ((1 - p) + p / n);
  const progress = (n, t, p, setup) => speed(n, p) * Math.max(0, t - setup * (n - 1));
  const fmt = (v, d = 1) => v.toFixed(d);

  function axes(ctx, c, box, xmax, ymax, xlabel, ylabel) {
    const { l, t, r, b } = box;
    ctx.strokeStyle = c.line;
    ctx.fillStyle = c.muted;
    ctx.font = `11px ${c.sans}`;
    ctx.lineWidth = 1;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const xs = niceTicks(xmax, 5);
    for (const v of xs) {
      const x = l + (v / xmax) * (r - l);
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, b); ctx.stroke();
      ctx.fillText(String(v), x, b + 4);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of niceTicks(ymax, 4)) {
      const y = b - (v / ymax) * (b - t);
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(r, y); ctx.stroke();
      ctx.fillText(String(v), l - 5, y);
    }
    ctx.strokeStyle = c.strongLine;
    ctx.beginPath(); ctx.moveTo(l, t); ctx.lineTo(l, b); ctx.lineTo(r, b); ctx.stroke();
    ctx.fillStyle = c.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(xlabel, r, b + 17);
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(ylabel, l, t - 3);
  }

  function niceTicks(max, count) {
    const raw = max / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
    return out;
  }

  // Widget 1: how the cap decides the best team size.
  Dives.mount("cap-explorer", (el) => {
    el.append(Dives.el("p", "widget-title", "Progress by the deadline, by team size"));
    const controls = Dives.el("div", "widget-controls");
    el.append(controls);
    const plot = Dives.canvas(el, { aspect: 0.62, label: "Progress against time for teams of 1, 2, 4, and 8 agents, with a vertical line at the deadline" });
    const out = Dives.readout(el);
    const redraw = () => plot.redraw();
    const p = Dives.slider(controls, { label: "Parallel fraction p", min: 0.5, max: 0.99, step: 0.01, value: 0.9, format: (v) => v.toFixed(2), onInput: redraw });
    const setup = Dives.slider(controls, { label: "Setup per extra agent", min: 0, max: 2, step: 0.05, value: 0.6, format: (v) => `${v.toFixed(2)} min`, onInput: redraw });
    const cap = Dives.slider(controls, { label: "Deadline", min: 1, max: 40, step: 0.5, value: 5, format: (v) => `${v} min`, onInput: redraw });
    const shown = [1, 2, 4, 8];

    plot.draw = (ctx, w, h) => {
      const c = Dives.colors();
      const P = p.get(), S = setup.get(), T = cap.get();
      const xmax = Math.max(8, Math.ceil(T * 1.6));
      let ymax = 0;
      for (let n = 1; n <= 8; n++) ymax = Math.max(ymax, progress(n, xmax, P, S));
      ymax = Math.max(ymax, xmax);
      const box = { l: 38, t: 16, r: w - 46, b: h - 32 };
      axes(ctx, c, box, xmax, ymax, "minutes", "work done");
      const X = (t) => box.l + (t / xmax) * (box.r - box.l);
      const Y = (v) => box.b - (v / ymax) * (box.b - box.t);
      let best = 1, bestV = -1;
      for (let n = 1; n <= 8; n++) {
        const v = progress(n, T, P, S);
        if (v > bestV + 1e-9) { best = n; bestV = v; }
      }
      const drawn = shown.includes(best) ? shown : [...shown, best].sort((a, b) => a - b);
      drawn.forEach((n) => {
        const col = c.series[(n === best ? 0 : shown.indexOf(n) + 1) % 6];
        ctx.strokeStyle = n === best ? c.accent : col;
        ctx.lineWidth = n === best ? 3 : 2;
        ctx.setLineDash(shown.includes(n) ? [] : [5, 3]);
        ctx.beginPath();
        for (let i = 0; i <= 120; i++) {
          const t = (i / 120) * xmax;
          const x = X(t), y = Y(progress(n, t, P, S));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = n === best ? c.accentInk : c.ink;
        ctx.font = `600 11px ${c.sans}`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(`n=${n}`, X(xmax) + 4, Y(progress(n, xmax, P, S)));
      });
      ctx.strokeStyle = c.error;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(X(T), box.t); ctx.lineTo(X(T), box.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = c.error;
      ctx.font = `600 11px ${c.sans}`;
      ctx.textAlign = X(T) > box.r - 50 ? "right" : "left";
      ctx.textBaseline = "top";
      ctx.fillText("deadline", X(T) + (X(T) > box.r - 50 ? -4 : 4), box.t + 2);
      ctx.fillStyle = c.accent;
      ctx.beginPath(); ctx.arc(X(T), Y(bestV), 4.5, 0, Math.PI * 2); ctx.fill();

      const s8 = speed(8, P);
      const cross = s8 > 1 ? (s8 * S * 7) / (s8 - 1) : Infinity;
      const one = progress(1, T, P, S);
      out.textContent = `At a ${T}-minute deadline the best team has ${best} agent${best > 1 ? "s" : ""} and gets ${fmt(bestV)} units, against ${fmt(one)} for one agent. Eight agents overtake one agent after ${fmt(cross, 2)} minutes.`;
    };
    el.append(Dives.el("p", "widget-caption", "Move the deadline right: the best team size grows, because a longer deadline gives a big team time to repay its setup tax. The accent line marks the best team size at the deadline."));
    plot.redraw();
  });

  // Widget 2: one flat advantage per rollout, under a time cap or a latency penalty.
  Dives.mount("flat-advantage", (el) => {
    el.append(Dives.el("p", "widget-title", "One advantage for every token in a rollout"));
    const controls = Dives.el("div", "widget-controls");
    el.append(controls);
    const rows = Dives.el("div", "widget-controls");
    el.append(rows);
    const plot = Dives.canvas(el, { height: 4 * 50 + 12, label: "Four rollouts drawn as strips of tokens split by agent, each strip one color set by its advantage" });
    const out = Dives.readout(el);
    // Token counts per agent (in thousands) are illustrative and ours.
    const tokens = [[30, 12, 8], [25, 20], [10, 10, 9, 8, 6], [40, 15, 15, 12]];
    const tests = [3, 5, 5, 9];
    const times = [90, 70, 40, 55];
    const sTests = [], sTimes = [];
    const redraw = () => { update(); plot.redraw(); };
    const mode = Dives.select(controls, { label: "Latency pressure", options: [["cap", "Time cap (anytime return)"], ["pen", "Explicit latency penalty"]], value: "cap", onChange: () => { syncDisabled(); redraw(); } });
    const beta = Dives.slider(controls, { label: "Penalty strength β", min: 0, max: 1, step: 0.05, value: 0.5, format: (v) => v.toFixed(2), onInput: redraw });
    for (let i = 0; i < 4; i++) {
      sTests.push(Dives.slider(rows, { label: `Rollout ${i + 1}: tests passed`, min: 0, max: 10, step: 1, value: tests[i], onInput: redraw }));
      sTimes.push(Dives.slider(rows, { label: `Rollout ${i + 1}: finish time`, min: 10, max: 100, step: 5, value: times[i], format: (v) => `${v} min`, onInput: redraw }));
    }
    function syncDisabled() {
      const cap = mode.get() === "cap";
      beta.input.disabled = cap;
      sTimes.forEach((s) => { s.input.disabled = cap; });
    }
    let state = null;
    function update() {
      const cap = mode.get() === "cap";
      const score = sTests.map((s) => s.get());
      const T = sTimes.map((s) => s.get());
      const pbar = score.reduce((a, b) => a + b, 0) / score.length / 10;
      const reward = score.map((s, i) => (cap ? s : s / 10 - beta.get() * pbar * T[i] / 100));
      const mean = reward.reduce((a, b) => a + b, 0) / reward.length;
      const std = Math.sqrt(reward.reduce((a, b) => a + (b - mean) ** 2, 0) / reward.length);
      const adv = reward.map((r) => (std < 1e-9 ? 0 : (r - mean) / std));
      state = { cap, reward, mean, std, adv };
      out.textContent = `Rewards ${reward.map((r) => fmt(r, cap ? 0 : 3)).join(", ")}. Mean ${fmt(mean, 2)}, standard deviation ${fmt(std, 2)}. Advantages ${adv.map((a) => (a >= 0 ? "+" : "") + fmt(a, 2)).join(", ")}.`;
    }
    plot.draw = (ctx, w) => {
      if (!state) return;
      const c = Dives.colors();
      ctx.font = `12px ${c.sans}`;
      state.adv.forEach((a, i) => {
        const y = 6 + i * 50;
        ctx.fillStyle = c.ink;
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(`Rollout ${i + 1}: advantage ${(a >= 0 ? "+" : "") + fmt(a, 2)} on every token`, 0, y);
        const gap = 6;
        let x = 0;
        tokens[i].forEach((n, k) => {
          const width = (n / 85) * (w - gap * 3);
          ctx.fillStyle = a >= 0 ? c.forest : c.error;
          ctx.globalAlpha = 0.12 + 0.88 * Math.min(1, Math.abs(a) / 2);
          ctx.fillRect(x, y + 18, Math.max(2, width), 18);
          ctx.globalAlpha = 1;
          ctx.strokeStyle = c.strongLine;
          ctx.strokeRect(x + 0.5, y + 18.5, Math.max(2, width) - 1, 17);
          x += width + gap;
        });
      });
    };
    el.append(Dives.el("p", "widget-caption", "Under a time cap, rollouts 2 and 3 tie even though rollout 3 finishes sooner. Switch to the penalty to separate them. Every segment, whichever agent wrote it, gets the same value."));
    syncDisabled();
    update();
    plot.redraw();
  });

  // Widget 3: a policy learns a team size under a deadline schedule.
  function mulberry32(a) {
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const P0 = 0.9, S0 = 0.6, STEPS = 300;
  const schedules = {
    short: { label: "Fixed deadline 2", at: () => 2 },
    mid: { label: "Fixed deadline 5", at: () => 5 },
    long: { label: "Fixed deadline 20", at: () => 20 },
    ramp: { label: "Ramp 2 to 20", at: (s) => 2 + (18 * s) / (STEPS - 1) },
  };
  const bestN = (T) => {
    let b = 1, bv = -1;
    for (let n = 1; n <= 8; n++) { const v = progress(n, T, P0, S0); if (v > bv + 1e-9) { b = n; bv = v; } }
    return b;
  };
  function softmax(th) {
    const m = Math.max(...th);
    const e = th.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  Dives.mount("team-trainer", (el) => {
    el.append(Dives.el("p", "widget-title", "Train a team-size policy under a deadline"));
    const controls = Dives.el("div", "widget-controls");
    el.append(controls);
    const buttons = Dives.el("div", "widget-controls");
    el.append(buttons);
    const plot = Dives.canvas(el, { aspect: 0.95, label: "Bars show the probability of each team size from 1 to 8; a line below shows the mean team size over training steps" });
    const out = Dives.readout(el);
    let theta, step, rng, hist, running = false, raf = 0;
    const sched = Dives.select(controls, { label: "Deadline schedule", options: Object.entries(schedules).map(([k, v]) => [k, v.label]), value: "ramp", onChange: () => reset() });
    const seed = Dives.slider(controls, { label: "Seed", min: 0, max: 20, step: 1, value: 0, onInput: () => reset() });

    function reset() {
      running = false; cancelAnimationFrame(raf);
      theta = new Array(8).fill(0); step = 0; hist = []; rng = mulberry32(seed.get() + 1);
      runBtn.textContent = "Run";
      record(); plot.redraw(); say();
    }
    function record() {
      const pr = softmax(theta);
      hist.push(pr.reduce((a, v, i) => a + v * (i + 1), 0));
    }
    function one() {
      const T = schedules[sched.get()].at(step);
      const pr = softmax(theta);
      const ns = [];
      for (let g = 0; g < 16; g++) {
        let u = rng(), k = 0;
        while (k < 7 && u > pr[k]) { u -= pr[k]; k++; }
        ns.push(k + 1);
      }
      const r = ns.map((n) => progress(n, T, P0, S0));
      const mean = r.reduce((a, b) => a + b, 0) / r.length;
      const std = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / r.length);
      const grad = new Array(8).fill(0);
      ns.forEach((n, j) => {
        const a = (r[j] - mean) / (std + 1e-8);
        for (let k = 0; k < 8; k++) grad[k] += a * ((k === n - 1 ? 1 : 0) - pr[k]);
      });
      for (let k = 0; k < 8; k++) theta[k] += (0.5 * grad[k]) / 16;
      step++;
      record();
    }
    function advance(count) {
      for (let i = 0; i < count && step < STEPS; i++) one();
      plot.redraw(); say();
    }
    function say() {
      const T = schedules[sched.get()].at(Math.min(step, STEPS - 1));
      const pr = softmax(theta);
      const mode = pr.indexOf(Math.max(...pr)) + 1;
      out.textContent = `Step ${step} of ${STEPS}. Deadline ${fmt(T)} min, best team size ${bestN(T)}. The policy's most likely team size is ${mode}; mean ${fmt(hist[hist.length - 1], 2)}.`;
    }
    function loop() {
      if (!running) return;
      advance(3);
      if (step >= STEPS) { running = false; runBtn.textContent = "Run"; return; }
      raf = requestAnimationFrame(loop);
    }
    const runBtn = Dives.button(buttons, "Run", () => {
      if (running) { running = false; runBtn.textContent = "Run"; return; }
      if (step >= STEPS) reset();
      if (Dives.reducedMotion) { advance(STEPS); return; }
      running = true; runBtn.textContent = "Pause"; loop();
    });
    Dives.button(buttons, "Step 10", () => { if (!running && step < STEPS) advance(10); });
    Dives.button(buttons, "Reset", reset);

    plot.draw = (ctx, w, h) => {
      const c = Dives.colors();
      const T = schedules[sched.get()].at(Math.min(step, STEPS - 1));
      const pr = softmax(theta);
      const top = { l: 30, t: 20, r: w - 8, b: h * 0.5 };
      const bw = (top.r - top.l) / 8;
      ctx.strokeStyle = c.strongLine;
      ctx.beginPath(); ctx.moveTo(top.l, top.b); ctx.lineTo(top.r, top.b); ctx.stroke();
      ctx.font = `11px ${c.sans}`;
      ctx.fillStyle = c.muted;
      ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText("probability of each team size", top.l, top.t - 4);
      const best = bestN(T);
      pr.forEach((v, i) => {
        const x = top.l + i * bw + bw * 0.15;
        const bh = v * (top.b - top.t);
        ctx.fillStyle = i + 1 === best ? c.accent : c.forest;
        ctx.globalAlpha = i + 1 === best ? 1 : 0.7;
        ctx.fillRect(x, top.b - bh, bw * 0.7, bh);
        ctx.globalAlpha = 1;
        ctx.fillStyle = c.ink;
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(String(i + 1), x + bw * 0.35, top.b + 3);
      });
      ctx.fillStyle = c.muted;
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText("1.0", top.l - 4, top.t);
      ctx.fillText("0", top.l - 4, top.b);
      // Lower panel: mean team size over steps, and the best size for the deadline.
      const lo = { l: 30, t: h * 0.5 + 40, r: w - 8, b: h - 22 };
      ctx.strokeStyle = c.line;
      for (const n of [1, 4, 8]) {
        const y = lo.b - ((n - 1) / 7) * (lo.b - lo.t);
        ctx.beginPath(); ctx.moveTo(lo.l, y); ctx.lineTo(lo.r, y); ctx.stroke();
        ctx.fillStyle = c.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(String(n), lo.l - 4, y);
      }
      ctx.strokeStyle = c.strongLine;
      ctx.beginPath(); ctx.moveTo(lo.l, lo.t); ctx.lineTo(lo.l, lo.b); ctx.lineTo(lo.r, lo.b); ctx.stroke();
      ctx.fillStyle = c.muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
      ctx.fillText("mean (solid) and best (dashed) team size", lo.l, lo.t - 6);
      ctx.textAlign = "right"; ctx.textBaseline = "top";
      ctx.fillText("step", lo.r, lo.b + 5);
      const X = (s) => lo.l + (s / STEPS) * (lo.r - lo.l);
      const Y = (n) => lo.b - ((n - 1) / 7) * (lo.b - lo.t);
      ctx.strokeStyle = c.accent; ctx.lineWidth = 2; ctx.setLineDash([5, 3]);
      ctx.beginPath();
      for (let s = 0; s < STEPS; s++) { const x = X(s), y = Y(bestN(schedules[sched.get()].at(s))); if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = c.forest; ctx.lineWidth = 2.5;
      ctx.beginPath();
      hist.forEach((m, s) => { const x = X(s), y = Y(m); if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      ctx.stroke(); ctx.lineWidth = 1;
    };
    el.append(Dives.el("p", "widget-caption", "Run each schedule. A short deadline teaches a small team and a long one teaches the biggest team. Try the ramp with several seeds: some runs lock in a small team and stop learning. The sampling here is JavaScript's, so its numbers differ from the Python cell's."));
    reset();
  });

  // Widget 4: draw the result of the Python cell.
  Dives.mount("cell-result", (el) => {
    el.append(Dives.el("p", "widget-title", "Learned team-size probabilities from the cell"));
    const plot = Dives.canvas(el, { aspect: 0.6, label: "A grid of probabilities: one row per deadline, one column per team size from 1 to 8" });
    let data = null;
    document.addEventListener("cell:result", (e) => {
      if (e.detail.name !== "train-a-swarm/reconstruct") return;
      data = e.detail.result;
      plot.redraw();
    });
    plot.draw = (ctx, w, h) => {
      const c = Dives.colors();
      ctx.font = `12px ${c.sans}`;
      if (!data) {
        ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("Run the reconstruction cell above to fill this grid.", w / 2, h / 2);
        return;
      }
      const rows = data.tmax.length;
      const box = { l: 64, t: 22, r: w - 8, b: h - 8 };
      const cw = (box.r - box.l) / 8, rh = (box.b - box.t) / rows;
      ctx.fillStyle = c.muted; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      for (let n = 1; n <= 8; n++) ctx.fillText(String(n), box.l + (n - 0.5) * cw, box.t - 4);
      ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText("agents", box.l - 6, box.t - 12);
      data.tmax.forEach((T, i) => {
        ctx.fillStyle = c.ink; ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillText(`cap ${T}`, box.l - 6, box.t + (i + 0.5) * rh);
        data.probs[i].forEach((v, j) => {
          ctx.fillStyle = c.forest; ctx.globalAlpha = 0.08 + 0.92 * v;
          ctx.fillRect(box.l + j * cw + 1, box.t + i * rh + 1, cw - 2, rh - 2);
          ctx.globalAlpha = 1;
          if (v >= 0.05) {
            ctx.fillStyle = v > 0.5 ? c.paper : c.ink; ctx.textAlign = "center";
            ctx.fillText(v.toFixed(2), box.l + (j + 0.5) * cw, box.t + (i + 0.5) * rh);
          }
        });
      });
    };
    el.append(Dives.el("p", "widget-caption", "Each row is one training run with a fixed deadline. The filled cell moves right as the deadline grows."));
    plot.redraw();
  });
})();

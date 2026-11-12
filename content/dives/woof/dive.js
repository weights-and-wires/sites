// Widgets for the Underdog Woof deep dive. Everything here is our own toy
// code; none of it runs Woof.
(() => {
  "use strict";

  function mulberry(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rnd) {
    const u = 1 - rnd(), v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ---------------------------------------------------------------- WER aligner
  function align(ref, hyp) {
    const n = ref.length, m = hyp.length;
    const d = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = 0; i <= n; i++) d[i][0] = i;
    for (let j = 0; j <= m; j++) d[0][j] = j;
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        d[i][j] = Math.min(d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1), d[i - 1][j] + 1, d[i][j - 1] + 1);
      }
    }
    const ops = [];
    let i = n, j = m;
    while (i || j) {
      if (i && j && d[i][j] === d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)) {
        ops.push(ref[i - 1] === hyp[j - 1] ? ["ok", ref[i - 1], hyp[j - 1]] : ["sub", ref[i - 1], hyp[j - 1]]);
        i--; j--;
      } else if (i && d[i][j] === d[i - 1][j] + 1) {
        ops.push(["del", ref[i - 1], ""]);
        i--;
      } else {
        ops.push(["ins", "", hyp[j - 1]]);
        j--;
      }
    }
    return ops.reverse();
  }

  Dives.mount("wer-aligner", (el) => {
    el.append(Dives.el("p", "widget-title", "Score a transcript"));
    const presets = [
      ["call zero two zero seven nine four six zero zero one eight", "call zero two zero seven nine four six zero zero one eight"],
      ["the fourteenth of march", "the fourteenth of march"],
      ["x k four four seven one", "x k four four seven one"],
    ];
    const controls = Dives.el("div", "widget-controls");
    el.append(controls);
    const refInput = Dives.el("input", "woof-input");
    refInput.type = "text";
    refInput.setAttribute("aria-label", "Reference script (what the voice was asked to say)");
    const hypInput = Dives.el("input", "woof-input");
    hypInput.type = "text";
    hypInput.setAttribute("aria-label", "Hypothesis (what the recognizer heard)");
    Dives.select(controls, {
      label: "Preset",
      options: [["0", "Phone number"], ["1", "Date"], ["2", "Code"]],
      value: "0",
      onChange: (v) => { refInput.value = presets[+v][0]; hypInput.value = presets[+v][1]; render(); },
    });
    el.append(Dives.el("p", "meta", "Reference script (edit freely)"), refInput, Dives.el("p", "meta", "What the recognizer heard (edit this one)"), hypInput);
    const grid = Dives.el("div", "woof-align");
    grid.setAttribute("aria-live", "polite");
    el.append(grid);
    const out = Dives.readout(el);

    function render() {
      const ref = refInput.value.toLowerCase().split(/\s+/).filter(Boolean);
      const hyp = hypInput.value.toLowerCase().split(/\s+/).filter(Boolean);
      const ops = align(ref, hyp);
      grid.replaceChildren();
      let s = 0, dl = 0, ins = 0;
      for (const [k, r, h] of ops) {
        if (k === "sub") s++; else if (k === "del") dl++; else if (k === "ins") ins++;
        const t = Dives.el("span", `woof-tok ${k}`);
        const top = Dives.el("span", "", r || "—");
        const bot = Dives.el("small", "", k === "ok" ? "ok" : k === "sub" ? `heard "${h}"` : k === "del" ? "deleted" : `inserted "${h}"`);
        t.append(top, bot);
        grid.append(t);
      }
      const e = s + dl + ins;
      out.textContent = ref.length
        ? `S ${s} + D ${dl} + I ${ins} = ${e} errors over ${ref.length} reference words: WER ${(100 * e / ref.length).toFixed(1)}%`
        : "Type a reference script.";
    }
    refInput.addEventListener("input", render);
    hypInput.addEventListener("input", render);
    refInput.value = "call zero two zero seven nine four six zero zero one eight";
    hypInput.value = "call zero two oh seven nine four six zero zero one eight";
    render();
    el.append(Dives.el("p", "widget-caption", "Change the hypothesis and watch the three error types change the score. One wrong word in a 12-word number costs 8.3 points on this sentence, which is why numbers-heavy test sets are unforgiving."));
  });

  // ---------------------------------------------------------------- int8 size and error
  Dives.mount("int8-size", (el) => {
    el.append(Dives.el("p", "widget-title", "Size and rounding error"));
    const controls = Dives.el("div", "widget-controls");
    el.append(controls);
    const params = Dives.slider(controls, { label: "Parameters (millions)", min: 1, max: 100, step: 0.1, value: 30.7, format: (v) => `${v.toFixed(1)} M`, onInput: () => update() });
    const bits = Dives.select(controls, { label: "Bits per weight", options: [["32", "fp32"], ["16", "fp16"], ["8", "int8"], ["4", "4-bit"]], value: "8", onChange: () => update() });
    const outlier = Dives.slider(controls, { label: "Outlier factor on row 0", min: 1, max: 40, step: 1, value: 1, format: (v) => `x${v}`, onInput: () => update() });
    const mode = Dives.select(controls, { label: "Scale", options: [["row", "one per row"], ["tensor", "one per tensor"]], value: "row", onChange: () => update() });
    const readout = Dives.readout(el);
    const plot = Dives.canvas(el, { aspect: 0.45, label: "Mean rounding error for eight rows of a toy weight matrix" });
    const rnd = mulberry(7);
    const rows = Array.from({ length: 8 }, () => Array.from({ length: 32 }, () => gauss(rnd) * 0.05));
    let errs = new Array(8).fill(0);
    let levels = 127;

    function update() {
      const b = +bits.get();
      levels = b >= 16 ? 0 : 2 ** (b - 1) - 1;
      const data = rows.map((r, i) => (i === 0 ? r.map((v) => v * outlier.get()) : r));
      const tmax = Math.max(...data.flat().map(Math.abs));
      errs = data.map((r) => {
        if (!levels) return 0;
        const m = mode.get() === "tensor" ? tmax : Math.max(...r.map(Math.abs));
        const s = m / levels;
        return r.reduce((a, v) => a + Math.abs(v - Math.round(v / s) * s), 0) / r.length;
      });
      const p = params.get() * 1e6;
      const weights = (p * b) / 8 / 1e6;
      // our arithmetic: one fp32 scale per row, assuming 512 weights per row
      const scales = levels && mode.get() === "row" ? (p / 512) * 4 / 1e6 : 0;
      readout.textContent = `Weights: ${weights.toFixed(1)} MB` + (scales ? ` + about ${scales.toFixed(2)} MB of row scales (our assumption: 512 weights per row)` : "") + `. The post reports 31.7 MB for 30.7 M parameters.`;
      plot.redraw();
    }
    plot.draw = (ctx, w, h) => {
      const c = Dives.colors();
      const pad = 28, bw = (w - pad * 2) / 8;
      const mx = Math.max(...errs, 1e-9);
      ctx.font = `11px ${c.sans}`;
      ctx.textAlign = "center";
      errs.forEach((e, i) => {
        const bh = ((h - pad * 2) * e) / mx;
        ctx.fillStyle = i === 0 ? c.accent : c.forest;
        ctx.fillRect(pad + i * bw + 4, h - pad - bh, bw - 8, bh);
        ctx.fillStyle = c.muted;
        ctx.fillText(`row ${i}`, pad + i * bw + bw / 2, h - 10);
      });
      ctx.textAlign = "left";
      ctx.fillStyle = c.ink;
      ctx.fillText(levels ? `Mean |error|, tallest bar = ${mx.toFixed(5)}` : "No rounding at fp16/fp32 in this toy (error shown as 0)", pad, 14);
    };
    update();
    el.append(Dives.el("p", "widget-caption", "Raise the outlier factor with one scale per tensor and every quiet row gets coarser. With one scale per row, only row 0 pays."));
  });

  // ---------------------------------------------------------------- toy generation loop
  Dives.mount("toy-loop", (el) => {
    el.append(Dives.el("p", "widget-title", "A toy generation loop (not Woof)"));
    const controls = Dives.el("div", "widget-controls");
    el.append(controls);
    const redraw = () => run();
    const temp = Dives.slider(controls, { label: "Temperature", min: 0, max: 1.5, step: 0.05, value: 0.5, format: (v) => v.toFixed(2), onInput: redraw });
    const thr = Dives.slider(controls, { label: "EOS threshold", min: 0.5, max: 0.99, step: 0.01, value: 0.9, format: (v) => v.toFixed(2), onInput: redraw });
    const minf = Dives.slider(controls, { label: "Minimum frames", min: 0, max: 40, step: 1, value: 0, onInput: redraw });
    const readout = Dives.readout(el);
    const plot = Dives.canvas(el, { aspect: 0.5, label: "Toy latent frames and end-of-speech logit" });
    let res = { frames: [], eos: [], stop: 0, t: 0 };

    function run() {
      const rnd = mulberry(3);
      const target = 18, eosT = Math.log(thr.get() / (1 - thr.get()));
      let state = 0;
      const frames = [], eos = [];
      for (let f = 0; f < 60; f++) {
        const e = 0.9 * (f - target) / 2 + 0.3 * state;
        eos.push(e);
        if (f > Math.max(3, minf.get()) && e > eosT) break;
        const z = gauss(rnd) * temp.get() + 0.8 * Math.sin(0.5 * f) - 0.4 * state;
        frames.push(z);
        state = 0.7 * state + 0.5 * z;
      }
      res = { frames, eos, eosT };
      readout.textContent = `Stopped after ${frames.length} frames (a frame-count cutoff of 60 applies). EOS logit threshold: ${eosT.toFixed(2)}.`;
      plot.redraw();
    }
    plot.draw = (ctx, w, h) => {
      const c = Dives.colors();
      const pad = 24, n = 60;
      const x = (i) => pad + ((w - pad * 2) * i) / (n - 1);
      const y = (v) => h / 2 - (v * (h / 2 - pad)) / 3;
      ctx.strokeStyle = c.line;
      ctx.beginPath(); ctx.moveTo(pad, h / 2); ctx.lineTo(w - pad, h / 2); ctx.stroke();
      const line = (arr, color) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
        arr.forEach((v, i) => (i ? ctx.lineTo(x(i), y(Math.max(-3, Math.min(3, v)))) : ctx.moveTo(x(i), y(v))));
        ctx.stroke(); ctx.lineWidth = 1;
      };
      line(res.frames, c.forest);
      line(res.eos, c.accent);
      ctx.strokeStyle = c.error; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(pad, y(res.eosT)); ctx.lineTo(w - pad, y(res.eosT)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.ink; ctx.font = `11px ${c.sans}`;
      ctx.fillText("frames (forest), EOS logit (accent)", pad, 14);
      ctx.fillText(`stop at frame ${res.frames.length}`, Math.min(w - 110, x(res.frames.length)), h - 8);
    };
    run();
    el.append(Dives.el("p", "widget-caption", "Lower the threshold and the loop stops sooner. Raise the temperature and the frames spread out. The dashed line is the threshold. The toy stop rule is hand-made, so only the control flow is faithful."));
  });
})();

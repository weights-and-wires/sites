// Shared helpers for deep-dive widgets. Each dive's own dive.js loads after
// this file and uses window.Dives to build controls and draw on canvases that
// follow the light and dark themes.
//
//   Dives.mount("name", (el) => { ... })   run for each [data-widget="name"]
//   Dives.colors()                         current theme colors by role
//   Dives.canvas(el, {aspect, height})     a crisp canvas that redraws on
//                                          resize and theme change
//   Dives.slider(parent, {...})            a labeled range input with readout
//   Dives.toggle(parent, {...})            a labeled checkbox
//   Dives.select(parent, {...})            a labeled select
//   Dives.button(parent, label, onClick)   a small button
//   Dives.readout(parent)                  a live text line for numbers
//   Dives.onTheme(fn)                      call fn when the theme changes
//   Dives.reducedMotion                    true when the reader asked for less motion
(() => {
  "use strict";
  const root = document.documentElement;
  const css = (name) => getComputedStyle(root).getPropertyValue(`--${name}`).trim();

  // colors returns the palette for the current theme. Read it inside your draw
  // function, not once at load, so a theme switch takes effect.
  function colors() {
    return {
      paper: css("paper"), surface: css("surface"), surfaceMuted: css("surface-muted"),
      ink: css("ink"), muted: css("muted"), line: css("line"), strongLine: css("strong-line"),
      forest: css("forest"), forestSoft: css("forest-soft"),
      accent: css("accent"), accentInk: css("accent-ink"), accentSoft: css("accent-soft"),
      error: css("error"), errorSoft: css("error-soft"),
      series: [1, 2, 3, 4, 5, 6].map((i) => css(`series-${i}`)),
      mono: css("mono") || "monospace", sans: css("sans") || "system-ui, sans-serif",
    };
  }

  const themeListeners = new Set();
  new MutationObserver(() => themeListeners.forEach((fn) => fn())).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => themeListeners.forEach((fn) => fn()));
  function onTheme(fn) { themeListeners.add(fn); return () => themeListeners.delete(fn); }

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // canvas appends a canvas to parent and sizes it to the parent's width at
  // the device pixel ratio. Set draw with c.draw = (ctx, w, h) => {...}; it
  // runs on resize and theme change, and whenever you call c.redraw().
  // Coordinates are CSS pixels.
  function canvas(parent, opts = {}) {
    const cv = el("canvas", "widget-canvas");
    if (opts.label) { cv.setAttribute("role", "img"); cv.setAttribute("aria-label", opts.label); }
    parent.append(cv);
    const ctx = cv.getContext("2d");
    const c = { el: cv, ctx, width: 0, height: 0, draw: null };
    let queued = false;
    c.redraw = () => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        const w = Math.max(1, Math.round(cv.clientWidth || parent.clientWidth));
        const h = opts.height ? (typeof opts.height === "function" ? opts.height(w) : opts.height) : Math.round(w * (opts.aspect || 0.5));
        const dpr = window.devicePixelRatio || 1;
        if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
          cv.width = Math.round(w * dpr);
          cv.height = Math.round(h * dpr);
          cv.style.height = `${h}px`;
        }
        c.width = w;
        c.height = h;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        if (c.draw) c.draw(ctx, w, h);
      });
    };
    new ResizeObserver(c.redraw).observe(parent);
    onTheme(c.redraw);
    return c;
  }

  let uid = 0;
  function labeled(parent, label, input, cls) {
    const id = `dive-ctl-${++uid}`;
    input.id = id;
    const wrap = el("div", `widget-control ${cls || ""}`.trim());
    const lab = el("label", "", label);
    lab.htmlFor = id;
    wrap.append(lab, input);
    parent.append(wrap);
    return wrap;
  }

  // slider: {label, min, max, step, value, format(v) -> string, onInput(v)}.
  // Returns {input, get(), set(v)}.
  function slider(parent, o) {
    const input = el("input");
    Object.assign(input, { type: "range", min: o.min, max: o.max, step: o.step ?? 1, value: o.value });
    const out = el("output", "widget-value");
    const wrap = labeled(parent, o.label, input, "widget-slider");
    wrap.append(out);
    const fmt = o.format || ((v) => String(v));
    const get = () => Number(input.value);
    const show = () => { out.textContent = fmt(get()); };
    input.addEventListener("input", () => { show(); if (o.onInput) o.onInput(get()); });
    show();
    return { input, get, set(v) { input.value = v; show(); if (o.onInput) o.onInput(get()); } };
  }

  // toggle: {label, checked, onChange(bool)}.
  function toggle(parent, o) {
    const input = el("input");
    input.type = "checkbox";
    input.checked = !!o.checked;
    const wrap = labeled(parent, o.label, input, "widget-toggle");
    wrap.prepend(input);
    input.addEventListener("change", () => o.onChange && o.onChange(input.checked));
    return { input, get: () => input.checked, set(v) { input.checked = v; if (o.onChange) o.onChange(v); } };
  }

  // select: {label, options: [[value, text], ...], value, onChange(value)}.
  function select(parent, o) {
    const input = el("select");
    for (const [v, t] of o.options) {
      const opt = el("option", "", t);
      opt.value = v;
      input.append(opt);
    }
    if (o.value !== undefined) input.value = o.value;
    labeled(parent, o.label, input, "widget-select");
    input.addEventListener("change", () => o.onChange && o.onChange(input.value));
    return { input, get: () => input.value, set(v) { input.value = v; if (o.onChange) o.onChange(v); } };
  }

  function button(parent, label, onClick) {
    const b = el("button", "button widget-button", label);
    b.type = "button";
    b.addEventListener("click", onClick);
    parent.append(b);
    return b;
  }

  function readout(parent) {
    const r = el("p", "widget-readout");
    r.setAttribute("aria-live", "polite");
    parent.append(r);
    return r;
  }

  // mount calls init(el) for each [data-widget=name]. A failing widget shows a
  // short message instead of breaking the rest of the page.
  function mount(name, init) {
    document.querySelectorAll(`[data-widget="${name}"]`).forEach((w) => {
      try {
        init(w);
      } catch (err) {
        console.error(`widget ${name}:`, err);
        w.append(el("p", "runner-error", `This widget failed to load: ${err.message}`));
      }
    });
  }

  window.Dives = {
    mount, colors, canvas, slider, toggle, select, button, readout, onTheme, el,
    reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  };
})();

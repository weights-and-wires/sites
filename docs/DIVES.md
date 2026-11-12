# Write a deep dive

A deep dive teaches one external technical post, such as a blog post or a lab
write-up, from first principles. It lives at `papers.ifkash.dev/dives/<id>/`
and uses the paper page layout, plus widgets that the reader can change and
Python cells that run in the browser.

## Files

A dive folder holds the following files:

```text
content/dives/<id>/
  index.md          # front matter plus the page body
  dive.js           # optional: the page's widgets
  dive.css          # optional: styles that only this dive needs
  cells/<name>.py   # optional: editable Python cells
  scripts/*.py      # optional: scripts that compute numbers on the page
```

## Front matter

The following example shows every field:

```yaml
---
id: train-a-swarm
title: "Train a swarm of agents"
summary: "Three plain-text sentences for the TL;DR and the index card."
source:
  title: "How to train a swarm"
  author: "Example Lab team"
  site: "Example Lab"
  date: "2026-10-26"      # ISO date; the index lists newest first
  url: "https://example.com/blog/how-to-train-a-swarm"
topics: [rl, multi-agent]
est_minutes: 40
prereqs:                   # paper IDs from content/papers/
  - {id: ppo, reason: "One sentence on what this dive needs from PPO."}
background: [policy-gradient]   # concept IDs from content/papers/concepts.yaml
course_links:
  - {title: "RL lesson 41: GRPO", url: "https://rl.ifkash.dev/..."}
---
```

The build fails when `title`, `summary`, `source.title`, `source.author`, or an
`https` `source.url` is missing, when a prerequisite isn't a paper or has no
reason, or when a background ID isn't in `concepts.yaml`.

## Body

The template renders the header with a credit and link to the original post,
the TL;DR, and the "Before you read" panel. The body supplies the rest, usually
in this order:

1. `## The problem`: what the post solves, and why the naive approach fails.
1. `## Key ideas`, with one `###` heading per idea. Give the intuition, the
   math where it helps, and a small numeric example the reader can check by
   hand.
1. `## Rebuild it`: a small, runnable version of the core technique, as one or
   more Python cells.
1. `## Pitfalls` or `## What the post doesn't say`: misreadings, limits, and
   anything the post leaves out.
1. `## Further reading`: the original post first, then code and papers. This
   section renders last.

Use `$...$` for inline math and `$$...$$` on its own lines for display math.
Explain in your own words. Quote only short phrases, and redraw figures as
widgets instead of copying them.

## Widgets

To place a widget, put an empty element on its own line in the body:

```html
<div class="widget" data-widget="cap-explorer"></div>
```

In `dive.js`, build it with the helpers in `static/dives.js`, which every dive
page loads first:

```js
Dives.mount("cap-explorer", (el) => {
  el.append(Dives.el("p", "widget-title", "Wall-clock cap"));
  const controls = Dives.el("div", "widget-controls");
  el.append(controls);
  const plot = Dives.canvas(el, { aspect: 0.5, label: "Return against team size" });
  const n = Dives.slider(controls, { label: "Agents", min: 1, max: 32, value: 4, onInput: () => plot.redraw() });
  plot.draw = (ctx, w, h) => {
    const c = Dives.colors(); // read colors inside draw so theme changes apply
    ctx.strokeStyle = c.forest;
    // draw with n.get()
  };
  plot.redraw();
  el.append(Dives.el("p", "widget-caption", "What the reader should notice."));
});
```

The helpers are `mount`, `el`, `colors`, `canvas`, `slider`, `toggle`,
`select`, `button`, `readout`, `onTheme`, and `reducedMotion`. For details, see
the comment at the top of `static/dives.js`.

Follow these rules for widgets:

- Use plain JavaScript with canvas or SVG. Don't add frameworks or load
  scripts from other hosts. The Content Security Policy blocks inline scripts
  and other origins.
- Take every color from `Dives.colors()` or a CSS variable, so the widget
  works in both themes.
- Make it work at 360 px wide. Use pointer events, not mouse events, for
  dragging.
- Keep each update fast enough to run on every slider input.

## Python cells

To place an editable cell, put a marker on its own line in the body:

```html
<!-- cell: reconstruct -->
```

The marker names `cells/reconstruct.py`. The build fails on a marker without a
file, and on a file without a marker. The reader can edit the cell and click
**Run**. The cell runs in a fresh namespace in Pyodide with NumPy, and the page
shows what it prints.

A cell that sets a global named `result` to JSON-compatible data (NumPy arrays
are converted with `tolist`) sends it to the page. A widget can listen for it
and draw it:

```js
document.addEventListener("cell:result", (e) => {
  if (e.detail.name === "train-a-swarm/reconstruct") draw(e.detail.result);
});
```

A cell imports only NumPy and the standard library, and runs in under
5 seconds.

## Validate

Run these commands from the repository root before you open a PR:

```bash
go run ./cmd/build -offline -check
```

```bash
python3 scripts/check_exercises.py content/dives/DIVE_ID
```

Replace `DIVE_ID` with the folder name. The second command runs every cell.
Then serve the site, open the page, and check both themes, every widget, and a
phone-width window:

```bash
go run ./cmd/serve -addr 127.0.0.1:8080
```

Open http://papers.localhost:8080/dives/DIVE_ID/.

# sites

This repository builds two static sites and holds the shared course design
system:

- papers.ifkash.dev: one page per paper, with prerequisites and exercises.
- courses.ifkash.dev: the hub that links every course.
- `ui/`: the design tokens, type, components, theme script, and Monaco themes
  that every course app vendors. For details, see [`ui/README.md`](ui/README.md).

## How it works

`cmd/build` renders markdown from `content/` with goldmark and chroma, renders
TeX once with KaTeX in Node (`scripts/katex.mjs`), and writes one directory per
site to `internal/assets/public/`. `cmd/serve` embeds that directory and serves
it, picking the site by the `Host` header. The container image is `FROM
scratch` with the one binary.

## Build and run locally

1. Install the build dependencies, and download the pinned Pyodide that the
   exercise runner uses:

   ```bash
   npm ci
   ```

   ```bash
   scripts/fetch-pyodide.sh
   ```

1. Build both sites and check the internal links:

   ```bash
   go run ./cmd/build -check
   ```

1. Serve them:

   ```bash
   go run ./cmd/serve -addr 127.0.0.1:8080
   ```

1. Open http://papers.localhost:8080/ or http://courses.localhost:8080/.

To check everything that CI checks, run `go vet ./...`, `go test ./...`,
`python3 scripts/check_exercises.py`, and `docker build -t sites:dev .`.

## Papers

Each paper lives in `content/papers/<id>/`. To write or edit a page, follow
[`docs/PAPER_PAGES.md`](docs/PAPER_PAGES.md). The build fails on a
prerequisite cycle, an unknown paper or concept ID, bad TeX, a full page
without exercises, or a broken internal link.

- `content/papers/concepts.yaml` maps each background concept to the course
  lesson that teaches it.
- `content/paths/` holds the reading paths.
- `scripts/import_papers.py` created the stub pages and paths from the
  research list. It never overwrites a page.
- Exercises run in the browser in a Web Worker with Pyodide and NumPy, served
  from `/pyodide/`. The worker starts on the first **Run tests** click.
  `scripts/check_exercises.py` runs the same harness
  (`static/runner/harness.py`) in CI: the starter must fail, and the solution
  must pass.
- `scripts/check_course_links.py` checks that every course lesson URL still
  exists on the course's default branch. CI runs it only when the
  `COURSES_READ_TOKEN` secret is set; run it locally after a course renumbers
  its lessons.

## Deploy

A push to `main` runs `.github/workflows/deploy.yml`. The workflow builds
`ghcr.io/weights-and-wires/sites`, pushes it, and then asks the VM to pull and
restart the `sites` service. The VM only pulls images.

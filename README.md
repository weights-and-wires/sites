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

1. Install the build dependencies:

   ```bash
   npm ci
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

To check everything that CI checks, run `go vet ./...`, `go test ./...`, and
`docker build -t sites:dev .`.

## Deploy

A push to `main` runs `.github/workflows/deploy.yml`. The workflow builds
`ghcr.io/weights-and-wires/sites`, pushes it, and then asks the VM to pull and
restart the `sites` service. The VM only pulls images.

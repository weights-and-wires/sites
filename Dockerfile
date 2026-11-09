# Builds papers.ifkash.dev and courses.ifkash.dev into one static Go binary.
#   1. node: installs KaTeX and the CodeMirror bundler (esbuild).
#   2. pyodide: downloads the pinned Pyodide core and NumPy, checking SHA-256.
#   3. go: renders the sites (calling Node for KaTeX and the editor bundle),
#      then compiles serve with the sites embedded.
#   4. scratch: the binary and CA certificates only.

FROM python:3.13-alpine AS pyodide
RUN apk add --no-cache curl
WORKDIR /src
COPY scripts/fetch-pyodide.sh scripts/
RUN sh scripts/fetch-pyodide.sh

FROM node:22-alpine AS node
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund

FROM golang:1.27-alpine AS build
RUN apk add --no-cache nodejs ca-certificates
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY --from=node /src/node_modules ./node_modules
COPY . .
COPY --from=pyodide /src/.cache/pyodide ./.cache/pyodide
ARG GIT_SHA=dev
ENV CGO_ENABLED=0
RUN go run ./cmd/build -check -release \
 && go test ./... \
 && go build -trimpath -ldflags "-s -w -X main.version=${GIT_SHA}" -o /out/serve ./cmd/serve

FROM scratch
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=build /out/serve /serve
ENV GOMAXPROCS=1 GOMEMLIMIT=24MiB GOGC=50
USER 65534:65534
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=3s --retries=3 CMD ["/serve", "-healthcheck"]
ENTRYPOINT ["/serve"]

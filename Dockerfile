# Builds papers.ifkash.dev and courses.ifkash.dev into one static Go binary.
#   1. node: installs KaTeX for build-time math.
#   2. go: renders the sites (calling Node for KaTeX), then compiles serve
#      with the sites embedded.
#   3. scratch: the binary and CA certificates only.

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
ARG GIT_SHA=dev
ENV CGO_ENABLED=0
RUN go run ./cmd/build -check \
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

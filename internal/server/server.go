// Package server serves the built sites. It picks a site by the Host header,
// serves precompressed siblings when the client accepts them, and sets the
// same security headers as the arm64 nginx config.
package server

import (
	"io"
	"io/fs"
	"mime"
	"net"
	"net/http"
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

// CSP allows only same-origin resources. 'wasm-unsafe-eval' lets the papers
// site run Pyodide.
const CSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; " +
	"style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; " +
	"font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'none'"

// hashed matches fingerprinted asset names such as site.3f2a9c1b.css.
var hashed = regexp.MustCompile(`\.[0-9a-f]{8,}\.[a-z0-9]+$`)

func init() {
	// The scratch image has no /etc/mime.types, so register what the sites ship.
	for ext, typ := range map[string]string{
		".woff2": "font/woff2",
		".zip":   "application/zip",
		".whl":   "application/zip",
		".py":    "text/x-python; charset=utf-8",
		".txt":   "text/plain; charset=utf-8",
		".ico":   "image/x-icon",
	} {
		mime.AddExtensionType(ext, typ)
	}
}

// Handler serves sites from root. hosts maps a Host header value (without a
// port) to a directory under root.
type Handler struct {
	root  fs.FS
	hosts map[string]string
	etags sync.Map
}

// New returns a Handler. Unknown hosts get a 404.
func New(root fs.FS, hosts map[string]string) *Handler {
	return &Handler{root: root, hosts: hosts}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	hdr := w.Header()
	hdr.Set("X-Content-Type-Options", "nosniff")
	hdr.Set("Referrer-Policy", "strict-origin-when-cross-origin")
	hdr.Set("X-Frame-Options", "DENY")
	hdr.Set("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
	hdr.Set("Content-Security-Policy", CSP)

	if r.URL.Path == "/health" {
		hdr.Set("Content-Type", "text/plain; charset=utf-8")
		hdr.Set("Cache-Control", "no-store")
		io.WriteString(w, "ok\n")
		return
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		hdr.Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	host := r.Host
	if hh, _, err := net.SplitHostPort(host); err == nil {
		host = hh
	}
	site, ok := h.hosts[strings.ToLower(host)]
	if !ok {
		http.Error(w, "unknown host", http.StatusNotFound)
		return
	}

	name := path.Clean("/" + r.URL.Path)
	if strings.HasSuffix(r.URL.Path, "/") || name == "/" {
		name = path.Join(name, "index.html")
	}
	file := path.Join(site, strings.TrimPrefix(name, "/"))
	if st, err := fs.Stat(h.root, file); err == nil && st.IsDir() {
		http.Redirect(w, r, r.URL.Path+"/", http.StatusMovedPermanently)
		return
	}
	if _, err := fs.Stat(h.root, file); err != nil {
		h.notFound(w, r, site)
		return
	}

	if hashed.MatchString(file) {
		hdr.Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		hdr.Set("Cache-Control", "no-cache")
	}
	h.serveFile(w, r, file, http.StatusOK)
}

func (h *Handler) notFound(w http.ResponseWriter, r *http.Request, site string) {
	page := path.Join(site, "404.html")
	if _, err := fs.Stat(h.root, page); err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Cache-Control", "no-cache")
	h.serveFile(w, r, page, http.StatusNotFound)
}

// serveFile writes file, or its .br or .gz sibling when the client accepts
// that encoding.
func (h *Handler) serveFile(w http.ResponseWriter, r *http.Request, file string, status int) {
	hdr := w.Header()
	ctype := mime.TypeByExtension(path.Ext(file))
	if ctype == "" {
		ctype = "application/octet-stream"
	}
	hdr.Set("Content-Type", ctype)
	hdr.Add("Vary", "Accept-Encoding")

	accept := r.Header.Get("Accept-Encoding")
	send := file
	for _, enc := range []struct{ token, ext string }{{"br", ".br"}, {"gzip", ".gz"}} {
		if acceptsEncoding(accept, enc.token) {
			if _, err := fs.Stat(h.root, file+enc.ext); err == nil {
				send = file + enc.ext
				hdr.Set("Content-Encoding", enc.token)
				break
			}
		}
	}

	f, err := h.root.Open(send)
	if err != nil {
		http.Error(w, "read error", http.StatusInternalServerError)
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		http.Error(w, "read error", http.StatusInternalServerError)
		return
	}
	hdr.Set("Content-Length", strconv.FormatInt(st.Size(), 10))
	if status == http.StatusOK {
		etag, err := h.etag(send)
		if err != nil {
			http.Error(w, "read error", http.StatusInternalServerError)
			return
		}
		hdr.Set("ETag", etag)
		if r.Header.Get("If-None-Match") == etag {
			hdr.Del("Content-Length")
			w.WriteHeader(http.StatusNotModified)
			return
		}
	}
	w.WriteHeader(status)
	if r.Method != http.MethodHead {
		// Stream the file: Pyodide's wasm is several MB, and the container
		// has a 32 MiB memory limit.
		io.Copy(w, f)
	}
}

// etag hashes a file once and caches the result. The files never change
// while the server runs.
func (h *Handler) etag(file string) (string, error) {
	if v, ok := h.etags.Load(file); ok {
		return v.(string), nil
	}
	f, err := h.root.Open(file)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := uint64(14695981039346656037)
	buf := make([]byte, 32<<10)
	for {
		n, err := f.Read(buf)
		for _, c := range buf[:n] {
			hash ^= uint64(c)
			hash *= 1099511628211
		}
		if err == io.EOF {
			break
		} else if err != nil {
			return "", err
		}
	}
	tag := `"` + strconv.FormatUint(hash, 16) + `"`
	h.etags.Store(file, tag)
	return tag, nil
}

func acceptsEncoding(header, token string) bool {
	for _, part := range strings.Split(header, ",") {
		name, params, _ := strings.Cut(strings.TrimSpace(part), ";")
		if strings.EqualFold(strings.TrimSpace(name), token) {
			return !strings.Contains(strings.ReplaceAll(params, " ", ""), "q=0")
		}
	}
	return false
}

// Server returns an http.Server with conservative timeouts.
func Server(addr string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    16 << 10,
	}
}

package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

func testHandler() *Handler {
	root := fstest.MapFS{
		"papers/index.html":               {Data: []byte("<h1>papers</h1>")},
		"papers/404.html":                 {Data: []byte("<h1>missing</h1>")},
		"papers/transformer/index.html":   {Data: []byte("<h1>transformer</h1>")},
		"papers/assets/site.0123abcd.css": {Data: []byte("body{}")},
		"courses/index.html":              {Data: []byte("<h1>hub</h1>")},
		"courses/index.html.gz":           {Data: []byte("gzipped")},
		"courses/index.html.br":           {Data: []byte("brotli")},
	}
	return New(root, map[string]string{
		"papers.ifkash.dev":  "papers",
		"courses.ifkash.dev": "courses",
	})
}

func get(t *testing.T, host, path string, header map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	req.Host = host
	for k, v := range header {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	testHandler().ServeHTTP(rec, req)
	return rec
}

func TestRoutesByHost(t *testing.T) {
	if body := get(t, "papers.ifkash.dev", "/", nil).Body.String(); body != "<h1>papers</h1>" {
		t.Errorf("papers / = %q", body)
	}
	if body := get(t, "courses.ifkash.dev:8080", "/", nil).Body.String(); body != "<h1>hub</h1>" {
		t.Errorf("courses / = %q", body)
	}
	if body := get(t, "papers.ifkash.dev", "/transformer/", nil).Body.String(); body != "<h1>transformer</h1>" {
		t.Errorf("papers /transformer/ = %q", body)
	}
}

func TestDirectoryWithoutSlashRedirects(t *testing.T) {
	rec := get(t, "papers.ifkash.dev", "/transformer", nil)
	if rec.Code != http.StatusMovedPermanently || rec.Header().Get("Location") != "/transformer/" {
		t.Errorf("got %d to %q", rec.Code, rec.Header().Get("Location"))
	}
}

func TestHealth(t *testing.T) {
	rec := get(t, "anything.example.com", "/health", nil)
	if rec.Code != 200 || rec.Body.String() != "ok\n" {
		t.Errorf("health = %d %q", rec.Code, rec.Body.String())
	}
}

func TestUnknownHostIs404(t *testing.T) {
	if rec := get(t, "evil.example.com", "/", nil); rec.Code != http.StatusNotFound {
		t.Errorf("unknown host = %d, want 404", rec.Code)
	}
}

func TestMissingPageServesSite404(t *testing.T) {
	rec := get(t, "papers.ifkash.dev", "/nope/", nil)
	if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), "missing") {
		t.Errorf("missing page = %d %q", rec.Code, rec.Body.String())
	}
}

func TestSecurityHeaders(t *testing.T) {
	h := get(t, "papers.ifkash.dev", "/", nil).Header()
	want := map[string]string{
		"Content-Security-Policy": CSP,
		"X-Content-Type-Options":  "nosniff",
		"X-Frame-Options":         "DENY",
		"Referrer-Policy":         "strict-origin-when-cross-origin",
	}
	for k, v := range want {
		if h.Get(k) != v {
			t.Errorf("%s = %q, want %q", k, h.Get(k), v)
		}
	}
	if !strings.Contains(h.Get("Content-Security-Policy"), "'wasm-unsafe-eval'") {
		t.Error("CSP must allow wasm for Pyodide")
	}
}

func TestPrecompressed(t *testing.T) {
	rec := get(t, "courses.ifkash.dev", "/", map[string]string{"Accept-Encoding": "gzip, br"})
	if rec.Header().Get("Content-Encoding") != "br" || rec.Body.String() != "brotli" {
		t.Errorf("br: encoding %q body %q", rec.Header().Get("Content-Encoding"), rec.Body.String())
	}
	rec = get(t, "courses.ifkash.dev", "/", map[string]string{"Accept-Encoding": "gzip"})
	if rec.Header().Get("Content-Encoding") != "gzip" || rec.Body.String() != "gzipped" {
		t.Errorf("gzip: encoding %q body %q", rec.Header().Get("Content-Encoding"), rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("Content-Type = %q, want text/html", ct)
	}
	rec = get(t, "courses.ifkash.dev", "/", nil)
	if rec.Header().Get("Content-Encoding") != "" || rec.Body.String() != "<h1>hub</h1>" {
		t.Errorf("identity: encoding %q body %q", rec.Header().Get("Content-Encoding"), rec.Body.String())
	}
}

func TestCacheControl(t *testing.T) {
	if cc := get(t, "papers.ifkash.dev", "/assets/site.0123abcd.css", nil).Header().Get("Cache-Control"); !strings.Contains(cc, "immutable") {
		t.Errorf("hashed asset Cache-Control = %q", cc)
	}
	if cc := get(t, "papers.ifkash.dev", "/", nil).Header().Get("Cache-Control"); cc != "no-cache" {
		t.Errorf("page Cache-Control = %q", cc)
	}
}

func TestRejectsPost(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/", nil)
	req.Host = "papers.ifkash.dev"
	rec := httptest.NewRecorder()
	testHandler().ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST = %d", rec.Code)
	}
}

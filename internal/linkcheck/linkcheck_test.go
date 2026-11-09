package linkcheck

import (
	"testing"
	"testing/fstest"
)

func TestCheck(t *testing.T) {
	root := fstest.MapFS{
		"papers/index.html":   {Data: []byte(`<a href="/a/">ok</a><a href="/missing/">bad</a><a href="https://courses.ifkash.dev/">ok</a><a href="https://example.com/">skip</a><a href="#top">skip</a>`)},
		"papers/a/index.html": {Data: []byte(`<img src="../assets/x.svg"><a href="https://courses.ifkash.dev/nope/">bad</a>`)},
		"papers/assets/x.svg": {Data: []byte(`<svg/>`)},
		"courses/index.html":  {Data: []byte(`ok`)},
	}
	got := Check(root, map[string]string{"papers.ifkash.dev": "papers", "courses.ifkash.dev": "courses"})
	want := []string{"papers/a/index.html: https://courses.ifkash.dev/nope/", "papers/index.html: /missing/"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("got %q, want %q", got, want)
	}
}

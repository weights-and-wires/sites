package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/weights-and-wires/sites/internal/content"
)

// renderHub builds the courses site's assets and hub page from the repo's
// real templates and courses.yaml, offline, and returns the page and assets.
func renderHub(t *testing.T) (string, Assets) {
	t.Helper()
	b := &builder{root: "../..", out: t.TempDir(), offline: true, built: "test"}
	hub, err := content.LoadHub(filepath.Join(b.root, "content/hub/courses.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	assets, err := b.writeAssets("courses", false)
	if err != nil {
		t.Fatal(err)
	}
	page := Page{Site: sites[1], Assets: assets, Title: "Courses", Built: "test", Hub: b.hubView(hub, nil, 0, 0)}
	if err := b.page("courses", "index.html", "hub.html", page); err != nil {
		t.Fatal(err)
	}
	html, err := os.ReadFile(filepath.Join(b.out, "courses/index.html"))
	if err != nil {
		t.Fatal(err)
	}
	return string(html), assets
}

func TestHubRendersProgressOnlyForOptedInCourses(t *testing.T) {
	html, assets := renderHub(t)
	want := []string{"ml", "rl", "speech", "gpu", "qwen"}
	for _, id := range want {
		attr := `data-progress-url="https://` + id + `.ifkash.dev/api/progress/summary"`
		if !strings.Contains(html, attr) {
			t.Errorf("hub page lacks %s", attr)
		}
	}
	if got := strings.Count(html, "data-progress-url="); got != len(want) {
		t.Errorf("data-progress-url appears %d times, want %d", got, len(want))
	}
	if got := strings.Count(html, `data-progress-out aria-live="polite"`); got != len(want) {
		t.Errorf("progress placeholders = %d, want %d", got, len(want))
	}
	for _, id := range []string{"os", "cpu", "tpu", "arm64", "papers"} {
		if strings.Contains(html, "https://"+id+".ifkash.dev/api/progress") {
			t.Errorf("course %s must not render a progress URL", id)
		}
	}
	if assets.HubJS == "" || !strings.Contains(html, `<script src="/assets/`+assets.HubJS+`" defer></script>`) {
		t.Errorf("hub page doesn't load hub.js (%q)", assets.HubJS)
	}
}

func TestHubScriptLoadsOnlyOnHub(t *testing.T) {
	b := &builder{root: "../..", out: t.TempDir()}
	a, err := b.writeAssets("papers", false)
	if err != nil {
		t.Skipf("papers assets need npm ci: %v", err)
	}
	if a.HubJS != "" {
		t.Error("papers site must not get hub.js")
	}
}

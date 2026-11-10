package content

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, path, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadHubSortsAndChecksStages(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "courses.yaml")
	write(t, f, "stages: [{id: a, label: A}]\ncourses:\n- {id: two, name: Two, url: https://two.example.com/, stage: a, order: 2}\n- {id: one, name: One, url: https://one.example.com/, stage: a, order: 1}\n")
	h, err := LoadHub(f)
	if err != nil {
		t.Fatal(err)
	}
	if h.Courses[0].ID != "one" {
		t.Errorf("courses not sorted by order: %+v", h.Courses)
	}
	write(t, f, "stages: [{id: a, label: A}]\ncourses:\n- {id: x, name: X, url: https://x.example.com/, stage: nope}\n")
	if _, err := LoadHub(f); err == nil || !strings.Contains(err.Error(), "unknown stage") {
		t.Errorf("want unknown stage error, got %v", err)
	}
}

func TestLoadPapersAndPaths(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "papers/a/index.md"), "---\nid: a\ntitle: A paper\nauthors: X et al.\nyear: 2020\narxiv: \"1\"\ntrack: foundations\ntier: core\nstatus: stub\nsummary: S.\n---\nBody\n")
	write(t, filepath.Join(dir, "papers/a/exercises/1-x/meta.json"), `{"title": "T", "goal": "G"}`)
	for _, f := range []string{"starter.py", "tests.py", "solution.py"} {
		write(t, filepath.Join(dir, "papers/a/exercises/1-x", f), "")
	}
	write(t, filepath.Join(dir, "paths/first.md"), "---\ntitle: First path\n---\nSteps\n")
	papers, err := LoadPapers(filepath.Join(dir, "papers"))
	if err != nil || len(papers) != 1 || len(papers[0].Exercises) != 1 || papers[0].Body != "Body\n" {
		t.Fatalf("LoadPapers = %+v, %v", papers, err)
	}
	paths, err := LoadPaths(filepath.Join(dir, "paths"))
	if err != nil || len(paths) != 1 || paths[0].ID != "first" || paths[0].Title != "First path" {
		t.Fatalf("LoadPaths = %+v, %v", paths, err)
	}
	if none, err := LoadPaths(filepath.Join(dir, "missing")); err != nil || len(none) != 0 {
		t.Fatalf("missing dir: %v, %v", none, err)
	}
}

func TestLoadPapersRejectsBadFrontMatter(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "b/index.md"), "---\nid: b\ntitle: B\nauthors: X\nyear: 2020\narxiv: \"1\"\ntrack: nope\ntier: core\nstatus: stub\nsummary: S.\n---\n")
	if _, err := LoadPapers(dir); err == nil || !strings.Contains(err.Error(), "unknown track") {
		t.Fatalf("want unknown track error, got %v", err)
	}
}

func TestLoadConceptsNeedsURL(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "c.yaml")
	write(t, f, "concepts:\n  x: {label: X, url: 'ftp://x', refresher: R}\n")
	if _, err := LoadConcepts(f); err == nil {
		t.Fatal("want an error for a non-https url")
	}
}

func TestHubProgressURL(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "courses.yaml")
	write(t, f, "stages: [{id: a, label: A}]\ncourses:\n- {id: on, name: On, url: https://on.example.com/, stage: a, progress: true}\n- {id: off, name: Off, url: https://off.example.com/, stage: a}\n")
	h, err := LoadHub(f)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range h.Courses {
		got := c.ProgressURL()
		if c.ID == "on" && got != "https://on.example.com/api/progress/summary" {
			t.Errorf("ProgressURL() = %q", got)
		}
		if c.ID == "off" && got != "" {
			t.Errorf("course without progress has URL %q", got)
		}
	}
	write(t, f, "stages: [{id: a, label: A}]\ncourses:\n- {id: x, name: X, url: http://x.example.com/, stage: a, progress: true}\n")
	if _, err := LoadHub(f); err == nil || !strings.Contains(err.Error(), "https://") {
		t.Errorf("want https error, got %v", err)
	}
}

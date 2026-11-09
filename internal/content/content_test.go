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
	write(t, filepath.Join(dir, "papers/a/index.md"), "---\nid: a\ntitle: A paper\n---\nBody\n")
	write(t, filepath.Join(dir, "papers/a/exercises/1-x/meta.json"), "{}")
	write(t, filepath.Join(dir, "paths/first.md"), "---\ntitle: First path\n---\nSteps\n")
	papers, err := LoadPapers(filepath.Join(dir, "papers"))
	if err != nil || len(papers) != 1 || papers[0].Exercises != 1 || papers[0].Body != "Body\n" {
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

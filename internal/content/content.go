// Package content loads the markdown sources and YAML data that the sites
// are built from.
package content

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Paper is one paper page, read from content/papers/<id>/index.md.
type Paper struct {
	ID         string   `yaml:"id"`
	Title      string   `yaml:"title"`
	Authors    string   `yaml:"authors"`
	Year       int      `yaml:"year"`
	Venue      string   `yaml:"venue"`
	Arxiv      string   `yaml:"arxiv"`
	Track      string   `yaml:"track"`
	Tier       string   `yaml:"tier"`
	Hours      float64  `yaml:"hours"`
	Status     string   `yaml:"status"`
	TLDR       string   `yaml:"tldr"`
	Prereqs    []Prereq `yaml:"prereqs"`
	Background []string `yaml:"background"`

	// Body is the markdown after the front matter.
	Body string `yaml:"-"`
	// Exercises counts the exercise folders under exercises/.
	Exercises int `yaml:"-"`
}

// Prereq is a direct prerequisite paper and the reason a reader needs it.
type Prereq struct {
	ID     string `yaml:"id"`
	Reason string `yaml:"reason"`
}

// IsStub reports whether the page is a placeholder rather than a full page.
func (p Paper) IsStub() bool { return p.Status == "" || p.Status == "stub" }

// SplitFrontMatter separates YAML front matter, fenced by "---" lines, from
// the markdown body.
func SplitFrontMatter(src []byte) (meta, body []byte, err error) {
	src = bytes.ReplaceAll(src, []byte("\r\n"), []byte("\n"))
	if !bytes.HasPrefix(src, []byte("---\n")) {
		return nil, src, nil
	}
	rest := src[4:]
	end := bytes.Index(rest, []byte("\n---\n"))
	if end < 0 {
		if bytes.HasSuffix(rest, []byte("\n---")) {
			return rest[:len(rest)-4], nil, nil
		}
		return nil, nil, fmt.Errorf("front matter has no closing ---")
	}
	return rest[:end], rest[end+5:], nil
}

// LoadPapers reads every content/papers/<id>/index.md under dir, sorted by ID.
func LoadPapers(dir string) ([]Paper, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var papers []Paper
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		path := filepath.Join(dir, e.Name(), "index.md")
		src, err := os.ReadFile(path)
		if os.IsNotExist(err) {
			continue
		} else if err != nil {
			return nil, err
		}
		meta, body, err := SplitFrontMatter(src)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		var p Paper
		if err := yaml.Unmarshal(meta, &p); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		if p.ID == "" {
			p.ID = e.Name()
		}
		if p.ID != e.Name() {
			return nil, fmt.Errorf("%s: id %q does not match folder %q", path, p.ID, e.Name())
		}
		if strings.TrimSpace(p.Title) == "" {
			return nil, fmt.Errorf("%s: title is empty", path)
		}
		p.Body = string(body)
		ex, _ := os.ReadDir(filepath.Join(dir, e.Name(), "exercises"))
		for _, x := range ex {
			if x.IsDir() {
				p.Exercises++
			}
		}
		papers = append(papers, p)
	}
	sort.Slice(papers, func(i, j int) bool { return papers[i].ID < papers[j].ID })
	return papers, nil
}

// Hub is the course catalogue in content/hub/courses.yaml.
type Hub struct {
	Title   string   `yaml:"title"`
	Lede    string   `yaml:"lede"`
	Stages  []Stage  `yaml:"stages"`
	Courses []Course `yaml:"courses"`
}

// Stage is one step of the learning roadmap strip.
type Stage struct {
	ID    string `yaml:"id"`
	Label string `yaml:"label"`
	Note  string `yaml:"note"`
}

// Course is one card on the hub.
type Course struct {
	ID       string   `yaml:"id"`
	Name     string   `yaml:"name"`
	Mark     string   `yaml:"mark"`
	URL      string   `yaml:"url"`
	Blurb    string   `yaml:"blurb"`
	Lessons  int      `yaml:"lessons"`
	Unit     string   `yaml:"unit"`
	Tracks   []string `yaml:"tracks"`
	Stage    string   `yaml:"stage"`
	Order    int      `yaml:"order"`
	Progress string   `yaml:"progress"`
	Repo     string   `yaml:"repo"`
	Glob     string   `yaml:"glob"`
	Paths    []string `yaml:"paths"`
}

// Path is a reading path, read from content/paths/<id>.md.
type Path struct {
	ID    string `yaml:"id"`
	Title string `yaml:"title"`
	Body  string `yaml:"-"`
}

// LoadPaths reads every content/paths/*.md. A missing directory means no
// paths yet.
func LoadPaths(dir string) ([]Path, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.md"))
	if err != nil {
		return nil, err
	}
	var paths []Path
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		meta, body, err := SplitFrontMatter(src)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		var p Path
		if err := yaml.Unmarshal(meta, &p); err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		if p.ID == "" {
			p.ID = strings.TrimSuffix(filepath.Base(f), ".md")
		}
		p.Body = string(body)
		paths = append(paths, p)
	}
	sort.Slice(paths, func(i, j int) bool { return paths[i].ID < paths[j].ID })
	return paths, nil
}

// LoadHub reads the hub catalogue and sorts courses by their suggested order.
func LoadHub(path string) (Hub, error) {
	var h Hub
	src, err := os.ReadFile(path)
	if err != nil {
		return h, err
	}
	if err := yaml.Unmarshal(src, &h); err != nil {
		return h, fmt.Errorf("%s: %w", path, err)
	}
	seen := map[string]bool{}
	stages := map[string]bool{}
	for _, st := range h.Stages {
		stages[st.ID] = true
	}
	for _, c := range h.Courses {
		if !stages[c.Stage] {
			return h, fmt.Errorf("%s: course %q has unknown stage %q", path, c.ID, c.Stage)
		}
		if c.ID == "" || c.Name == "" || c.URL == "" {
			return h, fmt.Errorf("%s: every course needs id, name, and url", path)
		}
		if seen[c.ID] {
			return h, fmt.Errorf("%s: duplicate course id %q", path, c.ID)
		}
		seen[c.ID] = true
	}
	sort.SliceStable(h.Courses, func(i, j int) bool { return h.Courses[i].Order < h.Courses[j].Order })
	return h, nil
}

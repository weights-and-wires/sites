// Package content loads the markdown sources and YAML data that the sites
// are built from.
package content

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Paper is one paper page, read from content/papers/<id>/index.md. The JSON
// schema for the front matter is docs/paper.schema.json.
type Paper struct {
	ID            string       `yaml:"id"`
	Title         string       `yaml:"title"`
	Authors       string       `yaml:"authors"`
	Year          int          `yaml:"year"`
	Venue         string       `yaml:"venue"`
	Arxiv         string       `yaml:"arxiv"`
	URL           string       `yaml:"url"`
	Track         string       `yaml:"track"`
	Tier          string       `yaml:"tier"`
	Prereqs       []Prereq     `yaml:"prereqs"`
	Background    []string     `yaml:"background"`
	RelatedCourse string       `yaml:"related_course"`
	CourseLinks   []CourseLink `yaml:"course_links"`
	Hours         float64      `yaml:"est_hours"`
	Summary       string       `yaml:"summary"`
	Status        string       `yaml:"status"`
	SelfCheck     []SelfCheck  `yaml:"self_check"`

	// Body is the markdown after the front matter.
	Body string `yaml:"-"`
	// Exercises are the folders under exercises/, in order.
	Exercises []Exercise `yaml:"-"`
	// Dir is the paper's folder.
	Dir string `yaml:"-"`
}

// Prereq is a direct prerequisite paper and the reason a reader needs it.
type Prereq struct {
	ID     string `yaml:"id"`
	Reason string `yaml:"reason"`
}

// CourseLink is a course lesson that teaches this paper in depth.
type CourseLink struct {
	Title string `yaml:"title"`
	URL   string `yaml:"url"`
}

// SelfCheck is one "can you answer this before you start" question. See names
// the paper ID or concept ID to read when the reader gets it wrong.
type SelfCheck struct {
	Q   string `yaml:"q"`
	A   string `yaml:"a"`
	See string `yaml:"see"`
}

// Exercise is one folder under content/papers/<id>/exercises/.
type Exercise struct {
	// Slug is the folder name, for example "1-softmax".
	Slug string `json:"-"`
	// N is the number before the first dash.
	N        int      `json:"-"`
	Title    string   `json:"title"`
	Goal     string   `json:"goal"`
	Hints    []string `json:"hints"`
	Runtime  string   `json:"runtime"` // "pyodide" (default) or "torch"
	Lab      string   `json:"lab"`     // optional course lab URL for torch exercises
	Starter  string   `json:"-"`
	Tests    string   `json:"-"`
	Solution string   `json:"-"`
}

// Valid tracks and tiers.
var (
	Tracks = []string{"foundations", "architectures", "training-optimization", "generative", "nlp-llm", "vision", "rl", "speech", "gpu-systems", "inference", "scaling-efficiency"}
	Tiers  = []string{"core", "extended"}
)

// IsStub reports whether the page is a placeholder rather than a full page.
func (p Paper) IsStub() bool { return p.Status == "" || p.Status == "stub" }

// Link is the paper's arXiv abstract page, or its official URL.
func (p Paper) Link() string {
	if p.Arxiv != "" {
		return "https://arxiv.org/abs/" + p.Arxiv
	}
	return p.URL
}

// Validate checks the fields every paper needs.
func (p Paper) Validate() error {
	var problems []string
	if !contains(Tracks, p.Track) {
		problems = append(problems, fmt.Sprintf("unknown track %q", p.Track))
	}
	if !contains(Tiers, p.Tier) {
		problems = append(problems, fmt.Sprintf("unknown tier %q", p.Tier))
	}
	if p.Status != "stub" && p.Status != "full" {
		problems = append(problems, fmt.Sprintf("status must be stub or full, got %q", p.Status))
	}
	if p.Year == 0 || p.Authors == "" || p.Summary == "" {
		problems = append(problems, "year, authors, and summary are required")
	}
	if p.Arxiv == "" && p.URL == "" {
		problems = append(problems, "set arxiv or url")
	}
	if p.Status == "full" {
		if len(p.SelfCheck) < 1 {
			problems = append(problems, "a full page needs self_check questions")
		}
		for _, r := range p.Prereqs {
			if strings.TrimSpace(r.Reason) == "" {
				problems = append(problems, fmt.Sprintf("prerequisite %q needs a reason", r.ID))
			}
		}
	}
	if len(problems) > 0 {
		return fmt.Errorf("%s", strings.Join(problems, "; "))
	}
	return nil
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

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
		dec := yaml.NewDecoder(bytes.NewReader(meta))
		dec.KnownFields(true)
		if err := dec.Decode(&p); err != nil {
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
		if err := p.Validate(); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		p.Body = string(body)
		p.Dir = filepath.Join(dir, e.Name())
		if p.Exercises, err = LoadExercises(filepath.Join(p.Dir, "exercises")); err != nil {
			return nil, fmt.Errorf("%s: %w", p.ID, err)
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
	Progress bool     `yaml:"progress"`
	Repo     string   `yaml:"repo"`
	Glob     string   `yaml:"glob"`
	Paths    []string `yaml:"paths"`
}

// ProgressURL returns the course's progress summary endpoint, or "" when the
// course hasn't opted in with `progress: true`.
func (c Course) ProgressURL() string {
	if !c.Progress {
		return ""
	}
	return strings.TrimRight(c.URL, "/") + ProgressPath
}

// ProgressPath is the path of the progress summary endpoint on a course host.
const ProgressPath = "/api/progress/summary"

// Path is a reading path, read from content/paths/<id>.md.
type Path struct {
	ID      string   `yaml:"id"`
	Title   string   `yaml:"title"`
	Assumes string   `yaml:"assumes"`
	Papers  []string `yaml:"papers"`
	Body    string   `yaml:"-"`
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
		if c.Progress && !strings.HasPrefix(c.URL, "https://") {
			return h, fmt.Errorf("%s: course %q sets progress, so its url must start with https://", path, c.ID)
		}
		seen[c.ID] = true
	}
	sort.SliceStable(h.Courses, func(i, j int) bool { return h.Courses[i].Order < h.Courses[j].Order })
	return h, nil
}

// LoadExercises reads every exercise folder under dir. A folder is named
// <n>-<slug> and holds meta.json, starter.py, tests.py, and solution.py.
func LoadExercises(dir string) ([]Exercise, error) {
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var out []Exercise
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		num, _, ok := strings.Cut(e.Name(), "-")
		n, err := strconv.Atoi(num)
		if !ok || err != nil {
			return nil, fmt.Errorf("exercise folder %q must be named <n>-<slug>", e.Name())
		}
		x := Exercise{Slug: e.Name(), N: n}
		read := func(name string) (string, error) {
			b, err := os.ReadFile(filepath.Join(dir, e.Name(), name))
			return string(b), err
		}
		meta, err := read("meta.json")
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(meta), &x); err != nil {
			return nil, fmt.Errorf("%s/meta.json: %w", e.Name(), err)
		}
		if x.Runtime == "" {
			x.Runtime = "pyodide"
		}
		if x.Runtime != "pyodide" && x.Runtime != "torch" {
			return nil, fmt.Errorf("%s/meta.json: runtime must be pyodide or torch", e.Name())
		}
		if x.Title == "" || x.Goal == "" {
			return nil, fmt.Errorf("%s/meta.json: title and goal are required", e.Name())
		}
		for name, dst := range map[string]*string{"starter.py": &x.Starter, "tests.py": &x.Tests, "solution.py": &x.Solution} {
			if *dst, err = read(name); err != nil {
				return nil, err
			}
		}
		out = append(out, x)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].N < out[j].N })
	for i := 1; i < len(out); i++ {
		if out[i].N == out[i-1].N {
			return nil, fmt.Errorf("two exercises numbered %d", out[i].N)
		}
	}
	return out, nil
}

// Concept is a background idea that a course lesson teaches.
type Concept struct {
	Label     string `yaml:"label"`
	URL       string `yaml:"url"`
	Lesson    string `yaml:"lesson"`
	Refresher string `yaml:"refresher"`
	Match     string `yaml:"anchor_match"`
}

// LoadConcepts reads content/papers/concepts.yaml.
func LoadConcepts(path string) (map[string]Concept, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Concepts map[string]Concept `yaml:"concepts"`
	}
	if err := yaml.Unmarshal(src, &doc); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	for id, c := range doc.Concepts {
		if c.Label == "" || !strings.HasPrefix(c.URL, "https://") || c.Refresher == "" {
			return nil, fmt.Errorf("%s: concept %q needs label, an https url, and refresher", path, id)
		}
	}
	return doc.Concepts, nil
}

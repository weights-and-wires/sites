package main

import (
	"archive/zip"
	"bytes"
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/weights-and-wires/sites/internal/content"
	"github.com/weights-and-wires/sites/internal/graph"
	"github.com/weights-and-wires/sites/internal/render"
)

// exercisesMarker is where a paper's body places the exercises. Without it,
// the exercises follow the body.
const exercisesMarker = "<!-- exercises -->"

// mdQueue collects markdown snippets so that every formula on the site goes
// to KaTeX in one batch. Add snippets, call Render once, then read HTML.
type mdQueue struct {
	src      []string
	where    []string
	formulas []render.Math
	html     []string
	heads    [][]render.Heading
}

func (q *mdQueue) add(src, where string) int {
	q.src = append(q.src, render.ExtractMath(src, where, &q.formulas))
	q.where = append(q.where, where)
	return len(q.src) - 1
}

func (q *mdQueue) render(katexScript string) error {
	rendered, err := render.KaTeX(katexScript, q.formulas)
	if err != nil {
		return err
	}
	for i, s := range q.src {
		body, heads, err := render.Markdown(s)
		if err != nil {
			return fmt.Errorf("%s: %w", q.where[i], err)
		}
		q.html = append(q.html, render.RestoreMath(body, rendered))
		q.heads = append(q.heads, heads)
	}
	return nil
}

var singlePara = regexp.MustCompile(`(?s)^<p>(.*)</p>\s*$`)

// inline returns snippet i without a single wrapping <p>.
func (q *mdQueue) inline(i int) template.HTML {
	h := strings.TrimSpace(q.html[i])
	if m := singlePara.FindStringSubmatch(h); m != nil && !strings.Contains(m[1], "<p>") {
		h = m[1]
	}
	return template.HTML(h)
}

func (q *mdQueue) block(i int) template.HTML { return template.HTML(q.html[i]) }

// PaperView is everything paper.html needs.
type PaperView struct {
	content.Paper
	Link       string
	Prereqs    []PaperLink
	Chain      []PaperLink
	Concepts   []ConceptView
	SelfCheck  []SelfCheckView
	Next       []PaperLink
	Exercises  []ExerciseView
	BodyBefore template.HTML
	BodyAfter  template.HTML
	Further    template.HTML
	Headings   []render.Heading
	Text       string
}

// ConceptView is one "Concepts you need" row.
type ConceptView struct {
	ID string
	content.Concept
}

// SelfCheckView is one self-check question with its answer.
type SelfCheckView struct {
	Q, A    template.HTML
	SeeURL  string
	SeeText string
}

// ExerciseView is one exercise block.
type ExerciseView struct {
	content.Exercise
	ID       string
	GoalHTML template.HTML
	Hints    []template.HTML
	Zip      string
	Torch    bool
}

type paperJobs struct {
	body, after, goal int
	hints             []int
	checks            [][2]int
	exGoal            [][]int // per exercise: goal, hints...
}

// papersSite renders every paper page and returns the search entries and the
// rendered views.
func (b *builder) paperViews(papers []content.Paper, g *graph.Graph, concepts map[string]content.Concept) ([]PaperView, error) {
	byID := map[string]content.Paper{}
	for _, p := range papers {
		byID[p.ID] = p
	}
	var problems []string
	for _, p := range papers {
		for _, c := range p.Background {
			if _, ok := concepts[c]; !ok {
				problems = append(problems, fmt.Sprintf("%s: background concept %q is not in concepts.yaml", p.ID, c))
			}
		}
		for _, sc := range p.SelfCheck {
			if _, ok := byID[sc.See]; !ok && sc.See != "" {
				if _, ok := concepts[sc.See]; !ok {
					problems = append(problems, fmt.Sprintf("%s: self_check see %q is neither a paper nor a concept", p.ID, sc.See))
				}
			}
		}
	}
	if len(problems) > 0 {
		return nil, fmt.Errorf("papers:\n  %s", strings.Join(problems, "\n  "))
	}

	q := &mdQueue{}
	jobs := make([]paperJobs, len(papers))
	for i, p := range papers {
		where := "papers/" + p.ID
		before, after, _ := strings.Cut(p.Body, exercisesMarker)
		jobs[i].body = q.add(before, where)
		jobs[i].after = q.add(after, where)
		for _, sc := range p.SelfCheck {
			jobs[i].checks = append(jobs[i].checks, [2]int{q.add(sc.Q, where+" self_check"), q.add(sc.A, where+" self_check")})
		}
		for _, x := range p.Exercises {
			ids := []int{q.add(x.Goal, where+"/"+x.Slug)}
			for _, h := range x.Hints {
				ids = append(ids, q.add(h, where+"/"+x.Slug+" hint"))
			}
			jobs[i].exGoal = append(jobs[i].exGoal, ids)
		}
	}
	if err := q.render(b.katex); err != nil {
		return nil, err
	}
	b.math = len(q.formulas) > 0

	link := func(id, reason string) PaperLink {
		p := byID[id]
		return PaperLink{ID: id, Title: p.Title, Reason: reason, Year: p.Year, Track: p.Track, Stub: p.IsStub()}
	}
	views := make([]PaperView, len(papers))
	for i, p := range papers {
		v := PaperView{Paper: p, Link: p.Link()}
		v.BodyBefore = q.block(jobs[i].body)
		after := q.html[jobs[i].after]
		if k := strings.Index(after, `<h2 id="further-reading"`); k >= 0 {
			v.Further = template.HTML(after[k:])
			after = after[:k]
		}
		v.BodyAfter = template.HTML(after)
		for _, r := range p.Prereqs {
			v.Prereqs = append(v.Prereqs, link(r.ID, r.Reason))
		}
		for _, id := range g.Chain(p.ID) {
			v.Chain = append(v.Chain, link(id, ""))
		}
		for _, c := range p.Background {
			v.Concepts = append(v.Concepts, ConceptView{ID: c, Concept: concepts[c]})
		}
		for k, sc := range p.SelfCheck {
			sv := SelfCheckView{Q: q.inline(jobs[i].checks[k][0]), A: q.inline(jobs[i].checks[k][1])}
			if pp, ok := byID[sc.See]; ok {
				sv.SeeURL, sv.SeeText = "/"+pp.ID+"/", pp.Title
			} else if c, ok := concepts[sc.See]; ok {
				sv.SeeURL, sv.SeeText = c.URL, c.Lesson
			}
			v.SelfCheck = append(v.SelfCheck, sv)
		}
		for _, n := range g.Next(p.ID) {
			reason := ""
			for _, r := range byID[n].Prereqs {
				if r.ID == p.ID {
					reason = r.Reason
				}
			}
			v.Next = append(v.Next, link(n, reason))
		}
		for k, x := range p.Exercises {
			ids := jobs[i].exGoal[k]
			ev := ExerciseView{Exercise: x, ID: fmt.Sprintf("exercise-%d", x.N), GoalHTML: q.block(ids[0]), Torch: x.Runtime == "torch"}
			for _, h := range ids[1:] {
				ev.Hints = append(ev.Hints, q.inline(h))
			}
			ev.Zip = fmt.Sprintf("/%s/exercises/%d.zip", p.ID, x.N)
			v.Exercises = append(v.Exercises, ev)
		}
		v.Headings = paperHeadings(v, q.heads[jobs[i].body], q.heads[jobs[i].after])
		v.Text = render.PlainText(q.html[jobs[i].body] + " " + q.html[jobs[i].after])
		views[i] = v
	}
	return views, nil
}

// headings returns the table of contents for a paper page: the fixed panels
// plus the body's own h2 and h3 headings, in page order.
func paperHeadings(v PaperView, beforeHeads, afterHeads []render.Heading) []render.Heading {
	heads := []render.Heading{{Level: 2, ID: "before-you-read", Text: "Before you read"}}
	heads = append(heads, beforeHeads...)
	if len(v.Exercises) > 0 {
		heads = append(heads, render.Heading{Level: 2, ID: "implement-it", Text: "Implement it"})
	}
	var further []render.Heading
	for _, h := range afterHeads {
		if h.ID == "further-reading" || len(further) > 0 {
			further = append(further, h)
		} else {
			heads = append(heads, h)
		}
	}
	if len(v.Next) > 0 {
		heads = append(heads, render.Heading{Level: 2, ID: "what-came-next", Text: "What came next"})
	}
	if len(v.CourseLinks) > 0 {
		heads = append(heads, render.Heading{Level: 2, ID: "course-links", Text: "Course links"})
	}
	return append(heads, further...)
}

// exerciseZip packs one exercise for local runs: the starter as
// submission.py, the tests, and a README. Entries carry a fixed time so the
// archive is the same on every build.
func exerciseZip(p content.Paper, x content.Exercise) ([]byte, error) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	when := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	runtime := "NumPy"
	install := "python -m pip install numpy pytest"
	if x.Runtime == "torch" {
		runtime = "PyTorch (CPU is enough)"
		install = "python -m pip install numpy pytest torch --index-url https://download.pytorch.org/whl/cpu"
	}
	readme := fmt.Sprintf(`%s: exercise %d, %s

From https://papers.ifkash.dev/%s/#exercise-%d

Edit submission.py until every test passes. This exercise needs %s.

    %s
    python -m pytest tests.py

The tests start by failing. That is expected.
`, p.Title, x.N, x.Title, p.ID, x.N, runtime, install)
	dir := fmt.Sprintf("%s-exercise-%d/", p.ID, x.N)
	for _, f := range []struct{ name, body string }{
		{"README.txt", readme},
		{"submission.py", x.Starter},
		{"tests.py", x.Tests},
	} {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: dir + f.name, Method: zip.Deflate, Modified: when})
		if err != nil {
			return nil, err
		}
		if _, err := w.Write([]byte(f.body)); err != nil {
			return nil, err
		}
	}
	if err := zw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// copyPyodide copies the vendored Pyodide (scripts/fetch-pyodide.sh) into
// public/papers/pyodide/. In a release build it must exist.
func (b *builder) copyPyodide(require bool) error {
	matches, _ := filepath.Glob(filepath.Join(b.root, ".cache/pyodide/*/dist"))
	sort.Strings(matches)
	if len(matches) == 0 {
		if require {
			return fmt.Errorf("pyodide is missing: run scripts/fetch-pyodide.sh")
		}
		fmt.Fprintln(os.Stderr, "build: warning: no Pyodide in .cache/pyodide; Run tests won't work (run scripts/fetch-pyodide.sh)")
		return nil
	}
	src := matches[len(matches)-1]
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, e := range entries {
		data, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			return err
		}
		if err := b.write("papers", "pyodide/"+e.Name(), data); err != nil {
			return err
		}
	}
	return nil
}

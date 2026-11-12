package main

import (
	"fmt"
	"html"
	"html/template"
	"regexp"
	"sort"
	"strings"

	"github.com/weights-and-wires/sites/internal/content"
	"github.com/weights-and-wires/sites/internal/render"
	"github.com/weights-and-wires/sites/internal/search"
)

// DiveView is everything dive.html needs.
type DiveView struct {
	content.Dive
	Prereqs  []PaperLink
	Concepts []ConceptView
	Body     template.HTML
	Further  template.HTML
	HasCells bool
	Headings []render.Heading
	// JS and CSS are the fingerprinted names of dive.js and dive.css, or "".
	JS, CSS string
}

// cellMarker places an editable Python cell: <!-- cell: name -->.
var cellMarker = regexp.MustCompile(`<!--\s*cell:\s*([A-Za-z0-9_-]+)\s*-->`)

// diveViews renders every dive body, resolves its prerequisites, and swaps
// each cell marker for an editable cell.
func (b *builder) diveViews(dives []content.Dive, papers []content.Paper, concepts map[string]content.Concept) ([]DiveView, error) {
	byID := map[string]content.Paper{}
	for _, p := range papers {
		byID[p.ID] = p
	}
	var problems []string
	for _, d := range dives {
		for _, r := range d.Prereqs {
			if _, ok := byID[r.ID]; !ok {
				problems = append(problems, fmt.Sprintf("%s: prerequisite %q is not a paper", d.ID, r.ID))
			}
		}
		for _, c := range d.Background {
			if _, ok := concepts[c]; !ok {
				problems = append(problems, fmt.Sprintf("%s: background concept %q is not in concepts.yaml", d.ID, c))
			}
		}
	}
	if len(problems) > 0 {
		return nil, fmt.Errorf("dives:\n  %s", strings.Join(problems, "\n  "))
	}

	q := &mdQueue{}
	for _, d := range dives {
		q.add(d.Body, "dives/"+d.ID)
	}
	if err := q.render(b.katex); err != nil {
		return nil, err
	}
	if len(q.formulas) > 0 {
		b.math = true
	}

	views := make([]DiveView, len(dives))
	for i, d := range dives {
		v := DiveView{Dive: d}
		body := q.html[i]
		used := map[string]bool{}
		var missing []string
		body = cellMarker.ReplaceAllStringFunc(body, func(m string) string {
			name := cellMarker.FindStringSubmatch(m)[1]
			src, ok := d.Cells[name]
			if !ok {
				missing = append(missing, name)
				return m
			}
			used[name] = true
			return cellHTML(d.ID, name, src)
		})
		if len(missing) > 0 {
			return nil, fmt.Errorf("dives/%s: no cells/%s.py for the cell markers", d.ID, strings.Join(missing, ".py, cells/"))
		}
		for name := range d.Cells {
			if !used[name] {
				return nil, fmt.Errorf("dives/%s: cells/%s.py has no <!-- cell: %s --> marker", d.ID, name, name)
			}
		}
		v.HasCells = len(used) > 0
		if len(d.Prereqs)+len(d.Background)+len(d.CourseLinks) > 0 {
			v.Headings = append(v.Headings, render.Heading{Level: 2, ID: "before-you-read", Text: "Before you read"})
		}
		v.Headings = append(v.Headings, q.heads[i]...)
		if k := strings.Index(body, `<h2 id="further-reading"`); k >= 0 {
			v.Further = template.HTML(body[k:])
			body = body[:k]
		}
		v.Body = template.HTML(body)
		for _, r := range d.Prereqs {
			p := byID[r.ID]
			v.Prereqs = append(v.Prereqs, PaperLink{ID: r.ID, Title: p.Title, Reason: r.Reason, Year: p.Year, Track: p.Track, Stub: p.IsStub()})
		}
		for _, c := range d.Background {
			v.Concepts = append(v.Concepts, ConceptView{ID: c, Concept: concepts[c]})
		}
		views[i] = v
	}
	return views, nil
}

// cellHTML is one editable cell. static/papers.js mounts the editor and runs
// the code in the Pyodide worker.
func cellHTML(dive, name, src string) string {
	code := html.EscapeString(src)
	return fmt.Sprintf(`<div class="py-cell" data-cell="%[1]s/%[2]s">
<template data-code>%[3]s</template>
<div class="code-block editor-block">
<div class="code-toolbar"><span>%[2]s.py</span><button type="button" data-run class="run-button">Run</button><button type="button" data-reset>Reset</button></div>
<div class="editor-host" data-editor><textarea class="editor-fallback" spellcheck="false" aria-label="Python cell %[2]s">%[3]s</textarea></div>
</div>
<div class="runner-output" data-output aria-live="polite" hidden></div>
</div>`, dive, name, code)
}

// writeDives renders /dives/ and /dives/<id>/ with each dive's own script and
// styles, and returns search entries for them.
func (b *builder) writeDives(base Page, views []DiveView) ([]search.Entry, error) {
	var index []search.Entry
	for i := range views {
		v := &views[i]
		if v.Dive.JS != nil {
			v.JS = hashed("dive-"+v.ID, ".js", v.Dive.JS)
			if err := b.write("papers", "assets/"+v.JS, v.Dive.JS); err != nil {
				return nil, err
			}
		}
		if v.Dive.CSS != nil {
			v.CSS = hashed("dive-"+v.ID, ".css", v.Dive.CSS)
			if err := b.write("papers", "assets/"+v.CSS, v.Dive.CSS); err != nil {
				return nil, err
			}
		}
		page := base
		page.Title = v.Title + " · Deep dives"
		page.Description = v.Summary
		page.Dive = v
		page.Headings = v.Headings
		page.Math = strings.Contains(string(v.Body)+string(v.Further), `class="katex`)
		if err := b.page("papers", "dives/"+v.ID+"/index.html", "dive.html", page); err != nil {
			return nil, err
		}
		index = append(index, search.NewEntry("Deep dive: "+v.Title, "/dives/"+v.ID+"/", page.Headings, v.Summary+" "+render.PlainText(string(v.Body))))
	}
	list := append([]DiveView(nil), views...)
	sort.SliceStable(list, func(i, j int) bool { return list[i].Source.Date > list[j].Source.Date })
	page := base
	page.Title, page.Description, page.Extra = "Deep dives · Papers", "Interactive walkthroughs of technical blog posts, rebuilt from first principles.", list
	if err := b.page("papers", "dives/index.html", "dives.html", page); err != nil {
		return nil, err
	}
	index = append(index, search.NewEntry("Deep dives", "/dives/", nil, page.Description))
	return index, nil
}

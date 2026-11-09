package main

import (
	"fmt"
	"html/template"
	"sort"
	"strings"

	"github.com/weights-and-wires/sites/internal/content"
	"github.com/weights-and-wires/sites/internal/graph"
	"github.com/weights-and-wires/sites/internal/render"
	"github.com/weights-and-wires/sites/internal/search"
)

// trackNames are the display names of the tracks, in display order.
var trackNames = map[string]string{
	"foundations":           "Foundations",
	"architectures":         "Architectures",
	"training-optimization": "Training and optimization",
	"generative":            "Generative models",
	"nlp-llm":               "Language models",
	"vision":                "Vision",
	"rl":                    "Reinforcement learning",
	"speech":                "Speech",
	"gpu-systems":           "GPU systems",
	"inference":             "Inference",
	"scaling-efficiency":    "Scaling and efficiency",
}

// TrackView is one track: its papers in reading order.
type TrackView struct {
	ID, Name string
	Papers   []PaperLink
	Full     int
}

// PathView is one reading path.
type PathView struct {
	content.Path
	Intro  template.HTML
	Papers []PaperLink
}

// GraphView is the /graph/ page.
type GraphView struct {
	Layout graph.Layout
	Nodes  []GraphNode
	Tracks []TrackView
}

// GraphNode is one node of the drawing with what the template needs.
type GraphNode struct {
	graph.Node
	Title, Track, Tier string
	Stub               bool
	Needs              string // space-separated prerequisite IDs
	LabelX, LabelY     float64
}

// HomeView is the papers home page.
type HomeView struct {
	Paths  []PathView
	Tracks []TrackView
	Total  int
	Full   int
}

// siteViews builds the track, path, graph, and home views. Papers within a
// track are in reading order: by layer, then by year.
func siteViews(papers []content.Paper, paths []content.Path, g *graph.Graph) (HomeView, []TrackView, []PathView, GraphView, error) {
	byID := map[string]content.Paper{}
	for _, p := range papers {
		byID[p.ID] = p
	}
	layer := g.Layer()
	link := func(id string) PaperLink {
		p := byID[id]
		return PaperLink{ID: id, Title: p.Title, Reason: p.Summary, Year: p.Year, Track: p.Track, Stub: p.IsStub()}
	}
	ordered := append([]content.Paper(nil), papers...)
	sort.SliceStable(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		if layer[a.ID] != layer[b.ID] {
			return layer[a.ID] < layer[b.ID]
		}
		if a.Year != b.Year {
			return a.Year < b.Year
		}
		return a.ID < b.ID
	})
	var tracks []TrackView
	home := HomeView{Total: len(papers)}
	for _, t := range content.Tracks {
		tv := TrackView{ID: t, Name: trackNames[t]}
		for _, p := range ordered {
			if p.Track == t {
				tv.Papers = append(tv.Papers, link(p.ID))
				if !p.IsStub() {
					tv.Full++
				}
			}
		}
		home.Full += tv.Full
		tracks = append(tracks, tv)
	}

	var pathViews []PathView
	q := &mdQueue{}
	for _, p := range paths {
		pv := PathView{Path: p}
		for _, id := range p.Papers {
			if _, ok := byID[id]; !ok {
				return home, nil, nil, GraphView{}, fmt.Errorf("path %s: unknown paper %q", p.ID, id)
			}
			pv.Papers = append(pv.Papers, link(id))
		}
		pathViews = append(pathViews, pv)
		q.add(p.Body, "paths/"+p.ID)
	}
	if err := q.render(""); err != nil {
		return home, nil, nil, GraphView{}, err
	}
	for i := range pathViews {
		pathViews[i].Intro = q.block(i)
	}
	home.Paths, home.Tracks = pathViews, tracks

	lay := g.Layout(152, 34, 128, 24)
	gv := GraphView{Layout: lay, Tracks: tracks}
	for _, n := range lay.Nodes {
		p := byID[n.ID]
		gv.Nodes = append(gv.Nodes, GraphNode{Node: n, Title: p.Title, Track: p.Track, Tier: p.Tier, Stub: p.IsStub(),
			Needs: strings.Join(g.Needs(n.ID), " "), LabelX: n.X + 8, LabelY: n.Y + 16})
	}
	sort.SliceStable(gv.Nodes, func(i, j int) bool {
		if gv.Nodes[i].Layer != gv.Nodes[j].Layer {
			return gv.Nodes[i].Layer < gv.Nodes[j].Layer
		}
		return gv.Nodes[i].Row < gv.Nodes[j].Row
	})
	return home, tracks, pathViews, gv, nil
}

// writeSitePages renders /, /graph/, /paths/<id>/, /tracks/<track>/, and the
// index pages for paths and tracks. It returns search entries for them.
func (b *builder) writeSitePages(base Page, papers []content.Paper, paths []content.Path, g *graph.Graph) ([]search.Entry, error) {
	home, tracks, pathViews, gv, err := siteViews(papers, paths, g)
	if err != nil {
		return nil, err
	}
	var index []search.Entry
	pages := []struct {
		path, tmpl, title, desc string
		extra                   any
	}{
		{"index.html", "papers_home.html", "Papers · ifkash.dev", "Paper pages with prerequisites and in-browser exercises.", home},
		{"graph/index.html", "graph.html", "Prerequisite graph · Papers", "Every paper and what it builds on, as a layered graph.", gv},
		{"paths/index.html", "paths.html", "Reading paths · Papers", "Ordered reading lists from first principles to current models.", pathViews},
		{"tracks/index.html", "tracks.html", "Tracks · Papers", "Papers grouped by topic.", tracks},
	}
	for _, p := range pathViews {
		pages = append(pages, struct {
			path, tmpl, title, desc string
			extra                   any
		}{"paths/" + p.ID + "/index.html", "path.html", p.Title + " · Reading path", render.PlainText(string(p.Intro)), p})
		index = append(index, search.NewEntry("Reading path: "+p.Title, "/paths/"+p.ID+"/", nil, render.PlainText(string(p.Intro))))
	}
	for _, t := range tracks {
		pages = append(pages, struct {
			path, tmpl, title, desc string
			extra                   any
		}{"tracks/" + t.ID + "/index.html", "track.html", t.Name + " · Papers", fmt.Sprintf("%d papers on %s.", len(t.Papers), strings.ToLower(t.Name)), t})
		index = append(index, search.NewEntry("Track: "+t.Name, "/tracks/"+t.ID+"/", nil, fmt.Sprintf("%d papers", len(t.Papers))))
	}
	for _, p := range pages {
		page := base
		page.Title, page.Description, page.Extra = p.title, p.desc, p.extra
		if err := b.page("papers", p.path, p.tmpl, page); err != nil {
			return nil, err
		}
	}
	return index, nil
}

// trackName is a template helper.
func trackName(id string) string {
	if n, ok := trackNames[id]; ok {
		return n
	}
	return id
}

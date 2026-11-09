// Package graph checks the paper prerequisite graph and answers ordering
// questions about it.
package graph

import (
	"fmt"
	"sort"
	"strings"

	"github.com/weights-and-wires/sites/internal/content"
)

// Graph is the prerequisite DAG. An edge runs from a prerequisite to the
// paper that needs it.
type Graph struct {
	papers map[string]content.Paper
	needs  map[string][]string // paper -> direct prerequisites
	next   map[string][]string // paper -> papers that list it as a prerequisite
}

// Build checks every prerequisite ID and rejects cycles. It also rejects a
// full (non-stub) paper with no exercises.
func Build(papers []content.Paper) (*Graph, error) {
	g := &Graph{papers: map[string]content.Paper{}, needs: map[string][]string{}, next: map[string][]string{}}
	for _, p := range papers {
		g.papers[p.ID] = p
	}
	var problems []string
	for _, p := range papers {
		for _, r := range p.Prereqs {
			if _, ok := g.papers[r.ID]; !ok {
				problems = append(problems, fmt.Sprintf("%s: unknown prerequisite %q", p.ID, r.ID))
				continue
			}
			g.needs[p.ID] = append(g.needs[p.ID], r.ID)
			g.next[r.ID] = append(g.next[r.ID], p.ID)
		}
		if !p.IsStub() && len(p.Exercises) == 0 {
			problems = append(problems, fmt.Sprintf("%s: a full paper page needs at least one exercise", p.ID))
		}
	}
	for _, list := range g.next {
		sort.Strings(list)
	}
	if len(problems) > 0 {
		return nil, fmt.Errorf("paper graph:\n  %s", strings.Join(problems, "\n  "))
	}
	if cycle := g.findCycle(); cycle != nil {
		return nil, fmt.Errorf("paper graph: prerequisite cycle %s", strings.Join(cycle, " -> "))
	}
	return g, nil
}

func (g *Graph) findCycle() []string {
	const (
		white = iota
		grey
		black
	)
	color := map[string]int{}
	var stack []string
	var found []string
	var visit func(string) bool
	visit = func(id string) bool {
		color[id] = grey
		stack = append(stack, id)
		for _, r := range g.needs[id] {
			if color[r] == grey {
				for i, s := range stack {
					if s == r {
						found = append(append([]string{}, stack[i:]...), r)
					}
				}
				return true
			}
			if color[r] == white && visit(r) {
				return true
			}
		}
		stack = stack[:len(stack)-1]
		color[id] = black
		return false
	}
	ids := make([]string, 0, len(g.papers))
	for id := range g.papers {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if color[id] == white && visit(id) {
			return found
		}
	}
	return nil
}

// Next returns the papers that list id as a direct prerequisite.
func (g *Graph) Next(id string) []string { return g.next[id] }

// Chain returns every transitive prerequisite of id, prerequisites first.
func (g *Graph) Chain(id string) []string {
	seen := map[string]bool{}
	var order []string
	var walk func(string)
	walk = func(cur string) {
		for _, r := range g.needs[cur] {
			if !seen[r] {
				seen[r] = true
				walk(r)
				order = append(order, r)
			}
		}
	}
	walk(id)
	return order
}

// Layer returns each paper's layer: 0 for papers with no prerequisites, and
// otherwise one more than its deepest prerequisite.
func (g *Graph) Layer() map[string]int {
	layer := map[string]int{}
	var walk func(string) int
	walk = func(id string) int {
		if l, ok := layer[id]; ok {
			return l
		}
		l := 0
		for _, r := range g.needs[id] {
			if d := walk(r) + 1; d > l {
				l = d
			}
		}
		layer[id] = l
		return l
	}
	for id := range g.papers {
		walk(id)
	}
	return layer
}

// Needs returns the direct prerequisites of id.
func (g *Graph) Needs(id string) []string { return g.needs[id] }

// Node is a paper placed on the layered drawing.
type Node struct {
	ID    string
	Layer int
	Row   int
	X, Y  float64
}

// Edge joins a prerequisite (From) to the paper that needs it (To).
type Edge struct {
	From, To string
	Path     string // SVG path data
}

// Layout is a layered (Sugiyama-style) drawing of the graph: layers are
// columns from left to right, and rows inside a column are ordered to reduce
// edge crossings.
type Layout struct {
	Nodes         []Node
	Edges         []Edge
	Width, Height float64
	NodeW, NodeH  float64
}

// Layout places every paper. colW and rowH set the column and row pitch.
func (g *Graph) Layout(colW, rowH, nodeW, nodeH float64) Layout {
	layer := g.Layer()
	maxLayer := 0
	cols := map[int][]string{}
	ids := make([]string, 0, len(layer))
	for id := range layer {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		l := layer[id]
		cols[l] = append(cols[l], id)
		if l > maxLayer {
			maxLayer = l
		}
	}
	pos := map[string]float64{}
	setPos := func(l int) {
		for i, id := range cols[l] {
			pos[id] = float64(i)
		}
	}
	for l := 0; l <= maxLayer; l++ {
		setPos(l)
	}
	// Barycenter sweeps: order each column by the mean position of its
	// neighbours in the previous column (down), then the next (up).
	bary := func(l int, neighbours func(string) []string) {
		col := cols[l]
		key := map[string]float64{}
		for _, id := range col {
			ns := neighbours(id)
			if len(ns) == 0 {
				key[id] = pos[id]
				continue
			}
			sum := 0.0
			for _, n := range ns {
				sum += pos[n]
			}
			key[id] = sum / float64(len(ns))
		}
		sort.SliceStable(col, func(i, j int) bool { return key[col[i]] < key[col[j]] })
		setPos(l)
	}
	for sweep := 0; sweep < 8; sweep++ {
		for l := 1; l <= maxLayer; l++ {
			bary(l, func(id string) []string { return g.needs[id] })
		}
		for l := maxLayer - 1; l >= 0; l-- {
			bary(l, func(id string) []string { return g.next[id] })
		}
	}
	tallest := 0
	for _, col := range cols {
		if len(col) > tallest {
			tallest = len(col)
		}
	}
	const pad = 16.0
	lay := Layout{NodeW: nodeW, NodeH: nodeH}
	lay.Width = pad*2 + float64(maxLayer)*colW + nodeW
	lay.Height = pad*2 + float64(tallest-1)*rowH + nodeH
	at := map[string]Node{}
	for l := 0; l <= maxLayer; l++ {
		col := cols[l]
		offset := float64(tallest-len(col)) * rowH / 2
		for i, id := range col {
			n := Node{ID: id, Layer: l, Row: i, X: pad + float64(l)*colW, Y: pad + offset + float64(i)*rowH}
			at[id] = n
			lay.Nodes = append(lay.Nodes, n)
		}
	}
	for _, n := range lay.Nodes {
		for _, r := range g.needs[n.ID] {
			a := at[r]
			x1, y1 := a.X+nodeW, a.Y+nodeH/2
			x2, y2 := n.X, n.Y+nodeH/2
			dx := (x2 - x1) / 2
			lay.Edges = append(lay.Edges, Edge{From: r, To: n.ID,
				Path: fmt.Sprintf("M%.1f %.1fC%.1f %.1f %.1f %.1f %.1f %.1f", x1, y1, x1+dx, y1, x2-dx, y2, x2, y2)})
		}
	}
	return lay
}

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
		if !p.IsStub() && p.Exercises == 0 {
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

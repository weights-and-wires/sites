package graph

import (
	"strings"
	"testing"

	"github.com/weights-and-wires/sites/internal/content"
)

func p(id string, prereqs ...string) content.Paper {
	paper := content.Paper{ID: id, Title: id, Status: "stub"}
	for _, r := range prereqs {
		paper.Prereqs = append(paper.Prereqs, content.Prereq{ID: r})
	}
	return paper
}

func TestChainOrdersPrerequisitesFirst(t *testing.T) {
	g, err := Build([]content.Paper{p("a"), p("b", "a"), p("c", "b", "a")})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(g.Chain("c"), ","); got != "a,b" {
		t.Errorf("Chain(c) = %s, want a,b", got)
	}
	if got := strings.Join(g.Next("a"), ","); got != "b,c" {
		t.Errorf("Next(a) = %s, want b,c", got)
	}
}

func TestBuildRejectsCycle(t *testing.T) {
	_, err := Build([]content.Paper{p("a", "c"), p("b", "a"), p("c", "b")})
	if err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("want a cycle error, got %v", err)
	}
}

func TestBuildRejectsUnknownPrerequisite(t *testing.T) {
	_, err := Build([]content.Paper{p("a", "missing")})
	if err == nil || !strings.Contains(err.Error(), "unknown prerequisite") {
		t.Fatalf("want an unknown-prerequisite error, got %v", err)
	}
}

func TestBuildRejectsFullPaperWithoutExercises(t *testing.T) {
	full := p("a")
	full.Status = "full"
	if _, err := Build([]content.Paper{full}); err == nil {
		t.Fatal("want an error for a full paper with no exercises")
	}
}

func TestLayoutPlacesLayersLeftToRight(t *testing.T) {
	g, err := Build([]content.Paper{p("a"), p("b", "a"), p("c", "b"), p("d", "a")})
	if err != nil {
		t.Fatal(err)
	}
	lay := g.Layout(100, 30, 80, 20)
	x := map[string]float64{}
	for _, n := range lay.Nodes {
		x[n.ID] = n.X
	}
	if !(x["a"] < x["b"] && x["b"] < x["c"] && x["b"] == x["d"]) {
		t.Errorf("unexpected columns: %v", x)
	}
	if len(lay.Edges) != 3 {
		t.Errorf("want 3 edges, got %d", len(lay.Edges))
	}
	// No two nodes in one column overlap.
	seen := map[[2]float64]bool{}
	for _, n := range lay.Nodes {
		k := [2]float64{n.X, n.Y}
		if seen[k] {
			t.Errorf("two nodes at %v", k)
		}
		seen[k] = true
	}
}

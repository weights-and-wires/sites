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

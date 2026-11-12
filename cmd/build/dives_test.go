package main

import (
	"strings"
	"testing"

	"github.com/weights-and-wires/sites/internal/content"
)

func diveBuilder(t *testing.T) *builder {
	t.Helper()
	return &builder{root: "../..", out: t.TempDir(), katex: "../../scripts/katex.mjs", offline: true, built: "test"}
}

func TestDiveCellMarkerBecomesEditableCell(t *testing.T) {
	d := content.Dive{ID: "demo", Title: "Demo", Body: "## Rebuild it\n\n<!-- cell: toy -->\n", Cells: map[string]string{"toy": "print(1 < 2)\n"}}
	views, err := diveBuilder(t).diveViews([]content.Dive{d}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	body := string(views[0].Body)
	for _, want := range []string{`data-cell="demo/toy"`, `print(1 &lt; 2)`, `data-run`} {
		if !strings.Contains(body, want) {
			t.Errorf("body lacks %q:\n%s", want, body)
		}
	}
	if !views[0].HasCells {
		t.Error("HasCells is false")
	}
}

func TestDiveCellMismatchFails(t *testing.T) {
	for name, d := range map[string]content.Dive{
		"marker without file":  {ID: "a", Body: "<!-- cell: gone -->\n"},
		"file without marker":  {ID: "b", Body: "text\n", Cells: map[string]string{"extra": "pass\n"}},
		"unknown prerequisite": {ID: "c", Body: "text\n", Prereqs: []content.Prereq{{ID: "no-such-paper", Reason: "x"}}},
	} {
		if _, err := diveBuilder(t).diveViews([]content.Dive{d}, nil, nil); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
}

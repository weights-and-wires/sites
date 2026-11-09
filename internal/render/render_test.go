package render

import (
	"strings"
	"testing"
)

func TestExtractMathSkipsCode(t *testing.T) {
	src := "Cost $O(n^2)$ and `$HOME` and a price of \\$5.\n\n```sh\necho $PATH\n```\n\n$$\na^2 + b^2\n$$\n"
	var items []Math
	out := ExtractMath(src, "test", &items)
	if len(items) != 2 {
		t.Fatalf("got %d formulas, want 2: %+v", len(items), items)
	}
	if items[0].TeX != "O(n^2)" || items[0].Display {
		t.Errorf("inline formula = %+v", items[0])
	}
	if items[1].TeX != "a^2 + b^2" || !items[1].Display {
		t.Errorf("display formula = %+v", items[1])
	}
	for _, keep := range []string{"`$HOME`", "echo $PATH", `\$5`} {
		if !strings.Contains(out, keep) {
			t.Errorf("output lost %q:\n%s", keep, out)
		}
	}
}

func TestMarkdownRestoresMathAndListsHeadings(t *testing.T) {
	var items []Math
	src := ExtractMath("## The idea\n\nScore $x$.\n\n$$y$$\n", "test", &items)
	html, heads, err := Markdown(src)
	if err != nil {
		t.Fatal(err)
	}
	// Display math is extracted first, so it takes index 0.
	html = RestoreMath(html, []string{"<D/>", "<I/>"})
	if !strings.Contains(html, "Score <I/>.") || !strings.Contains(html, `<div class="math-display"><D/></div>`) {
		t.Errorf("math not restored:\n%s", html)
	}
	if len(heads) != 1 || heads[0].ID != "the-idea" || heads[0].Text != "The idea" {
		t.Errorf("headings = %+v", heads)
	}
}

func TestCodeBlockUsesSlab(t *testing.T) {
	html, _, err := Markdown("```python\nx = 1\n```\n")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(html, `class="code-block"`) || !strings.Contains(html, "<span>python</span>") {
		t.Errorf("code block not wrapped:\n%s", html)
	}
}

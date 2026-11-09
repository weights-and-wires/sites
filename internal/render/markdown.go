package render

import (
	"bytes"
	"fmt"
	"html"
	"io"
	"regexp"
	"strings"

	chromahtml "github.com/alecthomas/chroma/v2/formatters/html"
	"github.com/yuin/goldmark"
	highlighting "github.com/yuin/goldmark-highlighting/v2"
	"github.com/yuin/goldmark/extension"
	"github.com/yuin/goldmark/parser"
	gmhtml "github.com/yuin/goldmark/renderer/html"
	"github.com/yuin/goldmark/util"
)

// Heading is an h2 or h3 on a page, for the table of contents and search.
type Heading struct {
	Level int
	ID    string
	Text  string
}

var md = goldmark.New(
	goldmark.WithExtensions(
		extension.GFM,
		extension.Footnote,
		highlighting.NewHighlighting(
			highlighting.WithFormatOptions(chromahtml.WithClasses(true)),
			highlighting.WithWrapperRenderer(codeWrapper),
		),
	),
	goldmark.WithParserOptions(parser.WithAutoHeadingID()),
	goldmark.WithRendererOptions(gmhtml.WithUnsafe()),
)

// codeWrapper draws the arm64 code slab: a toolbar with the language label
// above the highlighted block.
func codeWrapper(w util.BufWriter, ctx highlighting.CodeBlockContext, entering bool) {
	lang, _ := ctx.Language()
	label := "code"
	if len(lang) > 0 {
		label = string(lang)
	}
	if entering {
		fmt.Fprintf(w, `<div class="code-block"><div class="code-toolbar"><span>%s</span></div>`, html.EscapeString(label))
		return
	}
	io.WriteString(w, `</div>`)
}

var headingRe = regexp.MustCompile(`(?s)<h([23]) id="([^"]+)">(.*?)</h[23]>`)
var tagRe = regexp.MustCompile(`<[^>]+>`)

// Markdown renders markdown to HTML and lists its h2 and h3 headings.
func Markdown(src string) (string, []Heading, error) {
	var buf bytes.Buffer
	if err := md.Convert([]byte(src), &buf); err != nil {
		return "", nil, err
	}
	out := buf.String()
	var heads []Heading
	for _, m := range headingRe.FindAllStringSubmatch(out, -1) {
		level := 2
		if m[1] == "3" {
			level = 3
		}
		text := html.UnescapeString(strings.TrimSpace(tagRe.ReplaceAllString(m[3], "")))
		heads = append(heads, Heading{Level: level, ID: m[2], Text: text})
	}
	return out, heads, nil
}

// PlainText strips tags from rendered HTML, for search snippets.
func PlainText(htmlSrc string) string {
	text := html.UnescapeString(tagRe.ReplaceAllString(htmlSrc, " "))
	return strings.Join(strings.Fields(text), " ")
}

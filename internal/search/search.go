// Package search builds the search.json index for the "/" dialog.
package search

import "github.com/weights-and-wires/sites/internal/render"

// Entry is one page in the index.
type Entry struct {
	Title    string   `json:"title"`
	URL      string   `json:"url"`
	Headings []string `json:"headings"`
	Excerpt  string   `json:"excerpt"`
}

// NewEntry indexes a page's title, its h2 and h3 text, and the first 300
// characters of its text.
func NewEntry(title, url string, heads []render.Heading, text string) Entry {
	e := Entry{Title: title, URL: url}
	for _, h := range heads {
		e.Headings = append(e.Headings, h.Text)
	}
	r := []rune(text)
	if len(r) > 300 {
		r = r[:300]
	}
	e.Excerpt = string(r)
	return e
}

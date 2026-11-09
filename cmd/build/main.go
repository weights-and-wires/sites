// Command build renders papers.ifkash.dev and courses.ifkash.dev into static
// files, one directory per site:
//
//	go run ./cmd/build            # writes internal/assets/public/{papers,courses}
//	go run ./cmd/build -check     # also checks every internal link
//
// It needs Node with the katex package installed (npm ci) to render math.
package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/andybalholm/brotli"

	"github.com/weights-and-wires/sites/internal/content"
	"github.com/weights-and-wires/sites/internal/counts"
	"github.com/weights-and-wires/sites/internal/graph"
	"github.com/weights-and-wires/sites/internal/linkcheck"
	"github.com/weights-and-wires/sites/internal/render"
	"github.com/weights-and-wires/sites/internal/search"
)

// Site describes one of the two hosts.
type Site struct {
	ID       string
	Name     string
	Wordmark string
	Subtitle string
	Host     string
}

var sites = []Site{
	{ID: "papers", Name: "Papers", Wordmark: "papers", Subtitle: "reading paths", Host: "papers.ifkash.dev"},
	{ID: "courses", Name: "Courses", Wordmark: "courses", Subtitle: "ifkash.dev", Host: "courses.ifkash.dev"},
}

// Assets are the fingerprinted file names every page links to.
type Assets struct {
	CSS     string
	ThemeJS string
	Favicon string
}

// Page is the data every template gets.
type Page struct {
	Site        Site
	Assets      Assets
	Title       string
	Description string
	Built       string
	Math        bool

	Paper    content.Paper
	Prereqs  []PaperLink
	Next     []PaperLink
	Body     template.HTML
	Headings []render.Heading
	Papers   []content.Paper
	Hub      HubView
}

// HubView is the hub page: the catalogue with live counts and resolved paths.
type HubView struct {
	Title        template.HTML
	Lede         string
	Stages       []StageView
	Courses      []CourseView
	TotalLessons int
}

// StageView is one step of the roadmap strip.
type StageView struct {
	content.Stage
	Number  string
	Courses []CourseView
}

// CourseView is one hub card.
type CourseView struct {
	content.Course
	Number     string
	Count      int
	Live       bool
	StageLabel string
	Paths      []content.Path
}

// PaperLink is a paper reference with its title resolved.
type PaperLink struct {
	ID     string
	Title  string
	Reason string
}

type builder struct {
	root, out string
	katex     string
	built     string
	offline   bool
}

func main() {
	root := flag.String("root", ".", "repository root")
	out := flag.String("out", "internal/assets/public", "output directory")
	check := flag.Bool("check", false, "check internal links after the build")
	offline := flag.Bool("offline", false, "use the lesson counts in courses.yaml instead of asking GitHub")
	flag.Parse()

	b := &builder{root: *root, out: *out, katex: filepath.Join(*root, "scripts/katex.mjs"), built: time.Now().UTC().Format("2006-01-02"), offline: *offline}
	if err := b.run(); err != nil {
		log.Fatal(err)
	}
	if *check {
		hosts := map[string]string{}
		for _, s := range sites {
			hosts[s.Host] = s.ID
		}
		if problems := linkcheck.Check(os.DirFS(*out), hosts); len(problems) > 0 {
			log.Fatalf("broken links:\n  %s", strings.Join(problems, "\n  "))
		}
		log.Print("link check: no broken internal links")
	}
}

func (b *builder) run() error {
	// Keep .gitkeep so the embed directive always has a file to match.
	entries, _ := os.ReadDir(b.out)
	for _, e := range entries {
		if e.Name() != ".gitkeep" {
			if err := os.RemoveAll(filepath.Join(b.out, e.Name())); err != nil {
				return err
			}
		}
	}

	papers, err := content.LoadPapers(filepath.Join(b.root, "content/papers"))
	if err != nil {
		return err
	}
	g, err := graph.Build(papers)
	if err != nil {
		return err
	}
	hub, err := content.LoadHub(filepath.Join(b.root, "content/hub/courses.yaml"))
	if err != nil {
		return err
	}
	paths, err := content.LoadPaths(filepath.Join(b.root, "content/paths"))
	if err != nil {
		return err
	}
	hubView := b.hubView(hub, paths, len(papers))

	// Pull every formula out first, so KaTeX runs once for the whole build.
	var formulas []render.Math
	sources := make([]string, len(papers))
	for i, p := range papers {
		sources[i] = render.ExtractMath(p.Body, "papers/"+p.ID, &formulas)
	}
	rendered, err := render.KaTeX(b.katex, formulas)
	if err != nil {
		return err
	}

	byID := map[string]content.Paper{}
	for _, p := range papers {
		byID[p.ID] = p
	}
	link := func(id, reason string) PaperLink { return PaperLink{ID: id, Title: byID[id].Title, Reason: reason} }

	for _, site := range sites {
		assets, err := b.writeAssets(site.ID, site.ID == "papers" && len(formulas) > 0)
		if err != nil {
			return err
		}
		base := Page{Site: site, Assets: assets, Built: b.built}

		switch site.ID {
		case "papers":
			var index []search.Entry
			for i, p := range papers {
				body, heads, err := render.Markdown(sources[i])
				if err != nil {
					return fmt.Errorf("papers/%s: %w", p.ID, err)
				}
				body = render.RestoreMath(body, rendered)
				page := base
				page.Title = p.Title + " · Papers"
				page.Description = p.TLDR
				page.Paper, page.Body, page.Headings = p, template.HTML(body), heads
				page.Math = strings.Contains(body, `class="katex`)
				for _, r := range p.Prereqs {
					page.Prereqs = append(page.Prereqs, link(r.ID, r.Reason))
				}
				for _, n := range g.Next(p.ID) {
					page.Next = append(page.Next, link(n, ""))
				}
				if err := b.page(site.ID, p.ID+"/index.html", "paper.html", page); err != nil {
					return err
				}
				index = append(index, search.NewEntry(p.Title, "/"+p.ID+"/", heads, render.PlainText(body)))
			}
			home := base
			home.Title = "Papers · ifkash.dev"
			home.Description = "Paper pages with prerequisites and in-browser exercises."
			home.Papers = papers
			if err := b.page(site.ID, "index.html", "papers_home.html", home); err != nil {
				return err
			}
			data, err := json.Marshal(index)
			if err != nil {
				return err
			}
			if err := b.write(site.ID, "search.json", data); err != nil {
				return err
			}
		case "courses":
			home := base
			home.Title = "Courses · ifkash.dev"
			home.Description = hub.Lede
			home.Hub = hubView
			if err := b.page(site.ID, "index.html", "hub.html", home); err != nil {
				return err
			}
		}
		notFound := base
		notFound.Title = "Page not found · " + site.Name
		if err := b.page(site.ID, "404.html", "404.html", notFound); err != nil {
			return err
		}
	}
	return b.compress()
}

// hubView fills in lesson counts and reading paths. A course whose count
// can't be fetched keeps the count from courses.yaml.
func (b *builder) hubView(hub content.Hub, paths []content.Path, papers int) HubView {
	token := os.Getenv("COURSES_READ_TOKEN")
	if token == "" {
		token = os.Getenv("GITHUB_TOKEN")
	}
	client := &http.Client{}
	byPath := map[string]content.Path{}
	for _, p := range paths {
		byPath[p.ID] = p
	}
	stageLabel := map[string]string{}
	for _, st := range hub.Stages {
		stageLabel[st.ID] = st.Label
	}

	v := HubView{Title: template.HTML(hub.Title), Lede: hub.Lede}
	for i, c := range hub.Courses {
		cv := CourseView{Course: c, Number: fmt.Sprintf("%02d", i+1), Count: c.Lessons, StageLabel: stageLabel[c.Stage]}
		switch {
		case c.ID == "papers":
			cv.Count, cv.Live = papers, true
		case c.Repo != "" && c.Glob != "" && !b.offline:
			n, err := counts.Count(context.Background(), client, c.Repo, c.Glob, token)
			if err != nil {
				log.Printf("hub: %s: using the courses.yaml count (%d): %v", c.ID, c.Lessons, err)
			} else {
				cv.Count, cv.Live = n, true
			}
		}
		for _, id := range c.Paths {
			if p, ok := byPath[id]; ok {
				cv.Paths = append(cv.Paths, p)
			}
		}
		if c.Unit != "papers" {
			v.TotalLessons += cv.Count
		}
		v.Courses = append(v.Courses, cv)
	}
	for i, st := range hub.Stages {
		sv := StageView{Stage: st, Number: fmt.Sprintf("%02d", i+1)}
		for _, cv := range v.Courses {
			if cv.Stage == st.ID {
				sv.Courses = append(sv.Courses, cv)
			}
		}
		v.Stages = append(v.Stages, sv)
	}
	return v
}

// writeAssets copies the design system and site styles into assets/ with a
// content hash in each name, so the server can cache them forever.
func (b *builder) writeAssets(site string, math bool) (Assets, error) {
	var css bytes.Buffer
	css.WriteString("@layer base, components, site;\n")
	for _, f := range []string{"ui/tokens.css", "ui/type.css", "ui/components.css", "static/site.css"} {
		data, err := os.ReadFile(filepath.Join(b.root, f))
		if err != nil {
			return Assets{}, err
		}
		css.Write(data)
		css.WriteString("\n")
	}
	theme, err := os.ReadFile(filepath.Join(b.root, "ui/theme.js"))
	if err != nil {
		return Assets{}, err
	}
	favicon, err := os.ReadFile(filepath.Join(b.root, "static/favicon.svg"))
	if err != nil {
		return Assets{}, err
	}
	a := Assets{
		CSS:     hashed("site", ".css", css.Bytes()),
		ThemeJS: hashed("theme", ".js", theme),
		Favicon: hashed("favicon", ".svg", favicon),
	}
	for name, data := range map[string][]byte{a.CSS: css.Bytes(), a.ThemeJS: theme, a.Favicon: favicon} {
		if err := b.write(site, "assets/"+name, data); err != nil {
			return a, err
		}
	}
	if math {
		if err := b.copyKaTeX(site); err != nil {
			return a, err
		}
	}
	return a, nil
}

// copyKaTeX ships the KaTeX stylesheet and its woff2 fonts only.
func (b *builder) copyKaTeX(site string) error {
	dist := filepath.Join(b.root, "node_modules/katex/dist")
	css, err := os.ReadFile(filepath.Join(dist, "katex.min.css"))
	if err != nil {
		return fmt.Errorf("katex css (run npm ci): %w", err)
	}
	if err := b.write(site, "assets/katex/katex.min.css", css); err != nil {
		return err
	}
	fonts, err := filepath.Glob(filepath.Join(dist, "fonts/*.woff2"))
	if err != nil {
		return err
	}
	for _, f := range fonts {
		data, err := os.ReadFile(f)
		if err != nil {
			return err
		}
		if err := b.write(site, "assets/katex/fonts/"+filepath.Base(f), data); err != nil {
			return err
		}
	}
	return nil
}

func hashed(stem, ext string, data []byte) string {
	sum := sha256.Sum256(data)
	return stem + "." + hex.EncodeToString(sum[:])[:10] + ext
}

func (b *builder) page(site, path, tmpl string, data Page) error {
	t, err := template.ParseFiles(filepath.Join(b.root, "templates/base.html"), filepath.Join(b.root, "templates", tmpl))
	if err != nil {
		return err
	}
	var buf bytes.Buffer
	if err := t.ExecuteTemplate(&buf, "base", data); err != nil {
		return fmt.Errorf("%s/%s: %w", site, path, err)
	}
	return b.write(site, path, buf.Bytes())
}

func (b *builder) write(site, path string, data []byte) error {
	full := filepath.Join(b.out, site, path)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		return err
	}
	return os.WriteFile(full, data, 0o644)
}

// compress writes .gz and .br siblings for text files over 1 KiB.
func (b *builder) compress() error {
	return filepath.WalkDir(b.out, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		switch filepath.Ext(path) {
		case ".html", ".css", ".js", ".json", ".svg", ".txt":
		default:
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil || len(data) < 1024 {
			return err
		}
		var gz bytes.Buffer
		zw, _ := gzip.NewWriterLevel(&gz, gzip.BestCompression)
		zw.Write(data)
		zw.Close()
		var br bytes.Buffer
		bw := brotli.NewWriterLevel(&br, brotli.BestCompression)
		bw.Write(data)
		bw.Close()
		if err := os.WriteFile(path+".gz", gz.Bytes(), 0o644); err != nil {
			return err
		}
		return os.WriteFile(path+".br", br.Bytes(), 0o644)
	})
}

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
	"os/exec"
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
	CSS      string
	ThemeJS  string
	Favicon  string
	PapersJS string
	HubJS    string
	WorkerJS string
	Harness  string
	EditorJS string
	DivesJS  string
}

// Page is the data every template gets.
type Page struct {
	Site        Site
	Assets      Assets
	Title       string
	Description string
	Built       string
	Math        bool

	PaperView PaperView
	Dive      *DiveView
	Headings  []render.Heading
	Papers    []content.Paper
	Hub       HubView
	Extra     any
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
	Year   int
	Track  string
	Stub   bool
}

type builder struct {
	root, out string
	katex     string
	built     string
	offline   bool
	release   bool
	math      bool // some paper page has a formula
}

func main() {
	root := flag.String("root", ".", "repository root")
	out := flag.String("out", "internal/assets/public", "output directory")
	check := flag.Bool("check", false, "check internal links after the build")
	offline := flag.Bool("offline", false, "use the lesson counts in courses.yaml instead of asking GitHub")
	release := flag.Bool("release", false, "fail if Pyodide or the editor bundle is missing")
	flag.Parse()

	b := &builder{root: *root, out: *out, katex: filepath.Join(*root, "scripts/katex.mjs"), built: time.Now().UTC().Format("2006-01-02"), offline: *offline, release: *release}
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
	dives, err := content.LoadDives(filepath.Join(b.root, "content/dives"))
	if err != nil {
		return err
	}
	hubView := b.hubView(hub, paths, len(papers), len(dives))

	concepts, err := content.LoadConcepts(filepath.Join(b.root, "content/papers/concepts.yaml"))
	if err != nil {
		return err
	}
	views, err := b.paperViews(papers, g, concepts)
	if err != nil {
		return err
	}
	diveViews, err := b.diveViews(dives, papers, concepts)
	if err != nil {
		return err
	}

	for _, site := range sites {
		assets, err := b.writeAssets(site.ID, site.ID == "papers" && b.math)
		if err != nil {
			return err
		}
		base := Page{Site: site, Assets: assets, Built: b.built}

		switch site.ID {
		case "papers":
			var index []search.Entry
			for _, v := range views {
				page := base
				page.Title = v.Title + " · Papers"
				page.Description = v.Summary
				page.PaperView = v
				page.Headings = v.Headings
				page.Math = strings.Contains(string(v.BodyBefore)+string(v.BodyAfter)+string(v.Further), `class="katex`) || len(v.Exercises) > 0
				if err := b.page(site.ID, v.ID+"/index.html", "paper.html", page); err != nil {
					return err
				}
				for _, x := range v.Paper.Exercises {
					data, err := exerciseZip(v.Paper, x)
					if err != nil {
						return err
					}
					if err := b.write(site.ID, fmt.Sprintf("%s/exercises/%d.zip", v.ID, x.N), data); err != nil {
						return err
					}
				}
				index = append(index, search.NewEntry(v.Title, "/"+v.ID+"/", v.Headings, v.Summary+" "+v.Text))
			}
			if err := b.copyPyodide(b.release); err != nil {
				return err
			}
			more, err := b.writeSitePages(base, papers, paths, g)
			if err != nil {
				return err
			}
			index = append(index, more...)
			more, err = b.writeDives(base, diveViews)
			if err != nil {
				return err
			}
			index = append(index, more...)
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
func (b *builder) hubView(hub content.Hub, paths []content.Path, papers, dives int) HubView {
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
		case c.ID == "dives":
			cv.Count, cv.Live = dives, true
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
		if c.Stage != "papers" {
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
	files := map[string][]byte{a.CSS: css.Bytes(), a.ThemeJS: theme, a.Favicon: favicon}
	if site == "courses" {
		data, err := os.ReadFile(filepath.Join(b.root, "static/hub.js"))
		if err != nil {
			return a, err
		}
		a.HubJS = hashed("hub", ".js", data)
		files[a.HubJS] = data
	}
	if site == "papers" {
		for _, f := range []struct {
			src, stem, ext string
			dst            *string
		}{
			{"static/papers.js", "papers", ".js", &a.PapersJS},
			{"static/dives.js", "dives", ".js", &a.DivesJS},
			{"static/runner/worker.js", "worker", ".js", &a.WorkerJS},
			{"static/runner/harness.py", "harness", ".py", &a.Harness},
		} {
			data, err := os.ReadFile(filepath.Join(b.root, f.src))
			if err != nil {
				return a, err
			}
			*f.dst = hashed(f.stem, f.ext, data)
			files[*f.dst] = data
		}
		editor, err := b.bundleEditor()
		if err != nil {
			return a, err
		}
		a.EditorJS = hashed("editor", ".js", editor)
		files[a.EditorJS] = editor
	}
	for name, data := range files {
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

// bundleEditor builds the CodeMirror bundle with esbuild (scripts/editor.mjs).
func (b *builder) bundleEditor() ([]byte, error) {
	tmp, err := os.MkdirTemp("", "editor")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)
	out := filepath.Join(tmp, "editor.js")
	cmd := exec.Command("node", filepath.Join(b.root, "scripts/editor.mjs"), out)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("editor bundle (run npm ci): %w", err)
	}
	return os.ReadFile(out)
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

var funcs = template.FuncMap{
	"add":       func(a, b int) int { return a + b },
	"trackName": trackName,
	"pct": func(a, b int) int {
		if b == 0 {
			return 0
		}
		return a * 100 / b
	},
}

func hashed(stem, ext string, data []byte) string {
	sum := sha256.Sum256(data)
	return stem + "." + hex.EncodeToString(sum[:])[:10] + ext
}

func (b *builder) page(site, path, tmpl string, data Page) error {
	t, err := template.New("base").Funcs(funcs).ParseFiles(filepath.Join(b.root, "templates/base.html"), filepath.Join(b.root, "templates", tmpl))
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
		case ".html", ".css", ".js", ".json", ".svg", ".txt", ".py", ".wasm":
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

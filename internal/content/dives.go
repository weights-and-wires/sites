package content

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Dive is one deep-dive page, read from content/dives/<id>/index.md. A dive
// teaches an external blog post or write-up with interactive widgets. Its
// folder holds the following files:
//
//	index.md        front matter plus the page body
//	dive.js         optional: the page's widgets, loaded after static/dives.js
//	dive.css        optional: styles for those widgets
//	cells/<name>.py optional: editable Python cells, placed in the body with
//	                <!-- cell: name -->
//	scripts/*.py    optional: scripts that compute the numbers on the page
type Dive struct {
	ID          string       `yaml:"id"`
	Title       string       `yaml:"title"`
	Summary     string       `yaml:"summary"`
	Source      DiveSource   `yaml:"source"`
	Topics      []string     `yaml:"topics"`
	Minutes     int          `yaml:"est_minutes"`
	Prereqs     []Prereq     `yaml:"prereqs"`
	Background  []string     `yaml:"background"`
	CourseLinks []CourseLink `yaml:"course_links"`

	// Body is the markdown after the front matter.
	Body string `yaml:"-"`
	// JS and CSS are dive.js and dive.css, or nil.
	JS  []byte `yaml:"-"`
	CSS []byte `yaml:"-"`
	// Cells maps a cell name to its Python source.
	Cells map[string]string `yaml:"-"`
}

// DiveSource is the post that a dive teaches.
type DiveSource struct {
	Title  string `yaml:"title"`
	Author string `yaml:"author"`
	Site   string `yaml:"site"`
	Date   string `yaml:"date"`
	URL    string `yaml:"url"`
}

// LoadDives reads every content/dives/<id>/index.md, sorted by ID. A missing
// directory means no dives yet.
func LoadDives(dir string) ([]Dive, error) {
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var dives []Dive
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		d, err := loadDive(filepath.Join(dir, e.Name()), e.Name())
		if err != nil {
			return nil, err
		}
		dives = append(dives, d)
	}
	sort.Slice(dives, func(i, j int) bool { return dives[i].ID < dives[j].ID })
	return dives, nil
}

func loadDive(dir, name string) (Dive, error) {
	var d Dive
	path := filepath.Join(dir, "index.md")
	src, err := os.ReadFile(path)
	if err != nil {
		return d, err
	}
	meta, body, err := SplitFrontMatter(src)
	if err != nil {
		return d, fmt.Errorf("%s: %w", path, err)
	}
	dec := yaml.NewDecoder(bytes.NewReader(meta))
	dec.KnownFields(true)
	if err := dec.Decode(&d); err != nil {
		return d, fmt.Errorf("%s: %w", path, err)
	}
	if d.ID == "" {
		d.ID = name
	}
	if d.ID != name {
		return d, fmt.Errorf("%s: id %q does not match folder %q", path, d.ID, name)
	}
	var problems []string
	if strings.TrimSpace(d.Title) == "" || strings.TrimSpace(d.Summary) == "" {
		problems = append(problems, "title and summary are required")
	}
	if d.Source.Title == "" || d.Source.Author == "" || !strings.HasPrefix(d.Source.URL, "https://") {
		problems = append(problems, "source needs title, author, and an https url")
	}
	for _, r := range d.Prereqs {
		if strings.TrimSpace(r.Reason) == "" {
			problems = append(problems, fmt.Sprintf("prerequisite %q needs a reason", r.ID))
		}
	}
	if len(problems) > 0 {
		return d, fmt.Errorf("%s: %s", path, strings.Join(problems, "; "))
	}
	d.Body = string(body)
	for name, dst := range map[string]*[]byte{"dive.js": &d.JS, "dive.css": &d.CSS} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil && !os.IsNotExist(err) {
			return d, err
		}
		*dst = data
	}
	cells, err := filepath.Glob(filepath.Join(dir, "cells", "*.py"))
	if err != nil {
		return d, err
	}
	d.Cells = map[string]string{}
	for _, f := range cells {
		data, err := os.ReadFile(f)
		if err != nil {
			return d, err
		}
		d.Cells[strings.TrimSuffix(filepath.Base(f), ".py")] = string(data)
	}
	return d, nil
}

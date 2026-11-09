// Package linkcheck finds internal links in built HTML that point at files
// that don't exist.
package linkcheck

import (
	"fmt"
	"io/fs"
	"net/url"
	"path"
	"regexp"
	"sort"
	"strings"
)

var attr = regexp.MustCompile(`(?:href|src)="([^"]+)"`)

// Check walks every .html file in root (one directory per site) and returns
// one line per broken link. hosts maps a host name to its site directory, so
// absolute links between the sites are checked too.
func Check(root fs.FS, hosts map[string]string) []string {
	var problems []string
	fs.WalkDir(root, ".", func(file string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || path.Ext(file) != ".html" {
			return err
		}
		data, err := fs.ReadFile(root, file)
		if err != nil {
			return err
		}
		site := strings.SplitN(file, "/", 2)[0]
		for _, m := range attr.FindAllStringSubmatch(string(data), -1) {
			target, ok := resolve(m[1], site, path.Dir(file), hosts)
			if !ok {
				continue
			}
			if !exists(root, target) {
				problems = append(problems, fmt.Sprintf("%s: %s", file, m[1]))
			}
		}
		return nil
	})
	sort.Strings(problems)
	return problems
}

func resolve(ref, site, dir string, hosts map[string]string) (string, bool) {
	u, err := url.Parse(ref)
	if err != nil || u.Path == "" && u.Host == "" {
		return "", false
	}
	switch {
	case u.Scheme == "https" || u.Scheme == "http":
		target, ok := hosts[u.Host]
		if !ok {
			return "", false
		}
		p := u.Path
		if p == "" {
			p = "/"
		}
		return path.Join(target, p) + trailing(p), true
	case u.Scheme != "":
		return "", false
	case strings.HasPrefix(u.Path, "/"):
		return path.Join(site, u.Path) + trailing(u.Path), true
	default:
		return path.Join(dir, u.Path) + trailing(u.Path), true
	}
}

func trailing(p string) string {
	if strings.HasSuffix(p, "/") {
		return "/"
	}
	return ""
}

func exists(root fs.FS, target string) bool {
	if strings.HasSuffix(target, "/") {
		target = path.Join(target, "index.html")
	}
	st, err := fs.Stat(root, target)
	if err != nil {
		return false
	}
	if st.IsDir() {
		_, err = fs.Stat(root, path.Join(target, "index.html"))
		return err == nil
	}
	return true
}
